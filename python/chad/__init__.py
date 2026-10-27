"""CHAD: Combinatory Homomorphic Automatic Differentiation, in Python.

A small reference implementation of the CHAD source-code transformation
(Vákár & Smeding, https://arxiv.org/abs/2103.15776) for a higher-order
language with tuples, sum types and maps over vectors.

    import chad
    from chad import sin, cos

    @chad.function
    def f(x, y):
        z = x * y
        return sin(z) + 2 * x

    f(1.0, 2.0)                       # evaluate the program
    chad.grad(f)(1.0, 2.0)            # reverse mode: (df/dx, df/dy)
    chad.jvp(f, (1.0, 2.0), (1.0, 0.0))  # forward mode: (f(x, y), df/dx)
    print(chad.reverse(f).source)     # the derivative program, as Python
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import runtime as rt
from .ast import Expr, Var, fresh, reserve
from .codegen import RESERVED, compile_expr
from .runtime import ZERO, Inl, Inr
from .simplify import simplify as _simplify
from .trace import (Tracer, case, cond, cos, exp, fn, inl, inr, log,
                    replicate, scope, sin, sqrt, sum, tanh, vmap)
from .transform import df, dr

__all__ = [
    "function", "Function", "reverse", "forward", "grad", "value_and_grad", "vjp", "jvp",
    "fn", "vmap", "cond", "case", "inl", "inr",
    "sin", "cos", "exp", "log", "sqrt", "tanh", "sum", "replicate",
    "Inl", "Inr", "ZERO", "dr", "df",
]


@dataclass
class Compiled:
    """An expression compiled to a Python function."""

    expr: Expr
    fn: Callable
    source: str

    def __call__(self, *args):
        return _tidy(self.fn(*map(_coerce, args)))

    def __repr__(self) -> str:
        return self.source


class Function:
    """A Python function traced into a CHAD source-language program.

    The program's free variables are the function's parameters: as in the
    paper, a program is a function of its environment.
    """

    def __init__(self, f: Callable):
        import inspect

        self.__wrapped__ = f
        name = getattr(f, "__name__", "f")
        self.__name__ = name if name.isidentifier() else "f"
        self.__doc__ = getattr(f, "__doc__", None)
        self.params = [_safe_name(p) for p in inspect.signature(f).parameters]
        self.expr: Expr = scope(lambda: f(*[Tracer(Var(p)) for p in self.params]))
        self._cache: dict = {}

    def _compiled(self, key, make: Callable[[], Expr], name: str) -> Compiled:
        if key not in self._cache:
            with _deep_recursion():
                e = make()
                self._cache[key] = Compiled(e, *compile_expr(self.params, e, name))
        return self._cache[key]

    @property
    def primal(self) -> Compiled:
        return self._compiled("primal", lambda: self.expr, self.__name__)

    @property
    def source(self) -> str:
        """The traced program, as Python source."""
        return self.primal.source

    def __call__(self, *args):
        return self.primal(*args)

    def __repr__(self) -> str:
        return f"<chad.Function {self.__name__}({', '.join(self.params)})>"


def function(f: Callable) -> Function:
    """Decorator: trace a Python function into a differentiable program."""
    return f if isinstance(f, Function) else Function(f)


def reverse(f, simplify: bool = True) -> Compiled:
    """The CHAD reverse derivative of ``f``.

    Calling the result on primal arguments returns ``(output, backprop)`` where
    ``backprop(d)`` maps an output cotangent to a sparse dict of cotangents of
    the parameters.
    """
    f = function(f)
    return f._compiled(("rev", simplify), lambda: _maybe_simplify(dr(f.expr), simplify), f.__name__ + "_rev")


def forward(f, simplify: bool = True) -> Compiled:
    """The CHAD forward derivative of ``f``.

    Calling the result on primal arguments returns ``(output, tangent)`` where
    ``tangent(dt)`` maps a (sparse) dict of parameter tangents to the output
    tangent.
    """
    f = function(f)
    return f._compiled(("fwd", simplify), lambda: _maybe_simplify(df(f.expr), simplify), f.__name__ + "_fwd")


def vjp(f, *args, simplify: bool = True):
    """Return ``(f(*args), pullback)``; ``pullback(d)`` gives a tuple of cotangents, one per argument."""
    f = function(f)
    args = tuple(map(_coerce, args))
    out, backprop = reverse(f, simplify)(*args)

    def pullback(d):
        ct = backprop(_coerce(d))
        return _tidy(tuple(rt.densify(rt.get(ct, p), a) for p, a in zip(f.params, args)))

    return out, pullback


def jvp(f, args, tangents, simplify: bool = True):
    """Return ``(f(*args), tangent of the output)`` for the given argument tangents."""
    f = function(f)
    args, tangents = tuple(map(_coerce, args)), tuple(map(_coerce, tangents))
    out, tangent = forward(f, simplify)(*args)
    return out, _tidy(rt.densify(tangent(dict(zip(f.params, tangents))), out))


def value_and_grad(f, simplify: bool = True):
    """Like ``grad``, but also return the value: ``(f(*args), gradient)``."""
    f = function(f)

    def vg(*args):
        out, pullback = vjp(f, *args, simplify=simplify)
        if isinstance(out, tuple) or np.ndim(out) != 0:
            raise TypeError(f"grad requires a scalar output, got {out!r}")
        g = pullback(1.0)
        return out, g[0] if len(g) == 1 else g

    return vg


def grad(f, simplify: bool = True):
    """The gradient of a scalar-valued ``f``: one entry per argument (unwrapped if there is one)."""
    vg = value_and_grad(f, simplify)
    return lambda *args: vg(*args)[1]


# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _deep_recursion(limit: int = 100_000):
    """The transformations recurse over the program; allow long programs."""
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, limit))
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


def _maybe_simplify(e: Expr, simplify: bool) -> Expr:
    return _simplify(e) if simplify else e


def _safe_name(p: str) -> str:
    """Keep parameter names, unless they clash with a runtime helper."""
    return fresh(p + "_") if p in RESERVED else reserve(p)


def _tidy(v):
    """numpy scalars -> Python floats, for friendlier results."""
    if isinstance(v, tuple):
        return tuple(map(_tidy, v))
    if isinstance(v, np.floating):
        return float(v)
    return v


def _coerce(v):
    if isinstance(v, tuple):
        return tuple(map(_coerce, v))
    if isinstance(v, list):
        return np.asarray(v, dtype=float)
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, np.floating):
        return float(v)
    return v
