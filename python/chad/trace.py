"""Tracing: turning ordinary-looking Python into source-language programs.

Calling a ``@chad.function`` on symbolic ``Tracer`` arguments records every
primitive operation as a let-binding (A-normal form), so sharing in the Python
code is preserved as sharing in the program. Python-level control flow and
closures are simply executed away during tracing; to put *real* lambdas, maps
and case distinctions in the program (so that CHAD differentiates them as
such) use ``fn``, ``vmap``, ``cond`` and ``case``.
"""

from __future__ import annotations

import inspect
import numbers

import numpy as np

from . import runtime as rt
from .ast import (App, Case, Const, Expr, Inl, Inr, Lam, Let, Map, Op, Proj,
                  Tuple, Var, fresh)
from .ops import PRIMITIVES

_scopes: list[list[tuple[str, Expr]]] = []


class Tracer:
    """A symbolic value standing for a variable of the program being traced."""

    __slots__ = ("expr",)
    __array_priority__ = 1000  # make numpy defer to our operators

    def __init__(self, expr: Expr):
        self.expr = expr

    def __repr__(self) -> str:
        return f"Tracer({self.expr.name if isinstance(self.expr, Var) else self.expr})"

    # arithmetic
    def __add__(self, o): return _op("add", self, o)
    def __radd__(self, o): return _op("add", o, self)
    def __sub__(self, o): return _op("sub", self, o)
    def __rsub__(self, o): return _op("sub", o, self)
    def __mul__(self, o): return _op("mul", self, o)
    def __rmul__(self, o): return _op("mul", o, self)
    def __truediv__(self, o): return _op("div", self, o)
    def __rtruediv__(self, o): return _op("div", o, self)
    def __neg__(self): return _op("neg", self)
    def __pos__(self): return self

    def __pow__(self, n):
        if isinstance(n, Tracer):
            raise TypeError("only constant exponents are supported; use exp(n * log(x))")
        return _op("pow", self, n)

    # comparisons produce booleans: the sum type () + ()
    def __lt__(self, o): return _op("lt", self, o)
    def __le__(self, o): return _op("le", self, o)
    def __gt__(self, o): return _op("gt", self, o)
    def __ge__(self, o): return _op("ge", self, o)

    def __getitem__(self, i: int):
        """Tuple projection."""
        if not isinstance(i, int):
            raise TypeError("traced values can only be indexed by integer tuple positions")
        return _emit(Proj(i, self.expr))

    def __call__(self, *args):
        """Apply a traced function (see ``fn``)."""
        arg = to_expr(args[0]) if len(args) == 1 else Tuple(tuple(map(to_expr, args)))
        return _emit(App(self.expr, (arg,)))

    def __bool__(self):
        raise TypeError("a traced value cannot be used in Python control flow; use chad.cond(...)")

    def __iter__(self):
        raise TypeError("traced tuples cannot be unpacked; index them with t[0], t[1], ...")


def to_expr(v) -> Expr:
    if isinstance(v, Tracer):
        return v.expr
    if isinstance(v, tuple):
        return Tuple(tuple(map(to_expr, v)))
    if v is None:
        return Tuple(())
    if isinstance(v, (bool, np.bool_)):
        return Const(rt.Inl(()) if v else rt.Inr(()))
    if isinstance(v, numbers.Integral):
        return Const(int(v))
    if isinstance(v, numbers.Real):
        return Const(float(v))
    if isinstance(v, (np.ndarray, list)):
        return Const(np.asarray(v, dtype=float))
    raise TypeError(f"cannot use {type(v).__name__} value {v!r} in a chad program")


def _emit(e: Expr, prefix: str = "v") -> Tracer:
    if not _scopes:
        raise RuntimeError("chad.fn / vmap / cond / case / inl / inr can only be used while tracing a chad.function")
    name = fresh(prefix)
    _scopes[-1].append((name, e))
    return Tracer(Var(name))


def _op(name: str, *args):
    if not any(isinstance(a, Tracer) for a in args):
        return PRIMITIVES[name].impl(*args)  # plain values: just compute
    return _emit(Op(name, tuple(to_expr(a) for a in args)))


def scope(build) -> Expr:
    """Run ``build()`` in a fresh scope; return its bindings wrapped around its result."""
    _scopes.append([])
    try:
        out = to_expr(build())
    finally:
        bindings = _scopes.pop()
    for name, rhs in reversed(bindings):
        out = Let(name, rhs, out)
    return out


def _param_names(f, default: str = "x") -> list[str]:
    try:
        return [p.name for p in inspect.signature(f).parameters.values()]
    except (TypeError, ValueError):
        return [default]


# ---------------------------------------------------------------------------
# User-facing language constructs


def fn(f) -> Tracer:
    """Make ``f`` a first-class function *in the program* (a lambda).

    Multi-argument functions take their arguments as a tuple.
    """
    names = _param_names(f)
    p = fresh(names[0] if len(names) == 1 else "p")  # unary in the program

    def body():
        if len(names) == 0:
            return f()
        if len(names) == 1:
            return f(Tracer(Var(p)))
        return f(*[_emit(Proj(i, Var(p)), n) for i, n in enumerate(names)])

    return _emit(Lam((p,), scope(body)), "f")


def vmap(f, xs) -> Tracer:
    """``map f xs`` over a vector of scalars. ``f`` may close over other values."""
    x = fresh(_param_names(f)[0])
    return _emit(Map(x, scope(lambda: f(Tracer(Var(x)))), to_expr(xs)))


def inl(v) -> Tracer:
    """Left injection into a sum type."""
    return _emit(Inl(to_expr(v)))


def inr(v) -> Tracer:
    """Right injection into a sum type."""
    return _emit(Inr(to_expr(v)))


def case(s, on_inl, on_inr):
    """Case distinction on a sum-type value: ``on_inl(a)`` or ``on_inr(b)``."""
    if isinstance(s, (rt.Inl, rt.Inr)):  # not traced: decide now
        return (on_inl if isinstance(s, rt.Inl) else on_inr)(s.value)
    xl, xr = fresh(_param_names(on_inl)[0]), fresh(_param_names(on_inr)[0])
    left = scope(lambda: on_inl(Tracer(Var(xl))))
    right = scope(lambda: on_inr(Tracer(Var(xr))))
    return _emit(Case(to_expr(s), xl, left, xr, right))


def cond(pred, then, else_):
    """``then()`` if ``pred`` else ``else_()``, where ``pred`` is a comparison."""
    if isinstance(pred, (bool, np.bool_)):
        return then() if pred else else_()
    return case(pred, lambda _: then(), lambda _: else_())


def _unary(name: str):
    def f(x):
        return _op(name, x)

    f.__name__ = name
    f.__doc__ = f"Elementwise {name}."
    return f


sin, cos, exp, log, sqrt, tanh = map(_unary, ["sin", "cos", "exp", "log", "sqrt", "tanh"])


def sum(v):
    """Sum of a vector."""
    return _op("sum", v)


def replicate(x, n: int):
    """A vector of length ``n`` filled with ``x``."""
    return _op("replicate", x, n)
