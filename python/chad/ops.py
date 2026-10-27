"""Primitive operations of the source language and their derivatives.

This is the analogue of ``Operation.hs`` together with ``dfOp`` / ``drOp``.
Each primitive gives

* ``impl``: how to evaluate it,
* ``jvp(xs, dxs)``: its derivative, as a target expression linear in the
  tangents ``dxs`` of its arguments,
* ``vjp(xs, d)``: its transposed derivative, as a list of target expressions
  (one per argument) linear in the output cotangent ``d``.

CHAD's correctness theorem says that if these per-primitive rules are right,
the derivative of every composite program is right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import runtime as rt
from .ast import Const, Expr, op

Z = Const(rt.ZERO)


@dataclass(frozen=True)
class Primitive:
    impl: Callable
    jvp: Callable[[list[Expr], list[Expr]], Expr]
    vjp: Callable[[list[Expr], Expr], list[Expr]]


def _unary(impl, deriv):
    """A unary scalar/elementwise op ``f`` with derivative ``f' = deriv(x)``."""
    return Primitive(
        impl,
        jvp=lambda xs, dxs: op("scale", deriv(xs[0]), dxs[0]),
        vjp=lambda xs, d: [op("scale", deriv(xs[0]), d)],
    )


def _nondiff(impl):
    """An op whose output has no (non-zero) derivative, e.g. comparisons."""
    return Primitive(impl, jvp=lambda xs, dxs: Z, vjp=lambda xs, d: [Z] * len(xs))


def _bool(b) -> rt.Inl | rt.Inr:
    """Booleans are the sum type ``() + ()``: true is ``inl ()``."""
    return rt.Inl(()) if b else rt.Inr(())


PRIMITIVES: dict[str, Primitive] = {
    "add": Primitive(
        lambda x, y: x + y,
        jvp=lambda xs, dxs: op("plus", dxs[0], dxs[1]),
        vjp=lambda xs, d: [op("sum_to", xs[0], d), op("sum_to", xs[1], d)],
    ),
    "sub": Primitive(
        lambda x, y: x - y,
        jvp=lambda xs, dxs: op("plus", dxs[0], op("lneg", dxs[1])),
        vjp=lambda xs, d: [op("sum_to", xs[0], d), op("sum_to", xs[1], op("lneg", d))],
    ),
    "mul": Primitive(
        lambda x, y: x * y,
        jvp=lambda xs, dxs: op("plus", op("scale", xs[1], dxs[0]), op("scale", xs[0], dxs[1])),
        vjp=lambda xs, d: [
            op("sum_to", xs[0], op("scale", xs[1], d)),
            op("sum_to", xs[1], op("scale", xs[0], d)),
        ],
    ),
    "div": Primitive(
        lambda x, y: x / y,
        jvp=lambda xs, dxs: op(
            "plus",
            op("scale", op("div", Const(1.0), xs[1]), dxs[0]),
            op("scale", op("neg", op("div", xs[0], op("mul", xs[1], xs[1]))), dxs[1]),
        ),
        vjp=lambda xs, d: [
            op("sum_to", xs[0], op("scale", op("div", Const(1.0), xs[1]), d)),
            op("sum_to", xs[1], op("scale", op("neg", op("div", xs[0], op("mul", xs[1], xs[1]))), d)),
        ],
    ),
    "neg": Primitive(
        lambda x: -x,
        jvp=lambda xs, dxs: op("lneg", dxs[0]),
        vjp=lambda xs, d: [op("lneg", d)],
    ),
    # x ** n for a constant exponent n (the second argument is not differentiated)
    "pow": Primitive(
        lambda x, n: x**n,
        jvp=lambda xs, dxs: op("scale", op("mul", xs[1], op("pow", xs[0], op("sub", xs[1], Const(1.0)))), dxs[0]),
        vjp=lambda xs, d: [op("scale", op("mul", xs[1], op("pow", xs[0], op("sub", xs[1], Const(1.0)))), d), Z],
    ),
    "sin": _unary(np.sin, lambda x: op("cos", x)),
    "cos": _unary(np.cos, lambda x: op("neg", op("sin", x))),
    "exp": _unary(np.exp, lambda x: op("exp", x)),
    "log": _unary(np.log, lambda x: op("div", Const(1.0), x)),
    "sqrt": _unary(np.sqrt, lambda x: op("div", Const(0.5), op("sqrt", x))),
    "tanh": _unary(np.tanh, lambda x: op("sub", Const(1.0), op("pow", op("tanh", x), Const(2.0)))),
    # Vector <-> scalar
    "sum": Primitive(
        lambda v: float(np.sum(v)),
        jvp=lambda xs, dxs: op("lsum", dxs[0]),
        vjp=lambda xs, d: [op("expand", xs[0], d)],
    ),
    "replicate": Primitive(  # replicate(x, n) for a constant length n
        lambda x, n: np.full(n, x, dtype=float),
        jvp=lambda xs, dxs: op("lreplicate", xs[1], dxs[0]),
        vjp=lambda xs, d: [op("lsum", d), Z],
    ),
    # Comparisons produce booleans, i.e. values of the sum type () + ()
    "lt": _nondiff(lambda x, y: _bool(x < y)),
    "le": _nondiff(lambda x, y: _bool(x <= y)),
    "gt": _nondiff(lambda x, y: _bool(x > y)),
    "ge": _nondiff(lambda x, y: _bool(x >= y)),
}
