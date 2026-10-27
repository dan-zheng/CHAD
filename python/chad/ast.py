"""The expression language.

One small lambda calculus serves as both the *source* language (what users
write, via tracing) and the *target* language (what CHAD produces). As in the
paper, the source language is a sublanguage of the target: the target merely
uses extra primitive operations (``plus``, ``scale``, ``get``, ...) that
implement the commutative-monoid structure of tangent/cotangent types.

Variables are named (not De Bruijn as in the Haskell implementation); every
binder introduced by tracing or by the transformations gets a globally unique
name, so substitution never needs to worry about capture.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

_counter = itertools.count(1)
_taken: set[str] = set()


def fresh(prefix: str = "v") -> str:
    """A globally unique variable name."""
    while (name := f"{prefix}{next(_counter)}") in _taken:
        pass
    return name


def reserve(name: str) -> str:
    """Make sure ``fresh`` never returns ``name`` (e.g. a user's parameter name)."""
    _taken.add(name)
    return name


class Expr:
    __slots__ = ()


@dataclass(frozen=True, eq=False)
class Var(Expr):
    name: str


@dataclass(frozen=True, eq=False)
class Const(Expr):
    value: Any  # a float, numpy array, int (static argument), str (variable name), or ZERO


@dataclass(frozen=True, eq=False)
class Lam(Expr):
    params: tuple[str, ...]  # source lambdas are unary; target helpers may take several
    body: Expr
    linear: bool = False  # the paper's LinFun: a monoid homomorphism (so f(ZERO) = ZERO)


@dataclass(frozen=True, eq=False)
class App(Expr):
    fun: Expr
    args: tuple[Expr, ...]


@dataclass(frozen=True, eq=False)
class Let(Expr):
    name: str
    rhs: Expr
    body: Expr


@dataclass(frozen=True, eq=False)
class Tuple(Expr):
    items: tuple[Expr, ...]  # Tuple(()) is the unit value


@dataclass(frozen=True, eq=False)
class Proj(Expr):
    index: int
    tup: Expr


@dataclass(frozen=True, eq=False)
class Inl(Expr):
    value: Expr


@dataclass(frozen=True, eq=False)
class Inr(Expr):
    value: Expr


@dataclass(frozen=True, eq=False)
class Case(Expr):
    scrut: Expr
    lname: str
    left: Expr
    rname: str
    right: Expr


@dataclass(frozen=True, eq=False)
class Map(Expr):
    """``map (\\name -> body) xs`` over a vector of scalars (the paper's map1)."""

    name: str
    body: Expr
    xs: Expr


@dataclass(frozen=True, eq=False)
class Op(Expr):
    """A primitive operation: a differentiable source operation (see ``ops``)
    or a target-language runtime helper (see ``runtime``)."""

    name: str
    args: tuple[Expr, ...]


# ---------------------------------------------------------------------------
# Small constructors used throughout the transformations


def var(name: str) -> Var:
    return Var(name)


def lam(*params_and_body):
    *params, body = params_and_body
    return Lam(tuple(params), body)


def app(f: Expr, *args: Expr) -> App:
    return App(f, tuple(args))


def tup(*items: Expr) -> Tuple:
    return Tuple(tuple(items))


def op(name: str, *args: Expr) -> Op:
    return Op(name, tuple(args))


def fst(e: Expr) -> Proj:
    return Proj(0, e)


def snd(e: Expr) -> Proj:
    return Proj(1, e)


def let(rhs: Expr, body_fn, prefix: str = "v") -> Let:
    """HOAS-style let: ``let(rhs, lambda x: body)`` binds a fresh variable."""
    name = fresh(prefix)
    return Let(name, rhs, body_fn(Var(name)))


def linlam(body_fn, prefix: str = "d") -> Lam:
    """A (linear) lambda with a fresh parameter: ``linlam(lambda d: ...)``."""
    name = fresh(prefix)
    return Lam((name,), body_fn(Var(name)), linear=True)
