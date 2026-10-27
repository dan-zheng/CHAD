"""Runtime support for generated code.

The paper requires every (co)tangent type to be a commutative monoid, with
type-directed ``zero`` and ``plus``. Our language is untyped, so instead we use
a single polymorphic symbolic zero, ``ZERO``, that acts as the additive
identity at every type, and a ``plus`` that dispatches on the runtime
representation:

==================  ==========================  ===============================
type                tangent / cotangent         representation
==================  ==========================  ===============================
scalar, vector      itself                      float, numpy array
tuple               tuple of (co)tangents       tuple (right-padded with ZERO)
sum  a + b          LEither: zero, inl, inr     ZERO, Inl, Inr
function (fwd)      ``Df1 a -> Df2 b``          Python callable
function (rev)      copower ``!(Dr1 a) ⊗ Dr2 b``  ``Copower`` (list of pairs)
environment         product of all variables    sparse ``dict`` name -> value
==================  ==========================  ===============================

Every helper marked *linear* is a monoid homomorphism in its linear argument,
which therefore may be ``ZERO``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class _Zero:
    __slots__ = ()

    def __repr__(self) -> str:
        return "ZERO"

    def __reduce__(self):
        return "ZERO"


ZERO = _Zero()


@dataclass(frozen=True)
class Inl:
    value: Any


@dataclass(frozen=True)
class Inr:
    value: Any


class Copower(list):
    """A formal sum of pairs ``(x, d)``: the cotangent of a function value."""

    def __repr__(self) -> str:
        return f"Copower({list.__repr__(self)})"


# ---------------------------------------------------------------------------
# The monoid structure


def plus(a, b):
    if a is ZERO:
        return b
    if b is ZERO:
        return a
    if isinstance(a, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = plus(out[k], v) if k in out else v
        return out
    if isinstance(a, tuple):
        n = max(len(a), len(b))
        a = a + (ZERO,) * (n - len(a))
        b = b + (ZERO,) * (n - len(b))
        return tuple(plus(x, y) for x, y in zip(a, b))
    if isinstance(a, Copower):
        return Copower(a + b)
    if isinstance(a, Inl) and isinstance(b, Inl):
        return Inl(plus(a.value, b.value))
    if isinstance(a, Inr) and isinstance(b, Inr):
        return Inr(plus(a.value, b.value))
    if isinstance(a, (Inl, Inr)) or isinstance(b, (Inl, Inr)):
        raise ValueError("plus: inconsistent sum-type cotangents")
    if callable(a):
        return lambda x: plus(a(x), b(x))
    return a + b


def _linear(f):
    """Make ``f`` return ZERO when its last (linear) argument is ZERO."""

    def wrapped(*args):
        return ZERO if args[-1] is ZERO else f(*args)

    wrapped.__name__ = f.__name__
    return wrapped


# ---------------------------------------------------------------------------
# Sparse environments (the paper's Df2Env / Dr2Env products)


def single(name, v):
    """One-hot environment (co)tangent. Linear in ``v``."""
    return ZERO if v is ZERO else {name: v}


def get(env, name):
    """Project one variable out of an environment (co)tangent. Linear."""
    return ZERO if env is ZERO else env.get(name, ZERO)


def drop(env, name):
    """Remove a variable from an environment (co)tangent. Linear."""
    if env is ZERO or name not in env:
        return env
    out = dict(env)
    del out[name]
    return out or ZERO


def extend(env, name, v):
    """Add a variable to an environment tangent. Linear in (env, v)."""
    if v is ZERO:
        return env
    out = {} if env is ZERO else dict(env)
    out[name] = v
    return out


# ---------------------------------------------------------------------------
# Linear operations


@_linear
def lproj(i, t):
    return t[i] if i < len(t) else ZERO


@_linear
def scale(c, d):
    return c * d


@_linear
def lneg(d):
    return -d


@_linear
def lsum(d):
    return float(np.sum(d))


@_linear
def lreplicate(n, d):
    return np.full(n, d, dtype=float)


@_linear
def expand(like, d):
    """Broadcast a scalar cotangent to the shape of vector ``like``."""
    return np.full(np.shape(like), d, dtype=float)


@_linear
def sum_to(like, d):
    """Sum a cotangent down to the shape of ``like`` (undoes broadcasting)."""
    if np.ndim(like) == 0 and np.ndim(d) > 0:
        return float(np.sum(d))
    return d


@_linear
def tapp(x, f):
    """Apply a forward-mode function tangent to a primal argument."""
    return f(x)


@_linear
def from_inl(d):
    if isinstance(d, Inl):
        return d.value
    raise ValueError("cotangent of inl is not inl")


@_linear
def from_inr(d):
    if isinstance(d, Inr):
        return d.value
    raise ValueError("cotangent of inr is not inr")


@_linear
def copow(x, d):
    """The singleton copower ``[(x, d)]``. Linear in ``d``."""
    return Copower([(x, d)])


def fold(f, c):
    """``sum [f x d | (x, d) <- c]`` for a copower ``c``. Linear in ``c``."""
    out = ZERO
    if c is not ZERO:
        for x, d in c:
            out = plus(out, f(x, d))
    return out


@_linear
def zip_(xs, ds):
    """Zip a primal vector with a cotangent vector into a copower. Linear in ``ds``."""
    return Copower(zip(xs, ds))


@_linear
def zipwith(f, xs, ds):
    """Elementwise ``f x d`` where ``f x`` is linear. Linear in ``ds``."""
    return np.array([densify(f(x, d), 0.0) for x, d in zip(xs, ds)], dtype=float)


def map_(f, xs):
    return np.array([densify(f(x), 0.0) for x in xs], dtype=float)


# ---------------------------------------------------------------------------
# Converting sparse (co)tangents back to dense values at the API boundary


def densify(d, like):
    """Turn a (co)tangent with ``ZERO``s into a concrete value shaped like ``like``."""
    if isinstance(like, tuple):
        if d is ZERO:
            d = ()
        d = tuple(d) + (ZERO,) * (len(like) - len(d))
        return tuple(densify(x, y) for x, y in zip(d, like))
    if d is ZERO:
        if isinstance(like, np.ndarray):
            return np.zeros_like(like, dtype=float)
        if isinstance(like, (int, float)):
            return 0.0
        return ZERO
    if isinstance(like, (Inl, Inr)) and isinstance(d, (Inl, Inr)):
        return type(d)(densify(d.value, like.value))
    return d
