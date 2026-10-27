"""Tests: CHAD forward mode, CHAD reverse mode and finite differences agree.

This mirrors ``src/Test.hs``: for every example program we check that the
primal results of both derivative programs match the original program, and
that the Jacobians computed by forward AD, reverse AD (with and without
simplification) and central finite differences all agree.
"""

import numpy as np
import pytest

import chad
from chad import Inl, Inr, case, cond, cos, exp, fn, inl, inr, log, replicate, sin, sqrt, tanh, vmap

# ---------------------------------------------------------------------------
# Flattening values to vectors, so we can build Jacobian matrices


def flatten(v) -> list[float]:
    if isinstance(v, tuple):
        return [x for item in v for x in flatten(item)]
    if isinstance(v, (Inl, Inr)):
        return flatten(v.value)
    return list(np.ravel(np.asarray(v, dtype=float)))


def unflatten(like, flat: list[float]):
    """Rebuild a value shaped like ``like`` from the front of ``flat`` (consumed)."""
    if isinstance(like, tuple):
        return tuple(unflatten(item, flat) for item in like)
    if isinstance(like, (Inl, Inr)):
        return type(like)(unflatten(like.value, flat))
    if isinstance(like, np.ndarray):
        return np.array([flat.pop(0) for _ in range(like.size)]).reshape(like.shape)
    return flat.pop(0)


def basis(like):
    n = len(flatten(like))
    return [unflatten(like, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]


def jac_forward(f, args, simplify=True):
    cols = [flatten(chad.jvp(f, args, t, simplify=simplify)[1]) for t in basis(args)]
    return np.array(cols).T


def jac_reverse(f, args, simplify=True):
    out, pullback = chad.vjp(f, *args, simplify=simplify)
    return np.array([flatten(pullback(e)) for e in basis(out)])


def jac_fd(f, args, h=1e-6):
    x0 = flatten(args)
    cols = []
    for j in range(len(x0)):
        hi = list(x0); hi[j] += h
        lo = list(x0); lo[j] -= h
        cols.append((np.array(flatten(f(*unflatten(args, hi)))) - np.array(flatten(f(*unflatten(args, lo))))) / (2 * h))
    return np.array(cols).T


def check(f, *args):
    f = chad.function(f)
    args = tuple(chad._coerce(a) for a in args)
    expected = f(*args)
    np.testing.assert_allclose(flatten(chad.reverse(f)(*args)[0]), flatten(expected))
    np.testing.assert_allclose(flatten(chad.forward(f)(*args)[0]), flatten(expected))
    fd = jac_fd(f, args)
    for simplify in (True, False):
        np.testing.assert_allclose(jac_forward(f, args, simplify), fd, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(jac_reverse(f, args, simplify), fd, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# The four examples from the paper


def paper_ex1(x):
    y = 2 * x
    z = x * y
    w = cos(z)
    return ((y, z), w)


def paper_ex2(x1, x2, x3, x4):
    y = x1 * x4 + 2 * x2
    z = y * x3
    w = z + x4
    return sin(w)


def paper_ex3(x):
    f = fn(lambda z: x * z + 1)
    zs = replicate(x, 4)
    return vmap(lambda z: f(z), zs)


def paper_ex4(x1, x2):
    f = fn(lambda x2i: x1 * x2i)
    ys = vmap(lambda x2i: f(x2i), x2)
    return chad.sum(ys)


VEC = np.array([0.5, -1.25, 2.0])

EXAMPLES = [
    (paper_ex1, (0.7,)),
    (paper_ex2, (0.3, -1.1, 0.8, 1.7)),
    (paper_ex3, (1.3,)),
    (paper_ex4, (1.5, VEC)),
]


# ---------------------------------------------------------------------------
# More programs: arithmetic, tuples, higher-order functions, sums, maps


def polynomial(x):
    square = fn(lambda y: y * y)
    return 2 * square(x) + 7 * x + 3


def ops(x, y):
    return exp(x) / (1 + y * y) - log(1 + x * x) + sqrt(2 + sin(y)) * tanh(x) + (-x) ** 3


def tuples(p, y):
    return (p[0] * y, (p[1] - y, p[0] * p[1]))


def apply_twice(x, y):
    twice = fn(lambda g, v: g(g(v)))
    scale_y = fn(lambda v: y * v + x)
    return twice(scale_y, x)


def curried(x, y):
    add = fn(lambda a: fn(lambda b: a * b + sin(a)))
    return add(x)(y) + add(y)(x)


def closure_returned(x, y):
    make = fn(lambda a: fn(lambda b: a * b * x))
    g = make(y)
    return g(x) + g(y)


def abs_(x):
    return cond(x < 0, lambda: -x, lambda: x)


def piecewise(x, y):
    s = cond(x * y > 1, lambda: inl(x * y), lambda: inr((x, y)))
    return case(s, lambda a: sin(a), lambda p: p[0] * cos(p[1]))


def relu_map(v, b):
    return vmap(lambda x: cond(x + b > 0, lambda: x + b, lambda: 0.0), v)


def map_closure(v, w):
    return chad.sum(vmap(lambda x: x * x * w + sin(x * w), v))


def map_of_function(v, a):
    f = fn(lambda x: a * x * x)
    return vmap(lambda x: f(x) + f(a), v)


def broadcasting(v, s):
    return chad.sum(s * v + v * v / s)


def softmax_xent(logits, target):
    m = exp(logits)
    return log(chad.sum(m)) - chad.sum(logits * target)


EXAMPLES += [
    (polynomial, (1.3,)),
    (ops, (0.4, 0.9)),
    (tuples, ((1.2, -0.4), 0.6)),
    (apply_twice, (0.8, 1.3)),
    (curried, (0.8, 1.3)),
    (closure_returned, (0.8, 1.3)),
    (abs_, (-0.6,)),
    (abs_, (0.6,)),
    (piecewise, (0.5, 0.7)),
    (piecewise, (1.5, 1.7)),
    (relu_map, (VEC, 0.3)),
    (map_closure, (VEC, 0.7)),
    (map_of_function, (VEC, 0.7)),
    (broadcasting, (VEC, 1.7)),
    (softmax_xent, (VEC, np.array([0.0, 1.0, 0.0]))),
]


@pytest.mark.parametrize("f,args", EXAMPLES, ids=lambda x: getattr(x, "__name__", ""))
def test_derivatives_agree(f, args):
    check(f, *args)


# ---------------------------------------------------------------------------
# Specific values and API


def test_paper_ex4_gradient():
    # d/dx1 = sum x2, d/dx2 = replicate x1
    g1, g2 = chad.grad(paper_ex4)(1.5, VEC)
    assert g1 == pytest.approx(VEC.sum())
    np.testing.assert_allclose(g2, np.full(3, 1.5))


def test_paper_ex3_vjp():
    # equivalent to replicate (x * x + 1), so the derivative is 2 * x * sum d
    out, pullback = chad.vjp(paper_ex3, 1.3)
    np.testing.assert_allclose(out, np.full(4, 1.3 * 1.3 + 1))
    (g,) = pullback(np.array([1.0, 2.0, 3.0, 4.0]))
    assert g == pytest.approx(2 * 1.3 * 10)


def test_grad_single_argument_unwrapped():
    assert chad.grad(polynomial)(2.0) == pytest.approx(4 * 2.0 + 7)


def test_value_and_grad_plain_function():
    v, (gx, gy) = chad.value_and_grad(lambda x, y: x * y)(3.0, 4.0)
    assert (v, gx, gy) == (12.0, 4.0, 3.0)


def test_unused_argument_gets_zero():
    assert chad.grad(lambda x, y: x * x)(3.0, 4.0) == (6.0, 0.0)
    np.testing.assert_array_equal(chad.grad(lambda x, v: x)(1.0, VEC)[1], np.zeros(3))


def test_grad_requires_scalar_output():
    with pytest.raises(TypeError):
        chad.grad(paper_ex1)(1.0)


def test_python_control_flow_is_rejected():
    def bad(x):
        return x if x > 0 else -x

    with pytest.raises(TypeError, match="cond"):
        chad.function(bad)


def test_source_is_shown():
    f = chad.function(paper_ex1)
    assert "cos(" in f.source
    assert "def paper_ex1_rev(x):" in chad.reverse(f).source
    assert "def paper_ex1_fwd(x):" in chad.forward(f).source


def test_simplification_shrinks_code():
    f = chad.function(paper_ex2)
    assert len(chad.reverse(f).source) < len(chad.reverse(f, simplify=False).source) / 2
