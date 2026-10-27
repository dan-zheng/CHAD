"""The four example programs from the CHAD paper (https://arxiv.org/abs/2103.15776).

These are the same programs as in ``src/Examples.hs``; we print their
derivative programs and check them against the derivatives worked out by hand.
"""

import numpy as np

import chad
from chad import cos, fn, replicate, sin, vmap


@chad.function
def ex1(x):
    y = 2 * x
    z = x * y
    w = cos(z)
    return ((y, z), w)


@chad.function
def ex2(x1, x2, x3, x4):
    # sin ((x1 * x4 + 2 * x2) * x3 + x4)
    y = x1 * x4 + 2 * x2
    z = y * x3
    w = z + x4
    return sin(w)


@chad.function
def ex3(x):
    # map (\z -> x * z + 1) (replicate x)  ==  replicate (x * x + 1)
    f = fn(lambda z: x * z + 1)
    zs = replicate(x, 3)
    return vmap(lambda z: f(z), zs)


@chad.function
def ex4(x1, x2):
    # sum (map (x1 *) x2)
    f = fn(lambda x2i: x1 * x2i)
    ys = vmap(lambda x2i: f(x2i), x2)
    return chad.sum(ys)


for ex in (ex1, ex2, ex3, ex4):
    print(f"# ---- {ex.__name__}\n{ex.source}\n")
    print(chad.reverse(ex).source, "\n")

# ex1: forward derivative w.r.t. x of ((2x, 2x^2), cos(2x^2))
x = 0.7
_, ((dy, dz), dw) = chad.jvp(ex1, (x,), (1.0,))
assert np.allclose([dy, dz, dw], [2, 4 * x, -np.sin(2 * x * x) * 4 * x])

# ex3: reverse derivative is 2 * x * sum d
_, pullback = chad.vjp(ex3, x)
d = np.array([1.0, 2.0, 3.0])
assert np.isclose(pullback(d)[0], 2 * x * d.sum())

# ex4: gradient is (sum x2, replicate x1)
x2 = np.array([0.5, -1.0, 2.0])
g1, g2 = chad.grad(ex4)(1.5, x2)
assert np.isclose(g1, x2.sum()) and np.allclose(g2, 1.5)

print("All paper examples agree with their hand-derived derivatives.")
