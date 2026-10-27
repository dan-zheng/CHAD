"""A tour of the API."""

import numpy as np

import chad
from chad import cos, sin


@chad.function
def f(x, y):
    z = x * y
    return sin(z) + 2 * x


# A chad.function is an ordinary callable...
print("f(1, 2) =", f(1.0, 2.0))

# ...that can be differentiated in reverse mode,
print("grad f  =", chad.grad(f)(1.0, 2.0))  # (y cos(xy) + 2, x cos(xy))

# with arbitrary output cotangents,
out, pullback = chad.vjp(f, 1.0, 2.0)
print("vjp     =", pullback(10.0))

# or in forward mode.
out, tangent = chad.jvp(f, (1.0, 2.0), (1.0, 0.0))
print("jvp     =", tangent)

# Outputs can be tuples; inputs can be tuples and numpy vectors.
@chad.function
def g(p, v):
    return (p[0] * chad.sum(v), cos(p[1]) * v)

out, pullback = chad.vjp(g, (2.0, 0.5), np.array([1.0, 2.0, 3.0]))
print("g       =", out)
print("vjp g   =", pullback((1.0, np.array([0.0, 1.0, 0.0]))))

# CHAD is a source-code transformation: derivatives are programs too.
print("\n# The traced program:")
print(f.source)
print("\n# Its reverse derivative: returns (primal, backpropagator)")
print(chad.reverse(f).source)
print("\n# Its forward derivative: returns (primal, tangent map)")
print(chad.forward(f).source)
