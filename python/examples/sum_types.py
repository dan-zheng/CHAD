"""Sum types and control flow.

Comparisons produce booleans, which are values of the sum type ``() + ()``.
``chad.cond`` branches on them; ``chad.case`` branches on general sums made
with ``inl`` / ``inr``. The cotangent of a sum value is a "linear either":
zero, or a cotangent for whichever side was actually taken.
"""

import numpy as np

import chad
from chad import case, cond, cos, inl, inr, sin, vmap


@chad.function
def abs_(x):
    return cond(x < 0, lambda: -x, lambda: x)


print("abs' at -2, 3:", chad.grad(abs_)(-2.0), chad.grad(abs_)(3.0))


@chad.function
def relu_layer(v, w, b):
    """Elementwise relu(w * x + b), summed."""
    return chad.sum(vmap(lambda x: cond(w * x + b > 0, lambda: w * x + b, lambda: 0.0), v))


v = np.array([-1.0, 0.5, 2.0])
print("relu_layer grad:", chad.grad(relu_layer)(v, 1.5, -0.25))


@chad.function
def safe_log_or_square(x):
    # A sum value carrying different payloads in each branch
    s = cond(x > 1, lambda: inl(x), lambda: inr((x, x * x)))
    return case(s, lambda a: chad.log(a), lambda p: p[1] * cos(p[0]))


for x in (0.5, 2.0):
    print(f"d/dx at {x}:", chad.grad(safe_log_or_square)(x))

print("\n# Reverse derivative of abs':")
print(chad.reverse(abs_).source)
