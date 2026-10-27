"""Higher-order functions: CHAD differentiates lambdas and closures directly.

Python closures are normally just inlined away while tracing. ``chad.fn``
instead makes a function a *value in the program*, which CHAD then
differentiates as such: the reverse-mode cotangent of a function value is a
copower, i.e. a list of (argument, output cotangent) pairs collected from every
call site, which is folded back into the cotangents of the variables the
function captured.
"""

import chad
from chad import fn, sin


@chad.function
def apply_twice(x, y):
    twice = fn(lambda g, v: g(g(v)))  # takes a function as an argument
    affine = fn(lambda v: y * v + x)  # captures x and y
    return twice(affine, x)  # y * (y * x + x) + x


print("apply_twice(0.8, 1.3) =", apply_twice(0.8, 1.3))
print("gradient              =", chad.grad(apply_twice)(0.8, 1.3))
x, y = 0.8, 1.3
print("expected              =", (y * y + y + 1, 2 * x * y + x))


@chad.function
def curried(a, b):
    mul_plus_sin = fn(lambda p: fn(lambda q: p * q + sin(p)))  # returns a closure
    return mul_plus_sin(a)(b)


print("\ncurried gradient      =", chad.grad(curried)(0.5, 2.0))


@chad.function
def compose_many(x, k):
    # Build a function by composing closures, then apply it.
    step = fn(lambda v: sin(v) * k)
    compose = fn(lambda f, g: fn(lambda v: g(f(v))))
    h = compose(step, compose(step, step))
    return h(x)


print("compose gradient      =", chad.grad(compose_many)(0.3, 1.7))

print("\n# Reverse derivative of apply_twice:")
print(chad.reverse(apply_twice).source)
