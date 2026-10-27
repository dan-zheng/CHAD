"""Fit a tiny model by gradient descent, using CHAD reverse mode.

The model is ``y = a * tanh(w * x + b) + c``, applied to a whole data vector
with ``vmap``; the loss is the mean squared error.
"""

import numpy as np

import chad
from chad import tanh, vmap

rng = np.random.default_rng(0)
xs = np.linspace(-2, 2, 40)
ys = 1.5 * np.tanh(2.0 * xs - 0.5) + 0.3 + 0.05 * rng.normal(size=xs.shape)


@chad.function
def loss(params, xs, ys):
    a, w, b, c = params[0], params[1], params[2], params[3]
    preds = vmap(lambda x: a * tanh(w * x + b) + c, xs)
    err = preds - ys
    return chad.sum(err * err) / 40


value_and_grad = chad.value_and_grad(loss)
params = (1.0, 1.0, 0.0, 0.0)
for step in range(301):
    l, (g, _, _) = value_and_grad(params, xs, ys)
    params = tuple(p - 0.2 * gp for p, gp in zip(params, g))
    if step % 50 == 0:
        print(f"step {step:3d}  loss {l:.5f}")

print("fitted (a, w, b, c) =", tuple(round(float(p), 3) for p in params))
print("true   (a, w, b, c) = (1.5, 2.0, -0.5, 0.3)")
