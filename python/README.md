# CHAD in Python

A small Python library for automatic differentiation using **CHAD**
(Combinatory Homomorphic Automatic Differentiation;
[Vákár & Smeding, 2021](https://arxiv.org/abs/2103.15776)). It is a port of the
Haskell reference implementation in this repository.

CHAD is a *source-code transformation*. A program becomes a new program that
computes the original result together with a **linear function**: the
derivative (forward mode) or the transposed derivative (reverse mode). The
transformation is *homomorphic*: each language construct is mapped to its own
derivative, independently of everything around it. This makes it easy to see
that it is correct, and it covers higher-order functions, tuples and sum types
as well as arithmetic.

```python
import chad
from chad import sin

@chad.function
def f(x, y):
    z = x * y
    return sin(z) + 2 * x

f(1.0, 2.0)                           # 2.909...
chad.grad(f)(1.0, 2.0)                # (1.167..., -0.416...)   reverse mode
chad.jvp(f, (1.0, 2.0), (1.0, 0.0))   # (2.909..., 1.167...)    forward mode
print(chad.reverse(f).source)         # the derivative is a program; here it is
```

```python
def f_rev(x, y):
    v1 = (x * y)
    v2 = sin(v1)
    v3 = (2 * x)
    def f70(d34):
        d11 = scale(cos(v1), sum_to(v2, d34))
        return plus(single('x', plus(sum_to(x, scale(2, sum_to(v3, d34))), sum_to(x, scale(y, d11)))),
                    single('y', sum_to(y, scale(x, d11))))
    return ((v2 + v3), f70)
```

(`sum_to(like, d)` undoes numpy broadcasting between scalars and vectors. It
does nothing when the shapes already match.)

## Installation

```bash
pip install -e python/
```

The only dependency is numpy. To run the tests and examples:

```bash
cd python && pytest && python examples/quickstart.py
```

## API

| | |
|---|---|
| `@chad.function` | Trace a Python function into a program. The result is still callable. |
| `chad.grad(f)(*args)` | Gradient of a scalar-valued `f` (a tuple, or a single value for one argument). |
| `chad.value_and_grad(f)(*args)` | `(f(*args), gradient)`. |
| `chad.vjp(f, *args)` | `(out, pullback)`: `pullback(d)` gives one cotangent per argument. |
| `chad.jvp(f, args, tangents)` | `(out, tangent of out)`. |
| `chad.reverse(f)` / `chad.forward(f)` | The compiled derivative program. Calling it gives `(out, linear_fn)`. `.source` shows its Python code. |
| `f.source` | The traced program itself. |

`grad`, `vjp` and `jvp` also accept plain Python functions. Values can be floats,
1-D numpy arrays (vectors), tuples of these, and sum values (`chad.Inl` /
`chad.Inr`).

### The language

Inside a traced function you can use the arithmetic operators (`+ - * / **`
with a constant exponent, and comparisons), plus `sin cos exp log sqrt tanh`,
`chad.sum(v)` and `replicate(x, n)`. Tuples are indexed with `t[0]`, `t[1]`, …

Python control flow and Python closures run *during tracing*, so they never
appear in the program. These constructs put real higher-order and branching
code into the program, where CHAD differentiates it:

| | |
|---|---|
| `fn(lambda x: ...)` | A first-class function value (a lambda) in the program. It may capture variables, be passed around, returned, and called. Multi-argument lambdas take a tuple. |
| `vmap(lambda x: ..., v)` | Map over a vector. The body may capture variables. |
| `cond(pred, lambda: a, lambda: b)` | Branch on a comparison. Booleans are the sum type `() + ()`. |
| `inl(v)`, `inr(v)`, `case(s, lambda a: ..., lambda b: ...)` | General sum types. |

```python
@chad.function
def apply_twice(x, y):
    twice = fn(lambda g, v: g(g(v)))   # takes a function as argument
    affine = fn(lambda v: y * v + x)   # closes over x and y
    return twice(affine, x)

chad.grad(apply_twice)(0.8, 1.3)       # (3.99, 2.88)
```

## Examples

* [`examples/quickstart.py`](examples/quickstart.py): a tour of the API.
* [`examples/paper_examples.py`](examples/paper_examples.py): the four example programs from the paper, checked against their hand-derived derivatives.
* [`examples/higher_order.py`](examples/higher_order.py): functions as arguments, closures, currying, composition.
* [`examples/sum_types.py`](examples/sum_types.py): `cond`, `case`, and a relu layer.
* [`examples/regression.py`](examples/regression.py): fitting a small model by gradient descent.

## How it works

The modules match the Haskell implementation:

| Python | Haskell | What |
|---|---|---|
| [`chad/ast.py`](chad/ast.py) | `SourceLanguage`, `TargetLanguage` | One small lambda calculus with named variables. The target language is the source language plus some runtime helpers. |
| [`chad/ops.py`](chad/ops.py) | `Operation`, `dfOp`, `drOp` | Primitive operations and their (transposed) derivatives. |
| [`chad/transform.py`](chad/transform.py) | `ForwardAD`, `ReverseAD` | **The CHAD transformations `df` and `dr`.** |
| [`chad/runtime.py`](chad/runtime.py) | `Types` (`LT`, `LFun`, `Copower`, `LEither`) | The monoid structure (`zero`, `plus`) of tangent and cotangent values. |
| [`chad/simplify.py`](chad/simplify.py) | `TargetLanguage.Simplify` | Cleans up the transformed code. |
| [`chad/codegen.py`](chad/codegen.py) | `Concrete`, `ToConcrete` | Compiles target programs to readable Python. |
| [`chad/trace.py`](chad/trace.py) | (none) | Tracing frontend: Python functions become source programs. |
| [`tests/test_chad.py`](tests/test_chad.py) | `Test`, `FinDiff` | Forward mode, reverse mode and finite differences must agree. |

**Programs are functions of their environment.** For a term `Γ ⊢ t : τ`, CHAD produces

```
df(t) : (Df₁[τ], Df₂[Γ] ⊸ Df₂[τ])      forward
dr(t) : (Dr₁[τ], Dr₂[τ] ⊸ Dr₂[Γ])      reverse
```

the primal value paired with a linear map between tangent or cotangent spaces.
Each rule in `transform.py` translates one constructor of the Haskell `df`/`dr`.
For example, the reverse rule for `let`:

```python
case Let(x, rhs, body):
    # let r = dr(rhs); x = fst r; b = dr(body)
    # in (fst b, \d. let ct = snd b d in drop(ct, x) + snd r (ct[x]))
```

This port differs from the Haskell code in a few places:

* **Untyped, with a symbolic zero.** CHAD needs every (co)tangent type to be a
  commutative monoid. Haskell gets `zero`/`plus` from types; here a single
  polymorphic `ZERO` is the identity at every type, and `plus` dispatches on
  the runtime value. Each type is represented as follows:

  | type | tangent / cotangent |
  |---|---|
  | scalar, vector | float, numpy array |
  | tuple | tuple (right-padded with `ZERO`) |
  | `a + b` | `ZERO`, `Inl(da)` or `Inr(db)`: the paper's `LEither` |
  | `a → b`, forward | a function `Df₁[a] → Df₂[b]` |
  | `a → b`, reverse | a `Copower`: a list of `(argument, cotangent)` pairs |
  | environment `Γ` | a sparse `dict` from variable names to (co)tangents |

* **Named variables, not De Bruijn indices.** Every binder gets a globally
  unique name, so substitution cannot capture variables.
* **Linear functions are ordinary lambdas** flagged `linear=True`. The
  simplifier uses the flag (`f(ZERO) = ZERO`). Like the Haskell `Concrete`
  language, there is no separate linear sublanguage.
* **n-ary tuples and multi-argument primitives**, where the paper uses pairs
  and unary operations on pairs.
* **Broadcasting** between scalars and vectors in arithmetic, made
  differentiable by `sum_to`.

The reverse mode is the straightforward CHAD algorithm from the paper. It
recomputes a function's primal inside the backward pass for each call, and it
does not use the efficiency improvements from later work. It is a reference
implementation, meant to be read and checked rather than to be fast.
