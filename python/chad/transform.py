"""CHAD: the forward- and reverse-mode AD code transformations.

This is the heart of the library and a direct transcription of
``ForwardAD.hs`` and ``ReverseAD.hs``. For a term ``Γ ⊢ t : τ``:

* forward:  ``Df₁[Γ] ⊢ df(t) : (Df₁[τ], Df₂[Γ] ⊸ Df₂[τ])``
* reverse:  ``Dr₁[Γ] ⊢ dr(t) : (Dr₁[τ], Dr₂[τ] ⊸ Dr₂[Γ])``

That is, each term becomes a pair of its primal value and a *linear* function
between (co)tangent spaces, where the (co)tangent of the environment ``Γ`` is a
sparse dict from variable names to (co)tangents. The transformation is
homomorphic: every rule only combines the transformed sub-terms, so each case
below can be checked in isolation.

Primal variables keep their names, so e.g. a source variable ``x`` stands for
``Df₁[x]`` in the transformed program.
"""

from __future__ import annotations

from .ast import (App, Case, Const, Expr, Inl, Inr, Lam, Let, Map, Op, Proj,
                  Tuple, Var, app, fresh, fst, let, linlam, op, snd, tup)
from .ops import PRIMITIVES, Z


def _name(n: str) -> Const:
    return Const(n)


def _plus_all(es: list[Expr]) -> Expr:
    out = es[0]
    for e in es[1:]:
        out = op("plus", out, e)
    return out


# ---------------------------------------------------------------------------
# Reverse mode


def dr(e: Expr) -> Expr:
    match e:
        case Var(x):
            return tup(Var(x), linlam(lambda d: op("single", _name(x), d)))

        case Const():
            return tup(e, linlam(lambda d: Z))

        case Let(x, rhs, body):
            # let r = dr(rhs); x = fst r; b = dr(body)
            # in (fst b, \d. let ct = snd b d in drop(ct, x) + snd r (ct[x]))
            return let(dr(rhs), lambda r: Let(x, fst(r), let(dr(body), lambda b: tup(
                fst(b),
                linlam(lambda d: let(app(snd(b), d), lambda ct: op(
                    "plus",
                    op("drop", ct, _name(x)),
                    app(snd(r), op("get", ct, _name(x))))))))), "r")

        case Lam((x,), body):
            # Primal: a function returning its own (primal, backpropagator) pair.
            # Its cotangent is a copower: a formal sum of (argument, cotangent)
            # pairs, which we fold into the environment cotangent.
            return let(Lam((x,), dr(body)), lambda g: tup(
                lam1(lambda a: let(app(g, a), lambda r: tup(
                    fst(r),
                    linlam(lambda d: op("get", app(snd(r), d), _name(x))))), "x"),
                linlam(lambda c: op("fold", lam2(lambda a, d: op(
                    "drop", app(snd(app(g, a)), d), _name(x))), c), "c")), "g")

        case App(f, (a,)):
            return let(dr(a), lambda ra: let(dr(f), lambda rf: let(app(fst(rf), fst(ra)), lambda res: tup(
                fst(res),
                linlam(lambda d: op(
                    "plus",
                    app(snd(ra), app(snd(res), d)),
                    app(snd(rf), op("copow", fst(ra), d))))), "y"), "r"), "r")

        case Tuple(items):
            return _let_all([dr(x) for x in items], lambda rs: tup(
                tup(*[fst(r) for r in rs]),
                linlam(lambda d: _plus_all(
                    [app(snd(r), op("lproj", Const(i), d)) for i, r in enumerate(rs)]
                ) if rs else Z)))

        case Proj(i, t):
            return let(dr(t), lambda r: tup(
                Proj(i, fst(r)),
                linlam(lambda d: app(snd(r), tup(*[Z] * i, d)))), "r")

        case Inl(v):
            return let(dr(v), lambda r: tup(
                Inl(fst(r)), linlam(lambda d: app(snd(r), op("from_inl", d)))), "r")

        case Inr(v):
            return let(dr(v), lambda r: tup(
                Inr(fst(r)), linlam(lambda d: app(snd(r), op("from_inr", d)))), "r")

        case Case(s, xl, left, xr, right):
            def branch(r, x, body, inj):
                return let(dr(body), lambda b: tup(
                    fst(b),
                    linlam(lambda d: let(app(snd(b), d), lambda ct: op(
                        "plus",
                        op("drop", ct, _name(x)),
                        app(snd(r), inj(op("get", ct, _name(x)))))))), "r")

            return let(dr(s), lambda r: Case(
                fst(r), xl, branch(r, xl, left, Inl), xr, branch(r, xr, right, Inr)), "r")

        case Map(x, body, xs):
            # The paper's map1: the body may use x and anything in scope. Its
            # backpropagator recomputes the per-element pullbacks.
            return let(Lam((x,), dr(body)), lambda g: let(dr(xs), lambda r: tup(
                op("map_", lam1(lambda a: fst(app(g, a))), fst(r)),
                linlam(lambda d: op(
                    "plus",
                    op("fold", lam2(lambda a, da: op("drop", app(snd(app(g, a)), da), _name(x))),
                       op("zip_", fst(r), d)),
                    app(snd(r), op("zipwith", lam2(lambda a, da: op("get", app(snd(app(g, a)), da), _name(x))),
                                   fst(r), d))))), "r"), "g")

        case Op(name, args):
            prim = PRIMITIVES[name]
            return _let_all([dr(a) for a in args], lambda rs: _let_all([fst(r) for r in rs], lambda xs: tup(
                Op(name, tuple(xs)),
                linlam(lambda d: _plus_all(
                    [app(snd(r), ct) for r, ct in zip(rs, prim.vjp(xs, d))] or [Z]))), "x"))

    raise TypeError(f"dr: unsupported expression {e!r}")


# ---------------------------------------------------------------------------
# Forward mode


def df(e: Expr) -> Expr:
    match e:
        case Var(x):
            return tup(Var(x), linlam(lambda dt: op("get", dt, _name(x)), "dt"))

        case Const():
            return tup(e, linlam(lambda dt: Z, "dt"))

        case Let(x, rhs, body):
            # (fst b, \dt. snd b (dt extended with x := snd r dt))
            return let(df(rhs), lambda r: Let(x, fst(r), let(df(body), lambda b: tup(
                fst(b),
                linlam(lambda dt: app(snd(b), op("extend", dt, _name(x), app(snd(r), dt))), "dt")), "r")), "r")

        case Lam((x,), body):
            # Primal: tangent w.r.t. the argument only. Tangent: a function from
            # primal arguments to the tangent w.r.t. the environment.
            return let(Lam((x,), df(body)), lambda g: tup(
                lam1(lambda a: let(app(g, a), lambda r: tup(
                    fst(r),
                    linlam(lambda da: app(snd(r), op("single", _name(x), da)), "dt")), "r"), "x"),
                linlam(lambda dt: lam1(lambda a: app(snd(app(g, a)), dt), "x"), "dt")), "g")

        case App(f, (a,)):
            return let(df(a), lambda ra: let(df(f), lambda rf: let(app(fst(rf), fst(ra)), lambda res: tup(
                fst(res),
                linlam(lambda dt: op(
                    "plus",
                    op("tapp", fst(ra), app(snd(rf), dt)),
                    app(snd(res), app(snd(ra), dt))), "dt")), "y"), "r"), "r")

        case Tuple(items):
            return _let_all([df(x) for x in items], lambda rs: tup(
                tup(*[fst(r) for r in rs]),
                linlam(lambda dt: tup(*[app(snd(r), dt) for r in rs]) if rs else Z, "dt")))

        case Proj(i, t):
            return let(df(t), lambda r: tup(
                Proj(i, fst(r)), linlam(lambda dt: op("lproj", Const(i), app(snd(r), dt)), "dt")), "r")

        case Inl(v):
            return let(df(v), lambda r: tup(Inl(fst(r)), linlam(lambda dt: Inl(app(snd(r), dt)), "dt")), "r")

        case Inr(v):
            return let(df(v), lambda r: tup(Inr(fst(r)), linlam(lambda dt: Inr(app(snd(r), dt)), "dt")), "r")

        case Case(s, xl, left, xr, right):
            def branch(r, x, body, proj):
                return let(df(body), lambda b: tup(
                    fst(b),
                    linlam(lambda dt: app(snd(b), op("extend", dt, _name(x), op(proj, app(snd(r), dt)))), "dt")), "r")

            return let(df(s), lambda r: Case(
                fst(r), xl, branch(r, xl, left, "from_inl"), xr, branch(r, xr, right, "from_inr")), "r")

        case Map(x, body, xs):
            return let(Lam((x,), df(body)), lambda g: let(df(xs), lambda r: tup(
                op("map_", lam1(lambda a: fst(app(g, a))), fst(r)),
                linlam(lambda dt: op(
                    "plus",
                    op("zipwith", lam2(lambda a, da: app(snd(app(g, a)), op("single", _name(x), da))),
                       fst(r), app(snd(r), dt)),
                    op("map_", lam1(lambda a: app(snd(app(g, a)), dt)), fst(r))), "dt")), "r"), "g")

        case Op(name, args):
            prim = PRIMITIVES[name]
            return _let_all([df(a) for a in args], lambda rs: _let_all([fst(r) for r in rs], lambda xs: tup(
                Op(name, tuple(xs)),
                linlam(lambda dt: prim.jvp(xs, [app(snd(r), dt) for r in rs]), "dt")), "x"))

    raise TypeError(f"df: unsupported expression {e!r}")


# ---------------------------------------------------------------------------
# Helpers


def lam1(body_fn, prefix: str = "a") -> Lam:
    n = fresh(prefix)
    return Lam((n,), body_fn(Var(n)))


def lam2(body_fn) -> Lam:
    a, d = fresh("a"), fresh("d")
    return Lam((a, d), body_fn(Var(a), Var(d)))


def _let_all(rhss: list[Expr], body_fn, prefix: str = "r") -> Expr:
    names = [fresh(prefix) for _ in rhss]
    out = body_fn([Var(n) for n in names])
    for n, rhs in reversed(list(zip(names, rhss))):
        out = Let(n, rhs, out)
    return out
