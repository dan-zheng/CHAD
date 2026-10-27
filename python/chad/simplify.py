"""A small simplifier for target programs.

The CHAD rules are compositional, which makes their raw output verbose:
every sub-term becomes a ``(primal, linear function)`` pair that is
immediately taken apart again. A handful of rewrites -- beta reduction, pair
projection, let-floating, inlining of single-use or trivial bindings, and the
monoid laws for ``ZERO`` -- clean that up.

All rewrites preserve meaning and never duplicate work: a non-trivial binding
is only inlined if it is used once and not under a lambda. Because all binders
have globally unique names, substitution is capture-free.

Each pass takes a census of variable usage, then walks the program once,
top-down, substituting inlined bindings as it goes; passes repeat until
nothing changes.
"""

from __future__ import annotations

from . import runtime as rt
from .ast import (App, Case, Const, Expr, Inl, Inr, Lam, Let, Map, Op, Proj,
                  Tuple, Var, fresh)

# For each linear runtime helper, the position of its linear argument:
# if that argument is ZERO, so is the result.
LINEAR_ARG = {
    "lproj": 1, "scale": 1, "lneg": 0, "lsum": 0, "lreplicate": 1, "expand": 1,
    "sum_to": 1, "tapp": 1, "from_inl": 0, "from_inr": 0, "copow": 1,
    "zip_": 1, "zipwith": 2, "single": 1, "get": 0, "drop": 0, "fold": 1,
}


def is_zero(e: Expr) -> bool:
    return isinstance(e, Const) and e.value is rt.ZERO


ZERO = Const(rt.ZERO)


def simplify(e: Expr, max_passes: int = 100) -> Expr:
    hints: dict = {}
    for _ in range(max_passes):
        s = _Simplifier(e, hints)
        e = s.go(e)
        if not s.changed:
            break
    return _rename(e, hints)


# ---------------------------------------------------------------------------
# Generic traversal helpers


def children_map(e: Expr, f) -> Expr:
    """Rebuild ``e`` with ``f`` applied to each direct sub-expression."""
    match e:
        case Var() | Const():
            return e
        case Lam(ps, b, lin):
            return Lam(ps, f(b), lin)
        case App(g, args):
            return App(f(g), tuple(f(a) for a in args))
        case Let(x, r, b):
            return Let(x, f(r), f(b))
        case Tuple(items):
            return Tuple(tuple(f(i) for i in items))
        case Proj(i, t):
            return Proj(i, f(t))
        case Inl(v):
            return Inl(f(v))
        case Inr(v):
            return Inr(f(v))
        case Case(s, xl, l, xr, r):
            return Case(f(s), xl, f(l), xr, f(r))
        case Map(x, b, xs):
            return Map(x, f(b), f(xs))
        case Op(n, args):
            return Op(n, tuple(f(a) for a in args))
    raise TypeError(e)


def census(e: Expr) -> tuple[dict[str, list], set[str]]:
    """Usage of every variable in ``e``.

    Returns ``{name: [count, used_under_a_lambda]}`` (relative to the
    variable's binder) and the set of variables that are components of a
    let-bound tuple or environment literal.
    """
    usage: dict[str, list] = {}
    depth_of: dict[str, int] = {}
    leaves: set[str] = set()

    def bind(names, depth):
        for n in names:
            depth_of[n] = depth
            usage.setdefault(n, [0, False])

    def walk(e: Expr, depth: int):
        match e:
            case Var(n):
                u = usage.setdefault(n, [0, False])
                u[0] += 1
                u[1] = u[1] or depth > depth_of.get(n, 0)
            case Lam(ps, b):
                bind(ps, depth + 1)
                walk(b, depth + 1)
            case Map(x, b, xs):
                bind([x], depth + 1)
                walk(b, depth + 1)
                walk(xs, depth)
            case Let(x, r, b):
                bind([x], depth)
                comps = r.items if isinstance(r, Tuple) else envlit_leaves(r) if is_envlit(r) else ()
                leaves.update(c.name for c in comps if isinstance(c, Var))
                walk(r, depth)
                walk(b, depth)
            case Case(sc, xl, l, xr, r):
                bind([xl, xr], depth)
                walk(sc, depth)
                walk(l, depth)
                walk(r, depth)
            case _:
                children_map(e, lambda c: walk(c, depth) or c)

    walk(e, 0)
    return usage, leaves


def is_trivial(e: Expr) -> bool:
    match e:
        case Var() | Const():
            return True
        case Proj(_, t):
            return is_trivial(t)
        case Tuple(items):
            return all(is_trivial(i) for i in items)
    return is_envlit(e) and all(is_trivial(l) for l in envlit_leaves(e))


def is_envlit(e: Expr) -> bool:
    """A literal environment (co)tangent, built from single / extend / plus."""
    match e:
        case Op("single", (Const(), _)):
            return True
        case Op("extend", (env, Const(), _)):
            return isinstance(env, Var) or is_envlit(env)
        case Op("plus", (a, b)):
            return is_envlit(a) and is_envlit(b)
    return False


def envlit_leaves(e: Expr) -> list[Expr]:
    """The values stored in an environment literal (and its base, if any)."""
    match e:
        case Op("single", (_, v)):
            return [v]
        case Op("extend", (env, _, v)):
            return ([env] if isinstance(env, Var) else envlit_leaves(env)) + [v]
        case Op("plus", (a, b)):
            return envlit_leaves(a) + envlit_leaves(b)
    return []


def map_envlit_leaves(e: Expr, f) -> Expr:
    match e:
        case Op("single", (n, v)):
            return Op("single", (n, f(v)))
        case Op("extend", (env, n, v)):
            return Op("extend", (env if isinstance(env, Var) else map_envlit_leaves(env, f), n, f(v)))
        case Op("plus", (a, b)):
            return Op("plus", (map_envlit_leaves(a, f), map_envlit_leaves(b, f)))
    raise TypeError(e)


def singles(e: Expr):
    """If ``e`` is a plus-tree of singles, its ``(name, value)`` pairs."""
    match e:
        case Op("single", (Const(n), v)):
            return [(n, v)]
        case Op("plus", (a, b)):
            sa, sb = singles(a), singles(b)
            return None if sa is None or sb is None else sa + sb
    return None


def _base(name: str) -> str:
    """``"r12"`` -> ``"r"``: the prefix of a generated name."""
    return name.rstrip("0123456789_") or "v"


def _rename(e: Expr, hints: dict[str, str]) -> Expr:
    """Give variables back the (source) names of the bindings they replaced."""
    bound: list[str] = []

    def collect(e: Expr):
        match e:
            case Lam(ps, _):
                bound.extend(ps)
            case Let(x, _, _) | Map(x, _, _):
                bound.append(x)
            case Case(_, xl, _, xr, _):
                bound.extend((xl, xr))
        children_map(e, lambda c: collect(c) or c)

    collect(e)
    used, ren = set(bound), {}
    for b in bound:
        name = b
        seen = set()
        while name in hints and name not in seen:
            seen.add(name)
            name = hints[name]
        if name != b and name not in used:
            ren[b] = name
            used.add(name)
    if not ren:
        return e

    def go(e: Expr) -> Expr:
        r = lambda n: ren.get(n, n)
        match e:
            case Var(n):
                return Var(r(n))
            case Lam(ps, b, lin):
                return Lam(tuple(map(r, ps)), go(b), lin)
            case Let(x, rhs, b):
                return Let(r(x), go(rhs), go(b))
            case Map(x, b, xs):
                return Map(r(x), go(b), go(xs))
            case Case(s, xl, l, xr, rr):
                return Case(go(s), r(xl), go(l), r(xr), go(rr))
        return children_map(e, go)

    return go(e)


# ---------------------------------------------------------------------------
# The rewrites


class _Simplifier:
    def __init__(self, e: Expr, hints: dict):
        self.changed = False
        self.usage, self.leaves = census(e)
        self.env: dict[str, Expr] = {}  # bindings being inlined
        self.linear: set[str] = set()  # variables bound to linear functions
        # Naming hints, to give variables back the names of bindings they
        # replaced: variable -> name, and (tuple variable, index) -> name.
        self.hints = hints
        self.proj_hints = hints.setdefault(None, {})

    def rewrote(self, e: Expr) -> Expr:
        self.changed = True
        return e

    def go(self, e: Expr) -> Expr:
        if isinstance(e, Var):
            return self.env.get(e.name, e)
        if isinstance(e, Let):
            # Walk let-chains iteratively to keep the recursion shallow.
            kept: list[tuple[str, Expr]] = []
            while isinstance(e, Let):
                kept += self.let(e.name, self.go(e.rhs))
                e = e.body
            out = self.go(e)
            for x, rhs in reversed(kept):
                out = Let(x, rhs, out)
            return out
        if isinstance(e, App):
            f, args = self.go(e.fun), tuple(self.go(a) for a in e.args)
            return self.beta(f, args) if isinstance(f, Lam) else self.node(App(f, args))
        return self.node(children_map(e, self.go))

    def beta(self, f: Lam, args: tuple[Expr, ...]) -> Expr:
        """``(\\params -> body) args``, simplified straight away."""
        self.changed = True
        kept = []
        for p, a in zip(f.params, args):
            kept += self.let(p, a)
        out = self.go(f.body)
        for x, rhs in reversed(kept):
            out = Let(x, rhs, out)
        return out

    def let(self, x: str, rhs: Expr) -> list[tuple[str, Expr]]:
        """Decide what to do with ``let x = rhs`` (``rhs`` already simplified).

        Returns the bindings to keep in its place; inlined bindings go to
        ``self.env`` instead.
        """
        count, under = self.usage.get(x, (1, True))
        if count == 0:
            self.changed = True
            return []
        kept = []
        # Float lets out of let-bound right-hand sides.
        while isinstance(rhs, Let):
            self.changed = True
            kept.append((rhs.name, rhs.rhs))
            rhs = rhs.body
        # Split let-bound tuples and environment literals into their components.
        if (isinstance(rhs, Tuple) or is_envlit(rhs)) and not is_trivial(rhs):
            def bind(i, v):
                if is_trivial(v):
                    return v
                n = fresh(_base(x))
                if (x, i) in self.proj_hints:
                    self.hints.setdefault(n, self.proj_hints[x, i])
                kept.append((n, v))
                return Var(n)

            if isinstance(rhs, Tuple):
                rhs = Tuple(tuple(bind(i, v) for i, v in enumerate(rhs.items)))
            else:
                rhs = map_envlit_leaves(rhs, lambda v: bind(None, v))
            self.changed = True
        if isinstance(rhs, Var):
            self.hints.setdefault(rhs.name, x)
        if isinstance(rhs, Proj) and isinstance(rhs.tup, Var):
            self.proj_hints.setdefault((rhs.tup.name, rhs.index), x)
        # Inline trivial bindings, and single-use ones where that does not
        # duplicate work (and does not undo a split).
        if is_trivial(rhs) or (
            count == 1 and (not under or isinstance(rhs, Lam)) and x not in self.leaves
        ):
            self.env[x] = rhs
            self.changed = True
            return kept
        if isinstance(rhs, Lam) and rhs.linear:
            self.linear.add(x)
        return kept + [(x, rhs)]

    def node(self, e: Expr) -> Expr:
        # Float lets out of (strict, non-binding) argument positions.
        match e:
            case Op(name, args) if any(isinstance(a, Let) for a in args):
                i = next(i for i, a in enumerate(args) if isinstance(a, Let))
                l = args[i]
                return self.rewrote(Let(l.name, l.rhs, Op(name, args[:i] + (l.body,) + args[i + 1:])))
            case App(f, args) if any(isinstance(a, Let) for a in args):
                i = next(i for i, a in enumerate(args) if isinstance(a, Let))
                l = args[i]
                return self.rewrote(Let(l.name, l.rhs, App(f, args[:i] + (l.body,) + args[i + 1:])))
            case Tuple(items) if any(isinstance(a, Let) for a in items):
                i = next(i for i, a in enumerate(items) if isinstance(a, Let))
                l = items[i]
                return self.rewrote(Let(l.name, l.rhs, Tuple(items[:i] + (l.body,) + items[i + 1:])))
            case Inl(Let(x, r, b)):
                return self.rewrote(Let(x, r, Inl(b)))
            case Inr(Let(x, r, b)):
                return self.rewrote(Let(x, r, Inr(b)))
        match e:
            case Proj(i, Tuple(items)) if i < len(items):
                return self.rewrote(items[i])
            case Proj(i, Let(x, r, b)):
                return self.rewrote(Let(x, r, Proj(i, b)))
            case App(Lam(params, body), args):
                out = body
                for p, a in reversed(list(zip(params, args))):
                    out = Let(p, a, out)
                return self.rewrote(out)
            case App(Let(x, r, f), args):
                return self.rewrote(Let(x, r, App(f, args)))
            case Case(Let(x, r, s), xl, l, xr, rr):
                return self.rewrote(Let(x, r, Case(s, xl, l, xr, rr)))
            case Case(Inl(v), xl, l, _, _):
                return self.rewrote(Let(xl, v, l))
            case Case(Inr(v), _, _, xr, r):
                return self.rewrote(Let(xr, v, r))
            case Tuple(items) if items and all(is_zero(i) for i in items):
                return self.rewrote(ZERO)
            case Inl(v) | Inr(v) if is_zero(v):
                # Only arises as a cotangent, where ZERO is equivalent.
                return self.rewrote(ZERO)
            case App(Var(f), (a,)) if f in self.linear and is_zero(a):
                return self.rewrote(ZERO)
            case App(Lam(_, _, True), (a,)) if is_zero(a):
                return self.rewrote(ZERO)
            case Op(name, args):
                return self.op(e, name, args)
        return e

    def op(self, e: Op, name: str, args: tuple[Expr, ...]) -> Expr:
        if name in LINEAR_ARG and is_zero(args[LINEAR_ARG[name]]):
            return self.rewrote(ZERO)
        match name, args:
            case "plus", (a, b) if is_zero(a):
                return self.rewrote(b)
            case "plus", (a, b) if is_zero(b):
                return self.rewrote(a)
            case "plus", (a, b) if (sa := singles(a)) is not None and (sb := singles(b)) is not None:
                merged: dict[str, list[Expr]] = {}
                for n, v in sa + sb:
                    merged.setdefault(n, []).append(v)
                if len(merged) < len(sa) + len(sb):  # some variable occurs twice
                    parts = []
                    for n, vs in merged.items():
                        v = vs[0]
                        for w in vs[1:]:
                            v = Op("plus", (v, w))
                        parts.append(Op("single", (Const(n), v)))
                    out = parts[0]
                    for p in parts[1:]:
                        out = Op("plus", (out, p))
                    return self.rewrote(out)
            case "drop", (Op("extend", (env, Const(m), v)), Const(n)):
                return self.rewrote(env if m == n else Op("extend", (Op("drop", (env, Const(n))), Const(m), v)))
            case "lproj", (Const(i), Tuple(items)):
                return self.rewrote(items[i] if i < len(items) else ZERO)
            case "get", (Op("single", (Const(m), v)), Const(n)):
                return self.rewrote(v if m == n else ZERO)
            case "drop", (Op("single", (Const(m), v)) as s, Const(n)):
                return self.rewrote(ZERO if m == n else s)
            case "get", (Op("extend", (env, Const(m), v)), Const(n)):
                return self.rewrote(v if m == n else Op("get", (env, Const(n))))
            case ("get" | "drop") as f, (Op("plus", (a, b)), n):
                return self.rewrote(Op("plus", (Op(f, (a, n)), Op(f, (b, n)))))
            case "extend", (env, n, v) if is_zero(env):
                return self.rewrote(Op("single", (n, v)))
            case "extend", (env, _, v) if is_zero(v):
                return self.rewrote(env)
            case "tapp", (x, Lam() as f):
                return self.beta(f, (x,))
            case "fold", (Lam() as f, Op("copow", (x, d))):
                return self.beta(f, (x, d))
            case "fold", (f, Op("copow", (x, d))):
                return self.rewrote(App(f, (x, d)))
            case "from_inl", (Inl(v),):
                return self.rewrote(v)
            case "from_inr", (Inr(v),):
                return self.rewrote(v)
        return e
