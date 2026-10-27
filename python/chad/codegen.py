"""Compile expressions to readable Python source code, and load them."""

from __future__ import annotations

import numpy as np

from . import runtime as rt
from .ast import (App, Case, Const, Expr, Inl, Inr, Lam, Let, Map, Op, Proj,
                  Tuple, Var, fresh)
from .ops import PRIMITIVES

INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}

RUNTIME_HELPERS = [
    "ZERO", "Inl", "Inr", "plus", "single", "get", "drop", "extend", "lproj",
    "scale", "lneg", "lsum", "lreplicate", "expand", "sum_to", "tapp",
    "from_inl", "from_inr", "copow", "fold", "zip_", "zipwith", "map_",
]


def base_namespace() -> dict:
    ns = {name: getattr(rt, name) for name in RUNTIME_HELPERS}
    ns.update({name: p.impl for name, p in PRIMITIVES.items()})
    return ns


RESERVED = frozenset(base_namespace()) | {"np"}


class _Gen:
    def __init__(self):
        self.lines: list[str] = []
        self.indent = 1
        self.consts: dict[str, object] = {}

    def emit(self, line: str):
        self.lines.append("    " * self.indent + line)

    def define(self, name: str, f: Lam):
        self.emit(f"def {name}({', '.join(f.params)}):")
        self.indent += 1
        self.emit(f"return {self.expr(f.body)}")
        self.indent -= 1

    def expr(self, e: Expr) -> str:
        """Emit statements for ``e`` and return a Python expression for its value."""
        while isinstance(e, Let):
            if isinstance(e.rhs, Lam):
                self.define(e.name, e.rhs)
            elif isinstance(e.rhs, Case):
                self.case(e.rhs, e.name)
            else:
                self.emit(f"{e.name} = {self.expr(e.rhs)}")
            e = e.body

        match e:
            case Var(n):
                return n
            case Const(v):
                return self.const(v)
            case Lam():
                name = fresh("f")
                self.define(name, e)
                return name
            case App(f, args):
                return f"{self.expr(f)}({', '.join(self.expr(a) for a in args)})"
            case Tuple(items):
                parts = [self.expr(i) for i in items]
                return f"({parts[0]},)" if len(parts) == 1 else f"({', '.join(parts)})"
            case Proj(i, t):
                return f"{self.expr(t)}[{i}]"
            case Inl(v):
                return f"Inl({self.expr(v)})"
            case Inr(v):
                return f"Inr({self.expr(v)})"
            case Case():
                return self.case(e, fresh("c"))
            case Map(x, body, xs):
                name = fresh("f")
                self.define(name, Lam((x,), body))
                return f"map_({name}, {self.expr(xs)})"
            case Op(name, args) if name in INFIX:
                a, b = (self.expr(x) for x in args)
                return f"({a} {INFIX[name]} {b})"
            case Op("neg", (a,)):
                return f"(-{self.expr(a)})"
            case Op(name, args):
                return f"{name}({', '.join(self.expr(a) for a in args)})"
        raise TypeError(f"codegen: unsupported expression {e!r}")

    def case(self, e: Case, res: str) -> str:
        """Emit an if/else assigning the value of ``e`` to ``res``."""
        sv = self.expr(e.scrut)
        if not sv.isidentifier():
            tmp = fresh("s")
            self.emit(f"{tmp} = {sv}")
            sv = tmp
        for cls, x, body in (("Inl", e.lname, e.left), (None, e.rname, e.right)):
            self.emit(f"if isinstance({sv}, {cls}):" if cls else "else:")
            self.indent += 1
            if not x.startswith("_"):  # unit binders of chad.cond are unused
                self.emit(f"{x} = {sv}.value")
            self.emit(f"{res} = {self.expr(body)}")
            self.indent -= 1
        return res

    def const(self, v) -> str:
        if v is rt.ZERO:
            return "ZERO"
        if isinstance(v, np.ndarray):
            name = f"_const{len(self.consts)}"
            self.consts[name] = v
            return name
        s = repr(v)
        return f"({s})" if s.startswith("-") else s


def compile_expr(params: list[str], e: Expr, name: str = "f"):
    """Compile ``e`` (with free variables ``params``) to a Python function.

    Returns ``(function, source)``.
    """
    g = _Gen()
    result = g.expr(e)
    source = "\n".join([f"def {name}({', '.join(params)}):", *g.lines, f"    return {result}"])
    ns = base_namespace()
    ns.update(g.consts)
    exec(compile(source, f"<chad:{name}>", "exec"), ns)
    return ns[name], source
