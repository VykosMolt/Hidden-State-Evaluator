"""Static name-binding checks: unbound globals, and may-be-unbound locals.

Both catch the SAME class of defect: a name that resolves in no scope at
runtime, on a path no test executes.  production_entry.calibration() read
`sha256_file` as a module global that sibling handlers had only imported
function-locally; it raised NameError on every real pod, after the pod, the
image pull and the checkpoint fetch had been paid for.

Two blind spots are covered deliberately:
  * the FL package is baked into the SAME image and runs on the same paid
    pod, so it is scanned too;
  * `is_global()` is false for any name assigned anywhere in a function, so
    UnboundLocalError -- the same "only crashes on the real path" class --
    is invisible to the global check and needs its own pass.

`unbound_locals` is deliberately conservative: a read inside a loop that also
binds the name is not flagged, because the binding feeds the next iteration.
It is validated against synthetic positives and negatives by its callers.
"""
from __future__ import annotations

import ast
import builtins
import os
import symtable

IMPLICIT_MODULE_NAMES = frozenset({
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__path__"})
_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
           ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)


def _own_nodes(fn):
    """Nodes belonging to THIS function scope, not to nested scopes.

    A nested def/lambda/comprehension is NEVER returned: including the node
    itself let ast.walk descend into it later, which attributed a nested
    helper's own parameters to the enclosing function.  Nested scopes still
    BIND their name in this scope, which _own_binds records separately.
    """
    out = []
    stack = [n for n in fn.body if not isinstance(n, _NESTED)]
    while stack:
        n = stack.pop()
        if isinstance(n, _NESTED):
            continue
        out.append(n)
        for child in ast.iter_child_nodes(n):
            if not isinstance(child, _NESTED):
                stack.append(child)
    return out


def _nested_scope_bindings(fn):
    """`def`/`class` statements bind their NAME in the ENCLOSING scope.

    Traverses compound statements (if/for/while/try/with/match) so a def
    inside an ``if`` branch counts as a binding -- but never descends INTO a
    nested def/class body, whose own statements belong to another scope.
    """
    binds: dict[str, list[int]] = {}
    stack = list(fn.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            binds.setdefault(n.name, []).append(n.lineno)
            continue                    # its body is a different scope
        for attr in ("body", "orelse", "finalbody", "handlers", "cases"):
            stack.extend(getattr(n, attr, []) or [])
    return binds


def unbound_globals(path):
    """Names a function reads as a global that exist in no scope."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    top = symtable.symtable(src, os.path.basename(path), "exec")
    bound = {s.get_name() for s in top.get_symbols()
             if s.is_assigned() or s.is_imported() or s.is_namespace()}
    bound |= IMPLICIT_MODULE_NAMES
    found, stack = [], [top]
    while stack:
        table = stack.pop()
        stack.extend(table.get_children())
        if table.get_type() != "function":
            continue
        for sym in table.get_symbols():
            if (sym.is_global() and not sym.is_assigned()
                    and sym.get_name() not in bound
                    and not hasattr(builtins, sym.get_name())):
                found.append(f"{table.get_name()}() -> {sym.get_name()}")
    return found


def unbound_locals(path):
    """Locals read strictly before their first binding (UnboundLocalError).

    Conservative on purpose: a read inside a loop that also binds the name is
    NOT flagged, because the binding feeds the next iteration.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        nodes = _own_nodes(fn)
        declared = set()
        for n in nodes:
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                declared.update(n.names)
        params = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
        binds: dict[str, list[int]] = {
            k: list(v) for k, v in _nested_scope_bindings(fn).items()}
        for n in nodes:
            for sub in ast.walk(n):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    binds.setdefault(sub.id, []).append(sub.lineno)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for al in sub.names:
                        nm = (al.asname or al.name).split(".")[0]
                        binds.setdefault(nm, []).append(sub.lineno)
                elif isinstance(sub, ast.ExceptHandler) and sub.name:
                    binds.setdefault(sub.name, []).append(sub.lineno)
        # lines covered by a loop that also binds the name
        loop_spans = [(n.lineno, n.end_lineno) for n in nodes
                      if isinstance(n, _LOOPS)]
        for n in nodes:
            if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)):
                continue
            name = n.id
            if name in params or name in declared or name not in binds:
                continue
            first = min(binds[name])
            if n.lineno >= first:
                continue
            if any(lo <= n.lineno <= hi
                   and any(lo <= b <= hi for b in binds[name])
                   for lo, hi in loop_spans):
                continue
            found.append(f"{fn.name}() -> {name} (read line {n.lineno}, "
                         f"first bound line {first})")
    return found


def scan_tree(root, skip_dirs=("__pycache__", "reports")):
    problems = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in skip_dirs)
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            p = os.path.join(base, name)
            for hit in unbound_globals(p):
                problems.append(f"{p}: GLOBAL {hit}")
            for hit in unbound_locals(p):
                problems.append(f"{p}: LOCAL {hit}")
    return problems
