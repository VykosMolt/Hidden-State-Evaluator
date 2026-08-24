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

KNOWN GAP, stated so nobody mistakes this for a proof: a nested function
that closes over a CONDITIONALLY bound enclosing local -- ``if c: v = 1``
then ``def inner(): return v`` -- is not flagged, because the name is a free
variable in the inner scope rather than a global or a local of it.  Both
trees were swept for that shape and no live instance was found, but the
check does not cover it.  `del x` then read, and an augmented assignment
before the first binding, are also missed.

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


def _module_binds(stmts):
    """Names bound by a flat list of module-level statements."""
    names = set()
    for n in stmts:
        for sub in ast.walk(n):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
                names.add(sub.name)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for al in sub.names:
                    names.add((al.asname or al.name).split(".")[0])
            elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                names.add(sub.id)
    return names


def _conditionally_bound_module_names(tree):
    """Module names whose ONLY binding is conditional at runtime."""
    unconditional, conditional = set(), set()
    for node in tree.body:
        if isinstance(node, ast.If):
            test = ast.dump(node.test)
            if "TYPE_CHECKING" in test:
                conditional |= _module_binds(node.body)
                unconditional |= _module_binds(node.orelse)
            else:
                unconditional |= _module_binds([node])
        elif isinstance(node, ast.Try):
            # A bare `except:` or `except Exception:` swallows an import
            # error just as effectively as `except ImportError:`; only
            # literal-matching the two import types missed the two commonest
            # shapes.
            def _swallows(h):
                if h.type is None:
                    return True             # bare except
                dumped = ast.dump(h.type)
                return any(tok in dumped for tok in
                           ("ImportError", "ModuleNotFoundError",
                            "'Exception'", "'BaseException'"))

            handled = any(_swallows(h) for h in node.handlers)
            if handled:
                conditional |= _module_binds(node.body)
                for h in node.handlers:
                    unconditional |= _module_binds(h.body)
            else:
                unconditional |= _module_binds([node])
        else:
            unconditional |= _module_binds([node])
    return conditional - unconditional


def unbound_globals(path):
    """Names a function reads as a global that exist in no scope."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    top = symtable.symtable(src, os.path.basename(path), "exec")
    bound = {s.get_name() for s in top.get_symbols()
             if s.is_assigned() or s.is_imported() or s.is_namespace()}
    bound |= IMPLICIT_MODULE_NAMES
    # symtable marks a name "assigned" even when the ONLY binding is inside
    # `if TYPE_CHECKING:` (false at runtime) or a module-level
    # `try: import x / except ImportError: pass`.  Both leave the name
    # genuinely absent at runtime -- the same shape as the sha256_file bug
    # this guard exists for -- so they do not count as bound.
    bound -= _conditionally_bound_module_names(ast.parse(src))
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


def _always_iterates(node):
    """True when this iterable is provably non-empty at parse time.

    ``for _ in range(64): x = ...`` binds x -- the loop cannot be skipped.
    Only literals count; anything computed is treated as possibly empty.
    """
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts) > 0
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "range":
        args = node.args
        if len(args) == 1 and isinstance(args[0], ast.Constant):
            return isinstance(args[0].value, int) and args[0].value > 0
        if len(args) >= 2 and all(isinstance(a, ast.Constant) for a in args[:2]):
            lo, hi = args[0].value, args[1].value
            if isinstance(lo, int) and isinstance(hi, int):
                return hi > lo
    return False


def _binds_every_path(stmts, name):
    """True when EVERY path through ``stmts`` binds ``name``.

    The previous version asked only whether the name was bound ANYWHERE in
    the subtree, so an ``if``/``elif`` chain with no ``else`` counted as a
    definite assignment -- the single commonest UnboundLocalError shape, and
    exactly what the branch-only pass exists to catch.
    """
    for n in stmts or []:
        if _stmt_binds_definitely(n, name):
            return True
    return False


def _stmt_binds_definitely(n, name):
    """True when this ONE statement binds ``name`` on every path through it."""
    # a plain binding at this level is unconditional
    if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign,
                      ast.Import, ast.ImportFrom, ast.FunctionDef,
                      ast.AsyncFunctionDef, ast.ClassDef)):
        return name in _module_binds([n])
    if isinstance(n, ast.If):
        if not n.orelse:
            return False                     # no else: the else path binds nothing
        return ((_binds_every_path(n.body, name) or _terminates(n.body))
                and (_binds_every_path(n.orelse, name)
                     or _terminates(n.orelse))
                and not (_terminates(n.body) and _terminates(n.orelse)))
    if isinstance(n, ast.Try):
        if not n.handlers:
            return _binds_every_path(n.body, name)
        # Two ways out of a try/except/else: the body succeeded and the
        # `else` ran, or a handler ran.  A binding in EITHER the body or the
        # else covers the first path -- requiring it in the body alone
        # rejected the ordinary `try: v = g() / except: return / else: w = v`
        # shape.
        success_ok = (_binds_every_path(n.body, name)
                      or _binds_every_path(n.orelse, name)
                      or _terminates(n.body) or _terminates(n.orelse))
        handlers_ok = all(_binds_every_path(h.body, name)
                          or _terminates(h.body) for h in n.handlers)
        return success_ok and handlers_ok
    if isinstance(n, (ast.With, ast.AsyncWith)):
        return _binds_every_path(n.body, name)
    if isinstance(n, (ast.For, ast.AsyncFor, ast.While)):
        if isinstance(n, ast.While) and _is_true_literal(n.test):
            # `while True:` always enters its body
            return _binds_every_path(n.body, name)
        if isinstance(n, ast.For) and _always_iterates(n.iter):
            return _binds_every_path(n.body, name)
        # `for ... else:` / `while ... else:` -- the else clause runs when
        # the loop finished without break, the body when it ran.  Binding in
        # BOTH covers every way out, so flagging it was a false positive that
        # would have blocked ordinary code.
        if n.orelse and _binds_every_path(n.body, name) \
                and _binds_every_path(n.orelse, name):
            return True
        return False
    if isinstance(n, ast.Match):
        cases = getattr(n, "cases", [])
        if cases and any(_is_wildcard_case(c) for c in cases):
            return all(_binds_every_path(c.body, name) or _terminates(c.body)
                       for c in cases)
        return False
    return False


def _is_true_literal(node):
    return isinstance(node, ast.Constant) and node.value is True


def _is_wildcard_case(case):
    pat = getattr(case, "pattern", None)
    return (isinstance(pat, ast.MatchAs) and pat.pattern is None
            and getattr(case, "guard", None) is None)


def _definitely_bound(nodes, name):
    """True when some construct in this scope binds ``name`` on every path."""
    return any(_stmt_binds_definitely(n, name) for n in nodes)


def _terminates(stmts):
    """True when this block cannot fall through to the following statement.

    A handler that re-raises, returns, breaks or continues never reaches the
    code after the construct, so it does not need to bind the name for a
    later read to be safe.  Without this, the overwhelmingly common
    ``try: x = f() / except E: raise ...`` shape reads as a defect.
    """
    for n in reversed(stmts or []):
        if isinstance(n, (ast.Raise, ast.Return, ast.Continue, ast.Break)):
            return True
        if isinstance(n, ast.If) and n.orelse:
            if _terminates(n.body) and _terminates(n.orelse):
                return True
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            fn = n.value.func
            nm = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if nm in ("exit", "_exit"):
                return True
        if not isinstance(n, ast.Pass):
            break
    return False


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
        # Branch-only bindings: EVERY binding sits inside a conditional and
        # the read does not.  Line order cannot see this -- it is the most
        # common UnboundLocalError shape there is.
        cond_spans = [(n.lineno, n.end_lineno) for n in nodes
                      if isinstance(n, (ast.If, ast.Try, ast.While,
                                        ast.For, ast.AsyncFor, ast.Match))]
        for name, blines in binds.items():
            if name in params or name in declared:
                continue
            if not blines or not cond_spans:
                continue
            if not all(any(lo <= b <= hi for lo, hi in cond_spans)
                       for b in blines):
                continue                     # some binding is unconditional
            if _definitely_bound(fn.body, name):
                continue     # every path through some construct binds it
            for n in nodes:
                if not (isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Load) and n.id == name):
                    continue
                if any(lo <= n.lineno <= hi for lo, hi in cond_spans):
                    continue                 # read is guarded too
                found.append(
                    f"{fn.name}() -> {name} (read line {n.lineno} is not "
                    f"guarded, but every binding is conditional)")
                break
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
