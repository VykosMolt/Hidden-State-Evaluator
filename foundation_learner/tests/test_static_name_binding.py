"""Every name the FL package reads must resolve in some scope at runtime.

The FL package is baked into the SAME container image as the O1 runner and
executes on the same paid pod, but the O1 suite's name guard only ever
walked ``o1_b200/``.  A NameError of the shape found in
``production_entry.calibration()`` (a name only sibling handlers had
imported function-locally) would therefore have been invisible here -- on
code that runs for hours AFTER the O1 phase, i.e. at the most expensive
possible moment to crash.

``UnboundLocalError`` gets its own pass: ``symtable``'s ``is_global()`` is
false for any name assigned anywhere in a function, so the global check
cannot see it, and it fails exactly the same way -- only on the real path.
"""
from __future__ import annotations

import os
import tempfile

from . import _namecheck

_FL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: (source, expected unbound globals, expected may-be-unbound locals)
_SELF_TEST_FIXTURES = (
    # --- true positives: these really do raise at runtime ---
    ("def f():\n    return missing_name\n", 1, 0),
    ("def f():\n    print(x)\n    x = 1\n", 0, 1),
    # branch-only binding, and the if/elif chain with NO else -- the single
    # commonest UnboundLocalError shape, and the one an earlier version of
    # this checker reported as a definite assignment
    ("def f(c):\n    if c:\n        x = 1\n    return x\n", 0, 1),
    ("def f(c):\n    if c == 1:\n        x = 1\n    elif c == 2:\n"
     "        x = 2\n    return x\n", 0, 1),
    ("def f(g):\n    try:\n        d = g()\n    except ValueError:\n"
     "        pass\n    return d\n", 0, 1),
    ("def f(xs, g):\n    for _ in xs:\n        s = g()\n    return s\n", 0, 1),
    # bound ONLY under TYPE_CHECKING or a swallowed import: absent at
    # runtime, the same shape as the sha256_file bug this guard exists for.
    # `except Exception` and a bare `except` swallow just as effectively.
    ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import foo\n"
     "def f():\n    return foo.bar()\n", 1, 0),
    ("try:\n    import foo\nexcept ImportError:\n    pass\n"
     "def f():\n    return foo.bar()\n", 1, 0),
    ("try:\n    import foo\nexcept Exception:\n    pass\n"
     "def f():\n    return foo.bar()\n", 1, 0),
    ("try:\n    import foo\nexcept:\n    pass\n"
     "def f():\n    return foo.bar()\n", 1, 0),
    # --- true negatives: flagging any of these would block real code ---
    ("def f(x):\n    print(x)\n    x = 2\n", 0, 0),
    ("g = 1\ndef f():\n    global g\n    print(g)\n    g = 2\n", 0, 0),
    ("def o():\n    v = 1\n    def i():\n        return v\n    return i\n", 0, 0),
    ("def m():\n    d = {}\n    def rec(name, ok):\n        d[name] = ok\n"
     "    rec('a', 1)\n", 0, 0),
    ("def m(t):\n    if t:\n        def g():\n            return 1\n"
     "        return g\n    def g():\n        return 2\n    return g\n", 0, 0),
    ("def f(c):\n    if c:\n        x = 1\n    else:\n        x = 2\n"
     "    return x\n", 0, 0),
    ("def f(c):\n    if c == 1:\n        x = 1\n    elif c == 2:\n"
     "        x = 2\n    else:\n        x = 3\n    return x\n", 0, 0),
    ("def f(g):\n    try:\n        d = g()\n    except ValueError:\n"
     "        raise RuntimeError('x')\n    return d\n", 0, 0),
    ("def f(g):\n    try:\n        v = g()\n    except ValueError:\n"
     "        return None\n    else:\n        w = v\n    return w\n", 0, 0),
    ("def f(g):\n    try:\n        out = g()\n    finally:\n        pass\n"
     "    return out\n", 0, 0),
    ("def f(g):\n    while True:\n        x = g()\n        break\n"
     "    return x\n", 0, 0),
    ("def f(g):\n    with g() as h:\n        v = h\n    return v\n", 0, 0),
    ("def f(g):\n    for _ in range(64):\n        s = g()\n        break\n"
     "    return s\n", 0, 0),
    ("def f():\n    for _ in range(3):\n        try:\n            print(t)\n"
     "        except NameError:\n            pass\n        t = 1\n", 0, 0),
    ("try:\n    import foo\nexcept ImportError:\n    foo = None\n"
     "def f():\n    return foo\n", 0, 0),
    ("import os\ndef f():\n    return os.sep\n", 0, 0),
    ("def f(xs):\n    return [y for y in xs]\n", 0, 0),
    ("def f():\n    return len([1])\n", 0, 0),
    # loop-else is conservatively treated as NOT definite, even when both
    # clauses bind: a `break` before the binding exits with the name unbound
    # and this analysis has no notion of `break`.  A version that accepted
    # the both-clauses-bind shape let THIS real UnboundLocalError through:
    ("def f(xs, g):\n    for x in xs:\n        if g(x):\n            break\n"
     "        v = x\n    else:\n        v = None\n    return v\n", 0, 1),
    ("def f(xs, g):\n    for x in xs:\n        v = g(x)\n    else:\n"
     "        pass\n    return v\n", 0, 1),
)


def test_the_checker_itself_is_correct():
    """A guard nobody validated is a guard nobody should trust."""
    for src, want_globals, want_locals in _SELF_TEST_FIXTURES:
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as fh:
            fh.write(src)
            probe = fh.name
        try:
            got = (len(_namecheck.unbound_globals(probe)),
                   len(_namecheck.unbound_locals(probe)))
        finally:
            os.unlink(probe)
        assert got == (want_globals, want_locals), (
            f"checker wrong on {src!r}: got {got}, "
            f"want {(want_globals, want_locals)}")


def test_no_name_in_the_fl_package_resolves_in_no_scope():
    offenders = _namecheck.scan_tree(_FL_ROOT)
    assert not offenders, (
        "these names raise NameError/UnboundLocalError when the line runs: "
        + "; ".join(o.replace(_FL_ROOT + os.sep, "") for o in offenders))
