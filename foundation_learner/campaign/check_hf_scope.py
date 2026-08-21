#!/usr/bin/env python3
"""FL credential-scope preflight: prove the token before the ladder starts.

One HF_TOKEN serves both directions of an interruptible FL session:

  * READ   ``FL_PREGEN_SOURCE``           — the pregenerated episode corpus;
  * WRITE  ``fl_durable_destination``     — the supervisor journal and every
    atomic training checkpoint, mirrored continuously so a fresh pod can
    resume from the last valid checkpoint.

A read-only token pulls the corpus, trains for hours, and silently loses
every mirror push.  Under interruptible capacity that is the worst failure
available: the arm looks healthy right up to the eviction that destroys it,
and the resumed pod finds nothing to restore.

Read is proven by auth_check; write is proven by writing (a few bytes under
``.preflight/``, deleted again).  A non-``hf://`` destination is checked as a
filesystem path instead.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .hf_transfer import child_env
from .o1_isolation import MODE_READ, MODE_WRITE, guard_path
from .redaction import redact


class ScopeError(RuntimeError):
    pass


def parse_repo(uri: str) -> str | None:
    """``hf://ns/repo[/prefix]`` -> ``ns/repo``; None for anything else."""
    if not uri or not uri.startswith("hf://"):
        return None
    parts = uri[len("hf://"):].strip("/").split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ScopeError(f"malformed hf:// URI {uri!r}")
    return "/".join(parts[:2])


def _helper_error(text: str, limit: int = 400) -> str:
    """The last meaningful line, not a raw tail.

    A Python traceback ends with the exception line — the one thing the
    operator needs ("Invalid user token", "401 Unauthorized").  Slicing the
    last N characters instead lands mid-frame and prints source fragments.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1][:limit] if lines else "(helper produced no output)"


def _run_helper(repo: str, mode: str, timeout: float) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "foundation_learner.campaign.hf_transfer",
         "scope", "--repo", repo, "--mode", mode],
        capture_output=True, text=True, timeout=timeout,
        env=child_env(os.environ.get("HF_TOKEN")))
    if proc.returncode != 0:
        raise ScopeError(redact(
            f"{mode.upper()} scope check failed for {repo}: "
            f"{_helper_error(proc.stdout + proc.stderr)}"))
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise ScopeError(
            f"{mode} scope check for {repo} produced no result") from None


def _check_local_write(path: str) -> dict:
    # The durable mirror is FL's own; routing it through the guard means an
    # operator who points it at one of the FROZEN O1 roots is refused here
    # rather than discovering the violation mid-session.  (Roots discovered
    # later from the O1 manifests are not yet known at this point, so this
    # is the frozen set only.)
    probe = os.path.join(guard_path(path, MODE_WRITE), ".preflight_write_probe")
    try:
        os.makedirs(path, exist_ok=True)
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("probe\n")
        os.remove(probe)
    except OSError as exc:
        raise ScopeError(
            f"the durable destination {path!r} is not writable: {exc}") from None
    return {"repo": path, "mode": "write", "read": True, "write": True,
            "identity": "local-filesystem"}


def check(read_uri: str, write_uri: str, timeout: float = 300.0,
          runner=None) -> dict:
    run = runner or (lambda repo, mode: _run_helper(repo, mode, timeout))
    checks = []
    read_repo = parse_repo(read_uri)
    write_repo = parse_repo(write_uri)
    if read_repo:
        checks.append(run(read_repo, "read"))
    if write_repo:
        checks.append(run(write_repo, "write"))
    elif write_uri and not write_uri.startswith("UNRESOLVED"):
        checks.append(_check_local_write(write_uri))
    else:
        raise ScopeError(
            "fl_durable_destination is unset or UNRESOLVED; an interruptible "
            "FL session with no durable mirror cannot survive an eviction")
    for res in checks:
        if res["mode"] == "read" and not res.get("read"):
            raise ScopeError(
                f"the token cannot read {res['repo']}; the pod could not "
                f"fetch the episode corpus and the FL ladder could not run")
        if res["mode"] == "write" and not res.get("write"):
            raise ScopeError(
                f"the token can read {res['repo']} but cannot write to it; "
                f"the journal and every training checkpoint would be lost "
                f"on eviction")
    return {"schema": "flb200.hf_scope_report.v1", "checks": checks,
            "read_checked": bool(read_repo), "write_checked": bool(write_repo)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="session config; supplies fl_durable_destination")
    p.add_argument("--read-source", default=os.environ.get("FL_PREGEN_SOURCE", ""))
    p.add_argument("--write-destination", default="")
    p.add_argument("--out")
    a = p.parse_args()

    write = a.write_destination
    read_source = a.read_source
    pregen_root = None
    rehearsal = False
    if a.config and (not write or not read_source):
        try:
            with open(guard_path(a.config, MODE_READ), encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"REFUSED: cannot read {a.config}: {exc}", file=sys.stderr)
            return 2
        write = write or cfg.get("fl_durable_destination") or ""
        # the session config is the operator's single binding surface and is
        # exactly where fetch_pregen takes the source from; probing a
        # different (empty) source here would verify nothing
        read_source = read_source or cfg.get("fl_pregen_source") or ""
        pregen_root = cfg.get("pregen_root")
        rehearsal = bool(cfg.get("rehearsal", False))
    if str(read_source).startswith("UNRESOLVED"):
        read_source = ""
    if (not read_source and not rehearsal
            and not (pregen_root and os.path.isdir(str(pregen_root)))):
        print("REFUSED: no pregen read source is bound and the pregen root "
              "is absent; a fetch is due and its credential scope cannot be "
              "verified", file=sys.stderr)
        return 2

    if rehearsal and (not write or str(write).startswith("UNRESOLVED")):
        print("*** DRESS_REHEARSAL: no durable destination bound; credential "
              "scope NOT verified ***", file=sys.stderr)
        return 0
    if not os.environ.get("HF_TOKEN") and not (
            write and not str(write).startswith("hf://")):
        print("REFUSED: HF_TOKEN is unset; the pod can neither fetch the "
              "episode corpus nor mirror the journal", file=sys.stderr)
        return 2
    try:
        report = check(read_source, write)
    except ScopeError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        with open(guard_path(a.out, MODE_WRITE), "w", encoding="utf-8") as fh:
            fh.write(payload)
    print(payload.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
