#!/usr/bin/env python3
"""Pod-side credential-scope preflight: prove the token before spending time.

One HF_TOKEN has to serve both directions of an interruptible session:

  * READ   ``O1_B200_ARTIFACT_SOURCE``   — the staged checkpoint and cohorts;
  * WRITE  ``O1_B200_RESULT_DESTINATION`` — committed rows and checkpoints,
    pushed continuously so an eviction is recoverable.

A read-only token passes artifact ingestion, passes every hardware and
scientific gate, runs for hours, and then loses every durability push.  Under
interruptible capacity that is the worst available failure: the run looks
healthy right up to the eviction that destroys it.  Nothing else in the stack
detects it, because nothing else writes until there are results to write.

So this runs FIRST, before the multi-gigabyte fetch, and it fails closed.
Read is proven by auth_check; write is proven by writing — the probe uploads
a few bytes under a dedicated ``.preflight/`` prefix and deletes them again.
Non-``hf://`` destinations (a mounted path) are checked directly instead.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

from .hf_transfer import child_env

READ_ENV = "O1_B200_ARTIFACT_SOURCE"
WRITE_ENV = "O1_B200_RESULT_DESTINATION"


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


#: HTTP status lines worth keeping: for HfHubHTTPError the status is on the
#: FIRST line of a multi-line message while the human blurb is last, so a
#: naive "last line" loses exactly the part that distinguishes a transient
#: outage from a genuinely wrong token.
_STATUS_RE = re.compile(r"\b([45]\d\d)\s+(?:Client|Server)\s+Error\b")

#: Statuses that are the caller's fault and will recur on a fresh pod.
DETERMINISTIC_STATUSES = (401, 403, 404)


def http_status(text: str) -> int | None:
    m = _STATUS_RE.search(text)
    return int(m.group(1)) if m else None


def _helper_error(text: str, limit: int = 400) -> str:
    """The informative line, not a raw tail.

    A Python traceback ends with the exception line, which is usually what
    the operator needs — but for an HTTP failure the status line comes
    first.  Keep the status when there is one, then the final line; slicing
    the last N characters instead lands mid-frame and prints source
    fragments.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return "(helper produced no output)"
    tail = lines[-1]
    status_line = next((ln for ln in lines if _STATUS_RE.search(ln)), None)
    if status_line and status_line != tail:
        return f"{status_line[:limit]} | {tail[:limit]}"
    return tail[:limit]


class TransientScopeError(ScopeError):
    """The hub was unreachable or erroring; a fresh attempt may succeed."""


def _run_helper(repo: str, mode: str, timeout: float,
                attempts: int = 4, sleep=time.sleep) -> dict:
    """Probe once, retrying only what a retry can fix.

    Without this, a single HF 5xx/429/timeout at pod start killed the
    entrypoint, which the driver reads as an eviction and answers with a
    reacquisition — and because evictions are counted cumulatively, one hub
    hiccup could end a session that had already produced hours of rows.
    401/403/404 are the caller's fault and recur on a fresh pod, so they are
    raised immediately as deterministic.
    """
    from ..provider.runpod.redaction import redact
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "o1_b200.runner.hf_transfer", "scope",
                 "--repo", repo, "--mode", mode],
                capture_output=True, text=True, timeout=timeout,
                env=child_env(os.environ.get("HF_TOKEN")))
        except subprocess.TimeoutExpired:
            last = f"helper timed out after {timeout:.0f}s"
            if attempt < attempts:
                sleep(min(30.0, 2.0 ** attempt))
                continue
            raise TransientScopeError(
                f"{mode.upper()} scope check for {repo}: {last}") from None
        if proc.returncode == 0:
            try:
                return json.loads(proc.stdout.strip().splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                raise ScopeError(
                    f"{mode} scope check for {repo} produced no "
                    f"result") from None
        combined = proc.stdout + proc.stderr
        last = _helper_error(combined)
        status = http_status(combined)
        if status in DETERMINISTIC_STATUSES:
            raise ScopeError(redact(
                f"{mode.upper()} scope check failed for {repo}: {last}"))
        if attempt < attempts:
            sleep(min(30.0, 2.0 ** attempt))
    raise TransientScopeError(redact(
        f"{mode.upper()} scope check for {repo} failed {attempts}x "
        f"(last: {last}); treating as a transient hub failure"))


def _check_local_write(path: str) -> dict:
    probe = os.path.join(path, ".preflight_write_probe")
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
        # Same repo in both directions still needs the write probe: read
        # access says nothing about whether pushes will be accepted.
        checks.append(run(write_repo, "write"))
    elif write_uri and not write_uri.startswith("UNRESOLVED"):
        checks.append(_check_local_write(write_uri))
    else:
        # production_entry requires this anyway; refusing here means the
        # session dies in seconds rather than after the gates.  A verified
        # read with no verified write is the exact shape of the failure
        # this preflight exists to prevent.
        raise ScopeError(
            f"{WRITE_ENV} is unset or UNRESOLVED; an interruptible session "
            f"with no durable destination loses every committed row on "
            f"eviction")
    if not checks:
        raise ScopeError(
            f"neither {READ_ENV} nor {WRITE_ENV} names a destination this "
            f"preflight can verify; refusing to start a session whose "
            f"durability is unproven")
    for res in checks:
        if res["mode"] == "write" and not res.get("write"):
            raise ScopeError(
                f"the token can read {res['repo']} but cannot write to it; "
                f"an interruptible session with no durable push loses "
                f"everything on eviction")
    return {"schema": "o1b300.hf_scope_report.v1", "checks": checks}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--read-source", default=os.environ.get(READ_ENV, ""))
    p.add_argument("--write-destination", default=os.environ.get(WRITE_ENV, ""))
    p.add_argument("--artifacts-root", default=None,
                   help="where artifacts must appear (default /artifacts)")
    p.add_argument("--manifest", default=None)
    p.add_argument("--out")
    a = p.parse_args()

    # An EMPTY read source is not "nothing to check".  If the pod manifest
    # still expects artifacts that are not on disk, an empty source means
    # the fetch is guaranteed to refuse a few seconds from now — and,
    # worse, parse_repo("") returns None, so the read side would be silently
    # skipped and this preflight would PASS on the write probe alone.  That
    # is exactly how a session config with no artifact_source at all reached
    # a paid pod.
    try:
        from .fetch_artifacts import ARTIFACTS_ROOT, required_from_manifest
        root = a.artifacts_root or ARTIFACTS_ROOT
        manifest = a.manifest or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "deploy", "POD_TRANSFER_MANIFEST.json")
        wanted = required_from_manifest(manifest, root)
        missing = [r for r in wanted
                   if not os.path.exists(os.path.join(root, r))]
    except OSError:
        missing = []
    if missing and not str(a.read_source).startswith("hf://"):
        print("ZERO_TOUCH_ABORTED_AT_HF_SCOPE")
        print(f"REFUSED: {missing} must still be fetched but "
              f"{READ_ENV} is {a.read_source!r}; the read side of this "
              f"preflight cannot be proven and the fetch will refuse "
              f"moments from now", file=sys.stderr)
        return 2

    # A token is only required when a hub repo is actually involved: a pod
    # with mounted artifacts and a mounted durable path needs none, and
    # refusing that configuration would be an invented requirement.
    needs_token = any(str(uri).startswith("hf://")
                      for uri in (a.read_source, a.write_destination))
    if needs_token and not os.environ.get("HF_TOKEN"):
        print("REFUSED: HF_TOKEN is unset but the session names an hf:// "
              "source or destination; the pod can neither fetch the staged "
              "checkpoint nor push results", file=sys.stderr)
        return 2
    try:
        report = check(a.read_source, a.write_destination)
    except TransientScopeError as exc:
        # No marker: a fresh pod may well succeed, so let the driver treat
        # this as an eviction and reacquire rather than ending the session.
        print(f"REFUSED (transient): {exc}", file=sys.stderr)
        return 2
    except ScopeError as exc:
        # The credential or the destination is wrong, and a fresh pod would
        # fail identically.  Emit the marker the off-pod driver greps for so
        # it raises DeterministicPodFailure and stops, instead of paying for
        # a reacquisition that repeats this exact refusal.
        print("ZERO_TOUCH_ABORTED_AT_HF_SCOPE")
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(payload)
    print(payload.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
