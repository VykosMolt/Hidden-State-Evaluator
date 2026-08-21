#!/usr/bin/env python3
"""Pod-side artifact ingestion: bring the staged artifacts onto the pod.

The container bakes the runner, the sealed O1 package, the axis package,
the cohorts and the manifests — but NOT the ~5 GB Ouro-RLTT checkpoint,
which is deliberately never in the image and never in Git.  Something has
to put it on the pod before anything can be verified or loaded, and until
this module existed nothing did: the pod declared no persistent or network
storage and no start-up step fetched anything, so artifact verification
could only ever fail.

Source of truth is O1_B200_ARTIFACT_SOURCE (an ``hf://<ns>/<repo>``
staging URI, part of the deployment identity the authorization commits
to).  The staging repo's layout mirrors /artifacts/ exactly, as written by
provider/runpod/stage_artifacts_hf.py.

Network access goes through the isolated hf_transfer helper, so the pod
stays HF_HUB_OFFLINE-locked for model loading (see that module).  Nothing
here loads a model, and nothing here is trusted: deploy/verify_artifacts.py
re-hashes every fetched artifact against the pinned manifest afterwards,
and the sealed orchestrator hashes them again against its own sealed
values before generating a single row.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from .hf_transfer import child_env

#: destination under the artifacts root -> staging path-in-repo.  Mirrors
#: STAGED in provider/runpod/stage_artifacts_hf.py (identity mapping there,
#: kept explicit here so a future staging rename is a one-line change).
STAGED_LAYOUT = {
    "ouro_rltt_local": "ouro_rltt_local",
    "TOKENIZER_BINDING.json": "TOKENIZER_BINDING.json",
    "AXIS_PACKAGE_V2": "AXIS_PACKAGE_V2",
    "COHORTS": "COHORTS",
    "calibration_seed_matrix.json": "calibration_seed_matrix.json",
    "O1_oracle_reachability_v2.1.0_verified.zip":
        "O1_oracle_reachability_v2.1.0_verified.zip",
}

#: The one artifact that MUST arrive this way; everything else is baked.
REQUIRED = ("ouro_rltt_local",)

ARTIFACTS_ROOT = "/artifacts"


class TransientFetchError(RuntimeError):
    """The transfer failed for a reason a fresh pod may not repeat (network,
    5xx, timeout).  NO deterministic marker: the driver may reacquire."""


class ArtifactFetchError(RuntimeError):
    pass


def parse_hf_source(uri: str) -> str:
    if not uri.startswith("hf://"):
        raise ArtifactFetchError(
            f"artifact source {uri!r} is not an hf:// staging URI; the pod "
            f"has no other ingestion path")
    parts = uri[len("hf://"):].strip("/").split("/")
    if len(parts) < 2 or not all(parts[:2]):
        raise ArtifactFetchError(f"malformed staging URI {uri!r}")
    return "/".join(parts[:2])


def _run_helper(args: list[str], timeout: float) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "o1_b200.runner.hf_transfer", *args],
        capture_output=True, text=True, timeout=timeout,
        env=child_env(os.environ.get("HF_TOKEN")))
    if proc.returncode != 0:
        from ..provider.runpod.redaction import redact
        from .check_hf_scope import (DETERMINISTIC_STATUSES, _helper_error,
                                     http_status)
        combined = proc.stdout + proc.stderr
        msg = redact(f"artifact fetch failed: {_helper_error(combined)}")
        # 401/403/404 repeat on every pod; anything else (5xx, reset,
        # timeout, rate limit) is a hub/network condition a new pod may
        # not see
        if http_status(combined) in DETERMINISTIC_STATUSES:
            raise ArtifactFetchError(msg)
        raise TransientFetchError(msg)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise ArtifactFetchError("artifact fetch produced no result") from None


def required_from_manifest(manifest_path: str,
                           artifacts_root: str = ARTIFACTS_ROOT) -> list[str]:
    """Which artifacts the POD manifest expects under the artifacts root.

    Derived from the manifest rather than hard-coded, so the fetch list can
    never drift from what verification is about to demand.  Everything else
    in the manifest is baked into the image and needs no transfer.
    """
    # Names come from the manifest, which pins the canonical /artifacts
    # location; artifacts_root only says where THIS pod materialises them
    # (tests relocate it).
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    canonical = os.path.abspath(ARTIFACTS_ROOT)
    wanted = []
    for _name, spec in sorted(manifest["artifacts"].items()):
        path = os.path.abspath(spec["path"])
        if path == canonical or path.startswith(canonical + os.sep):
            wanted.append(os.path.relpath(path, canonical))
    return wanted


def fetch(source_uri: str, artifacts_root: str = ARTIFACTS_ROOT,
          timeout: float = 7200.0, runner=None,
          manifest_path: str | None = None) -> dict:
    """Materialise every artifact the pod manifest expects under /artifacts."""
    run = runner or (lambda args: _run_helper(args, timeout))
    os.makedirs(artifacts_root, exist_ok=True)
    wanted = (required_from_manifest(manifest_path, artifacts_root)
              if manifest_path else list(REQUIRED))
    missing = [rel for rel in wanted
               if not os.path.exists(os.path.join(artifacts_root, rel))]
    if not missing:
        # already mounted or fetched on this pod; verification still has
        # the last word on whether the bytes are the right ones
        return {"repo": None, "fetched": [], "already_present": wanted,
                "artifacts_root": artifacts_root}
    repo = parse_hf_source(source_uri)
    fetched = []
    # Stage, then publish atomically.  Downloading straight into
    # /artifacts left a PARTIAL tree behind on any interruption, and the
    # next start saw "the path exists" and skipped the fetch entirely --
    # so verification failed forever, which the driver reads as an
    # eviction and answers with reacquisitions that fail identically.
    staging = os.path.join(artifacts_root, ".incoming")
    for rel in missing:
        remote = STAGED_LAYOUT.get(rel, rel)
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging, exist_ok=True)
        run(["snapshot", "--repo", repo, "--prefix", remote,
             "--local", staging])
        staged = os.path.join(staging, remote)
        if not os.path.exists(staged):
            raise ArtifactFetchError(
                f"artifact {rel!r} is absent from the staging repo {repo!r} "
                f"after fetch; the pod cannot proceed")
        target = os.path.join(artifacts_root, rel)
        os.makedirs(os.path.dirname(os.path.abspath(target)) or "/",
                    exist_ok=True)
        if os.path.exists(target):
            shutil.rmtree(target, ignore_errors=True) \
                if os.path.isdir(target) else os.remove(target)
        os.replace(staged, target)
        fetched.append(rel)
    shutil.rmtree(staging, ignore_errors=True)
    return {"repo": repo, "fetched": fetched,
            "already_present": [r for r in wanted if r not in missing],
            "artifacts_root": artifacts_root}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=os.environ.get(
        "O1_B200_ARTIFACT_SOURCE", ""))
    p.add_argument("--artifacts-root", default=ARTIFACTS_ROOT)
    p.add_argument("--manifest", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "deploy", "POD_TRANSFER_MANIFEST.json"))
    p.add_argument("--out")
    a = p.parse_args()
    # Work out what is actually missing FIRST: a pod whose artifacts are
    # already mounted needs no staging source at all.
    try:
        wanted = required_from_manifest(a.manifest, a.artifacts_root)
    except OSError as exc:
        print(f"REFUSED: cannot read the pod manifest {a.manifest}: {exc}",
              file=sys.stderr)
        return 2
    missing = [rel for rel in wanted
               if not os.path.exists(os.path.join(a.artifacts_root, rel))]
    if missing and (not a.source or a.source.startswith("UNRESOLVED")):
        print(f"REFUSED: {missing} must be transferred but "
              f"O1_B200_ARTIFACT_SOURCE is unset; the pod cannot obtain the "
              f"checkpoint and nothing scientific may run", file=sys.stderr)
        return 2
    try:
        report = fetch(a.source, a.artifacts_root, manifest_path=a.manifest)
    except (TransientFetchError, subprocess.TimeoutExpired) as exc:
        print(f"REFUSED (transient): {exc}", file=sys.stderr)
        return 2
    except ArtifactFetchError as exc:
        # wrong credential, absent artifact, digest mismatch: identical on
        # every pod, so the driver must not pay to repeat it
        print("ZERO_TOUCH_ABORTED_AT_ARTIFACT_FETCH")
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(payload)
    print(payload.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
