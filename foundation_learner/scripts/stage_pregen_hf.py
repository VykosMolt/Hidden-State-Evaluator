#!/usr/bin/env python3
"""Stage the pregenerated episode corpus to the PRIVATE staging repo.

The ~480 MB corpus is neither in Git nor in the container image, and
``campaign/entry.py`` refuses a session whose ``pregen_root`` does not exist.
Without this step a combined O1 -> FL rental acquires an accelerator, runs O1,
hands over, and only then discovers there is nothing to train on.

Usage (WRITE-capable HF_TOKEN in the environment):

    python -m foundation_learner.scripts.stage_pregen_hf \\
        --repo Vykos/o1-b200-staging upload
    python -m foundation_learner.scripts.stage_pregen_hf \\
        --repo Vykos/o1-b200-staging verify

``upload`` refuses a public repository and mirrors ``artifacts_fl/pregen``
under the same path-in-repo the pod's default remote prefix expects.

``verify`` is the real test: it re-downloads the corpus through
``campaign/fetch_pregen.py`` -- the pod's own ingestion path, offline-flag
stripping and all -- and re-hashes every shard against SHARD_SUMS.json.
Sealed shards are checked by their on-disk digest, so SEALED_TEST is never
opened.  Writes PREGEN_STAGING_RECORD.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from foundation_learner.campaign import fetch_pregen  # noqa: E402
from foundation_learner.campaign.hf_transfer import child_env  # noqa: E402

LOCAL_PREGEN = os.path.join(_ROOT, "artifacts_fl", "pregen")
REMOTE_PREFIX = fetch_pregen.DEFAULT_REMOTE_PREFIX          # artifacts_fl/pregen
RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "deploy", "PREGEN_STAGING_RECORD.json")


def _helper(args: list[str], timeout: float = 7200.0) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "foundation_learner.campaign.hf_transfer",
         *args],
        capture_output=True, text=True, timeout=timeout, cwd=_ROOT,
        env=child_env(os.environ.get("HF_TOKEN")))
    if proc.returncode != 0:
        raise SystemExit(f"helper failed: {(proc.stdout + proc.stderr)[-800:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def upload(repo: str) -> int:
    if not os.path.isdir(LOCAL_PREGEN):
        print(f"REFUSED: {LOCAL_PREGEN} does not exist", file=sys.stderr)
        return 2
    stats = fetch_pregen.verify_tree(LOCAL_PREGEN)
    print(f"local corpus verified before upload: {stats}")
    out = _helper(["upload-folder", "--repo", repo,
                   "--local", LOCAL_PREGEN, "--prefix", REMOTE_PREFIX])
    print(f"uploaded {out['count']} file(s) -> {repo}/{REMOTE_PREFIX}")
    print("upload complete; now run: ... verify")
    return 0


def verify(repo: str) -> int:
    """Re-download through the POD's ingestion path and re-hash everything."""
    tmp = tempfile.mkdtemp(prefix="fl_pregen_verify_")
    try:
        report = fetch_pregen.fetch(f"hf://{repo}/{REMOTE_PREFIX}",
                                    os.path.join(tmp, "pregen"))
        record = {
            "schema": "flb200.pregen_staging_record.v1",
            "repo": repo,
            "remote_prefix": REMOTE_PREFIX,
            "pod_fetch_path": "campaign/fetch_pregen.py (the same module the "
                              "pod runs; SHARD_SUMS.json re-hashed, sealed "
                              "shards verified by on-disk digest and never "
                              "opened)",
            "shards_verified": report["shards_verified"],
            "sealed_shards": report["sealed_shards"],
            "pregen_download_test": "PASS",
        }
        with open(os.path.abspath(RECORD), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"PREGEN DOWNLOAD TEST: PASS "
              f"({report['shards_verified']} shards, "
              f"{report['sealed_shards']} sealed) -> {os.path.abspath(RECORD)}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["upload", "verify"])
    p.add_argument("--repo", required=True)
    a = p.parse_args()
    if not os.environ.get("HF_TOKEN"):
        print("REFUSED: HF_TOKEN is unset", file=sys.stderr)
        return 2
    return upload(a.repo) if a.action == "upload" else verify(a.repo)


if __name__ == "__main__":
    raise SystemExit(main())
