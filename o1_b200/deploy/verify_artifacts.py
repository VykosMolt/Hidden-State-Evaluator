#!/usr/bin/env python3
"""Artifact verification script (deployment side).

Consumes TRANSFER_MANIFEST.json and hash-verifies EVERY listed artifact
(including the out-of-band model checkpoint tree) before any scientific
process may start.  Exits non-zero on the first mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: Excluded from every tree digest, on the build host AND on the pod:
#: the two manifests (a tree that contains its own digest has no fixed
#: point) and bytecode (present on a host that ran the tests, absent from
#: the image by .dockerignore).  make_transfer_manifest applies the same
#: rule; the two MUST agree or ARTIFACT_VERIFY refuses every pod.
TREE_DIGEST_EXCLUDED_NAMES = frozenset(
    {"TRANSFER_MANIFEST.json", "POD_TRANSFER_MANIFEST.json"})
TREE_DIGEST_EXCLUDED_DIRS = frozenset({"__pycache__"})


def tree_digest_includes(rel: str, name: str) -> bool:
    if name.endswith(".pyc"):
        return False
    if name in TREE_DIGEST_EXCLUDED_NAMES and os.sep not in rel and "/" not in rel:
        return False
    return True


def sha256_tree(path: str) -> str:
    if os.path.isfile(path):
        return sha256_file(path)
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in TREE_DIGEST_EXCLUDED_DIRS)
        for name in sorted(files):
            full = os.path.join(root, name)
            if not tree_digest_includes(
                    os.path.relpath(full, path).replace(os.sep, "/"), name):
                continue
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(bytes.fromhex(sha256_file(full)))
            h.update(b"\n")
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", default=".")
    a = p.parse_args()
    with open(a.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema") != "o1b200.transfer_manifest.v1":
        print("FATAL: unknown manifest schema", file=sys.stderr)
        return 2
    failures = 0
    for name, spec in sorted(manifest["artifacts"].items()):
        path = spec["path"]
        if not os.path.isabs(path):
            path = os.path.join(a.root, path)
        if not os.path.exists(path):
            print(f"MISSING  {name}: {path}", file=sys.stderr)
            failures += 1
            continue
        got = sha256_tree(path)
        if got != spec["sha256"]:
            print(f"MISMATCH {name}: {got[:16]}... != {spec['sha256'][:16]}...",
                  file=sys.stderr)
            failures += 1
        else:
            print(f"OK       {name}")
    if failures:
        print(f"FATAL: {failures} artifact verification failures — no "
              f"scientific process may start", file=sys.stderr)
        return 1
    print("ALL ARTIFACTS VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
