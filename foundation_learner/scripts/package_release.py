#!/usr/bin/env python3
"""Deterministic release packaging (contract §19).

Builds ``FOUNDATION_LEARNER_B200_V0.1.0.zip`` with the O1 algorithm — sorted
entries, ``date_time=(1980,1,1,0,0,0)``, ``external_attr = 0o644 << 16``,
``ZIP_DEFLATED`` level 9, secret-pattern deny-list — over an EXPLICIT content
manifest:

* ``foundation_learner/**`` — code, docs, tests, configs, deploy files;
* ``artifacts_fl/pregen/**`` — the pre-generated shards (also bitwise
  reproducible from ``scripts/pregenerate_all.py``, which contains no
  wall-clock value; staging them is a convenience, not a trust root);

minus ``reports/local_runs/``, ``__pycache__/``, ``*.pyc``, ``*.tmp`` and the
release artefacts themselves.

Also written:

* ``foundation_learner/SHA256SUMS`` — O1 two-space format over the whole
  content manifest, EXACT coverage (a bijection between the listed files and
  the files actually present: neither a missing nor an extra file passes);
* ``<zip>.sha256`` — the sidecar digest.

Usage::

    python -m foundation_learner.scripts.package_release --dry-run
    python -m foundation_learner.scripts.package_release
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from foundation_learner.campaign import o1_isolation  # noqa: E402
from foundation_learner.campaign import result_verifier as rv  # noqa: E402
from foundation_learner.ecology.base import sha256_file  # noqa: E402

PACKAGE_NAME = "FOUNDATION_LEARNER_B200_V0"
ZIP_NAME = "FOUNDATION_LEARNER_B200_V0.1.0.zip"
SUMS_NAME = "SHA256SUMS"
MANIFEST_DIR_NAME = "MANIFESTS"

#: mirrored into ``artifacts_fl/pregen*/MANIFESTS/`` so that packaging (and
#: review) can find them while the raw shard directories stay git-ignored.
PREGEN_MANIFEST_FILES = ("PREGEN_MANIFEST.json", "SHARD_SUMS.json",
                         "family_split_manifest.json")

EXCLUDED_DIRS = ("__pycache__", "local_runs", ".pytest_cache", ".git")
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".tmp", ".orig", ".rej")
EXCLUDED_NAMES = (SUMS_NAME, ZIP_NAME, ZIP_NAME + ".sha256")


class PackagingError(RuntimeError):
    """A packaging invariant refused."""


def _excluded(rel: str) -> bool:
    parts = rel.replace(os.sep, "/").split("/")
    if any(part in EXCLUDED_DIRS for part in parts):
        return True
    if parts[-1] in EXCLUDED_NAMES:
        return True
    return any(rel.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def _tree(root: str, repo_root: str) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, repo_root).replace(os.sep, "/")
            if _excluded(rel):
                continue
            out.append(full)
    return sorted(out)


def mirror_pregen_manifests(pregen_root: str) -> dict:
    """Copy the pregen manifests into ``<pregen_root>/MANIFESTS/``.

    The raw shard directories are git-ignored (they are large and bitwise
    reproducible), but their manifests must stay findable — for packaging, for
    the campaign manifest, and for review.  The mirror is verified by hash on
    every run, so a stale copy is an error rather than a silent divergence.
    """
    pregen_root = os.path.abspath(pregen_root)
    target = os.path.join(pregen_root, MANIFEST_DIR_NAME)
    os.makedirs(target, exist_ok=True)
    mirrored = {}
    for name in PREGEN_MANIFEST_FILES:
        source = os.path.join(pregen_root, name)
        if not os.path.isfile(source):
            raise PackagingError(
                f"pregeneration is incomplete: {name} missing under "
                f"{pregen_root}")
        destination = os.path.join(target, name)
        payload = open(source, "rb").read()
        if not os.path.isfile(destination) or \
                open(destination, "rb").read() != payload:
            tmp = destination + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, destination)
        if sha256_file(source) != sha256_file(destination):  # pragma: no cover
            raise PackagingError(f"manifest mirror mismatch for {name}")
        mirrored[name] = sha256_file(destination)
    return mirrored


def content_manifest(repo_root: str, *, pregen_root: str | None,
                     include_pregen: bool = True) -> list[str]:
    """The explicit list of files the release contains."""
    repo_root = os.path.abspath(repo_root)
    package = os.path.join(repo_root, "foundation_learner")
    if not os.path.isdir(package):
        raise PackagingError(f"package directory not found: {package}")
    files = _tree(package, repo_root)
    if include_pregen:
        if not pregen_root or not os.path.isdir(pregen_root):
            raise PackagingError(
                f"pregenerated data not found at {pregen_root!r}; run "
                "scripts/pregenerate_all.py, or pass --no-pregen for a "
                "code-only dry run (which is NOT a release)")
        files += _tree(os.path.abspath(pregen_root), repo_root)
    return sorted(files)


def write_sha256sums(files: list[str], repo_root: str, out_path: str) -> str:
    lines = []
    for path in files:
        rel = os.path.relpath(path, repo_root).replace(os.sep, "/")
        lines.append(f"{sha256_file(path)}  {rel}")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    return sha256_file(out_path)


def verify_exact_coverage(sums_path: str, repo_root: str, *,
                          pregen_root: str | None,
                          include_pregen: bool = True) -> dict:
    """Bijection check: every listed file exists and every file is listed."""
    listed: dict[str, str] = {}
    with open(sums_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            digest, rel = line.rstrip("\n").split("  ", 1)
            if rel in listed:
                raise PackagingError(f"{sums_path}:{lineno}: duplicate entry {rel}")
            listed[rel] = digest
    present = {os.path.relpath(p, repo_root).replace(os.sep, "/")
               for p in content_manifest(repo_root, pregen_root=pregen_root,
                                         include_pregen=include_pregen)}
    missing = sorted(set(listed) - present)
    unlisted = sorted(present - set(listed))
    bad = []
    for rel, digest in sorted(listed.items()):
        full = os.path.join(repo_root, rel)
        if os.path.isfile(full) and sha256_file(full) != digest:
            bad.append(rel)
    report = {"listed": len(listed), "present": len(present),
              "missing": missing, "unlisted": unlisted, "mismatched": bad,
              "ok": not missing and not unlisted and not bad}
    if not report["ok"]:
        raise PackagingError(
            f"SHA256SUMS coverage is not exact: {len(missing)} listed-but-absent, "
            f"{len(unlisted)} present-but-unlisted, {len(bad)} hash mismatches "
            f"(first: {missing[:1]} {unlisted[:1]} {bad[:1]})")
    return report


def build(repo_root: str, *, pregen_root: str | None, out_dir: str,
          include_pregen: bool = True, label: str = "RELEASE",
          guard: o1_isolation.IsolationGuard | None = None) -> dict:
    """Write SHA256SUMS, the deterministic zip, and its sidecar."""
    guard = guard or o1_isolation.default_guard()
    repo_root = os.path.abspath(repo_root)
    os.makedirs(out_dir, exist_ok=True)
    mirrored = None
    if include_pregen and pregen_root:
        mirrored = mirror_pregen_manifests(pregen_root)
    files = content_manifest(repo_root, pregen_root=pregen_root,
                             include_pregen=include_pregen)
    for path in files:
        rv.check_no_secrets(path)
    sums_path = os.path.join(repo_root, "foundation_learner", SUMS_NAME)
    sums_sha256 = write_sha256sums(files, repo_root, sums_path)
    coverage = verify_exact_coverage(sums_path, repo_root,
                                     pregen_root=pregen_root,
                                     include_pregen=include_pregen)
    zip_path = os.path.join(out_dir, ZIP_NAME)
    zip_sha256 = rv.deterministic_zip(files + [sums_path], repo_root, zip_path,
                                      guard=guard)
    sidecar = zip_path + ".sha256"
    with open(sidecar, "w", encoding="utf-8") as fh:
        fh.write(f"{zip_sha256}  {ZIP_NAME}\n")
    return {
        "schema": "flb200.package_release.v1",
        "label": label,
        "repo_root": repo_root,
        "zip_path": zip_path,
        "zip_name": ZIP_NAME,
        "zip_sha256": zip_sha256,
        "zip_bytes": os.path.getsize(zip_path),
        "sha256sums_path": sums_path,
        "sha256sums_sha256": sums_sha256,
        "entries": len(files) + 1,
        "coverage": coverage,
        "include_pregen": bool(include_pregen),
        "pregen_root": pregen_root if include_pregen else None,
        "pregen_manifests_mirrored": mirrored,
        "excluded": {"dirs": list(EXCLUDED_DIRS),
                     "suffixes": list(EXCLUDED_SUFFIXES),
                     "names": list(EXCLUDED_NAMES)},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic FL release zip")
    parser.add_argument("--repo-root", default=_PKG_PARENT)
    parser.add_argument("--pregen",
                        default=os.path.join(_PKG_PARENT, "artifacts_fl", "pregen"),
                        help="pre-generated data root")
    parser.add_argument("--out", default=_PKG_PARENT,
                        help="where to write the zip (default: repo root)")
    parser.add_argument("--no-pregen", action="store_true",
                        help="code-only build; NOT a release (dry runs only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build into reports/local_runs/ and label it a "
                             "THROWAWAY dry run")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out
    label = "RELEASE"
    if args.dry_run:
        out_dir = os.path.join(args.repo_root, "foundation_learner", "reports",
                               "local_runs", "package_dry_run")
        label = "THROWAWAY_DRY_RUN"
    report = build(args.repo_root, pregen_root=args.pregen, out_dir=out_dir,
                   include_pregen=not args.no_pregen, label=label)
    print(json.dumps({k: v for k, v in report.items() if k != "coverage"},
                     indent=2, sort_keys=True))
    print(f"coverage: {report['coverage']['listed']} files, exact="
          f"{report['coverage']['ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
