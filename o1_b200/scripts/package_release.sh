#!/usr/bin/env bash
# Build the deterministic O1_B300_RUNNER release archive.
#
# Deterministic by construction: sorted entry order, fixed timestamps, fixed
# permissions, no compression-level drift — so two builds of the same tree
# produce byte-identical archives with the same SHA-256.  Generated run
# outputs under reports/local_runs and every __pycache__ are excluded; the
# top-level reports (the validation evidence) are included.
#
# Historical packages are NEVER overwritten: the script refuses if the
# target archive already exists.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="$(cat "$ROOT/o1_b200/VERSION")"
NAME="${1:-O1_B300_RUNNER_v${VERSION}_RUNPOD_PRERENTAL}"
OUT="$ROOT/o1_packages/${NAME}.zip"

if [[ -e "$OUT" ]]; then
  echo "REFUSED: $OUT already exists; validated historical packages are" >&2
  echo "         never overwritten. Bump VERSION or pass a new name." >&2
  exit 3
fi

cd "$ROOT"
# The archive is defined by what is COMMITTED (see the git ls-files below),
# so refuse a dirty tree before touching anything -- otherwise the release
# describes a state that exists in no commit and cannot be reproduced from
# a fresh clone.
DIRTY="$(git status --porcelain -- o1_b200 || true)"
if [[ -n "$DIRTY" ]]; then
  echo "REFUSED: o1_b200/ has uncommitted changes; commit them first so the" >&2
  echo "         release is reproducible from a fresh clone:" >&2
  echo "$DIRTY" | head -20 >&2
  exit 4
fi
# refresh the transfer manifests: they pin tree hashes of policies/, runner/
# and deploy/, and the pod refuses at ARTIFACT_VERIFY if they are stale
# the generator imports the sealed package (torch): use the project
# interpreter, never the system python3
PY="${O1_B200_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PY="${PY:-$(command -v python3)}"
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "REFUSED: $PY cannot import torch; set O1_B200_PYTHON to the project" >&2
  echo "         interpreter (the manifest generator imports the sealed package)." >&2
  exit 4
fi
"$PY" -m o1_b200.runner.make_transfer_manifest >/dev/null
if [[ -n "$(git status --porcelain -- o1_b200/deploy/TRANSFER_MANIFEST.json o1_b200/deploy/POD_TRANSFER_MANIFEST.json || true)" ]]; then
  echo "REFUSED: the transfer manifests were stale and have been regenerated;" >&2
  echo "         commit them, then re-run (a pod built from the stale pair" >&2
  echo "         fails verify_artifacts.py deterministically)." >&2
  exit 4
fi
# refresh the package checksum manifest: it ships inside the archive
./o1_b200/deploy/checksums.sh write >/dev/null
if [[ -n "$(git status --porcelain -- o1_b200/SHA256SUMS || true)" ]]; then
  echo "REFUSED: o1_b200/SHA256SUMS was stale and has been refreshed;" >&2
  echo "         commit it, then re-run so the archive ships a manifest" >&2
  echo "         that matches the commit it claims to be." >&2
  exit 4
fi

python3 - "$OUT" <<'PYEOF'
import os, stat, sys, zipfile

out = sys.argv[1]
root = os.getcwd()
FIXED_DATE = (1980, 1, 1, 0, 0, 0)

#: Regenerable per-run working directories under reports/ — scratch, not
#: evidence.  The evidence is the top-level *_REPORT.json files.
SCRATCH_REPORT_DIRS = ("local_runs", "rehearsal_calibration", "downloaded")


def included(path: str) -> bool:
    parts = path.split(os.sep)
    if "__pycache__" in parts or path.endswith(".pyc"):
        return False
    if parts[:2] == ["o1_b200", "reports"] and len(parts) > 3:
        sub = parts[2]
        if sub in SCRATCH_REPORT_DIRS or sub.startswith("eq_"):
            return False
    return True

# GIT-TRACKED files define the archive.  Walking the working tree shipped
# whatever happened to be lying under o1_b200/ -- a scratch JSON, an editor
# backup, a local config with a credential -- inside an archive advertised
# as deterministic and reproducible.  Tracked-ness is also what makes a
# fresh clone of the pushed commit reproduce the same zip.
import subprocess as _sp
_ls = _sp.run(["git", "ls-files", "-z", "--", "o1_b200"],
              cwd=root, capture_output=True, check=True)
tracked = [r for r in _ls.stdout.decode().split("\0") if r]
if not tracked:
    raise SystemExit("REFUSED: git ls-files listed nothing under o1_b200/; "
                     "refusing to build a release from an unknown tree")
files = sorted(rel for rel in tracked
               if included(rel) and os.path.isfile(os.path.join(root, rel)))

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for rel in files:
        info = zipfile.ZipInfo(rel, date_time=FIXED_DATE)
        mode = os.stat(rel).st_mode
        executable = bool(mode & stat.S_IXUSR)
        info.external_attr = (0o755 if executable else 0o644) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(rel, "rb") as fh:
            zf.writestr(info, fh.read())
print(f"{len(files)} files -> {out}")
PYEOF

( cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256" )
echo "archive: $OUT"
cat "$OUT.sha256"
