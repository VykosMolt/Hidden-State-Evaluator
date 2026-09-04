#!/usr/bin/env bash
# Build and upload an immutable JLens stage package.
#
# The archive contains the complete selected source tree, MANIFEST.sha256,
# and PINNED_INPUTS.json.  The manifest is verified locally before any HF
# mutation.  Runtime result paths are separate and include a run_id.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PYTHON:-venv/bin/python}
STAGING=${STAGING:-Vykos/ouro-jlens-staging}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if ! [[ "$STAGING" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}(/[A-Za-z0-9][A-Za-z0-9_.-]{0,95})?$ ]]; then
  echo "STAGING is not a valid Hugging Face repository" >&2
  exit 2
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

archive="$tmp/ouro_jlens_stage.tar.gz"
metadata=$("$PY" -m ouro_jlens.publish stage-create --root . --output "$archive")
printf '%s\n' "$metadata"
"$PY" -m ouro_jlens.publish stage-verify --archive "$archive"
manifest_sha=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "$metadata")
remote_stage=${REMOTE_STAGE:-"stages/$manifest_sha/ouro_jlens_stage.tar.gz"}
if [[ "$remote_stage" != *"$manifest_sha"* ]]; then
  echo "REMOTE_STAGE must contain the verified manifest digest" >&2
  exit 2
fi
"$PY" -c 'from ouro_jlens.publish import safe_relative; import re,sys; p=safe_relative(sys.argv[1]); d=re.findall(r"[0-9a-f]{64}", p); ok=bool(re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", p)) and d == [sys.argv[2]]; sys.exit(0 if ok else "REMOTE_STAGE must be a safe path containing exactly one verified manifest digest")' "$remote_stage" "$manifest_sha"

if [[ "${DRY_RUN:-0}" == "1" || "${1:-}" == "--dry-run" ]]; then
  printf 'DRY RUN: verified stage only; no Hugging Face mutation (path=%s)\n' "$remote_stage"
  exit 0
fi

# --exist-ok makes the intended idempotency explicit.  No failure is hidden:
# authentication, network, or upload errors terminate this script.
hf repo create "$STAGING" --private --exist-ok --quiet
# A digest-addressed stage path is immutable.  Inspect an existing object
# before attempting upload; only an explicit not-found response permits the
# first upload, while authentication/network uncertainty fails closed.
existing_dir="$tmp/existing"
mkdir -p "$existing_dir"
set +e
hf download "$STAGING" "$remote_stage" --local-dir "$existing_dir" --quiet \
  > /dev/null 2> "$tmp/download.err"
download_rc=$?
set -e
if [[ "$download_rc" -eq 0 ]]; then
  existing_archive="$existing_dir/$remote_stage"
  if [[ ! -f "$existing_archive" ]]; then
    existing_archive="$existing_dir/$(basename "$remote_stage")"
  fi
  test -f "$existing_archive"
  cmp -s "$archive" "$existing_archive" || {
    echo "immutable stage path already contains different bytes" >&2
    exit 2
  }
  "$PY" -m ouro_jlens.publish stage-verify --archive "$existing_archive"
  printf 'stage_path=%s manifest_sha256=%s archive_sha256=' "$remote_stage" "$manifest_sha"
  sha256sum "$archive" | cut -d' ' -f1
  exit 0
fi
if ! grep -Eiq '404|not found|does not exist|cannot find' "$tmp/download.err"; then
  echo "could not determine whether the immutable stage path exists" >&2
  exit 2
fi
# Read the object back and verify both its bytes and its internal manifest.
# A successful upload response alone is not an acknowledgement.
hf upload "$STAGING" "$archive" "$remote_stage" --quiet
remote_dir="$tmp/remote"
mkdir -p "$remote_dir"
hf download "$STAGING" "$remote_stage" --local-dir "$remote_dir" --quiet
remote_archive="$remote_dir/$remote_stage"
if [[ ! -f "$remote_archive" ]]; then
  remote_archive="$remote_dir/$(basename "$remote_stage")"
fi
test -f "$remote_archive"
cmp -s "$archive" "$remote_archive"
"$PY" -m ouro_jlens.publish stage-verify --archive "$remote_archive"
printf 'stage_path=%s manifest_sha256=%s archive_sha256=' "$remote_stage" "$manifest_sha"
sha256sum "$archive" | cut -d' ' -f1
