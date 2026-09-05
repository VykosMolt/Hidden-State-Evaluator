#!/usr/bin/env bash
# Build and upload an immutable JLens stage package.
#
# The archive contains the complete selected source tree, MANIFEST.sha256,
# and PINNED_INPUTS.json.  The manifest is verified locally before any HF
# mutation.  Runtime result paths are separate and include a run_id.
set -euo pipefail

# The stage archive is public-source data, so no credential belongs in the
# environment used to build or verify it.  A private token file is required
# only after the explicit dry-run exit below.
unset HF_TOKEN JLENS_HF_TOKEN_BOOTSTRAP

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

JLENS_HF_TOKEN_FILE=${JLENS_HF_TOKEN_FILE:?JLENS_HF_TOKEN_FILE is required for stage publication}
"$PY" - "$JLENS_HF_TOKEN_FILE" <<'PY'
import sys

from ouro_jlens.publish import read_hf_token_file

read_hf_token_file(sys.argv[1])
PY

# HfPublisher submits the archive through HfApi.create_commit with an exact
# parent OID.  It also reconciles unknown outcomes by digest and rejects a
# differing immutable object; this script must not reintroduce a CLI
# read-before-write race.
"$PY" -m ouro_jlens.publish stage-publish \
  --repo "$STAGING" --token-file "$JLENS_HF_TOKEN_FILE" \
  --archive "$archive" --remote "$remote_stage"
