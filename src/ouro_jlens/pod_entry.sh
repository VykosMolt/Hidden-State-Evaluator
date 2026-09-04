#!/usr/bin/env bash
# Remote entry point for one immutable, supervised JLens run.
# Required environment: RUN_ID, HF_TOKEN, and (optionally) RESULTS/STAGING.
set -euo pipefail

STAGING=${STAGING:-Vykos/ouro-jlens-staging}
RESULTS=${RESULTS:?RESULTS repository is required}
STAGE_PATH=${STAGE_PATH:?STAGE_PATH must be the digest-pinned path emitted by stage_upload.sh}
RUN_ID=${RUN_ID:?RUN_ID is required; every run must have an immutable namespace}
HF_TOKEN=${HF_TOKEN:?HF_TOKEN is required by the staged setup}
N=${N_PROMPTS:-100}
DB=${DIM_BATCH:-32}
WORK=${WORK:-/workspace/ouro_project}
PY=${PYTHON:-python}

if ! [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "RUN_ID is not a safe immutable path component" >&2
  exit 2
fi
if ! [[ "$RESULTS" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}(/[A-Za-z0-9][A-Za-z0-9_.-]{0,95})?$ ]] ||
   ! [[ "$STAGING" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}(/[A-Za-z0-9][A-Za-z0-9_.-]{0,95})?$ ]]; then
  echo "RESULTS/STAGING is not a safe Hugging Face repository" >&2
  exit 2
fi
if ! [[ "$STAGE_PATH" =~ ^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$ ]] ||
   [[ "$STAGE_PATH" == *".."* ]] || ! [[ "$STAGE_PATH" =~ [0-9a-f]{64} ]]; then
  echo "STAGE_PATH is not a safe manifest-digest path" >&2
  exit 2
fi
stage_path_digest=$("$PY" -c 'import re,sys; m=re.findall(r"[0-9a-f]{64}", sys.argv[1]); print(m[0] if len(m) == 1 else "")' "$STAGE_PATH")
if [[ -z "$stage_path_digest" ]]; then
  echo "STAGE_PATH must contain exactly one manifest digest" >&2
  exit 2
fi

[[ ! -L "$WORK" ]] || {
  echo "work root is a symlink: $WORK" >&2
  exit 2
}
cd "$WORK" 2>/dev/null || {
  mkdir -p "$WORK"
  cd "$WORK"
}
export RESULTS STAGING STAGE_PATH RUN_ID HF_TOKEN

# Download only the pinned stage archive. Verify it in isolation before
# extracting into a restartable workspace; the extractor rejects traversal,
# links, special files, and manifest/pin mismatches.
stage_tmp=$(mktemp -d "$WORK/.jlens-stage.XXXXXX")
hf download "$STAGING" "$STAGE_PATH" --local-dir "$stage_tmp" --quiet
stage_archive="$stage_tmp/$STAGE_PATH"
if [[ ! -f "$stage_archive" ]]; then
stage_archive="$stage_tmp/$(basename "$STAGE_PATH")"
fi
test -f "$stage_archive"
stage_archive_metadata=$("$PY" -m ouro_jlens.publish stage-verify --archive "$stage_archive")
archive_manifest=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "$stage_archive_metadata")
if [[ "$archive_manifest" != "$stage_path_digest" ]]; then
  echo "stage archive does not match its digest-addressed path" >&2
  exit 2
fi
"$PY" -m ouro_jlens.publish stage-extract --archive "$stage_archive" --destination "$WORK"
rm -rf -- "$stage_tmp"
export PYTHONPATH="$WORK/src${PYTHONPATH:+:$PYTHONPATH}"
stage_metadata=$("$PY" -m ouro_jlens.publish stage-verify --root "$WORK" --allow-extra)
export STAGE_MANIFEST_SHA256=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "$stage_metadata")

RUN_ROOT="$WORK/artifacts/jlens/runs/$RUN_ID"
[[ ! -L "$RUN_ROOT" ]] || {
  echo "run output root is a symlink: $RUN_ROOT" >&2
  exit 2
}
mkdir -p "$RUN_ROOT"

publish_status() {
  local state=$1 stage=$2 code=${3:-}
  local -a extra=()
  if [[ -n "$code" ]]; then
    extra+=(--exit-code "$code")
  fi
  # Status records are immutable heartbeat artifacts. If this command fails,
  # the caller retains the original job status and reports the publication
  # failure separately.
  "$PY" -m ouro_jlens.publish status \
    --repo "$RESULTS" --run-id "$RUN_ID" --local-root "$RUN_ROOT" \
    --receipt-root "$RUN_ROOT/receipts" --state "$state" --stage "$stage" \
    "${extra[@]}" --kind heartbeat
}

on_exit() {
  local job_rc=$? status_rc=0
  trap - EXIT
  if [[ "$job_rc" -eq 0 ]]; then
    publish_status succeeded complete 0 || status_rc=$?
  else
    publish_status failed run "$job_rc" || status_rc=$?
  fi
  # Publish an index for failed runs too, so the controller can recover every
  # acknowledged shard and status even when the computation's exit is nonzero.
  if [[ "$status_rc" -eq 0 ]]; then
    "$PY" -m ouro_jlens.publish index \
      --repo "$RESULTS" --run-id "$RUN_ID" --local-root "$RUN_ROOT" \
      --receipt-root-path "$RUN_ROOT/receipts" || status_rc=$?
  fi
  # A successful computation with an unacknowledged terminal status/index is not a
  # successful run. Failed computations retain their original exit code.
  if [[ "$job_rc" -eq 0 && "$status_rc" -ne 0 ]]; then
    exit "$status_rc"
  fi
  exit "$job_rc"
}
trap on_exit EXIT

bash src/ouro_jlens/setup_b300.sh
publish_status running setup
bash src/ouro_jlens/run_b300.sh "$N" "$DB"
