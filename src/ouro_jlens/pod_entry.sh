#!/usr/bin/env bash
# Remote entry point for one immutable, supervised JLens run.
# Required environment: RUN_ID, JLENS_ATTEMPT_ID, JLENS_HF_TOKEN_BOOTSTRAP,
# JLENS_IMAGE_DIGEST,
# JLENS_LAUNCH_NONCE, EXPECTED_STAGE_MANIFEST_SHA256,
# EXPECTED_STAGE_SOURCE_HEAD, and JLENS_WORKER_DEADLINE_EPOCH (plus
# RESULTS/STAGING).
set -euo pipefail

# Importing ouro_jlens from the freshly extracted stage must not mutate that
# stage before its root verification.  The private bootstrap umask would make
# Python's implicit __pycache__ directory mode 0700, correctly tripping the
# deterministic-mode verifier.  Keep bootstrap and workload imports read-only;
# bytecode caching is irrelevant for this one-shot run.
export PYTHONDONTWRITEBYTECODE=1

# Hugging Face recommends Xet's high-throughput profile only with at least
# 64 GiB of host RAM. Enable it from the kernel-reported total, not from an
# assumption about the GPU SKU. The normal adaptive Xet path remains active
# on smaller hosts.
host_mem_kib=0
while read -r key value _unit; do
  if [[ "$key" == "MemTotal:" && "$value" =~ ^[0-9]+$ ]]; then
    host_mem_kib=$value
    break
  fi
done < /proc/meminfo
if (( host_mem_kib >= 64 * 1024 * 1024 )); then
  export HF_XET_HIGH_PERFORMANCE=1
else
  unset HF_XET_HIGH_PERFORMANCE
fi
unset host_mem_kib key value _unit

# The provider injects the credential only as a bootstrap value.  Convert it
# immediately into a private 0600 file, then remove both credential variables
# before any Python, pip, git, or network command can run.  The file is kept
# under a freshly-created private temporary directory and is deleted by every
# exit path below.  Do not print or export the bootstrap bytes.
umask 077
token_bootstrap=${JLENS_HF_TOKEN_BOOTSTRAP-}
unset HF_TOKEN JLENS_HF_TOKEN_BOOTSTRAP
if [[ -z "$token_bootstrap" || ! "$token_bootstrap" =~ ^hf_[A-Za-z0-9][A-Za-z0-9_-]{2,}$ ]]; then
  echo "JLENS_HF_TOKEN_BOOTSTRAP is missing or malformed" >&2
  exit 2
fi
JLENS_TOKEN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ouro-jlens-hf-token.XXXXXXXX")
JLENS_HF_TOKEN_FILE="$JLENS_TOKEN_DIR/token"
cleanup_token_channel() {
  if [[ -n "${JLENS_HF_TOKEN_FILE:-}" ]]; then
    rm -f -- "$JLENS_HF_TOKEN_FILE" 2>/dev/null || true
  fi
  if [[ -n "${JLENS_TOKEN_DIR:-}" ]]; then
    rmdir -- "$JLENS_TOKEN_DIR" 2>/dev/null || true
  fi
}
trap cleanup_token_channel EXIT
if ! (umask 077; set -C; printf '%s' "$token_bootstrap" > "$JLENS_HF_TOKEN_FILE"); then
  echo "could not create the private HF token file" >&2
  exit 2
fi
unset token_bootstrap
if [[ ! -f "$JLENS_HF_TOKEN_FILE" || -L "$JLENS_HF_TOKEN_FILE" ]]; then
  echo "private HF token file is not a regular file" >&2
  exit 2
fi
if ! token_mode=$(stat -c '%a' -- "$JLENS_HF_TOKEN_FILE"); then
  echo "could not inspect private HF token file" >&2
  exit 2
fi
if [[ "$token_mode" != "600" ]]; then
  echo "private HF token file does not have mode 0600" >&2
  exit 2
fi

STAGING=${STAGING:-Vykos/ouro-jlens-staging}
RESULTS=${RESULTS:?RESULTS repository is required}
STAGE_PATH=${STAGE_PATH:?STAGE_PATH must be the digest-pinned path emitted by stage_upload.sh}
RUN_ID=${RUN_ID:?RUN_ID is required; every run must have an immutable namespace}
JLENS_ATTEMPT_ID=${JLENS_ATTEMPT_ID:?JLENS_ATTEMPT_ID is required}
N=${N_PROMPTS:-100}
DB=${DIM_BATCH:-32}
WORK=${WORK:-/workspace/ouro_project}
PY=${PYTHON:-python}
JLENS_IMAGE_DIGEST=${JLENS_IMAGE_DIGEST:?JLENS_IMAGE_DIGEST is required}
JLENS_LAUNCH_NONCE=${JLENS_LAUNCH_NONCE:?JLENS_LAUNCH_NONCE is required}
EXPECTED_STAGE_MANIFEST_SHA256=${EXPECTED_STAGE_MANIFEST_SHA256:?EXPECTED_STAGE_MANIFEST_SHA256 is required}
EXPECTED_STAGE_SOURCE_HEAD=${EXPECTED_STAGE_SOURCE_HEAD:?EXPECTED_STAGE_SOURCE_HEAD is required}
JLENS_WORKER_DEADLINE_EPOCH=${JLENS_WORKER_DEADLINE_EPOCH:?JLENS_WORKER_DEADLINE_EPOCH is required}
export JLENS_IMAGE_DIGEST
export JLENS_LAUNCH_NONCE EXPECTED_STAGE_MANIFEST_SHA256 EXPECTED_STAGE_SOURCE_HEAD \
  JLENS_WORKER_DEADLINE_EPOCH

# The bootstrap package came from the locally verified remote stage. Reject
# a wrong image identity before downloading any stage or model data; setup
# repeats this check after the complete stage is extracted.
"$PY" -c '
from ouro_jlens.publish import RUNTIME_IMAGE
import sys
if sys.argv[1] != RUNTIME_IMAGE:
    raise SystemExit("JLENS_IMAGE_DIGEST does not match the approved image")
' "$JLENS_IMAGE_DIGEST"

assert_writable_tree() {
  "$PY" -c 'from pathlib import Path; import sys; from ouro_jlens.publish import _reject_symlink_ancestors; path = Path(sys.argv[1]); _reject_symlink_ancestors(path / "placeholder"); sys.exit(2 if path.is_symlink() else 0)' "$1"
}

if ! [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "RUN_ID is not a safe immutable path component" >&2
  exit 2
fi
if ! [[ "$JLENS_ATTEMPT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "JLENS_ATTEMPT_ID is not a safe immutable path component" >&2
  exit 2
fi
if ! [[ "$JLENS_LAUNCH_NONCE" =~ ^[A-Za-z0-9_-]{32,128}$ ]]; then
  echo "JLENS_LAUNCH_NONCE is not a safe nonce" >&2
  exit 2
fi
if ! [[ "$EXPECTED_STAGE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_STAGE_MANIFEST_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
fi
if ! [[ "$EXPECTED_STAGE_SOURCE_HEAD" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]]; then
  echo "EXPECTED_STAGE_SOURCE_HEAD must be a lowercase 40- or 64-hex git object ID" >&2
  exit 2
fi
if ! [[ "$JLENS_WORKER_DEADLINE_EPOCH" =~ ^[0-9]{1,20}$ ]]; then
  echo "JLENS_WORKER_DEADLINE_EPOCH must be a decimal epoch integer" >&2
  exit 2
fi
if ! "$PY" - "$JLENS_WORKER_DEADLINE_EPOCH" <<'PY'
import sys
import time

deadline = int(sys.argv[1])
if deadline <= time.time():
    raise SystemExit("JLENS_WORKER_DEADLINE_EPOCH is already past")
PY
then
  echo "JLENS_WORKER_DEADLINE_EPOCH must be in the future" >&2
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
if [[ "$stage_path_digest" != "$EXPECTED_STAGE_MANIFEST_SHA256" ]]; then
  echo "STAGE_PATH digest does not match the controller-approved stage manifest" >&2
  exit 2
fi

# The controller derives this deadline from the provider terminateAfter value
# with a large safety margin.  Reserve the final 120 seconds for a graceful
# TERM/EXIT publication window; GNU timeout's KILL therefore lands no later
# than the worker deadline and still precedes provider termination by >=120s.
JLENS_TIMEOUT_GRACE_SECONDS=120
JLENS_MIN_FREE_BYTES=$((80 * 1024 * 1024 * 1024))
timeout_version=$(timeout --version 2>&1 || true)
if [[ "$timeout_version" != *"GNU coreutils"* ]]; then
  echo "GNU timeout is required for the worker deadline" >&2
  exit 2
fi
deadline_remaining() {
  "$PY" - "$JLENS_WORKER_DEADLINE_EPOCH" "$JLENS_TIMEOUT_GRACE_SECONDS" <<'PY'
import math
import sys
import time

deadline = int(sys.argv[1])
grace = int(sys.argv[2])
remaining = deadline - time.time()
if not math.isfinite(remaining) or remaining <= grace:
    raise SystemExit("worker deadline leaves no graceful timeout window")
print(f"{remaining - grace:.3f}")
PY
}

run_with_deadline() {
  local duration
  duration=$(deadline_remaining) || return 124
  timeout --signal=TERM --kill-after="${JLENS_TIMEOUT_GRACE_SECONDS}s" \
    "${duration}s" "$@"
}

run_stage_download_with_token() {
  local duration
  duration=$(deadline_remaining) || return 124
  # The pinned base image has Python but intentionally does not assume the
  # third-party ``hf`` CLI exists before setup.  The bootstrap publisher uses
  # urllib, reads the protected token file itself, and writes only the one
  # digest-addressed stage archive.
  timeout --signal=TERM --kill-after="${JLENS_TIMEOUT_GRACE_SECONDS}s" \
    "${duration}s" "$PY" -m ouro_jlens.publish stage-download \
    --repo "$STAGING" --token-file "$JLENS_HF_TOKEN_FILE" \
    --remote "$STAGE_PATH" --output "$stage_tmp/$STAGE_PATH"
}

check_free_space() {
  # Check the filesystem that will hold the model, lenses, reports, and
  # receipts after extraction.  f_bavail excludes blocks unavailable to the
  # worker, and statvfs fails closed for missing, linked, or non-directory
  # workspace paths.
  "$PY" - "$WORK" "$JLENS_MIN_FREE_BYTES" <<'PY'
import os
import stat
import sys

path = sys.argv[1]
required = int(sys.argv[2])
try:
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or os.path.islink(path):
        raise OSError("workspace is not a regular directory")
    usage = os.statvfs(path)
    available = usage.f_bavail * usage.f_frsize
except (OSError, ValueError, TypeError) as exc:
    raise SystemExit(f"could not inspect workspace free space: {exc}") from exc
if usage.f_bavail <= 0 or usage.f_frsize <= 0 or available < required:
    raise SystemExit(
        f"workspace has insufficient free space: {available} bytes available, "
        f"{required} required"
    )
PY
}

# Query the physical lease before downloading any stage or model bytes. A
# successful command with more than one non-empty name is not acceptable: the
# paid contract binds this run to one B300 device, not merely to a CUDA host.
# ``timeout`` is deliberately fixed and fail-closed.
gpu_names=""
if ! gpu_names=$(timeout --kill-after=2s 30s nvidia-smi \
    --query-gpu=name --format=csv,noheader,nounits 2>&1); then
  echo "nvidia-smi did not complete successfully; refusing stage download" >&2
  exit 2
fi
mapfile -t gpu_lines < <(printf '%s\n' "$gpu_names" | awk 'NF { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); print }')
if (( ${#gpu_lines[@]} != 1 )); then
  echo "expected exactly one GPU from nvidia-smi, found ${#gpu_lines[@]}" >&2
  exit 2
fi
if ! "$PY" - "${gpu_lines[0]}" <<'PY'
import re
import sys

name = sys.argv[1]
if re.search(r"(?i)(?<![A-Za-z0-9_])B300(?![A-Za-z0-9_])", name) is None:
    raise SystemExit("the leased GPU name does not contain a word-boundary B300")
PY
then
  echo "nvidia-smi did not report the required B300 device" >&2
  exit 2
fi

if [[ -e "$WORK" || -L "$WORK" ]]; then
  echo "work root already exists; a paid attempt requires a fresh workspace: $WORK" >&2
  exit 2
fi
work_parent=$(dirname -- "$WORK")
mkdir -p -- "$work_parent"
assert_writable_tree "$work_parent"
assert_writable_tree "$WORK"
export RESULTS STAGING STAGE_PATH RUN_ID JLENS_ATTEMPT_ID JLENS_HF_TOKEN_FILE JLENS_IMAGE_DIGEST \
  JLENS_LAUNCH_NONCE EXPECTED_STAGE_MANIFEST_SHA256 EXPECTED_STAGE_SOURCE_HEAD \
  JLENS_WORKER_DEADLINE_EPOCH

# Download only the pinned stage archive. Verify it in isolation before
# atomically installing a fresh workspace; the extractor rejects traversal,
# links, special files, destination reuse, and manifest/pin mismatches.
stage_tmp=$(mktemp -d "/tmp/ouro-jlens-stage.${JLENS_ATTEMPT_ID}.XXXXXX")
stage_install_complete=0
cleanup_stage() {
  rm -rf -- "$stage_tmp" 2>/dev/null || true
  if [[ "$stage_install_complete" -ne 1 ]]; then
    cleanup_token_channel
  fi
}
trap cleanup_stage EXIT
if run_stage_download_with_token; then
  :
else
  stage_download_rc=$?
  echo "pinned stage download failed or exceeded the worker deadline" >&2
  exit "$stage_download_rc"
fi
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
stage_install_complete=1
cleanup_stage
trap - EXIT
trap cleanup_token_channel EXIT
cd "$WORK"
assert_writable_tree "$WORK"
export PYTHONPATH="$WORK/src${PYTHONPATH:+:$PYTHONPATH}"
stage_metadata=$("$PY" -m ouro_jlens.publish stage-verify --root "$WORK" --allow-extra)
export STAGE_MANIFEST_SHA256=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "$stage_metadata")
if [[ "$STAGE_MANIFEST_SHA256" != "$EXPECTED_STAGE_MANIFEST_SHA256" ]]; then
  echo "extracted stage manifest does not match the controller-approved digest" >&2
  exit 2
fi
stage_source_head=$("$PY" - "$WORK/PINNED_INPUTS.json" <<'PY'
import json
import re
import sys
from pathlib import Path

try:
    document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    source = document["source"]
    head = source["head"]
except (KeyError, OSError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"malformed PINNED_INPUTS.json source identity: {exc}") from exc
if not isinstance(source, dict) or not isinstance(head, str):
    raise SystemExit("PINNED_INPUTS.json source.head is malformed")
if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head) is None:
    raise SystemExit("PINNED_INPUTS.json source.head is not a lowercase git object ID")
print(head)
PY
)
if [[ "$stage_source_head" != "$EXPECTED_STAGE_SOURCE_HEAD" ]]; then
  echo "extracted stage source.head does not match the controller-approved git object ID" >&2
  exit 2
fi
if ! check_free_space; then
  echo "workspace free-space preflight failed; refusing paid setup" >&2
  exit 2
fi

RUN_ROOT="$WORK/artifacts/jlens/runs/$RUN_ID"
[[ ! -L "$RUN_ROOT" ]] || {
  echo "run output root is a symlink: $RUN_ROOT" >&2
  exit 2
}
assert_writable_tree "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
assert_writable_tree "$RUN_ROOT"

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
    --repo "$RESULTS" --token-file "$JLENS_HF_TOKEN_FILE" \
    --run-id "$RUN_ID" --local-root "$RUN_ROOT" \
    --receipt-root "$RUN_ROOT/receipts" --state "$state" --stage "$stage" \
    "${extra[@]}" --kind heartbeat
}

on_exit() {
  local job_rc=$? status_rc=0 index_rc=0 final_rc
  trap - EXIT
  if [[ "$job_rc" -eq 0 ]]; then
    publish_status succeeded complete 0 || status_rc=$?
  else
    publish_status failed run "$job_rc" || status_rc=$?
  fi
  # Publish a digest-versioned cumulative index on *every* trapped exit,
  # including a failed status publication.  The index is independently
  # discoverable from all receipt generations, so a status/network failure
  # cannot hide earlier shard acknowledgements.
  "$PY" -m ouro_jlens.publish index \
    --repo "$RESULTS" --token-file "$JLENS_HF_TOKEN_FILE" \
    --run-id "$RUN_ID" --local-root "$RUN_ROOT" \
    --receipt-root-path "$RUN_ROOT/receipts" || index_rc=$?
  # A successful computation with an unacknowledged terminal status/index is not a
  # successful run. Failed computations retain their original exit code.
  final_rc=$job_rc
  if [[ "$job_rc" -eq 0 ]]; then
    if [[ "$status_rc" -ne 0 ]]; then
      final_rc=$status_rc
    elif [[ "$index_rc" -ne 0 ]]; then
      final_rc=$index_rc
    fi
  fi
  cleanup_token_channel
  exit "$final_rc"
}
trap on_exit EXIT

if run_with_deadline bash src/ouro_jlens/setup_b300.sh; then
  :
else
  setup_rc=$?
  echo "B300 setup failed or exceeded the worker deadline" >&2
  exit "$setup_rc"
fi
publish_status running setup
if run_with_deadline bash src/ouro_jlens/run_b300.sh "$N" "$DB"; then
  :
else
  run_rc=$?
  echo "B300 run failed or exceeded the worker deadline" >&2
  exit "$run_rc"
fi
