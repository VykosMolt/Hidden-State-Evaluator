#!/usr/bin/env bash
# Full Ouro-2.6B recurrent Jacobian-lens job for one supervised GPU lease.
#
# Every output is published and acknowledged below RUN_ID before the next
# stage starts.  A shard is resumable only after its .pt and .json sidecar
# receipts have both been verified; no file type is excluded.
set -euo pipefail

# Only the pathname of the private token file belongs in this worker
# environment.  Never pass either credential variable to Python or a hostile
# child such as pip/git.
unset HF_TOKEN JLENS_HF_TOKEN_BOOTSTRAP

cd "$(dirname "$0")/../.."
N=${1:-100}
DB=${2:-32}
SHARD=${SHARD_SIZE:-25}
PY=${PYTHON:-python}
RUN_ID=${RUN_ID:?RUN_ID is required}
JLENS_ATTEMPT_ID=${JLENS_ATTEMPT_ID:?JLENS_ATTEMPT_ID is required}
RESULTS=${RESULTS:?RESULTS repository is required}
JLENS_IMAGE_DIGEST=${JLENS_IMAGE_DIGEST:?JLENS_IMAGE_DIGEST is required}
JLENS_LAUNCH_NONCE=${JLENS_LAUNCH_NONCE:?JLENS_LAUNCH_NONCE is required}
EXPECTED_STAGE_MANIFEST_SHA256=${EXPECTED_STAGE_MANIFEST_SHA256:?EXPECTED_STAGE_MANIFEST_SHA256 is required}
EXPECTED_STAGE_SOURCE_HEAD=${EXPECTED_STAGE_SOURCE_HEAD:?EXPECTED_STAGE_SOURCE_HEAD is required}
JLENS_WORKER_DEADLINE_EPOCH=${JLENS_WORKER_DEADLINE_EPOCH:?JLENS_WORKER_DEADLINE_EPOCH is required}
JLENS_HF_TOKEN_FILE=${JLENS_HF_TOKEN_FILE:?JLENS_HF_TOKEN_FILE is required}
export JLENS_IMAGE_DIGEST JLENS_LAUNCH_NONCE EXPECTED_STAGE_MANIFEST_SHA256 EXPECTED_STAGE_SOURCE_HEAD \
  JLENS_WORKER_DEADLINE_EPOCH JLENS_HF_TOKEN_FILE
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

if int(sys.argv[1]) <= time.time():
    raise SystemExit("JLENS_WORKER_DEADLINE_EPOCH is already past")
PY
then
  echo "JLENS_WORKER_DEADLINE_EPOCH must be in the future" >&2
  exit 2
fi
# pod_entry has already compared the extracted manifest against this value.
# Bind the status/inventory field to that same controller expectation rather
# than to an independently supplied mutable environment value.
if [[ -n "${STAGE_MANIFEST_SHA256:-}" && "$STAGE_MANIFEST_SHA256" != "$EXPECTED_STAGE_MANIFEST_SHA256" ]]; then
  echo "STAGE_MANIFEST_SHA256 does not match the controller-approved stage manifest" >&2
  exit 2
fi
STAGE_MANIFEST_SHA256="$EXPECTED_STAGE_MANIFEST_SHA256"
export STAGE_MANIFEST_SHA256
if ! [[ "$N" =~ ^[1-9][0-9]*$ && "$DB" =~ ^[1-9][0-9]*$ && "$SHARD" =~ ^[1-9][0-9]*$ ]]; then
  echo "N, DIM_BATCH, and SHARD_SIZE must be positive integers" >&2
  exit 2
fi
if (( N < 80 )); then
  echo "B300 contract requires N >= 80; target-ut=3 must include the [0,8), [8,32), [32,56), and [56,80) prefixes" >&2
  exit 2
fi
PUBLISH_ROOT=${PUBLISH_ROOT:-artifacts/jlens/runs/$RUN_ID}
RECEIPT_ROOT=${RECEIPT_ROOT:-$PUBLISH_ROOT/receipts}
LENS="artifacts/jlens/lens/n$N"
EVAL_ROOT=artifacts/jlens/eval
FINAL_ROOT=artifacts/jlens/final
CONVERGENCE_OUT="$EVAL_ROOT/fitsize_convergence.json"
FIT_SIZE_OUT="$FINAL_ROOT/fit_size.json"
TRANSPORT_OUT="$FINAL_ROOT/transport.json"
ANALYSIS_OUT="$FINAL_ROOT/analysis.json"
CLAIMS_OUT="$FINAL_ROOT/claim_status.json"
VERIFICATION_OUT="$FINAL_ROOT/verification.json"
PREFIX_N8="$LENS/exit3_n8.pt"
PREFIX_N32="$LENS/exit3_n32.pt"
PREFIX_N56="$LENS/exit3_n56.pt"
PREFIX_N80="$LENS/exit3_n80.pt"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$JLENS_HF_TOKEN_FILE" <<'PY'
import sys

from ouro_jlens.publish import read_hf_token_file

read_hf_token_file(sys.argv[1])
PY
assert_writable_tree() {
  "$PY" -c 'from pathlib import Path; import sys; from ouro_jlens.publish import _reject_symlink_ancestors; path = Path(sys.argv[1]); _reject_symlink_ancestors(path / "placeholder"); sys.exit(2 if path.is_symlink() else 0)' "$1"
}
for output_root in "$LENS" "$PUBLISH_ROOT" "$EVAL_ROOT" "$FINAL_ROOT" "artifacts/jlens/logs"; do
  [[ ! -L "$output_root" ]] || {
    echo "output root is a symlink: $output_root" >&2
    exit 2
  }
  assert_writable_tree "$output_root"
done
assert_writable_tree "$RECEIPT_ROOT"
mkdir -p "$LENS" artifacts/jlens/logs "$PUBLISH_ROOT" "$RECEIPT_ROOT" "$FINAL_ROOT"

"$PY" -c 'from ouro_jlens.publish import validate_run_id; validate_run_id(__import__("os").environ["RUN_ID"])'

# A process-substitution tee can still be draining when the EXIT trap hashes
# the log, and tee's failure does not become the job's status.  Create one
# exclusive regular file and write stdout/stderr directly to it.  The EXIT
# trap fsyncs the descriptor before publication, so final bytes are stable.
log=$(mktemp "artifacts/jlens/logs/run_b300_n${N}_${RUN_ID}_XXXXXX.log")
[[ -f "$log" && ! -L "$log" ]] || {
  echo "could not create a regular run log" >&2
  exit 2
}
log_identity=$("$PY" -c 'import os, stat, sys; fd = os.open(sys.argv[1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)); st = os.fstat(fd); os.close(fd); sys.exit(2) if not stat.S_ISREG(st.st_mode) else print("%s:%s" % (st.st_dev, st.st_ino))' "$log")
exec >>"$log" 2>&1

sync_log() {
  "$PY" -c '
import os, stat, sys
fd = os.open(sys.argv[1], os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or "%s:%s" % (st.st_dev, st.st_ino) != sys.argv[2]:
        raise OSError("run log is not a regular file")
    os.fsync(fd)
finally:
    os.close(fd)
' "$log" "$log_identity"
}

publish_file() {
  local source=$1 relative=$2 kind=$3
  [[ -f "$source" ]] || { echo "missing artifact: $source" >&2; return 2; }
  "$PY" -m ouro_jlens.publish publish-file \
    --repo "$RESULTS" --token-file "$JLENS_HF_TOKEN_FILE" \
    --run-id "$RUN_ID" --file "$source" --relative "$relative" \
    --kind "$kind" --receipt-root "$RECEIPT_ROOT"
}

write_immutable_marker() {
  local path=$1 value=$2
  "$PY" -c 'from pathlib import Path; import sys; from ouro_jlens.publish import _write_immutable; _write_immutable(Path(sys.argv[1]), (sys.argv[2] + "\\n").encode())' "$path" "$value"
}

verify_receipt() {
  local source=$1 receipt=$2
  "$PY" -c 'from pathlib import Path; import sys; from ouro_jlens.publish import verify_local_receipt; verify_local_receipt(Path(sys.argv[1]), Path(sys.argv[2]), run_id=sys.argv[3])' "$source" "$receipt" "$RUN_ID"
}

publish_status() {
  local state=$1 stage=$2 code=${3:-}
  local -a extra=()
  if [[ -n "$code" ]]; then
    extra+=(--exit-code "$code")
  fi
  "$PY" -m ouro_jlens.publish status \
    --repo "$RESULTS" --token-file "$JLENS_HF_TOKEN_FILE" \
    --run-id "$RUN_ID" --local-root "$PUBLISH_ROOT" \
    --receipt-root "$RECEIPT_ROOT" --state "$state" --stage "$stage" \
    "${extra[@]}" --kind heartbeat
}

restore_artifact() {
  local destination=$1 relative=$2 result
  result=$("$PY" -m ouro_jlens.publish restore-file \
    --repo "$RESULTS" --token-file "$JLENS_HF_TOKEN_FILE" \
    --run-id "$RUN_ID" --file "$destination" --relative "$relative")
  case "$result" in
    RESTORED|ABSENT) return 0 ;;
    *) echo "unexpected restore result for $relative: $result" >&2; return 2 ;;
  esac
}

# Return the exact prompt intervals owned by one target.  The eventual target
# deliberately keeps the four prefix boundaries needed by the fit-size
# report, then appends only the final [80,N) interval for N>80.  All other
# targets use the ordinary positive SHARD_SIZE partition.
plan_intervals() {
  local target=$1 s end
  if [[ "$target" == 3 ]]; then
    printf '%s %s\n' 0 8
    printf '%s %s\n' 8 32
    printf '%s %s\n' 32 56
    printf '%s %s\n' 56 80
    if (( N > 80 )); then
      printf '%s %s\n' 80 "$N"
    fi
    return 0
  fi
  for ((s = 0; s < N; s += SHARD)); do
    end=$((s + SHARD))
    if (( end > N )); then
      end=$N
    fi
    printf '%s %s\n' "$s" "$end"
  done
}

shard_path() {
  local ut=$1 start=$2 end=$3
  printf '%s/exit%s_shard_%04d_%04d.pt\n' "$LENS" "$ut" "$start" "$end"
}

shard_relative() {
  local path=$1
  printf 'lens/n%s/%s\n' "$N" "$(basename "$path")"
}

validate_lens_pair() {
  local path=$1 kind=$2
  local sidecar="${path%.pt}.json"
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "missing or linked lens binary: $path" >&2
    return 2
  }
  [[ -f "$sidecar" && ! -L "$sidecar" ]] || {
    echo "missing or linked lens sidecar: $sidecar" >&2
    return 2
  }
  "$PY" -c 'from ouro_jlens.fit_lens import validate_sidecar; import sys; validate_sidecar(sys.argv[1], kind=sys.argv[2])' "$path" "$kind"
}

validate_json_status() {
  local path=$1 expected=$2
  "$PY" -c '
import json, sys
from pathlib import Path
path, expected = sys.argv[1:]
try:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid JSON product {path}: {exc}") from exc
if not isinstance(value, dict) or value.get("status") != expected:
    raise SystemExit(f"JSON product {path} has unexpected status")
' "$path" "$expected"
}

publish_verified_pair() {
  local path=$1 kind=$2
  local sidecar="${path%.pt}.json" rel side_rel
  rel=$(shard_relative "$path")
  side_rel="${rel%.pt}.json"
  publish_file "$path" "$rel" "$kind"
  verify_receipt "$path" "$RECEIPT_ROOT/$rel.receipt.json"
  publish_file "$sidecar" "$side_rel" sidecar
  verify_receipt "$sidecar" "$RECEIPT_ROOT/$side_rel.receipt.json"
}

discard_completed_checkpoint_pair() {
  local output=$1 checkpoint sidecar
  checkpoint="${output}.ckpt"
  sidecar="${output}.ckpt.json"
  # A checkpoint is recovery material only until its final lens payload and
  # sidecar have both been acknowledged.  Never publish a completed fit's
  # checkpoint: discard the paired local files after those two receipt checks.
  # EXIT recovery below handles only checkpoint pairs still present when a fit
  # is interrupted, fails, or never reaches this function.
  if [[ -e "$checkpoint" || -L "$checkpoint" || -e "$sidecar" || -L "$sidecar" ]]; then
    [[ -f "$checkpoint" && ! -L "$checkpoint" && -f "$sidecar" && ! -L "$sidecar" ]] || {
      echo "completed shard has an incomplete checkpoint pair: $checkpoint" >&2
      return 2
    }
    rm -f -- "$checkpoint" "$sidecar"
    [[ ! -e "$checkpoint" && ! -L "$checkpoint" &&
       ! -e "$sidecar" && ! -L "$sidecar" ]] || {
      echo "completed checkpoint pair could not be removed: $checkpoint" >&2
      return 2
    }
  fi
}

quarantine_orphan() {
  local source=$1 reason=$2 destination
  [[ -e "$source" && ! -L "$source" ]] || return 0
  destination="$PUBLISH_ROOT/quarantine/$(basename "$source").${reason}.$(date +%s%N).$$"
  mkdir -p -- "$(dirname "$destination")"
  [[ ! -e "$destination" && ! -L "$destination" ]] || {
    echo "quarantine destination already exists: $destination" >&2
    return 2
  }
  mv -- "$source" "$destination"
}

publish_shard() {
  local ut=$1 start=$2 end=$3
  local out
  out=$(shard_path "$ut" "$start" "$end")
  local rel sidecar side_rel
  rel=$(shard_relative "$out")
  sidecar="${out%.pt}.json"
  side_rel="${rel%.pt}.json"
  publish_file "$out" "$rel" shard
  verify_receipt "$out" "$RECEIPT_ROOT/$rel.receipt.json"
  publish_file "$sidecar" "$side_rel" sidecar
  verify_receipt "$sidecar" "$RECEIPT_ROOT/$side_rel.receipt.json"
  discard_completed_checkpoint_pair "$out"
}

verify_eval_receipts() {
  local root=$1 tag=$2
  "$PY" - "$root" "$tag" "$RECEIPT_ROOT" "$RUN_ID" <<'PY'
import sys
from pathlib import Path

from ouro_jlens.publish import verify_local_receipt

root = Path(sys.argv[1])
tag = sys.argv[2]
receipt_root = Path(sys.argv[3])
run_id = sys.argv[4]
files = []
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"evaluation tree contains a symlink: {path}")
    if path.is_file():
        files.append(path)
if not files:
    raise SystemExit(f"expected evaluation output tree is empty: {root}")
for path in files:
    relative = f"eval/{tag}/{path.relative_to(root).as_posix()}"
    receipt = receipt_root / f"{relative}.receipt.json"
    try:
        verify_local_receipt(path, receipt, run_id=run_id)
    except Exception as exc:
        raise SystemExit(f"expected evaluation receipt is missing or invalid: {receipt}: {exc}") from exc
PY
}

publish_interrupted_checkpoints() {
  # fit_lens seals a checkpoint sidecar only after hashing the complete
  # checkpoint.  A fit process can still exit before publish_shard is reached
  # (OOM, signal, or a failed child), so recover every paired, hash-valid
  # checkpoint from the EXIT trap.  Publication errors are reported but never
  # replace the computation's original exit status.
  local checkpoint sidecar rel side_rel recovery_failed=0 recovered=0
  for checkpoint in "$LENS"/*.pt.ckpt; do
    [[ -f "$checkpoint" && ! -L "$checkpoint" ]] || {
      [[ -L "$checkpoint" ]] && {
        echo "WARNING: refusing linked interrupted checkpoint: $checkpoint" >&2
        recovery_failed=1
      }
      continue
    }
    sidecar="$checkpoint.json"
    if [[ ! -f "$sidecar" || -L "$sidecar" ]]; then
      echo "WARNING: unpaired interrupted checkpoint left local: $checkpoint" >&2
      recovery_failed=1
      continue
    fi
    if ! "$PY" -c '
import hashlib, json, re, sys
from pathlib import Path
checkpoint, sidecar = map(Path, sys.argv[1:])
try:
    metadata = json.loads(sidecar.read_text())
    record = metadata.get("checkpoint")
    digest = record.get("sha256") if isinstance(record, dict) else None
    size = record.get("size") if isinstance(record, dict) else None
    if metadata.get("kind") != "fit_checkpoint" or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
        raise ValueError("checkpoint sidecar is not sealed")
    if not isinstance(size, int) or size < 0 or checkpoint.stat().st_size != size:
        raise ValueError("checkpoint size does not match sidecar")
    found = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            found.update(chunk)
    if found.hexdigest() != digest:
        raise ValueError("checkpoint digest does not match sidecar")
except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
    print(f"{sidecar}: {exc}", file=sys.stderr)
    raise SystemExit(1)
' "$checkpoint" "$sidecar"; then
      echo "WARNING: refusing unsealed or hash-invalid checkpoint: $checkpoint" >&2
      recovery_failed=1
      continue
    fi
    rel="lens/n$N/$(basename "$checkpoint")"
    side_rel="lens/n$N/$(basename "$sidecar")"
    if publish_file "$checkpoint" "$rel" checkpoint; then
      if ! verify_receipt "$checkpoint" "$RECEIPT_ROOT/$rel.receipt.json"; then
        echo "WARNING: interrupted checkpoint payload receipt was not acknowledged: $checkpoint" >&2
        recovery_failed=1
      elif ! publish_file "$sidecar" "$side_rel" checkpoint_sidecar; then
        echo "WARNING: interrupted checkpoint sidecar publication was not acknowledged: $sidecar" >&2
        recovery_failed=1
      elif ! verify_receipt "$sidecar" "$RECEIPT_ROOT/$side_rel.receipt.json"; then
        echo "WARNING: interrupted checkpoint sidecar receipt was not acknowledged: $sidecar" >&2
        recovery_failed=1
      else
        recovered=$((recovered + 1))
      fi
    else
      echo "WARNING: interrupted checkpoint payload publication was not acknowledged: $checkpoint" >&2
      recovery_failed=1
    fi
  done
  if (( recovery_failed )); then
    echo "WARNING: one or more interrupted checkpoints need recovery" >&2
  fi
  local recovery_path="$PUBLISH_ROOT/recovery/${JLENS_ATTEMPT_ID}.json"
  if ! "$PY" - "$recovery_path" "$RUN_ID" "$JLENS_ATTEMPT_ID" "$recovered" "$recovery_failed" \
    "$JLENS_LAUNCH_NONCE" "$EXPECTED_STAGE_MANIFEST_SHA256" "$EXPECTED_STAGE_SOURCE_HEAD" \
    "$JLENS_WORKER_DEADLINE_EPOCH" <<'PY'
import sys
from pathlib import Path
from ouro_jlens.publish import _write_immutable, canonical_json

(
    path, run_id, attempt_id, recovered, failed, launch_nonce,
    stage_manifest_sha256, stage_source_head, worker_deadline_epoch,
) = sys.argv[1:]
_write_immutable(Path(path), canonical_json({
    "schema_version": 1,
    "run_id": run_id,
    "attempt_id": attempt_id,
    "launch_nonce": launch_nonce,
    "stage_manifest_sha256": stage_manifest_sha256,
    "stage_source_head": stage_source_head,
    "worker_deadline_epoch": int(worker_deadline_epoch),
    "recovered_checkpoint_pairs": int(recovered),
    "complete": int(failed) == 0,
}))
PY
  then
    echo "WARNING: checkpoint recovery status could not be written" >&2
    recovery_failed=1
  elif ! publish_file "$recovery_path" "recovery/${JLENS_ATTEMPT_ID}.json" recovery_status; then
    echo "WARNING: checkpoint recovery status was not acknowledged" >&2
    recovery_failed=1
  elif ! verify_receipt "$recovery_path" "$RECEIPT_ROOT/recovery/${JLENS_ATTEMPT_ID}.json.receipt.json"; then
    echo "WARNING: checkpoint recovery status receipt is invalid" >&2
    recovery_failed=1
  fi
  (( recovery_failed == 0 ))
}

on_exit() {
  local job_rc=$? log_rc=0 recovery_rc=0
  local exit_log=""
  trap - EXIT
  # The main computation log ends at this cutover.  Recovery, terminal
  # heartbeat, fsync, and payload publication all emit diagnostics, so move
  # both descriptors to a separate log before doing any of them.  Otherwise a
  # publish command's stdout would append bytes after the main log had been
  # acknowledged.  The exit-publication log is intentionally local-only: it
  # is not part of the scientific inventory and publishing it would recurse
  # by writing its own acknowledgement into itself.
  if exit_log=$(mktemp "${log%.log}.exit.XXXXXX.log" 2>/dev/null) &&
     [[ -f "$exit_log" && ! -L "$exit_log" ]] &&
     exec >>"$exit_log" 2>&1; then
    :
  else
    exec >>/dev/null 2>&1
    log_rc=2
  fi
  publish_interrupted_checkpoints || recovery_rc=$?
  if [[ "$job_rc" -ne 0 ]]; then
    # Record the terminal failure after the main-log cutover.  Preserve the
    # computation's original status even when this publication cannot be
    # acknowledged.
    if ! publish_status failed run "$job_rc"; then
      echo "WARNING: failed-run heartbeat publication was not acknowledged" >&2
    fi
  fi
  if ! sync_log; then
    echo "WARNING: run log could not be fsynced before publication" >&2
    log_rc=2
  fi
  if [[ "$log_rc" -eq 0 && -f "$log" ]]; then
    if ! publish_file "$log" "logs/$(basename "$log")" run_log; then
      echo "WARNING: run log publication was not acknowledged" >&2
      log_rc=2
    fi
  fi
  if [[ "$job_rc" -eq 0 && ( "$log_rc" -ne 0 || "$recovery_rc" -ne 0 ) ]]; then
    job_rc=2
  fi
  exit "$job_rc"
}
trap on_exit EXIT

fit_target() {
  local ut=$1 s end out rel sidecar side_rel planned
  local -a shards=()
  publish_status running "fit_exit${ut}"
  planned=$(plan_intervals "$ut")
  while read -r s end; do
    [[ -n "$s" && -n "$end" ]] || { echo "planned shard interval is malformed" >&2; return 2; }
    out=$(shard_path "$ut" "$s" "$end")
    shards+=("$out")
    rel=$(shard_relative "$out")
    sidecar="${out%.pt}.json"
    side_rel="${rel%.pt}.json"
    if [[ -L "$out" || -L "$sidecar" || -L "$out.ckpt" || -L "$out.ckpt.json" ]]; then
      echo "lens shard or checkpoint path is a symlink: $out" >&2
      return 2
    fi
    if [[ ! -f "$out" ]]; then
      restore_artifact "$out" "$rel"
    fi
    if [[ ! -f "$sidecar" ]]; then
      restore_artifact "$sidecar" "$side_rel"
    fi
    if [[ ! -f "$out" && ! -f "$sidecar" ]]; then
      # A process can die after fit_lens sealed a checkpoint but before it
      # sealed the final lens. Restore both checkpoint files so the generator
      # can resume; a lone checkpoint payload is never trusted.
      restore_artifact "$out.ckpt" "$rel.ckpt"
      restore_artifact "$out.ckpt.json" "$rel.ckpt.json"
    fi
    if [[ -f "$out.ckpt" && ! -f "$out.ckpt.json" ]]; then
      echo "checkpoint payload has no sidecar: $out.ckpt" >&2
      return 2
    fi
    if [[ -f "$out.ckpt.json" && ! -f "$out.ckpt" ]]; then
      echo "checkpoint sidecar has no payload: $out.ckpt.json" >&2
      return 2
    fi
    # Any unpaired artifact is an invalid run state.  Do not silently
    # quarantine and recompute: that would turn missing provenance into a
    # successful paid result.
    if [[ -e "$out" || -L "$out" || -e "$sidecar" || -L "$sidecar" ]]; then
      [[ -f "$out" && ! -L "$out" && -f "$sidecar" && ! -L "$sidecar" ]] || {
        echo "shard payload/sidecar pair is incomplete: $out" >&2
        return 2
      }
      validate_lens_pair "$out" fit
    fi
    if [[ -e "$out.ckpt" || -L "$out.ckpt" || -e "$out.ckpt.json" || -L "$out.ckpt.json" ]]; then
      [[ -f "$out.ckpt" && ! -L "$out.ckpt" && -f "$out.ckpt.json" && ! -L "$out.ckpt.json" ]] || {
        echo "checkpoint payload/sidecar pair is incomplete: $out.ckpt" >&2
        return 2
      }
    fi
    if [[ ! -f "$out" ]]; then
      "$PY" src/ouro_jlens/fit_lens.py fit --target-ut "$ut" --start "$s" --end "$end" \
        --dim-batch "$DB" --checkpoint-every 10 --out "$out"
      validate_lens_pair "$out" fit
    fi
    # This also handles a crash after fit completion but before upload ACK.
    publish_shard "$ut" "$s" "$end"
  done <<< "$planned"

  for out in "${shards[@]}"; do
    validate_lens_pair "$out" fit
  done
  local merged="$LENS/exit${ut}.pt"
  local merged_meta="${merged%.pt}.json"
  local merged_rel merged_meta_rel
  merged_rel=$(shard_relative "$merged")
  merged_meta_rel="${merged_rel%.pt}.json"
  if [[ -L "$merged" || -L "$merged_meta" ]]; then
    echo "merged lens path is a symlink: $merged" >&2
    return 2
  fi
  if [[ ! -f "$merged" ]]; then
    restore_artifact "$merged" "$merged_rel"
  fi
  if [[ ! -f "$merged_meta" ]]; then
    restore_artifact "$merged_meta" "$merged_meta_rel"
  fi
  if [[ -e "$merged" || -L "$merged" || -e "$merged_meta" || -L "$merged_meta" ]]; then
    [[ -f "$merged" && ! -L "$merged" && -f "$merged_meta" && ! -L "$merged_meta" ]] || {
      echo "merged lens payload/sidecar pair is incomplete: $merged" >&2
      return 2
    }
  else
    "$PY" src/ouro_jlens/fit_lens.py merge --out "$merged" "${shards[@]}"
  fi
  validate_lens_pair "$merged" merged
  "$PY" - "${merged}" "${shards[@]}" <<'PY'
import sys
from ouro_jlens.fit_lens import validate_merged_shards
validate_merged_shards(sys.argv[1], sys.argv[2:])
PY
  publish_file "$merged" "$merged_rel" merged_lens
  verify_receipt "$merged" "$RECEIPT_ROOT/$merged_rel.receipt.json"
  publish_file "$merged_meta" "$merged_meta_rel" sidecar
  verify_receipt "$merged_meta" "$RECEIPT_ROOT/$merged_meta_rel.receipt.json"
  publish_status running "fit_exit${ut}_complete"
}

merge_prefix() {
  local n=$1 out=$2
  shift 2
  local sidecar="${out%.pt}.json" rel side_rel
  rel=$(shard_relative "$out")
  side_rel="${rel%.pt}.json"
  if [[ -L "$out" || -L "$sidecar" ]]; then
    echo "prefix lens path is a symlink: $out" >&2
    return 2
  fi
  if [[ ! -f "$out" ]]; then
    restore_artifact "$out" "$rel"
  fi
  if [[ ! -f "$sidecar" ]]; then
    restore_artifact "$sidecar" "$side_rel"
  fi
  if [[ -e "$out" || -e "$sidecar" ]]; then
    [[ -f "$out" && -f "$sidecar" ]] || {
      echo "prefix lens payload/sidecar pair is incomplete: $out" >&2
      return 2
    }
  else
    "$PY" src/ouro_jlens/fit_lens.py merge --out "$out" "$@"
  fi
  validate_lens_pair "$out" merged
  "$PY" - "$out" "$@" <<'PY'
import sys
from ouro_jlens.fit_lens import validate_merged_shards
validate_merged_shards(sys.argv[1], sys.argv[2:])
PY
  publish_verified_pair "$out" prefix_lens
  publish_status running "prefix_n${n}_complete"
}

build_prefix_lenses() {
  local p0 p8 p32 p56 prefix_shard
  p0=$(shard_path 3 0 8)
  p8=$(shard_path 3 8 32)
  p32=$(shard_path 3 32 56)
  p56=$(shard_path 3 56 80)
  for prefix_shard in "$p0" "$p8" "$p32" "$p56"; do
    validate_lens_pair "$prefix_shard" fit
  done
  merge_prefix 8 "$PREFIX_N8" "$p0"
  merge_prefix 32 "$PREFIX_N32" "$p0" "$p8"
  merge_prefix 56 "$PREFIX_N56" "$p0" "$p8" "$p32"
  merge_prefix 80 "$PREFIX_N80" "$p0" "$p8" "$p32" "$p56"
}

publish_status running start
"$PY" src/ouro_jlens/validate.py
if [[ -d artifacts/jlens/validation ]]; then
  "$PY" -m ouro_jlens.publish publish-tree --repo "$RESULTS" --token-file "$JLENS_HF_TOKEN_FILE" \
    --run-id "$RUN_ID" \
    --root artifacts/jlens/validation --prefix validation --kind validation --receipt-root "$RECEIPT_ROOT"
fi
publish_status running validation
"$PY" src/ouro_jlens/bench.py 128 8 16 "$DB" 64
publish_status running benchmark

fit_target 3
build_prefix_lenses

validate_eval() {
  local out=$1 policy=$2
  shift 2
  "$PY" - "$out" "$policy" "$@" <<'PY'
import sys
from pathlib import Path
from ouro_jlens.evaluate import validate_evaluation_provenance

output, policy = sys.argv[1:3]
specs = []
for raw in sys.argv[3:]:
    target, path = raw.split("=", 1)
    specs.append((int(target), Path(path)))
validate_evaluation_provenance(
    output,
    lens_paths=specs,
    expected_prompt_policy=policy,
)
PY
}

run_eval_impl() {
  local tag=$1 position=$2
  shift 2
  local out="$EVAL_ROOT/$tag"
  local marker="$PUBLISH_ROOT/markers/${tag}.complete"
  local -a specs=("$@") eval_args=()
  local spec
  for spec in "${specs[@]}"; do
    eval_args+=(--lens "$spec")
  done
  if [[ "$position" != -1 ]]; then
    eval_args+=(--position "$position")
  fi
  [[ ! -L "$out" ]] || {
    echo "evaluation output root is a symlink: $out" >&2
    return 2
  }
  [[ ! -L "$marker" ]] || {
    echo "evaluation completion marker is a symlink: $marker" >&2
    return 2
  }
  if [[ -d "$out" ]] && find "$out" -type l -print -quit | grep -q .; then
    echo "evaluation tree contains a symlink: $out" >&2
    return 2
  fi
  mkdir -p "$out" "$(dirname "$marker")"
  publish_status running "evaluate_${tag}"
  if [[ -f "$marker" ]]; then
    [[ "$(<"$marker")" == "$RUN_ID:$tag" ]] || {
      echo "evaluation marker belongs to another or incomplete run: $marker" >&2
      return 2
    }
    validate_eval "$out" identical "${specs[@]}"
    [[ -f "$out/summary.json" && -f "$out/summary.md" ]] || {
      echo "completed evaluation is missing analysis products: $out" >&2
      return 2
    }
  else
    "$PY" src/ouro_jlens/evaluate.py "${eval_args[@]}" --out "$out"
    validate_eval "$out" identical "${specs[@]}"
    "$PY" src/ouro_jlens/analyze.py --eval "$out"
    validate_eval "$out" identical "${specs[@]}"
    [[ -f "$out/summary.json" && -f "$out/summary.md" ]] || {
      echo "analysis did not leave complete summary products: $out" >&2
      return 2
    }
  fi
  # Publish every regular file.  There is no checkpoint/lens/extension
  # exclusion here: arrays, figures, summaries, and provenance all travel.
  "$PY" -m ouro_jlens.publish publish-tree --repo "$RESULTS" --token-file "$JLENS_HF_TOKEN_FILE" \
    --run-id "$RUN_ID" \
    --root "$out" --prefix "eval/$tag" --kind eval_output --receipt-root "$RECEIPT_ROOT"
  verify_eval_receipts "$out" "$tag"
  if [[ ! -f "$marker" ]]; then
    write_immutable_marker "$marker" "${RUN_ID}:${tag}"
  fi
  publish_file "$marker" "markers/${tag}.complete" marker
  verify_receipt "$marker" "$RECEIPT_ROOT/markers/${tag}.complete.receipt.json"
  publish_status running "evaluate_${tag}_complete"
}

run_eval() {
  local tag=$1
  shift
  run_eval_impl "$tag" -1 "$@"
}

run_eval_position() {
  local tag=$1 position=$2
  shift 2
  run_eval_impl "$tag" "$position" "$@"
}

run_eval "n${N}_exit3" "3=$LENS/exit3.pt"

# The four nested prefix evaluations and reports are built immediately after
# target-ut=3. They are independent of the later local-exit fits and make the
# fit-size products available even if a later target fails.
run_eval "fitsize_n8" "3=$PREFIX_N8"
run_eval "fitsize_n32" "3=$PREFIX_N32"
run_eval "fitsize_n56" "3=$PREFIX_N56"
run_eval "fitsize_n80" "3=$PREFIX_N80"

generate_fit_reports() {
  publish_status running fit_size_reports
  "$PY" src/ouro_jlens/lens_convergence.py \
    "$(shard_path 3 56 80)" "$PREFIX_N56" --out "$CONVERGENCE_OUT"
  [[ -f "$CONVERGENCE_OUT" && ! -L "$CONVERGENCE_OUT" ]] || {
    echo "fit-size convergence report is missing: $CONVERGENCE_OUT" >&2
    return 2
  }
  validate_json_status "$CONVERGENCE_OUT" INCONCLUSIVE_TWO_POINT_DIAGNOSTIC
  "$PY" src/ouro_jlens/fitsize_report.py --root "$EVAL_ROOT" \
    --out "$FIT_SIZE_OUT" --markdown "$EVAL_ROOT/fitsize_summary.md" --sizes 8 32 56 80
  [[ -f "$FIT_SIZE_OUT" && ! -L "$FIT_SIZE_OUT" &&
     -f "$FINAL_ROOT/fit_size.status.json" && ! -L "$FINAL_ROOT/fit_size.status.json" ]] || {
    echo "fit-size report products are incomplete" >&2
    return 2
  }
  validate_json_status "$FIT_SIZE_OUT" LARGE_FIT_INCONCLUSIVE
  validate_json_status "$FINAL_ROOT/fit_size.status.json" COMPLETE_CURRENT_PROVENANCE
  "$PY" src/ouro_jlens/transport_report.py \
    --lens "8=$PREFIX_N8" --lens "32=$PREFIX_N32" \
    --lens "56=$PREFIX_N56" --lens "80=$PREFIX_N80" --out "$TRANSPORT_OUT"
  [[ -f "$TRANSPORT_OUT" && ! -L "$TRANSPORT_OUT" ]] || {
    echo "transport report is missing: $TRANSPORT_OUT" >&2
    return 2
  }
  validate_json_status "$TRANSPORT_OUT" MODELED_ASSOCIATION_ONLY
  for report in "$CONVERGENCE_OUT" "$EVAL_ROOT/fitsize_summary.md" "$FIT_SIZE_OUT" \
                "$FINAL_ROOT/fit_size.status.json" "$TRANSPORT_OUT"; do
    [[ -f "$report" && ! -L "$report" ]] || {
      echo "report product is missing or linked: $report" >&2
      return 2
    }
    local rel="${report#artifacts/jlens/}"
    publish_file "$report" "$rel" report
    verify_receipt "$report" "$RECEIPT_ROOT/$rel.receipt.json"
  done
}

generate_fit_reports

for ut in 2 1 0; do
  fit_target "$ut"
done

ALL_SPECS=("3=$LENS/exit3.pt" "2=$LENS/exit2.pt" "1=$LENS/exit1.pt" "0=$LENS/exit0.pt")
run_eval "n${N}_allexits" "${ALL_SPECS[@]}"
run_eval_position "n${N}_allexits_pos-2" -2 "${ALL_SPECS[@]}"

generate_canonical_report() {
  publish_status running canonical_report
  "$PY" src/ouro_jlens/report.py \
    --eval "$EVAL_ROOT/fitsize_n80" \
    --local-eval "$EVAL_ROOT/n${N}_allexits" \
    --validation artifacts/jlens/validation/milestones.json \
    --fit-size "$FIT_SIZE_OUT" --transport "$TRANSPORT_OUT" \
    --checkpoints '' \
    --out "$ANALYSIS_OUT" --claims-out "$CLAIMS_OUT"
  for report in "$ANALYSIS_OUT" "$CLAIMS_OUT"; do
    [[ -f "$report" && ! -L "$report" ]] || {
      echo "canonical report product is missing or linked: $report" >&2
      return 2
    }
    local rel="${report#artifacts/jlens/}"
    publish_file "$report" "$rel" report
    verify_receipt "$report" "$RECEIPT_ROOT/$rel.receipt.json"
  done
}

generate_canonical_report

# The paid verifier is deliberately separate from the local arithmetic
# verifier.  It checks current B300 validation, every evaluation/lens lineage,
# exact canonical regeneration, and direct transport recomputation before the
# terminal success state can be published.
publish_status running paid_verification
"$PY" -m ouro_jlens.verify_paid \
  --main "$EVAL_ROOT/fitsize_n80" \
  --local-eval "$EVAL_ROOT/n${N}_allexits" \
  --eval-root "$EVAL_ROOT" \
  --lens-root "$LENS" \
  --validation artifacts/jlens/validation/milestones.json \
  --fit-size "$FIT_SIZE_OUT" \
  --transport "$TRANSPORT_OUT" \
  --analysis "$ANALYSIS_OUT" \
  --claims "$CLAIMS_OUT" \
  --checkpoints '' \
  --out "$VERIFICATION_OUT"
[[ -f "$VERIFICATION_OUT" && ! -L "$VERIFICATION_OUT" ]] || {
  echo "paid verification output is missing or linked: $VERIFICATION_OUT" >&2
  exit 2
}
"$PY" - "$VERIFICATION_OUT" "$ANALYSIS_OUT" <<'PY'
import json
import os
import sys
from pathlib import Path

verification, analysis = (json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:])
if not isinstance(verification, dict) or verification.get("status") != "PASS" or verification.get("accepted") is not True:
    raise SystemExit("paid verification did not produce exact PASS/accepted status")
if verification.get("run_id") != os.environ.get("RUN_ID"):
    raise SystemExit("paid verification artifact lineage does not match RUN_ID")
if verification.get("image_digest") != os.environ.get("JLENS_IMAGE_DIGEST") or verification.get("controller_approved_image_digest") != os.environ.get("JLENS_IMAGE_DIGEST"):
    raise SystemExit("paid verification image identity is not bound to the controller environment")
if verification.get("validation_status") != "SUPPORTED_CURRENT_B300_VALIDATION" or verification.get("b300_validation_status") != "SUPPORTED_CURRENT_B300_VALIDATION":
    raise SystemExit("paid verification validation/B300 status is not current")
if verification.get("canonical_report_status") != "CURRENT_GENERATION_EVIDENCE_VERIFIED":
    raise SystemExit("paid verification canonical lineage is not current")
claims = analysis.get("claims") if isinstance(analysis, dict) else None
if not isinstance(claims, dict) or claims.get("instrumentation", {}).get("status") != "SUPPORTED_CURRENT_VALIDATION" or claims.get("b300_validation", {}).get("status") != "SUPPORTED_CURRENT_B300_VALIDATION":
    raise SystemExit("canonical report validation/B300 claims are not current")
PY
publish_file "$VERIFICATION_OUT" "final/verification.json" verification
verify_receipt "$VERIFICATION_OUT" "$RECEIPT_ROOT/final/verification.json.receipt.json"

verify_expected_inventory() {
  "$PY" - "$PUBLISH_ROOT" "$RECEIPT_ROOT" "$RUN_ID" "$N" "$SHARD" \
    "$LENS" "$EVAL_ROOT" "$JLENS_ATTEMPT_ID" "$JLENS_LAUNCH_NONCE" \
    "$EXPECTED_STAGE_MANIFEST_SHA256" "$EXPECTED_STAGE_SOURCE_HEAD" \
    "$JLENS_WORKER_DEADLINE_EPOCH" <<'PY'
import json
import os
import sys
from pathlib import Path

from ouro_jlens.evaluate import validate_evaluation_provenance
from ouro_jlens.fit_lens import validate_merged_shards, validate_sidecar
from ouro_jlens.publish import _write_immutable, canonical_json, sha256_file, verify_local_receipt

(
    publish_root, receipt_root, run_id, raw_n, raw_shard, lens_root, eval_root,
    attempt_id, launch_nonce, stage_manifest_sha256, stage_source_head,
    worker_deadline_epoch,
) = sys.argv[1:]
n = int(raw_n)
shard_size = int(raw_shard)
lens_root = Path(lens_root)
eval_root = Path(eval_root)
publish_root = Path(publish_root)
receipt_root = Path(receipt_root)


def fail(message):
    raise SystemExit(message)


def interval_plan(target):
    if target == 3:
        result = [(0, 8), (8, 32), (32, 56), (56, 80)]
        if n > 80:
            result.append((80, n))
        return result
    return [(start, min(start + shard_size, n)) for start in range(0, n, shard_size)]


def ensure_regular(path, label):
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is missing or linked: {path}")


def record(path, relative, label, kind):
    ensure_regular(path, label)
    receipt = receipt_root / f"{relative}.receipt.json"
    ensure_regular(receipt, f"{label} receipt")
    try:
        verify_local_receipt(path, receipt, run_id=run_id)
    except Exception as exc:
        fail(f"{label} receipt is invalid: {receipt}: {exc}")
    return {
        "path": relative,
        "receipt": f"{relative}.receipt.json",
        "kind": kind,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def lens_pair(path, *, kind, relative_kind):
    sidecar = path.with_suffix(".json")
    relative = f"lens/n{n}/{path.name}"
    side_relative = f"lens/n{n}/{sidecar.name}"
    try:
        validate_sidecar(path, kind=kind)
    except Exception as exc:
        fail(f"{relative} sidecar validation failed: {exc}")
    return {
        "payload": record(path, relative, f"{relative} payload", relative_kind),
        "sidecar": record(sidecar, side_relative, f"{side_relative} sidecar", "sidecar"),
    }


def eval_files(tag, root):
    ensure_regular(root / "arrays.npz", f"evaluation {tag} arrays")
    ensure_regular(root / "items.json", f"evaluation {tag} items")
    ensure_regular(root / "task_names.json", f"evaluation {tag} task names")
    ensure_regular(root / "provenance.json", f"evaluation {tag} provenance")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            fail(f"evaluation {tag} contains a symlink: {path}")
        if path.is_file():
            relative = f"eval/{tag}/{path.relative_to(root).as_posix()}"
            files.append(record(path, relative, f"evaluation {tag} file", "eval_output"))
    if not files:
        fail(f"evaluation {tag} output tree is empty")
    return files


def validate_eval(tag, specs):
    root = eval_root / tag
    try:
        validate_evaluation_provenance(
            root,
            lens_paths=[(target, path) for target, path in specs],
            expected_prompt_policy="identical",
        )
    except Exception as exc:
        fail(f"evaluation {tag} provenance is invalid: {exc}")
    return eval_files(tag, root)


shard_intervals = {str(target): interval_plan(target) for target in range(4)}
lenses = {}
variable_shards = []
shard_paths = {}
for target in range(4):
    entries = []
    for start, end in interval_plan(target):
        path = lens_root / f"exit{target}_shard_{start:04d}_{end:04d}.pt"
        pair = lens_pair(path, kind="fit", relative_kind="shard")
        pair.update({"target_ut": target, "start": start, "end": end})
        entries.append(pair)
        shard_paths.setdefault(target, []).append(path)
        if target == 3:
            variable_shards.append(pair)
    merged = lens_root / f"exit{target}.pt"
    merged_pair = lens_pair(merged, kind="merged", relative_kind="merged_lens")
    try:
        validate_merged_shards(merged, shard_paths[target])
    except Exception as exc:
        fail(f"merged lens target {target} is not sealed to current shards: {exc}")
    lenses[str(target)] = {
        "intervals": interval_plan(target),
        "shards": entries,
        "merged": merged_pair,
    }

prefix_specs = [
    (8, lens_root / "exit3_n8.pt", shard_paths[3][:1]),
    (32, lens_root / "exit3_n32.pt", shard_paths[3][:2]),
    (56, lens_root / "exit3_n56.pt", shard_paths[3][:3]),
    (80, lens_root / "exit3_n80.pt", shard_paths[3][:4]),
]
prefix_lenses = []
for size, path, sources in prefix_specs:
    pair = lens_pair(path, kind="merged", relative_kind="prefix_lens")
    try:
        validate_merged_shards(path, sources)
    except Exception as exc:
        fail(f"prefix lens n={size} is not sealed to exact shards: {exc}")
    pair.update({"n_prompts": size, "sources": [p.name for p in sources]})
    prefix_lenses.append(pair)

main_specs = {
    f"n{n}_exit3": [(3, lens_root / "exit3.pt")],
    f"n{n}_allexits": [
        (3, lens_root / "exit3.pt"), (2, lens_root / "exit2.pt"),
        (1, lens_root / "exit1.pt"), (0, lens_root / "exit0.pt"),
    ],
    f"n{n}_allexits_pos-2": [
        (3, lens_root / "exit3.pt"), (2, lens_root / "exit2.pt"),
        (1, lens_root / "exit1.pt"), (0, lens_root / "exit0.pt"),
    ],
}
main_evaluations = [
    {"tag": tag, "files": validate_eval(tag, specs)}
    for tag, specs in main_specs.items()
]
fit_evaluations = []
for size, path, _sources in prefix_specs:
    tag = f"fitsize_n{size}"
    fit_evaluations.append({"tag": tag, "n_prompts": size,
                            "files": validate_eval(tag, [(3, path)])})

report_paths = [
    (Path("artifacts/jlens/eval/fitsize_convergence.json"), "report"),
    (Path("artifacts/jlens/eval/fitsize_summary.md"), "report"),
    (Path("artifacts/jlens/final/fit_size.json"), "report"),
    (Path("artifacts/jlens/final/fit_size.status.json"), "report"),
    (Path("artifacts/jlens/final/transport.json"), "report"),
    (Path("artifacts/jlens/final/analysis.json"), "report"),
    (Path("artifacts/jlens/final/claim_status.json"), "report"),
    (Path("artifacts/jlens/final/verification.json"), "verification"),
]
report_products = []
for path, kind in report_paths:
    relative = path.relative_to(Path("artifacts/jlens")).as_posix()
    report_products.append(record(path, relative, "report product", kind))

verification_payload = {}
try:
    verification_payload = json.loads(
        (Path("artifacts/jlens/final/verification.json")).read_text(encoding="utf-8")
    )
except (OSError, json.JSONDecodeError) as exc:
    fail(f"paid verification output is unavailable or invalid: {exc}")
if (
    not isinstance(verification_payload, dict)
    or verification_payload.get("status") != "PASS"
    or verification_payload.get("accepted") is not True
    or verification_payload.get("run_id") != run_id
    or verification_payload.get("image_digest") != os.environ.get("JLENS_IMAGE_DIGEST")
    or verification_payload.get("controller_approved_image_digest") != os.environ.get("JLENS_IMAGE_DIGEST")
    or verification_payload.get("validation_status") != "SUPPORTED_CURRENT_B300_VALIDATION"
    or verification_payload.get("b300_validation_status") != "SUPPORTED_CURRENT_B300_VALIDATION"
    or verification_payload.get("canonical_report_status") != "CURRENT_GENERATION_EVIDENCE_VERIFIED"
):
    fail("paid verification status is not exact PASS/current B300 lineage")

validation_files = []
validation_root = Path("artifacts/jlens/validation")
for path in sorted(validation_root.rglob("*")):
    if path.is_symlink():
        fail(f"validation tree contains a symlink: {path}")
    if path.is_file():
        relative = f"validation/{path.relative_to(validation_root).as_posix()}"
        validation_files.append(record(path, relative, "validation product", "validation"))
if not validation_files:
    fail("validation product tree is empty")

inventory = {
    "schema_version": 2,
    "run_id": run_id,
    "attempt_id": attempt_id,
    "launch_nonce": launch_nonce,
    "stage_manifest_sha256": stage_manifest_sha256,
    "stage_source_head": stage_source_head,
    "worker_deadline_epoch": int(worker_deadline_epoch),
    "n_prompts": n,
    "shard_size": shard_size,
    "shard_intervals": shard_intervals,
    "variable_shards": variable_shards,
    "lens": lenses,
    "prefix_lenses": prefix_lenses,
    "main_evaluations": main_evaluations,
    "fit_size_evaluations": fit_evaluations,
    "report_products": report_products,
    "validation_products": validation_files,
    "evaluation_tags": list(main_specs),
}
inventory_path = publish_root / "expected-inventory.json"
_write_immutable(inventory_path, canonical_json(inventory))
PY
  publish_file "$PUBLISH_ROOT/expected-inventory.json" "inventory/expected-inventory.json" expected_inventory
  verify_receipt "$PUBLISH_ROOT/expected-inventory.json" "$RECEIPT_ROOT/inventory/expected-inventory.json.receipt.json"
}

verify_expected_inventory
publish_status succeeded complete 0
echo "== $(date -Is) done run_id=$RUN_ID"
