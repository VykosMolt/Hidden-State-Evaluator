#!/usr/bin/env bash
# Full Ouro-2.6B recurrent Jacobian-lens job for one supervised GPU lease.
#
# Every output is published and acknowledged below RUN_ID before the next
# stage starts.  A shard is resumable only after its .pt and .json sidecar
# receipts have both been verified; no file type is excluded.
set -euo pipefail

cd "$(dirname "$0")/../.."
N=${1:-100}
DB=${2:-32}
SHARD=${SHARD_SIZE:-25}
PY=${PYTHON:-python}
RUN_ID=${RUN_ID:?RUN_ID is required}
RESULTS=${RESULTS:?RESULTS repository is required}
if ! [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "RUN_ID is not a safe immutable path component" >&2
  exit 2
fi
if ! [[ "$N" =~ ^[1-9][0-9]*$ && "$DB" =~ ^[1-9][0-9]*$ && "$SHARD" =~ ^[1-9][0-9]*$ ]]; then
  echo "N, DIM_BATCH, and SHARD_SIZE must be positive integers" >&2
  exit 2
fi
PUBLISH_ROOT=${PUBLISH_ROOT:-artifacts/jlens/runs/$RUN_ID}
RECEIPT_ROOT=${RECEIPT_ROOT:-$PUBLISH_ROOT/receipts}
LENS="artifacts/jlens/lens/n$N"
EVAL_ROOT=artifacts/jlens/eval
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
for output_root in "$LENS" "$PUBLISH_ROOT" "$EVAL_ROOT"; do
  [[ ! -L "$output_root" ]] || {
    echo "output root is a symlink: $output_root" >&2
    exit 2
  }
done
mkdir -p "$LENS" artifacts/jlens/logs "$PUBLISH_ROOT" "$RECEIPT_ROOT"

"$PY" -c 'from ouro_jlens.publish import validate_run_id; validate_run_id(__import__("os").environ["RUN_ID"])'

log="artifacts/jlens/logs/run_b300_n${N}_${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1

publish_file() {
  local source=$1 relative=$2 kind=$3
  [[ -f "$source" ]] || { echo "missing artifact: $source" >&2; return 2; }
  "$PY" -m ouro_jlens.publish publish-file \
    --repo "$RESULTS" --run-id "$RUN_ID" --file "$source" --relative "$relative" \
    --kind "$kind" --receipt-root "$RECEIPT_ROOT"
}

publish_status() {
  local state=$1 stage=$2 code=${3:-}
  local -a extra=()
  if [[ -n "$code" ]]; then
    extra+=(--exit-code "$code")
  fi
  "$PY" -m ouro_jlens.publish status \
    --repo "$RESULTS" --run-id "$RUN_ID" --local-root "$PUBLISH_ROOT" \
    --receipt-root "$RECEIPT_ROOT" --state "$state" --stage "$stage" \
    "${extra[@]}" --kind heartbeat
}

restore_artifact() {
  local destination=$1 relative=$2 result
  result=$("$PY" -m ouro_jlens.publish restore-file \
    --repo "$RESULTS" --run-id "$RUN_ID" --file "$destination" --relative "$relative")
  case "$result" in
    RESTORED|ABSENT) return 0 ;;
    *) echo "unexpected restore result for $relative: $result" >&2; return 2 ;;
  esac
}

publish_shard() {
  local ut=$1 start=$2 end=$3
  local out="$LENS/exit${ut}_shard_$(printf '%04d' "$start").pt"
  local rel="lens/n$N/$(basename "$out")"
  local sidecar="${out%.pt}.json" side_rel="lens/n$N/$(basename "$sidecar")"
  publish_file "$out" "$rel" shard
  publish_file "$sidecar" "$side_rel" sidecar
  # A checkpoint left by an interrupted fit is itself evidence.  Publish it
  # and its identity sidecar before cleanup; on restart an unacknowledged
  # checkpoint remains local and fit_lens can resume it safely.
  if [[ -f "$out.ckpt" ]]; then
    [[ -f "$out.ckpt.json" ]] || { echo "checkpoint sidecar missing: $out.ckpt.json" >&2; return 2; }
    publish_file "$out.ckpt" "$rel.ckpt" checkpoint
    publish_file "$out.ckpt.json" "$rel.ckpt.json" checkpoint_sidecar
    rm -f -- "$out.ckpt" "$out.ckpt.json"
  fi
}

publish_interrupted_checkpoints() {
  # fit_lens seals a checkpoint sidecar only after hashing the complete
  # checkpoint.  A fit process can still exit before publish_shard is reached
  # (OOM, signal, or a failed child), so recover every paired, hash-valid
  # checkpoint from the EXIT trap.  Publication errors are reported but never
  # replace the computation's original exit status.
  local checkpoint sidecar rel side_rel recovery_failed=0
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
    if ! publish_file "$checkpoint" "$rel" checkpoint; then
      echo "WARNING: interrupted checkpoint payload publication was not acknowledged: $checkpoint" >&2
      recovery_failed=1
    fi
    if ! publish_file "$sidecar" "$side_rel" checkpoint_sidecar; then
      echo "WARNING: interrupted checkpoint sidecar publication was not acknowledged: $sidecar" >&2
      recovery_failed=1
    fi
  done
  if (( recovery_failed )); then
    echo "WARNING: one or more interrupted checkpoints need recovery" >&2
  fi
  return 0
}

on_exit() {
  local job_rc=$?
  trap - EXIT
  publish_interrupted_checkpoints
  if [[ -f "$log" ]]; then
    if ! publish_file "$log" "logs/$(basename "$log")" run_log; then
      echo "WARNING: run log publication was not acknowledged" >&2
    fi
  fi
  if [[ "$job_rc" -ne 0 ]]; then
    # Preserve the computation's original status even when the terminal
    # heartbeat cannot be published.
    if ! publish_status failed run "$job_rc"; then
      echo "WARNING: failed-run heartbeat publication was not acknowledged" >&2
    fi
  fi
  exit "$job_rc"
}
trap on_exit EXIT

fit_target() {
  local ut=$1 s end out rel sidecar side_rel quarantine
  publish_status running "fit_exit${ut}"
  for ((s = 0; s < N; s += SHARD)); do
    end=$((s + SHARD))
    if (( end > N )); then
      end=$N
    fi
    out="$LENS/exit${ut}_shard_$(printf '%04d' "$s").pt"
    rel="lens/n$N/$(basename "$out")"
    sidecar="${out%.pt}.json"
    side_rel="lens/n$N/$(basename "$sidecar")"
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
    if [[ ! -f "$out" && -f "$sidecar" ]]; then
      # A sidecar without its bound payload is not a resumable completion.
      # Preserve the orphan for audit and recompute the shard from source.
      quarantine="${PUBLISH_ROOT}/quarantine/${ut}_$(basename "$sidecar").orphan.$(date +%s%N).$$"
      mkdir -p "$(dirname "$quarantine")"
      mv -- "$sidecar" "$quarantine"
    fi
    if [[ ! -f "$out" ]]; then
      # A process can die after fit_lens sealed a checkpoint but before it
      # sealed the final lens. Restore both checkpoint files so the generator
      # can resume; a lone checkpoint payload is never trusted.
      restore_artifact "$out.ckpt" "$rel.ckpt"
      restore_artifact "$out.ckpt.json" "$rel.ckpt.json"
      if [[ -f "$out.ckpt" && ! -f "$out.ckpt.json" ]]; then
        echo "checkpoint payload has no acknowledged sidecar: $out.ckpt" >&2
        return 2
      fi
    fi
    if [[ -f "$out" && ! -f "$sidecar" ]]; then
      # A binary without fit_lens's substantive sidecar is not a completed
      # shard. Preserve it for audit and recompute rather than inventing
      # model/configuration metadata from the binary alone.
      quarantine="$PUBLISH_ROOT/quarantine/${ut}_$(basename "$out").orphan.$(date +%s%N).$$"
      mkdir -p "$(dirname "$quarantine")"
      mv -- "$out" "$quarantine"
    fi
    if [[ ! -f "$out" ]]; then
      "$PY" src/ouro_jlens/fit_lens.py fit --target-ut "$ut" --start "$s" --end "$end" \
        --dim-batch "$DB" --checkpoint-every 10 --out "$out"
    else
      "$PY" -c 'from ouro_jlens.fit_lens import validate_sidecar; import sys; validate_sidecar(sys.argv[1], kind="fit")' "$out"
    fi
    # This also handles a crash after fit completion but before upload ACK.
    publish_shard "$ut" "$s" "$end"
  done

  local -a shards=()
  for ((s = 0; s < N; s += SHARD)); do
    shards+=("$LENS/exit${ut}_shard_$(printf '%04d' "$s").pt")
  done
  for out in "${shards[@]}"; do
    [[ -f "$out" ]] || { echo "missing shard: $out" >&2; return 2; }
  done
  local merged="$LENS/exit${ut}.pt"
  local merged_meta="${merged%.pt}.json"
  local merged_rel="lens/n$N/$(basename "$merged")" merged_meta_rel="lens/n$N/$(basename "$merged_meta")"
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
  if [[ ! -f "$merged" && -f "$merged_meta" ]]; then
    quarantine="${PUBLISH_ROOT}/quarantine/${ut}_$(basename "$merged_meta").orphan.$(date +%s%N).$$"
    mkdir -p "$(dirname "$quarantine")"
    mv -- "$merged_meta" "$quarantine"
  fi
  if [[ -f "$merged" && ! -f "$merged_meta" ]]; then
    quarantine="$PUBLISH_ROOT/quarantine/${ut}_$(basename "$merged").orphan.$(date +%s%N).$$"
    mkdir -p "$(dirname "$quarantine")"
    mv -- "$merged" "$quarantine"
  fi
  if [[ ! -f "$merged" || ! -f "$merged_meta" ]]; then
    "$PY" src/ouro_jlens/fit_lens.py merge --out "$merged" "${shards[@]}"
  else
    "$PY" -c 'from ouro_jlens.fit_lens import validate_sidecar; import sys; validate_sidecar(sys.argv[1], kind="merged")' "$merged"
  fi
  publish_file "$merged" "$merged_rel" merged_lens
  publish_file "$merged_meta" "$merged_meta_rel" sidecar
  publish_status running "fit_exit${ut}_complete"
}

publish_status running start
"$PY" src/ouro_jlens/validate.py
if [[ -d artifacts/jlens/validation ]]; then
  "$PY" -m ouro_jlens.publish publish-tree --repo "$RESULTS" --run-id "$RUN_ID" \
    --root artifacts/jlens/validation --prefix validation --kind validation --receipt-root "$RECEIPT_ROOT"
fi
publish_status running validation
"$PY" src/ouro_jlens/bench.py 128 8 16 "$DB" 64
publish_status running benchmark

fit_target 3

run_eval() {
  local tag=$1
  shift
  local out="$EVAL_ROOT/$tag"
  local marker="$PUBLISH_ROOT/markers/${tag}.complete"
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
  if [[ ! -f "$marker" || "$(<"$marker")" != "$RUN_ID:$tag" ||
        ! -f "$out/arrays.npz" || ! -f "$out/items.json" ||
        ! -f "$out/task_names.json" || ! -f "$out/summary.json" ]]; then
    "$PY" src/ouro_jlens/evaluate.py "$@" --out "$out"
    "$PY" src/ouro_jlens/analyze.py --eval "$out"
    # Publish every regular file.  There is no checkpoint/lens/extension
    # exclusion here: arrays, figures, summaries, and metadata all travel.
    "$PY" -m ouro_jlens.publish publish-tree --repo "$RESULTS" --run-id "$RUN_ID" \
      --root "$out" --prefix "eval/$tag" --kind eval_output --receipt-root "$RECEIPT_ROOT"
    printf '%s\n' "${RUN_ID}:${tag}" > "$marker"
    publish_file "$marker" "markers/${tag}.complete" marker
  else
    # A valid marker means the prior process acknowledged the complete tree;
    # re-scan it to verify bytes before proceeding after a restart.
    "$PY" -m ouro_jlens.publish publish-tree --repo "$RESULTS" --run-id "$RUN_ID" \
      --root "$out" --prefix "eval/$tag" --kind eval_output --receipt-root "$RECEIPT_ROOT"
    publish_file "$marker" "markers/${tag}.complete" marker
  fi
  publish_status running "evaluate_${tag}_complete"
}

run_eval "n${N}_exit3" --lens "3=$LENS/exit3.pt"

for ut in 2 1 0; do
  fit_target "$ut"
done

ALL=(--lens "3=$LENS/exit3.pt" --lens "2=$LENS/exit2.pt" --lens "1=$LENS/exit1.pt" --lens "0=$LENS/exit0.pt")
run_eval "n${N}_allexits" "${ALL[@]}"
run_eval "n${N}_allexits_pos-2" "${ALL[@]}" --position -2

publish_status succeeded complete 0
echo "== $(date -Is) done run_id=$RUN_ID"
