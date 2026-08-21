#!/usr/bin/env bash
# Pod-side launch entrypoint for the accelerator session (B300 primary /
# B200 explicit fallback, INTERRUPTIBLE capacity).
#
# DOES NOTHING SCIENTIFIC BY ITSELF.  Order matters and is load-bearing:
#
#   1. FETCH the staged artifacts.  The ~5 GB Ouro-RLTT checkpoint is
#      deliberately never baked into the image and never in Git, so
#      without this step there is nothing to verify and nothing to load.
#   2. VERIFY every artifact against POD_TRANSFER_MANIFEST.json — the
#      manifest whose paths are the CONTAINER's.  (deploy/
#      TRANSFER_MANIFEST.json records build-host paths for provenance and
#      would fail every entry here.)
#   3. VALIDATE the environment.
#   4. Hand control to the zero-touch state machine, which enforces every
#      remaining gate — hardware identity + representative workloads,
#      non-O1 equivalence, the frozen staged benchmark, frozen backend
#      selection, replacement precommit, external commitment verification
#      and the affordability gate — before the sealed v2.1 calibration
#      orchestrator runs a single row.
#
# Committed rows stream continuously to the durable result destination, so
# eviction of the interruptible pod is recoverable at the next-missing-row
# boundary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${O1_B200_PYTHON:-/opt/venv/bin/python}"
OUT="${O1_B200_OUT:-/outputs}"
ARTIFACTS="${O1_B200_ARTIFACTS_ROOT:-/artifacts}"

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
# Model loading must never reach the network.  Hub access for artifact
# ingestion, durability and result transfer happens ONLY inside
# runner/hf_transfer.py, which is spawned with these flags stripped.
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="$ROOT"

mkdir -p "$OUT" "$ARTIFACTS"

# Every pre-entry step below fails DETERMINISTICALLY (same image, same
# config, same artifacts -> same refusal), so a non-zero exit must carry the
# marker the off-pod driver greps for.  Without it the driver reads the
# EXITED pod as an eviction and reacquires; the zero-progress guard cannot
# help because no durable row count exists before the first calibration
# checkpoint.  Result: four paid acquisitions of a run that refuses each
# time.  (check_hf_scope distinguishes TRANSIENT hub failures itself and
# deliberately omits the marker for those; this wrapper only fires when a
# step exited without having printed one.)
step() {
  local name="$1"; shift
  local rc=0
  local log="$OUT/.pre_entry_${name}.log"
  "$@" 2>&1 | tee "$log" || rc=${PIPESTATUS[0]}
  if [[ "$rc" -ne 0 ]]; then
    if ! grep -q "ZERO_TOUCH_ABORTED_AT_" "$log" 2>/dev/null \
        && ! grep -q "REFUSED (transient)" "$log" 2>/dev/null; then
      echo "ZERO_TOUCH_ABORTED_AT_PRE_ENTRY_${name}"
    fi
    exit "$rc"
  fi
}

# Credential scope FIRST: one HF_TOKEN must cover both the staged-artifact
# read and the durable-result write.  A read-only token passes ingestion and
# every gate, then loses every durability push — under interruptible capacity
# the run looks healthy until the eviction that destroys it.  Seconds here.
echo "[start_b300] credential scope preflight (read source + write destination)"
step HF_SCOPE "$PY" -m o1_b200.runner.check_hf_scope --out "$OUT/HF_SCOPE_REPORT.json"

echo "[start_b300] artifact ingestion (checkpoint is not baked into the image)"
step ARTIFACT_FETCH "$PY" -m o1_b200.runner.fetch_artifacts \
  --artifacts-root "$ARTIFACTS" \
  --out "$OUT/ARTIFACT_FETCH_REPORT.json"

echo "[start_b300] artifact verification (container-path manifest)"
step ARTIFACT_VERIFY "$PY" "$ROOT/o1_b200/deploy/verify_artifacts.py" \
  --manifest "${O1_B200_TRANSFER_MANIFEST:-$ROOT/o1_b200/deploy/POD_TRANSFER_MANIFEST.json}"

echo "[start_b300] environment validation"
step ENV_VALIDATE "$PY" "$ROOT/o1_b200/deploy/validate_environment.py" --out "$OUT/environment_report.local.json"

# Combined O1 -> Foundation Learner session: the FL supervisor owns the
# session and runs O1 as an opaque configured subprocess (INTEGRATION.md
# section 3).  It is selected only when the FL package is actually present
# and a session config names it, so an O1-only rental is unaffected.
FL_ENTRY="${O1_FL_ENTRY:-/opt/foundation_learner/foundation_learner/deploy/fl_b200_entry.sh}"
FL_CONFIG="${O1_FL_SESSION_CONFIG:-}"

# RECURSION GUARD.  In a combined session the FL supervisor runs THIS script
# again as the O1 phase (session config field o1_entry_command), inheriting
# the pod environment — O1_FL_SESSION_CONFIG included.  Without this guard
# the inner invocation would dispatch to the FL entry a second time, which
# would start another supervisor, which would run this script again:
# unbounded recursion on a paid accelerator.  The marker is exported below,
# so any descendant of the first entry runs the O1 phase only.  The FL
# supervisor strips the config from the child environment as well; either
# alone is sufficient.
if [[ -n "${O1_B300_ENTRY_ACTIVE:-}" && -n "$FL_CONFIG" ]]; then
  echo "[start_b300] already inside a combined session (depth marker set):" \
       "running the O1 phase, not re-dispatching to FL"
  FL_CONFIG=""
fi
export O1_B300_ENTRY_ACTIVE=1
if [[ -n "$FL_CONFIG" ]]; then
  if [[ ! -f "$FL_CONFIG" ]]; then
    echo "[start_b300] REFUSED: O1_FL_SESSION_CONFIG names $FL_CONFIG," >&2
    echo "            which does not exist on this pod.  A combined session" >&2
    echo "            was requested and cannot be honoured; refusing now" >&2
    echo "            rather than after the accelerator is paid for." >&2
    exit 78
  fi
  if [[ ! -x "$FL_ENTRY" ]]; then
    echo "[start_b300] REFUSED: O1_FL_SESSION_CONFIG is set but the FL entry" >&2
    echo "            $FL_ENTRY is absent — a combined session was requested" >&2
    echo "            and cannot be honoured; refusing rather than silently" >&2
    echo "            running O1 only." >&2
    exit 78
  fi
  echo "[start_b300] combined session: handing over to the FL supervisor"
  exec "$FL_ENTRY" --config "$FL_CONFIG" --out "${FL_B200_OUT:-/workspace/foundation_learner/session}"
fi

echo "[start_b300] O1-only session: handing over to the zero-touch state machine"
exec "$PY" -m o1_b200.runner.production_entry
