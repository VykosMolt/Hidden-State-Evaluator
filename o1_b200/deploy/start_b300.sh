#!/usr/bin/env bash
# Pod-side launch entrypoint for the accelerator session (B300 primary /
# B200 explicit fallback, INTERRUPTIBLE capacity).
#
# DOES NOTHING SCIENTIFIC BY ITSELF: it validates the environment and the
# artifacts, then hands control to the production zero-touch state machine
# (runner/production_entry.py), which enforces every gate — hardware
# identity + representative workloads, non-O1 equivalence, the frozen
# staged benchmark, frozen backend selection, replacement precommit,
# external commitment verification, and the affordability gate — before
# the sealed v2.1 calibration orchestrator runs a single row.  Committed
# rows stream continuously to the durable result destination, so eviction
# of the interruptible pod is recoverable at the next-missing-row boundary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${O1_B200_PYTHON:-/opt/venv/bin/python}"
OUT="${O1_B200_OUT:-/outputs}"

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="$ROOT"

mkdir -p "$OUT"

echo "[start_b300] environment validation"
"$PY" "$ROOT/o1_b200/deploy/validate_environment.py" --out "$OUT/environment_report.local.json"

echo "[start_b300] artifact verification"
"$PY" "$ROOT/o1_b200/deploy/verify_artifacts.py" \
  --manifest "${O1_B200_TRANSFER_MANIFEST:-$ROOT/o1_b200/deploy/TRANSFER_MANIFEST.json}"

echo "[start_b300] handing over to the production zero-touch state machine"
exec "$PY" -m o1_b200.runner.production_entry
