#!/usr/bin/env bash
# Measure the eventual-exit lens at nested fit sizes (8, 32, 56, 80).
#
# Existing lens binaries are accepted only with a current hash-valid sidecar.
# Evaluation resumes from the complete evaluate.py output set, never from a
# bare summary.json.  The convergence diagnostic always receives a durable
# --out path.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PYTHON:-venv/bin/python}
L=artifacts/jlens/lens/exit3
FIT=src/ouro_jlens/fit_lens.py
EVAL_ROOT=artifacts/jlens/eval
CONVERGENCE_OUT=${CONVERGENCE_OUT:-$EVAL_ROOT/fitsize_convergence.json}

[[ -x "$PY" || "$PY" == */python || "$PY" == python ]] || {
  echo "PYTHON is not executable: $PY" >&2
  exit 2
}

validate_lens() {
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
  "$PY" -c 'from ouro_jlens.fit_lens import validate_sidecar; import sys; validate_sidecar(sys.argv[1], kind=sys.argv[2])' \
    "$path" "$kind"
}

ensure_merge() {
  local out=$1
  local sidecar="${out%.pt}.json"
  shift
  if [[ -e "$out" || -L "$out" || -e "$sidecar" || -L "$sidecar" ]]; then
    validate_lens "$out" merged
    return 0
  fi
  local input
  for input in "$@"; do
    validate_lens "$input" fit
  done
  "$PY" "$FIT" merge --out "$out" "$@"
  validate_lens "$out" merged
}

eval_base_complete() {
  local out=$1
  [[ -d "$out" ]] || return 1
  local name
  for name in arrays.npz items.json task_names.json provenance.json; do
    [[ -f "$out/$name" && ! -L "$out/$name" ]] || return 1
  done
  "$PY" -c '
import hashlib, json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]).resolve()
try:
    arrays = np.load(root / "arrays.npz")
    if not arrays.files:
        raise ValueError("arrays.npz is empty")
    arrays.close()
    json.loads((root / "items.json").read_text())
    json.loads((root / "task_names.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    outputs = provenance.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("evaluation provenance has no outputs")
    for key, filename in (("arrays", "arrays.npz"), ("items", "items.json"), ("task_names", "task_names.json")):
        record = outputs.get(key)
        target = root / filename
        if not isinstance(record, dict) or Path(record.get("path", "")).resolve() != target:
            raise ValueError(f"{key} provenance path mismatch")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if record.get("sha256") != digest or record.get("size") != target.stat().st_size:
            raise ValueError(f"{key} provenance digest mismatch")
except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
    print(f"invalid evaluation output: {exc}", file=sys.stderr)
    raise SystemExit(1)
' "$out"
}

analysis_complete() {
  local out=$1
  [[ -f "$out/summary.json" && ! -L "$out/summary.json" &&
     -f "$out/summary.md" && ! -L "$out/summary.md" ]] || return 1
  "$PY" -c 'import json,sys; from pathlib import Path; value=json.loads(Path(sys.argv[1]).read_text()); assert isinstance(value,dict) and value' \
    "$out/summary.json"
}

run_eval() {
  local n=$1 lens=$2
  local out="$EVAL_ROOT/fitsize_n$n"
  mkdir -p "$EVAL_ROOT"
  # Never let an atomic evaluator write replace a pre-existing symlink.
  if [[ -L "$out" ]]; then
    echo "evaluation output root is a symlink: $out" >&2
    return 2
  fi
  if [[ -d "$out" ]] && find "$out" -type l -print -quit | grep -q .; then
    echo "evaluation tree contains a symlink: $out" >&2
    return 2
  fi
  if ! eval_base_complete "$out"; then
    "$PY" src/ouro_jlens/evaluate.py --lens "3=$lens" --out "$out"
    eval_base_complete "$out"
  fi
  # Analysis is always rerun after the complete base output is established;
  # a stale summary cannot make the run skip its current input validation.
  "$PY" src/ouro_jlens/analyze.py --eval "$out" > /dev/null
  analysis_complete "$out"
}

[[ ! -L "$EVAL_ROOT" ]] || {
  echo "output root is a symlink: $EVAL_ROOT" >&2
  exit 2
}
mkdir -p "$L" "$EVAL_ROOT"
P0_8="$L/exit3_p0-8.pt"
P8_32="$L/exit3_p8-32.pt"
P32_56="$L/exit3_p32-56.pt"
P56_80="$L/exit3_p56-80.pt"
N32="$L/exit3_merged.pt"
N56="$L/exit3_n56.pt"
N80="$L/exit3_n80.pt"

ensure_merge "$N32" "$P0_8" "$P8_32"
ensure_merge "$N56" "$P0_8" "$P8_32" "$P32_56"
ensure_merge "$N80" "$P0_8" "$P8_32" "$P32_56" "$P56_80"
validate_lens "$P0_8" fit
validate_lens "$P56_80" fit

run_eval 8 "$P0_8"
run_eval 32 "$N32"
run_eval 56 "$N56"
run_eval 80 "$N80"

if [[ -L "$CONVERGENCE_OUT" ]]; then
  echo "convergence output is a symlink: $CONVERGENCE_OUT" >&2
  exit 2
fi
mkdir -p "$(dirname "$CONVERGENCE_OUT")"
"$PY" src/ouro_jlens/lens_convergence.py "$P56_80" "$N56" --out "$CONVERGENCE_OUT"
[[ -f "$CONVERGENCE_OUT" && ! -L "$CONVERGENCE_OUT" ]] || {
  echo "convergence did not leave a durable output: $CONVERGENCE_OUT" >&2
  exit 2
}
"$PY" src/ouro_jlens/fitsize_report.py
