#!/usr/bin/env bash
# Crash-safe local continuation: fit the initial exit family, evaluate it,
# extend the eventual-exit lens, then evaluate again.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PYTHON:-venv/bin/python}
L=artifacts/jlens/lens
FIT=src/ouro_jlens/fit_lens.py

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

shard() {
  local target=$1 start=$2 end=$3
  local dir="$L/exit$target" out="$L/exit$target/exit${target}_p${start}-${end}.pt"
  local sidecar="${out%.pt}.json"
  [[ ! -L "$dir" ]] || {
    echo "lens output directory is a symlink: $dir" >&2
    return 2
  }
  mkdir -p "$dir"
  if [[ -e "$out" || -L "$out" || -e "$sidecar" || -L "$sidecar" ]]; then
    validate_lens "$out" fit
    return 0
  fi
  "$PY" "$FIT" fit --target-ut "$target" --start "$start" --end "$end" \
    --dim-batch 2 --checkpoint-every 8 --out "$out"
  validate_lens "$out" fit
  if [[ -e "$out.ckpt" || -L "$out.ckpt" || -e "$out.ckpt.json" || -L "$out.ckpt.json" ]]; then
    [[ -f "$out.ckpt" && ! -L "$out.ckpt" && -f "$out.ckpt.json" && ! -L "$out.ckpt.json" ]] || {
      echo "checkpoint pair is incomplete after successful fit: $out" >&2
      return 2
    }
    rm -f -- "$out.ckpt" "$out.ckpt.json"
  fi
}

[[ ! -L "$L" ]] || {
  echo "lens output root is a symlink: $L" >&2
  exit 2
}
mkdir -p "$L"
shard 3 0 8
shard 3 8 32
shard 0 0 24
shard 1 0 24
shard 2 0 24

bash src/ouro_jlens/eval_round.sh round1_exit3x32

shard 3 32 56
shard 3 56 80
shard 3 80 104

bash src/ouro_jlens/eval_round.sh round2_exit3x104
echo "== local continuation complete $(date -Is)"
