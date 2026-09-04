#!/usr/bin/env bash
# One local evaluation round using the current cross-validated probe contract.
#
# Lens binaries must have current fit/merged sidecars.  Evaluation resumes
# only from a complete evaluate.py output set; probe_cv's GPU and CPU phases
# are checked separately, so a dead worker cannot leave an infinite wait loop.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PYTHON:-venv/bin/python}
L=artifacts/jlens/lens
EVAL_ROOT=artifacts/jlens/eval
PROBE_ROOT=artifacts/jlens/probe
CACHE=${PROBE_CACHE:-artifacts/jlens/probe/n80_v2/gpu_cache.npz}
CACHE_PROVENANCE="${CACHE%.npz}.provenance.json"
TAG=${1:?usage: eval_round.sh TAG}

[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]] || {
  echo "TAG is not a safe artifact path component" >&2
  exit 2
}
[[ -x "$PY" || "$PY" == */python || "$PY" == python ]] || {
  echo "PYTHON is not executable: $PY" >&2
  exit 2
}

# A probe result is only resumable when its complete provenance chain still
# names the exact bytes that this wrapper will consume.  Keep this check in a
# single Python helper so the shell's resume branches cannot accidentally grow
# a weaker, presence-only variant.
verify_probe_metadata() {
  "$PY" - "$@" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from ouro_jlens.evidence import aggregate_sha256


def fail(message):
    raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check_no_link_ancestors(path, label):
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            fail(f"{label} path traverses a symlink: {current}")
        current = current.parent


def check_record(record, label, expected=None, *, allow_symlink=False):
    if not isinstance(record, dict):
        fail(f"{label} record is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        fail(f"{label} record path is malformed")
    path = Path(raw_path)
    target = path if expected is None else Path(expected)
    if expected is not None and path.resolve() != target.resolve():
        fail(f"{label} provenance path mismatch")
    check_no_link_ancestors(path, label)
    if target.is_symlink() and not allow_symlink:
        fail(f"{label} target is a symlink")
    if path.is_symlink() and not allow_symlink:
        fail(f"{label} record points through a symlink")
    if not path.is_file():
        fail(f"{label} file is missing")
    size = record.get("size")
    if type(size) is not int or size != path.stat().st_size:
        fail(f"{label} size mismatch")
    expected_digest = record.get("sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        fail(f"{label} digest is malformed")
    if digest(path) != expected_digest:
        fail(f"{label} digest mismatch")
    return path.resolve()


def check_sources(document, names, label):
    records = document.get("source_files")
    if not isinstance(records, list) or len(records) != len(names):
        fail(f"{label} source manifest is incomplete")
    expected = {Path(name).resolve() for name in names}
    resolved = []
    for index, record in enumerate(records):
        path = check_record(record, f"{label} source[{index}]")
        if path not in expected:
            fail(f"{label} source file is not current: {path}")
        resolved.append(path)
    if len(set(resolved)) != len(expected):
        fail(f"{label} source manifest has duplicate files")
    aggregate = aggregate_sha256({str(record["path"]): record["sha256"] for record in records})
    if document.get("source_sha256") != aggregate:
        fail(f"{label} source aggregate mismatch")


def check_npz(path, required, label, *, first_shape=None):
    check_no_link_ancestors(path, label)
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is missing or linked")
    with np.load(path, allow_pickle=False) as values:
        missing = set(required) - set(values.files)
        if missing:
            fail(f"{label} is missing arrays: {sorted(missing)}")
        if first_shape is not None:
            for name in required:
                value = values[name]
                if not value.shape or value.shape[0] != first_shape:
                    fail(f"{label} array has an invalid shape: {name}")


def verify_cache(cache, provenance):
    if provenance.is_symlink() or not provenance.is_file():
        fail("hidden-state cache provenance is missing or linked")
    document = json.loads(provenance.read_text())
    if document.get("schema_version") != 1:
        fail("unsupported hidden-state cache provenance schema")
    if document.get("status") != "FRESH_CURRENT_SOURCE_AND_MODEL":
        fail("hidden-state cache is not bound to current source and model")
    design = document.get("design")
    if design != {"hidden_width": 2048, "prompts": 648, "virtual_locations": 192}:
        fail("hidden-state cache design is not the current 648-prompt design")
    check_record(document.get("output"), "hidden-state cache", cache)
    check_npz(cache, {"H", "correct"}, "hidden-state cache")
    with np.load(cache, allow_pickle=False) as values:
        if values["H"].shape != (648, 192, 2048):
            fail("hidden-state cache H shape is not [648, 192, 2048]")
        if values["correct"].shape != (648,):
            fail("hidden-state cache correct shape is not [648]")
    check_sources(document, (
        "src/ouro_jlens/probe_cv.py",
        "src/ouro_jlens/probe.py",
        "src/ouro_jlens/recurrent.py",
    ), "hidden-state cache")
    model_records = document.get("model_files")
    if not isinstance(model_records, list):
        fail("hidden-state cache model manifest is missing")
    from ouro_jlens.recurrent import OURO_SNAPSHOT
    model_names = {"model.safetensors", "config.json", "modeling_ouro.py", "tokenizer.json"}
    expected = {(
        OURO_SNAPSHOT / name
    ).resolve(): name for name in model_names}
    if len(model_records) != len(expected):
        fail("hidden-state cache model manifest is incomplete")
    seen = set()
    for index, record in enumerate(model_records):
        path = check_record(record, f"hidden-state cache model[{index}]", allow_symlink=True)
        if path not in expected:
            fail(f"hidden-state cache model is not the pinned runtime snapshot: {path}")
        seen.add(path)
    if seen != set(expected):
        fail("hidden-state cache model manifest has duplicate files")


def verify_gpu(cache, provenance, hidden_cache, lens):
    if provenance.is_symlink() or not provenance.is_file():
        fail("lens-score provenance is missing or linked")
    document = json.loads(provenance.read_text())
    if document.get("schema_version") != 1 or document.get("status") != "FRESH_CURRENT_SOURCE":
        fail("lens scores are not bound to current source")
    required = {"jl_cand", "jl_one", "jl_vocab", "ll_cand", "ll_one", "ll_vocab"}
    check_npz(cache, required, "lens-score cache", first_shape=648)
    with np.load(cache, allow_pickle=False) as values:
        for name in required:
            if values[name].shape != (648, 192):
                fail(f"lens-score array has an invalid shape: {name}")
        if "lens" not in values.files:
            fail("lens-score cache is missing its lens binding")
        lens_value = values["lens"].item()
        if not isinstance(lens_value, str) or Path(lens_value).resolve() != lens.resolve():
            fail("lens-score cache lens binding mismatch")
    check_record(document.get("output"), "lens-score output", cache)
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        fail("lens-score inputs are missing")
    check_record(inputs.get("gpu_cache"), "lens-score hidden-state input", hidden_cache)
    check_record(inputs.get("lens"), "lens-score lens input", lens)
    check_sources(document, (
        "src/ouro_jlens/probe_cv.py",
        "src/ouro_jlens/probe.py",
        "src/ouro_jlens/evaluate.py",
        "src/ouro_jlens/recurrent.py",
    ), "lens scores")


def verify_cpu(arrays, design, summary, hidden_cache, lens_scores, lens_provenance):
    required = {
        "probe_rank", "chosen_C", "folds", "labels", "correct",
        "jl_cand", "jl_one", "jl_vocab", "ll_cand", "ll_one", "ll_vocab",
    }
    check_npz(arrays, required, "probe CPU arrays")
    with np.load(arrays, allow_pickle=False) as values:
        for name in required - {"chosen_C"}:
            if values[name].shape[0] != 648:
                fail(f"probe CPU array has an invalid prompt count: {name}")
        if values["chosen_C"].shape != (5, 192):
            fail("probe CPU chosen_C shape is not [5, 192]")
    design_document = json.loads(design.read_text())
    if design_document.get("schema_version") != 1:
        fail("unsupported probe CPU design schema")
    if design_document.get("seed") != 0:
        fail("probe CPU design was not produced with the frozen seed")
    if design_document.get("lens_score_status") != "FRESH_CURRENT_SOURCE":
        fail("probe CPU design uses unverified lens scores")
    inputs = design_document.get("inputs")
    if not isinstance(inputs, dict):
        fail("probe CPU design inputs are missing")
    check_record(inputs.get("gpu_cache"), "probe CPU hidden-state input", hidden_cache)
    check_record(inputs.get("lens_scores"), "probe CPU lens-score input", lens_scores)
    check_record(design_document.get("lens_score_provenance"),
                 "probe CPU lens-score provenance", lens_provenance)
    summary_document = json.loads(summary.read_text())
    if summary_document.get("schema_version") != 1:
        fail("unsupported probe CPU summary schema")
    if summary_document.get("lens_score_status") != "FRESH_CURRENT_SOURCE":
        fail("probe CPU summary uses unverified lens scores")
    if not isinstance(summary_document.get("readouts"), dict) or not summary_document["readouts"]:
        fail("probe CPU summary readouts are missing")
    check_record(summary_document.get("input"), "probe CPU summary arrays input", arrays)


try:
    mode, *raw = sys.argv[1:]
    paths = [Path(value) for value in raw]
    if mode == "cache" and len(paths) == 2:
        verify_cache(*paths)
    elif mode == "gpu" and len(paths) == 4:
        verify_gpu(*paths)
    elif mode == "cpu" and len(paths) == 6:
        verify_cpu(*paths)
    else:
        fail(f"invalid probe metadata arguments for {mode}")
except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, ImportError) as exc:
    print(f"invalid probe metadata: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
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
  local tag=$1 position=$2
  local out="$EVAL_ROOT/$tag"
  mkdir -p "$EVAL_ROOT"
  if [[ -L "$out" ]]; then
    echo "evaluation output root is a symlink: $out" >&2
    return 2
  fi
  if [[ -d "$out" ]] && find "$out" -type l -print -quit | grep -q .; then
    echo "evaluation tree contains a symlink: $out" >&2
    return 2
  fi
  if ! eval_base_complete "$out"; then
    local -a position_arg=()
    if [[ "$position" != "-1" ]]; then
      position_arg=(--position "$position")
    fi
    "$PY" src/ouro_jlens/evaluate.py --lens "3=$MERGED" \
      "${LENSES[@]}" "${position_arg[@]}" --out "$out"
    eval_base_complete "$out"
  fi
  "$PY" src/ouro_jlens/analyze.py --eval "$out" > /dev/null
  analysis_complete "$out"
}

for output_root in "$EVAL_ROOT" "$PROBE_ROOT"; do
  [[ ! -L "$output_root" ]] || {
    echo "output root is a symlink: $output_root" >&2
    exit 2
  }
done
mkdir -p "$L/exit3" "$EVAL_ROOT" "$PROBE_ROOT"
MERGED="$L/exit3/exit3_merged.pt"
MERGED_SIDECAR="${MERGED%.pt}.json"
shopt -s nullglob
shards=("$L"/exit3/exit3_p*.pt)
(( ${#shards[@]} > 0 )) || {
  echo "no exit-3 shards are available for a complete merge" >&2
  exit 2
}
for shard in "${shards[@]}"; do
  validate_lens "$shard" fit
done
shard_end=$("$PY" -c '
import sys
from ouro_jlens.fit_lens import validate_sidecar
ends = []
for path in sys.argv[1:]:
    metadata = validate_sidecar(path, kind="fit")
    end = metadata.get("end")
    if type(end) is not int:
        raise ValueError(f"invalid shard end in {path}")
    ends.append(end)
print(max(ends))
' "${shards[@]}")
merge_matches_current_shards() {
  "$PY" -c '
import sys
from pathlib import Path
from ouro_jlens.fit_lens import validate_sidecar
merged, expected_end, *shards = sys.argv[1:]
metadata = validate_sidecar(merged, kind="merged")
if metadata.get("end") != int(expected_end):
    raise ValueError("merged lens does not cover the current shard range")
recorded = metadata.get("shards")
if not isinstance(recorded, list) or {Path(value).resolve() for value in recorded} != {Path(value).resolve() for value in shards}:
    raise ValueError("merged lens is not bound to the current shard set")
' "$MERGED" "$shard_end" "${shards[@]}"
}
merged_matches=0
if [[ -e "$MERGED" || -L "$MERGED" || -e "$MERGED_SIDECAR" || -L "$MERGED_SIDECAR" ]]; then
  validate_lens "$MERGED" merged
  if merge_matches_current_shards; then
    merged_matches=1
  else
    # Preserve the old immutable merge and materialize a new path for an
    # extended shard set rather than silently evaluating stale bytes.
    MERGED="$L/exit3_merged_n${shard_end}.pt"
  fi
fi
if (( ! merged_matches )); then
  merged_sidecar="${MERGED%.pt}.json"
  if [[ -e "$MERGED" || -L "$MERGED" || -e "$merged_sidecar" || -L "$merged_sidecar" ]]; then
    validate_lens "$MERGED" merged
  else
    "$PY" src/ouro_jlens/fit_lens.py merge --out "$MERGED" "${shards[@]}"
  fi
  validate_lens "$MERGED" merged
  merge_matches_current_shards
fi

LENSES=()
for u in 0 1 2; do
  lens="$L/exit$u/exit${u}_p0-24.pt"
  validate_lens "$lens" fit
  LENSES+=(--lens "$u=$lens")
done
validate_lens "$MERGED" merged

run_eval "$TAG" -1
run_eval "${TAG}_pos-2" -2

[[ -f "$CACHE" && ! -L "$CACHE" ]] || {
  echo "probe_cv hidden-state cache is missing or linked: $CACHE" >&2
  exit 2
}
[[ -f "$CACHE_PROVENANCE" && ! -L "$CACHE_PROVENANCE" ]] || {
  echo "probe_cv hidden-state cache provenance is missing or linked: $CACHE_PROVENANCE" >&2
  exit 2
}
verify_probe_metadata cache "$CACHE" "$CACHE_PROVENANCE"
PROBE="$PROBE_ROOT/$TAG"
if [[ -L "$PROBE" ]]; then
  echo "probe output root is a symlink: $PROBE" >&2
  exit 2
fi
mkdir -p "$PROBE"
GPU_CACHE="$PROBE/lens_all648.npz"
GPU_PROVENANCE="$PROBE/lens_all648.provenance.json"
if [[ -e "$GPU_CACHE" || -L "$GPU_CACHE" || -e "$GPU_PROVENANCE" || -L "$GPU_PROVENANCE" ]]; then
  [[ -f "$GPU_CACHE" && ! -L "$GPU_CACHE" && -f "$GPU_PROVENANCE" && ! -L "$GPU_PROVENANCE" ]] || {
    echo "probe_cv GPU output is only partially present; refusing to replace it" >&2
    exit 2
  }
  verify_probe_metadata gpu "$GPU_CACHE" "$GPU_PROVENANCE" "$CACHE" "$MERGED"
else
  "$PY" src/ouro_jlens/probe_cv.py --cache "$CACHE" --lens "$MERGED" --out "$PROBE" \
    --jobs "${PROBE_JOBS:-8}" --lens-only
  verify_probe_metadata gpu "$GPU_CACHE" "$GPU_PROVENANCE" "$CACHE" "$MERGED"
fi

# The CPU phase writes arrays/design/summary atomically and can be rerun from
# the already verified lens_all648 cache after a crash.  A symlink is the one
# state that must never be replaced by an atomic output write.
for probe_output in arrays.npz design.json summary.json; do
  [[ ! -L "$PROBE/$probe_output" ]] || {
    echo "probe_cv CPU output is a symlink: $PROBE/$probe_output" >&2
    exit 2
  }
done
if ! verify_probe_metadata cpu "$PROBE/arrays.npz" "$PROBE/design.json" "$PROBE/summary.json" \
    "$CACHE" "$GPU_CACHE" "$GPU_PROVENANCE"; then
  "$PY" src/ouro_jlens/probe_cv.py --cache "$CACHE" --lens "$MERGED" --out "$PROBE" \
    --jobs "${PROBE_JOBS:-8}"
  verify_probe_metadata cpu "$PROBE/arrays.npz" "$PROBE/design.json" "$PROBE/summary.json" \
    "$CACHE" "$GPU_CACHE" "$GPU_PROVENANCE"
fi
"$PY" src/ouro_jlens/analyze.py --eval "$EVAL_ROOT/$TAG" --probe "$PROBE" > /dev/null
analysis_complete "$EVAL_ROOT/$TAG"
