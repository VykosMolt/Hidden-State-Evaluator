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


def check_record(record, label, expected=None, *, logical_path=None, allow_symlink=False):
    if not isinstance(record, dict):
        fail(f"{label} record is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        fail(f"{label} record path is malformed")
    path = Path(raw_path)
    target = path if expected is None else Path(expected)
    if logical_path is not None:
        if raw_path != logical_path:
            fail(f"{label} logical provenance path mismatch")
    elif expected is not None and path.resolve() != target.resolve():
        fail(f"{label} provenance path mismatch")
    check_no_link_ancestors(target, label)
    if target.is_symlink() and not allow_symlink:
        fail(f"{label} target is a symlink")
    if expected is None and path.is_symlink() and not allow_symlink:
        fail(f"{label} record points through a symlink")
    if not target.is_file():
        fail(f"{label} file is missing")
    size = record.get("size")
    if type(size) is not int or size != target.stat().st_size:
        fail(f"{label} size mismatch")
    expected_digest = record.get("sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        fail(f"{label} digest is malformed")
    if digest(target) != expected_digest:
        fail(f"{label} digest mismatch")
    return target.resolve()


def check_sources(document, names, label):
    records = document.get("source_files")
    if not isinstance(records, list) or len(records) != len(names):
        fail(f"{label} source manifest is incomplete")
    expected = {Path(name).resolve() for name in names}
    resolved = []
    for index, record in enumerate(records):
        name = names[index]
        path = check_record(
            record,
            f"{label} source[{index}]",
            expected=Path(name).resolve(),
            logical_path=name,
        )
        if path not in expected:
            fail(f"{label} source file is not current: {path}")
        resolved.append(path)
    if len(set(resolved)) != len(expected):
        fail(f"{label} source manifest has duplicate files")
    aggregate = aggregate_sha256({str(record["path"]): record["sha256"] for record in records})
    if document.get("source_sha256") != aggregate:
        fail(f"{label} source aggregate mismatch")


def check_nested_sources(document, names, label):
    """Validate the schema-2 CPU design's nested source manifest.

    The CPU design deliberately does not reuse the legacy top-level
    ``source_files``/``source_sha256`` fields.  Keeping this accessor strict
    prevents a schema-1 record from being made to look current merely by
    adding a new status string.
    """
    source = document.get("source")
    if not isinstance(source, dict):
        fail(f"{label} source block is missing")
    records = source.get("files")
    if not isinstance(records, list) or len(records) != len(names):
        fail(f"{label} source manifest is incomplete")
    expected = {Path(name).resolve() for name in names}
    resolved = []
    for index, record in enumerate(records):
        name = names[index]
        path = check_record(
            record,
            f"{label} source[{index}]",
            expected=Path(name).resolve(),
            logical_path=name,
        )
        if path not in expected:
            fail(f"{label} source file is not current: {path}")
        resolved.append(path)
    if len(set(resolved)) != len(expected):
        fail(f"{label} source manifest has duplicate files")
    aggregate = aggregate_sha256({str(record["path"]): record["sha256"] for record in records})
    if source.get("sha256") != aggregate:
        fail(f"{label} source aggregate mismatch")


def current_runtime_versions():
    import importlib.metadata
    import platform

    versions = {"python": platform.python_version()}
    for distribution in (
        "numpy", "torch", "scikit-learn", "threadpoolctl", "jlens", "transformers",
        "safetensors", "accelerate", "huggingface-hub",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    return versions


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
    # Preserve the useful byte-level failure for legacy partial records before
    # rejecting their obsolete schema.  A complete schema-1 record still fails
    # closed immediately below.
    if document.get("schema_version") != 2:
        check_record(document.get("output"), "hidden-state cache", cache)
        fail("unsupported hidden-state cache provenance schema")
    if document.get("status") != "FRESH_CURRENT_SOURCE_AND_MODEL":
        fail("hidden-state cache is not bound to current source and model")
    design = document.get("design")
    if design != {"hidden_width": 2048, "prompts": 648, "virtual_locations": 192}:
        fail("hidden-state cache design is not the current 648-prompt design")
    check_record(document.get("output"), "hidden-state cache", cache, logical_path="gpu_cache")
    check_npz(cache, {"H", "correct"}, "hidden-state cache")
    with np.load(cache, allow_pickle=False) as values:
        if values["H"].shape != (648, 192, 2048):
            fail("hidden-state cache H shape is not [648, 192, 2048]")
        if values["correct"].shape != (648,):
            fail("hidden-state cache correct shape is not [648]")
    check_sources(document, (
        "src/ouro_jlens/probe_cv.py",
        "src/ouro_jlens/probe.py",
        "src/ouro_jlens/evaluate.py",
        "src/ouro_jlens/evaldata.py",
        "src/ouro_jlens/recurrent.py",
        "src/ouro_jlens/evidence.py",
    ), "hidden-state cache")
    model_records = document.get("model_files")
    if not isinstance(model_records, list):
        fail("hidden-state cache model manifest is missing")
    from ouro_jlens.recurrent import OURO_SNAPSHOT, model_snapshot_files
    model_paths = model_snapshot_files(OURO_SNAPSHOT)
    model_names = tuple(
        path.relative_to(OURO_SNAPSHOT).as_posix() for path in model_paths
    )
    expected = {path.resolve(): name for path, name in zip(model_paths, model_names, strict=True)}
    if len(model_records) != len(expected):
        fail("hidden-state cache model manifest is incomplete")
    seen = set()
    for index, (name, path, record) in enumerate(zip(model_names, model_paths, model_records, strict=True)):
        path = check_record(
            record,
            f"hidden-state cache model[{index}]",
            expected=path,
            logical_path=f"model_snapshot/{name}",
        )
        if path not in expected:
            fail(f"hidden-state cache model is not the pinned runtime snapshot: {path}")
        seen.add(path)
    if seen != set(expected):
        fail("hidden-state cache model manifest has duplicate files")


def verify_gpu(cache, provenance, hidden_cache, lens):
    if provenance.is_symlink() or not provenance.is_file():
        fail("lens-score provenance is missing or linked")
    document = json.loads(provenance.read_text())
    accepted_statuses = {
        "FRESH_CURRENT_SOURCE_HASH_BOUND_LENS",
        "FRESH_CURRENT_SOURCE_RETAINED_PRE_CUSTODY_LENS",
    }
    if document.get("schema_version") != 2 or document.get("status") not in accepted_statuses:
        fail("lens scores are not bound to current source")
    required = {"jl_cand", "jl_one", "jl_vocab", "ll_cand", "ll_one", "ll_vocab"}
    check_npz(cache, required, "lens-score cache", first_shape=648)
    with np.load(cache, allow_pickle=False) as values:
        for name in required:
            if values[name].shape != (648, 192):
                fail(f"lens-score array has an invalid shape: {name}")
        if "lens_sha256" not in values.files:
            fail("lens-score cache is missing its lens binding")
        lens_digest = values["lens_sha256"].item()
        if not isinstance(lens_digest, str) or lens_digest != digest(lens):
            fail("lens-score cache lens binding mismatch")
    check_record(document.get("output"), "lens-score output", cache, logical_path="lens_scores")
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        fail("lens-score inputs are missing")
    check_record(
        inputs.get("gpu_cache"), "lens-score hidden-state input", hidden_cache,
        logical_path="gpu_cache",
    )
    check_record(
        inputs.get("gpu_cache_provenance"), "lens-score hidden-state provenance",
        hidden_cache.with_suffix(".provenance.json"), logical_path="gpu_cache_provenance",
    )
    check_record(
        inputs.get("lens"), "lens-score lens input", lens, logical_path="lens",
    )
    check_record(
        inputs.get("lens_sidecar"), "lens-score lens sidecar", lens.with_suffix(".json"),
        logical_path="lens_sidecar",
    )
    check_sources(document, (
        "src/ouro_jlens/probe_cv.py",
        "src/ouro_jlens/probe.py",
        "src/ouro_jlens/evaluate.py",
        "src/ouro_jlens/evaldata.py",
        "src/ouro_jlens/recurrent.py",
        "src/ouro_jlens/evidence.py",
        "src/ouro_jlens/fit_lens.py",
    ), "lens scores")


def verify_cpu(arrays, design, summary, hidden_cache, lens_scores, lens_provenance):
    accepted_statuses = {
        "FRESH_CURRENT_SOURCE_HASH_BOUND_LENS",
        "FRESH_CURRENT_SOURCE_RETAINED_PRE_CUSTODY_LENS",
    }
    required = {
        "probe_rank", "chosen_C", "selection_accuracy", "folds", "labels", "correct",
        "jl_cand", "jl_one", "jl_vocab", "ll_cand", "ll_one", "ll_vocab",
    }
    check_npz(arrays, required, "probe CPU arrays")
    with np.load(arrays, allow_pickle=False) as values:
        for name in required - {"chosen_C", "selection_accuracy"}:
            if values[name].shape[0] != 648:
                fail(f"probe CPU array has an invalid prompt count: {name}")
        if values["chosen_C"].shape != (5, 192):
            fail("probe CPU chosen_C shape is not [5, 192]")
        if values["selection_accuracy"].shape != (5, 192):
            fail("probe CPU selection_accuracy shape is not [5, 192]")
        if not np.isfinite(values["selection_accuracy"]).all() \
                or np.any(values["selection_accuracy"] < 0) \
                or np.any(values["selection_accuracy"] > 1):
            fail("probe CPU selection_accuracy must be finite and within [0, 1]")
        if not np.isin(values["chosen_C"], (0.01, 0.1, 1.0)).all():
            fail("probe CPU chosen_C contains a value outside the frozen C grid")
    design_document = json.loads(design.read_text())
    if design_document.get("schema_version") != 2:
        fail("unsupported probe CPU design schema")
    if design_document.get("status") != "FRESH_CURRENT_SOURCE_INNER_SELECTED_OUTER_SCORED":
        fail("probe CPU design status is not current")
    if design_document.get("seed") != 0:
        fail("probe CPU design was not produced with the frozen seed")
    # Schema-2 CPU design provenance is grouped under source/runtime/inputs/
    # outputs.  Do not consult the removed schema-1 top-level lens fields.
    if any(key in design_document for key in (
        "lens_score_status", "source_files", "source_sha256", "runtime_versions",
        "output", "lens_score_provenance",
    )):
        fail("probe CPU design uses legacy top-level provenance fields")
    check_nested_sources(design_document, (
        "src/ouro_jlens/probe_cv.py",
        "src/ouro_jlens/probe.py",
        "src/ouro_jlens/probe_report.py",
        "src/ouro_jlens/evidence.py",
    ), "probe CPU design")
    runtime = design_document.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("versions") != current_runtime_versions():
        fail("probe CPU runtime dependency versions changed")
    inputs = design_document.get("inputs")
    if not isinstance(inputs, dict) or not {
        "gpu_cache", "lens_scores", "lens_score_provenance",
    }.issubset(inputs):
        fail("probe CPU design inputs are missing")
    check_record(
        inputs.get("gpu_cache"), "probe CPU hidden-state input", hidden_cache,
        logical_path="gpu_cache",
    )
    check_record(
        inputs.get("lens_scores"), "probe CPU lens-score input", lens_scores,
        logical_path="lens_scores",
    )
    check_record(
        inputs.get("lens_score_provenance"), "probe CPU lens-score provenance",
        lens_provenance, logical_path="lens_score_provenance",
    )
    outputs = design_document.get("outputs")
    if not isinstance(outputs, dict) or "arrays" not in outputs:
        fail("probe CPU design outputs are missing")
    check_record(outputs.get("arrays"), "probe CPU arrays output", arrays,
                 logical_path="probe_arrays")
    summary_document = json.loads(summary.read_text())
    # summary.json is the intentionally separate report schema emitted by
    # probe_report.py; its input bytes are still checked, while CPU design
    # provenance above is the current schema-2 gate.
    if summary_document.get("schema_version") != 1:
        fail("unsupported probe CPU summary schema")
    if summary_document.get("lens_score_status") not in accepted_statuses:
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

eval_complete() {
  local out=$1
  local prompt_policy=$2
  shift 2
  "$PY" -c '
from pathlib import Path
import sys
from ouro_jlens.evaluate import validate_evaluation_provenance
specs = []
for value in sys.argv[3:]:
    target, path = value.split("=", 1)
    specs.append((int(target), Path(path)))
validate_evaluation_provenance(
    sys.argv[1], lens_paths=specs, expected_prompt_policy=sys.argv[2]
)
' "$out" "$prompt_policy" "$@"
}

analysis_complete() {
  local out=$1
  [[ -f "$out/summary.json" && ! -L "$out/summary.json" &&
     -f "$out/summary.md" && ! -L "$out/summary.md" ]] || return 1
  "$PY" -c 'import json,sys; from pathlib import Path; value=json.loads(Path(sys.argv[1]).read_text()); (isinstance(value,dict) and bool(value)) or sys.exit("summary JSON must be a non-empty object")' \
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
  if ! eval_complete "$out" nested "3=$MERGED" "${LENS_SPECS[@]}"; then
    local -a position_arg=()
    if [[ "$position" != "-1" ]]; then
      position_arg=(--position "$position")
    fi
    "$PY" src/ouro_jlens/evaluate.py --lens "3=$MERGED" \
      "${LENSES[@]}" --prompt-policy nested "${position_arg[@]}" --out "$out"
    eval_complete "$out" nested "3=$MERGED" "${LENS_SPECS[@]}"
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
from ouro_jlens.fit_lens import validate_sidecar, validate_merged_shards
merged, expected_end, *shards = sys.argv[1:]
metadata = validate_sidecar(merged, kind="merged")
if metadata.get("end") != int(expected_end):
    raise ValueError("merged lens does not cover the current shard range")
validate_merged_shards(merged, shards)
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
LENS_SPECS=()
for u in 0 1 2; do
  lens="$L/exit$u/exit${u}_p0-24.pt"
  validate_lens "$lens" fit
  LENSES+=(--lens "$u=$lens")
  LENS_SPECS+=("$u=$lens")
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
