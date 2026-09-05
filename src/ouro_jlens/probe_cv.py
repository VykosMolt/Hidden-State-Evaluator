"""Cross-validated supervised probe on ``(a + b) * c = ``.

`probe.py` fits one probe on 384 prompts from 24 of the 45 unordered operand pairs and
scores it on a fixed 176-prompt test split. That is under-powered, which is why an
untrained logit lens beats it at loop 1, and it leaves the comparison on a quarter of
the family. Here the 45 unordered pairs are split into 5 folds: each prompt is a test
prompt in exactly one fold, C is chosen inside the fold on validation pairs drawn from
that fold's training pairs only, and the probe is refit on all training pairs before
scoring the held-out ones. Because a+b is symmetric, unordered pairs are the split unit
throughout, so a test pair's mirror is never in training (asserted per fold).

All 648 prompts receive a held-out score, but 72 have a label absent from their
training fold.  They are retained in the raw arrays and explicitly excluded by
``probe_report.py`` from the fair 576-prompt comparison.  Supervised layer
selection uses the same inner validation partition as C selection and is then
held fixed while scoring the untouched outer fold; the outer predictions never
select their own layer.  A raw max over held-out scores is descriptive only and
is not the reported estimand.

Hidden states are reused from a hash- and source-bound GPU cache
(`gpu_cache.npz`, key ``H``, [648, 192, 2048] fp16), so the supervised fit
needs no GPU.  The two lenses are re-scored on all 648 prompts with
`probe.lens_candidate_ranks` so the three readouts are compared on identical
data.  Existing cache/score files are reused only after their complete
provenance and byte identities validate; existence alone is never a resume
condition.

  # GPU step (needs the lock), writes lens_all648.npz next to --out
  python src/ouro_jlens/probe_cv.py --cache artifacts/jlens/probe/n80_v2/gpu_cache.npz \
      --lens artifacts/jlens/lens/exit3/exit3_n80.pt --out <dir> --lens-only
  # CPU step
  python src/ouro_jlens/probe_cv.py --cache artifacts/jlens/probe/n80_v2/gpu_cache.npz \
      --lens artifacts/jlens/lens/exit3/exit3_n80.pt --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouro_jlens.evidence import (  # noqa: E402
    atomic_savez,
    atomic_write_json,
    file_record,
    sha256_file,
)
from ouro_jlens.probe import (  # noqa: E402
    C_GRID,
    LABELS,
    _invalidate_outputs,
    _reject_writable_symlinks,
    _source_manifest as _probe_source_manifest,
    make_prompts,
)

N_UT, N_LAYER = 4, 48
N_FOLDS = 5
N_VAL_PAIRS = 8  # validation pairs held out of the 36 training pairs, for choosing C


def unordered_pairs() -> list[tuple[int, int]]:
    return [(a, b) for a in range(1, 10) for b in range(a, 10)]


def fold_assignment(seed: int = 0) -> dict[tuple[int, int], int]:
    """Assign each of the 45 unordered pairs to one of 5 folds (9 pairs each)."""
    pairs = unordered_pairs()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    return {pairs[p]: int(k % N_FOLDS) for k, p in enumerate(order)}


def prompt_folds(prompts: list[dict], seed: int = 0) -> np.ndarray:
    """Fold index per prompt, keyed by the *unordered* pair so mirrors share a fold."""
    assign = fold_assignment(seed)
    return np.array([assign[(min(q["a"], q["b"]), max(q["a"], q["b"]))] for q in prompts], int)


def validation_pairs(folds: np.ndarray, pair_of: list[tuple[int, int]], seed: int = 0
                     ) -> list[list[tuple[int, int]]]:
    """Persistable validation-pair selection used for regularisation tuning."""
    rng = np.random.default_rng(seed + 1)
    selected = []
    for fold in range(N_FOLDS):
        train_pairs = sorted({pair_of[i] for i in np.where(folds != fold)[0]})
        selected.append(sorted(train_pairs[i] for i in rng.permutation(len(train_pairs))[:N_VAL_PAIRS]))
    return selected


def _scaled(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return StandardScaler().fit(X[rows]).transform(X)


def _fit_location(H: np.ndarray, location: int, labels: np.ndarray, tr: np.ndarray,
                  va: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit/tune one location.  Scaling is shared across C candidates."""
    # BLAS/OpenMP reduction order changed discrete predictions in practice.
    # Freeze every location fit to one native thread so --jobs controls only
    # independent process-level parallelism and the estimator is reproducible
    # regardless of the caller's ambient thread variables.
    with threadpool_limits(limits=1):
        X = H[:, location]
        Xs = _scaled(X.astype(np.float32), tr)
        best, best_acc = C_GRID[0], -1.0
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(
                Xs[tr], labels[tr]
            )
            acc = clf.score(Xs[va], labels[va])
            if acc > best_acc:
                best, best_acc = C, acc
        fit_rows = tr | va
        Xs = _scaled(X.astype(np.float32), fit_rows)
        clf = LogisticRegression(C=best, max_iter=2000, solver="lbfgs").fit(
            Xs[fit_rows], labels[fit_rows]
        )
        return _rank_of_true(clf, Xs, labels, te), best, float(best_acc)


def _rank_of_true(clf, Xs: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Rank of the true label among the 17 candidates. Labels absent from the training
    fold keep probability 0, exactly as in probe.py."""
    proba = np.zeros((rows.sum(), len(LABELS)))
    proba[:, [LABELS.index(c) for c in clf.classes_]] = clf.predict_proba(Xs[rows])
    true_p = proba[np.arange(len(proba)), [LABELS.index(l) for l in y[rows]]]
    return (proba > true_p[:, None]).sum(1)


def cv_probe(H: np.ndarray, labels: np.ndarray, folds: np.ndarray, pair_of: list[tuple[int, int]],
             seed: int = 0, jobs: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outer-fold ranks plus inner-validation C and layer-selection scores.

    ``selection_accuracy[f, v]`` is measured only on validation pairs inside
    outer fold ``f``'s training partition.  It is therefore safe to select a
    location later scored on outer fold ``f``; predictions from classifiers
    trained on ``f`` never feed that choice.
    """
    if not np.isfinite(H).all():
        raise ValueError("hidden-state input contains non-finite values")
    n_loc = H.shape[1]
    ranks = np.full((len(H), n_loc), -1, np.int32)
    chosen = np.zeros((N_FOLDS, n_loc))
    selection_accuracy = np.full((N_FOLDS, n_loc), np.nan)
    val_pairs_by_fold = validation_pairs(folds, pair_of, seed)
    for f in range(N_FOLDS):
        train_pairs = sorted({pair_of[i] for i in np.where(folds != f)[0]})
        val_pairs = set(val_pairs_by_fold[f])
        te = folds == f
        va = np.array([p in val_pairs for p in pair_of]) & ~te
        tr = ~te & ~va
        # leak checks: unordered pair (hence its mirror) never spans train and test
        overlap = {pair_of[i] for i in np.where(tr | va)[0]} & {
            pair_of[i] for i in np.where(te)[0]
        }
        if overlap:
            raise ValueError(f"outer fold {f} leaks unordered pairs: {sorted(overlap)}")
        if not te.sum() or not va.sum() or not tr.sum():
            raise ValueError(f"outer fold {f} has an empty train, validation, or test partition")
        fitted = Parallel(n_jobs=jobs, prefer="processes", max_nbytes="10M")(
            delayed(_fit_location)(H, v, labels, tr, va, te) for v in range(n_loc)
        )
        for v, (location_ranks, best, best_acc) in enumerate(fitted):
            ranks[te, v] = location_ranks
            chosen[f, v] = best
            selection_accuracy[f, v] = best_acc
        print(f"fold {f}: test {int(te.sum())} val {int(va.sum())} train {int(tr.sum())} "
              f"prompts; loop-1 best top1 {(ranks[te, :N_LAYER] == 0).mean(0).max():.3f}", flush=True)
    if not (ranks >= 0).all():
        raise RuntimeError("every prompt must be scored exactly once")
    if not np.isfinite(selection_accuracy).all():
        raise RuntimeError("one or more inner-validation selection scores are missing")
    return ranks, chosen, selection_accuracy


def _logical_record(path: Path, logical_path: str) -> dict:
    """Record bytes under a relocatable, schema-defined identity."""

    record = file_record(path)
    record["path"] = logical_path
    return record


def _source_manifest(paths: list[Path]) -> dict:
    """Build a project-plus-complete-installed-jlens source manifest."""

    return _probe_source_manifest(paths)


def _runtime_versions() -> dict[str, str]:
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


def _check_record(
    record: object,
    expected: Path,
    label: str,
    *,
    logical_path: str,
    allow_symlink: bool = False,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record is missing")
    if not expected.is_file() or (expected.is_symlink() and not allow_symlink):
        raise ValueError(f"{label} is missing or linked: {expected}")
    try:
        if record.get("path") != logical_path:
            raise ValueError(f"{label} logical identity mismatch")
        if record.get("size") != expected.stat().st_size or record.get("sha256") != sha256_file(expected):
            raise ValueError(f"{label} byte identity mismatch")
    except (KeyError, OSError, TypeError) as exc:
        raise ValueError(f"{label} record is malformed") from exc


def _check_source_manifest(document: dict, expected: list[Path], label: str) -> None:
    records = document.get("source_files")
    if not isinstance(records, list):
        raise ValueError(f"{label} source manifest is incomplete")
    expected_manifest = _source_manifest(expected)
    if records != expected_manifest["files"]:
        raise ValueError(f"{label} source manifest does not bind current source bytes")
    if document.get("source_sha256") != expected_manifest["sha256"]:
        raise ValueError(f"{label} source aggregate mismatch")


def _cache_sources() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        here,
        here.with_name("probe.py"),
        here.with_name("evaluate.py"),
        here.with_name("evaldata.py"),
        here.with_name("recurrent.py"),
        here.with_name("evidence.py"),
    ]


def _score_sources() -> list[Path]:
    return [*_cache_sources(), Path(__file__).resolve().with_name("fit_lens.py")]


def _cpu_sources() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        here,
        here.with_name("probe.py"),
        here.with_name("probe_report.py"),
        here.with_name("evidence.py"),
    ]


def validate_hidden_cache(cache_path: Path) -> dict:
    """Validate the complete current-source/model provenance of hidden states."""

    provenance = cache_path.with_suffix(".provenance.json")
    if provenance.is_symlink() or not provenance.is_file():
        raise ValueError("hidden-state cache provenance is missing or linked")
    document = json.loads(provenance.read_text())
    if document.get("schema_version") != 2 or document.get("status") != "FRESH_CURRENT_SOURCE_AND_MODEL":
        raise ValueError("hidden-state cache is not current schema/source/model evidence")
    _check_record(
        document.get("output"), cache_path, "hidden-state cache", logical_path="gpu_cache"
    )
    _check_source_manifest(document, _cache_sources(), "hidden-state cache")
    if document.get("design") != {"prompts": 648, "virtual_locations": 192, "hidden_width": 2048}:
        raise ValueError("hidden-state cache design mismatch")
    if document.get("runtime_versions") != _runtime_versions():
        raise ValueError("hidden-state runtime dependency versions changed")
    from ouro_jlens.recurrent import OURO_SNAPSHOT, model_snapshot_files

    model_paths = model_snapshot_files(OURO_SNAPSHOT)
    model_records = document.get("model_files")
    if not isinstance(model_records, list) or len(model_records) != len(model_paths):
        raise ValueError("hidden-state model manifest is incomplete")
    for index, (record, path) in enumerate(zip(model_records, model_paths, strict=True)):
        _check_record(
            record,
            path,
            f"hidden-state model[{index}]",
            logical_path=f"model_snapshot/{path.relative_to(OURO_SNAPSHOT).as_posix()}",
            allow_symlink=True,
        )
    with np.load(cache_path, allow_pickle=False) as values:
        if set(values.files) != {"H", "correct"} \
                or values["H"].shape != (648, 192, 2048) \
                or values["H"].dtype != np.float16 \
                or values["correct"].shape != (648,) \
                or values["correct"].dtype != np.bool_:
            raise ValueError("hidden-state cache arrays have unexpected shape")
        if not np.isfinite(values["H"]).all():
            raise ValueError("hidden-state cache contains non-finite values")
    return document


def validate_lens_scores(score_path: Path, cache_path: Path, lens_path: Path) -> dict:
    """Reject every stale, partial, or mismatched GPU-score resume."""

    provenance = score_path.with_suffix(".provenance.json")
    if provenance.is_symlink() or not provenance.is_file():
        raise ValueError("lens-score provenance is missing or linked")
    document = json.loads(provenance.read_text())
    accepted = {
        "FRESH_CURRENT_SOURCE_HASH_BOUND_LENS",
        "FRESH_CURRENT_SOURCE_RETAINED_PRE_CUSTODY_LENS",
    }
    if document.get("schema_version") != 2 or document.get("status") not in accepted:
        raise ValueError("lens scores are not current schema/source evidence")
    _check_record(
        document.get("output"), score_path, "lens-score output", logical_path="lens_scores"
    )
    _check_source_manifest(document, _score_sources(), "lens scores")
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("lens-score input manifest is missing")
    _check_record(
        inputs.get("gpu_cache"), cache_path, "lens-score hidden cache", logical_path="gpu_cache"
    )
    _check_record(
        inputs.get("gpu_cache_provenance"),
        cache_path.with_suffix(".provenance.json"),
        "lens-score hidden-cache provenance",
        logical_path="gpu_cache_provenance",
    )
    _check_record(inputs.get("lens"), lens_path, "lens-score lens", logical_path="lens")
    _check_record(
        inputs.get("lens_sidecar"),
        lens_path.with_suffix(".json"),
        "lens-score lens sidecar",
        logical_path="lens_sidecar",
    )
    cache_document = validate_hidden_cache(cache_path)
    from ouro_jlens.probe import _lens_lineage

    lineage = _lens_lineage(lens_path)
    expected_status = (
        "FRESH_CURRENT_SOURCE_HASH_BOUND_LENS"
        if lineage == "HASH_BOUND"
        else "FRESH_CURRENT_SOURCE_RETAINED_PRE_CUSTODY_LENS"
    )
    if document.get("status") != expected_status or document.get("lens_lineage_status") != lineage:
        raise ValueError("lens-score lineage status does not match the retained sidecar")
    if document.get("model_files") != cache_document.get("model_files"):
        raise ValueError("lens-score model manifest differs from hidden-cache provenance")
    if document.get("runtime_versions") != _runtime_versions():
        raise ValueError("lens-score runtime dependency versions changed")
    with np.load(score_path, allow_pickle=False) as values:
        required = {"jl_cand", "jl_one", "jl_vocab", "ll_cand", "ll_one", "ll_vocab"}
        if not required <= set(values.files):
            raise ValueError("lens-score arrays are incomplete")
        for name in required:
            value = values[name]
            if value.shape != (648, 192) or not np.issubdtype(value.dtype, np.integer) or value.min() < 0:
                raise ValueError(f"lens-score array {name} has an invalid shape or value")
            if name.endswith(("cand", "one")) and value.max() >= len(LABELS):
                raise ValueError(f"lens-score candidate array {name} exceeds its support")
        if "lens_sha256" not in values.files \
                or str(values["lens_sha256"].item()) != sha256_file(lens_path):
            raise ValueError("lens-score archive is bound to another lens")
    return document


def validate_cpu_evidence(
    arrays_path: Path,
    design_path: Path,
    cache_path: Path,
    score_path: Path,
    *,
    seed: int,
    lens_path: Path | None = None,
) -> dict:
    """Validate the complete CPU probe product and its GPU-input ancestry."""

    if design_path.is_symlink() or not design_path.is_file():
        raise ValueError("probe CPU design/provenance is missing or linked")
    try:
        document = json.loads(design_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("probe CPU design/provenance is unreadable") from exc
    if document.get("schema_version") != 2 \
            or document.get("status") != "FRESH_CURRENT_SOURCE_INNER_SELECTED_OUTER_SCORED" \
            or document.get("seed") != seed or isinstance(document.get("seed"), bool):
        raise ValueError("probe CPU evidence schema, status, or seed is invalid")
    _check_source_manifest(document, _cpu_sources(), "probe CPU evidence")
    if document.get("runtime_versions") != _runtime_versions():
        raise ValueError("probe CPU runtime dependency versions changed")
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("probe CPU input manifest is missing")

    # The CPU archive is only meaningful through the complete GPU ancestry.
    # Validate those producers recursively before comparing any copied fields;
    # a forged design sidecar must not be able to bless stale cache/score bytes.
    cache_document = validate_hidden_cache(cache_path)
    if lens_path is None:
        raise ValueError("probe CPU validation requires the lens path for score ancestry")
    lens_path = Path(lens_path)
    score_document = validate_lens_scores(score_path, cache_path, lens_path)

    _check_record(inputs.get("gpu_cache"), cache_path, "probe CPU cache", logical_path="gpu_cache")
    _check_record(
        inputs.get("gpu_cache_provenance"),
        cache_path.with_suffix(".provenance.json"),
        "probe CPU hidden-cache provenance",
        logical_path="gpu_cache_provenance",
    )
    _check_record(
        inputs.get("lens_scores"), score_path, "probe CPU lens scores", logical_path="lens_scores"
    )
    _check_record(
        inputs.get("lens_score_provenance"),
        score_path.with_suffix(".provenance.json"),
        "probe CPU lens-score provenance",
        logical_path="lens_score_provenance",
    )
    _check_record(document.get("output"), arrays_path, "probe CPU arrays", logical_path="probe_arrays")
    if inputs["gpu_cache"] != cache_document["output"]:
        raise ValueError("probe CPU cache identity is not the validated hidden-cache output")
    if inputs["lens_scores"] != score_document["output"]:
        raise ValueError("probe CPU score identity is not the validated lens-score output")
    if inputs["lens_score_provenance"] != _logical_record(
        score_path.with_suffix(".provenance.json"), "lens_score_provenance"
    ):
        raise ValueError("probe CPU score provenance identity changed")
    if inputs["gpu_cache_provenance"] != _logical_record(
        cache_path.with_suffix(".provenance.json"), "gpu_cache_provenance"
    ):
        raise ValueError("probe CPU cache provenance identity changed")
    if score_document.get("model_files") != cache_document.get("model_files"):
        raise ValueError("probe CPU score/cache model identities differ")

    prompts = make_prompts()
    folds = prompt_folds(prompts, seed)
    pairs = [(min(q["a"], q["b"]), max(q["a"], q["b"])) for q in prompts]
    expected_validation = [
        [[int(a), int(b)] for a, b in values]
        for values in validation_pairs(folds, pairs, seed)
    ]
    expected_design = {
        "fold_assignment_unit": "unordered_operand_pair",
        "fold_sizes": [int((folds == fold).sum()) for fold in range(N_FOLDS)],
        "validation_pairs_by_fold": expected_validation,
        "selection_ancestry": (
            "for outer fold f, classifiers and selection_accuracy[f] use only its training and "
            "inner-validation pairs; the outer fold is scored only after selection"
        ),
    }
    if document.get("design") != expected_design:
        raise ValueError("probe CPU design no longer matches the frozen ancestry contract")

    required = {
        "probe_rank", "chosen_C", "selection_accuracy", "folds", "labels", "correct",
        "lens_sha256", "jl_cand", "jl_one", "jl_vocab", "ll_cand", "ll_one", "ll_vocab",
    }
    with np.load(arrays_path, allow_pickle=False) as arrays, \
            np.load(score_path, allow_pickle=False) as scores, \
            np.load(cache_path, allow_pickle=False) as cache:
        if set(arrays.files) != required:
            raise ValueError("probe CPU archive fields are incomplete or unexpected")
        if not np.array_equal(arrays["folds"], folds):
            raise ValueError("probe CPU archive fold assignment changed")
        labels = np.asarray([q["label"] for q in prompts])
        if not np.array_equal(arrays["labels"], labels):
            raise ValueError("probe CPU archive label order changed")
        if not np.array_equal(arrays["correct"], cache["correct"]):
            raise ValueError("probe CPU correctness vector differs from the hidden cache")
        for name in ("lens_sha256", "jl_cand", "jl_one", "jl_vocab", "ll_cand", "ll_one", "ll_vocab"):
            if not np.array_equal(arrays[name], scores[name]):
                raise ValueError(f"probe CPU archive changed GPU-derived field {name}")
        ranks = arrays["probe_rank"]
        selection = arrays["selection_accuracy"]
        chosen = arrays["chosen_C"]
        if ranks.shape != (648, 192) or not np.issubdtype(ranks.dtype, np.integer) \
                or ranks.min() < 0 or ranks.max() >= len(LABELS):
            raise ValueError("probe CPU supervised ranks are malformed")
        if selection.shape != (N_FOLDS, 192) or not np.isfinite(selection).all() \
                or np.any((selection < 0) | (selection > 1)):
            raise ValueError("probe CPU inner-validation selection scores are malformed")
        if chosen.shape != selection.shape or not np.isin(chosen, np.asarray(C_GRID)).all():
            raise ValueError("probe CPU C choices are outside the frozen grid")
    return document


def score_lenses_all(cache_path: Path, lens_path: str, out: Path) -> dict:
    """GPU: candidate/one-form/vocab ranks for both lenses on ALL 648 prompts."""
    import torch

    import jlens
    from ouro_jlens.evaldata import single_token_ids, surface_forms
    from ouro_jlens.evaluate import stacked_jacobians
    from ouro_jlens.probe import lens_candidate_ranks
    from ouro_jlens.recurrent import load_ouro

    cache_document = validate_hidden_cache(cache_path)
    lens_binary = Path(lens_path)
    lens_sidecar = lens_binary.with_suffix(".json")
    if lens_binary.is_symlink() or lens_sidecar.is_symlink() or not lens_sidecar.is_file():
        raise ValueError("lens score requires a regular lens binary and retained sidecar")
    from ouro_jlens.probe import _lens_lineage

    lens_status = _lens_lineage(lens_binary)
    status = (
        "FRESH_CURRENT_SOURCE_HASH_BOUND_LENS"
        if lens_status == "HASH_BOUND"
        else "FRESH_CURRENT_SOURCE_RETAINED_PRE_CUSTODY_LENS"
    )
    with np.load(cache_path, allow_pickle=False) as cache_values:
        H = torch.from_numpy(cache_values["H"])
    prompts = make_prompts()
    labels = np.array([q["label"] for q in prompts])
    m = load_ouro()
    label_tokens = [single_token_ids(m.tokenizer, surface_forms(str(s))) for s in LABELS]
    if not all(label_tokens):
        raise ValueError("one or more arithmetic labels have no single-token surface form")
    t0 = time.perf_counter()
    lens = jlens.JacobianLens.load(str(lens_binary))
    J = stacked_jacobians(m, lens, m.exit_index(m.n_ut - 1))
    jl = lens_candidate_ranks(m, H, labels, J, label_tokens)
    del J, lens
    torch.cuda.empty_cache()
    ll = lens_candidate_ranks(m, H, labels, None, label_tokens)
    print(f"lens scoring on all {len(H)} prompts in {time.perf_counter()-t0:.0f}s", flush=True)
    arrays = dict(zip(("jl_cand", "jl_one", "jl_vocab"), jl)) | dict(zip(("ll_cand", "ll_one", "ll_vocab"), ll))
    atomic_savez(out, lens_sha256=np.asarray(sha256_file(lens_binary)), **arrays)
    source = _source_manifest(_score_sources())
    atomic_write_json(out.with_suffix(".provenance.json"), {
        "schema_version": 2,
        "status": status,
        "lens_lineage_status": lens_status,
        "inputs": {
            "gpu_cache": _logical_record(cache_path, "gpu_cache"),
            "gpu_cache_provenance": _logical_record(
                cache_path.with_suffix(".provenance.json"), "gpu_cache_provenance"
            ),
            "lens": _logical_record(lens_binary, "lens"),
            "lens_sidecar": _logical_record(lens_sidecar, "lens_sidecar"),
        },
        "model_files": cache_document["model_files"],
        "runtime_versions": _runtime_versions(),
        "source_files": source["files"],
        "source_sha256": source["sha256"],
        "output": _logical_record(out, "lens_scores"),
    })
    validate_lens_scores(out, cache_path, lens_binary)
    return arrays


def rebuild_gpu_cache(cache_path: Path) -> None:
    """Regenerate all 648 hidden states under current source and bind the model bytes."""
    import gc
    import torch

    from ouro_jlens.probe import cache as cache_hidden
    from ouro_jlens.recurrent import OURO_SNAPSHOT, load_ouro, model_snapshot_files

    prompts = make_prompts()
    model = load_ouro()
    hidden, correct = cache_hidden(model, prompts)
    if hidden.dtype != torch.float16 or not torch.isfinite(hidden).all():
        raise ValueError("hidden-state capture has an invalid dtype or non-finite values")
    if len(correct) != len(prompts):
        raise ValueError("hidden-state capture correctness vector is incomplete")
    atomic_savez(cache_path, H=hidden.numpy(), correct=np.asarray(correct))
    model_files = model_snapshot_files(OURO_SNAPSHOT)
    source = _source_manifest(_cache_sources())
    atomic_write_json(cache_path.with_suffix(".provenance.json"), {
        "schema_version": 2,
        "status": "FRESH_CURRENT_SOURCE_AND_MODEL",
        "design": {"prompts": len(prompts), "virtual_locations": int(hidden.shape[1]),
                   "hidden_width": int(hidden.shape[2])},
        "model_files": [
            _logical_record(
                path, f"model_snapshot/{path.relative_to(OURO_SNAPSHOT).as_posix()}"
            ) for path in model_files
        ],
        "runtime_versions": _runtime_versions(),
        "source_files": source["files"],
        "source_sha256": source["sha256"],
        "output": _logical_record(cache_path, "gpu_cache"),
    })
    del model, hidden
    gc.collect()
    torch.cuda.empty_cache()
    validate_hidden_cache(cache_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="artifacts/jlens/probe/n80_v2/gpu_cache.npz")
    p.add_argument("--lens", default="artifacts/jlens/lens/exit3/exit3_n80.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--lens-only", action="store_true", help="run only the GPU step and exit")
    p.add_argument("--rebuild-gpu-cache", action="store_true",
                   help="replace the hidden-state cache under current source/model before scoring")
    p.add_argument("--rescore-lenses", action="store_true",
                   help="atomically replace lens scores after validating the hidden cache")
    args = p.parse_args()
    out = Path(args.out)
    _reject_writable_symlinks(out)
    out.mkdir(parents=True, exist_ok=True)
    _reject_writable_symlinks(out)
    cache, lens_all = Path(args.cache), out / "lens_all648.npz"
    _reject_writable_symlinks(cache)

    if args.rebuild_gpu_cache:
        # Invalidate every upstream and downstream artifact before replacing
        # the hidden states.  A crash can leave missing evidence, never a
        # mixed-generation score/archive/design/summary chain.
        _invalidate_outputs(
            out,
            (
                "lens_all648.npz",
                "lens_all648.provenance.json",
                "arrays.npz",
                "design.json",
                "summary.json",
            ),
        )
        _invalidate_outputs(
            cache.parent,
            (cache.name, cache.with_suffix(".provenance.json").name),
        )
        rebuild_gpu_cache(cache)
    elif args.rescore_lenses:
        # Replacing scores invalidates all CPU products that copied them.
        _invalidate_outputs(
            out,
            (
                "lens_all648.npz",
                "lens_all648.provenance.json",
                "arrays.npz",
                "design.json",
                "summary.json",
            ),
        )
    if args.rescore_lenses:
        validate_hidden_cache(cache)
        score_lenses_all(cache, args.lens, lens_all)
    elif lens_all.exists() or lens_all.with_suffix(".provenance.json").exists():
        if not lens_all.is_file() or lens_all.is_symlink():
            raise ValueError("lens-score output is partial or linked")
        validate_lens_scores(lens_all, cache, Path(args.lens))
    else:
        score_lenses_all(cache, args.lens, lens_all)
    if args.lens_only:
        return
    # The supervised CPU fit is a distinct derived generation.  Invalidate
    # its previous products even when the validated GPU cache/scores are
    # reused, before any fitting begins.
    _invalidate_outputs(out, ("arrays.npz", "design.json", "summary.json"))
    L = dict(np.load(lens_all))
    lens_provenance = lens_all.with_suffix(".provenance.json")
    lens_document = validate_lens_scores(lens_all, cache, Path(args.lens))
    lens_score_status = str(lens_document["status"])

    prompts = make_prompts()
    labels = np.array([q["label"] for q in prompts])
    pair_of = [(min(q["a"], q["b"]), max(q["a"], q["b"])) for q in prompts]
    folds = prompt_folds(prompts, args.seed)
    c = np.load(cache)
    H, correct = c["H"], c["correct"]
    if H.shape[0] != len(prompts) or len(prompts) != 648:
        raise ValueError("validated hidden cache no longer matches the frozen prompt family")

    t0 = time.perf_counter()
    ranks, chosen, selection_accuracy = cv_probe(H, labels, folds, pair_of, args.seed, jobs=args.jobs)
    print(f"cross-validated probes fitted in {time.perf_counter()-t0:.0f}s", flush=True)

    arrays_path = out / "arrays.npz"
    atomic_savez(arrays_path, probe_rank=ranks, chosen_C=chosen,
                 selection_accuracy=selection_accuracy, folds=folds,
                 labels=labels, correct=correct, **L)
    vals = validation_pairs(folds, pair_of, args.seed)
    design_path = out / "design.json"
    cpu_source = _source_manifest(_cpu_sources())
    atomic_write_json(design_path, {
        "schema_version": 2,
        "status": "FRESH_CURRENT_SOURCE_INNER_SELECTED_OUTER_SCORED",
        "seed": args.seed,
        "design": {
            "fold_assignment_unit": "unordered_operand_pair",
            "fold_sizes": [int((folds == f).sum()) for f in range(N_FOLDS)],
            "validation_pairs_by_fold": [
                [[int(a), int(b)] for a, b in pairs] for pairs in vals
            ],
            "selection_ancestry": (
                "for outer fold f, classifiers and selection_accuracy[f] use only its training and "
                "inner-validation pairs; the outer fold is scored only after selection"
            ),
        },
        "inputs": {
            "gpu_cache": _logical_record(cache, "gpu_cache"),
            "gpu_cache_provenance": _logical_record(
                cache.with_suffix(".provenance.json"), "gpu_cache_provenance"
            ),
            "lens_scores": _logical_record(lens_all, "lens_scores"),
            "lens_score_provenance": _logical_record(
                lens_provenance, "lens_score_provenance"
            ),
        },
        "runtime_versions": _runtime_versions(),
        "source_files": cpu_source["files"],
        "source_sha256": cpu_source["sha256"],
        "output": _logical_record(arrays_path, "probe_arrays"),
    })
    cpu_document = validate_cpu_evidence(
        arrays_path, design_path, cache, lens_all, seed=args.seed, lens_path=Path(args.lens)
    )
    from ouro_jlens.probe_report import build_report

    report = build_report(arrays_path, seed=args.seed)
    report["lens_score_status"] = lens_score_status
    report["lens_lineage_status"] = lens_document["lens_lineage_status"]
    report["input_provenance_status"] = "VALIDATED_CURRENT_SOURCE_MODEL_AND_SCORE_BYTES"
    report["status"] = (
        "LOCAL_CURRENT_SOURCE_HASH_BOUND_UNFROZEN"
        if lens_document["lens_lineage_status"] == "HASH_BOUND"
        else "LOCAL_CURRENT_SOURCE_MIXED_PRE_CUSTODY_LENS_UNFROZEN"
    )
    report["readout_evidence_status"] = {
        "supervised_probe": "FRESH_CURRENT_SOURCE_INNER_SELECTED_OUTER_SCORED",
        "logit_lens": "FRESH_CURRENT_SOURCE_AND_MODEL",
        "eventual_exit_jacobian_lens": lens_score_status,
    }
    report["upstream_evidence"] = {
        "hidden_cache": lens_document["inputs"]["gpu_cache"],
        "hidden_cache_provenance": lens_document["inputs"]["gpu_cache_provenance"],
        "lens": lens_document["inputs"]["lens"],
        "lens_sidecar": lens_document["inputs"]["lens_sidecar"],
        "lens_scores": lens_document["output"],
        "lens_score_source_sha256": lens_document["source_sha256"],
        "model_files": lens_document["model_files"],
        "cpu_arrays": cpu_document["output"],
        "cpu_design_provenance": _logical_record(design_path, "probe_cpu_design_provenance"),
        "cpu_source_sha256": cpu_document["source_sha256"],
    }
    atomic_write_json(out / "summary.json", report)
    for name, values in report["readouts"].items():
        print(f"{name}: cross-fitted eligible-population per-loop {values['per_loop']}")


if __name__ == "__main__":
    main()
