"""Deterministic, leakage-aware reporting for the arithmetic probe experiment.

The probe is cross-validated over unordered operand pairs.  Some held-out folds
contain labels that do not occur in their training folds, so the comparison
population is the subset whose label is trainable in its fold.  For the learned
probe, layer and C selection use only an inner validation split inside each
outer training partition.  Fixed lens readouts select on the other outer folds.
No classifier trained on an outer test fold can influence its selected layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ouro_jlens.evidence import atomic_write_json, file_record
from ouro_jlens.probe import C_GRID, LABELS, make_prompts
from ouro_jlens.probe_cv import N_FOLDS, N_LAYER, N_UT, prompt_folds, validation_pairs

READOUTS = {
    "supervised_probe": "probe_rank",
    "logit_lens": "ll_cand",
    "eventual_exit_jacobian_lens": "jl_cand",
}


def _logical_input_record(path: Path) -> dict:
    """Record the supplied archive without embedding its physical location."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("probe report input archive is missing or linked")
    record = file_record(path)
    record["path"] = "probe_arrays"
    return record


def design(arrays: dict[str, np.ndarray], seed: int) -> dict:
    prompts = make_prompts()
    labels = np.asarray(arrays["labels"])
    folds = np.asarray(arrays["folds"])
    expected_folds = prompt_folds(prompts, seed)
    if len(prompts) != 648 or labels.shape != (648,) or not np.array_equal(folds, expected_folds):
        raise ValueError("probe arrays do not match the frozen 648-prompt/five-fold design")
    expected_labels = np.asarray([q["label"] for q in prompts])
    if not np.array_equal(labels, expected_labels):
        raise ValueError("probe labels do not match make_prompts() ordering")

    pairs = [(min(q["a"], q["b"]), max(q["a"], q["b"])) for q in prompts]
    eligible = np.zeros(len(prompts), dtype=bool)
    for fold in range(N_FOLDS):
        train = folds != fold
        test = folds == fold
        eligible |= test & np.isin(labels, np.unique(labels[train]))

    cluster_ids = np.asarray([f"{a}+{b}" for a, b in pairs])
    all_clusters = sorted(set(cluster_ids.tolist()))
    eligible_clusters = sorted(set(cluster_ids[eligible].tolist()))
    excluded_clusters = sorted(set(cluster_ids[~eligible].tolist()))
    if eligible.sum() != 576 or len(eligible_clusters) != 39 or len(excluded_clusters) != 6:
        raise ValueError("label-coverage population changed; review the estimand before reporting")
    return {
        "labels": labels,
        "folds": folds,
        "eligible": eligible,
        "cluster_ids": cluster_ids,
        "all_clusters": all_clusters,
        "eligible_clusters": eligible_clusters,
        "excluded_clusters": excluded_clusters,
        "validation_pairs_by_fold": validation_pairs(folds, pairs, seed),
    }


def cross_fitted_point(
    ranks: np.ndarray,
    folds: np.ndarray,
    eligible: np.ndarray,
    rows: np.ndarray | None = None,
    *,
    selection_accuracy: np.ndarray | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    """Return per-loop accuracy and fold-specific selected physical layers.

    ``rows`` may contain repeated indices, which is used by the cluster bootstrap.
    """
    ranks = np.asarray(ranks)
    if ranks.shape != (len(folds), N_UT * N_LAYER):
        raise ValueError(f"unexpected rank shape {ranks.shape}")
    rows = np.flatnonzero(eligible) if rows is None else np.asarray(rows, dtype=int)
    hit = ranks == 0
    if selection_accuracy is not None:
        selection_accuracy = np.asarray(selection_accuracy, dtype=float)
        if selection_accuracy.shape != (N_FOLDS, N_UT * N_LAYER) \
                or not np.isfinite(selection_accuracy).all() \
                or np.any((selection_accuracy < 0) | (selection_accuracy > 1)):
            raise ValueError("supervised selection accuracy has an invalid shape or value")
    values: list[float] = []
    choices: list[list[int]] = []
    for loop in range(N_UT):
        selected: list[int] = []
        numerator = 0.0
        denominator = 0
        block = slice(loop * N_LAYER, (loop + 1) * N_LAYER)
        for fold in range(N_FOLDS):
            test_rows = rows[(folds[rows] == fold) & eligible[rows]]
            if not len(test_rows):
                raise ValueError("bootstrap/design draw omitted a required fold")
            if selection_accuracy is None:
                train_rows = rows[(folds[rows] != fold) & eligible[rows]]
                if not len(train_rows):
                    raise ValueError("bootstrap/design draw omitted selection folds")
                layer = int(hit[train_rows, block].mean(0).argmax())
            else:
                # These scores were produced on inner validation pairs by
                # classifiers whose training/validation data both exclude the
                # outer fold.  Keep them fixed under the outer test-cluster
                # bootstrap; the interval is conditional on hyperparameter
                # selection rather than a leaky re-selection.
                layer = int(selection_accuracy[fold, block].argmax())
            selected.append(layer)
            numerator += float(hit[test_rows, loop * N_LAYER + layer].sum())
            denominator += len(test_rows)
        values.append(numerator / denominator)
        choices.append(selected)
    return np.asarray(values), choices


def _cluster_rows(sampled: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    return np.concatenate([np.flatnonzero(cluster_ids == cluster) for cluster in sampled])


def cluster_bootstrap(readouts: dict[str, np.ndarray], d: dict, *, seed: int,
                      draws: int, selection_accuracy: np.ndarray
                      ) -> tuple[dict[str, list[list[float]]], dict[str, list[list[float]]]]:
    """Outer-test cluster intervals with ancestry-safe layer selection."""
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("bootstrap draws must be a positive integer")
    rng = np.random.default_rng(seed)
    clusters = np.asarray(d["eligible_clusters"])
    samples = {name: np.empty((draws, N_UT), float) for name in readouts}
    for draw in range(draws):
        # Rejection is deterministic and vanishingly rare; every fold is required so
        # cross-fitting never silently changes its estimand.
        while True:
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            rows = _cluster_rows(sampled, d["cluster_ids"])
            if len(set(d["folds"][rows].tolist())) == N_FOLDS:
                break
        for name, ranks in readouts.items():
            samples[name][draw] = cross_fitted_point(
                ranks,
                d["folds"],
                d["eligible"],
                rows=rows,
                selection_accuracy=selection_accuracy if name == "supervised_probe" else None,
            )[0]

    ci = {
        name: np.percentile(values, [2.5, 97.5], axis=0).T.round(4).tolist()
        for name, values in samples.items()
    }
    contrasts = {}
    for left, right in (
        ("supervised_probe", "logit_lens"),
        ("eventual_exit_jacobian_lens", "logit_lens"),
        ("eventual_exit_jacobian_lens", "supervised_probe"),
    ):
        key = f"{left}_minus_{right}"
        delta = samples[left] - samples[right]
        contrasts[key] = np.percentile(delta, [2.5, 97.5], axis=0).T.round(4).tolist()
    return ci, contrasts


def build_report(arrays_path: Path, *, seed: int = 0, draws: int = 5000) -> dict:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("bootstrap draws must be a positive integer")
    arrays_path = Path(arrays_path)
    raw = np.load(arrays_path, allow_pickle=False)
    arrays = {key: raw[key] for key in raw.files}
    d = design(arrays, seed)
    if "selection_accuracy" not in arrays:
        raise ValueError("probe arrays lack outer-safe inner-validation layer scores")
    selection_accuracy = np.asarray(arrays["selection_accuracy"], dtype=float)
    if selection_accuracy.shape != (N_FOLDS, N_UT * N_LAYER) \
            or not np.isfinite(selection_accuracy).all() \
            or np.any((selection_accuracy < 0) | (selection_accuracy > 1)):
        raise ValueError("inner-validation selection scores are malformed")
    chosen_c = np.asarray(arrays.get("chosen_C"))
    if chosen_c.shape != selection_accuracy.shape \
            or not np.isin(chosen_c, np.asarray(C_GRID)).all():
        raise ValueError("per-fold probe regularisation choices are missing or outside C_GRID")
    readouts = {}
    for name, key in READOUTS.items():
        ranks = np.asarray(arrays[key])
        if not np.issubdtype(ranks.dtype, np.integer) or ranks.shape != (648, N_UT * N_LAYER) \
                or ranks.min() < 0 or ranks.max() >= len(LABELS):
            raise ValueError(f"readout {name} has invalid rank values")
        readouts[name] = ranks
    points, layers = {}, {}
    for name, ranks in readouts.items():
        point, selected = cross_fitted_point(
            ranks,
            d["folds"],
            d["eligible"],
            selection_accuracy=selection_accuracy if name == "supervised_probe" else None,
        )
        points[name] = point
        layers[name] = selected
    ci, contrast_ci = cluster_bootstrap(
        readouts, d, seed=seed + 991, draws=draws, selection_accuracy=selection_accuracy
    )

    labels = d["labels"]
    eligible = d["eligible"]
    report = {
        "schema_version": 1,
        "status": "DERIVED_FROM_SUPPLIED_ARRAYS_PROVENANCE_NOT_VALIDATED",
        "input_provenance_status": "NOT_VALIDATED_BY_STANDALONE_REPORT_GENERATOR",
        "population": {
            "total_prompts": int(len(labels)),
            "eligible_prompts": int(eligible.sum()),
            "excluded_untrainable_prompts": int((~eligible).sum()),
            "total_unordered_pair_clusters": len(d["all_clusters"]),
            "eligible_unordered_pair_clusters": len(d["eligible_clusters"]),
            "excluded_clusters": d["excluded_clusters"],
            "validation_pairs_by_fold": [
                [[int(a), int(b)] for a, b in pairs] for pairs in d["validation_pairs_by_fold"]
            ],
            "fold_sizes_all": [int((d["folds"] == f).sum()) for f in range(N_FOLDS)],
            "fold_sizes_eligible": [int(((d["folds"] == f) & eligible).sum()) for f in range(N_FOLDS)],
        },
        "estimand": (
            "candidate-set top-1 on fold-trainable prompts; supervised layer/C chosen by "
            "inner validation wholly inside each outer training partition, then scored on its untouched outer fold"
        ),
        "baselines": {
            "uniform_17_way": round(1 / len(LABELS), 6),
            "majority_all_648": round(float(np.bincount(labels).max() / len(labels)), 6),
            "majority_eligible_576": round(float(np.bincount(labels[eligible]).max() / eligible.sum()), 6),
        },
        "readouts": {},
        "contrasts": {},
        "bootstrap": {
            "unit": "outer-test unordered_operand_pair",
            "draws": draws,
            "seed": seed + 991,
            "supervised_layer_selection": "fixed inner-validation choice; interval conditional on selection",
            "fixed_lens_layer_selection": "repeated from other outer folds per draw",
        },
        "selection_ancestry": {
            "supervised_probe": (
                "for outer fold f, selection_accuracy[f] uses only its training and inner-validation pairs; "
                "fold f is absent from classifier training, C selection, and layer selection"
            ),
            "fixed_lenses": "no trained parameters; layer selected on held-out predictions from other outer folds",
        },
        "input": _logical_input_record(arrays_path),
    }
    for name, point in points.items():
        report["readouts"][name] = {
            "per_loop": point.round(6).tolist(),
            "ci95": ci[name],
            "selected_physical_layer_by_heldout_fold": layers[name],
        }
    for left, right in (
        ("supervised_probe", "logit_lens"),
        ("eventual_exit_jacobian_lens", "logit_lens"),
        ("eventual_exit_jacobian_lens", "supervised_probe"),
    ):
        key = f"{left}_minus_{right}"
        report["contrasts"][key] = {
            "per_loop": (points[left] - points[right]).round(6).tolist(),
            "ci95": contrast_ci[key],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    report = build_report(Path(args.arrays), seed=args.seed, draws=args.draws)
    atomic_write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
