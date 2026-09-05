"""Independent arithmetic verifier for the canonical JLens artifact claims.

This intentionally does not import ``analyze.py``, ``report.py``,
``probe_report.py``, or ``transport_report.py``.  It duplicates the small
amount of aggregation needed to catch common-mode generator errors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ouro_jlens.evidence import atomic_write_json, file_record

N_UT, N_LAYER, K = 4, 48, 10
OPERATIONS = {"addition", "subtraction", "multiplication", "division", "mod", "squared"}


class VerificationError(ValueError):
    """The supplied artifacts do not satisfy the verifier's frozen contract."""


def _require(condition: bool, message: str) -> None:
    # Unlike ``assert``, evidence verification must remain active under
    # ``python -O``.
    if not condition:
        raise VerificationError(message)


def _logical_record(path: Path, logical_path: str) -> dict:
    """Bind an input file while keeping the verification result relocatable."""

    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"verification input is missing or linked: {path}")
    record = file_record(path)
    record["path"] = logical_path
    return record


def _invalidate_output(path: Path) -> None:
    """Remove a prior verifier verdict without traversing linked parents."""

    lexical = Path(os.path.abspath(path))
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink():
            raise VerificationError(f"verification output traverses a link: {candidate}")
    if lexical.is_dir():
        raise VerificationError(f"verification output is a directory: {lexical}")
    lexical.unlink(missing_ok=True)


def _verify_transport_shape(transport: dict) -> list[int]:
    """Validate the transport contract before consuming its numeric fields."""

    _require(isinstance(transport, dict), "transport must be a JSON object")
    _require(transport.get("schema_version") == 1, "transport schema changed")
    _require(transport.get("status") == "MODELED_ASSOCIATION_ONLY", "transport status is not accepted")
    raw_sizes = transport.get("sample_sizes")
    _require(
        isinstance(raw_sizes, list)
        and len(raw_sizes) >= 3
        and all(type(value) is int and value > 0 for value in raw_sizes)
        and len(set(raw_sizes)) == len(raw_sizes),
        "transport sample_sizes must be distinct positive integers",
    )
    per_loop = transport.get("per_loop")
    _require(isinstance(per_loop, list) and len(per_loop) == N_UT, "transport loop rows are incomplete")
    for index, row in enumerate(per_loop, start=1):
        _require(isinstance(row, dict) and row.get("loop") == index, "transport loop labels are invalid")
        raw = row.get("raw_fitted_map_norm_by_n")
        _require(isinstance(raw, dict), "transport raw fit-size map is missing")
        _require(
            set(raw) == {str(value) for value in raw_sizes}
            and all(type(value) in (int, float) and np.isfinite(value) and value >= 0 for value in raw.values()),
            "transport raw fit-size map is not finite/nonnegative",
        )
    negative = transport.get("negative_sigma_squared_total")
    _require(type(negative) is int and negative >= 0, "transport invalid-moment count is not an integer")
    return list(raw_sizes)


def _finite_vector(value: object, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    _require(array.shape == (length,) and np.isfinite(array).all(), f"{label} is malformed")
    return array


def verify_transport(transport: dict) -> dict:
    """Recompute the fit-size regression from retained sufficient statistics."""

    sizes = _verify_transport_shape(transport)
    per_source = transport.get("per_source")
    _require(isinstance(per_source, dict), "transport per-source statistics are missing")
    raw = per_source.get("squared_norms_by_n")
    _require(isinstance(raw, dict) and set(raw) == {str(n) for n in sizes},
             "transport sufficient statistics do not cover every fit size")
    rows = [np.asarray(raw[str(n)], dtype=float) for n in sizes]
    _require(rows and all(row.ndim == 1 and row.shape == rows[0].shape for row in rows),
             "transport squared-norm arrays have inconsistent shapes")
    squared = np.stack(rows)
    _require(squared.shape[1] > 0 and squared.shape[1] % N_LAYER == N_LAYER - 1,
             "transport source count does not match recurrent virtual locations")
    _require(np.isfinite(squared).all() and np.all(squared >= 0),
             "transport squared norms must be finite and nonnegative")
    design = np.column_stack([np.ones(len(sizes)), 1.0 / np.asarray(sizes, dtype=float)])
    mu2, sigma2 = np.linalg.lstsq(design, squared, rcond=None)[0]
    residual = squared - design @ np.stack([mu2, sigma2])
    np.testing.assert_allclose(
        _finite_vector(per_source.get("mu_squared"), len(mu2), "transport mu_squared"),
        mu2, rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        _finite_vector(per_source.get("sigma_squared"), len(sigma2), "transport sigma_squared"),
        sigma2, rtol=1e-12, atol=1e-12,
    )
    valid = np.asarray(per_source.get("valid"))
    _require(valid.shape == mu2.shape and valid.dtype == np.bool_, "transport validity mask is malformed")
    _require(np.array_equal(valid, (mu2 >= 0) & (sigma2 >= 0)),
             "transport validity mask disagrees with recomputed moments")
    _require(transport.get("n_sources") == len(mu2), "transport source count is inconsistent")
    _require(transport.get("negative_mu_squared_total") == int((mu2 < 0).sum()),
             "transport negative-mu count is inconsistent")
    _require(transport.get("negative_sigma_squared_total") == int((sigma2 < 0).sum()),
             "transport negative-sigma count is inconsistent")
    np.testing.assert_allclose(float(transport.get("rmse")), float(np.sqrt(np.mean(residual ** 2))),
                               rtol=1e-12, atol=1e-12)

    loop_rows = transport["per_loop"]
    recomputed = []
    for loop, row in enumerate(loop_rows):
        sl = slice(loop * N_LAYER, min((loop + 1) * N_LAYER, len(mu2)))
        m, s = mu2[sl], sigma2[sl]
        expected_raw = {
            str(n): round(float(np.sqrt(squared[index, sl]).mean()), 6)
            for index, n in enumerate(sizes)
        }
        _require(row.get("raw_fitted_map_norm_by_n") == expected_raw,
                 f"transport loop {loop + 1} raw norms disagree with sufficient statistics")
        checks = {
            "mu_norm_mean_valid": np.where(m >= 0, np.sqrt(np.maximum(m, 0)), np.nan),
            "sigma_rms_scatter_mean_valid": np.where(s >= 0, np.sqrt(np.maximum(s, 0)), np.nan),
            "modeled_single_prompt_rms_mean_valid": np.where(
                (m >= 0) & (s >= 0), np.sqrt(np.maximum(m + s, 0)), np.nan
            ),
            "modeled_total_rms_mean_where_sum_nonnegative": np.where(
                m + s >= 0, np.sqrt(np.maximum(m + s, 0)), np.nan
            ),
        }
        for name, values in checks.items():
            finite = values[np.isfinite(values)]
            expected = None if not len(finite) else float(finite.mean())
            if expected is None:
                _require(row.get(name) is None, f"transport loop {loop + 1} {name} should be null")
            else:
                np.testing.assert_allclose(float(row.get(name)), expected, rtol=1e-12, atol=1e-12)
        _require(row.get("negative_mu_squared") == int((m < 0).sum()),
                 f"transport loop {loop + 1} negative-mu count changed")
        _require(row.get("negative_sigma_squared") == int((s < 0).sum()),
                 f"transport loop {loop + 1} negative-sigma count changed")
        recomputed.append({"loop": loop + 1, "raw_fitted_map_norm_by_n": expected_raw})
    return {
        "sample_sizes": sizes,
        "n_sources": len(mu2),
        "negative_mu_squared": int((mu2 < 0).sum()),
        "negative_sigma_squared": int((sigma2 < 0).sum()),
        "per_loop": recomputed,
    }


def boundary_correct(continuation: str, target: str) -> bool:
    c, t = continuation.strip().strip('"').lower(), target.strip().lower()
    return c.startswith(t) and (len(c) == len(t) or not c[len(t)].isalnum())


def _item_mean(values: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    return np.stack([values[item_ids == item].mean(0) for item in np.unique(item_ids)])


def independent_excess(arrays: dict, items: list[dict], names: dict, task: str, key: str
                       ) -> tuple[np.ndarray, int, int]:
    valid = np.zeros((len(items), arrays[key].shape[1]), bool)
    own = np.zeros_like(valid)
    is_op = np.zeros_like(valid)
    rows, cols = [], []
    for i, item in enumerate(items):
        task_names = names[item["task"]]
        valid[i, :len(task_names)] = True
        is_op[i, :len(task_names)] = [name in OPERATIONS for name in task_names]
        for index in item["own_index"]:
            if index >= 0:
                own[i, index] = True
        for slot, index in enumerate(item["own_index"]):
            if index < 0 or slot >= len(item["intermediates"]):
                continue
            numeric = item["intermediates"][slot] not in OPERATIONS
            selected = item["task"] == task and item["scorable"][slot] and not item["leaked"][slot]
            selected &= task != "order-ops" or numeric
            if selected:
                rows.append(i)
                cols.append(index)
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    ranks = arrays[key]
    own_hit = (ranks[rows, cols].reshape(-1, N_UT, N_LAYER) < K).max(2).astype(float)
    control = np.zeros_like(own_hit)
    for slot, (row, col) in enumerate(zip(rows, cols)):
        candidates = valid[row] & ~own[row] & (is_op[row] == is_op[row, col])
        hits = (ranks[row, candidates].reshape(-1, N_UT, N_LAYER) < K).max(2)
        control[slot] = hits.mean(0)
    return _item_mean(own_hit - control, rows).mean(0), len(np.unique(rows)), len(rows)


def verify_main(root: Path, analysis: dict | None = None) -> dict:
    arrays = dict(np.load(root / "arrays.npz", allow_pickle=False))
    items = json.loads((root / "items.json").read_text())
    names = json.loads((root / "task_names.json").read_text())
    _require(isinstance(items, list) and all(isinstance(item, dict) for item in items),
             "main item metadata is malformed")
    _require(isinstance(names, dict), "main task-name metadata is malformed")
    expected_counts = {"multihop": (90, 100), "order-ops": (51, 51)}
    observed = {}
    for task, counts in expected_counts.items():
        j, n_items, n_slots = independent_excess(arrays, items, names, task, "jlens_exit3_allrank")
        ll, n_items2, n_slots2 = independent_excess(arrays, items, names, task, "logitlens_allrank")
        _require(
            (n_items, n_slots) == counts == (n_items2, n_slots2),
            f"{task} item/slot counts changed",
        )
        observed[task] = {"n_items": n_items, "n_slots": n_slots,
                          "j_excess": j.round(6).tolist(), "logit_excess": ll.round(6).tolist()}
    correct = sum(boundary_correct(i["continuation"], i["target"]) for i in items)
    _require(len(items) == 148, "main evaluation item population changed")
    for task in ("multihop", "order-ops"):
        rows = np.asarray([i for i, item in enumerate(items) if item["task"] == task])
        got = [(arrays["exit_top1"][rows, u] == arrays["exit_top1"][rows, -1]).mean() for u in range(N_UT)]
        observed[task]["exit_agreement_all_items"] = np.round(got, 6).tolist()
    observed["correct_items"] = correct
    observed["total_items"] = len(items)

    if analysis is not None:
        _require(isinstance(analysis, dict) and isinstance(analysis.get("main"), dict),
                 "canonical analysis main section is missing")
        main = analysis["main"]
        populations = main.get("populations")
        _require(isinstance(populations, dict), "canonical analysis populations are missing")
        for task, public in (("multihop", "multihop"), ("order-ops", "order_ops_numeric")):
            population = populations.get(public)
            _require(isinstance(population, dict), f"canonical analysis population {public} is missing")
            readouts = population.get("readouts")
            _require(isinstance(readouts, dict), f"canonical analysis readouts for {public} are missing")
            # Canonical analysis intentionally publishes these metrics to
            # three decimal places.  Compare at exactly that serialization
            # precision while still rejecting a one-unit change in the last
            # published digit.
            report_atol = 0.0005000001
            np.testing.assert_allclose(
                observed[task]["j_excess"],
                readouts["eventual_exit_jacobian_lens"]["excess_pass10"],
                rtol=0.0,
                atol=report_atol,
            )
            np.testing.assert_allclose(
                observed[task]["logit_excess"],
                readouts["logit_lens"]["excess_pass10"],
                rtol=0.0,
                atol=report_atol,
            )
            exit_key = "multihop" if task == "multihop" else "order-ops numeric"
            np.testing.assert_allclose(
                observed[task]["exit_agreement_all_items"],
                main["lens_free_exits"][exit_key]["exit_top1_equals_final_top1_all_items"],
                rtol=0.0,
                atol=report_atol,
            )
        accuracy = main.get("corrected_model_accuracy")
        _require(
            isinstance(accuracy, dict)
            and accuracy.get("n_correct_all_items") == correct
            and accuracy.get("n_items") == len(items),
            "canonical analysis corrected model accuracy disagrees with inputs",
        )
    return observed


def verify_probe(path: Path, summary: dict | None = None) -> dict:
    arrays = dict(np.load(path, allow_pickle=False))
    labels, folds = arrays["labels"], arrays["folds"]
    selection = np.asarray(arrays["selection_accuracy"], dtype=float)
    _require(
        selection.shape == (5, N_UT * N_LAYER)
        and np.isfinite(selection).all()
        and not np.any((selection < 0) | (selection > 1)),
        "probe inner-validation selection scores are malformed",
    )
    chosen_c = np.asarray(arrays["chosen_C"])
    _require(
        chosen_c.shape == selection.shape and np.isin(chosen_c, [0.01, 0.1, 1.0]).all(),
        "probe C selections are missing or outside the frozen grid",
    )
    pairs = [(a, b) for a in range(1, 10) for b in range(a, 10)]
    order = np.random.default_rng(0).permutation(len(pairs))
    assignment = {pairs[index]: int(position % 5) for position, index in enumerate(order)}
    expected_labels = np.asarray([
        a + b for a in range(1, 10) for b in range(1, 10) for _ in range(2, 10)
    ])
    expected_folds = np.asarray([
        assignment[(min(a, b), max(a, b))]
        for a in range(1, 10) for b in range(1, 10) for _ in range(2, 10)
    ])
    _require(
        labels.shape == (648,) and np.array_equal(labels, expected_labels),
        "probe label order differs from the frozen arithmetic family",
    )
    _require(
        folds.shape == (648,) and np.array_equal(folds, expected_folds),
        "probe folds differ from the frozen unordered-pair assignment",
    )
    eligible = np.zeros(len(labels), bool)
    for fold in range(5):
        eligible |= (folds == fold) & np.isin(labels, np.unique(labels[folds != fold]))
    _require(int(eligible.sum()) == 576, "probe eligible population changed")
    summary_names = {
        "probe_rank": "supervised_probe",
        "ll_cand": "logit_lens",
        "jl_cand": "eventual_exit_jacobian_lens",
    }
    out = {}
    for key, public in summary_names.items():
        _require(key in arrays, f"probe array {key} is missing")
        _require(
            arrays[key].shape == (648, N_UT * N_LAYER)
            and np.issubdtype(arrays[key].dtype, np.integer)
            and arrays[key].min() >= 0
            and arrays[key].max() < 17,
            f"probe array {key} has invalid shape, dtype, or rank values",
        )
        hit = arrays[key] == 0
        values = []
        selected_by_loop = []
        for loop in range(N_UT):
            numerator = denominator = 0
            selected = []
            for fold in range(5):
                test = eligible & (folds == fold)
                if key == "probe_rank":
                    # Inner-validation accuracy for this outer fold was
                    # produced without fitting or selecting on that fold.
                    layer = int(selection[fold, loop * N_LAYER:(loop + 1) * N_LAYER].argmax())
                else:
                    train = eligible & (folds != fold)
                    layer = int(hit[train, loop * N_LAYER:(loop + 1) * N_LAYER].mean(0).argmax())
                selected.append(layer)
                numerator += int(hit[test, loop * N_LAYER + layer].sum())
                denominator += int(test.sum())
            values.append(numerator / denominator)
            selected_by_loop.append(selected)
        out[key] = {"per_loop": values, "selected_physical_layer_by_heldout_fold": selected_by_loop}
        if summary is not None:
            _require(isinstance(summary, dict) and isinstance(summary.get("readouts"), dict),
                     "probe summary readouts are missing")
            claimed = summary["readouts"].get(public)
            _require(isinstance(claimed, dict), f"probe summary readout {public} is missing")
            np.testing.assert_allclose(values, claimed.get("per_loop"), atol=5e-7)
            _require(
                claimed.get("selected_physical_layer_by_heldout_fold") == selected_by_loop,
                f"probe summary selected layers for {public} disagree with held-out calculation",
            )
    _require(np.bincount(labels).max() / len(labels) == 1 / 9, "full-cohort majority baseline changed")
    _require(
        np.bincount(labels[eligible]).max() / eligible.sum() == 1 / 8,
        "eligible-cohort majority baseline changed",
    )
    if summary is not None:
        population = summary.get("population")
        _require(
            isinstance(population, dict)
            and population.get("total_prompts") == len(labels)
            and population.get("eligible_prompts") == int(eligible.sum()),
            "probe summary population disagrees with frozen design",
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", default="artifacts/jlens/eval/fitsize_n80")
    parser.add_argument("--analysis", default="artifacts/jlens/final/analysis.json")
    parser.add_argument("--probe", default="artifacts/jlens/probe/cv_all648/arrays.npz")
    parser.add_argument("--probe-summary", default="artifacts/jlens/probe/cv_all648/summary.json")
    parser.add_argument("--transport", default="artifacts/jlens/final/transport.json")
    parser.add_argument("--out", default="artifacts/jlens/final/verification.json")
    args = parser.parse_args()
    output_path = Path(args.out)
    # A failed rerun must not leave a prior PASS-shaped artifact in place.
    _invalidate_output(output_path)
    main_root = Path(args.main)
    analysis_path = Path(args.analysis)
    probe_path = Path(args.probe)
    probe_summary_path = Path(args.probe_summary)
    transport_path = Path(args.transport)
    try:
        transport = json.loads(transport_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"transport input is unavailable or invalid: {transport_path}") from exc
    try:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        probe_summary = json.loads(probe_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError("canonical analysis or probe summary is unavailable or invalid") from exc
    transport_result = verify_transport(transport)
    result = {
        "schema_version": 1,
        "status": "MECHANICAL_ARITHMETIC_PASS",
        "scope": (
            "independent arithmetic recomputation from retained sufficient statistics and frozen designs; "
            "not provenance, semantic review, external replication, or claim acceptance"
        ),
        "main": verify_main(main_root, analysis),
        "probe": verify_probe(probe_path, probe_summary),
        "transport": transport_result,
        "inputs": [
            _logical_record(main_root / "arrays.npz", "main/arrays.npz"),
            _logical_record(main_root / "items.json", "main/items.json"),
            _logical_record(main_root / "task_names.json", "main/task_names.json"),
            _logical_record(analysis_path, "analysis.json"),
            _logical_record(probe_path, "probe/arrays.npz"),
            _logical_record(probe_summary_path, "probe/summary.json"),
            _logical_record(transport_path, "transport.json"),
        ],
    }
    atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
