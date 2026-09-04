"""Independent arithmetic verifier for the canonical JLens artifact claims.

This intentionally does not import ``analyze.py``, ``report.py``,
``probe_report.py``, or ``transport_report.py``.  It duplicates the small
amount of aggregation needed to catch common-mode generator errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ouro_jlens.evidence import atomic_write_json, file_record

N_UT, N_LAYER, K = 4, 48, 10
OPERATIONS = {"addition", "subtraction", "multiplication", "division", "mod", "squared"}


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


def verify_main(root: Path) -> dict:
    arrays = dict(np.load(root / "arrays.npz", allow_pickle=False))
    items = json.loads((root / "items.json").read_text())
    names = json.loads((root / "task_names.json").read_text())
    expected = {
        "multihop": {
            "counts": (90, 100),
            "j": np.asarray([0.003, -0.025, 0.084, 0.451]),
            "logit": np.asarray([0.358, 0.458, 0.500, 0.483]),
        },
        "order-ops": {
            "counts": (51, 51),
            "j": np.asarray([0.054, 0.273, 0.181, 0.193]),
            "logit": np.asarray([0.306, 0.152, 0.131, 0.213]),
        },
    }
    observed = {}
    for task, exp in expected.items():
        j, n_items, n_slots = independent_excess(arrays, items, names, task, "jlens_exit3_allrank")
        ll, n_items2, n_slots2 = independent_excess(arrays, items, names, task, "logitlens_allrank")
        assert (n_items, n_slots) == exp["counts"] == (n_items2, n_slots2)
        np.testing.assert_allclose(j, exp["j"], atol=5e-4)
        np.testing.assert_allclose(ll, exp["logit"], atol=5e-4)
        observed[task] = {"n_items": n_items, "n_slots": n_slots,
                          "j_excess": j.round(6).tolist(), "logit_excess": ll.round(6).tolist()}
    assert sum(boundary_correct(i["continuation"], i["target"]) for i in items) == 72
    for task, wanted in (("multihop", [0.409, 0.656, 0.785, 1.0]),
                         ("order-ops", [0.527, 0.855, 0.964, 1.0])):
        rows = np.asarray([i for i, item in enumerate(items) if item["task"] == task])
        got = [(arrays["exit_top1"][rows, u] == arrays["exit_top1"][rows, -1]).mean() for u in range(N_UT)]
        np.testing.assert_allclose(got, wanted, atol=5e-4)
        observed[task]["exit_agreement_all_items"] = np.round(got, 6).tolist()
    return observed


def verify_probe(path: Path) -> dict:
    arrays = dict(np.load(path, allow_pickle=False))
    labels, folds = arrays["labels"], arrays["folds"]
    eligible = np.zeros(len(labels), bool)
    for fold in range(5):
        eligible |= (folds == fold) & np.isin(labels, np.unique(labels[folds != fold]))
    assert len(labels) == 648 and eligible.sum() == 576
    expected = {
        "probe_rank": [0.614583, 0.590278, 0.336806, 0.236111],
        "ll_cand": [0.763889, 0.500000, 0.178819, 0.310764],
        "jl_cand": [0.392361, 0.500000, 0.147569, 0.119792],
    }
    out = {}
    for key, wanted in expected.items():
        hit = arrays[key] == 0
        values = []
        for loop in range(N_UT):
            numerator = denominator = 0
            for fold in range(5):
                train = eligible & (folds != fold)
                test = eligible & (folds == fold)
                layer = int(hit[train, loop * N_LAYER:(loop + 1) * N_LAYER].mean(0).argmax())
                numerator += int(hit[test, loop * N_LAYER + layer].sum())
                denominator += int(test.sum())
            values.append(numerator / denominator)
        np.testing.assert_allclose(values, wanted, atol=5e-7)
        out[key] = values
    assert np.bincount(labels).max() / len(labels) == 1 / 9
    assert np.bincount(labels[eligible]).max() / eligible.sum() == 1 / 8
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", default="artifacts/jlens/eval/fitsize_n80")
    parser.add_argument("--probe", default="artifacts/jlens/probe/cv_all648/arrays.npz")
    parser.add_argument("--transport", default="artifacts/jlens/final/transport.json")
    parser.add_argument("--out", default="artifacts/jlens/final/verification.json")
    args = parser.parse_args()
    transport = json.loads(Path(args.transport).read_text())
    assert transport["negative_sigma_squared_total"] == 2
    raw = [row["raw_fitted_map_norm_by_n"]["80"] for row in transport["per_loop"]]
    np.testing.assert_allclose(raw, [0.159232, 0.178018, 0.271623, 0.810091], atol=5e-7)
    mu = [row["mu_norm_mean_valid"] for row in transport["per_loop"]]
    sigma = [row["sigma_rms_scatter_mean_valid"] for row in transport["per_loop"]]
    total = [row["modeled_total_rms_mean_where_sum_nonnegative"] for row in transport["per_loop"]]
    np.testing.assert_allclose(mu, [0.133634, 0.173291, 0.269261, 0.806608], atol=5e-7)
    np.testing.assert_allclose(sigma, [0.707673, 0.324169, 0.321085, 0.746810], atol=5e-7)
    np.testing.assert_allclose(total, [0.721079, 0.368432, 0.419324, 1.102747], atol=5e-7)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "main": verify_main(Path(args.main)),
        "probe": verify_probe(Path(args.probe)),
        "transport": {"n80_raw_norm": raw, "mu_norm_valid": mu,
                      "sigma_scatter_valid": sigma, "modeled_total_rms": total,
                      "negative_sigma_squared": 2},
        "inputs": [file_record(Path(args.main) / "arrays.npz"), file_record(Path(args.probe)),
                   file_record(Path(args.transport))],
    }
    atomic_write_json(Path(args.out), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
