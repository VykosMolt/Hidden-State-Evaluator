"""Fit-size report using both thresholded hits and the ranks below the floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ouro_jlens import analyze
from ouro_jlens.evidence import atomic_write_json, atomic_write_text, file_record

SIZES = [8, 32, 56, 80]
TASKS = {"multihop": "multihop", "order_ops_numeric": "order-ops numeric"}


def _best_slot_ranks(ev: analyze.Eval, mask_name: str) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = ev.own_slots(ev.slot_mask[mask_name])
    own = ev.arrays["jlens_exit3_allrank"][ii, jj].reshape(-1, analyze.N_UT, analyze.N_LAYER)
    return own.min(axis=2), ii


def _vector_ci(values: np.ndarray, seed: int = 0,
               draws: int = analyze.N_BOOT) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    boot = np.stack([values[rng.integers(0, len(values), len(values))].mean(0) for _ in range(draws)])
    return np.percentile(boot, [2.5, 97.5], axis=0).T.round(3).tolist()


def build_report(root: Path, sizes: list[int] = SIZES) -> dict:
    evaluations = {n: analyze.Eval(root / f"fitsize_n{n}") for n in sizes}
    report = {
        "schema_version": 1,
        "status": "LARGE_FIT_INCONCLUSIVE",
        "fit_sizes": sizes,
        "design": "nested prompt prefixes; no fixed-n replicate variance",
        "inputs": {str(n): file_record(root / f"fitsize_n{n}" / "arrays.npz") for n in sizes},
        "populations": {},
    }
    for public, mask_name in TASKS.items():
        per_n, log_ranks = {}, {}
        for n, ev in evaluations.items():
            sc = ev.scores(ev.arrays["jlens_exit3_allrank"], ev.slot_mask[mask_name])
            slot_rank, item_ids = _best_slot_ranks(ev, mask_name)
            item_log_rank = analyze.by_item(np.log10(slot_rank + 1), item_ids)
            log_ranks[n] = item_log_rank
            per_n[str(n)] = {
                "excess_pass10_any_layer": analyze.any_layer_items(sc)["excess_pass10"].mean(0).round(3).tolist(),
                "slot_median_best_rank": np.median(slot_rank, axis=0).round().astype(int).tolist(),
                "item_median_geometric_best_rank": np.rint(10 ** np.median(item_log_rank, axis=0) - 1).astype(int).tolist(),
            }
        delta = log_ranks[sizes[0]] - log_ranks[sizes[-1]]
        report["populations"][public] = {
            "n_items": int(len(log_ranks[sizes[0]])),
            "by_fit_size": per_n,
            f"n{sizes[-1]}_minus_n{sizes[0]}_mean_log10_rank_improvement": delta.mean(0).round(3).tolist(),
            "improvement_ci95": _vector_ci(delta),
        }
    return report


def markdown(report: dict) -> str:
    lines = [
        "# Fit-size sensitivity",
        "",
        "Fits are nested prefixes (8, 32, 56, 80), not independent fixed-size replicates. "
        "The sweep therefore does not establish behavior at 1000 prompts.",
        "",
        "| population | fit n | loop 1 | loop 2 | loop 3 | loop 4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for population, data in report["populations"].items():
        for n, values in data["by_fit_size"].items():
            row = values["excess_pass10_any_layer"]
            lines.append(f"| {population} | {n} | " + " | ".join(f"{v:+.3f}" for v in row) + " |")
    lines.extend(["", "Mean improvement in log10(rank+1), n=8 to n=80 (positive is better):", ""])
    for population, data in report["populations"].items():
        delta = data["n80_minus_n8_mean_log10_rank_improvement"]
        ci = data["improvement_ci95"]
        text = ", ".join(f"L{i+1} {v:+.3f} [{bounds[0]:+.3f}, {bounds[1]:+.3f}]"
                         for i, (v, bounds) in enumerate(zip(delta, ci)))
        lines.append(f"- {population}: {text}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/jlens/eval")
    parser.add_argument("--out", default="artifacts/jlens/final/fit_size.json")
    parser.add_argument("--markdown", default="artifacts/jlens/eval/fitsize_summary.md")
    args = parser.parse_args()
    report = build_report(Path(args.root))
    atomic_write_json(Path(args.out), report)
    atomic_write_text(Path(args.markdown), markdown(report))
    print(json.dumps({"status": report["status"], "populations": report["populations"]}, indent=2))


if __name__ == "__main__":
    main()
