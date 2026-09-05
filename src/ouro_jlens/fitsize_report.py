"""Fit-size report using both thresholded hits and the ranks below the floor."""

from __future__ import annotations

import argparse
import json
import os
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
    values = np.asarray(values, dtype=float)
    if (isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0
            or values.ndim != 2 or not len(values) or not np.isfinite(values).all()):
        raise ValueError("fit-size bootstrap requires finite item vectors and positive draws")
    rng = np.random.default_rng(seed)
    boot = np.stack([values[rng.integers(0, len(values), len(values))].mean(0) for _ in range(draws)])
    return np.percentile(boot, [2.5, 97.5], axis=0).T.round(3).tolist()


def _logical_record(path: Path, logical_path: str) -> dict:
    record = file_record(path)
    record["path"] = logical_path
    return record


def _fit_identity(provenance: dict, declared_n: int) -> dict:
    lenses = provenance.get("lens_inputs")
    if not isinstance(lenses, list) or len(lenses) != 1:
        raise ValueError("fit-size evaluation must contain exactly one eventual-exit lens")
    entry = lenses[0]
    identity = entry.get("identity") if isinstance(entry, dict) else None
    prompt_slice = identity.get("prompt_slice") if isinstance(identity, dict) else None
    if (entry.get("target_ut") != 3 or identity.get("target_ut") != 3
            or not isinstance(prompt_slice, dict)
            or prompt_slice.get("start") != 0
            or prompt_slice.get("end") != declared_n
            or prompt_slice.get("count") != declared_n
            or identity.get("n_requested") != declared_n
            or identity.get("n_fitted") != declared_n
            or identity.get("n_prompts") != declared_n):
        raise ValueError(
            f"fit-size label n={declared_n} is not authenticated by the evaluation lens"
        )
    digest = prompt_slice.get("sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or digest != identity.get("prompt_slice_sha256")):
        raise ValueError(f"fit-size n={declared_n} prompt-slice identity is malformed")
    return identity


def _population_identity(provenance: dict) -> dict:
    config = provenance.get("config")
    inputs = provenance.get("inputs")
    outputs = provenance.get("outputs")
    model = provenance.get("model")
    if not all(isinstance(value, dict) for value in (config, inputs, outputs, model)):
        raise ValueError("evaluation provenance is missing population identity")
    if (config.get("tasks") != ["multihop", "order-ops"]
            or config.get("position") != -1
            or config.get("prompt_policy") != "identical"
            or config.get("max_intermediates") != 3
            or config.get("max_names") != 128
            or config.get("greedy_steps") != 4):
        raise ValueError("fit-size evaluation config differs from the frozen experiment")
    try:
        result = {
            "config": config,
            "evaluation_input_sha256": inputs["evaluation_input_sha256"],
            "items_sha256": outputs["items"]["sha256"],
            "task_names_sha256": outputs["task_names"]["sha256"],
            "item_metadata_sha256": provenance["item_metadata_sha256"],
            "item_count": provenance["item_count"],
            "correct_count": provenance["correct_count"],
            "model_snapshot_sha256": provenance["model_snapshot_sha256"],
            "model_shape": {
                key: model[key] for key in ("revision", "n_physical", "n_ut", "n_layers", "d_model")
            },
            "jlens_commit": provenance["jlens_commit"],
            "source_sha256": provenance["source_sha256"],
        }
        if (result["model_shape"]["n_physical"] != 48
                or result["model_shape"]["n_ut"] != 4
                or result["model_shape"]["n_layers"] != 192):
            raise ValueError("fit-size evaluation model topology differs from the frozen experiment")
        return result
    except (KeyError, TypeError) as exc:
        raise ValueError("evaluation provenance population identity is incomplete") from exc


def _fit_shared_identity(identity: dict) -> dict:
    excluded = {"prompt_slice", "prompt_slice_sha256", "n_requested", "n_fitted", "n_prompts"}
    return {key: value for key, value in identity.items() if key not in excluded}


def build_report(root: Path, sizes: list[int] = SIZES) -> dict:
    if (not isinstance(sizes, list) or len(sizes) < 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in sizes)
            or len(set(sizes)) != len(sizes) or sizes != sorted(sizes)):
        raise ValueError("fit sizes must be at least two distinct increasing positive integers")
    evaluations = {n: analyze.Eval(root / f"fitsize_n{n}") for n in sizes}
    provenances = {n: evaluations[n].provenance for n in sizes}
    if any(not isinstance(value, dict) for value in provenances.values()):
        raise ValueError("fit-size evaluations require complete current provenance")
    fit_identities = {n: _fit_identity(provenances[n], n) for n in sizes}
    population_identities = {n: _population_identity(provenances[n]) for n in sizes}
    reference_population = population_identities[sizes[0]]
    if any(identity != reference_population for identity in population_identities.values()):
        raise ValueError("fit-size evaluations do not use the exact same population/model/code")
    reference_fit = _fit_shared_identity(fit_identities[sizes[0]])
    if any(_fit_shared_identity(identity) != reference_fit for identity in fit_identities.values()):
        raise ValueError("fit-size lenses differ in model, corpus, or fitting contract")
    if len({identity["prompt_slice"]["sha256"] for identity in fit_identities.values()}) != len(sizes):
        raise ValueError("fit-size prompt-prefix identities are not distinct")
    inputs = {}
    for n in sizes:
        directory = root / f"fitsize_n{n}"
        inputs[str(n)] = {
            name: _logical_record(directory / filename, f"fitsize_n{n}/{filename}")
            for name, filename in (
                ("arrays", "arrays.npz"),
                ("items", "items.json"),
                ("task_names", "task_names.json"),
                ("provenance", "provenance.json"),
            )
        }
    report = {
        "schema_version": 1,
        "status": "LARGE_FIT_INCONCLUSIVE",
        "evidence_status": "CURRENT_PROVENANCE_VERIFIED_NESTED_PREFIXES",
        "fit_sizes": sizes,
        "design": "authenticated nested prompt prefixes; correlated sizes; no fixed-n replicate variance",
        "inputs": inputs,
        "population_identity": reference_population,
        "shared_fit_identity": reference_fit,
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
    first, last = report["fit_sizes"][0], report["fit_sizes"][-1]
    lines = [
        "# Fit-size sensitivity",
        "",
        f"Fits are authenticated nested prefixes ({', '.join(map(str, report['fit_sizes']))}), "
        "not independent fixed-size replicates. "
        f"The sweep therefore establishes sensitivity only through {last} prompts, not large-fit behavior.",
        "",
        "| population | fit n | loop 1 | loop 2 | loop 3 | loop 4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for population, data in report["populations"].items():
        for n, values in data["by_fit_size"].items():
            row = values["excess_pass10_any_layer"]
            lines.append(f"| {population} | {n} | " + " | ".join(f"{v:+.3f}" for v in row) + " |")
    lines.extend(["", f"Mean improvement in log10(rank+1), n={first} to n={last} (positive is better):", ""])
    for population, data in report["populations"].items():
        delta = data[f"n{last}_minus_n{first}_mean_log10_rank_improvement"]
        ci = data["improvement_ci95"]
        text = ", ".join(f"L{i+1} {v:+.3f} [{bounds[0]:+.3f}, {bounds[1]:+.3f}]"
                         for i, (v, bounds) in enumerate(zip(delta, ci)))
        lines.append(f"- {population}: {text}")
    return "\n".join(lines) + "\n"


def _invalidate_products(paths: list[Path]) -> None:
    for path in paths:
        lexical = Path(os.path.abspath(path))
        for candidate in (lexical, *lexical.parents):
            if candidate.is_symlink():
                raise ValueError(f"fit-size output traverses a link: {candidate}")
        if lexical.is_dir():
            raise ValueError(f"fit-size output is a directory: {lexical}")
        lexical.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/jlens/eval")
    parser.add_argument("--out", default="artifacts/jlens/final/fit_size.json")
    parser.add_argument("--markdown", default="artifacts/jlens/eval/fitsize_summary.md")
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    args = parser.parse_args()
    output, markdown_path = Path(args.out), Path(args.markdown)
    status_path = output.with_suffix(".status.json")
    _invalidate_products([output, markdown_path])
    atomic_write_json(status_path, {"schema_version": 1, "status": "RUNNING_INCOMPLETE"})
    try:
        report = build_report(Path(args.root), list(args.sizes))
        atomic_write_json(output, report)
        atomic_write_text(markdown_path, markdown(report))
    except BaseException as exc:
        _invalidate_products([output, markdown_path])
        atomic_write_json(status_path, {
            "schema_version": 1,
            "status": "FAILED_INCOMPLETE",
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    atomic_write_json(status_path, {
        "schema_version": 1,
        "status": "COMPLETE_CURRENT_PROVENANCE",
        "outputs": [
            _logical_record(output, output.name),
            _logical_record(markdown_path, markdown_path.name),
        ],
    })
    print(json.dumps({"status": report["status"], "populations": report["populations"]}, indent=2))


if __name__ == "__main__":
    main()
