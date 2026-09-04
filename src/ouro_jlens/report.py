"""Build the canonical, machine-readable JLens result from retained artifacts.

This module deliberately separates measurements from claim status.  It never
turns a missing control, an old GPU transcript, or an interval touching zero
into completion.  The JSON output is the authority used by the concise docs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ouro_jlens import analyze
from ouro_jlens.evidence import atomic_write_json, file_record


CLAIMS = {
    "instrumentation": {
        "status": "PENDING_FRESH_VALIDATION",
        "claim": "Virtual recurrent locations and VJP plumbing require a fresh current-source validation run.",
    },
    "multihop_relative_deficit": {
        "status": "SUPPORTED_LOCAL_UNFROZEN",
        "claim": (
            "On the retained n=80 artifact, the final-exit average Jacobian lens is below the "
            "logit lens for task-defined multihop intermediate-token readout at loops 1-3."
        ),
    },
    "arithmetic_relative_deficit": {
        "status": "SUPPORTED_LOOP1_ONLY",
        "claim": "The relative deficit is resolved at loop 1; loops 2-4 are inconclusive.",
    },
    "local_eventual_convergence": {
        "status": "REFUTED_PRE_FINAL",
        "claim": "The specified local/eventual gap does not shrink monotonically over loops 1-3.",
    },
    "lens_free_exit_agreement": {
        "status": "SUPPORTED_LOCAL_UNFROZEN",
        "claim": "Item-weighted exit agreement is descriptive on the retained 93/55-stimulus populations.",
    },
    "large_fit_robustness": {
        "status": "INCONCLUSIVE",
        "claim": "Nested fits through n=80 do not establish behavior at n=1000 or fixed-n variability.",
    },
    "cross_loop_association": {
        "status": "SUPPORTED_ASSOCIATION_MULTIHOP",
        "claim": (
            "At n=32 and n=80 the multihop transfer matrix is dominated by the loop where "
            "the Jacobian was fitted; this is an association, not a causal mechanism."
        ),
    },
    "transport_mechanism": {
        "status": "INCONCLUSIVE",
        "claim": "Norm/scatter patterns are consistent with cancellation but do not establish prompt-specific rewriting.",
    },
    "supervised_probe": {
        "status": "SUPPORTED_LOCAL_UNFROZEN",
        "claim": (
            "Cross-fitted point estimates and pair-cluster intervals are retained for the "
            "576-prompt fold-trainable population; they are not results on all 648 prompts."
        ),
    },
    "probe_familywide_inference": {
        "status": "INCONCLUSIVE_UNADJUSTED_INTERVALS",
        "claim": "Probe intervals are pointwise and do not establish a multiplicity-adjusted family-wide result.",
    },
    "checkpoint_comparison": {
        "status": "SUPPORTED_LOCAL_UNFROZEN",
        "claim": "Retained checkpoint summaries show distributional convergence without categorical alignment.",
    },
    "estimator_independence": {
        "status": "INCONCLUSIVE",
        "claim": "The matched 2x2 position-by-reduction control has not been run.",
    },
    "architecture_cause": {
        "status": "PLAUSIBLE_HYPOTHESIS",
        "claim": "Shared-head recurrent supervision is a plausible contributor, not a demonstrated cause.",
    },
    "b300_validation": {
        "status": "NOT_RETAINED",
        "claim": "No hash-bound B300 validation or fit artifact survived the paid run.",
    },
    "submission_readiness": {
        "status": "NOT_ESTABLISHED",
        "claim": "The local observational result is not a completed mechanism study.",
    },
}


def _record_hashes(value: object) -> dict[str, str]:
    """Collect path-to-digest bindings from nested file records."""

    found: dict[str, str] = {}
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            found[value["path"]] = value["sha256"]
        for child in value.values():
            found.update(_record_hashes(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_record_hashes(child))
    return dict(sorted(found.items()))


def _complete_claim_records(report: dict) -> None:
    """Attach the evidence contract to every claim, including negative claims.

    A status-only table invites prose to drift away from the numbers.  Every
    record therefore has the same machine-checkable fields; unavailable values
    stay explicit ``null`` rather than being replaced by suggestive prose.
    """

    claims = report["claims"]
    generator_hash = report["generator"]["sha256"]
    for record in claims.values():
        record.update({
            "population": None,
            "estimand": None,
            "value": None,
            "uncertainty": None,
            "input_hashes": {},
            "generator_sha256": generator_hash,
        })

    main = report["main"]
    main_hashes = _record_hashes(main["input"])
    for claim_name, population_name in (
        ("multihop_relative_deficit", "multihop"),
        ("arithmetic_relative_deficit", "order_ops_numeric"),
    ):
        population = main["populations"][population_name]
        paired = population["paired_j_minus_logit"]
        claims[claim_name].update({
            "population": {
                key: population[key] for key in (
                    "n_raw_items", "n_clean_items", "n_clean_slots"
                )
            },
            "estimand": main["estimand"],
            "value": paired["excess_pass10_per_loop"],
            "uncertainty": {"ci95_unadjusted": paired["ci95_unadjusted"]},
            "input_hashes": main_hashes,
        })

    claims["lens_free_exit_agreement"].update({
        "population": {"multihop": 93, "order_ops": 55},
        "estimand": "fraction of unique items whose exit-k top token equals the final-exit top token",
        "value": main["lens_free_exits"],
        "input_hashes": main_hashes,
    })
    claims["local_eventual_convergence"].update({
        "population": "retained round1_exit3x32 multihop population",
        "estimand": "local-exit versus eventual-exit readout divergence at pre-final loops",
        "value": report["local_vs_eventual"]["rows"],
        "input_hashes": _record_hashes(report["local_vs_eventual"]),
    })

    if "validation" in report:
        validation = report["validation"]
        claims["instrumentation"].update({
            "population": "current source, fixed local Ouro snapshot",
            "estimand": "M1-M5 mechanical validation gate",
            "value": {
                "numerical_pass": validation.get("numerical_pass"),
                "bit_exact": validation.get("bit_exact"),
            },
            "input_hashes": _record_hashes(validation),
        })
    if "fit_size" in report:
        fit_size = report["fit_size"]
        multihop = fit_size["populations"]["multihop"]
        claims["large_fit_robustness"].update({
            "population": {"n_items": multihop["n_items"], "nested_fit_sizes": fit_size["fit_sizes"]},
            "estimand": "n=8 to n=80 change in item-mean log10(best-rank+1); positive is improvement",
            "value": multihop["n80_minus_n8_mean_log10_rank_improvement"],
            "uncertainty": {"ci95_unadjusted": multihop["improvement_ci95"]},
            "input_hashes": _record_hashes(fit_size),
        })
    if "cross_loop" in report:
        claims["cross_loop_association"].update({
            "population": "retained n=32 and n=80 clean multihop and arithmetic items",
            "estimand": "any-layer excess-hit@10 matrix indexed by fit loop and state loop",
            "value": report["cross_loop"]["fits"],
            "input_hashes": _record_hashes(report["cross_loop"]),
        })
    if "transport" in report:
        transport = report["transport"]
        claims["transport_mechanism"].update({
            "population": {"sample_sizes": transport.get("sample_sizes"),
                           "n_sources": transport.get("n_sources")},
            "estimand": transport.get("model"),
            "value": {
                "per_loop": transport.get("per_loop"),
                "negative_sigma_squared_total": transport.get("negative_sigma_squared_total"),
            },
            "input_hashes": _record_hashes(transport),
        })
    if "probe" in report:
        probe = report["probe"]
        probe_fields = {
            "population": probe.get("population"),
            "estimand": probe.get("estimand"),
            "value": {"readouts": probe.get("readouts"), "baselines": probe.get("baselines")},
            "uncertainty": {"contrasts": probe.get("contrasts"), "bootstrap": probe.get("bootstrap")},
            "input_hashes": _record_hashes(probe),
        }
        claims["supervised_probe"].update(probe_fields)
        claims["probe_familywide_inference"].update(probe_fields)
    if "checkpoints" in report:
        checkpoints = report["checkpoints"]
        measured = checkpoints.get("checkpoints")
        if measured is None:
            # The retained pre-repair artifact used checkpoint names as its
            # top-level keys.  Preserve it explicitly without pretending that
            # it satisfies the newer COMPLETE/INCOMPLETE schema.
            measured = {
                name: value for name, value in checkpoints.items()
                if name not in {"_record", "schema_version", "status", "missing"}
            }
        claims["checkpoint_comparison"].update({
            "population": "retained base, Thinking, and RLTT checkpoint evaluation stimuli",
            "estimand": "exit-to-final KL, JS, entropy, final-token rank, and top-1 agreement",
            "value": measured,
            "input_hashes": _record_hashes(checkpoints),
        })


def _ci(values: np.ndarray) -> list[list[float]]:
    # ``analyze.boot_ci`` is scalar-oriented; calculate vector intervals here.
    rng = np.random.default_rng(analyze.BOOT_SEED)
    draws = np.stack([
        values[rng.integers(0, len(values), len(values))].mean(0)
        for _ in range(analyze.N_BOOT)
    ])
    return np.percentile(draws, [2.5, 97.5], axis=0).T.round(3).tolist()


def main_readout(eval_dir: Path) -> tuple[dict, analyze.Eval]:
    ev = analyze.Eval(eval_dir)
    out = {
        "input": {
            name: file_record(eval_dir / name)
            for name in ("arrays.npz", "items.json", "task_names.json")
        },
        "fit_prompts": 80,
        "estimand": (
            "item-mean of own-name hit@10 at any layer in a loop minus the mean of the "
            "same any-layer statistic over matched control names"
        ),
        "bootstrap": {"unit": "unique_item", "draws": analyze.N_BOOT, "seed": analyze.BOOT_SEED},
        "populations": {},
    }
    task_specs = (("multihop", "multihop"), ("order_ops_numeric", "order-ops numeric"))
    for public_name, mask_name in task_specs:
        mask = ev.slot_mask[mask_name]
        ii, _ = ev.own_slots(mask)
        raw_task = "order-ops" if mask_name.startswith("order-ops") else mask_name
        raw_count = sum(item["task"] == raw_task for item in ev.items)
        population = {
            "n_raw_items": raw_count,
            "n_clean_items": int(len(np.unique(ii))),
            "n_clean_slots": int(len(ii)),
            "n_clean_model_correct_items": int(len(np.unique(ev.own_slots(
                ev.slot_mask[f"{mask_name} (model correct)"]
            )[0]))),
            "readouts": {},
        }
        item_values = {}
        for label, key in (
            ("eventual_exit_jacobian_lens", "jlens_exit3_allrank"),
            ("logit_lens", "logitlens_allrank"),
        ):
            values = analyze.any_layer_items(ev.scores(ev.arrays[key], mask))
            item_values[label] = values
            population["readouts"][label] = {
                metric: array.mean(0).round(3).tolist()
                for metric, array in values.items()
            }
        delta = item_values["eventual_exit_jacobian_lens"]["excess_pass10"] - item_values["logit_lens"]["excess_pass10"]
        population["paired_j_minus_logit"] = {
            "excess_pass10_per_loop": delta.mean(0).round(3).tolist(),
            "ci95_unadjusted": _ci(delta),
        }
        out["populations"][public_name] = population
    out["corrected_model_accuracy"] = {
        "n_correct_all_items": int(sum(item["correct"] for item in ev.items)),
        "n_items": len(ev.items),
        "definition": "boundary-aware prefix match recomputed from retained continuation and target",
    }
    out["lens_free_exits"] = analyze.model_exits(ev)
    return out, ev


def local_eventual(local_eval_dir: Path) -> dict:
    ev = analyze.Eval(local_eval_dir)
    arrays = ev.arrays
    rows = []
    for loop in range(analyze.N_UT - 1):
        block = slice(loop * analyze.N_LAYER, (loop + 1) * analyze.N_LAYER)
        local_key = f"jlens_exit{loop}"
        if f"{local_key}_allrank" not in arrays:
            raise ValueError(f"missing {local_key} in {local_eval_dir}")
        local_values = analyze.loc_maps(ev.scores(
            arrays[f"{local_key}_allrank"], ev.slot_mask["multihop"]
        ))["excess"][loop]
        eventual_values = analyze.loc_maps(ev.scores(
            arrays["jlens_exit3_allrank"], ev.slot_mask["multihop"]
        ))["excess"][loop]
        rows.append({
            "loop": loop + 1,
            "mean_kl_local_to_eventual": round(float(
                arrays[f"{local_key}_kl_to_eventual_readout"][:, block].mean()
            ), 3),
            "top1_agreement_last8_layers": round(float((
                arrays[f"{local_key}_top1"][:, block][:, -8:]
                == arrays["jlens_exit3_top1"][:, block][:, -8:]
            ).mean()), 3),
            "multihop_mean_layer_excess_local": round(float(local_values.mean()), 3),
            "multihop_mean_layer_excess_eventual": round(float(eventual_values.mean()), 3),
            "local_minus_eventual": round(float((local_values - eventual_values).mean()), 3),
        })
    return {
        "status": "REFUTED_PRE_FINAL",
        "fit_note": "local-exit lenses use n=24; eventual-exit lens in this artifact uses n=32",
        "rows": rows,
        "input": file_record(local_eval_dir / "arrays.npz"),
    }


def cross_loop_by_fit_size(eval_dirs: dict[int, Path]) -> dict:
    result = {
        "status": "SUPPORTED_ASSOCIATION_MULTIHOP_ARITHMETIC_INCONCLUSIVE",
        "interpretation": (
            "Rows index the loop where J was fitted and columns the loop producing the state. "
            "A row-dominated matrix is descriptive and does not identify remaining horizon as a cause."
        ),
        "fits": {},
    }
    for n, directory in sorted(eval_dirs.items()):
        ev = analyze.Eval(directory)
        result["fits"][str(n)] = {
            "input": file_record(directory / "arrays.npz"),
            "multihop": analyze.cross_loop_summary(ev, ev.slot_mask["multihop"]),
            "order_ops_numeric": analyze.cross_loop_summary(
                ev, ev.slot_mask["order-ops numeric"]
            ),
        }
    return result


def build_report(eval_dir: Path, local_eval_dir: Path, *, validation: Path | None = None,
                 fit_size: Path | None = None, transport: Path | None = None,
                 probe: Path | None = None,
                 checkpoints: Path | None = None) -> dict:
    main, main_ev = main_readout(eval_dir)
    claims = json.loads(json.dumps(CLAIMS))
    report = {
        "schema_version": 1,
        "overall_status": "SUPPORTED_LOCAL_OBSERVATIONAL_RESULT_UNFROZEN",
        "scope": (
            "Fixed local Ouro-2.6B snapshot and retained stimuli; task-defined intermediate-token readout "
            "under a prompt-averaged Jacobian-lens estimator"
        ),
        "main": main,
        "local_vs_eventual": local_eventual(local_eval_dir),
        "claims": claims,
        # Canonical project-relative path keeps report bytes identical in a
        # fresh accepted-state worktree when invoked from the repository root.
        "generator": file_record(Path("src/ouro_jlens/report.py")),
    }
    n32 = eval_dir.parent / "fitsize_n32"
    if main_ev is not None and n32.is_dir():
        report["cross_loop"] = cross_loop_by_fit_size({32: n32, 80: eval_dir})
    optional = {
        "validation": validation,
        "fit_size": fit_size,
        "transport": transport,
        "probe": probe,
        "checkpoints": checkpoints,
    }
    for name, path in optional.items():
        if path is not None:
            report[name] = json.loads(path.read_text())
            report[name]["_record"] = file_record(path)
    if validation is not None:
        val = report["validation"]
        if val.get("numerical_pass") is True:
            claims["instrumentation"]["status"] = "SUPPORTED_CURRENT_VALIDATION"
            claims["instrumentation"]["claim"] = (
                "Current-source validation passes M1-M5, including human loop 1 VJPs; "
                "the retained report records numerical and bit-exact status separately."
            )
        else:
            claims["instrumentation"]["status"] = "FAILED_OR_INCOMPLETE_CURRENT_VALIDATION"
    _complete_claim_records(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="artifacts/jlens/eval/fitsize_n80")
    parser.add_argument("--local-eval", default="artifacts/jlens/eval/round1_exit3x32")
    parser.add_argument("--validation")
    parser.add_argument("--fit-size", default="artifacts/jlens/final/fit_size.json")
    parser.add_argument("--transport")
    parser.add_argument("--probe")
    parser.add_argument("--checkpoints", default="artifacts/jlens/checkpoints/exit_divergence.json")
    parser.add_argument("--out", default="artifacts/jlens/final/analysis.json")
    parser.add_argument("--claims-out", default="artifacts/jlens/final/claim_status.json")
    args = parser.parse_args()
    report = build_report(
        Path(args.eval), Path(args.local_eval),
        validation=Path(args.validation) if args.validation else None,
        fit_size=Path(args.fit_size) if args.fit_size else None,
        transport=Path(args.transport) if args.transport else None,
        probe=Path(args.probe) if args.probe else None,
        checkpoints=Path(args.checkpoints) if args.checkpoints else None,
    )
    atomic_write_json(Path(args.out), report)
    atomic_write_json(Path(args.claims_out), report["claims"])
    print(json.dumps({"overall_status": report["overall_status"], "claims": report["claims"]}, indent=2))


if __name__ == "__main__":
    main()
