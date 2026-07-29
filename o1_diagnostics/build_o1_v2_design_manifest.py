#!/usr/bin/env python3
"""Resolve the non-calibration fields of the O1 v2 design manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def build(root: Path, commit: str, output: Path) -> dict:
    package = root / "o1_packages/O1_oracle_reachability_v2.0.0_source/o1_v200"
    run = root / "o1_runs/O1_V2_AXIS_BANK_REDESIGN"
    axis = run / "AXIS_PACKAGE_V2"
    template = json.loads((package / "FREEZE_MANIFEST.template.json").read_text())
    axis_manifest = json.loads((axis / "AXIS_MANIFEST.json").read_text())
    gram = json.loads((axis / "GRAM_MATRIX.json").read_text())
    diag = json.loads((run / "ACTUATOR_MEAN_DIAGNOSTICS.json").read_text())

    template["code"].update({
        "repo": "https://github.com/VykosMolt/ouro_project.git",
        "commit": commit,
        "analysis_module_sha256": file_hash(package / "o1_analysis.py"),
        "fixture_suite_sha256": file_hash(package / "test_o1_fixtures.py"),
        "python": "3.14.6",
        "numpy": "2.4.4",
        "torch": "2.12.0.dev20260407+cu128",
        "transformers": "4.54.1",
        "axis_verifier_sha256": file_hash(package / "verify_axis_artifact.py"),
        "generation_module_sha256": file_hash(package / "run_o1_v2_generation.py"),
        "calibration_module_sha256": file_hash(package / "calibration_analysis.py"),
        "repository_clean_required": True,
        "repository_diff_sha256": hashlib.sha256(b"").hexdigest(),
    })
    template["model"]["checkpoint"] = (
        "/home/moloch/ouro_project/models/ouro_rltt_local"
    )
    template["model"]["checkpoint_sha256"] = (
        "a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889"
    )
    template["model"]["checkpoint_project_relative_tree_sha256"] = (
        "7dbbe4ef91de369bd712bad587c1af4f7c5dc42deff5377d4d70086c4c08a802"
    )
    generator_hash = file_hash(package / "horizon_logic_generator_v2.py")
    template["generator"].update({
        "revision": (
            "O1_HORIZON_LOGIC_V2_PROPOSITIONAL_20260729"
            f"@sha256:{generator_hash}"
        ),
        "hash_ordered_pool_offsets_calibration": (
            "namespace=calibration; settings rules_2/rules_3/rules_4; "
            "first 32 accepted tasks per setting in deterministic nonce order"
        ),
        "hash_ordered_pool_offsets_confirmatory": (
            "namespace=confirmatory_candidate; first 800 accepted tasks per "
            "setting; final band-filtered prefix selected mechanically after calibration"
        ),
    })
    axes_by_name = {row["name"]: row for row in axis_manifest["axes"]}
    frozen_axes = template["action_space"]["structured_axes"]["axes"]
    for name in frozen_axes:
        frozen_axes[name]["sha256"] = axes_by_name[name]["sha256"]
    frozen_axes["A1_READABLE"].update({
        "training_task_manifest_sha256": (
            "NOT_SEPARATELY_MATERIALIZED; source split_integrity.json sha256 "
            "74c146e78e68daa20e2e6d129e57635141c71f19660ffc02d85ffd3123742c7d"
        ),
        "fit_code_sha256": (
            "214ab1bc5e00ff7afb0396f7abd7e0e097d9eaadd54452feac5f3e70800610dc"
        ),
    })
    frozen_axes["A2_WRITABLE"]["source_run"] = (
        "s1_s3_exact_injection_orthogonality_null_audit_2026-06-17; "
        "bundle sha256 ebb47fb80efacdf60e91f70a49d11e4229c15189bcfe74e199de6529cdad2a38"
    )
    frozen_axes["A3_CAUSAL_MEAN"].update({
        "source_matrix_sha256": diag["A3_CAUSAL_MEAN"]["source_matrix_file_sha256"],
        "reference_set_sha256": diag["reference"]["file_sha256"],
    })
    frozen_axes["A4_SEQUENCE_MEAN"].update({
        "source_matrix_sha256": diag["A4_SEQUENCE_MEAN"]["source_matrix_file_sha256"],
        "reference_set_sha256": diag["reference"]["file_sha256"],
    })
    structured = template["action_space"]["structured_axes"]
    structured["gram_matrix"] = gram["structured_gram"]
    structured["pairwise_cosines"] = gram["structured_cosine"]
    structured["norms_before_after"] = {
        "unit_rms_after": [1.0, 1.0, 1.0, 1.0],
        "A3_raw_mean_l2": diag["A3_CAUSAL_MEAN"]["raw_mean_l2_norm"],
        "A4_raw_mean_l2": diag["A4_SEQUENCE_MEAN"]["raw_mean_l2_norm"],
    }
    template["cohorts"]["calibration_task_manifest_sha256"] = file_hash(
        run / "COHORTS/calibration_tasks.jsonl"
    )
    template["cohorts"]["confirmatory_task_manifest_sha256"] = (
        "PENDING_POSTCALIBRATION_MECHANICAL_SELECTION_FROM_"
        + file_hash(run / "COHORTS/confirmatory_candidate_pool.jsonl")
    )
    template["cohorts"]["seed_matrix_sha256"] = "PENDING_SEED_MATRIX_BUILD"
    template["artifact_hashes"].update({
        "tokenizer": file_hash(run / "TOKENIZER_BINDING.json"),
        "prompt_template": file_hash(package / "o1_prompt_template_v2.py"),
        "parser": file_hash(package / "o1_answer_parser_v2.py"),
        "verifier_implementation": file_hash(package / "o1_truth_table_verifier_v2.py"),
        "structured_axis_tensor": file_hash(axis / "axes_l3_24.npy"),
        "random_axis_tensor": file_hash(axis / "random_axes_l3_24.npy"),
    })
    # sha256_tree values are filled by the caller after using the package's
    # canonical tree hasher, so this script never risks an incompatible rule.
    template["axis_artifact"]["package_sha256"] = "PENDING_CANONICAL_TREE_HASH"
    output.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
    return {
        "output": str(output),
        "sha256": file_hash(output),
        "generator_source_sha256": generator_hash,
        "unresolved_calibration_outputs": True,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(build(
        Path(a.root), a.commit, Path(a.output),
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
