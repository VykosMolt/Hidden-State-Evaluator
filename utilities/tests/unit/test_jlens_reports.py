from __future__ import annotations

import json
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import jlens

from ouro_jlens import lens_convergence
from ouro_jlens.fit_lens import IntegrityError
from ouro_jlens.probe import make_prompts
from ouro_jlens.probe_cv import prompt_folds
from ouro_jlens.probe_report import cross_fitted_point, design
from ouro_jlens.analyze import boundary_prefix_match
from ouro_jlens.transport_report import estimate_from_squared_norms, summarize
from ouro_jlens import evidence
from ouro_jlens import manifest, publish as publish_module, report as report_module
from ouro_jlens.report import (
    build_report,
    checkpoint_claim_text,
    classify_loop_effect,
    derive_loop_decisions,
)
from ouro_jlens.validate import derive_milestone_passes, paid_validation_accepted


def _probe_design_arrays():
    prompts = make_prompts()
    return {
        "labels": np.asarray([q["label"] for q in prompts]),
        "folds": prompt_folds(prompts, 0),
    }


def test_boundary_prefix_match_rejects_alphanumeric_extensions():
    assert boundary_prefix_match(" Atlantic Ocean", "Atlantic")
    assert boundary_prefix_match(" day.", "day")
    assert not boundary_prefix_match("11", "1")
    assert not boundary_prefix_match("daytime", "day")


def test_probe_design_exposes_the_exact_label_coverage_population():
    d = design(_probe_design_arrays(), seed=0)
    labels, eligible = d["labels"], d["eligible"]
    assert len(labels) == 648
    assert int(eligible.sum()) == 576
    assert len(d["all_clusters"]) == 45
    assert len(d["eligible_clusters"]) == 39
    assert len(d["excluded_clusters"]) == 6
    assert np.bincount(labels).max() / len(labels) == 1 / 9
    assert np.bincount(labels[eligible]).max() / eligible.sum() == 1 / 8


def test_cross_fitted_layer_selection_never_scores_its_selection_fold():
    arrays = _probe_design_arrays()
    d = design(arrays, seed=0)
    ranks = np.ones((648, 192), dtype=np.int32)
    # Each held-out fold has a different perfect layer. Selection on the other
    # folds must not simply recover that fold's private layer.
    for fold in range(5):
        rows = d["folds"] == fold
        ranks[rows, fold] = 0
    values, choices = cross_fitted_point(ranks, d["folds"], d["eligible"])
    assert values[0] < 1.0
    assert all(0 <= layer < 48 for layer in choices[0])
    assert values.shape == (4,)


def test_transport_regression_reports_negative_moments_instead_of_hiding_them():
    n = np.asarray([8, 32, 56, 80])
    mu2 = np.full(96, 0.25)
    sigma2 = np.full(96, 0.5)
    sigma2[-1] = -0.1
    y = np.stack([mu2 + sigma2 / size for size in n])
    fitted = estimate_from_squared_norms(n, y)
    assert np.allclose(fitted["mu_squared"], mu2)
    assert np.allclose(fitted["sigma_squared"], sigma2)
    report = summarize(n, y)
    assert report["negative_sigma_squared_total"] == 1
    assert set(report["per_source"]["squared_norms_by_n"]) == {"8", "32", "56", "80"}
    assert np.allclose(report["per_source"]["squared_norms_by_n"]["32"], y[1])
    assert report["per_loop"][1]["valid_moment_locations"] == 47
    assert report["per_loop"][1]["negative_total_squared"] == 0
    assert report["per_loop"][1]["legacy_zero_clipped_sigma_mean"] != report["per_loop"][1]["sigma_rms_scatter_mean_valid"]


def _minimal_valid_main(tmp_path, *, fit_prompts=None):
    records = {}
    for name in ("arrays.npz", "items.json", "task_names.json"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        records[name] = evidence.file_record(path)
    return {
        "input": records,
        "fit_prompts": fit_prompts,
        "estimand": "test estimand",
        "populations": {
            name: {
                "n_raw_items": 1,
                "n_clean_items": 1,
                "n_clean_slots": 1,
                "paired_j_minus_logit": {
                    "excess_pass10_per_loop": [0, 0, 0, 0],
                    "ci95_unadjusted": [[0, 0]] * 4,
                },
            }
            for name in ("multihop", "order_ops_numeric")
        },
        "lens_free_exits": {},
    }


def _current_validation_copy(tmp_path, monkeypatch):
    payload = json.loads(Path("artifacts/jlens/validation/milestones.json").read_text())
    # Keep this fixture tied to the current source bytes even if a neighboring
    # implementation test edits evidence.py during the same checkout.
    source_files = []
    for name in ("validate.py", "recurrent.py", "evidence.py"):
        source_files.append(evidence.file_record(Path("src/ouro_jlens") / name))
    payload["provenance"]["source_files"] = source_files
    payload["provenance"]["source_sha256"] = evidence.aggregate_sha256(
        {row["path"]: row["sha256"] for row in source_files}
    )
    payload["provenance"]["package_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "jlens", "numpy", "safetensors", "accelerate", "huggingface-hub")
    }
    payload["status"] = "NUMERICAL_AND_PROVENANCE_PASS"
    payload["provenance"]["status"] = "HASH_BOUND"
    payload["provenance"]["image_digest"] = None
    monkeypatch.delenv("JLENS_IMAGE_DIGEST", raising=False)
    model_root = tmp_path / "model_snapshot"
    model_root.mkdir()
    model_records = []
    for name in ("model.safetensors", "config.json", "modeling_ouro.py", "tokenizer.json"):
        path = model_root / name
        path.write_bytes(name.encode())
        record = evidence.file_record(path)
        record["path"] = f"model_snapshot/{name}"
        model_records.append(record)
    payload["provenance"]["model_bytes"] = {
        "status": "HASH_BOUND",
        "files": model_records,
        "aggregate_sha256": evidence.aggregate_sha256(
            {row["path"]: row["sha256"] for row in model_records}
        ),
    }
    jlens_root = Path(jlens.__file__).resolve().parent
    jlens_records = []
    for path in sorted(jlens_root.rglob("*.py")):
        record = evidence.file_record(path)
        record["path"] = f"jlens/{path.relative_to(jlens_root).as_posix()}"
        jlens_records.append(record)
    payload["provenance"]["jlens_source"] = {
        "files": jlens_records,
        "aggregate_sha256": evidence.aggregate_sha256(
            {row["path"]: row["sha256"] for row in jlens_records}
        ),
    }
    monkeypatch.setattr(report_module, "OURO_SNAPSHOT", model_root)
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(payload))
    return path


def test_report_reads_complete_current_validator_schema(monkeypatch, tmp_path):
    main = _minimal_valid_main(tmp_path, fit_prompts=80)
    monkeypatch.setattr("ouro_jlens.report.main_readout", lambda path: (main, None))
    monkeypatch.setattr("ouro_jlens.report.local_eventual", lambda path: {"rows": []})
    validation = _current_validation_copy(tmp_path, monkeypatch)
    result = build_report(tmp_path, tmp_path, validation=validation, checkpoints=None)
    assert result["claims"]["instrumentation"]["status"] == "SUPPORTED_CURRENT_VALIDATION"
    assert result["claims"]["b300_validation"]["status"] == "NOT_CURRENT_B300_VALIDATION"
    required = {"status", "population", "estimand", "value", "uncertainty",
                "input_hashes", "generator_sha256"}
    assert all(required <= record.keys() for record in result["claims"].values())


def test_validation_rollup_rejects_contradictory_milestone_and_primitive_fields(
    monkeypatch, tmp_path,
):
    validation = _current_validation_copy(tmp_path, monkeypatch)
    payload = json.loads(validation.read_text())
    assert all(derive_milestone_passes(payload).values())

    payload["m3_recurrent_identity"]["pass"] = False
    payload["m3_recurrent_identity"]["fire_order_ok"] = False
    info = report_module._validation_info(payload)
    assert not info["verified"]
    assert any("m3_recurrent_identity.pass" in error for error in info["errors"])


def test_validation_rollup_rejects_tampered_bit_exact_comparison(
    monkeypatch, tmp_path,
):
    validation = _current_validation_copy(tmp_path, monkeypatch)
    payload = json.loads(validation.read_text())
    payload["bit_exact_comparisons"][0]["equal"] = False
    info = report_module._validation_info(payload)
    assert not info["verified"]
    assert any("bit_exact_comparisons disagree" in error for error in info["errors"])
    assert not paid_validation_accepted(payload)


@pytest.mark.parametrize(
    ("effect", "interval", "expected"),
    [
        (-1.0, [-2.0, -0.1], "below"),
        (1.0, [0.1, 2.0], "above"),
        (0.0, [-0.1, 0.1], "inconclusive"),
        (-1.0, [-2.0, 0.1], "inconclusive"),
        ("malformed", [-2.0, -0.1], "inconclusive"),
        (-1.0, [float("nan"), -0.1], "inconclusive"),
        (1.0, [2.0, 0.1], "inconclusive"),
    ],
)
def test_loop_effect_claims_are_derived_from_effect_and_unadjusted_interval(
    effect, interval, expected,
):
    assert classify_loop_effect(effect, interval) == expected


def test_loop_effect_decisions_retain_all_four_loops_and_malformed_entries():
    assert derive_loop_decisions(
        [-1.0, 1.0, 0.0, "bad"],
        [[-2.0, -0.1], [0.1, 2.0], [-1.0, 1.0], None],
    ) == ["below", "above", "inconclusive", "inconclusive"]


def test_malformed_main_keeps_per_loop_classification_but_cannot_promote(
    monkeypatch, tmp_path,
):
    main = _minimal_valid_main(tmp_path, fit_prompts=80)
    paired = main["populations"]["multihop"]["paired_j_minus_logit"]
    paired["excess_pass10_per_loop"] = [-1.0, "bad", 1.0, 0.0]
    paired["ci95_unadjusted"] = [[-2.0, -0.1], [0.1, 2.0], [0.1, 2.0], [-1.0, 1.0]]
    monkeypatch.setattr("ouro_jlens.report.main_readout", lambda path: (main, None))
    monkeypatch.setattr("ouro_jlens.report.local_eventual", lambda path: {"rows": []})
    result = build_report(tmp_path, tmp_path, checkpoints=None)
    claim = result["claims"]["multihop_relative_deficit"]
    assert claim["status"] == "INCONCLUSIVE_MAIN_EVIDENCE_INVALID"
    assert claim["loop_decisions"] == ["below", "inconclusive", "above", "inconclusive"]


def test_cross_loop_retains_each_fit_lineage_and_degrades_stale_n32(
    monkeypatch, tmp_path,
):
    n32 = tmp_path / "fitsize_n32"
    n80 = tmp_path / "fitsize_n80"
    n32.mkdir()
    n80.mkdir()

    def lineage(directory):
        current = directory == n80
        return {
            "verified": current,
            "status": report_module.CURRENT_GENERATION if current else "INVALID_EVALUATION_PROVENANCE",
            "fit_prompts": 80 if current else None,
            "errors": [] if current else ["stale n32"],
        }

    fake_eval = SimpleNamespace(slot_mask={"multihop": "m", "order-ops numeric": "o"})
    monkeypatch.setattr(report_module, "_evaluation_info", lineage)
    monkeypatch.setattr(report_module, "_load_eval", lambda directory: fake_eval)
    monkeypatch.setattr(report_module, "file_record", lambda path: {"path": str(path), "size": 1, "sha256": "0" * 64})
    monkeypatch.setattr(report_module.analyze, "cross_loop_summary", lambda ev, mask: {"matrix": []})

    result = report_module.cross_loop_by_fit_size({32: n32, 80: n80})
    assert set(result["fits"]) == {"32", "80"}
    assert result["fits"]["32"]["lineage"]["status"] == "INVALID_EVALUATION_PROVENANCE"
    assert result["fits"]["80"]["lineage"]["status"] == report_module.CURRENT_GENERATION
    assert result["lineage_verified"] is False
    assert result["lineage_status"] == report_module.LINEAGE_UNFROZEN


def test_report_promotes_only_verified_b300_validation(monkeypatch, tmp_path):
    main = _minimal_valid_main(tmp_path, fit_prompts=80)
    monkeypatch.setattr("ouro_jlens.report.main_readout", lambda path: (main, None))
    monkeypatch.setattr("ouro_jlens.report.local_eventual", lambda path: {"rows": []})
    validation = _current_validation_copy(tmp_path, monkeypatch)
    payload = json.loads(validation.read_text())
    payload["provenance"]["cuda"]["devices"] = [{
        "index": 0,
        "name": "NVIDIA B300 SXM6",
        "capability": [10, 3],
        "total_memory": 288_000_000_000,
    }]
    payload["provenance"]["cuda"]["device_count"] = 1
    payload["provenance"]["cuda"]["available"] = True
    payload["provenance"]["image_digest"] = publish_module.RUNTIME_IMAGE
    monkeypatch.setenv("JLENS_IMAGE_DIGEST", publish_module.RUNTIME_IMAGE)
    validation.write_text(json.dumps(payload))

    result = build_report(tmp_path, tmp_path, validation=validation, checkpoints=None)
    claim = result["claims"]["b300_validation"]
    assert claim["status"] == "SUPPORTED_CURRENT_B300_VALIDATION"
    assert claim["value"]["devices"][0]["name"] == "NVIDIA B300 SXM6"


def test_report_does_not_promote_one_field_validation(monkeypatch, tmp_path):
    main = _minimal_valid_main(tmp_path, fit_prompts=80)
    monkeypatch.setattr("ouro_jlens.report.main_readout", lambda path: (main, None))
    monkeypatch.setattr("ouro_jlens.report.local_eventual", lambda path: {"rows": []})
    validation = tmp_path / "validation.json"
    validation.write_text('{"numerical_pass": true}')
    result = build_report(tmp_path, tmp_path, validation=validation, checkpoints=None)
    assert result["claims"]["instrumentation"]["status"] == "FAILED_OR_INCOMPLETE_CURRENT_VALIDATION"


def test_report_does_not_promote_failed_or_missing_milestone(monkeypatch, tmp_path):
    main = _minimal_valid_main(tmp_path, fit_prompts=80)
    monkeypatch.setattr("ouro_jlens.report.main_readout", lambda path: (main, None))
    monkeypatch.setattr("ouro_jlens.report.local_eventual", lambda path: {"rows": []})
    validation_payload = json.loads(Path("artifacts/jlens/validation/milestones.json").read_text())
    del validation_payload["m5_stock_consistency"]
    validation_payload["provenance"]["source_files"] = [
        evidence.file_record(Path("src/ouro_jlens") / name)
        for name in ("validate.py", "recurrent.py", "evidence.py")
    ]
    validation_payload["provenance"]["source_sha256"] = evidence.aggregate_sha256(
        {row["path"]: row["sha256"] for row in validation_payload["provenance"]["source_files"]}
    )
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps(validation_payload))
    result = build_report(tmp_path, tmp_path, validation=validation, checkpoints=None)
    assert result["claims"]["instrumentation"]["status"] == "FAILED_OR_INCOMPLETE_CURRENT_VALIDATION"


def test_report_downgrades_non_80_main_and_retains_generator_set(monkeypatch, tmp_path):
    main = _minimal_valid_main(tmp_path, fit_prompts=79)
    monkeypatch.setattr("ouro_jlens.report.main_readout", lambda path: (main, None))
    monkeypatch.setattr("ouro_jlens.report.local_eventual", lambda path: {"rows": []})
    result = build_report(tmp_path, tmp_path, checkpoints=None)
    claim = result["claims"]["multihop_relative_deficit"]
    assert claim["status"] == "INCONCLUSIVE_NON_80_MAIN_INPUT"
    assert "n=80" not in claim["claim"]
    paths = {row["path"] for row in claim["generator_set"]}
    assert {
        "src/ouro_jlens/report.py",
        "src/ouro_jlens/analyze.py",
        "src/ouro_jlens/evaluate.py",
        "src/ouro_jlens/evaldata.py",
    } <= paths
    assert claim["generator_set_sha256"] == evidence.aggregate_sha256(
        {row["path"]: row["sha256"] for row in claim["generator_set"]}
    )


def test_report_does_not_trust_n80_field_without_verified_evaluation_lineage(monkeypatch, tmp_path):
    main = _minimal_valid_main(tmp_path, fit_prompts=80)
    # A producer-looking fit size without its complete evaluation provenance
    # must remain observational; a directory/field alone cannot establish n=80.
    main["lineage"] = {"verified": True, "status": "CURRENT_GENERATION_EVIDENCE_VERIFIED"}
    monkeypatch.setattr("ouro_jlens.report.main_readout", lambda path: (main, None))
    monkeypatch.setattr("ouro_jlens.report.local_eventual", lambda path: {"rows": []})
    result = build_report(tmp_path, tmp_path, checkpoints=None)
    claim = result["claims"]["multihop_relative_deficit"]
    assert claim["status"] == "LOCAL_OBSERVATIONAL_ARITHMETIC_REPRODUCIBLE_LINEAGE_UNFROZEN"
    assert result["overall_status"] == "LOCAL_OBSERVATIONAL_ARITHMETIC_REPRODUCIBLE_LINEAGE_UNFROZEN"
    assert "n=80" not in claim["claim"]


def test_report_downgrades_changed_main_input_bytes(monkeypatch, tmp_path):
    main = _minimal_valid_main(tmp_path, fit_prompts=80)
    main["input"]["arrays.npz"]["sha256"] = "0" * 64
    monkeypatch.setattr("ouro_jlens.report.main_readout", lambda path: (main, None))
    monkeypatch.setattr("ouro_jlens.report.local_eventual", lambda path: {"rows": []})
    result = build_report(tmp_path, tmp_path, checkpoints=None)
    assert result["claims"]["multihop_relative_deficit"]["status"] == "INCONCLUSIVE_MAIN_EVIDENCE_INVALID"
    assert "unavailable" in result["claims"]["multihop_relative_deficit"]["claim"]


def test_report_downgrades_degraded_probe_summary(monkeypatch, tmp_path):
    main = _minimal_valid_main(tmp_path, fit_prompts=80)
    monkeypatch.setattr("ouro_jlens.report.main_readout", lambda path: (main, None))
    monkeypatch.setattr("ouro_jlens.report.local_eventual", lambda path: {"rows": []})
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"schema_version": 1, "status": "DEGRADED"}))
    result = build_report(tmp_path, tmp_path, probe=probe, checkpoints=None)
    assert result["claims"]["supervised_probe"]["status"] == "INCONCLUSIVE_PROBE_DEGRADED_OR_UNVERIFIED"
    assert result["claims"]["probe_familywide_inference"]["status"] == "INCONCLUSIVE_PROBE_DEGRADED_OR_UNVERIFIED"


def test_probe_source_validator_accepts_only_complete_mixed_manifest(monkeypatch, tmp_path):
    names = (
        "probe_cv.py", "probe.py", "evaluate.py", "evaldata.py",
        "recurrent.py", "evidence.py",
    )
    monkeypatch.setattr(report_module, "PROJECT_ROOT", tmp_path)
    source_root = tmp_path / "src" / "ouro_jlens"
    source_root.mkdir(parents=True)

    records = []
    for name in names:
        path = source_root / name
        path.write_text(f"# {name}\n", encoding="utf-8")
        record = evidence.file_record(path)
        record["path"] = f"src/ouro_jlens/{name}"
        records.append(record)

    from ouro_jlens.probe import _jlens_package_root, _jlens_source_paths

    package_root = _jlens_package_root()
    for path in _jlens_source_paths():
        record = evidence.file_record(path)
        record["path"] = f"dependency/jlens/{path.relative_to(package_root).as_posix()}"
        records.append(record)
    digest = evidence.aggregate_sha256({record["path"]: record["sha256"] for record in records})

    valid, returned, errors = report_module._probe_source_records_valid(records, digest)
    assert valid, errors
    assert returned == records

    valid, _, errors = report_module._probe_source_records_valid(records[:-1], digest)
    assert not valid
    assert "probe dependency source manifest is incomplete" in errors


def test_checkpoint_claim_keeps_distributional_and_categorical_trends_separate():
    text = checkpoint_claim_text({
        "trends": {
            "kl": {"multihop": "decreases", "order-ops": "mixed"},
            "js": {"multihop": "decreases", "order-ops": "mixed"},
            "agreement": {"multihop": "mixed", "order-ops": "increases"},
        }
    })
    assert "multihop KL decreases and JS decreases" in text
    assert "order-ops KL mixed and JS mixed" in text
    assert "top-1 agreement mixed" in text
    assert "no blanket agreement-improvement claim" in text


def test_manifest_rejects_stripped_contract(monkeypatch, tmp_path):
    # A recomputed checksum over a one-group document is still not a custody
    # manifest.  Avoid depending on the live external jlens checkout for this
    # fail-closed shape test.
    monkeypatch.setattr(manifest, "inventory", lambda: {})
    monkeypatch.setattr(manifest, "_jlens_identity", lambda: ([], "UNKNOWN"))
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps({"groups": {"source": []}}))
    result = manifest.verify_manifest(path)
    assert result["status"] == "FAIL"
    assert any(mismatch.get("field") == "top_level" for mismatch in result["mismatches"])
    assert any(mismatch.get("field") == "groups" for mismatch in result["mismatches"])


def test_convergence_requires_hash_valid_disjoint_prompt_slices(monkeypatch, tmp_path):
    class Lens:
        def __init__(self, n):
            self.n_prompts = n
            self.source_layers = [0]
            self.d_model = 2
            self.jacobians = {0: torch.eye(2)}

    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    lenses = {str(first): Lens(2), str(second): Lens(3)}
    monkeypatch.setattr(lens_convergence.JacobianLens, "load", lambda path: lenses[path])
    metadata = {
        str(first): {"kind": "fit", "prompt_file_sha256": "f" * 64,
                     "prompt_slice_sha256": "a" * 64, "start": 0, "end": 2,
                     "n_prompts": 2, "n_fitted": 2},
        str(second): {"kind": "fit", "prompt_file_sha256": "f" * 64,
                      "prompt_slice_sha256": "b" * 64, "start": 1, "end": 4,
                      "n_prompts": 3, "n_fitted": 3},
    }
    for path in (first, second):
        path.with_suffix(".json").write_text('{"kind":"fit"}')
    monkeypatch.setattr(lens_convergence, "validate_sidecar",
                        lambda path, kind: metadata[str(path)])
    with pytest.raises(IntegrityError, match="overlap"):
        lens_convergence.build(str(first), str(second))

    metadata[str(second)].update({"start": 2, "end": 5})
    report = lens_convergence.build(str(first), str(second))
    assert report["status"] == "INCONCLUSIVE_TWO_POINT_DIAGNOSTIC"
    assert report["disjoint_prompt_evidence"]["slice_a"]["end"] == 2
    assert report["disjoint_prompt_evidence"]["slice_b"]["start"] == 2
