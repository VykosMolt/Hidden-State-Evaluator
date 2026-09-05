"""Focused provenance and fail-closed tests for the JLens probe chain."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ouro_jlens import (
    checkpoints,
    probe,
    probe_cv,
    probe_report,
    transport_report,
    verify_artifacts,
    verify_paid,
)


def test_probe_source_manifest_binds_complete_jlens_tree_with_logical_paths():
    manifest = probe._source_manifest()
    package_root = probe._jlens_package_root()
    expected = {
        f"dependency/jlens/{path.relative_to(package_root).as_posix()}"
        for path in probe._jlens_source_paths()
    }
    actual = {
        record["path"]
        for record in manifest["files"]
        if record["path"].startswith("dependency/jlens/")
    }
    assert actual == expected
    assert all(not Path(path).is_absolute() for path in actual)


def test_probe_source_manifest_changes_when_installed_jlens_bytes_change(
    monkeypatch: pytest.MonkeyPatch,
):
    original = probe._source_manifest()
    target = probe._jlens_source_paths()[0]
    real_record = probe.file_record

    def changed_record(path: Path) -> dict:
        record = real_record(path)
        if Path(path) == target:
            record["sha256"] = "0" * 64
        return record

    monkeypatch.setattr(probe, "file_record", changed_record)
    changed = probe._source_manifest()
    assert changed["sha256"] != original["sha256"]
    assert any(
        record["path"].startswith("dependency/jlens/")
        and record["sha256"] == "0" * 64
        for record in changed["files"]
    )


def test_probe_and_cv_reject_nonfinite_hidden_states():
    with pytest.raises(ValueError, match="non-finite"):
        probe.lens_candidate_ranks(
            SimpleNamespace(input_device="cpu", n_layers=1),
            torch.tensor([[[float("nan")]]]),
            np.asarray([2]),
            None,
            [[1]],
        )
    with pytest.raises(ValueError, match="non-finite"):
        probe_cv.cv_probe(
            np.asarray([[float("inf")]]),
            np.asarray([2]),
            np.asarray([0]),
            [(1, 1)],
        )


def test_probe_location_fit_forces_single_native_thread(monkeypatch: pytest.MonkeyPatch):
    entered: list[int] = []

    class Guard:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(probe_cv, "threadpool_limits", lambda *, limits: Guard())
    hidden = np.random.default_rng(0).normal(size=(12, 1, 4)).astype(np.float32)
    labels = np.asarray([2, 3, 4] * 4)
    train = np.asarray([True] * 6 + [False] * 6)
    validation = np.asarray([False] * 6 + [True] * 3 + [False] * 3)
    test = ~(train | validation)

    ranks, chosen_c, accuracy = probe_cv._fit_location(
        hidden, 0, labels, train, validation, test,
    )
    assert entered == [1]
    assert ranks.shape == (3,)
    assert chosen_c in probe.C_GRID
    assert 0.0 <= accuracy <= 1.0


def test_rebuild_invalidates_every_probe_derivative_before_gpu_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    out = tmp_path / "probe"
    out.mkdir()
    cache = tmp_path / "cache.npz"
    names = (
        "lens_all648.npz",
        "lens_all648.provenance.json",
        "arrays.npz",
        "design.json",
        "summary.json",
    )
    for name in names:
        (out / name).write_bytes(b"stale")
    cache.write_bytes(b"stale cache")
    cache.with_suffix(".provenance.json").write_bytes(b"stale cache provenance")

    class RebuildObserved(RuntimeError):
        pass

    def observe(path: Path) -> None:
        assert all(not (out / name).exists() for name in names)
        assert not cache.exists()
        assert not cache.with_suffix(".provenance.json").exists()
        raise RebuildObserved

    monkeypatch.setattr(probe_cv, "rebuild_gpu_cache", observe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_cv.py",
            "--cache", str(cache),
            "--lens", str(tmp_path / "lens.pt"),
            "--out", str(out),
            "--rebuild-gpu-cache",
            "--lens-only",
        ],
    )
    with pytest.raises(RebuildObserved):
        probe_cv.main()


def test_paid_verifier_invalidates_old_verdict_before_fail_closed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "verification.json"
    output.write_text('{"status":"PASS","accepted":true}')
    monkeypatch.setenv(
        "JLENS_IMAGE_DIGEST",
        "registry.invalid/runtime@sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_paid.py",
            "--validation", str(tmp_path / "missing-validation.json"),
            "--out", str(output),
        ],
    )
    assert verify_paid.main() == 1
    assert not output.exists()


def test_paid_verifier_binds_controller_approved_image_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        verify_paid,
        "approved_b300_image_digest",
        lambda: "registry.invalid/runtime@sha256:" + "1" * 64,
    )
    monkeypatch.setenv(
        "JLENS_IMAGE_DIGEST",
        "registry.invalid/runtime@sha256:" + "2" * 64,
    )
    with pytest.raises(verify_paid.PaidVerificationError, match="controller-approved"):
        verify_paid._image_digest()


def test_paid_verifier_rejects_symlink_alias_for_exact_evaluation_directory(tmp_path):
    expected = tmp_path / "eval" / "fitsize_n80"
    expected.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(expected, target_is_directory=True)

    with pytest.raises(verify_paid.PaidVerificationError, match="linked"):
        verify_paid._exact_directory(alias, expected, "main evaluation path")


def test_cpu_validation_recursively_rejects_stale_hidden_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    design = tmp_path / "design.json"
    design.write_text(json.dumps({
        "schema_version": 2,
        "status": "FRESH_CURRENT_SOURCE_INNER_SELECTED_OUTER_SCORED",
        "seed": 0,
        "runtime_versions": {},
        "inputs": {},
    }))
    monkeypatch.setattr(probe_cv, "_check_source_manifest", lambda *args: None)
    monkeypatch.setattr(probe_cv, "_runtime_versions", lambda: {})
    called: list[str] = []

    def stale_cache(path: Path) -> dict:
        called.append("cache")
        raise ValueError("hidden cache is stale")

    monkeypatch.setattr(probe_cv, "validate_hidden_cache", stale_cache)
    with pytest.raises(ValueError, match="hidden cache is stale"):
        probe_cv.validate_cpu_evidence(
            tmp_path / "arrays.npz",
            design,
            tmp_path / "cache.npz",
            tmp_path / "scores.npz",
            seed=0,
            lens_path=tmp_path / "lens.pt",
        )
    assert called == ["cache"]


def test_cpu_validation_recursively_rejects_stale_lens_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    design = tmp_path / "design.json"
    design.write_text(json.dumps({
        "schema_version": 2,
        "status": "FRESH_CURRENT_SOURCE_INNER_SELECTED_OUTER_SCORED",
        "seed": 0,
        "runtime_versions": {},
        "inputs": {},
    }))
    monkeypatch.setattr(probe_cv, "_check_source_manifest", lambda *args: None)
    monkeypatch.setattr(probe_cv, "_runtime_versions", lambda: {})
    called: list[str] = []
    monkeypatch.setattr(
        probe_cv,
        "validate_hidden_cache",
        lambda path: called.append("cache") or {"output": {"path": "gpu_cache"}},
    )

    def stale_scores(*args) -> dict:
        called.append("scores")
        raise ValueError("lens scores are stale")

    monkeypatch.setattr(probe_cv, "validate_lens_scores", stale_scores)
    with pytest.raises(ValueError, match="lens scores are stale"):
        probe_cv.validate_cpu_evidence(
            tmp_path / "arrays.npz",
            design,
            tmp_path / "cache.npz",
            tmp_path / "scores.npz",
            seed=0,
            lens_path=tmp_path / "lens.pt",
        )
    assert called == ["cache", "scores"]


def _minimal_probe_arrays(path: Path) -> None:
    prompts = probe.make_prompts()
    ranks = np.ones((len(prompts), probe_cv.N_UT * probe_cv.N_LAYER), dtype=np.int32)
    np.savez_compressed(
        path,
        labels=np.asarray([item["label"] for item in prompts]),
        folds=probe_cv.prompt_folds(prompts, seed=0),
        probe_rank=ranks,
        ll_cand=ranks,
        jl_cand=ranks,
        selection_accuracy=np.zeros((probe_cv.N_FOLDS, probe_cv.N_UT * probe_cv.N_LAYER)),
        chosen_C=np.full((probe_cv.N_FOLDS, probe_cv.N_UT * probe_cv.N_LAYER), 0.01),
    )


def test_probe_report_requires_positive_draws_and_records_logical_input(tmp_path: Path):
    arrays = tmp_path / "arrays.npz"
    _minimal_probe_arrays(arrays)
    with pytest.raises(ValueError, match="positive integer"):
        probe_report.build_report(arrays, draws=0)
    report = probe_report.build_report(arrays, draws=1)
    assert report["input"]["path"] == "probe_arrays"
    assert not Path(report["input"]["path"]).is_absolute()


def test_checkpoint_provenance_contains_jlens_sources_and_full_runtime():
    manifest = checkpoints._source_manifest()
    dependency = [
        record for record in manifest["files"]
        if record["path"].startswith("dependency/jlens/")
    ]
    assert dependency
    assert manifest["sha256"]
    runtime = checkpoints._runtime_versions()
    assert runtime["python"]
    assert runtime["jlens"]
    assert runtime["transformers"]


def test_checkpoint_measurement_failure_leaves_nonaccepting_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    model_root = tmp_path / "model"
    model_root.mkdir()
    output = tmp_path / "out"

    class FakeTokenizer:
        pass

    class Failure(RuntimeError):
        pass

    monkeypatch.setattr(
        checkpoints,
        "CHECKPOINTS",
        {"base": model_root},
    )
    monkeypatch.setattr(
        checkpoints,
        "_checkpoint_files",
        lambda path: [{"path": "checkpoint/model.safetensors", "size": 1, "sha256": "0" * 64}],
    )
    monkeypatch.setattr(
        checkpoints,
        "load_ouro",
        lambda path: SimpleNamespace(model_revision="test", tokenizer=None),
    )
    monkeypatch.setattr(checkpoints, "load_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(checkpoints, "measure", lambda *args: (_ for _ in ()).throw(Failure("boom")))
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(sys, "argv", ["checkpoints.py", "--out", str(output), "--only", "base"])
    with pytest.raises(Failure, match="boom"):
        checkpoints.main()
    document = json.loads((output / "exit_divergence.json").read_text())
    assert document["status"] == "INTERRUPTED_OR_FAILED"
    assert document["checkpoints"] == {}
    assert any(record["path"].startswith("dependency/jlens/") for record in document["source_files"])


def test_verify_transport_rejects_noninteger_sample_sizes_and_requires_status():
    valid = {
        "schema_version": 1,
        "status": "MODELED_ASSOCIATION_ONLY",
        "sample_sizes": [8, 32, 56],
        "per_loop": [
            {"loop": loop, "raw_fitted_map_norm_by_n": {str(size): 0.1 for size in [8, 32, 56]}}
            for loop in range(1, 5)
        ],
        "negative_sigma_squared_total": 0,
    }
    assert verify_artifacts._verify_transport_shape(valid) == [8, 32, 56]
    invalid = {**valid, "sample_sizes": [8, True, 56]}
    with pytest.raises(verify_artifacts.VerificationError, match="sample_sizes"):
        verify_artifacts._verify_transport_shape(invalid)
    invalid_status = {**valid, "status": "PASS"}
    with pytest.raises(verify_artifacts.VerificationError, match="status"):
        verify_artifacts._verify_transport_shape(invalid_status)


def test_verify_transport_recomputes_fresh_statistics_and_rejects_tampering():
    sizes = np.asarray([8, 32, 56, 80])
    mu2 = np.linspace(0.1, 0.3, 191)
    sigma2 = np.linspace(0.2, 0.6, 191)
    squared = np.stack([mu2 + sigma2 / n for n in sizes])
    document = transport_report.summarize(sizes, squared)
    document.update({"schema_version": 1, "status": "MODELED_ASSOCIATION_ONLY"})
    checked = verify_artifacts.verify_transport(document)
    assert checked["sample_sizes"] == sizes.tolist()
    assert checked["n_sources"] == 191

    tampered = json.loads(json.dumps(document))
    tampered["per_source"]["sigma_squared"][0] += 0.01
    with pytest.raises(AssertionError):
        verify_artifacts.verify_transport(tampered)


def test_verifier_compares_fresh_main_and_probe_reports_instead_of_frozen_numbers():
    analysis = json.loads(Path("artifacts/jlens/final/analysis.json").read_text())
    main_root = Path("artifacts/jlens/eval/fitsize_n80")
    observed_main = verify_artifacts.verify_main(main_root, analysis)
    assert observed_main["total_items"] == 148

    changed_analysis = json.loads(json.dumps(analysis))
    changed_analysis["main"]["populations"]["multihop"]["readouts"][
        "eventual_exit_jacobian_lens"
    ]["excess_pass10"][0] += 0.01
    with pytest.raises(AssertionError):
        verify_artifacts.verify_main(main_root, changed_analysis)

    probe_path = Path("artifacts/jlens/probe/cv_all648/arrays.npz")
    summary = json.loads(Path("artifacts/jlens/probe/cv_all648/summary.json").read_text())
    observed_probe = verify_artifacts.verify_probe(probe_path, summary)
    assert set(observed_probe) == {"probe_rank", "ll_cand", "jl_cand"}

    changed_summary = json.loads(json.dumps(summary))
    changed_summary["readouts"]["supervised_probe"][
        "selected_physical_layer_by_heldout_fold"
    ][0][0] += 1
    with pytest.raises(verify_artifacts.VerificationError, match="selected layers"):
        verify_artifacts.verify_probe(probe_path, changed_summary)


def test_verify_input_records_are_logical_and_include_auxiliary_json(tmp_path: Path):
    files = {}
    for name in ("arrays.npz", "items.json", "task_names.json"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = verify_artifacts._logical_record(path, f"main/{name}")
    assert {record["path"] for record in files.values()} == {
        "main/arrays.npz", "main/items.json", "main/task_names.json"
    }
    assert all(not Path(record["path"]).is_absolute() for record in files.values())


def test_verifier_invalidates_stale_verdict_and_rejects_linked_parent(tmp_path: Path):
    verdict = tmp_path / "verification.json"
    verdict.write_text('{"status":"PASS"}')
    verify_artifacts._invalidate_output(verdict)
    assert not verdict.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(verify_artifacts.VerificationError, match="traverses a link"):
        verify_artifacts._invalidate_output(linked / "verification.json")
