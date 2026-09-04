"""CPU-only regression tests for the JLens evidence contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jlens.lens import JacobianLens
from ouro_jlens import bench, evidence, evaluate, fit_lens, manifest, validate


class FakeModel:
    n_physical = 2
    n_ut = 2
    n_layers = 4
    d_model = 2
    model_revision = "fake-model-r1"

    def exit_index(self, ut: int) -> int:
        if not 0 <= ut < self.n_ut:
            raise AssertionError(ut)
        return ut * self.n_physical + self.n_physical - 1


def _args(tmp_path, prompts_path, start, end, out, **overrides):
    values = {
        "target_ut": 1,
        "prompts": str(prompts_path),
        "start": start,
        "end": end,
        "dim_batch": 1,
        "max_seq_len": 16,
        "skip_first": 0,
        "checkpoint_every": None,
        "checkpoint": None,
        "out": str(out),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_fit(model, prompts, **kwargs):
    sources = list(kwargs["source_layers"])
    return JacobianLens(
        {layer: torch.eye(model.d_model) * (layer + 1) for layer in sources},
        n_prompts=len(prompts),
        d_model=model.d_model,
    )


def test_atomic_helpers_replace_complete_files(tmp_path):
    json_path = tmp_path / "nested" / "result.json"
    npz_path = tmp_path / "nested" / "arrays.npz"
    evidence.atomic_write_text(json_path, "hello")
    evidence.atomic_write_json(json_path, {"b": 2, "a": 1})
    evidence.atomic_savez(npz_path, x=np.arange(3))
    assert json.loads(json_path.read_text()) == {"a": 1, "b": 2}
    np.testing.assert_array_equal(np.load(npz_path)["x"], np.arange(3))
    record = evidence.file_record(npz_path)
    assert record == {
        "path": str(npz_path),
        "size": npz_path.stat().st_size,
        "sha256": evidence.sha256_file(npz_path),
    }


def test_fit_publishes_prompt_and_output_identity(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c", "d"]))
    monkeypatch.setattr(fit_lens, "load_ouro", lambda: FakeModel())
    monkeypatch.setattr(fit_lens.jlens, "fit", _fake_fit)
    out = tmp_path / "shard.pt"

    fit_lens.fit(_args(tmp_path, prompt_path, 1, 3, out))
    metadata_path = out.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["prompt_slice"] == {
        "start": 1,
        "end": 3,
        "count": 2,
        "sha256": evidence.prompt_slice_sha256(["b", "c"]),
    }
    assert metadata["output_sha256"] == evidence.sha256_file(out)
    assert metadata["output"]["size"] == out.stat().st_size
    assert metadata["model_revision"] == "fake-model-r1"
    assert metadata["model"]["bytes_status"] == "REVISION_AND_SHAPE_ONLY"
    assert len(metadata["model_snapshot_sha256"]) == 64
    assert metadata["jlens_commit"]

    # Idempotent verification is allowed only when bytes and identity agree.
    fit_lens.fit(_args(tmp_path, prompt_path, 1, 3, out))
    out.write_bytes(out.read_bytes() + b"tamper")
    with pytest.raises(fit_lens.IntegrityError, match="output hash mismatch"):
        fit_lens.fit(_args(tmp_path, prompt_path, 1, 3, out))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"target_ut": 2}, "target_ut"),
        ({"start": -1}, "invalid prompt range"),
        ({"start": 3, "end": 3}, "invalid prompt range"),
        ({"start": 0, "end": 5}, "invalid prompt range"),
        ({"dim_batch": 0}, "dim_batch"),
    ],
)
def test_fit_rejects_invalid_contract_before_fit(monkeypatch, tmp_path, kwargs, message):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c", "d"]))
    monkeypatch.setattr(fit_lens, "load_ouro", lambda: FakeModel())
    called = False

    def should_not_run(*args, **kw):
        nonlocal called
        called = True
        raise AssertionError("heavy fit was called")

    monkeypatch.setattr(fit_lens.jlens, "fit", should_not_run)
    values = {"start": 0, "end": 2}
    values.update(kwargs)
    with pytest.raises(fit_lens.IntegrityError, match=message):
        overrides = dict(kwargs)
        overrides.pop("start", None)
        overrides.pop("end", None)
        fit_lens.fit(
            _args(
                tmp_path,
                prompt_path,
                values["start"],
                values["end"],
                tmp_path / "x.pt",
                **overrides,
            )
        )
    assert not called


def test_merge_rejects_gap_and_accepts_adjacent_verified_shards(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c", "d"]))
    monkeypatch.setattr(fit_lens, "load_ouro", lambda: FakeModel())
    monkeypatch.setattr(fit_lens.jlens, "fit", _fake_fit)
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    fit_lens.fit(_args(tmp_path, prompt_path, 0, 2, first))
    fit_lens.fit(_args(tmp_path, prompt_path, 2, 4, second))
    merged = tmp_path / "merged.pt"
    fit_lens.merge(SimpleNamespace(out=str(merged), shards=[str(first), str(second)]))
    assert merged.exists() and merged.with_suffix(".json").exists()
    assert json.loads(merged.with_suffix(".json").read_text())["start"] == 0

    metadata_path = second.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["start"] = 3
    metadata["prompt_slice"]["start"] = 3
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(fit_lens.IntegrityError, match="gap"):
        fit_lens.merge(SimpleNamespace(out=str(tmp_path / "bad.pt"), shards=[str(first), str(second)]))


def test_unsealed_checkpoint_is_never_resumed(tmp_path):
    checkpoint = tmp_path / "partial.ckpt"
    checkpoint.write_bytes(b"possibly partial")
    expected = {"schema_version": 1, "kind": "fit"}
    evidence.atomic_write_json(
        fit_lens.checkpoint_sidecar_path(checkpoint),
        fit_lens._checkpoint_contract(expected),
    )
    with pytest.raises(fit_lens.IntegrityError, match="never sealed"):
        fit_lens._prepare_checkpoint(checkpoint, expected)


def test_evaluator_accepts_verified_fit_or_merge_sidecars(monkeypatch, tmp_path):
    binary = tmp_path / "lens.pt"
    binary.write_bytes(b"lens")
    sidecar = binary.with_suffix(".json")
    for kind in ("fit", "merged"):
        sidecar.write_text(json.dumps({"kind": kind}))
        monkeypatch.setattr(evaluate, "validate_sidecar", lambda path, *, kind: {"kind": kind})
        assert evaluate.load_lens_metadata(binary)["kind"] == kind


def test_stacked_jacobians_requires_exact_contiguous_sources():
    model = FakeModel()
    lens = JacobianLens({0: torch.eye(2), 2: torch.eye(2)}, n_prompts=1, d_model=2)
    from ouro_jlens.evaluate import IntegrityError, stacked_jacobians

    with pytest.raises(IntegrityError, match="exactly contiguous"):
        stacked_jacobians(model, lens, target=3)


def test_bench_returns_failure_when_every_configuration_is_oom(monkeypatch):
    monkeypatch.setattr(bench, "load_ouro", lambda: object())

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("synthetic OOM")

    monkeypatch.setattr(bench, "bench", fail)
    assert bench.main(["8", "1", "2"]) == 1


def test_validator_rollup_lists_nested_comparisons_and_separates_passes():
    report = {
        "model": {"n_ut": 2},
        "m1_noninterference": {
            "hooked_vs_plain": {"equal": True, "close": True},
            "after_vs_plain": {"equal": False, "close": True},
        },
        "m2_exit_equality": {
            "ut0": {"equal": True, "close": True},
            "ut1": {"equal": True, "close": True},
            "default_forward_vs_final": {"equal": True, "close": True},
        },
        "m3_recurrent_identity": {
            "norm_of_recorded_L47_equals_native_hidden_states_list": {
                "ut0": {"equal": True, "close": True}
            },
            "next_loop_input_equals_normed_L47": {},
            "loop1_layer0_input_equals_embeddings": {"equal": True, "close": True},
            "stock_recorder_returns_last_loop": True,
        },
        "m4_distinct_vjps": {"self_grad_is_cotangent": True},
        "m5_stock_consistency": {
            "stock_L40_vs_recurrent_final_L40": {"equal": False, "close": True}
        },
    }
    comparisons, bit_exact = validate.bit_exact_rollup(report)
    assert any(item["path"] == "m2_exit_equality.default_forward_vs_final" for item in comparisons)
    assert not bit_exact


def test_m3_expected_order_counts_one_forward_not_the_stock_recorder():
    source = Path(validate.__file__).read_text()
    removal = source.index("for handle in handles:\n            handle.remove()")
    stock_forward = source.index("with ActivationRecorder(m.blocks")
    assert removal < stock_forward


def test_manifest_verifier_detects_tampering(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "PROJECT_ROOT", tmp_path)
    payload = tmp_path / "input.bin"
    payload.write_bytes(b"accepted bytes")
    record = evidence.file_record(payload)
    declared = {"path": "input.bin", "size": record["size"], "sha256": record["sha256"],
                "tracked": False}
    manifest_path = tmp_path / "MANIFEST.json"
    evidence.atomic_write_json(manifest_path, {
        "schema_version": 1,
        "groups": {"raw": [declared]},
        "aggregate_sha256": evidence.sha256_json({"input.bin": record["sha256"]}),
    })
    assert manifest.verify_manifest(manifest_path)["status"] == "PASS"
    payload.write_bytes(b"tampered")
    result = manifest.verify_manifest(manifest_path)
    assert result["status"] == "FAIL"
    assert result["mismatches"]


def test_manifest_verifier_rejects_inventory_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "PROJECT_ROOT", tmp_path)
    payload = tmp_path / "input.bin"
    payload.write_bytes(b"accepted bytes")
    record = evidence.file_record(payload)
    record["path"] = "input.bin"
    record["tracked"] = False
    groups = {"raw": [Path("input.bin")], "external_jlens_source": []}
    manifest_path = tmp_path / "MANIFEST.json"
    evidence.atomic_write_json(manifest_path, {
        "schema_version": 1,
        "groups": {"raw": [record]},
        "inventory_sha256": evidence.sha256_json(manifest._inventory_paths(groups)),
        "jlens_revision": "UNKNOWN",
        "aggregate_sha256": evidence.sha256_json({"input.bin": record["sha256"]}),
    })
    monkeypatch.setattr(manifest, "inventory", lambda: {"raw": [Path("input.bin")]})
    monkeypatch.setattr(manifest, "_jlens_identity", lambda: ([], "UNKNOWN"))
    assert manifest.verify_manifest(manifest_path)["status"] == "PASS"
    monkeypatch.setattr(
        manifest, "inventory", lambda: {"raw": [Path("input.bin"), Path("new-source.py")]}
    )
    result = manifest.verify_manifest(manifest_path)
    assert result["status"] == "FAIL"
    assert any(row.get("field") == "inventory_sha256" for row in result["mismatches"])


def test_manifest_verifier_rejects_relative_path_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "PROJECT_ROOT", tmp_path)
    manifest_path = tmp_path / "MANIFEST.json"
    evidence.atomic_write_json(manifest_path, {
        "schema_version": 1,
        "groups": {"raw": [{"path": "../outside", "size": 0,
                              "sha256": "0" * 64, "tracked": False}]},
        "aggregate_sha256": evidence.sha256_json({}),
    })
    result = manifest.verify_manifest(manifest_path)
    assert result["status"] == "FAIL"
    assert any("escapes" in row.get("error", "") for row in result["mismatches"])


def test_evaluator_rejects_symlink_output_root_before_loading_model(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(evaluate, "load_ouro", lambda: (_ for _ in ()).throw(
        AssertionError("model must not load before output path validation")
    ))
    with pytest.raises(fit_lens.IntegrityError, match="symlink"):
        evaluate.main(["--lens", "3=unused.pt", "--out", str(linked)])
