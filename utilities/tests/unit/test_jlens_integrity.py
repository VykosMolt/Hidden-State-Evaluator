"""CPU-only regression tests for the JLens evidence contract."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jlens.lens import JacobianLens
from ouro_jlens import bench, evidence, evaldata, evaluate, fit_lens, manifest, validate


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


class HashBoundFakeModel(FakeModel):
    model_revision = "fake-model-hash-bound-r1"

    def __init__(self, snapshot_path: Path):
        self.snapshot_path = snapshot_path


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


def _fit_fake(monkeypatch, tmp_path, prompt_path, start, end, out, *, model=None, **overrides):
    loaded_model = model or FakeModel()
    monkeypatch.setattr(fit_lens, "load_ouro", lambda: loaded_model)
    monkeypatch.setattr(fit_lens.jlens, "fit", _fake_fit)
    fit_lens.fit(_args(tmp_path, prompt_path, start, end, out, **overrides))
    return json.loads(out.with_suffix(".json").read_text())


def _snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in fit_lens.MODEL_FILE_NAMES:
        (snapshot / name).write_bytes(name.encode())
    return snapshot


def test_default_fit_prompt_uses_pinned_corpus_without_replacing_legacy_file():
    assert fit_lens.DEFAULT_PROMPTS.name == "wikitext_prompts_b08601e.json"
    assert fit_lens.DEFAULT_PROMPTS.name != "wikitext_prompts.json"


def test_fit_binds_paired_prompt_provenance_and_source_revision(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompts = ["a", "b", "c"]
    prompt_path.write_text(json.dumps(prompts))
    provenance_path = prompt_path.with_suffix(".provenance.json")
    evidence.atomic_write_json(
        provenance_path,
        {
            "schema_version": 1,
            "status": "FRESH_PINNED_DATASET_REVISION",
            "source": {
                "dataset": "test/dataset",
                "config": "test-config",
                "split": "train",
                "revision": "a" * 40,
                "minimum_characters": 1,
                "requested_prompts": len(prompts),
            },
            "output": {
                "path": "wikitext_prompts",
                "size": prompt_path.stat().st_size,
                "sha256": evidence.sha256_file(prompt_path),
            },
        },
    )
    out = tmp_path / "shard.pt"
    metadata = _fit_fake(
        monkeypatch,
        tmp_path,
        prompt_path,
        0,
        2,
        out,
    )
    assert metadata["prompt_provenance"]["path"] == "prompts.provenance.json"
    assert metadata["prompt_source_revision"] == "a" * 40
    assert fit_lens.validate_sidecar(out)["prompt_source"] == {
        "dataset": "test/dataset",
        "config": "test-config",
        "split": "train",
        "revision": "a" * 40,
        "minimum_characters": 1,
        "requested_prompts": len(prompts),
    }

    changed = json.loads(provenance_path.read_text())
    changed["source"]["revision"] = "b" * 40
    evidence.atomic_write_json(provenance_path, changed)
    with pytest.raises(fit_lens.IntegrityError, match="prompt provenance record mismatch"):
        fit_lens.validate_sidecar(out)


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


@pytest.mark.parametrize("returned_count", [0, 1, 3, 5])
def test_fit_rejects_zero_partial_and_over_count_before_publication(
    monkeypatch, tmp_path, returned_count
):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c", "d"]))
    monkeypatch.setattr(fit_lens, "load_ouro", lambda: FakeModel())

    def incomplete_fit(model, prompts, **kwargs):
        return JacobianLens(
            {layer: torch.eye(model.d_model) for layer in kwargs["source_layers"]},
            n_prompts=returned_count,
            d_model=model.d_model,
        )

    monkeypatch.setattr(fit_lens.jlens, "fit", incomplete_fit)
    out = tmp_path / f"bad_{returned_count}.pt"
    with pytest.raises(fit_lens.IntegrityError, match="incomplete prompt count"):
        fit_lens.fit(_args(tmp_path, prompt_path, 1, 3, out))
    assert not out.exists()
    assert not out.with_suffix(".json").exists()


def test_sidecar_rejects_duplicate_count_and_digest_fields(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c", "d"]))
    out = tmp_path / "shard.pt"
    _fit_fake(monkeypatch, tmp_path, prompt_path, 1, 3, out)
    sidecar = out.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    metadata["n_prompts"] += 1
    with pytest.raises(fit_lens.IntegrityError, match="n_prompts"):
        sidecar.write_text(json.dumps(metadata))
        fit_lens.validate_sidecar(out)
    metadata = json.loads(sidecar.read_text())
    metadata["prompt_slice_sha256"] = "0" * 64
    metadata["n_prompts"] = 2
    with pytest.raises(fit_lens.IntegrityError, match="digest fields disagree"):
        sidecar.write_text(json.dumps(metadata))
        fit_lens.validate_sidecar(out)


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
    prompt_path.write_text(json.dumps(["a", "b", "c", "d", "e", "f"]))
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
    metadata["end"] = 5
    metadata["prompt_slice"].update(
        {
            "start": 3,
            "end": 5,
            "sha256": evidence.prompt_slice_sha256(["d", "e"]),
        }
    )
    metadata["prompt_slice_sha256"] = metadata["prompt_slice"]["sha256"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(fit_lens.IntegrityError, match="gap"):
        fit_lens.merge(SimpleNamespace(out=str(tmp_path / "bad.pt"), shards=[str(first), str(second)]))


def test_merge_rejects_wrong_merged_count_before_output_publication(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c", "d"]))
    monkeypatch.setattr(fit_lens, "load_ouro", lambda: FakeModel())
    monkeypatch.setattr(fit_lens.jlens, "fit", _fake_fit)
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    fit_lens.fit(_args(tmp_path, prompt_path, 0, 2, first))
    fit_lens.fit(_args(tmp_path, prompt_path, 2, 4, second))
    original_merge = fit_lens.jlens.JacobianLens.merge

    def wrong_count(lenses):
        merged = original_merge(lenses)
        return JacobianLens(
            merged.jacobians,
            n_prompts=merged.n_prompts - 1,
            d_model=merged.d_model,
        )

    monkeypatch.setattr(fit_lens.jlens.JacobianLens, "merge", wrong_count)
    out = tmp_path / "wrong_count.pt"
    with pytest.raises(fit_lens.IntegrityError, match="merged lens prompt count mismatch"):
        fit_lens.merge(SimpleNamespace(out=str(out), shards=[str(first), str(second)]))
    assert not out.exists()
    assert not out.with_suffix(".json").exists()


def test_existing_merge_rejects_same_path_shard_replacement(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c", "d"]))
    monkeypatch.setattr(fit_lens, "load_ouro", lambda: FakeModel())
    monkeypatch.setattr(fit_lens.jlens, "fit", _fake_fit)
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    fit_lens.fit(_args(tmp_path, prompt_path, 0, 2, first))
    fit_lens.fit(_args(tmp_path, prompt_path, 2, 4, second))
    merged = tmp_path / "merged.pt"
    fit_lens.merge(SimpleNamespace(out=str(merged), shards=[str(first), str(second)]))

    replacement = JacobianLens(
        {layer: torch.eye(2) * (layer + 10) for layer in (0, 1, 2)},
        n_prompts=2,
        d_model=2,
    )
    fit_lens._atomic_lens_save(replacement, second)
    sidecar = second.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    output = fit_lens._record_for_path(
        second,
        root=tmp_path,
        root_kind=fit_lens.TEST_BUNDLE_ROOT_KIND,
    )
    metadata["output"] = output
    metadata["output_sha256"] = output["sha256"]
    evidence.atomic_write_json(sidecar, metadata)
    with pytest.raises(fit_lens.IntegrityError, match="existing merge output"):
        fit_lens.merge(SimpleNamespace(out=str(merged), shards=[str(first), str(second)]))


def test_evaluator_rejects_disjoint_prompt_populations(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c", "d"]))
    snapshot = _snapshot(tmp_path)
    model = HashBoundFakeModel(snapshot)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    _fit_fake(
        monkeypatch,
        tmp_path,
        prompt_path,
        0,
        2,
        first,
        model=model,
        validation_root=str(tmp_path),
    )
    second_metadata = _fit_fake(
        monkeypatch,
        tmp_path,
        prompt_path,
        2,
        4,
        second,
        model=model,
        validation_root=str(tmp_path),
    )
    first_metadata = json.loads(first.with_suffix(".json").read_text())
    with pytest.raises(fit_lens.IntegrityError, match="prompt slices differ"):
        evaluate.validate_lens_metadata(
            second_metadata,
            target_ut=1,
            model=model,
            baseline=first_metadata,
            prompt_policy="identical",
            validation_root=tmp_path,
        )
    with pytest.raises(fit_lens.IntegrityError, match="common prompt start"):
        evaluate.validate_lens_metadata(
            second_metadata,
            target_ut=1,
            model=model,
            baseline=first_metadata,
            prompt_policy="nested",
            validation_root=tmp_path,
        )


def test_evaluator_recomputes_model_snapshot_digest(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c"]))
    snapshot = _snapshot(tmp_path)
    model = HashBoundFakeModel(snapshot)
    out = tmp_path / "lens.pt"
    metadata = _fit_fake(
        monkeypatch,
        tmp_path,
        prompt_path,
        0,
        2,
        out,
        model=model,
        validation_root=str(tmp_path),
    )
    (snapshot / "config.json").write_bytes(b"changed model bytes")
    with pytest.raises(fit_lens.IntegrityError, match="snapshot digest"):
        evaluate.validate_lens_metadata(
            metadata,
            target_ut=1,
            model=model,
            validation_root=tmp_path,
        )


def test_evaluator_rejects_revision_only_model_identity(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c"]))
    out = tmp_path / "lens.pt"
    metadata = _fit_fake(monkeypatch, tmp_path, prompt_path, 0, 2, out)
    with pytest.raises(fit_lens.IntegrityError, match="HASH_BOUND"):
        evaluate.validate_lens_metadata(
            metadata,
            target_ut=1,
            model=FakeModel(),
            validation_root=tmp_path,
        )


def test_sidecar_logical_records_survive_bundle_relocation(monkeypatch, tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    prompt_path = origin / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c"]))
    snapshot = _snapshot(origin)
    model = HashBoundFakeModel(snapshot)
    out = origin / "lens.pt"
    metadata = _fit_fake(
        monkeypatch,
        origin,
        prompt_path,
        0,
        2,
        out,
        model=model,
        validation_root=str(origin),
    )
    assert metadata["validation"] == {"path": ".", "kind": fit_lens.TEST_BUNDLE_ROOT_KIND}
    assert metadata["prompt_file"]["path"] == "prompts.json"
    assert metadata["output"]["path"] == "lens.pt"
    dependency_records = [
        record for record in metadata["source_files"]
        if record.get("root_kind") == fit_lens.DEPENDENCY_ROOT_KIND
    ]
    assert dependency_records
    assert all(record["path"].startswith("dependency/jlens/") for record in dependency_records)

    relocated = tmp_path / "relocated"
    shutil.move(str(origin), str(relocated))
    moved = relocated / "lens.pt"
    assert fit_lens.validate_sidecar(moved)["validation"] == {
        "path": ".",
        "kind": fit_lens.TEST_BUNDLE_ROOT_KIND,
    }
    assert fit_lens.validate_sidecar(moved, validation_root=relocated)["output"]["path"] == "lens.pt"


def test_atomic_lens_save_rejects_symlink_destination_and_temp(monkeypatch, tmp_path):
    target = tmp_path / "target.pt"
    target.write_bytes(b"sentinel")
    linked = tmp_path / "linked.pt"
    linked.symlink_to(target)
    lens = JacobianLens({0: torch.eye(2)}, n_prompts=1, d_model=2)
    with pytest.raises(fit_lens.IntegrityError, match="symlink"):
        fit_lens._atomic_lens_save(lens, linked)
    assert target.read_bytes() == b"sentinel"

    preplanted = tmp_path / "preplanted.tmp"
    preplanted.symlink_to(target)
    fd = fit_lens.os.open(os.devnull, os.O_RDONLY)
    monkeypatch.setattr(
        fit_lens.tempfile,
        "mkstemp",
        lambda **kwargs: (fd, str(preplanted)),
    )
    with pytest.raises(fit_lens.IntegrityError, match="temporary lens path"):
        fit_lens._atomic_lens_save(lens, tmp_path / "new.pt")
    assert target.read_bytes() == b"sentinel"
    assert preplanted.is_symlink()
    assert not (tmp_path / "new.pt").exists()


def test_fit_rejects_nonfinite_jacobian_before_publication(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c"]))
    monkeypatch.setattr(fit_lens, "load_ouro", lambda: FakeModel())

    def nonfinite_fit(model, prompts, **kwargs):
        matrix = torch.eye(model.d_model)
        matrix[0, 0] = torch.nan
        return JacobianLens(
            {layer: matrix.clone() for layer in kwargs["source_layers"]},
            n_prompts=len(prompts),
            d_model=model.d_model,
        )

    monkeypatch.setattr(fit_lens.jlens, "fit", nonfinite_fit)
    out = tmp_path / "nonfinite.pt"
    with pytest.raises(fit_lens.IntegrityError, match="not finite"):
        fit_lens.fit(_args(tmp_path, prompt_path, 0, 2, out))
    assert not out.exists()
    assert not out.with_suffix(".json").exists()


def _write_valid_evaluation_bundle(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(["a", "b", "c"]))
    snapshot = _snapshot(tmp_path)
    model = HashBoundFakeModel(snapshot)
    lens_path = tmp_path / "lens.pt"
    metadata = _fit_fake(
        monkeypatch,
        tmp_path,
        prompt_path,
        0,
        2,
        lens_path,
        model=model,
        validation_root=str(tmp_path),
    )
    data_root = tmp_path / "evaldata"
    data_root.mkdir()
    stimulus = data_root / "lens-eval-toy.json"
    stimulus.write_text(json.dumps([]))
    monkeypatch.setattr(evaldata, "JLENS_DATA", data_root)
    output_root = tmp_path / "evaluation"
    output_root.mkdir()
    arrays_path = output_root / "arrays.npz"
    items_path = output_root / "items.json"
    names_path = output_root / "task_names.json"
    evidence.atomic_savez(arrays_path, values=np.arange(2))
    evidence.atomic_write_json(items_path, [])
    evidence.atomic_write_json(names_path, {})
    validation_root_kind = fit_lens.TEST_BUNDLE_ROOT_KIND
    model_identity = fit_lens.model_snapshot_identity(
        model,
        validation_root=tmp_path,
        validation_root_kind=validation_root_kind,
    )
    source = evaluate._evaluation_source_manifest(
        validation_root=tmp_path,
        validation_root_kind=validation_root_kind,
    )
    stimuli = evaluate._evaluation_inputs(
        ["toy"],
        validation_root=tmp_path,
        validation_root_kind=validation_root_kind,
    )
    lens_record = {
        "target_ut": 1,
        "binary": fit_lens._record_for_path(
            lens_path, root=tmp_path, root_kind=validation_root_kind
        ),
        "sidecar": fit_lens._record_for_path(
            lens_path.with_suffix(".json"), root=tmp_path, root_kind=validation_root_kind
        ),
        "identity": {
            key: metadata.get(key)
            for key in (
                "target_ut",
                "target_virtual",
                "source_layers",
                "model_revision",
                "jlens_commit",
                "prompt_file_sha256",
                "prompt_slice_sha256",
                "prompt_slice",
                "n_requested",
                "n_fitted",
                "n_prompts",
                "prompt_provenance_sha256",
                "prompt_source_revision",
                "source_sha256",
                "generator_sha256",
            )
        },
    }
    provenance = {
        "schema_version": fit_lens.SCHEMA_VERSION,
        "validation": {"path": ".", "kind": validation_root_kind},
        "config": {"tasks": ["toy"], "prompt_policy": "identical"},
        "model": model_identity,
        "model_snapshot_sha256": model_identity["aggregate_sha256"],
        "jlens_commit": metadata["jlens_commit"],
        "inputs": {
            "evaluation_files": stimuli,
            "evaluation_input_sha256": evidence.sha256_json(stimuli),
        },
        "lens_inputs": [lens_record],
        "source_files": source["files"],
        "source_sha256": source["sha256"],
        "outputs": {
            "arrays": evaluate._output_record(arrays_path, output_root),
            "items": evaluate._output_record(items_path, output_root),
            "task_names": evaluate._output_record(names_path, output_root),
        },
        "item_count": 0,
        "correct_count": 0,
        "item_metadata_sha256": evidence.sha256_json([]),
    }
    evidence.atomic_write_json(output_root / "provenance.json", provenance)
    return output_root, model, stimulus


def test_evaluation_resume_rejects_stale_stimulus_bytes(monkeypatch, tmp_path):
    output_root, model, stimulus = _write_valid_evaluation_bundle(monkeypatch, tmp_path)
    assert evaluate.validate_evaluation_provenance(
        output_root,
        validation_root=tmp_path,
        model=model,
    )["schema_version"] == fit_lens.SCHEMA_VERSION
    stimulus.write_text(json.dumps(["stale input"]))
    with pytest.raises(fit_lens.IntegrityError, match="evaluation stimuli"):
        evaluate.validate_evaluation_provenance(
            output_root,
            validation_root=tmp_path,
            model=model,
        )


def test_evaluation_resume_rejects_stale_model_bytes(monkeypatch, tmp_path):
    output_root, _model, _stimulus = _write_valid_evaluation_bundle(monkeypatch, tmp_path)
    (tmp_path / "snapshot" / "config.json").write_bytes(b"changed model bytes")
    with pytest.raises(fit_lens.IntegrityError, match="evaluation model"):
        evaluate.validate_evaluation_provenance(
            output_root,
            validation_root=tmp_path,
        )


def test_evaluation_provenance_rejects_revision_only_lens(monkeypatch, tmp_path):
    output_root, model, _stimulus = _write_valid_evaluation_bundle(monkeypatch, tmp_path)
    lens_sidecar = tmp_path / "lens.json"
    metadata = json.loads(lens_sidecar.read_text())
    metadata["model"]["bytes_status"] = "REVISION_AND_SHAPE_ONLY"
    evidence.atomic_write_json(lens_sidecar, metadata)
    provenance_path = output_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["lens_inputs"][0]["sidecar"] = fit_lens._record_for_path(
        lens_sidecar,
        root=tmp_path,
        root_kind=fit_lens.TEST_BUNDLE_ROOT_KIND,
    )
    evidence.atomic_write_json(provenance_path, provenance)
    with pytest.raises(fit_lens.IntegrityError, match="HASH_BOUND"):
        evaluate.validate_evaluation_provenance(
            output_root,
            validation_root=tmp_path,
            model=model,
        )


def test_eval_round_cpu_verifier_requires_nested_schema2_provenance(tmp_path):
    """The wrapper must consume the current nested CPU design, not legacy fields."""

    script = Path("src/ouro_jlens/eval_round.sh").read_text()
    start = script.index("verify_probe_metadata() {")
    start = script.index("<<'PY'", start) + len("<<'PY'\n")
    end = script.index("\nPY\n", start)
    helper = script[start:end]

    arrays_path = tmp_path / "arrays.npz"
    rank = np.zeros((648, 192), dtype=np.int32)
    np.savez_compressed(
        arrays_path,
        probe_rank=rank,
        chosen_C=np.full((5, 192), 0.01),
        selection_accuracy=np.zeros((5, 192)),
        folds=np.zeros(648, dtype=np.int32),
        labels=np.zeros(648, dtype=np.int32),
        correct=np.zeros(648, dtype=np.bool_),
        jl_cand=rank,
        jl_one=rank,
        jl_vocab=rank,
        ll_cand=rank,
        ll_one=rank,
        ll_vocab=rank,
    )
    hidden_cache = tmp_path / "gpu_cache.npz"
    hidden_cache.write_bytes(b"cache")
    lens_scores = tmp_path / "lens_all648.npz"
    lens_scores.write_bytes(b"scores")
    lens_provenance = tmp_path / "lens_all648.provenance.json"
    lens_provenance.write_bytes(b"provenance")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "schema_version": 1,
        "lens_score_status": "FRESH_CURRENT_SOURCE_HASH_BOUND_LENS",
        "readouts": {"supervised_probe": {}},
        "input": evidence.file_record(arrays_path),
    }))

    source_names = (
        "src/ouro_jlens/probe_cv.py",
        "src/ouro_jlens/probe.py",
        "src/ouro_jlens/probe_report.py",
        "src/ouro_jlens/evidence.py",
    )
    source_records = []
    for name in source_names:
        record = evidence.file_record(Path(name))
        record["path"] = name
        source_records.append(record)
    runtime = {"python": __import__("platform").python_version()}
    import importlib.metadata

    for distribution in (
            "numpy", "torch", "scikit-learn", "threadpoolctl", "jlens", "transformers",
        "safetensors", "accelerate", "huggingface-hub",
    ):
        try:
            runtime[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            runtime[distribution] = "NOT_INSTALLED"

    def logical_record(path, logical):
        record = evidence.file_record(path)
        record["path"] = logical
        return record

    design = tmp_path / "design.json"
    design.write_text(json.dumps({
        "schema_version": 2,
        "status": "FRESH_CURRENT_SOURCE_INNER_SELECTED_OUTER_SCORED",
        "seed": 0,
        "design": {},
        "source": {
            "files": source_records,
            "sha256": evidence.aggregate_sha256({r["path"]: r["sha256"] for r in source_records}),
        },
        "runtime": {"versions": runtime},
        "inputs": {
            "gpu_cache": logical_record(hidden_cache, "gpu_cache"),
            "lens_scores": logical_record(lens_scores, "lens_scores"),
            "lens_score_provenance": logical_record(lens_provenance, "lens_score_provenance"),
        },
        "outputs": {"arrays": logical_record(arrays_path, "probe_arrays")},
    }))
    command = [
        sys.executable,
        "-",
        "cpu",
        str(arrays_path),
        str(design),
        str(summary),
        str(hidden_cache),
        str(lens_scores),
        str(lens_provenance),
    ]
    environment = {**os.environ, "PYTHONPATH": "src"}
    valid = subprocess.run(
        command, input=helper, text=True, capture_output=True, env=environment, check=False
    )
    assert valid.returncode == 0, valid.stderr

    legacy = json.loads(design.read_text())
    legacy["schema_version"] = 1
    legacy["lens_score_status"] = "FRESH_CURRENT_SOURCE"
    design.write_text(json.dumps(legacy))
    rejected = subprocess.run(
        command, input=helper, text=True, capture_output=True, env=environment, check=False
    )
    assert rejected.returncode == 1
    assert "unsupported probe CPU design schema" in rejected.stderr


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


def _strict_manifest_fixture(monkeypatch, tmp_path):
    """Build a complete synthetic custody manifest for verifier tests."""

    monkeypatch.setattr(manifest, "PROJECT_ROOT", tmp_path)
    groups = {}
    required_paths = []
    for group in sorted(manifest.REQUIRED_GROUPS - {"external_jlens_source"}):
        relative = Path(f"{group}.bin")
        payload = tmp_path / relative
        payload.write_bytes(group.encode())
        groups[group] = [relative]
        required_paths.append(str(relative))
    external = tmp_path / "jlens_external.py"
    external.write_bytes(b"external jlens source")
    groups["external_jlens_source"] = [external]
    required_paths.append(str(external))

    monkeypatch.setattr(manifest, "REQUIRED_PATHS", tuple(required_paths))
    monkeypatch.setattr(
        manifest,
        "inventory",
        lambda: {group: paths for group, paths in groups.items()
                 if group != "external_jlens_source"},
    )
    monkeypatch.setattr(manifest, "_jlens_identity", lambda: ([external], "jlens-test-r1"))
    monkeypatch.setattr(manifest, "_tracked", lambda _path: True)
    monkeypatch.setattr(manifest, "_git_head", lambda: "head-test")
    monkeypatch.setattr(manifest, "_versions", lambda: {"python": "test"})
    monkeypatch.setattr(manifest, "OURO_REVISION", "model-test-r1")
    monkeypatch.setattr(manifest, "_manifest_current_validation_status", lambda: "NOT_ESTABLISHED")
    monkeypatch.setattr(manifest, "_probe_score_provenance_status", lambda: "TEST_PROVENANCE")

    flat = {}
    declared = {}
    for group, paths in groups.items():
        declared[group] = []
        for path in paths:
            candidate = path if path.is_absolute() else tmp_path / path
            actual = evidence.file_record(candidate)
            record = {
                "path": str(path),
                "size": actual["size"],
                "sha256": actual["sha256"],
                "tracked": True,
            }
            declared[group].append(record)
            flat[record["path"]] = record["sha256"]
    document = {
        "schema_version": manifest.SCHEMA_VERSION,
        "status": "LOCAL_CUSTODY_BOUND_PREEXISTING_EMPIRICAL_ARTIFACTS_UNFROZEN",
        "scientific_effect": "NONE; byte identity is not semantic acceptance",
        "project_head": "head-test",
        "model_revision": "model-test-r1",
        "jlens_revision": "jlens-test-r1",
        "environment": {"python": "test"},
        "hardware": {},
        "reproduction_contract": {
            "seeds": {"analysis_bootstrap": 0, "probe_cv": 0, "probe_bootstrap": 0},
            "commands": ["test"],
        },
        "groups": declared,
        "required_groups": sorted(manifest.REQUIRED_GROUPS),
        "required_paths": required_paths,
        "inventory_sha256": evidence.sha256_json(manifest._inventory_paths(groups)),
        "aggregate_sha256": evidence.sha256_json(flat),
        "current_validation_status": "NOT_ESTABLISHED",
        "probe_score_provenance": "TEST_PROVENANCE",
    }
    path = tmp_path / "MANIFEST.json"
    evidence.atomic_write_json(path, document)
    return path, groups


def test_manifest_verifier_detects_tampering(monkeypatch, tmp_path):
    manifest_path, groups = _strict_manifest_fixture(monkeypatch, tmp_path)
    assert manifest.verify_manifest(manifest_path)["status"] == "PASS"
    payload = tmp_path / groups["source"][0]
    payload.write_bytes(b"tampered")
    result = manifest.verify_manifest(manifest_path)
    assert result["status"] == "FAIL"
    assert result["mismatches"]


def test_manifest_verifier_rejects_inventory_drift(monkeypatch, tmp_path):
    manifest_path, groups = _strict_manifest_fixture(monkeypatch, tmp_path)
    assert manifest.verify_manifest(manifest_path)["status"] == "PASS"
    drift = tmp_path / "new-source.py"
    drift.write_text("new source")
    current = {group: list(paths) for group, paths in groups.items()
               if group != "external_jlens_source"}
    current["source"].append(Path("new-source.py"))
    monkeypatch.setattr(
        manifest, "inventory", lambda: current,
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
