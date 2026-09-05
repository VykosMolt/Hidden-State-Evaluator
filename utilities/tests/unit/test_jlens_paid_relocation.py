"""Focused tests for paid-artifact relocation and immutable launch binding."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ouro_jlens import evaluate, report, verify_paid
from ouro_jlens.evidence import file_record, sha256_json


def _record(path: Path, logical: str) -> dict[str, object]:
    result = file_record(path)
    result["path"] = logical
    result["root_kind"] = "project"
    return result


def test_artifact_record_uses_relocated_bytes_but_source_stays_trusted(tmp_path: Path):
    project = tmp_path / "project"
    trusted = project / "src/ouro_jlens/evaluate.py"
    trusted.parent.mkdir(parents=True)
    trusted.write_bytes(b"trusted source")
    stale = project / "artifacts/jlens/eval/arrays.npz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale checkout artifact")

    relocated = tmp_path / "synchronized"
    remote = relocated / "eval/arrays.npz"
    remote.parent.mkdir(parents=True)
    remote.write_bytes(b"synchronized artifact")

    artifact_record = _record(remote, "artifacts/jlens/eval/arrays.npz")
    assert evaluate._record_matches_artifact(
        artifact_record,
        root=project,
        label="lens artifact",
        artifact_root=relocated,
    ) == remote

    source_record = _record(trusted, "src/ouro_jlens/evaluate.py")
    assert evaluate._record_matches_artifact(
        source_record,
        root=project,
        label="source",
        artifact_root=relocated,
    ) == trusted


@pytest.mark.parametrize(
    "logical",
    [
        "artifacts/jlens/../escape.bin",
        "artifacts/jlens/./escape.bin",
        "artifacts/jlens//escape.bin",
    ],
)
def test_artifact_path_rejects_traversal_and_noncanonical_spelling(
    tmp_path: Path, logical: str,
):
    root = tmp_path / "bundle"
    root.mkdir()
    with pytest.raises(ValueError, match="canonical|linked|missing"):
        evaluate.artifact_path(logical, root)


def test_artifact_path_rejects_symlink_components(tmp_path: Path):
    root = tmp_path / "bundle"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "bytes.bin").write_bytes(b"outside")
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|linked"):
        evaluate.artifact_path("artifacts/jlens/link/bytes.bin", root)


def test_relocated_merged_sidecar_adaptation_is_idempotent(tmp_path: Path):
    root = tmp_path / "bundle"
    root.mkdir()
    binary = root / "lens.pt"
    sidecar = root / "lens.json"
    binary.write_bytes(b"lens bytes")
    sidecar.write_bytes(b"sidecar bytes")
    binary_record = file_record(binary)
    binary_record.update({"path": "artifacts/jlens/lens.pt", "root_kind": "project"})
    shard_record = {
        "path": "artifacts/jlens/lens.pt",
        "root_kind": "project",
        "start": 0,
        "end": 1,
        "count": 1,
        "binary_sha256": binary_record["sha256"],
        "binary_size": binary_record["size"],
        "sidecar_path": "artifacts/jlens/lens.json",
        "sidecar_root_kind": "project",
        "sidecar_sha256": file_record(sidecar)["sha256"],
        "output_sha256": binary_record["sha256"],
    }
    metadata = {
        "output": binary_record,
        "output_sha256": binary_record["sha256"],
        "shards": [binary_record["path"]],
        "shard_records": [shard_record],
        "shard_records_sha256": sha256_json([shard_record]),
    }
    first = evaluate._relocated_sidecar_metadata(metadata, binary, root)
    second = evaluate._relocated_sidecar_metadata(first, binary, root)
    assert second == first


def test_relocation_adapter_restores_fit_lens_globals_on_failure(tmp_path: Path):
    import ouro_jlens.fit_lens as fit_lens

    root = tmp_path / "bundle"
    root.mkdir()
    original_reader = fit_lens._read_json
    with pytest.raises(RuntimeError, match="sentinel"):
        with evaluate.relocated_lens_validation(root):
            assert fit_lens._read_json is not original_reader
            raise RuntimeError("sentinel")
    assert fit_lens._read_json is original_reader


def test_checkpoints_none_is_omitted_and_bindings_are_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RUN_ID", "relocation-test")
    monkeypatch.setenv("JLENS_IMAGE_DIGEST", "registry.invalid/b300@sha256:" + "1" * 64)
    monkeypatch.setenv("JLENS_LAUNCH_NONCE", "nonce_01-safe-" + "x" * 20)
    monkeypatch.setenv("EXPECTED_STAGE_MANIFEST_SHA256", "2" * 64)
    monkeypatch.setenv("EXPECTED_STAGE_SOURCE_HEAD", "3" * 40)
    monkeypatch.setenv("JLENS_WORKER_DEADLINE_EPOCH", str(int(time.time()) + 3600))
    monkeypatch.setattr(verify_paid, "approved_b300_image_digest", lambda: os.environ["JLENS_IMAGE_DIGEST"])

    eval_root = tmp_path / "eval"
    lens_root = tmp_path / "lens" / "n100"
    (eval_root / "n100_allexits").mkdir(parents=True)
    (eval_root / "fitsize_n80").mkdir(parents=True)
    (eval_root / "fitsize_n80" / "provenance.json").write_text("{}", encoding="utf-8")
    lens_root.mkdir(parents=True)
    analysis = tmp_path / "analysis.json"
    claims = tmp_path / "claims.json"
    validation = tmp_path / "validation.json"
    transport = tmp_path / "transport.json"
    fit_size = tmp_path / "fit-size.json"
    analysis.write_text('{"claims": {}}', encoding="utf-8")
    claims.write_text("{}", encoding="utf-8")
    for path in (validation, transport, fit_size):
        path.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(verify_paid, "_validation_gate", lambda *args: {"status": "ok"})
    monkeypatch.setattr(verify_paid, "_validate_prefix_lenses", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(verify_paid, "_validate_evaluation_set", lambda *args, **kwargs: {"fitsize_n80": {}})
    monkeypatch.setattr(verify_paid, "_verify_transport_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify_paid, "_verify_fit_size_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify_paid, "_validate_analysis_shape", lambda *args: None)
    monkeypatch.setattr(verify_paid.verify_artifacts, "verify_main", lambda *args: {"total_items": 0})

    def fake_build_report(*args, **kwargs):
        captured.update(kwargs)
        return {"claims": {}}

    monkeypatch.setattr(report, "build_report", fake_build_report)
    result = verify_paid.verify_paid(
        main=eval_root / "fitsize_n80",
        local_eval=eval_root / "n100_allexits",
        eval_root=eval_root,
        lens_root=lens_root,
        validation=validation,
        transport=transport,
        analysis=analysis,
        claims=claims,
        fit_size=fit_size,
        checkpoints=None,
    )

    assert captured["checkpoints"] is None
    assert result["launch_nonce"] == "nonce_01-safe-" + "x" * 20
    assert result["stage_manifest_sha256"] == "2" * 64
    assert result["stage_source_head"] == "3" * 40
    assert result["worker_deadline_epoch"] == int(os.environ["JLENS_WORKER_DEADLINE_EPOCH"])


@pytest.mark.parametrize(
    "variable,value",
    [
        ("JLENS_LAUNCH_NONCE", "bad nonce"),
        ("EXPECTED_STAGE_MANIFEST_SHA256", "A" * 64),
        ("EXPECTED_STAGE_SOURCE_HEAD", "4" * 39),
        ("JLENS_WORKER_DEADLINE_EPOCH", "not-an-epoch"),
        ("JLENS_WORKER_DEADLINE_EPOCH", "0"),
    ],
)
def test_paid_bindings_require_strict_values(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str,
):
    monkeypatch.setenv("JLENS_LAUNCH_NONCE", "n" * 32)
    monkeypatch.setenv("EXPECTED_STAGE_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("EXPECTED_STAGE_SOURCE_HEAD", "b" * 40)
    monkeypatch.setenv("JLENS_WORKER_DEADLINE_EPOCH", str(int(time.time()) + 3600))
    monkeypatch.setenv(variable, value)
    with pytest.raises(verify_paid.PaidVerificationError):
        verify_paid._launch_bindings()


def test_paid_bindings_accept_64_hex_git_head(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JLENS_LAUNCH_NONCE", "n" * 32)
    monkeypatch.setenv("EXPECTED_STAGE_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("EXPECTED_STAGE_SOURCE_HEAD", "b" * 64)
    monkeypatch.setenv("JLENS_WORKER_DEADLINE_EPOCH", str(int(time.time()) + 3600))
    assert verify_paid._launch_bindings() == {
        "launch_nonce": "n" * 32,
        "stage_manifest_sha256": "a" * 64,
        "stage_source_head": "b" * 64,
        "worker_deadline_epoch": int(os.environ["JLENS_WORKER_DEADLINE_EPOCH"]),
    }


def test_evaluation_deadline_binding_is_checked_against_worker(
    monkeypatch: pytest.MonkeyPatch,
):
    deadline = int(time.time()) + 3600
    monkeypatch.setenv("JLENS_WORKER_DEADLINE_EPOCH", str(deadline))
    assert evaluate._check_worker_deadline({"worker_deadline_epoch": deadline}) == deadline
    with pytest.raises(ValueError, match="does not match"):
        evaluate._check_worker_deadline({"worker_deadline_epoch": deadline + 1})
    with pytest.raises(ValueError, match="no worker deadline"):
        evaluate._check_worker_deadline({})


def test_offline_deadline_binding_accepts_expired_positive_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    """Offline replay may verify a completed worker after its live cutoff."""

    monkeypatch.setenv("JLENS_WORKER_DEADLINE_EPOCH", "1")
    assert evaluate._check_worker_deadline({"worker_deadline_epoch": 1}) == 1
