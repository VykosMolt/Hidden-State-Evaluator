"""Offline tests for terminated-run extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouro_jlens import pod, recover
from ouro_jlens.publish import Receipt, canonical_json, payload_remote_path


IMAGE = pod.IMAGE or "runpod/pytorch@sha256:" + "a" * 64
TOKEN = "hf_test_token"
MANIFEST = "a" * 64
SOURCE_HEAD = "b" * 40


class FakePublisher:
    instances: list["FakePublisher"] = []
    preflight_data: bytes = b""

    def __init__(self, repository: str, token: str | None = None):
        self.repository = repository
        self.token = token
        self.downloads: list[tuple[str, int | None]] = []
        self.__class__.instances.append(self)

    def download_file(self, remote_path: str, destination: Path, *, expected_size: int | None = None) -> None:
        self.downloads.append((remote_path, expected_size))
        if self.repository == "org/staging":
            Path(destination).write_bytes(b"stage archive supplied to the patched bootstrap")
        else:
            Path(destination).write_bytes(self.preflight_data)


def _provider_deadline() -> tuple[str, int]:
    provider = "2030-01-01T00:00:00Z"
    # Keep this independent of the wall clock: the recovery validator checks
    # only the exact arithmetic binding, because a terminated run may be
    # recovered after the worker deadline has elapsed.
    from datetime import datetime, timezone

    provider_epoch = int(datetime.strptime(provider, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    return provider, provider_epoch - pod.WORKER_DEADLINE_MARGIN_SECONDS


def _state_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    token_path = tmp_path / "hf-token.txt"
    token_path.write_text(TOKEN + "\n", encoding="utf-8")
    token_path.chmod(0o600)
    sync_root = tmp_path / "sync"
    receipt_root = tmp_path / "receipts"
    attempt = "attempt-1"
    artifact = "artifact-1"
    nonce = "n" * 48
    provider_deadline, worker_deadline = _provider_deadline()
    token_sha = hashlib.sha256(TOKEN.encode()).hexdigest()
    preflight, preflight_data = recover._preflight_identity(
        repository="org/results",
        attempt_id=attempt,
        artifact_run_id=artifact,
        launch_nonce=nonce,
        stage_manifest_sha256=MANIFEST,
        stage_source_head=SOURCE_HEAD,
        hf_token_sha256=token_sha,
    )
    FakePublisher.preflight_data = preflight_data
    contract = {
        "schema_version": 1,
        "attempt_id": attempt,
        "artifact_run_id": artifact,
        "launch_nonce": nonce,
        "image": IMAGE,
        "stage_manifest_sha256": MANIFEST,
        "stage_source_head": SOURCE_HEAD,
        "staging_repository": "org/staging",
        "stage_path": f"stages/{MANIFEST}/stage.tar.gz",
        "results_repository": "org/results",
        "n_prompts": 100,
        "hf_token_path": str(token_path.absolute()),
        "hf_token_sha256": token_sha,
        "sync_root": str(sync_root.absolute()),
        "receipt_root": str(receipt_root.absolute()),
        "results_preflight": preflight,
    }
    state = {
        "schema_version": 2,
        "run_id": attempt,
        "attempt_id": attempt,
        "lease_attempt_id": attempt,
        "artifact_run_id": artifact,
        "artifact_lineage_id": artifact,
        "pod_name": pod._name_for(attempt),
        "status": "terminated",
        "termination_verified": True,
        "deployment_started": True,
        "launch_nonce": nonce,
        "image": IMAGE,
        "stage_manifest_sha256": MANIFEST,
        "stage_source_head": SOURCE_HEAD,
        "hf_token_sha256": token_sha,
        "gpu": pod.DEFAULT_GPU,
        "provider_gpu_display_name": pod.DEFAULT_GPU,
        "pod_id": "pod-1",
        "machine_id": "machine-1",
        "provider_terminate_after": provider_deadline,
        "worker_deadline_epoch": worker_deadline,
        "extraction_contract": contract,
        "results_preflight": preflight,
    }
    state_path = tmp_path / "state.json"
    state_path.write_bytes(canonical_json(state))
    state_path.chmod(0o600)
    return state_path, state, contract


def _install_stage_patches(monkeypatch, tmp_path: Path, contract: dict) -> None:
    stage_root = tmp_path / "verified-stage"
    (stage_root / "src/ouro_jlens").mkdir(parents=True)
    bootstrap = pod.StageBootstrap(
        package=stage_root / "src/ouro_jlens",
        root=stage_root,
        manifest_sha256=MANIFEST,
        source_head=SOURCE_HEAD,
        committed_payload_sha256={},
    )

    def prepare(stage_path, staging_repository, destination, *, publisher=None):
        assert stage_path == contract["stage_path"]
        assert staging_repository == contract["staging_repository"]
        assert isinstance(publisher, FakePublisher)
        return bootstrap

    monkeypatch.setattr(pod, "_prepare_stage_bootstrap", prepare)
    monkeypatch.setattr(
        pod,
        "verify_stage_root",
        lambda root, allow_extra=False: {
            "manifest_sha256": MANIFEST,
            "pinned_inputs": {"source": {"head": SOURCE_HEAD}},
        },
    )


def _install_sync_patch(monkeypatch, contract: dict, state: dict) -> dict:
    observed: dict = {}
    kinds = sorted(pod.REQUIRED_SUCCESS_RECEIPT_KINDS)
    receipts = [
        Receipt(
            schema_version=1,
            run_id=contract["artifact_run_id"],
            relative_path=f"final/{index:02d}-{kind}.json",
            remote_path=payload_remote_path(
                contract["artifact_run_id"], f"final/{index:02d}-{kind}.json"
            ),
            kind=kind,
            sha256="a" * 64,
            size=1,
        )
        for index, kind in enumerate(kinds)
    ]

    def sync(publisher, **kwargs):
        observed.update(kwargs)
        assert kwargs["compute_exit_code"] == 0
        assert kwargs["termination_verified"] is True
        assert kwargs["replay_source_root"].name == "verified-stage"
        receipt_root = Path(kwargs["receipt_root"])
        replay_dir = receipt_root / "local-replay"
        snapshots = receipt_root / "snapshots"
        replay_dir.mkdir(parents=True, exist_ok=True)
        snapshots.mkdir(parents=True, exist_ok=True)
        replay_payload = {
            "status": "PASS",
            "accepted": True,
            "run_id": contract["artifact_run_id"],
            "attempt_id": contract["attempt_id"],
            "launch_nonce": contract["launch_nonce"],
            "stage_manifest_sha256": contract["stage_manifest_sha256"],
            "stage_source_head": contract["stage_source_head"],
            "worker_deadline_epoch": state["worker_deadline_epoch"],
            "image_digest": contract["image"],
        }
        replay_data = canonical_json(replay_payload)
        replay_digest = hashlib.sha256(replay_data).hexdigest()
        replay_path = replay_dir / f"{replay_digest}.json"
        replay_path.write_bytes(replay_data)
        replay_path.chmod(0o600)
        records = [
            {
                "schema_version": receipt.schema_version,
                "run_id": receipt.run_id,
                "relative_path": receipt.relative_path,
                "remote_path": receipt.remote_path,
                "kind": receipt.kind,
                "sha256": receipt.sha256,
                "size": receipt.size,
            }
            for receipt in sorted(receipts, key=lambda item: item.relative_path)
        ]
        provider_lease = {
            "requested_gpu": pod.DEFAULT_GPU,
            "pod_id": state["pod_id"],
            "machine_id": state["machine_id"],
            "pod_name": state["pod_name"],
            "image": contract["image"],
            "gpu_display_name": pod.DEFAULT_GPU,
            "provider_terminate_after": state["provider_terminate_after"],
        }
        snapshot = {
            "schema_version": 1,
            "run_id": contract["artifact_run_id"],
            "attempt_id": contract["attempt_id"],
            "launch_nonce": contract["launch_nonce"],
            "stage_manifest_sha256": contract["stage_manifest_sha256"],
            "stage_source_head": contract["stage_source_head"],
            "worker_deadline_epoch": state["worker_deadline_epoch"],
            "compute_exit_code": 0,
            "termination_verified": True,
            "complete": True,
            "error": None,
            "provider_lease": provider_lease,
            "local_paid_replay": {
                "path": str(replay_path),
                "size": len(replay_data),
                "sha256": replay_digest,
                "status": "PASS",
                "accepted": True,
            },
            "receipts": records,
        }
        snapshot_data = canonical_json(snapshot)
        snapshot_digest = hashlib.sha256(snapshot_data).hexdigest()
        snapshot_path = snapshots / f"{snapshot_digest}.json"
        snapshot_path.write_bytes(snapshot_data)
        snapshot_path.chmod(0o600)
        return receipts

    monkeypatch.setattr(pod, "_sync_terminated_run", sync)
    return observed


def test_recovery_is_offline_from_runpod_and_repeats_with_same_bytes(tmp_path, monkeypatch):
    state_path, state, contract = _state_fixture(tmp_path)
    _install_stage_patches(monkeypatch, tmp_path, contract)
    observed = _install_sync_patch(monkeypatch, contract, state)
    FakePublisher.instances = []
    monkeypatch.setattr(pod, "RunPodClient", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RunPod client")))
    monkeypatch.setattr(pod, "api_key", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RunPod key")))
    monkeypatch.setattr(recover, "HfPublisher", FakePublisher)

    args = SimpleNamespace(state_file=state_path)
    assert recover.recover_terminated_run(args) == 0
    assert recover.recover_terminated_run(args) == 0
    assert observed["run_id"] == contract["artifact_run_id"]
    assert observed["attempt_id"] == contract["attempt_id"]
    assert observed["image"] == IMAGE
    assert observed["worker_deadline_epoch"] == state["worker_deadline_epoch"]
    assert observed["compute_exit_code"] == 0
    assert observed["termination_verified"] is True
    assert [(item.repository, item.token) for item in FakePublisher.instances] == [
        ("org/staging", TOKEN),
        ("org/results", TOKEN),
        ("org/staging", TOKEN),
        ("org/results", TOKEN),
    ]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda state: state.update({"status": "active"}), "terminated"),
        (lambda state: state["extraction_contract"].update({"unknown": True}), "exact"),
        (lambda state: state.update({"termination_verified": False}), "verified"),
        (lambda state: state.update({"worker_deadline_epoch": state["worker_deadline_epoch"] + 1}), "worker deadline"),
    ],
)
def test_recovery_rejects_unbound_or_unfinished_state(tmp_path, monkeypatch, mutation, message):
    state_path, state, _contract = _state_fixture(tmp_path)
    mutation(state)
    state_path.write_bytes(canonical_json(state))
    state_path.chmod(0o600)
    with pytest.raises(recover.RecoveryError, match=message):
        recover.recover_terminated_run(SimpleNamespace(state_file=state_path))


def test_recovery_rejects_token_fingerprint_change_without_publisher(tmp_path, monkeypatch):
    state_path, state, _contract = _state_fixture(tmp_path)
    token_path = Path(state["extraction_contract"]["hf_token_path"])
    token_path.write_text("hf_changed_token\n", encoding="utf-8")
    token_path.chmod(0o600)
    monkeypatch.setattr(recover, "HfPublisher", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("publisher")))
    with pytest.raises(recover.RecoveryError, match="fingerprint"):
        recover.recover_terminated_run(SimpleNamespace(state_file=state_path))


def test_recovery_requires_one_explicit_regular_state_file(tmp_path):
    state_path, _state, _contract = _state_fixture(tmp_path)
    with pytest.raises(recover.RecoveryError, match="exactly one"):
        recover.recover_terminated_run(SimpleNamespace())
    linked = tmp_path / "linked-state.json"
    linked.symlink_to(state_path)
    with pytest.raises(recover.RecoveryError, match="symlink"):
        recover.recover_terminated_run(SimpleNamespace(state_file=linked))


def test_controller_recover_subcommand_never_activates_runpod_key(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        pod, "_activate_key_selector",
        lambda _args: (_ for _ in ()).throw(AssertionError("RunPod selector activated")),
    )
    monkeypatch.setattr(
        recover, "recover_terminated_run",
        lambda args: observed.append(Path(args.state_file)) or 0,
    )
    assert pod.main(["recover", "--state", str(state_path)]) == 0
    assert observed == [state_path]


def test_recovery_rejects_preflight_readback_mismatch_before_sync(tmp_path, monkeypatch):
    state_path, state, contract = _state_fixture(tmp_path)
    _install_stage_patches(monkeypatch, tmp_path, contract)
    sync_called = False

    def sync(*args, **kwargs):
        nonlocal sync_called
        sync_called = True
        return []

    class WrongPreflight(FakePublisher):
        def download_file(self, remote_path, destination, *, expected_size=None):
            if self.repository == "org/results":
                Path(destination).write_bytes(b"wrong preflight")
            else:
                super().download_file(remote_path, destination, expected_size=expected_size)

    monkeypatch.setattr(recover, "HfPublisher", WrongPreflight)
    monkeypatch.setattr(pod, "_sync_terminated_run", sync)
    with pytest.raises(recover.RecoveryError, match="preflight"):
        recover.recover_terminated_run(SimpleNamespace(state_file=state_path))
    assert sync_called is False
