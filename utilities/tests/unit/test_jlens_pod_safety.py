"""Focused offline tests for the lease-attempt safety boundary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouro_jlens import pod, publish


IMAGE = pod.IMAGE or "runpod/pytorch@sha256:" + "a" * 64


def _future_provider_deadline(seconds: float = 1800.0) -> str:
    return pod._provider_termination_deadline(time.time(), seconds)


def _pod(name: str = "ouro-jlens-r1", pod_id: str = "p1", *, ports=None) -> dict:
    return {
        "id": pod_id,
        "name": name,
        "machineId": "m1",
        "machine": {"gpuDisplayName": "NVIDIA B300 SXM6 AC"},
        "costPerHr": 1.0,
        "desiredStatus": "RUNNING",
        "runtime": {"uptimeInSeconds": 0, "ports": list(ports or [])},
    }


class FakeClient:
    def __init__(self, *, balance: float = 100.0, pods=None, pod_sequence=None):
        self.current_balance = balance
        self.current_pods = list(pods or [])
        self.pod_sequence = list(pod_sequence or [])
        self.calls: list[object] = []
        self.deploy_calls: list[dict] = []
        self.terminate_calls: list[str] = []

    def balance(self):
        self.calls.append("balance")
        return self.current_balance

    def pods(self):
        self.calls.append("pods")
        if self.pod_sequence:
            return list(self.pod_sequence.pop(0))
        return list(self.current_pods)

    def deploy(self, **kwargs):
        self.calls.append("deploy")
        self.deploy_calls.append(kwargs)
        pod = _pod(kwargs["name"], "new-pod")
        self.current_pods = [pod]
        return pod

    def terminate(self, pod_id):
        self.calls.append(("terminate", pod_id))
        self.terminate_calls.append(pod_id)
        self.current_pods = [item for item in self.current_pods if item.get("id") != pod_id]


def _args(tmp_path: Path, *, run_id: str = "r1", image: str = IMAGE, command: str = "create"):
    pubkey = tmp_path / "id.pub"
    pubkey.write_text("ssh-ed25519 AAAA test\n")
    token = tmp_path / "hf-token"
    token.write_text("hf_test_token")
    token.chmod(0o600)
    identity = tmp_path / "id"
    identity.write_text("private-key\n")
    return pod._parser().parse_args([
        command, "--run-id", run_id, "--state-root", str(tmp_path / "states"),
        "--image", image, "--pubkey", str(pubkey), "--hf-token", str(token),
        "--identity-file", str(identity), "--min-balance", "20", "--max-spend", "20",
        "--poll-seconds", "0.01", "--termination-poll-seconds", "0",
    ])


@pytest.mark.parametrize("status", ["active", "terminated"])
def test_reused_state_path_is_rejected_even_after_termination(tmp_path, status):
    args = _args(tmp_path)
    state_file = pod.state_path("r1", tmp_path / "states")
    state_file.parent.mkdir(parents=True)
    state_file.write_text(json.dumps({"run_id": "r1", "pod_name": "ouro-jlens-r1", "status": status}))
    before = state_file.read_bytes()
    client = FakeClient()
    with pytest.raises(pod.SafetyError, match="state already exists"):
        pod._create_lease(args, client=client, monitor_launcher=lambda _path: "unit")
    assert state_file.read_bytes() == before
    assert "deploy" not in client.calls and "pods" not in client.calls


def test_same_name_pod_is_rejected_without_allow_second_escape(tmp_path):
    args = _args(tmp_path)
    client = FakeClient(pods=[_pod()])
    with pytest.raises(pod.SafetyError, match="concurrent JLens leases are disabled"):
        pod._create_lease(args, client=client, monitor_launcher=lambda _path: "unit")
    assert "deploy" not in client.calls
    assert not list((tmp_path / "states").rglob("state.json"))
    assert not hasattr(args, "allow_second")


def test_different_name_jlens_pod_also_blocks_a_second_lease(tmp_path):
    args = _args(tmp_path)
    client = FakeClient(pods=[_pod(name="ouro-jlens-another-attempt", pod_id="existing")])
    with pytest.raises(pod.SafetyError, match="concurrent JLens leases are disabled"):
        pod._create_lease(args, client=client, monitor_launcher=lambda _path: "unit")
    assert "deploy" not in client.calls
    assert not list((tmp_path / "states").rglob("state.json"))


def test_monitor_launch_failure_does_not_terminate_pod_that_appears_concurrently(tmp_path):
    args = _args(tmp_path)
    client = FakeClient()

    def launch(_path):
        client.current_pods = [_pod()]
        raise pod.SafetyError("monitor unavailable")

    with pytest.raises(pod.SafetyError, match="monitor unavailable"):
        pod._create_lease(args, client=client, monitor_launcher=launch)
    assert client.terminate_calls == []


def test_monitor_state_race_cannot_deploy_after_pending_attempt_was_terminated(
    tmp_path, monkeypatch
):
    args = _args(tmp_path)
    client = FakeClient()

    def launch(state_file):
        state = json.loads(state_file.read_text())
        state.update({"status": "terminated", "termination_verified": True})
        pod._atomic_json(state_file, state)
        return "unit"

    monkeypatch.setattr(pod, "start_systemd_monitor", launch)
    monkeypatch.setattr(pod, "stop_systemd_monitor", lambda unit: None)
    with pytest.raises(pod.SafetyError, match="terminated the pending attempt"):
        pod._create_lease(args, client=client)
    assert "deploy" not in client.calls
    assert client.terminate_calls == []


def test_systemd_monitor_rejects_symlinked_state_before_runner(tmp_path):
    real = tmp_path / "state.json"
    real.write_text('{"run_id":"r1","pod_name":"ouro-jlens-r1"}')
    linked = tmp_path / "linked.json"
    linked.symlink_to(real)

    def runner(*args, **kwargs):
        raise AssertionError("runner must not execute for a linked state")

    with pytest.raises(pod.SafetyError, match="symlink"):
        pod.start_systemd_monitor(linked, runner=runner)


def test_preflight_requires_reserve_plus_maximum_spend(tmp_path):
    args = _args(tmp_path)
    client = FakeClient(balance=39.99)
    with pytest.raises(pod.SafetyError, match="minimum balance plus maximum spend"):
        pod._create_lease(args, client=client, monitor_launcher=lambda _path: "unit")
    assert "pods" not in client.calls
    assert not list((tmp_path / "states").rglob("state.json"))


def test_invalid_local_argument_is_rejected_before_lease_mutation(tmp_path):
    args = _args(tmp_path)
    args.disk = 0
    client = FakeClient()
    with pytest.raises(pod.SafetyError, match="disk"):
        pod._create_lease(args, client=client, monitor_launcher=lambda _path: "unit")
    assert "deploy" not in client.calls and "pods" not in client.calls and "balance" not in client.calls
    assert not list((tmp_path / "states").rglob("state.json"))


def test_create_non_dry_is_disabled_and_dry_run_stays_offline(tmp_path, monkeypatch, capsys):
    args = _args(tmp_path)
    client = FakeClient()
    monkeypatch.setattr(pod, "RunPodClient", lambda: client)
    with pytest.raises(pod.SafetyError, match="provision-only create"):
        pod.create(args)
    assert client.calls == []

    dry = pod._parser().parse_args(["create", "--dry-run", "--run-id", "offline"])
    monkeypatch.setattr(pod, "api_key", lambda: (_ for _ in ()).throw(AssertionError("credential read")))
    pod.create(dry)
    assert "DRY RUN" in capsys.readouterr().out


def test_mutable_image_is_rejected_before_deploy(tmp_path):
    args = _args(tmp_path, image="runpod/pytorch:latest")
    client = FakeClient()
    with pytest.raises(pod.SafetyError, match="immutable image"):
        pod._create_lease(args, client=client, monitor_launcher=lambda _path: "unit")
    assert "deploy" not in client.calls
    assert not list((tmp_path / "states").rglob("state.json"))


def test_paid_run_requires_exact_attempt_confirmation_before_stage_or_account_access(
    tmp_path, monkeypatch,
):
    args = pod._parser().parse_args([
        "run", "--run-id", "attempt-1", "--results", "org/results",
        "--stage-path", "stages/" + "a" * 64 + "/stage.tar.gz",
        "--state-root", str(tmp_path), "--image", IMAGE,
    ])
    monkeypatch.setattr(
        pod,
        "_prepare_stage_bootstrap",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("stage access")),
    )
    with pytest.raises(pod.SafetyError, match="confirm-launch"):
        pod.run_job(args)
    args.confirm_launch = "another-attempt"
    with pytest.raises(pod.SafetyError, match="confirm-launch"):
        pod.run_job(args)


def test_paid_run_rejects_same_lineage_retry_before_stage_or_account_access(
    tmp_path,
    monkeypatch,
):
    args = pod._parser().parse_args([
        "run",
        "--run-id", "attempt-2",
        "--artifact-run-id", "artifact-1",
        "--confirm-launch", "attempt-2",
        "--results", "org/results",
        "--stage-path", "stages/" + "a" * 64 + "/stage.tar.gz",
        "--state-root", str(tmp_path),
        "--image", IMAGE,
    ])
    monkeypatch.setattr(
        pod,
        "_prepare_stage_bootstrap",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("stage access")),
    )
    with pytest.raises(pod.SafetyError, match="same-lineage retry is disabled"):
        pod.run_job(args)


def test_eventual_ssh_listing_receives_immutable_image_and_torch_environment(tmp_path, monkeypatch):
    args = _args(tmp_path)
    key = "rpa_" + "K" * 24
    key_file = tmp_path / "fidelio.txt"
    key_file.write_text(key + "\n")
    key_file.chmod(0o600)
    fingerprint = hashlib.sha256(key.encode()).hexdigest()
    monkeypatch.setattr(pod, "KEY_FILE", key_file)
    monkeypatch.setenv(pod.KEY_SELECTOR_ENV, fingerprint)
    args.runpod_key_sha256 = fingerprint
    ready_port = {"ip": "127.0.0.1", "publicPort": 2222, "privatePort": 22, "isIpPublic": True}
    created = _pod("ouro-jlens-r1", "new-pod")
    waiting = {**created, "runtime": {"uptimeInSeconds": 0, "ports": []}}
    ready = {**created, "runtime": {"uptimeInSeconds": 0, "ports": [ready_port]}}
    client = FakeClient(pod_sequence=[[], [], [], [waiting], [ready]])
    def launch(state_file):
        pod.LeaseSupervisor(client, state_file, pod._lease_config(args),
                            sleep=lambda _seconds: None)._authenticated_ready(
                                pod._read_json(state_file)
                            )
        return "ouro-jlens-monitor-r1.service"

    monkeypatch.setattr(pod, "verify_systemd_monitor_active", lambda _unit: None)
    state, supervisor = pod._create_lease(args, client=client, monitor_launcher=launch)
    assert state["image"] == IMAGE
    assert state["torch_version"] == pod.TORCH_VERSION
    assert client.deploy_calls[0]["image"] == IMAGE
    assert client.deploy_calls[0]["torch_version"] == pod.TORCH_VERSION
    assert client.deploy_calls[0]["gpu"] == pod.DEFAULT_GPU
    assert pod.PROVIDER_DATETIME_RE.fullmatch(client.deploy_calls[0]["terminate_after"])
    supervisor.terminate_verified()


def test_failed_run_sync_uses_remote_enumeration_without_local_index_fallback(tmp_path, monkeypatch):
    args = pod._parser().parse_args([
        "run", "--run-id", "r1", "--confirm-launch", "r1",
        "--results", "org/results", "--staging", "org/staging",
        "--stage-path", "stages/" + "a" * 64 + "/stage.tar.gz", "--state-root", str(tmp_path),
        "--sync-root", str(tmp_path / "sync"), "--image", IMAGE, "--n-prompts", "1",
        "--dim-batch", "1",
    ])

    class Supervisor:
        def terminate_verified(self):
            pass

    monkeypatch.setattr(pod, "_prepare_stage_bootstrap", lambda *_a, **_k: SimpleNamespace(
        package=Path("src/ouro_jlens"), root=tmp_path, manifest_sha256="a" * 64,
        source_head="b" * 40, committed_payload_sha256={},
    ))
    monkeypatch.setattr(pod, "_verify_controller_source_matches_stage", lambda _bootstrap: None)
    monkeypatch.setattr(
        pod, "_preflight_results_repository",
        lambda *_a, **_k: {"schema_version": 1, "remote_path": "canary", "size": 1, "sha256": "c" * 64},
    )
    provider_deadline = _future_provider_deadline(12_600)
    monkeypatch.setattr(pod, "_create_lease", lambda _args: ({
        "run_id": "r1", "artifact_run_id": "r1", "ssh_ip": "127.0.0.1", "ssh_port": 2222,
        "monitor_unit": None, "launch_nonce": args.launch_nonce,
        "stage_manifest_sha256": "a" * 64, "stage_source_head": "b" * 40,
        "provider_terminate_after": provider_deadline,
        "worker_deadline_epoch": pod._worker_deadline_epoch(provider_deadline, now=time.time()),
    }, Supervisor()))
    commands = []
    def run(command, check=False, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 17 if "bash" in command else 0)
    monkeypatch.setattr(pod.subprocess, "run", run)

    class Publisher:
        def __init__(self):
            self.reads = []

        def read_file(self, path):
            self.reads.append(path)
            raise AssertionError("controller must not read a terminal index")

    publisher = Publisher()
    calls = []
    monkeypatch.setattr(pod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(pod, "HfPublisher", lambda _repo, **_kwargs: publisher)
    monkeypatch.setattr(
        pod, "sync_and_verify",
        lambda _publisher, *, run_id, local_root, relative_paths, receipt_root: calls.append(
            (run_id, Path(local_root), relative_paths, Path(receipt_root))
        ) or [],
    )
    assert pod.run_job(args) == 17
    assert calls == [
        ("r1", tmp_path / "sync", None, tmp_path / "sync.receipts"),
        ("r1", tmp_path / "sync", None, tmp_path / "sync.receipts"),
    ]
    snapshots = list((tmp_path / "sync.receipts" / "snapshots").glob("*.json"))
    assert len(snapshots) == 1
    snapshot = json.loads(snapshots[0].read_text())
    assert snapshot["complete"] is False
    assert snapshot["compute_exit_code"] == 17
    assert snapshot["termination_verified"] is True
    remote_command = next(command for command in commands if "bash" in command)
    assert "JLENS_ATTEMPT_ID=r1" in remote_command
    assert f"JLENS_IMAGE_DIGEST={IMAGE}" in remote_command
    assert publisher.reads == []


def test_successful_sync_retries_stale_listing_and_seals_complete_snapshot(tmp_path, monkeypatch):
    first = publish.Receipt(
        1, "artifact-1", "status/one.json", "artifact-1/artifacts/status/one.json",
        "heartbeat", "a" * 64, 1,
    )
    second = publish.Receipt(
        1, "artifact-1", "inventory/expected-inventory.json",
        "artifact-1/artifacts/inventory/expected-inventory.json",
        "expected_inventory", "b" * 64, 2,
    )
    observations = [[first], [first, second]]
    calls = []
    monkeypatch.setattr(
        pod,
        "sync_and_verify",
        lambda *_args, **kwargs: calls.append(kwargs) or observations.pop(0),
    )

    def verify(receipts, *_args, **_kwargs):
        if len(receipts) != 2:
            raise publish.PublishError("stale listing")

    monkeypatch.setattr(pod, "_verify_success_inventory", verify)
    monkeypatch.setattr(pod, "_verify_extracted_paid_gate", lambda *_a, **_k: None)
    monkeypatch.setattr(pod, "_verify_synced_tree_exact", lambda *_a, **_k: None)
    def replay(*args, **kwargs):
        payload = {"status": "PASS", "accepted": True}
        data = publish.canonical_json(payload)
        digest = hashlib.sha256(data).hexdigest()
        destination = Path(args[1]) / "local-replay" / f"{digest}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return payload

    monkeypatch.setattr(pod, "_run_local_paid_replay", replay)
    receipt_root = tmp_path / "custody"
    result = pod._sync_terminated_run(
        object(),
        run_id="artifact-1",
        attempt_id="attempt-1",
        local_root=tmp_path / "payloads",
        receipt_root=receipt_root,
        n_prompts=100,
        image=IMAGE,
        launch_nonce="n" * 48,
        stage_manifest_sha256="a" * 64,
        stage_source_head="b" * 40,
        worker_deadline_epoch=int(time.time()) + 3600,
        replay_source_root=tmp_path,
        compute_exit_code=0,
        termination_verified=True,
        attempts=3,
        retry_seconds=0,
    )
    assert {receipt.relative_path for receipt in result} == {
        "status/one.json", "inventory/expected-inventory.json"
    }
    assert len(calls) == 2
    snapshots = list((receipt_root / "snapshots").glob("*.json"))
    assert len(snapshots) == 1
    snapshot = json.loads(snapshots[0].read_text())
    assert snapshot["complete"] is True
    assert snapshot["termination_verified"] is True
    assert snapshot["extraction_profile"] == "full_replay"


def test_application_extraction_excludes_only_large_lens_payloads():
    def receipt(relative, kind, size=1):
        return publish.Receipt(
            1,
            "artifact-1",
            relative,
            f"artifact-1/artifacts/{relative}",
            kind,
            "a" * 64,
            size,
        )

    receipts = [
        receipt("lens/n100/exit3_shard_0000_0008.pt", "shard", 1_600_000_000),
        receipt("lens/n100/exit3.pt", "merged_lens", 1_600_000_000),
        receipt("lens/n100/exit3.pt.ckpt", "checkpoint", 3_200_000_000),
        receipt("lens/n100/exit3.json", "sidecar"),
        receipt("eval/fitsize_n80/arrays.npz", "eval_output", 20_000_000),
        receipt("final/analysis.json", "report"),
    ]
    assert pod._application_extraction_paths(receipts, run_id="artifact-1") == {
        "lens/n100/exit3.json",
        "eval/fitsize_n80/arrays.npz",
        "final/analysis.json",
    }


def test_successful_application_sync_validates_remote_union_without_tensor_download(
    tmp_path, monkeypatch
):
    report = publish.Receipt(
        1, "artifact-1", "final/analysis.json",
        "artifact-1/artifacts/final/analysis.json", "report", "a" * 64, 1,
    )
    tensor = publish.Receipt(
        1, "artifact-1", "lens/n100/exit3.pt",
        "artifact-1/artifacts/lens/n100/exit3.pt", "merged_lens", "b" * 64,
        1_600_000_000,
    )
    verified_remote = []

    class Publisher:
        def verify_remote_receipts_metadata(self, receipts):
            verified_remote.append(list(receipts))
            return {"revision": "c" * 40, "count": len(receipts)}

    publisher = Publisher()
    monkeypatch.setattr(
        pod, "_read_remote_receipt_inventory", lambda *_a, **_k: [report, tensor]
    )
    sync_calls = []
    monkeypatch.setattr(
        pod,
        "sync_and_verify",
        lambda *_a, **kwargs: sync_calls.append(kwargs) or [report],
    )
    verified = []
    monkeypatch.setattr(
        pod,
        "_verify_success_inventory",
        lambda receipts, *_a, **kwargs: verified.append((receipts, kwargs)),
    )
    monkeypatch.setattr(pod, "_verify_extracted_paid_gate", lambda *_a, **_k: None)
    monkeypatch.setattr(pod, "_verify_synced_tree_exact", lambda *_a, **_k: None)
    monkeypatch.setattr(
        pod,
        "_run_local_paid_replay",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("quick extraction must not replay tensors")),
    )
    result = pod._sync_terminated_run(
        publisher, run_id="artifact-1", attempt_id="attempt-1",
        local_root=tmp_path / "payloads", receipt_root=tmp_path / "custody",
        n_prompts=100, image=IMAGE, launch_nonce="n" * 48,
        stage_manifest_sha256="a" * 64, stage_source_head="b" * 40,
        worker_deadline_epoch=int(time.time()) + 3600, replay_source_root=tmp_path,
        compute_exit_code=0, termination_verified=True, attempts=1,
        retry_seconds=0, full_replay=False,
    )
    assert result == [report, tensor]
    assert sync_calls[0]["relative_paths"] == ["final/analysis.json"]
    assert verified_remote == [[tensor]]
    assert verified[0][1]["required_local_paths"] == {"final/analysis.json"}
    retained_tensor_receipt = (
        tmp_path / "custody" / "lens/n100/exit3.pt.receipt.json"
    )
    assert publish.Receipt.from_bytes(retained_tensor_receipt.read_bytes()) == tensor
    snapshot_path = next((tmp_path / "custody" / "snapshots").glob("*.json"))
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["extraction_profile"] == "application"
    assert snapshot["remote_payload_custody"] == {
        "revision": "c" * 40,
        "count": 1,
    }


def test_application_sync_unions_nonmonotonic_receipt_listings(tmp_path, monkeypatch):
    report = publish.Receipt(
        1, "artifact-1", "final/analysis.json",
        "artifact-1/artifacts/final/analysis.json", "report", "a" * 64, 1,
    )
    tensor = publish.Receipt(
        1, "artifact-1", "lens/n100/exit3.pt",
        "artifact-1/artifacts/lens/n100/exit3.pt", "merged_lens", "b" * 64, 10,
    )
    observations = [[report], [tensor]]
    monkeypatch.setattr(
        pod, "_read_remote_receipt_inventory", lambda *_a, **_k: observations.pop(0)
    )
    sync_calls = []
    monkeypatch.setattr(
        pod, "sync_and_verify",
        lambda *_a, **kwargs: sync_calls.append(kwargs["relative_paths"]) or [report],
    )
    attempts = 0

    def verify(receipts, *_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if len(receipts) != 2:
            raise publish.PublishError("stale listing")

    monkeypatch.setattr(pod, "_verify_success_inventory", verify)
    monkeypatch.setattr(pod, "_verify_extracted_paid_gate", lambda *_a, **_k: None)
    monkeypatch.setattr(pod, "_verify_synced_tree_exact", lambda *_a, **_k: None)
    monkeypatch.setattr(
        pod,
        "_verify_deferred_remote_payloads",
        lambda _publisher, receipts, local_paths, **_kwargs: {
            "revision": "c" * 40,
            "count": len(receipts) - len(local_paths),
        },
    )
    result = pod._sync_terminated_run(
        object(), run_id="artifact-1", attempt_id="attempt-1",
        local_root=tmp_path / "payloads", receipt_root=tmp_path / "custody",
        n_prompts=100, image=IMAGE, launch_nonce="n" * 48,
        stage_manifest_sha256="a" * 64, stage_source_head="b" * 40,
        worker_deadline_epoch=int(time.time()) + 3600, replay_source_root=tmp_path,
        compute_exit_code=0, termination_verified=True, attempts=2,
        retry_seconds=0, full_replay=False,
    )
    assert set(result) == {report, tensor}
    assert sync_calls == [["final/analysis.json"], ["final/analysis.json"]]
    assert attempts == 2


def test_failed_full_sync_then_same_lineage_application_sync_reuses_verified_root(
    tmp_path,
    monkeypatch,
):
    remote = publish.LocalPublisher(tmp_path / "remote")
    sources = tmp_path / "sources"
    sources.mkdir()
    report_source = sources / "analysis.json"
    report_source.write_bytes(b"report")
    tensor_source = sources / "exit3.pt"
    tensor_source.write_bytes(b"tensor")
    report = publish.publish_file(
        remote,
        report_source,
        run_id="artifact-1",
        relative_path="final/analysis.json",
        kind="report",
    )
    tensor = publish.publish_file(
        remote,
        tensor_source,
        run_id="artifact-1",
        relative_path="lens/n100/exit3.pt",
        kind="merged_lens",
    )
    local_root = tmp_path / "retrieved" / "artifact-1"
    receipt_root = tmp_path / "retrieved" / "artifact-1.receipts"
    bindings = {
        "run_id": "artifact-1",
        "local_root": local_root,
        "receipt_root": receipt_root,
        "n_prompts": 100,
        "image": IMAGE,
        "launch_nonce": "n" * 48,
        "stage_manifest_sha256": "a" * 64,
        "stage_source_head": "b" * 40,
        "worker_deadline_epoch": int(time.time()) + 3600,
        "replay_source_root": tmp_path,
        "termination_verified": True,
        "attempts": 2,
        "retry_seconds": 0,
    }
    failed = pod._sync_terminated_run(
        remote,
        attempt_id="attempt-a",
        compute_exit_code=17,
        full_replay=False,
        **bindings,
    )
    assert set(failed) == {report, tensor}
    assert (local_root / "lens/n100/exit3.pt").read_bytes() == b"tensor"

    monkeypatch.setattr(pod, "_verify_success_inventory", lambda *_a, **_k: None)
    monkeypatch.setattr(pod, "_verify_extracted_paid_gate", lambda *_a, **_k: None)
    succeeded = pod._sync_terminated_run(
        remote,
        attempt_id="attempt-b",
        compute_exit_code=0,
        full_replay=False,
        **bindings,
    )
    assert set(succeeded) == {report, tensor}
    assert (local_root / "final/analysis.json").read_bytes() == b"report"
    assert (local_root / "lens/n100/exit3.pt").read_bytes() == b"tensor"


def test_application_sync_rejects_mismatched_previously_custodied_tensor(
    tmp_path,
):
    root = tmp_path / "retrieved"
    report_path = root / "final/analysis.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(b"report")
    tensor_path = root / "lens/n100/exit3.pt"
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(b"corrupt")
    report = publish.Receipt(
        1, "artifact-1", "final/analysis.json",
        "artifact-1/artifacts/final/analysis.json", "report",
        hashlib.sha256(b"report").hexdigest(), len(b"report"),
    )
    tensor = publish.Receipt(
        1, "artifact-1", "lens/n100/exit3.pt",
        "artifact-1/artifacts/lens/n100/exit3.pt", "merged_lens",
        hashlib.sha256(b"tensor").hexdigest(), len(b"tensor"),
    )
    with pytest.raises(publish.PublishError, match="differs from its receipt"):
        pod._verify_synced_tree_exact(
            root,
            [report],
            run_id="artifact-1",
            allowed_receipts=[report, tensor],
        )


def test_unverified_termination_can_only_seal_partial_snapshot(tmp_path, monkeypatch):
    receipt = publish.Receipt(
        1, "artifact-1", "logs/partial.log", "artifact-1/artifacts/logs/partial.log",
        "run_log", "a" * 64, 1,
    )
    monkeypatch.setattr(pod, "sync_and_verify", lambda *_a, **_k: [receipt])
    monkeypatch.setattr(pod, "_verify_synced_tree_exact", lambda *_a, **_k: None)
    monkeypatch.setattr(
        pod, "_verify_success_inventory",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not promote")),
    )
    receipt_root = tmp_path / "custody"
    pod._sync_terminated_run(
        object(),
        run_id="artifact-1",
        attempt_id="attempt-1",
        local_root=tmp_path / "payloads",
        receipt_root=receipt_root,
        n_prompts=100,
        image=IMAGE,
        launch_nonce="n" * 48,
        stage_manifest_sha256="a" * 64,
        stage_source_head="b" * 40,
        worker_deadline_epoch=int(time.time()) + 3600,
        replay_source_root=tmp_path,
        compute_exit_code=0,
        termination_verified=False,
        attempts=2,
        retry_seconds=0,
    )
    snapshot_path = next((receipt_root / "snapshots").glob("*.json"))
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["complete"] is False
    assert snapshot["compute_exit_code"] == 0
    assert snapshot["termination_verified"] is False
    assert snapshot["extraction_profile"] == "partial"


def test_extracted_paid_gate_rechecks_exact_run_image_and_claim_status(tmp_path, monkeypatch):
    root = tmp_path / "retrieved"
    claims = {
        "instrumentation": {"status": "SUPPORTED_CURRENT_VALIDATION"},
        "b300_validation": {"status": "SUPPORTED_CURRENT_B300_VALIDATION"},
    }
    payloads = {
        "final/verification.json": {
            "schema_version": 1,
            "status": "PASS",
            "accepted": True,
            "run_id": "artifact-1",
            "attempt_id": "attempt-1",
            "launch_nonce": "n" * 48,
            "stage_manifest_sha256": "a" * 64,
            "stage_source_head": "b" * 40,
            "worker_deadline_epoch": int(time.time()) + 3600,
            "image_digest": IMAGE,
            "controller_approved_image_digest": IMAGE,
            "validation_status": "SUPPORTED_CURRENT_B300_VALIDATION",
            "b300_validation_status": "SUPPORTED_CURRENT_B300_VALIDATION",
            "canonical_report_status": "CURRENT_GENERATION_EVIDENCE_VERIFIED",
            "evaluation_lineage_status": "CURRENT_GENERATION_EVIDENCE_VERIFIED",
        },
        "validation/milestones.json": {"provenance": {"image_digest": IMAGE}},
        "final/analysis.json": {
            "overall_status": "CURRENT_GENERATION_EVIDENCE_VERIFIED",
            "claims": claims,
        },
        "final/claim_status.json": claims,
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    bindings = {
        "attempt_id": "attempt-1",
        "launch_nonce": "n" * 48,
        "stage_manifest_sha256": "a" * 64,
        "stage_source_head": "b" * 40,
        "worker_deadline_epoch": int(time.time()) + 3600,
    }
    pod._verify_extracted_paid_gate(root, run_id="artifact-1", image=IMAGE, **bindings)
    payloads["final/verification.json"]["run_id"] = "stale-lineage"
    (root / "final/verification.json").write_text(
        json.dumps(payloads["final/verification.json"])
    )
    with pytest.raises(publish.PublishError, match="not exact PASS"):
        pod._verify_extracted_paid_gate(root, run_id="artifact-1", image=IMAGE, **bindings)


def test_synced_tree_must_equal_receipts_without_stale_or_linked_files(tmp_path):
    root = tmp_path / "synced"
    payload = root / "final" / "verification.json"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"{}")
    receipt = publish.Receipt(
        1, "r1", "final/verification.json",
        "r1/artifacts/final/verification.json", "verification",
        hashlib.sha256(b"{}").hexdigest(), 2,
    )
    pod._verify_synced_tree_exact(root, [receipt], run_id="r1")
    extra = root / "stale.json"
    extra.write_text("stale")
    with pytest.raises(publish.PublishError, match="receipt-exact"):
        pod._verify_synced_tree_exact(root, [receipt], run_id="r1")
    extra.unlink()
    linked = root / "linked.json"
    linked.symlink_to(payload)
    with pytest.raises(publish.PublishError, match="symlink"):
        pod._verify_synced_tree_exact(root, [receipt], run_id="r1")


def test_initial_state_creation_is_exclusive_and_does_not_replace(tmp_path):
    state_file = tmp_path / "state.json"
    pod._create_initial_json(state_file, {"run_id": "r1"})
    before = state_file.read_bytes()
    with pytest.raises(pod.SafetyError, match="state already exists"):
        pod._create_initial_json(state_file, {"run_id": "r2"})
    assert state_file.read_bytes() == before


def test_verified_terminal_state_cannot_be_resurrected_by_stale_monitor_save(tmp_path):
    state_file = tmp_path / "state.json"
    identity = {
        "run_id": "r1",
        "artifact_run_id": "r1",
        "pod_name": "ouro-jlens-r1",
        "pod_id": "p1",
        "machine_id": "m1",
        "launch_nonce": "n" * 48,
    }
    stale = {**identity, "status": "active", "termination_verified": False}
    state_file.write_text(json.dumps(stale))
    supervisor = pod.LeaseSupervisor(
        FakeClient(),
        state_file,
        pod.SafetyConfig(),
    )
    terminal = {
        **identity,
        "status": "terminated",
        "termination_verified": True,
        "terminated_at": 123.0,
    }
    supervisor._save(terminal)
    supervisor._save({**stale, "last_heartbeat_at": 124.0})
    assert pod._read_json(state_file) == terminal


def test_stale_nonterminal_heartbeat_cannot_erase_creator_lease_fields(tmp_path):
    state_file = tmp_path / "state.json"
    base = {
        "schema_version": 2,
        "run_id": "r1",
        "attempt_id": "r1",
        "lease_attempt_id": "r1",
        "artifact_run_id": "r1",
        "artifact_lineage_id": "r1",
        "pod_name": "ouro-jlens-r1",
        "launch_nonce": "n" * 48,
        "monitor_nonce": "monitor-nonce",
        "status": "pending",
        "deployment_started": False,
        "termination_verified": False,
    }
    state_file.write_text(json.dumps(base))
    supervisor = pod.LeaseSupervisor(FakeClient(), state_file, pod.SafetyConfig())
    creator = {
        **base,
        "status": "active",
        "deployment_started": True,
        "pod_id": "p1",
        "machine_id": "m1",
        "monitor_unit": "ouro-jlens-monitor-r1.service",
        "ssh_ip": "127.0.0.1",
        "ssh_port": 2222,
    }
    supervisor._save(creator)
    supervisor._save({
        **base,
        "machine_id": None,
        "monitor_status": "READY",
        "last_heartbeat_at": 123.0,
    })
    saved = pod._read_json(state_file)
    assert saved["status"] == "active"
    assert saved["deployment_started"] is True
    assert saved["pod_id"] == "p1"
    assert saved["machine_id"] == "m1"
    assert saved["monitor_unit"] == "ouro-jlens-monitor-r1.service"
    assert saved["ssh_ip"] == "127.0.0.1"
    assert saved["ssh_port"] == 2222
    assert saved["last_heartbeat_at"] == 123.0


@pytest.mark.parametrize(
    ("deploy_machine", "ready_machine", "expected", "error"),
    [
        (None, "m-ready", "m-ready", None),
        (None, None, None, "missing or invalid"),
        ("m-deploy", "m-other", None, "different machine"),
    ],
)
def test_machine_identity_is_bound_at_readiness_and_never_changes(
    tmp_path, monkeypatch, deploy_machine, ready_machine, expected, error
):
    args = _args(tmp_path)
    key = "rpa_" + "M" * 24
    key_file = tmp_path / "fidelio.txt"
    key_file.write_text(key + "\n")
    key_file.chmod(0o600)
    fingerprint = hashlib.sha256(key.encode()).hexdigest()
    monkeypatch.setattr(pod, "KEY_FILE", key_file)
    monkeypatch.setenv(pod.KEY_SELECTOR_ENV, fingerprint)
    args.runpod_key_sha256 = fingerprint
    ready_port = {
        "ip": "127.0.0.1", "publicPort": 2222,
        "privatePort": 22, "isIpPublic": True,
    }
    waiting = _pod("ouro-jlens-r1", "new-pod")
    waiting["machineId"] = deploy_machine
    ready = _pod("ouro-jlens-r1", "new-pod", ports=[ready_port])
    ready["machineId"] = ready_machine
    if deploy_machine is None and ready_machine == "m-ready":
        # This is the shape returned by RunPod in production: the concrete
        # machine and SSH endpoint are present, while gpuDisplayName uses the
        # short provider alias rather than the requested SKU string.
        ready["machine"] = {"gpuDisplayName": "B300"}

    class MachineClient(FakeClient):
        def deploy(self, **kwargs):
            created = super().deploy(**kwargs)
            created["machineId"] = deploy_machine
            if deploy_machine is None:
                created["machine"] = None
            self.current_pods = [created]
            return created

    client = MachineClient(pod_sequence=[[], [], [], [waiting], [ready]])

    def launch(state_file):
        pod.LeaseSupervisor(
            client, state_file, pod._lease_config(args),
            sleep=lambda _seconds: None,
        )._authenticated_ready(pod._read_json(state_file))
        return "ouro-jlens-monitor-r1.service"

    monkeypatch.setattr(pod, "verify_systemd_monitor_active", lambda _unit: None)
    if error is not None:
        with pytest.raises((pod.SafetyError, pod.IdentityMismatch), match=error):
            pod._create_lease(args, client=client, monitor_launcher=launch)
        assert client.terminate_calls
        assert set(client.terminate_calls) == {"new-pod"}
        return
    state, supervisor = pod._create_lease(
        args, client=client, monitor_launcher=launch
    )
    assert state["machine_id"] == expected
    assert state["provider_gpu_display_name"] == pod.DEFAULT_GPU
    assert pod._read_json(
        pod.state_path("r1", tmp_path / "states")
    )["machine_id"] == expected
    supervisor.terminate_verified()


@pytest.mark.parametrize("name", ["B300", "NVIDIA B300", "NVIDIA B300 SXM6 AC"])
def test_provider_b300_display_aliases_are_canonicalized(name):
    assert pod._optional_assigned_b300(
        {"machine": {"gpuDisplayName": name}}
    ) == pod.DEFAULT_GPU


def test_missing_provider_gpu_display_is_allowed_but_non_b300_is_rejected():
    assert pod._optional_assigned_b300({"machine": None}) is None
    with pytest.raises(pod.IdentityMismatch, match="identify.*B300"):
        pod._optional_assigned_b300(
            {"machine": {"gpuDisplayName": "NVIDIA H100 SXM"}}
        )


def test_env_key_derives_selector_and_rejects_explicit_mismatch(monkeypatch):
    key = "rpa_" + "E" * 24
    fingerprint = hashlib.sha256(key.encode()).hexdigest()
    args = pod._parser().parse_args(["create", "--dry-run", "--run-id", "env-selector"])
    monkeypatch.setenv("RUNPOD_API_KEY", key)
    monkeypatch.setenv(pod.KEY_SELECTOR_ENV, fingerprint)
    assert pod._activate_key_selector(args) == fingerprint
    assert args.runpod_key_sha256 == fingerprint
    assert pod.api_key(Path("/does/not/exist")) == key

    args.runpod_key_sha256 = "0" * 64
    with pytest.raises(pod.SafetyError, match="conflicts|does not match"):
        pod._activate_key_selector(args)


def test_selector_accepts_duplicate_copies_of_same_durable_credential(tmp_path):
    selected = "rpa_" + "D" * 24
    other = "rpa_" + "O" * 24
    key_file = tmp_path / "fidelio.txt"
    key_file.write_text(f"{selected}\n{other}\n{selected}\n")
    key_file.chmod(0o600)
    fingerprint = hashlib.sha256(selected.encode()).hexdigest()

    assert pod._select_file_key(key_file, fingerprint) == selected
    with pytest.raises(pod.CredentialSelectionError):
        pod._select_file_key(key_file)


def test_durable_runpod_credential_requires_bounded_regular_0600_file(tmp_path):
    key = "rpa_" + "R" * 24
    credential = tmp_path / "fidelio.txt"
    credential.write_text(key)
    credential.chmod(0o644)
    with pytest.raises(pod.SafetyError, match="0600"):
        pod._select_file_key(credential)

    credential.chmod(0o600)
    link = tmp_path / "linked-fidelio.txt"
    link.symlink_to(credential)
    with pytest.raises(pod.SafetyError, match="unsafe path"):
        pod._select_file_key(link)

    oversized = tmp_path / "oversized-fidelio.txt"
    oversized.write_text(key + "x" * pod.MAX_RUNPOD_CREDENTIAL_BYTES)
    oversized.chmod(0o600)
    with pytest.raises(pod.SafetyError, match="invalid size"):
        pod._select_file_key(oversized)


def test_paid_controller_rejects_hf_token_mode_other_than_exact_0600(tmp_path):
    args = _args(tmp_path)
    Path(args.hf_token).chmod(0o400)
    with pytest.raises(pod.SafetyError, match="0600"):
        pod._validate_hf_token_file(args.hf_token)


def test_env_only_credential_absent_from_file_blocks_predeploy(tmp_path, monkeypatch):
    args = _args(tmp_path)
    key = "rpa_" + "S" * 24
    monkeypatch.setenv("RUNPOD_API_KEY", key)
    monkeypatch.setenv(pod.KEY_SELECTOR_ENV, hashlib.sha256(key.encode()).hexdigest())
    monkeypatch.setattr(pod, "KEY_FILE", tmp_path / "missing-fidelio.txt")
    client = FakeClient()
    with pytest.raises(pod.SafetyError, match="cannot read RunPod credential file"):
        pod._create_lease(args, client=client, monitor_launcher=lambda _path: "unit")
    assert client.calls == []
    assert not list((tmp_path / "states").rglob("state.json"))


def test_main_dry_run_does_not_validate_or_read_credentials(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "SHELL_ONLY_SENTINEL")
    monkeypatch.setenv(pod.KEY_SELECTOR_ENV, "not-a-selector")
    assert pod.main(["create", "--dry-run", "--run-id", "offline"]) == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_paid_image_must_match_publisher_owned_canonical_identity():
    canonical = pod.RUNTIME_IMAGE
    assert pod._validate_paid_image(canonical) == canonical
    with pytest.raises(pod.SafetyError, match="publisher-approved"):
        pod._validate_paid_image("runpod/pytorch@sha256:" + "c" * 64)


def test_paid_image_fails_closed_when_publisher_identity_is_unavailable(monkeypatch):
    monkeypatch.setattr(pod, "_canonical_b300_image", lambda: None)
    with pytest.raises(pod.SafetyError, match="unavailable"):
        pod._validate_paid_image(pod.RUNTIME_IMAGE)


def test_monitor_handshake_writes_exact_ready_nonce_fingerprint_and_heartbeat(tmp_path):
    args = _args(tmp_path)
    fingerprint = "a" * 64
    state = pod._new_state("r1", args, credential_fingerprint=fingerprint)
    state_file = pod.state_path("r1", tmp_path / "states")
    state["protected_pod_ids"] = []
    pod._create_initial_json(state_file, state)
    client = FakeClient()
    supervisor = pod.LeaseSupervisor(client, state_file, sleep=lambda _seconds: None,
                                     clock=lambda: 100.0)
    acknowledged = supervisor._authenticated_ready(pod._read_json(state_file))
    saved = json.loads(state_file.read_text())
    assert client.calls == ["balance", "pods"]
    assert acknowledged["monitor_status"] == "READY"
    assert saved["monitor_status"] == "READY"
    assert saved["monitor_nonce"] == state["monitor_nonce"]
    assert saved["monitor_ready_nonce"] == state["monitor_nonce"]
    assert saved["runpod_key_sha256"] == fingerprint
    assert saved["monitor_ready_fingerprint"] == fingerprint
    assert saved["monitor_heartbeat_at"] == 100.0
    assert saved["last_heartbeat_at"] == 100.0


@pytest.mark.parametrize("mutation", [
    {"monitor_status": "arming"},
    {"monitor_status": "error"},
    {"monitor_status": "terminating"},
    {"termination_error": "cleanup failed"},
])
def test_creator_rejects_early_error_or_terminating_monitor_ready_state(mutation):
    state = {
        "monitor_status": "READY",
        "monitor_nonce": "nonce",
        "monitor_ready_nonce": "nonce",
        "runpod_key_sha256": "a" * 64,
        "monitor_ready_fingerprint": "a" * 64,
        "monitor_heartbeat_at": 100.0,
        "last_heartbeat_at": 100.0,
        "created_at": 99.0,
        "status": "pending",
        "deployment_started": False,
    }
    state.update(mutation)
    with pytest.raises(pod.SafetyError):
        pod._require_monitor_ready(state, nonce="nonce", fingerprint="a" * 64, now=100.0)


@pytest.mark.parametrize("missing", ["monitor_ready_nonce", "monitor_ready_fingerprint"])
def test_creator_requires_explicit_monitor_ready_identity(missing):
    state = {
        "monitor_status": "READY",
        "monitor_nonce": "nonce",
        "monitor_ready_nonce": "nonce",
        "runpod_key_sha256": "a" * 64,
        "monitor_ready_fingerprint": "a" * 64,
        "monitor_heartbeat_at": 100.0,
        "last_heartbeat_at": 100.0,
        "created_at": 99.0,
        "status": "pending",
        "deployment_started": False,
    }
    del state[missing]
    with pytest.raises(pod.SafetyError, match="missing or inconsistent"):
        pod._require_monitor_ready(state, nonce="nonce", fingerprint="a" * 64, now=100.0)


def test_creator_rejects_inactive_monitor_unit():
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 3, stdout="inactive\n", stderr="")

    with pytest.raises(pod.SafetyError, match="not active"):
        pod.verify_systemd_monitor_active("ouro-jlens-monitor-r1.service", runner=runner)


def test_monitor_bundle_command_is_content_addressed_and_tamper_checked(tmp_path):
    state_file = tmp_path / "r1" / "state.json"
    pod._create_initial_json(state_file, {
        "run_id": "r1",
        "pod_name": "ouro-jlens-r1",
        "status": "pending",
    })
    state = pod._read_json(state_file)
    state["monitor_bundle"] = pod._install_monitor_bundle(state_file)
    pod._atomic_json(state_file, state)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[0] == "loginctl":
            return subprocess.CompletedProcess(command, 0, stdout="yes\n")
        if command[:3] == ["systemctl", "--user", "is-enabled"]:
            return subprocess.CompletedProcess(command, 0, stdout="enabled\n")
        if command[0] == "systemctl" and "show" in command:
            return subprocess.CompletedProcess(command, 0, stdout="active\n")
        return subprocess.CompletedProcess(command, 0, stdout="")

    unit_dir = tmp_path / "systemd-user"
    unit = pod.start_systemd_monitor(
        state_file, script=Path("/mutable/pod.py"), runner=runner, unit_dir=unit_dir
    )
    unit_text = (unit_dir / unit).read_text()
    bundle_root = str(state_file.parent / state["monitor_bundle"]["root"])
    assert unit == "ouro-jlens-monitor-r1.service"
    assert f"Environment=PYTHONPATH={bundle_root}" in unit_text
    assert f"--state {state_file.absolute()}" in unit_text
    assert str(Path(bundle_root) / "ouro_jlens" / "pod.py") in unit_text
    assert str(Path("/mutable/pod.py")) not in unit_text
    assert "WantedBy=default.target" in unit_text

    imported = subprocess.run(
        [sys.executable, "-c", "import ouro_jlens.pod; print('monitor-import-ok')"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": bundle_root},
        text=True,
        capture_output=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == "monitor-import-ok"

    bundled_pod = Path(bundle_root) / "ouro_jlens" / "pod.py"
    bundled_pod.write_text("tampered\n")
    with pytest.raises(pod.SafetyError, match="hash verification"):
        pod.start_systemd_monitor(state_file, runner=lambda *_a, **_k: pytest.fail("runner called"))


def test_provider_deploy_is_exact_secure_b300_with_absolute_ttl():
    calls = []

    def gql(query, variables=None):
        calls.append((query, variables))
        return {"podFindAndDeployOnDemand": _pod("ouro-jlens-r1", "new-pod")}

    client = pod.RunPodClient(gql)
    guard = object()
    client._arm_deployment(guard)
    deadline = _future_provider_deadline()
    created = client.deploy(
        name="ouro-jlens-r1",
        gpu=pod.DEFAULT_GPU,
        disk=100,
        public_key="ssh-ed25519 AAAA test",
        hf_token="hf_test_token",
        image=IMAGE,
        terminate_after=deadline,
        _guard=guard,
    )
    assert created["id"] == "new-pod"
    assert len(calls) == 1
    request = calls[0][1]["input"]
    assert request["cloudType"] == "SECURE"
    assert request["gpuTypeId"] == "NVIDIA B300 SXM6 AC"
    assert request["gpuCount"] == 1
    assert request["terminateAfter"] == deadline
    assert request["imageName"] == IMAGE
    environment = {entry["key"]: entry["value"] for entry in request["env"]}
    assert environment["JLENS_HF_TOKEN_BOOTSTRAP"] == "hf_test_token"
    assert "HF_TOKEN" not in environment


def test_secure_b300_offer_is_exact_and_full_runtime_must_fit_budget():
    calls = []

    def gql(query, variables=None):
        calls.append((query, variables))
        return {"gpuTypes": [{
            "id": pod.DEFAULT_GPU,
            "displayName": "B300",
            "memoryInGb": 288,
            "secureCloud": True,
            "communityCloud": True,
            "lowestPrice": {
                "stockStatus": "Low",
                "uninterruptablePrice": 7.89,
                "availableGpuCounts": None,
            },
        }]}

    offer = pod.RunPodClient(gql).gpu_offer(pod.DEFAULT_GPU)
    assert offer["lowestPrice"]["uninterruptablePrice"] == 7.89
    assert calls[0][1] == {"id": pod.DEFAULT_GPU}
    config = pod.SafetyConfig(max_runtime=12_600, max_spend=30)
    assert pod._validate_offer_budget(offer, config) == pytest.approx(27.615)
    with pytest.raises(pod.SafetyError, match="full-runtime cost"):
        pod._validate_offer_budget(offer, pod.SafetyConfig(max_runtime=12_600, max_spend=27))


def test_worker_deadline_reserves_publication_window_before_provider_kill():
    deadline = pod._provider_termination_deadline(1_700_000_000, 12_600)
    worker = pod._worker_deadline_epoch(deadline, now=1_700_000_000)
    provider_epoch = int(pod.datetime.strptime(
        deadline, "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=pod.timezone.utc).timestamp())
    assert provider_epoch - worker == pod.WORKER_DEADLINE_MARGIN_SECONDS
    with pytest.raises(pod.SafetyError, match="publication window"):
        pod._worker_deadline_epoch(
            pod._provider_termination_deadline(1_700_000_000, 600),
            now=1_700_000_000,
        )


def test_paid_defaults_fit_declared_end_to_end_worker_floor():
    config = pod.SafetyConfig()
    assert config.min_balance == pod.HARD_MIN_BALANCE == 5.0
    assert config.max_spend == pod.HARD_MAX_SPEND == 30.0
    assert config.max_runtime == pod.HARD_MAX_RUNTIME == 12_600.0
    pod._validate_paid_workload_window(config)
    too_short = pod.SafetyConfig(max_runtime=11_699, max_spend=30)
    with pytest.raises(pod.SafetyError, match="end-to-end worker floor"):
        pod._validate_paid_workload_window(too_short)


def test_delayed_ssh_readiness_cannot_consume_worker_or_bootstrap_window(
    tmp_path, monkeypatch
):
    args = _args(tmp_path)
    key = "rpa_" + "W" * 24
    key_file = tmp_path / "fidelio.txt"
    key_file.write_text(key + "\n")
    key_file.chmod(0o600)
    fingerprint = hashlib.sha256(key.encode()).hexdigest()
    monkeypatch.setattr(pod, "KEY_FILE", key_file)
    monkeypatch.setenv(pod.KEY_SELECTOR_ENV, fingerprint)
    args.runpod_key_sha256 = fingerprint
    current = {"value": 1_700_000_000.0}
    monkeypatch.setattr(pod.time, "time", lambda: current["value"])
    client = FakeClient()

    def launch(state_file):
        pod.LeaseSupervisor(
            client,
            state_file,
            pod._lease_config(args),
            sleep=lambda _seconds: None,
            clock=lambda: current["value"],
        )._authenticated_ready(pod._read_json(state_file))
        return "ouro-jlens-monitor-r1.service"

    ready_port = {
        "ip": "127.0.0.1", "publicPort": 2222,
        "privatePort": 22, "isIpPublic": True,
    }

    def delayed_ready(_client, _pod_id, **_kwargs):
        current["value"] += 601
        return {
            "pod": _pod(
                "ouro-jlens-r1", "new-pod", ports=[ready_port]
            ),
            "port": ready_port,
        }

    monkeypatch.setattr(pod, "wait_for_ssh", delayed_ready)
    monkeypatch.setattr(pod, "verify_systemd_monitor_active", lambda _unit: None)
    with pytest.raises(pod.SafetyError, match="required end-to-end worker window"):
        pod._create_lease(args, client=client, monitor_launcher=launch)
    assert client.terminate_calls
    assert set(client.terminate_calls) == {"new-pod"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_balance": 4.99}, "balance floor"),
        ({"max_spend": 30.01}, "maximum spend"),
        ({"max_runtime": 12_601}, "maximum runtime"),
    ],
)
def test_paid_account_hard_caps_cannot_be_relaxed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pod.SafetyConfig(**kwargs)


def test_paid_account_hard_cap_boundaries_are_accepted():
    config = pod.SafetyConfig(
        min_balance=pod.HARD_MIN_BALANCE,
        max_spend=pod.HARD_MAX_SPEND,
        max_runtime=pod.HARD_MAX_RUNTIME,
    )
    assert (config.min_balance, config.max_spend, config.max_runtime) == (5.0, 30.0, 12_600.0)


def test_results_preflight_proves_write_readback_and_eventual_listing(tmp_path, monkeypatch):
    class Publisher:
        def __init__(self):
            self.files = {}
            self.list_calls = 0

        def ensure_repository(self):
            return None

        def put_file(self, source, remote_path):
            self.files[remote_path] = Path(source).read_bytes()

        def download_file(self, remote_path, destination, *, expected_size=None):
            data = self.files[remote_path]
            assert expected_size == len(data)
            Path(destination).write_bytes(data)

        def list_files(self, prefix=""):
            self.list_calls += 1
            if self.list_calls < 3:
                return []
            return sorted(path for path in self.files if path.startswith(prefix))

    sleeps = []
    monkeypatch.setattr(pod.time, "sleep", sleeps.append)
    publisher = Publisher()
    result = pod._preflight_results_repository(
        publisher,
        repository="org/results",
        attempt_id="attempt-1",
        artifact_run_id="artifact-1",
        launch_nonce="n" * 48,
        stage_manifest_sha256="a" * 64,
        stage_source_head="b" * 40,
        hf_token_sha256="c" * 64,
    )
    assert result["remote_path"] in publisher.files
    assert result["sha256"] == hashlib.sha256(
        publisher.files[result["remote_path"]]
    ).hexdigest()
    assert sleeps == [0.5, 1.0]


def test_results_preflight_fails_closed_when_listing_never_converges(tmp_path, monkeypatch):
    class Publisher:
        def ensure_repository(self):
            return None

        def put_file(self, _source, _remote_path):
            return None

        def download_file(self, _remote_path, destination, *, expected_size=None):
            # Deliberately wrong bytes exercise readback before listing.
            Path(destination).write_bytes(b"wrong")

        def list_files(self, prefix=""):
            return []

    monkeypatch.setattr(pod.time, "sleep", lambda _seconds: None)
    with pytest.raises(pod.SafetyError, match="write/read/list authority"):
        pod._preflight_results_repository(
            Publisher(), repository="org/results", attempt_id="attempt-1",
            artifact_run_id="artifact-1", launch_nonce="n" * 48,
            stage_manifest_sha256="a" * 64, stage_source_head="b" * 40,
            hf_token_sha256="c" * 64, attempts=2, retry_seconds=0,
        )


@pytest.mark.parametrize("mutation", [
    {"id": "NVIDIA B300"},
    {"displayName": "B300 SXM"},
    {"memoryInGb": 192},
    {"secureCloud": False},
    {"lowestPrice": {"stockStatus": "None", "uninterruptablePrice": 7.89}},
])
def test_secure_b300_offer_rejects_identity_or_availability_drift(mutation):
    offer = {
        "id": pod.DEFAULT_GPU,
        "displayName": "B300",
        "memoryInGb": 288,
        "secureCloud": True,
        "communityCloud": True,
        "lowestPrice": {"stockStatus": "Low", "uninterruptablePrice": 7.89},
    }
    offer.update(mutation)
    client = pod.RunPodClient(lambda *_a, **_k: {"gpuTypes": [offer]})
    with pytest.raises(pod.APIUncertain, match="malformed or unavailable"):
        client.gpu_offer(pod.DEFAULT_GPU)


@pytest.mark.parametrize("gpu", ["NVIDIA B200", "NVIDIA B300", "B300"])
def test_provider_deploy_rejects_every_nonexact_gpu_before_mutation(gpu):
    calls = []
    client = pod.RunPodClient(lambda *args, **kwargs: calls.append((args, kwargs)))
    guard = object()
    client._arm_deployment(guard)
    with pytest.raises(pod.SafetyError, match="exact GPU type"):
        client.deploy(
            name="ouro-jlens-r1", gpu=gpu, disk=100,
            public_key="ssh-ed25519 AAAA test", hf_token="hf_test_token",
            image=IMAGE, terminate_after=_future_provider_deadline(), _guard=guard,
        )
    assert calls == []


def test_termination_requires_three_consecutive_absences_after_reappearance(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({
        "schema_version": 2,
        "run_id": "r1",
        "attempt_id": "r1",
        "pod_name": "ouro-jlens-r1",
        "pod_id": "p1",
        "machine_id": "m1",
        "launch_nonce": "n" * 48,
        "status": "active",
    }))

    class EventuallyConsistent(FakeClient):
        def terminate(self, pod_id):
            self.calls.append(("terminate", pod_id))
            self.terminate_calls.append(pod_id)

    client = EventuallyConsistent(pod_sequence=[
        [_pod()],  # pre-mutation identity check
        [],        # one transient absence is not enough
        [_pod()],  # reappearance resets the absence counter
        [], [], [],
    ])
    supervisor = pod.LeaseSupervisor(
        client, state_file,
        pod.SafetyConfig(
            min_balance=5, termination_attempts=5, termination_poll_seconds=0,
        ),
        sleep=lambda _seconds: None,
    )
    supervisor.terminate_verified()
    assert client.terminate_calls == ["p1"] * 5
    saved = json.loads(state_file.read_text())
    assert saved["termination_verified"] is True


def test_persistent_monitor_stop_disables_verifies_and_removes_unit(tmp_path):
    unit = "ouro-jlens-monitor-r1.service"
    unit_dir = tmp_path / "systemd-user"
    unit_dir.mkdir()
    unit_path = unit_dir / unit
    unit_path.write_text("[Unit]\n")
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[:3] == ["systemctl", "--user", "is-active"]:
            return subprocess.CompletedProcess(command, 3, stdout="inactive\n")
        return subprocess.CompletedProcess(command, 0, stdout="")

    pod.stop_systemd_monitor(unit, runner=runner, unit_dir=unit_dir)
    assert not unit_path.exists()
    assert ["systemctl", "--user", "disable", "--now", unit] in calls
    assert ["systemctl", "--user", "daemon-reload"] in calls


def test_launch_nonce_is_fresh_even_when_visible_run_id_is_reused(tmp_path):
    args = _args(tmp_path)
    first = pod._new_state("r1", args)
    second = pod._new_state("r1", args)
    assert first["launch_nonce"] != second["launch_nonce"]
    assert pod.LAUNCH_NONCE_RE.fullmatch(first["launch_nonce"])
    assert pod.LAUNCH_NONCE_RE.fullmatch(second["launch_nonce"])


def test_postlease_state_validation_still_runs_termination_finally(tmp_path, monkeypatch):
    args = pod._parser().parse_args([
        "run", "--run-id", "r1", "--confirm-launch", "r1",
        "--results", "org/results", "--staging", "org/staging",
        "--stage-path", "stages/" + "a" * 64 + "/stage.tar.gz",
        "--state-root", str(tmp_path), "--sync-root", str(tmp_path / "sync"),
        "--image", IMAGE, "--n-prompts", "1", "--dim-batch", "1",
    ])

    class Supervisor:
        def __init__(self):
            self.terminated = 0

        def terminate_verified(self):
            self.terminated += 1

    supervisor = Supervisor()
    bootstrap = SimpleNamespace(
        package=Path("src/ouro_jlens"), root=tmp_path, manifest_sha256="a" * 64,
        source_head="b" * 40, committed_payload_sha256={},
    )
    monkeypatch.setattr(pod, "_prepare_stage_bootstrap", lambda *_a, **_k: bootstrap)
    monkeypatch.setattr(pod, "_verify_controller_source_matches_stage", lambda _value: None)
    monkeypatch.setattr(
        pod, "_preflight_results_repository",
        lambda *_a, **_k: {"schema_version": 1, "remote_path": "canary", "size": 1, "sha256": "c" * 64},
    )
    # Missing provider_terminate_after is deliberately detected only after
    # the lease exists.
    monkeypatch.setattr(pod, "_create_lease", lambda _args: ({
        "run_id": "r1", "artifact_run_id": "r1", "monitor_unit": None,
        "launch_nonce": args.launch_nonce,
        "stage_manifest_sha256": "a" * 64, "stage_source_head": "b" * 40,
    }, supervisor))
    monkeypatch.setattr(pod, "HfPublisher", lambda _repo, **_kwargs: object())
    monkeypatch.setattr(pod, "_sync_terminated_run", lambda *_a, **_k: [])

    with pytest.raises(pod.SafetyError, match="provider-enforced termination deadline"):
        pod.run_job(args)
    assert supervisor.terminated == 1
