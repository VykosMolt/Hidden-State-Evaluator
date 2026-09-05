"""Offline contract tests for the JLens rental and evidence protocol."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouro_jlens import pod, publish as publish_module
from ouro_jlens.publish import (
    ImmutableConflict,
    HfPublisher,
    LocalPublisher,
    PublishError,
    StageVerificationError,
    build_stage_tar,
    discover_receipt_paths,
    extract_stage_tar,
    publish_file,
    publish_index,
    publish_tree,
    read_index,
    restore_file,
    sync_and_verify,
    verify_runtime_compatibility,
    verify_stage_root,
    verify_stage_tar,
)


def _pod(name="ouro-jlens-r1", pod_id="p1", machine="m1", cost=1.0, uptime=0):
    return {
        "id": pod_id,
        "name": name,
        "machineId": machine,
        "machine": {"gpuDisplayName": "NVIDIA B300 SXM6 AC"},
        "costPerHr": cost,
        "desiredStatus": "RUNNING",
        "runtime": {"uptimeInSeconds": uptime, "ports": []},
    }


class FakeClient:
    def __init__(self, *, balance=100.0, pods=None):
        self.current_balance = balance
        self.current_pods = list(pods or [])
        self.calls = []
        self.terminate_calls = []

    def balance(self):
        self.calls.append("balance")
        return self.current_balance

    def pods(self):
        self.calls.append("pods")
        return list(self.current_pods)

    def terminate(self, pod_id):
        self.calls.append(("terminate", pod_id))
        self.terminate_calls.append(pod_id)
        self.current_pods = [p for p in self.current_pods if p["id"] != pod_id]


def _state(path: Path, *, pod_id="p1", name="ouro-jlens-r1", bound_at=0, cost=1.0):
    path.write_text(json.dumps({
        "schema_version": 2,
        "run_id": "r1",
        "launch_nonce": "n" * 48,
        "pod_name": name,
        "pod_id": pod_id,
        "machine_id": "m1",
        "cost_per_hr": cost,
        "created_at": bound_at,
        "bound_at": bound_at,
        "status": "active",
    }))


def _committed_payload_map(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): publish_module.sha256_file(path)
        for path in publish_module._stage_payload_paths(root)
        if "artifacts" not in path.relative_to(root).parts
    }


def _write_stage_corpus(root: Path) -> None:
    # Stage verification is intentionally anchored to the reviewed corpus,
    # not to a fixture that can rewrite both bytes and provenance together.
    repo_root = Path(__file__).resolve().parents[3]
    for relative in (
        publish_module.FITTING_CORPUS_GENERATOR_RELATIVE,
        publish_module.FITTING_CORPUS_RELATIVE,
        publish_module.FITTING_CORPUS_PROVENANCE_RELATIVE,
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo_root / relative).read_bytes())


def _build_extraction_stage(tmp_path: Path, *, executable: bool = False) -> Path:
    """Build one small, trusted stage archive for extraction transactions."""

    root = tmp_path / "stage-source"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    if executable:
        script = root / "src/ouro_jlens/entrypoint.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    archive = tmp_path / "stage.tar.gz"
    build_stage_tar(root, archive)
    return archive


def test_negative_balance_prevents_create_before_monitor_or_mutation(tmp_path, monkeypatch):
    pubkey = tmp_path / "pub"
    pubkey.write_text("ssh-ed25519 AAAA test\n")
    token = tmp_path / "hf"
    token.write_text("hf_test_token")
    token.chmod(0o600)
    identity = tmp_path / "identity"
    identity.write_text("private-key\n")
    args = pod._parser().parse_args([
        "create", "--run-id", "r1", "--state-root", str(tmp_path), "--min-balance", "20",
        "--image", "runpod/pytorch@sha256:" + "a" * 64,
        "--pubkey", str(pubkey), "--hf-token", str(token),
        "--identity-file", str(identity),
    ])
    client = FakeClient(balance=-0.54)
    monitor_started = []
    deploy_called = []
    client.deploy = lambda **kwargs: deploy_called.append(kwargs)
    with pytest.raises(pod.SafetyError, match="below"):
        pod._create_lease(client=client, monitor_launcher=lambda p: monitor_started.append(p), args=args)
    assert not monitor_started and not deploy_called
    assert not list(tmp_path.rglob("state.json"))


def test_negative_balance_is_blocked_at_hard_configured_floor(tmp_path):
    pubkey = tmp_path / "pub"
    pubkey.write_text("ssh-ed25519 AAAA test\n")
    token = tmp_path / "hf"
    token.write_text("hf_test_token")
    token.chmod(0o600)
    identity = tmp_path / "identity"
    identity.write_text("private-key\n")
    args = pod._parser().parse_args([
        "create", "--run-id", "r1", "--state-root", str(tmp_path), "--min-balance", "5",
        "--image", "runpod/pytorch@sha256:" + "a" * 64,
        "--pubkey", str(pubkey), "--hf-token", str(token),
        "--identity-file", str(identity),
    ])
    with pytest.raises(pod.SafetyError, match="below"):
        pod._create_lease(client=FakeClient(balance=-0.01),
                          monitor_launcher=lambda _path: None, args=args)


def test_dry_run_does_not_read_credentials_or_call_external(monkeypatch, capsys):
    args = pod._parser().parse_args(["create", "--dry-run", "--run-id", "offline"])
    monkeypatch.setenv("RUNPOD_API_KEY", "SENTINEL_DO_NOT_PRINT")
    monkeypatch.setattr(pod, "api_key", lambda: (_ for _ in ()).throw(AssertionError("API key read")))
    pod.create(args)
    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert "HF_TOKEN" not in output and "SENTINEL_DO_NOT_PRINT" not in output


def test_credential_file_requires_one_valid_key_without_key_fragment_selection(tmp_path):
    key_file = tmp_path / "credentials"
    valid_key = "rpa_" + "A" * 24
    key_file.write_text(f"comment\n{valid_key}\n")
    key_file.chmod(0o600)
    assert pod.api_key(key_file) == valid_key
    key_file.write_text("rpa_" + "B" * 24 + " rpa_" + "C" * 24)
    with pytest.raises(pod.SafetyError, match="exactly one"):
        pod.api_key(key_file)


def test_credential_selector_identifies_one_key_without_persisting_secret(monkeypatch, tmp_path):
    key_file = tmp_path / "credentials"
    first = "rpa_" + "B" * 24
    selected = "rpa_" + "C" * 24
    key_file.write_text(f"{first}\n{selected}\n")
    key_file.chmod(0o600)
    fingerprint = hashlib.sha256(selected.encode()).hexdigest()
    monkeypatch.setenv(pod.KEY_SELECTOR_ENV, fingerprint)
    assert pod.api_key(key_file) == selected

    args = pod._parser().parse_args([
        "create", "--dry-run", "--run-id", "selector-test",
        "--runpod-key-sha256", fingerprint,
    ])
    pod._activate_key_selector(args)
    state = pod._new_state("selector-test", args)
    encoded = json.dumps(state)
    assert state["runpod_key_sha256"] == fingerprint
    assert first not in encoded and selected not in encoded


def test_credential_selector_deduplicates_exact_key_and_rejects_missing_match(monkeypatch, tmp_path):
    key_file = tmp_path / "credentials"
    duplicate = "rpa_" + "D" * 24
    key_file.write_text(f"{duplicate}\n{duplicate}\n")
    key_file.chmod(0o600)
    monkeypatch.setenv(pod.KEY_SELECTOR_ENV, hashlib.sha256(duplicate.encode()).hexdigest())
    assert pod.api_key(key_file) == duplicate

    monkeypatch.setenv(pod.KEY_SELECTOR_ENV, "0" * 64)
    with pytest.raises(pod.SafetyError, match="exactly one"):
        pod.api_key(key_file)


def test_jlens_transformers_compatibility_override_is_explicit(monkeypatch):
    monkeypatch.setattr(
        publish_module.importlib_metadata,
        "requires",
        lambda name: ["torch", "transformers>=5.5", "numpy"]
        if name == "jlens" else [],
    )
    monkeypatch.setattr(
        publish_module.importlib_metadata,
        "version",
        lambda name: "4.54.1" if name == "transformers" else "0.0",
    )
    result = verify_runtime_compatibility()
    assert result == {
        "jlens_requirement": "transformers>=5.5",
        "transformers_runtime": "4.54.1",
        "compatibility_override": True,
    }


def test_jlens_transformers_override_fails_when_runtime_drifts(monkeypatch):
    monkeypatch.setattr(publish_module.importlib_metadata, "requires", lambda _name: ["transformers>=5.5"])
    monkeypatch.setattr(publish_module.importlib_metadata, "version", lambda _name: "5.5.0")
    with pytest.raises(StageVerificationError, match="compatibility override"):
        verify_runtime_compatibility()


@pytest.mark.parametrize("mutation", ["version", "hash", "image"])
def test_runtime_lock_rejects_tampered_version_hash_or_image(tmp_path, mutation):
    lock_path = tmp_path / "runtime.lock.json"
    lock = json.loads(Path("src/ouro_jlens/runtime.lock.json").read_text())
    if mutation == "version":
        lock["packages"]["transformers"]["version"] = "0.0.0"
    elif mutation == "hash":
        lock["packages"]["transformers"]["files_sha256"] = "0" * 64
    else:
        lock["image"] = "runpod/pytorch:latest"
    lock_path.write_bytes(publish_module.canonical_json(lock))
    with pytest.raises(StageVerificationError, match="lock digest"):
        publish_module.verify_runtime_lock(
            lock_path,
            image_digest=publish_module.RUNTIME_IMAGE,
        )


def test_setup_script_executes_pip_through_selected_python_and_checks_override():
    script = Path("src/ouro_jlens/setup_b300.sh").read_text()
    assert '"$PY" -m pip install -q' in script
    assert '"$PY" -m pip install -q -e "$JLENS_DIR" --no-deps' in script
    assert '"$PY" -m ouro_jlens.publish runtime-verify' in script
    assert "transformers>=5.5" in script and "transformers==4.54.1" in script
    assert publish_module.RUNTIME_IMAGE in script
    assert "JLENS_IMAGE_DIGEST is required" in script
    assert "runtime.lock.json" in script


def test_stage_upload_uses_cas_publisher_and_dry_run_has_no_external_access():
    script = Path("src/ouro_jlens/stage_upload.sh").read_text()
    assert "stage-publish" in script
    assert '"$PY" -m ouro_jlens.publish stage-publish' in script
    assert "hf upload" not in script
    assert "hf download" not in script
    assert "DRY_RUN" in script and "--dry-run" in script


def test_raw_client_deploy_requires_an_armed_supervisor():
    calls = []
    client = pod.RunPodClient(lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(pod.SafetyError, match="supervised lease"):
        client.deploy(name="ouro-jlens-r1", gpu="B300", disk=100,
                      public_key="ssh-ed25519 AAAA", hf_token="hf_test")
    assert not calls


def test_balance_breach_terminates_and_verifies_exact_pod(tmp_path):
    state_file = tmp_path / "state.json"
    _state(state_file)
    client = FakeClient(balance=9, pods=[_pod()])
    supervisor = pod.LeaseSupervisor(
        client,
        state_file,
        pod.SafetyConfig(min_balance=10, poll_seconds=1, termination_poll_seconds=0),
        sleep=lambda _: None,
        clock=lambda: 1,
        monotonic=lambda: 1,
    )
    with pytest.raises(pod.SafetyError, match="floor"):
        supervisor.monitor(once=True)
    assert client.terminate_calls == ["p1"] * pod.TERMINATION_ABSENCE_CONFIRMATIONS
    assert json.loads(state_file.read_text())["termination_verified"] is True


def test_runtime_and_budget_breach_are_independent_fail_closed(tmp_path):
    for limit, expected in [("max_runtime", "runtime"), ("max_spend", "spend")]:
        state_file = tmp_path / f"{limit}.json"
        _state(state_file, bound_at=0, cost=10)
        client = FakeClient(balance=100, pods=[_pod(cost=10)])
        config = pod.SafetyConfig(min_balance=5, max_runtime=5 if limit == "max_runtime" else 1000,
                                  max_spend=0.01 if limit == "max_spend" else 30,
                                  poll_seconds=1, termination_poll_seconds=0)
        supervisor = pod.LeaseSupervisor(client, state_file, config, sleep=lambda _: None,
                                         clock=lambda: 10, monotonic=lambda: 10)
        with pytest.raises(pod.SafetyError, match=expected):
            supervisor.monitor(once=True)
        assert client.terminate_calls == ["p1"] * pod.TERMINATION_ABSENCE_CONFIRMATIONS


def test_repeated_api_uncertainty_terminates_known_identity(tmp_path):
    state_file = tmp_path / "state.json"
    _state(state_file)

    class Uncertain(FakeClient):
        def __init__(self):
            super().__init__(balance=100, pods=[_pod()])
            self.n = 0

        def balance(self):
            self.n += 1
            raise pod.APIUncertain("network")

    client = Uncertain()
    supervisor = pod.LeaseSupervisor(
        client, state_file,
        pod.SafetyConfig(min_balance=5, max_api_failures=2, poll_seconds=0.01,
                         termination_poll_seconds=0),
        sleep=lambda _: None,
    )
    with pytest.raises(pod.APIUncertain):
        supervisor.monitor()
    assert client.terminate_calls == ["p1"] * pod.TERMINATION_ABSENCE_CONFIRMATIONS


def test_monitor_restart_after_verified_termination_is_clean_noop(tmp_path):
    state_file = tmp_path / "state.json"
    _state(state_file)
    state = json.loads(state_file.read_text())
    state.update({"status": "terminated", "termination_verified": True})
    state_file.write_text(json.dumps(state, sort_keys=True))
    before = state_file.read_bytes()
    client = FakeClient(balance=100, pods=[_pod()])
    supervisor = pod.LeaseSupervisor(
        client,
        state_file,
        pod.SafetyConfig(min_balance=5, poll_seconds=0.01, termination_poll_seconds=0),
        sleep=lambda _seconds: None,
    )
    supervisor.monitor()
    assert client.calls == []
    assert client.terminate_calls == []
    assert state_file.read_bytes() == before


def test_termination_verification_retries_until_absent(tmp_path):
    state_file = tmp_path / "state.json"
    _state(state_file)

    class Delayed(FakeClient):
        def terminate(self, pod_id):
            self.calls.append(("terminate", pod_id))
            self.terminate_calls.append(pod_id)
            # The provider continues listing the pod for one poll.
            if len(self.terminate_calls) > 1:
                self.current_pods = []

    client = Delayed(balance=100, pods=[_pod()])
    supervisor = pod.LeaseSupervisor(client, state_file,
                                     pod.SafetyConfig(min_balance=5, termination_attempts=4,
                                                      termination_poll_seconds=0))
    supervisor.terminate_verified()
    assert client.terminate_calls == ["p1"] * 4


def test_systemd_monitor_preflight_requires_disconnect_persistence(monkeypatch):
    monkeypatch.setenv("USER", "test-runner")
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="no" if command[0] == "loginctl" else "")

    with pytest.raises(pod.SafetyError, match="lingering"):
        pod.systemd_user_preflight(runner=runner)
    assert calls[-1][:3] == ["loginctl", "show-user", "test-runner"]


def test_publish_ack_is_immutable_and_sync_is_digest_verified(tmp_path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"important bytes")
    remote = LocalPublisher(tmp_path / "remote")
    receipts = tmp_path / "receipts"
    receipt = publish_file(remote, source, run_id="r1", relative_path="lens/shard.pt",
                           kind="shard", receipt_root=receipts)
    assert receipt.sha256
    assert (receipts / "lens/shard.pt.receipt.json").is_file()
    assert publish_file(remote, source, run_id="r1", relative_path="lens/shard.pt",
                        kind="shard") == receipt
    source.write_bytes(b"different")
    with pytest.raises(ImmutableConflict):
        publish_file(remote, source, run_id="r1", relative_path="lens/shard.pt", kind="shard")
    synced = tmp_path / "synced"
    # Restore the original bytes for a valid local verification target.
    source.write_bytes(b"important bytes")
    result = sync_and_verify(remote, run_id="r1", local_root=synced)
    assert len(result) == 1 and (synced / "lens/shard.pt").read_bytes() == b"important bytes"
    with pytest.raises(PublishError, match="must not overlap"):
        sync_and_verify(
            remote,
            run_id="r1",
            local_root=tmp_path / "overlap",
            receipt_root=tmp_path / "overlap" / "receipts",
        )


def test_publish_rejects_remote_payload_that_does_not_match_receipt(tmp_path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"important bytes")

    class CorruptPublisher(LocalPublisher):
        def put_file(self, path, remote_path):
            super().put_file(path, remote_path)
            if "/artifacts/" in remote_path:
                remote = self._path(remote_path)
                remote.chmod(0o600)
                remote.write_bytes(b"corrupted")

    with pytest.raises(ImmutableConflict, match="remote payload"):
        publish_file(CorruptPublisher(tmp_path / "remote"), source, run_id="r1",
                     relative_path="lens/shard.pt", kind="shard")


def test_publish_reconciles_transient_readback_after_successful_commit(tmp_path, monkeypatch):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"durable bytes")

    class EventuallyVisiblePublisher(LocalPublisher):
        def __init__(self, root):
            super().__init__(root)
            self.hidden = {}

        def put_file(self, path, remote_path):
            super().put_file(path, remote_path)
            self.hidden[remote_path] = 1

        def read_file(self, remote_path):
            if self.hidden.get(remote_path, 0) > 0:
                self.hidden[remote_path] -= 1
                raise PublishError("injected post-commit readback outage")
            return super().read_file(remote_path)

        def _download_file_to_fd(self, remote_path, destination_fd):
            if self.hidden.get(remote_path, 0) > 0:
                self.hidden[remote_path] -= 1
                raise PublishError("injected post-commit download outage")
            return super()._download_file_to_fd(remote_path, destination_fd)

    monkeypatch.setattr(publish_module.time, "sleep", lambda _seconds: None)
    remote = EventuallyVisiblePublisher(tmp_path / "remote")
    receipt = publish_file(
        remote, source, run_id="r1", relative_path="lens/shard.pt", kind="shard"
    )
    assert remote.read_file(receipt.remote_path) == b"durable bytes"
    assert publish_module.Receipt.from_bytes(
        remote.read_file("r1/receipts/lens/shard.pt.receipt.json")
    ) == receipt


def test_publish_resume_binds_orphan_payload_without_overwriting_it(tmp_path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"payload")
    remote = LocalPublisher(tmp_path / "remote")
    remote.put_file(source, "r1/artifacts/lens/shard.pt")
    receipt = publish_file(remote, source, run_id="r1", relative_path="lens/shard.pt", kind="shard")
    assert receipt.sha256
    assert remote.read_file("r1/artifacts/lens/shard.pt") == b"payload"


def test_restore_and_reconstruct_sidecar_after_worker_restart(tmp_path):
    from ouro_jlens.publish import write_sidecar

    source = tmp_path / "shard.pt"
    source.write_bytes(b"lens")
    remote = LocalPublisher(tmp_path / "remote")
    publish_file(remote, source, run_id="r1", relative_path="lens/shard.pt", kind="shard")
    restored = tmp_path / "restored.pt"
    receipt = restore_file(remote, run_id="r1", relative_path="lens/shard.pt", destination=restored)
    assert receipt is not None and restored.read_bytes() == b"lens"
    sidecar = tmp_path / "shard.json"
    write_sidecar(restored, sidecar, run_id="r1", relative_path="lens/shard.pt",
                  metadata={"target_ut": 3, "start": 0, "end": 25})
    assert json.loads(sidecar.read_text())["sha256"] == receipt.sha256


def test_publish_tree_has_no_extension_exclusion(tmp_path):
    source = tmp_path / "eval"
    source.mkdir()
    (source / "arrays.npz").write_bytes(b"npz")
    (source / "figure.png").write_bytes(b"png")
    (source / "summary.json").write_text("{}")
    remote = LocalPublisher(tmp_path / "remote")
    receipts = publish_tree(remote, source, run_id="r1", relative_prefix="eval/n100", kind="eval")
    assert {r.relative_path for r in receipts} == {
        "eval/n100/arrays.npz", "eval/n100/figure.png", "eval/n100/summary.json"
    }


def test_terminal_index_discovers_every_ack_for_controller_sync(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("payload")
    remote = LocalPublisher(tmp_path / "remote")
    cache = tmp_path / "receipts"
    publish_file(remote, source, run_id="r1", relative_path="lens/a.pt", kind="shard",
                 receipt_root=cache)
    index_receipt = publish_index(remote, run_id="r1", receipt_root=cache, local_root=tmp_path / "run")
    index = read_index(remote.read_file(index_receipt.remote_path), run_id="r1")
    assert index == ["lens/a.pt"]
    assert "receipt-index.json" not in index
    sync_and_verify(remote, run_id="r1", local_root=tmp_path / "synced",
                    relative_paths=[*index, "receipt-index.json"])


def test_immutable_install_does_not_follow_predictable_temp_symlink(tmp_path):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"trusted payload")
    attacker_target = tmp_path / "attacker-target"
    attacker_target.write_bytes(b"attacker bytes")
    remote = LocalPublisher(tmp_path / "remote")
    destination = remote._path("r1/artifacts/lens/shard.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictable = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    predictable.symlink_to(attacker_target)
    try:
        publish_file(remote, source, run_id="r1", relative_path="lens/shard.pt", kind="shard")
    finally:
        predictable.unlink(missing_ok=True)
    assert attacker_target.read_bytes() == b"attacker bytes"
    assert remote.read_file("r1/artifacts/lens/shard.pt") == b"trusted payload"


def _swap_owned_temp_path(temporary, attacker_kind: str, attacker_target: Path) -> None:
    """Replace a temporary pathname while retaining the original fd owner."""

    temporary.path.unlink()
    if attacker_kind == "symlink":
        temporary.path.symlink_to(attacker_target)
    else:
        temporary.path.write_bytes(b"attacker bytes")


@pytest.mark.parametrize("attacker_kind", ["symlink", "regular"])
def test_atomic_write_rejects_transition_temp_swap_without_unlinking_attacker(
    tmp_path, monkeypatch, attacker_kind
):
    destination = tmp_path / "latest.json"
    attacker_target = tmp_path / "attacker-target"
    attacker_target.write_bytes(b"must remain untouched")
    captured = {}
    original = publish_module._replace_owned

    def swap(temporary, target):
        captured["path"] = temporary.path
        _swap_owned_temp_path(temporary, attacker_kind, attacker_target)
        return original(temporary, target)

    monkeypatch.setattr(publish_module, "_replace_owned", swap)
    with pytest.raises(PublishError, match="owned inode"):
        publish_module._atomic_write(destination, b"trusted bytes")
    assert not destination.exists()
    assert attacker_target.read_bytes() == b"must remain untouched"
    if attacker_kind == "symlink":
        assert captured["path"].is_symlink()
    else:
        assert captured["path"].read_bytes() == b"attacker bytes"


@pytest.mark.parametrize("attacker_kind", ["symlink", "regular"])
def test_immutable_write_rejects_transition_temp_swap_without_unlinking_attacker(
    tmp_path, monkeypatch, attacker_kind
):
    destination = tmp_path / "receipt.json"
    attacker_target = tmp_path / "attacker-target"
    attacker_target.write_bytes(b"must remain untouched")
    captured = {}
    original = publish_module._link_immutable

    def swap(source, target, **kwargs):
        captured["path"] = source
        temporary = type("Owned", (), {"path": source})()
        _swap_owned_temp_path(temporary, attacker_kind, attacker_target)
        return original(source, target, **kwargs)

    monkeypatch.setattr(publish_module, "_link_immutable", swap)
    with pytest.raises(PublishError, match="owned inode"):
        publish_module._write_immutable(destination, b"trusted bytes")
    assert not destination.exists()
    assert attacker_target.read_bytes() == b"must remain untouched"
    if attacker_kind == "symlink":
        assert captured["path"].is_symlink()
    else:
        assert captured["path"].read_bytes() == b"attacker bytes"


@pytest.mark.parametrize("attacker_kind", ["symlink", "regular"])
def test_receipt_upload_rejects_transition_temp_swap_without_unlinking_attacker(
    tmp_path, monkeypatch, attacker_kind
):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"trusted payload")
    attacker_target = tmp_path / "attacker-target"
    attacker_target.write_bytes(b"must remain untouched")
    remote = LocalPublisher(tmp_path / "remote")
    original = LocalPublisher.put_file
    captured = {}

    def swap(publisher, owned, remote_path):
        if remote_path.endswith(".receipt.json"):
            captured["path"] = owned.path
            _swap_owned_temp_path(owned, attacker_kind, attacker_target)
        return original(publisher, owned, remote_path)

    monkeypatch.setattr(LocalPublisher, "put_file", swap)
    with pytest.raises(PublishError, match="owned inode"):
        publish_file(remote, source, run_id="r1", relative_path="lens/shard.pt", kind="shard")
    assert attacker_target.read_bytes() == b"must remain untouched"
    assert captured["path"].exists()
    if attacker_kind == "regular":
        assert not captured["path"].is_symlink()
    else:
        assert captured["path"].is_symlink()
    with pytest.raises(publish_module.MissingRemote):
        remote.read_file("r1/receipts/lens/shard.pt.receipt.json")


@pytest.mark.parametrize("attacker_kind", ["symlink", "regular"])
@pytest.mark.parametrize("operation", ["restore", "sync"])
def test_restore_and_sync_reject_transition_temp_swap_without_unlinking_attacker(
    tmp_path, monkeypatch, attacker_kind, operation
):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"trusted payload")
    attacker_target = tmp_path / "attacker-target"
    attacker_target.write_bytes(b"must remain untouched")
    remote = LocalPublisher(tmp_path / "remote")
    publish_file(remote, source, run_id="r1", relative_path="lens/shard.pt", kind="shard")
    captured = {}
    original = publish_module._download_remote

    def swap(publisher, remote_path, temporary, *, expected_size=None):
        captured["path"] = temporary.path
        _swap_owned_temp_path(temporary, attacker_kind, attacker_target)
        return original(
            publisher, remote_path, temporary, expected_size=expected_size
        )

    monkeypatch.setattr(publish_module, "_download_remote", swap)
    with pytest.raises(PublishError, match="owned inode"):
        if operation == "restore":
            restore_file(
                remote,
                run_id="r1",
                relative_path="lens/shard.pt",
                destination=tmp_path / "restored.pt",
            )
        else:
            sync_and_verify(remote, run_id="r1", local_root=tmp_path / "synced")
    assert attacker_target.read_bytes() == b"must remain untouched"
    assert captured["path"].exists()
    if attacker_kind == "symlink":
        assert captured["path"].is_symlink()
    else:
        assert captured["path"].read_bytes() == b"attacker bytes"


def test_concurrent_different_bytes_have_one_immutable_winner(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    remote = LocalPublisher(tmp_path / "remote")
    barrier = threading.Barrier(2)

    def publish(source):
        barrier.wait()
        try:
            return publish_file(remote, source, run_id="r1", relative_path="lens/shard.pt", kind="shard")
        except Exception as exc:  # assert the exact immutable failure below
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, (first, second)))
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, ImmutableConflict) for result in results) == 1
    assert len(remote.list_files("r1/receipts/")) == 1


def test_sidecar_creation_is_exclusive_under_different_bytes(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    output = tmp_path / "sidecar.json"
    barrier = threading.Barrier(2)

    def create(source):
        barrier.wait()
        try:
            return publish_module.write_sidecar(
                source, output, run_id="r1", relative_path="lens/shard.pt"
            )
        except Exception as exc:  # assert the exact immutable failure below
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, (first, second)))
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, ImmutableConflict) for result in results) == 1
    assert json.loads(output.read_text())["run_id"] == "r1"


def test_versioned_indexes_union_resume_and_survive_missing_terminal_generation(tmp_path):
    remote = LocalPublisher(tmp_path / "remote")
    cache = tmp_path / "receipts"
    run_root = tmp_path / "run"
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    publish_file(remote, first, run_id="r1", relative_path="lens/first.pt", kind="shard",
                 receipt_root=cache)
    first_index = publish_index(remote, run_id="r1", receipt_root=cache, local_root=run_root)
    publish_file(remote, second, run_id="r1", relative_path="lens/second.pt", kind="shard",
                 receipt_root=cache)
    second_index = publish_index(remote, run_id="r1", receipt_root=cache, local_root=run_root)
    assert first_index.relative_path != second_index.relative_path
    assert first_index.relative_path.startswith("receipt-index.")
    assert second_index.relative_path.startswith("receipt-index.")
    assert discover_receipt_paths(remote, "r1") == [
        "lens/first.pt", "lens/second.pt", first_index.relative_path, second_index.relative_path
    ]
    remote._path(second_index.remote_path).chmod(0o600)
    remote._path(second_index.remote_path).write_bytes(
        publish_module.canonical_json({"schema_version": 1, "run_id": "r1", "relative_paths": []})
    )
    with pytest.raises(publish_module.PublishError, match="bound to its path"):
        discover_receipt_paths(remote, "r1")
    remote._path(second_index.remote_path).write_bytes(
        publish_module.canonical_json({
            "schema_version": 1,
            "run_id": "r1",
            "relative_paths": ["lens/first.pt", "lens/second.pt"],
        })
    )
    synced = tmp_path / "synced"
    assert {receipt.relative_path for receipt in sync_and_verify(remote, run_id="r1", local_root=synced)} == set(
        discover_receipt_paths(remote, "r1")
    )
    # Remove the newest generation in both directions. The earlier receipts
    # and direct receipt listing still make the full artifact union recoverable.
    remote._path(second_index.remote_path).unlink()
    remote._path(f"r1/receipts/{second_index.relative_path}.receipt.json").unlink()
    discovered = discover_receipt_paths(remote, "r1")
    assert "lens/first.pt" in discovered and "lens/second.pt" in discovered
    assert second_index.relative_path not in discovered
    # A resumed run may have no terminal generation at all. Direct receipt
    # enumeration still recovers every acknowledged artifact.
    remote._path(first_index.remote_path).unlink()
    remote._path(f"r1/receipts/{first_index.relative_path}.receipt.json").unlink()
    assert discover_receipt_paths(remote, "r1") == ["lens/first.pt", "lens/second.pt"]


def test_success_inventory_accepts_index_history_but_requires_complete_generation(tmp_path):
    remote = LocalPublisher(tmp_path / "remote")
    cache = tmp_path / "receipts"
    run_root = tmp_path / "run"
    source_root = tmp_path / "sources"

    def acknowledge(relative, kind, data):
        source = source_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(data)
        receipt = publish_file(
            remote,
            source,
            run_id="r1",
            relative_path=relative,
            kind=kind,
            receipt_root=cache,
        )
        return {
            "path": relative,
            "receipt": f"{relative}.receipt.json",
            "kind": kind,
            "size": receipt.size,
            "sha256": receipt.sha256,
        }

    def pair(stem, payload_kind):
        return {
            "payload": acknowledge(f"{stem}.pt", payload_kind, f"{stem}:pt".encode()),
            "sidecar": acknowledge(f"{stem}.json", "sidecar", f"{stem}:json".encode()),
        }

    first = acknowledge("lens/n80/exit0_shard_0000_0080.pt", "shard", b"first shard")
    early_index = publish_index(remote, run_id="r1", receipt_root=cache, local_root=run_root)
    intervals = {
        "0": [[0, 80]], "1": [[0, 80]], "2": [[0, 80]],
        "3": [[0, 8], [8, 32], [32, 56], [56, 80]],
    }
    lenses = {}
    variable = []
    for target in range(4):
        shards = []
        for start, end in intervals[str(target)]:
            stem = f"lens/n80/exit{target}_shard_{start:04d}_{end:04d}"
            if target == 0:
                shard_pair = {
                    "payload": first,
                    "sidecar": acknowledge(f"{stem}.json", "sidecar", b"sidecar"),
                }
            else:
                shard_pair = pair(stem, "shard")
            shard_pair.update({"target_ut": target, "start": start, "end": end})
            shards.append(shard_pair)
            if target == 3:
                variable.append(shard_pair)
        lenses[str(target)] = {
            "intervals": intervals[str(target)],
            "shards": shards,
            "merged": pair(f"lens/n80/exit{target}", "merged_lens"),
        }

    target_three_sources = [
        f"exit3_shard_{start:04d}_{end:04d}.pt" for start, end in intervals["3"]
    ]
    prefixes = []
    for index, size in enumerate((8, 32, 56, 80), start=1):
        entry = pair(f"lens/n80/exit3_n{size}", "prefix_lens")
        entry.update({"n_prompts": size, "sources": target_three_sources[:index]})
        prefixes.append(entry)

    eval_names = (
        "arrays.npz", "items.json", "task_names.json", "provenance.json",
        "summary.json", "summary.md",
    )
    main_tags = ["n80_exit3", "n80_allexits", "n80_allexits_pos-2"]
    fit_tags = [f"fitsize_n{size}" for size in (8, 32, 56, 80)]
    def evaluation(tag):
        return {
            "tag": tag,
            "files": [
                acknowledge(f"eval/{tag}/{name}", "eval_output", f"{tag}:{name}".encode())
                for name in eval_names
            ],
        }
    main_evaluations = [evaluation(tag) for tag in main_tags]
    fit_evaluations = []
    for size, tag in zip((8, 32, 56, 80), fit_tags):
        entry = evaluation(tag)
        entry["n_prompts"] = size
        fit_evaluations.append(entry)

    report_products = [
        acknowledge(relative, "verification" if relative == "final/verification.json" else "report",
                    f"report:{relative}".encode())
        for relative in sorted(pod.EXPECTED_REPORT_PRODUCTS)
    ]
    validation_products = [
        acknowledge("validation/milestones.json", "validation", b"validation")
    ]
    launch_nonce = "n" * 48
    stage_manifest_sha256 = "a" * 64
    stage_source_head = "b" * 40
    worker_deadline_epoch = int(time.time()) + 3600
    inventory = {
        "schema_version": 2,
        "run_id": "r1",
        "attempt_id": "attempt-1",
        "launch_nonce": launch_nonce,
        "stage_manifest_sha256": stage_manifest_sha256,
        "stage_source_head": stage_source_head,
        "worker_deadline_epoch": worker_deadline_epoch,
        "n_prompts": 80,
        "shard_size": 80,
        "shard_intervals": intervals,
        "variable_shards": variable,
        "lens": lenses,
        "prefix_lenses": prefixes,
        "main_evaluations": main_evaluations,
        "fit_size_evaluations": fit_evaluations,
        "report_products": report_products,
        "validation_products": validation_products,
        "evaluation_tags": main_tags,
    }
    acknowledge("inventory/expected-inventory.json", "expected_inventory",
                publish_module.canonical_json(inventory))
    for tag in [*main_tags, *fit_tags]:
        acknowledge(f"markers/{tag}.complete", "marker", f"r1:{tag}\n".encode())
    acknowledge("logs/current.log", "run_log", b"complete log")
    acknowledge(
        "recovery/attempt-1.json",
        "recovery_status",
        publish_module.canonical_json({
            "schema_version": 1,
            "run_id": "r1",
            "attempt_id": "attempt-1",
            "launch_nonce": launch_nonce,
            "stage_manifest_sha256": stage_manifest_sha256,
            "stage_source_head": stage_source_head,
            "worker_deadline_epoch": worker_deadline_epoch,
            "recovered_checkpoint_pairs": 0,
            "complete": True,
        }),
    )
    acknowledge(
        "status/complete.json",
        "heartbeat",
        publish_module.canonical_json({
            "schema_version": 1,
            "run_id": "r1",
                "state": "succeeded",
                "stage": "complete",
                "exit_code": 0,
                "attempt_id": "attempt-1",
                "launch_nonce": launch_nonce,
                "stage_manifest_sha256": stage_manifest_sha256,
                "stage_source_head": stage_source_head,
                "worker_deadline_epoch": worker_deadline_epoch,
                "recorded_at": 1.0,
            }),
    )
    final_index = publish_index(remote, run_id="r1", receipt_root=cache, local_root=run_root)
    synced = tmp_path / "synced"
    local_receipts = tmp_path / "synced-receipts"
    receipts = sync_and_verify(
        remote, run_id="r1", local_root=synced, receipt_root=local_receipts
    )
    pod._verify_success_inventory(
        receipts, synced, run_id="r1", n_prompts=80, attempt_id="attempt-1",
        launch_nonce=launch_nonce, stage_manifest_sha256=stage_manifest_sha256,
        stage_source_head=stage_source_head,
        worker_deadline_epoch=worker_deadline_epoch,
    )
    assert len(list(local_receipts.rglob("*.receipt.json"))) == len(receipts)
    with pytest.raises(PublishError, match="current lease attempt"):
        pod._verify_success_inventory(
            receipts, synced, run_id="r1", n_prompts=80, attempt_id="attempt-2",
            launch_nonce=launch_nonce, stage_manifest_sha256=stage_manifest_sha256,
            stage_source_head=stage_source_head,
            worker_deadline_epoch=worker_deadline_epoch,
        )
    without_one_scientific = [
        receipt for receipt in receipts
        if receipt.relative_path != "eval/n80_exit3/arrays.npz"
    ]
    with pytest.raises(PublishError, match="inventory and scientific receipts differ"):
        pod._verify_success_inventory(
            without_one_scientific, synced, run_id="r1",
            n_prompts=80, attempt_id="attempt-1", launch_nonce=launch_nonce,
            stage_manifest_sha256=stage_manifest_sha256,
            stage_source_head=stage_source_head,
            worker_deadline_epoch=worker_deadline_epoch,
        )

    without_final = [receipt for receipt in receipts if receipt != final_index]
    assert early_index in without_final
    with pytest.raises(PublishError, match="no receipt index generation covers"):
        pod._verify_success_inventory(
            without_final, synced, run_id="r1", n_prompts=80, attempt_id="attempt-1",
            launch_nonce=launch_nonce, stage_manifest_sha256=stage_manifest_sha256,
            stage_source_head=stage_source_head,
            worker_deadline_epoch=worker_deadline_epoch,
        )


def test_hf_listing_is_lazy_and_fails_closed(tmp_path):
    class FakeApi:
        def __init__(self):
            self.calls = []

        def list_repo_tree(self, **kwargs):
            self.calls.append(kwargs)
            return [
                type("File", (), {"rfilename": "r1/receipts/a.receipt.json"})(),
                type("Folder", (), {"path": "r1/receipts"})(),
                {"rfilename": "r1/receipts/b.receipt.json"},
            ]

    api = FakeApi()
    publisher = HfPublisher("org/results", api=api)
    assert publisher.list_files("r1/receipts/") == [
        "r1/receipts/a.receipt.json", "r1/receipts/b.receipt.json"
    ]
    assert api.calls == [{"repo_id": "org/results", "path_in_repo": "r1/receipts", "recursive": True}]

    class BrokenApi:
        def list_repo_tree(self, **kwargs):
            raise RuntimeError("network unavailable")

    with pytest.raises(publish_module.PublishError, match="enumerate"):
        HfPublisher("org/results", api=BrokenApi()).list_files("r1/receipts/")


def test_status_receipt_binds_current_lease_attempt(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    local = tmp_path / "local"
    receipts = tmp_path / "receipts"
    monkeypatch.setenv("JLENS_ATTEMPT_ID", "attempt-7")
    monkeypatch.setenv("JLENS_LAUNCH_NONCE", "n" * 48)
    monkeypatch.setenv("EXPECTED_STAGE_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("EXPECTED_STAGE_SOURCE_HEAD", "b" * 40)
    monkeypatch.setenv("JLENS_WORKER_DEADLINE_EPOCH", str(int(time.time()) + 3600))
    args = SimpleNamespace(
        repo=None,
        local_publisher=remote,
        run_id="artifact-1",
        local_root=local,
        state="succeeded",
        stage="complete",
        exit_code=0,
        kind="heartbeat",
        receipt_root=receipts,
    )
    assert publish_module._cmd_status(args) == 0
    publisher = LocalPublisher(remote)
    paths = discover_receipt_paths(publisher, "artifact-1")
    assert len(paths) == 1 and paths[0].startswith("status/")
    payload = json.loads(publisher.read_file(f"artifact-1/artifacts/{paths[0]}"))
    assert payload["run_id"] == "artifact-1"
    assert payload["attempt_id"] == "attempt-7"
    assert payload["launch_nonce"] == "n" * 48
    assert payload["expected_stage_manifest_sha256"] == "a" * 64
    assert payload["expected_stage_source_head"] == "b" * 40


class _FakeHubParentConflict(Exception):
    response = SimpleNamespace(status_code=412)


class _ConcurrentFakeHub:
    """Small production-shaped fake for HfApi's head/CAS operations."""

    def __init__(self, *, synchronize_heads=False, unrelated_conflict=False):
        self.head = "a" * 40
        self.files = {}
        self.history = {self.head: {}}
        self.calls = []
        self.lock = threading.Lock()
        self._head_calls = 0
        self._heads_ready = threading.Barrier(2) if synchronize_heads else None
        self.unrelated_conflict = unrelated_conflict

    def repo_info(self, **kwargs):
        assert kwargs == {
            "repo_id": "org/results",
            "revision": "main",
            "repo_type": "model",
            "token": False,
        }
        with self.lock:
            self._head_calls += 1
            call = self._head_calls
            head = self.head
        if self._heads_ready is not None and call <= 2:
            self._heads_ready.wait(timeout=5)
        return SimpleNamespace(sha=head)

    def read_file(self, remote_path, *, expected_size=None, revision=None):
        with self.lock:
            files = self.files if revision is None else self.history.get(revision)
            if files is None or remote_path not in files:
                raise publish_module.MissingRemote(remote_path)
            payload = files[remote_path]
            if expected_size is not None and len(payload) != expected_size:
                raise publish_module.ImmutableConflict(remote_path)
            return payload

    def create_commit(self, **kwargs):
        assert kwargs["repo_id"] == "org/results"
        assert kwargs["repo_type"] == "model"
        assert kwargs["revision"] == "main"
        assert kwargs["commit_message"] == "ouro-jlens: publish immutable artifact"
        operation = list(kwargs["operations"])[0]
        with open(operation.path_or_fileobj, "rb") as handle:
            payload = handle.read()
        path = operation.path_in_repo
        with self.lock:
            self.calls.append((kwargs["parent_commit"], path, payload))
            if self.unrelated_conflict:
                self.unrelated_conflict = False
                self.head = hashlib.sha1((self.head + "unrelated").encode()).hexdigest()
                self.history[self.head] = dict(self.files)
                raise _FakeHubParentConflict()
            if kwargs["parent_commit"] != self.head:
                raise _FakeHubParentConflict()
            if path in self.files:
                # A real immutable publisher should never reach this branch:
                # the caller's target preflight rejects an existing path.
                raise AssertionError("CAS publisher attempted to overwrite a target")
            self.files[path] = payload
            self.head = hashlib.sha1((self.head + path + payload.hex()).encode()).hexdigest()
            self.history[self.head] = dict(self.files)


def _hf_fake_publisher(hub):
    publisher = HfPublisher("org/results", api=hub)
    publisher.read_file = hub.read_file
    return publisher


def test_hf_cas_different_concurrent_writers_have_one_winner(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    hub = _ConcurrentFakeHub(synchronize_heads=True)
    publishers = [_hf_fake_publisher(hub), _hf_fake_publisher(hub)]
    barrier = threading.Barrier(2)

    def publish(pair):
        publisher, source = pair
        barrier.wait(timeout=5)
        try:
            publisher.put_file(source, "immutable.bin")
            return None
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, zip(publishers, (first, second))))
    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, ImmutableConflict) for result in results) == 1
    assert len(hub.calls) == 2
    assert hub.files["immutable.bin"] in {b"first", b"second"}


def test_hf_cas_identical_existing_bytes_are_idempotent(tmp_path):
    source = tmp_path / "same.bin"
    source.write_bytes(b"same payload")
    hub = _ConcurrentFakeHub()
    _hf_fake_publisher(hub).put_file(source, "immutable.bin")
    _hf_fake_publisher(hub).put_file(source, "immutable.bin")
    assert hub.files["immutable.bin"] == b"same payload"
    assert len(hub.calls) == 1


def test_hf_cas_never_mutates_twice_after_unrelated_head_conflict(tmp_path):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    hub = _ConcurrentFakeHub(unrelated_conflict=True)
    with pytest.raises(PublishError, match="without a second mutation"):
        _hf_fake_publisher(hub).put_file(source, "immutable.bin")
    assert len(hub.calls) == 1
    assert "immutable.bin" not in hub.files


def test_hf_cas_target_check_is_bound_to_same_parent_as_commit(tmp_path):
    """A rival write in the old head/read gap must never be overwritten."""

    source = tmp_path / "candidate.bin"
    source.write_bytes(b"candidate")
    hub = _ConcurrentFakeHub()
    publisher = _hf_fake_publisher(hub)
    original_read = hub.read_file
    injected = False

    def read_with_rival(remote_path, *, expected_size=None, revision=None):
        nonlocal injected
        if revision is not None and not injected:
            injected = True
            with hub.lock:
                assert revision == "a" * 40
                hub.files[remote_path] = b"rival"
                hub.head = hashlib.sha1((hub.head + remote_path + "rival").encode()).hexdigest()
                hub.history[hub.head] = dict(hub.files)
        return original_read(
            remote_path,
            expected_size=expected_size,
            revision=revision,
        )

    publisher.read_file = read_with_rival
    with pytest.raises(ImmutableConflict, match="already differs"):
        publisher.put_file(source, "immutable.bin")
    assert len(hub.calls) == 1
    assert hub.files["immutable.bin"] == b"rival"


def test_stage_rechecks_head_and_status_after_archive(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    payloads = publish_module._stage_payload_paths(root)
    committed = {
        path.relative_to(root).as_posix(): publish_module.sha256_file(path)
        for path in payloads
        if "artifacts" not in path.relative_to(root).parts
    }
    calls = iter([
        {"head": "a" * 40, "status": "", "committed_payload_sha256": committed},
        {"head": "a" * 40, "status": " M src/ouro_jlens/example.py", "committed_payload_sha256": committed},
    ])
    monkeypatch.setattr(publish_module, "_git_pin", lambda *_args: next(calls))
    with pytest.raises(StageVerificationError, match="HEAD/status changed"):
        build_stage_tar(root, tmp_path / "stage.tar.gz", require_clean_source=True)


def test_stage_clean_policy_binds_payload_bytes_to_head_blob(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    source = root / "src/ouro_jlens/example.py"
    source.write_text("working tree bytes\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    committed = _committed_payload_map(root)
    committed["src/ouro_jlens/example.py"] = publish_module.sha256_bytes(b"committed HEAD bytes\n")
    monkeypatch.setattr(
        publish_module,
        "_git_pin",
        lambda *_args: {"head": "a" * 40, "status": "",
                        "committed_payload_sha256": committed},
    )
    with pytest.raises(StageVerificationError, match="committed HEAD blob"):
        build_stage_tar(root, tmp_path / "stage.tar.gz", require_clean_source=True)


def test_clean_stage_verifier_binds_manifest_to_committed_map(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    committed = _committed_payload_map(root)
    monkeypatch.setattr(
        publish_module,
        "_git_pin",
        lambda *_args: {"head": "a" * 40, "status": "",
                        "committed_payload_sha256": committed},
    )
    archive = tmp_path / "stage.tar.gz"
    build_stage_tar(root, archive, require_clean_source=True)
    extracted = tmp_path / "extracted"
    extract_stage_tar(archive, extracted)
    pinned_path = extracted / "PINNED_INPUTS.json"
    pinned = json.loads(pinned_path.read_text())
    first = sorted(pinned["committed_payload_sha256"])[0]
    pinned["committed_payload_sha256"][first] = "0" * 64
    pinned_path.write_bytes(publish_module.canonical_json(pinned))
    with pytest.raises(StageVerificationError, match="does not match"):
        verify_stage_root(extracted)


@pytest.mark.parametrize("existing_kind", ["file", "directory", "symlink"])
def test_stage_extract_rejects_any_existing_destination(tmp_path, existing_kind):
    archive = _build_extraction_stage(tmp_path)
    destination = tmp_path / "installed"
    target = tmp_path / "untouched-target"
    if existing_kind == "file":
        destination.write_bytes(b"keep me")
    elif existing_kind == "directory":
        destination.mkdir()
        (destination / "sentinel").write_bytes(b"keep me")
    else:
        target.mkdir()
        (target / "sentinel").write_bytes(b"keep me")
        destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(StageVerificationError, match="already exists"):
        extract_stage_tar(archive, destination)

    if existing_kind == "file":
        assert destination.read_bytes() == b"keep me"
    elif existing_kind == "directory":
        assert (destination / "sentinel").read_bytes() == b"keep me"
    else:
        assert destination.is_symlink()
        assert (target / "sentinel").read_bytes() == b"keep me"


def test_stage_extract_failure_leaves_no_destination_or_staging_tree(monkeypatch, tmp_path):
    archive = _build_extraction_stage(tmp_path)
    destination = tmp_path / "installed"
    original = publish_module._extract_regular_member
    calls = 0

    def fail_after_first(archive_file, member, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected extraction failure")
        return original(archive_file, member, target)

    monkeypatch.setattr(publish_module, "_extract_regular_member", fail_after_first)
    with pytest.raises(StageVerificationError, match="cannot extract"):
        extract_stage_tar(archive, destination)

    assert calls == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}.jlens-stage-*"))


def test_stage_extract_preserves_executable_member_mode(tmp_path):
    archive = _build_extraction_stage(tmp_path, executable=True)
    destination = tmp_path / "installed"

    extract_stage_tar(archive, destination)

    entrypoint = destination / "src/ouro_jlens/entrypoint.sh"
    assert entrypoint.read_text() == "#!/bin/sh\nexit 0\n"
    assert stat.S_IMODE(entrypoint.stat().st_mode) == 0o755


def test_publisher_entrypoints_close_checkpoint_and_log_residuals():
    run_script = Path("src/ouro_jlens/run_b300.sh").read_text()
    entry_script = Path("src/ouro_jlens/pod_entry.sh").read_text()
    setup_script = Path("src/ouro_jlens/setup_b300.sh").read_text()
    assert ">(tee" not in run_script
    assert "< <(" not in run_script
    assert "sync_log" in run_script and "os.fsync" in run_script and "log_identity" in run_script
    assert "verify_receipt" in run_script
    assert 'if publish_file "$checkpoint"' in run_script
    assert run_script.index('publish_status failed run') < run_script.index('if ! sync_log')
    assert "quarantine_orphan" in run_script
    assert '[[ -f "$out.ckpt" && ! -f "$out.ckpt.json" ]]' in run_script
    assert '[[ -f "$out.ckpt.json" && ! -f "$out.ckpt" ]]' in run_script
    assert "receipt-index.json" not in entry_script
    assert entry_script.count("ouro_jlens.publish index") == 1
    assert "EXPECTED_TORCH_VERSION='2.8.0+cu128'" in setup_script
    assert "torch.version.cuda" in setup_script
    assert "JLENS_IMAGE_DIGEST is required" in entry_script
    assert "JLENS_ATTEMPT_ID is required" in entry_script
    assert "a paid attempt requires a fresh workspace" in entry_script
    assert 'mktemp -d "/tmp/ouro-jlens-stage.' in entry_script
    assert "from ouro_jlens.publish import RUNTIME_IMAGE" in entry_script
    assert run_script.count("--checkpoints ''") == 2


def test_paid_b300_interval_plan_is_explicit_and_unambiguous():
    script = Path("src/ouro_jlens/run_b300.sh").read_text()
    # The eventual target owns the exact nested prefixes; N>80 contributes
    # only one final tail. Targets 0-2 remain ordinary SHARD_SIZE partitions.
    assert "printf '%s %s\\n' 0 8" in script
    assert "printf '%s %s\\n' 8 32" in script
    assert "printf '%s %s\\n' 32 56" in script
    assert "printf '%s %s\\n' 56 80" in script
    assert "printf '%s %s\\n' 80 \"$N\"" in script
    assert "for ((s = 0; s < N; s += SHARD)); do" in script
    assert "B300 contract requires N >= 80" in script
    # Including both boundaries in every filename prevents a variable tail
    # from colliding with a fixed-size shard during restore/publication.
    assert "exit%s_shard_%04d_%04d.pt" in script
    assert "exit${ut}_shard_$(printf '%04d' \"$s\")" not in script


def test_paid_b300_interval_plan_executes_exact_boundaries():
    script = Path("src/ouro_jlens/run_b300.sh").read_text()
    start = script.index("plan_intervals() {")
    end = script.index("\n}\n\nshard_path()", start) + 2
    function = script[start:end]

    def planned(n, target, shard=25):
        result = subprocess.run(
            ["bash", "-c", function + "\nplan_intervals \"$1\"", "plan", str(target)],
            env={**os.environ, "N": str(n), "SHARD": str(shard)},
            text=True,
            capture_output=True,
            check=True,
        )
        return [tuple(map(int, line.split())) for line in result.stdout.splitlines()]

    assert planned(100, 3) == [(0, 8), (8, 32), (32, 56), (56, 80), (80, 100)]
    assert planned(120, 3) == [(0, 8), (8, 32), (32, 56), (56, 80), (80, 120)]
    assert planned(100, 2) == [(0, 25), (25, 50), (50, 75), (75, 100)]


def test_paid_b300_wires_prefix_evaluations_reports_and_inventory():
    script = Path("src/ouro_jlens/run_b300.sh").read_text()
    for name in ("PREFIX_N8", "PREFIX_N32", "PREFIX_N56", "PREFIX_N80"):
        assert name in script
    for command in (
        "fit_lens.py merge",
        'run_eval "fitsize_n8"',
        'run_eval "fitsize_n32"',
        'run_eval "fitsize_n56"',
        'run_eval "fitsize_n80"',
        "lens_convergence.py",
        "fitsize_report.py",
        "transport_report.py",
        "report.py",
        "publish_file \"$report\"",
        "verify_expected_inventory",
    ):
        assert command in script
    for product in (
        "fitsize_convergence.json",
        'FIT_SIZE_OUT="$FINAL_ROOT/fit_size.json"',
        'TRANSPORT_OUT="$FINAL_ROOT/transport.json"',
        "prefix_lenses",
        "main_evaluations",
        "fit_size_evaluations",
        "report_products",
    ):
        assert product in script
    assert "_write_immutable(inventory_path" in script
    assert "verify_local_receipt" in script


def test_stage_archive_contains_manifest_and_rejects_tampering(tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    archive = tmp_path / "stage.tar.gz"
    result = build_stage_tar(root, archive)
    second_archive = tmp_path / "stage-second.tar.gz"
    build_stage_tar(root, second_archive)
    assert archive.read_bytes() == second_archive.read_bytes()
    assert result["payload_count"] == 5
    verified = verify_stage_tar(archive)
    assert verified["manifest_sha256"] == result["manifest_sha256"]
    with tarfile.open(archive, "r:*") as tar:
        names = tar.getnames()
    assert "MANIFEST.sha256" in names and "PINNED_INPUTS.json" in names


def test_stage_builder_rejects_forged_corpus_provenance_before_replacing_output(tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    provenance = root / publish_module.FITTING_CORPUS_PROVENANCE_RELATIVE
    forged = json.loads(provenance.read_text())
    forged["source"]["revision"] = "0" * 40
    provenance.write_text(json.dumps(forged))
    output = tmp_path / "stage.tar.gz"
    output.write_bytes(b"previous-stage")

    with pytest.raises(StageVerificationError, match="corpus provenance is not trusted"):
        build_stage_tar(root, output)
    assert output.read_bytes() == b"previous-stage"


def test_stage_rejects_self_consistent_fabricated_corpus(tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    corpus = root / publish_module.FITTING_CORPUS_RELATIVE
    corpus.write_bytes(json.dumps(["fabricated " + "x" * 600] * 1200).encode() + b"\n")
    provenance = root / publish_module.FITTING_CORPUS_PROVENANCE_RELATIVE
    document = json.loads(provenance.read_text())
    document["output"] = {
        "path": "wikitext_prompts",
        "size": corpus.stat().st_size,
        "sha256": publish_module.sha256_file(corpus),
    }
    provenance.write_bytes(publish_module.canonical_json(document))
    with pytest.raises(StageVerificationError, match="trusted corpus"):
        build_stage_tar(root, tmp_path / "stage.tar.gz")


def test_paid_stage_requires_clean_committed_payload(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    monkeypatch.setattr(
        publish_module,
        "_git_pin",
        lambda _root, _payloads: {"head": "a" * 40, "status": "?? src/ouro_jlens/example.py"},
    )
    output = tmp_path / "stage.tar.gz"
    output.write_bytes(b"previous-complete-stage")
    with pytest.raises(StageVerificationError, match="not clean and committed"):
        build_stage_tar(root, output, require_clean_source=True)
    assert output.read_bytes() == b"previous-complete-stage"


def test_stage_includes_available_docs_and_contract_tests(tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "docs/jlens").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    (root / "utilities/tests/unit/test_jlens_integrity.py").write_text("def test_integrity(): pass\n")
    (root / "utilities/tests/unit/test_jlens_future_guard.py").write_text("def test_future(): pass\n")
    (root / "docs/jlens/README.md").write_text("# Contract\n")
    (root / "docs/jlens/watch").mkdir()
    (root / "docs/jlens/watch/BOARD.md").write_text("mutable operator channel\n")
    archive = tmp_path / "stage.tar.gz"
    build_stage_tar(root, archive)
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert "docs/jlens/README.md" in names
    assert "docs/jlens/watch/BOARD.md" not in names
    assert "utilities/tests/unit/test_jlens_integrity.py" in names
    assert "utilities/tests/unit/test_jlens_future_guard.py" in names


def test_controller_bootstrap_comes_from_verified_clean_stage(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/publish.py").write_text("PIN = 'stage'\n")
    (root / "src/ouro_jlens/pod_entry.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    committed = _committed_payload_map(root)
    monkeypatch.setattr(
        publish_module,
        "_git_pin",
        lambda _root, _payloads: {"head": "a" * 40, "status": "",
                                  "committed_payload_sha256": committed},
    )
    archive = tmp_path / "stage.tar.gz"
    built = build_stage_tar(root, archive, require_clean_source=True)
    stage_path = f"stages/{built['manifest_sha256']}/stage.tar.gz"
    remote = LocalPublisher(tmp_path / "remote")
    remote.put_file(archive, stage_path)
    package = pod._prepare_stage_bootstrap(
        stage_path, "org/staging", tmp_path / "bootstrap", publisher=remote
    )
    assert (package.package / "publish.py").read_text() == "PIN = 'stage'\n"
    assert (package.package / "pod_entry.sh").is_file()
    assert package.manifest_sha256 == built["manifest_sha256"]
    assert package.source_head == "a" * 40


def test_controller_rejects_stage_whose_path_names_another_manifest(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/publish.py").write_text("PIN = 'stage'\n")
    (root / "src/ouro_jlens/pod_entry.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    committed = _committed_payload_map(root)
    monkeypatch.setattr(
        publish_module,
        "_git_pin",
        lambda _root, _payloads: {"head": "a" * 40, "status": "",
                                  "committed_payload_sha256": committed},
    )
    archive = tmp_path / "stage.tar.gz"
    build_stage_tar(root, archive, require_clean_source=True)
    stage_path = f"stages/{'b' * 64}/stage.tar.gz"
    remote = LocalPublisher(tmp_path / "remote")
    remote.put_file(archive, stage_path)
    with pytest.raises(pod.SafetyError, match="digest-addressed path"):
        pod._prepare_stage_bootstrap(
            stage_path, "org/staging", tmp_path / "bootstrap", publisher=remote
        )


def test_stage_allow_extra_rejects_python_shadow_but_allows_run_data(tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    archive = tmp_path / "stage.tar.gz"
    build_stage_tar(root, archive)
    extracted = tmp_path / "extracted"
    extract_stage_tar(archive, extracted)
    (extracted / "artifacts/jlens/extra.json").parent.mkdir(parents=True, exist_ok=True)
    (extracted / "artifacts/jlens/extra.json").write_text("{}\n")
    verify_stage_root(extracted, allow_extra=True)
    shadow = extracted / "src/ouro_jlens/unlisted.py"
    shadow.write_text("raise RuntimeError('shadow')\n")
    with pytest.raises(StageVerificationError, match="unlisted stage payload"):
        verify_stage_root(extracted, allow_extra=True)


def test_stage_and_local_publisher_reject_symlinked_roots(tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    _write_stage_corpus(root)
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    linked_root = tmp_path / "linked-repo"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(StageVerificationError, match="stage root is a symlink"):
        build_stage_tar(linked_root, tmp_path / "stage.tar.gz")
    linked_remote = tmp_path / "linked-remote"
    linked_remote.symlink_to(tmp_path / "remote-target", target_is_directory=True)
    with pytest.raises(publish_module.PublishError, match="publisher root is a symlink"):
        LocalPublisher(linked_remote)


def test_stage_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        import io
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(StageVerificationError):
        verify_stage_tar(archive)


def test_controller_preserves_remote_exit_status_and_releases_lease(monkeypatch, tmp_path):
    args = pod._parser().parse_args([
        "run", "--run-id", "r1", "--confirm-launch", "r1",
        "--results", "org/results", "--stage-path",
        "stages/" + "a" * 64 + "/stage.tar.gz", "--state-root", str(tmp_path),
        "--sync-root", str(tmp_path / "sync"), "--receipt-root", str(tmp_path / "receipts"),
        "--image", publish_module.RUNTIME_IMAGE,
    ])

    class Supervisor:
        def __init__(self):
            self.terminated = 0

        def terminate_verified(self):
            self.terminated += 1

    supervisor = Supervisor()
    bootstrap = SimpleNamespace(
        package=Path("src/ouro_jlens"),
        root=tmp_path,
        manifest_sha256="a" * 64,
        source_head="b" * 40,
        committed_payload_sha256={},
    )
    monkeypatch.setattr(
        pod,
        "_prepare_stage_bootstrap",
        lambda *_args, **_kwargs: bootstrap,
    )
    monkeypatch.setattr(pod, "_verify_controller_source_matches_stage", lambda _bootstrap: None)
    monkeypatch.setattr(
        pod, "_preflight_results_repository",
        lambda *_args, **_kwargs: {"schema_version": 1, "remote_path": "canary", "size": 1, "sha256": "c" * 64},
    )
    provider_deadline = pod._provider_termination_deadline(pod.time.time(), 12_600)
    monkeypatch.setattr(pod, "_create_lease", lambda _args: ({
        "run_id": "r1", "artifact_run_id": "r1",
        "ssh_ip": "127.0.0.1", "ssh_port": 2222,
        "monitor_unit": None,
        "launch_nonce": args.launch_nonce,
        "stage_manifest_sha256": "a" * 64,
        "stage_source_head": "b" * 40,
        "provider_terminate_after": provider_deadline,
        "worker_deadline_epoch": pod._worker_deadline_epoch(
            provider_deadline, now=pod.time.time()
        ),
    }, supervisor))
    commands = []
    def run(command, check=False, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 17 if "bash" in command else 0)

    monkeypatch.setattr(pod.subprocess, "run", run)
    class Publisher:
        def read_file(self, _path):
            from ouro_jlens.publish import MissingRemote
            raise MissingRemote("index")

    monkeypatch.setattr(pod, "HfPublisher", lambda repo, **_kwargs: Publisher())
    monkeypatch.setattr(pod, "sync_and_verify", lambda *a, **k: [])
    monkeypatch.setattr(pod.time, "sleep", lambda _seconds: None)
    assert pod.run_job(args) == 17
    assert supervisor.terminated == 1
    assert [command[0] for command in commands] == ["ssh", "scp", "ssh"]
    assert any(item.startswith("STAGE_PATH=stages/") for item in commands[2])
    assert any(item == "STAGING=Vykos/ouro-jlens-staging" for item in commands[2])
    assert any(item == "JLENS_ATTEMPT_ID=r1" for item in commands[2])
    assert any(
        item.startswith("JLENS_HF_TOKEN_BOOTSTRAP=hf_")
        for item in commands[2]
    )
    assert any(item == f"JLENS_IMAGE_DIGEST={publish_module.RUNTIME_IMAGE}" for item in commands[2])


def test_run_b300_exit_recovers_hash_valid_checkpoint_pair(tmp_path):
    repo = tmp_path / "repo"
    wrapper = repo / "src/ouro_jlens/run_b300.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(Path("src/ouro_jlens/run_b300.sh").read_text())
    fake_log = tmp_path / "fake-python.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import hashlib
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
log = Path(os.environ["FAKE_LOG"])
with log.open("a") as handle:
    handle.write(" ".join(args) + "\\n")
if "-m" in args and "restore-file" in args:
    print("ABSENT")
    raise SystemExit(0)
if any(value.endswith("fit_lens.py") for value in args) and "fit" in args:
    out = Path(args[args.index("--out") + 1])
    checkpoint = Path(str(out) + ".ckpt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = b"sealed checkpoint"
    checkpoint.write_bytes(payload)
    metadata = {
        "kind": "fit_checkpoint",
        "checkpoint": {
            "path": str(checkpoint),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    Path(str(checkpoint) + ".json").write_text(json.dumps(metadata))
    raise SystemExit(17)
if args and args[0] == "-" and len(args) >= 2 and "/recovery/" in args[1]:
    path = Path(args[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"complete": True}))
raise SystemExit(0)
"""
    )
    fake_python.chmod(0o755)
    hf_token_file = tmp_path / "hf-token"
    hf_token_file.write_text("hf_test_token", encoding="utf-8")
    hf_token_file.chmod(0o600)
    environment = os.environ.copy()
    environment.update({
        "PYTHON": str(fake_python),
        "FAKE_LOG": str(fake_log),
        "RUN_ID": "r1",
        "JLENS_ATTEMPT_ID": "attempt-1",
        "JLENS_LAUNCH_NONCE": "n" * 48,
        "EXPECTED_STAGE_MANIFEST_SHA256": "a" * 64,
        "EXPECTED_STAGE_SOURCE_HEAD": "b" * 40,
        "JLENS_WORKER_DEADLINE_EPOCH": str(int(pod.time.time()) + 1800),
        "JLENS_HF_TOKEN_FILE": str(hf_token_file),
        "RESULTS": "org/results",
        "SHARD_SIZE": "25",
        "JLENS_IMAGE_DIGEST": publish_module.RUNTIME_IMAGE,
    })
    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 17, result.stderr
    calls = fake_log.read_text()
    assert "--kind checkpoint" in calls
    assert "--kind checkpoint_sidecar" in calls
    assert "--kind recovery_status" in calls
    assert "recovery/attempt-1.json" in calls
    assert ".pt.ckpt" in calls and ".pt.ckpt.json" in calls


def test_run_local_only_skips_verified_lens_and_removes_checkpoint_pair(tmp_path):
    repo = tmp_path / "repo"
    wrapper = repo / "src/ouro_jlens/run_local.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(Path("src/ouro_jlens/run_local.sh").read_text())
    fake_log = tmp_path / "fake-python.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
log = Path(os.environ["FAKE_LOG"])
with log.open("a") as handle:
    handle.write(" ".join(args) + "\\n")
if any(value.endswith("fit_lens.py") for value in args) and "fit" in args:
    out = Path(args[args.index("--out") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"lens")
    out.with_suffix(".json").write_text(json.dumps({"kind": "fit"}))
    Path(str(out) + ".ckpt").write_bytes(b"checkpoint")
    Path(str(out) + ".ckpt.json").write_text(json.dumps({"kind": "fit_checkpoint"}))
raise SystemExit(0)
"""
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update({"PYTHON": str(fake_python), "FAKE_LOG": str(fake_log)})
    first = subprocess.run(["bash", str(wrapper)], cwd=repo, env=environment,
                           text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    outputs = sorted((repo / "artifacts/jlens/lens").glob("exit*/*.pt"))
    assert len(outputs) == 8
    assert all(path.with_suffix(".json").is_file() for path in outputs)
    assert not list((repo / "artifacts/jlens/lens").rglob("*.ckpt"))
    fit_calls = [line for line in fake_log.read_text().splitlines()
                 if "fit_lens.py fit" in line]
    assert len(fit_calls) == 8
    second = subprocess.run(["bash", str(wrapper)], cwd=repo, env=environment,
                            text=True, capture_output=True, check=False)
    assert second.returncode == 0, second.stderr
    all_calls = fake_log.read_text().splitlines()
    assert len([line for line in all_calls if "fit_lens.py fit" in line]) == 8
    assert len([line for line in all_calls if "validate_sidecar" in line]) >= 8


def test_local_wrappers_require_sidecars_and_current_probe_cv_contract():
    run_local = Path("src/ouro_jlens/run_local.sh").read_text()
    run_local2 = Path("src/ouro_jlens/run_local2.sh").read_text()
    fitsize = Path("src/ouro_jlens/fitsize.sh").read_text()
    eval_round = Path("src/ouro_jlens/eval_round.sh").read_text()
    for script in (run_local, run_local2):
        assert "validate_sidecar" in script
        assert ".ckpt.json" in script
    for script in (fitsize, eval_round):
        assert "validate_sidecar" in script
    assert "probe_cv.py" in eval_round
    assert "src/ouro_jlens/probe.py" in eval_round  # provenance binds the current source
    assert "python src/ouro_jlens/probe.py" not in eval_round
    assert "--lens-only" in eval_round
    assert "--out \"$CONVERGENCE_OUT\"" in fitsize
    assert "summary.json" in fitsize and "arrays.npz" in fitsize and "provenance.json" in fitsize


def test_eval_round_resume_binds_cache_scores_cpu_outputs_and_source_model():
    script = Path("src/ouro_jlens/eval_round.sh").read_text()
    for marker in (
        "CACHE_PROVENANCE",
        "FRESH_CURRENT_SOURCE_AND_MODEL",
        "model_files",
        "source_files",
        "lens-score hidden-state input",
        "lens-score lens input",
        "lens_score_provenance",
        "probe CPU summary arrays input",
        "sha256",
    ):
        assert marker in script
    assert "verify_probe_metadata cache" in script
    assert "verify_probe_metadata gpu" in script
    assert "verify_probe_metadata cpu" in script
    assert 'if [[ -L "$out" ]]' in script
    assert 'if [[ -L "$PROBE" ]]' in script


def test_eval_round_embedded_helper_rejects_cache_hash_tampering(tmp_path):
    script = Path("src/ouro_jlens/eval_round.sh").read_text()
    start = script.index("verify_probe_metadata() {")
    start = script.index("<<'PY'", start) + len("<<'PY'\n")
    end = script.index("\nPY\n", start)
    cache = tmp_path / "gpu_cache.npz"
    cache.write_bytes(b"cache bytes")
    provenance = cache.with_suffix(".provenance.json")
    provenance.write_text(json.dumps({
        "schema_version": 1,
        "status": "FRESH_CURRENT_SOURCE_AND_MODEL",
        "design": {"hidden_width": 2048, "prompts": 648, "virtual_locations": 192},
        "output": {"path": str(cache), "size": cache.stat().st_size, "sha256": "0" * 64},
    }))
    result = subprocess.run(
        [sys.executable, "-", "cache", str(cache), str(provenance)],
        input=script[start:end], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"}, check=False,
    )
    assert result.returncode == 1
    assert "hidden-state cache digest mismatch" in result.stderr


@pytest.mark.parametrize("root_name", ["eval", "probe"])
def test_eval_round_rejects_symlinked_output_root(tmp_path, root_name):
    repo = tmp_path / "repo"
    wrapper = repo / "src/ouro_jlens/eval_round.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(Path("src/ouro_jlens/eval_round.sh").read_text())
    roots = repo / "artifacts/jlens"
    roots.mkdir(parents=True)
    target = tmp_path / f"{root_name}-target"
    target.mkdir()
    (roots / root_name).symlink_to(target, target_is_directory=True)
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["PYTHON"] = str(fake_python)
    result = subprocess.run(["bash", str(wrapper), "symlink-check"], cwd=repo,
                            env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "output root is a symlink" in result.stderr
