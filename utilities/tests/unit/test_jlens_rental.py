"""Offline contract tests for the JLens rental and evidence protocol."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from ouro_jlens import pod, publish as publish_module
from ouro_jlens.publish import (
    ImmutableConflict,
    LocalPublisher,
    StageVerificationError,
    build_stage_tar,
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
        "pod_name": name,
        "pod_id": pod_id,
        "machine_id": "m1",
        "cost_per_hr": cost,
        "created_at": bound_at,
        "bound_at": bound_at,
        "status": "active",
    }))


def test_negative_balance_prevents_create_before_monitor_or_mutation(tmp_path, monkeypatch):
    args = pod._parser().parse_args([
        "create", "--run-id", "r1", "--state-root", str(tmp_path), "--min-balance", "20",
        "--pubkey", str(tmp_path / "pub"), "--hf-token", str(tmp_path / "hf"),
    ])
    client = FakeClient(balance=-0.54)
    monitor_started = []
    deploy_called = []
    client.deploy = lambda **kwargs: deploy_called.append(kwargs)
    with pytest.raises(pod.SafetyError, match="below"):
        pod._create_lease(client=client, monitor_launcher=lambda p: monitor_started.append(p), args=args)
    assert not monitor_started and not deploy_called
    assert not list(tmp_path.rglob("state.json"))


def test_negative_balance_is_blocked_even_with_zero_configured_floor(tmp_path):
    args = pod._parser().parse_args([
        "create", "--run-id", "r1", "--state-root", str(tmp_path), "--min-balance", "0",
        "--pubkey", str(tmp_path / "pub"), "--hf-token", str(tmp_path / "hf"),
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
    assert pod.api_key(key_file) == valid_key
    key_file.write_text("rpa_" + "B" * 24 + " rpa_" + "C" * 24)
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


def test_setup_script_executes_pip_through_selected_python_and_checks_override():
    script = Path("src/ouro_jlens/setup_b300.sh").read_text()
    assert '"$PY" -m pip install -q' in script
    assert '"$PY" -m pip install -q -e "$JLENS_DIR" --no-deps' in script
    assert '"$PY" -m ouro_jlens.publish runtime-verify' in script
    assert "transformers>=5.5" in script and "transformers==4.54.1" in script


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
    assert client.terminate_calls == ["p1"]
    assert json.loads(state_file.read_text())["termination_verified"] is True


def test_runtime_and_budget_breach_are_independent_fail_closed(tmp_path):
    for limit, expected in [("max_runtime", "runtime"), ("max_spend", "spend")]:
        state_file = tmp_path / f"{limit}.json"
        _state(state_file, bound_at=0, cost=10)
        client = FakeClient(balance=100, pods=[_pod(cost=10)])
        config = pod.SafetyConfig(min_balance=0, max_runtime=5 if limit == "max_runtime" else 1000,
                                  max_spend=0.01 if limit == "max_spend" else 1000,
                                  poll_seconds=1, termination_poll_seconds=0)
        supervisor = pod.LeaseSupervisor(client, state_file, config, sleep=lambda _: None,
                                         clock=lambda: 10, monotonic=lambda: 10)
        with pytest.raises(pod.SafetyError, match=expected):
            supervisor.monitor(once=True)
        assert client.terminate_calls == ["p1"]


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
        pod.SafetyConfig(min_balance=0, max_api_failures=2, poll_seconds=0.01,
                         termination_poll_seconds=0),
        sleep=lambda _: None,
    )
    with pytest.raises(pod.APIUncertain):
        supervisor.monitor()
    assert client.terminate_calls == ["p1"]


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
        pod.SafetyConfig(min_balance=0, poll_seconds=0.01, termination_poll_seconds=0),
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
                                     pod.SafetyConfig(min_balance=0, termination_attempts=3,
                                                      termination_poll_seconds=0))
    supervisor.terminate_verified()
    assert client.terminate_calls == ["p1", "p1"]


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


def test_publish_rejects_remote_payload_that_does_not_match_receipt(tmp_path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"important bytes")

    class CorruptPublisher(LocalPublisher):
        def put_file(self, path, remote_path):
            super().put_file(path, remote_path)
            if "/artifacts/" in remote_path:
                self._path(remote_path).write_bytes(b"corrupted")

    with pytest.raises(ImmutableConflict, match="remote payload"):
        publish_file(CorruptPublisher(tmp_path / "remote"), source, run_id="r1",
                     relative_path="lens/shard.pt", kind="shard")


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


def test_stage_archive_contains_manifest_and_rejects_tampering(tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    (root / "artifacts/jlens/data/wikitext_prompts.json").write_text("[]\n")
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    archive = tmp_path / "stage.tar.gz"
    result = build_stage_tar(root, archive)
    assert result["payload_count"] == 3
    verified = verify_stage_tar(archive)
    assert verified["manifest_sha256"] == result["manifest_sha256"]
    with tarfile.open(archive, "r:*") as tar:
        names = tar.getnames()
    assert "MANIFEST.sha256" in names and "PINNED_INPUTS.json" in names


def test_paid_stage_requires_clean_committed_payload(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/example.py").write_text("x = 1\n")
    (root / "artifacts/jlens/data/wikitext_prompts.json").write_text("[]\n")
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
    (root / "artifacts/jlens/data/wikitext_prompts.json").write_text("[]\n")
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    (root / "utilities/tests/unit/test_jlens_integrity.py").write_text("def test_integrity(): pass\n")
    (root / "docs/jlens/README.md").write_text("# Contract\n")
    archive = tmp_path / "stage.tar.gz"
    build_stage_tar(root, archive)
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert "docs/jlens/README.md" in names
    assert "utilities/tests/unit/test_jlens_integrity.py" in names


def test_controller_bootstrap_comes_from_verified_clean_stage(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/publish.py").write_text("PIN = 'stage'\n")
    (root / "src/ouro_jlens/pod_entry.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (root / "artifacts/jlens/data/wikitext_prompts.json").write_text("[]\n")
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    monkeypatch.setattr(
        publish_module,
        "_git_pin",
        lambda _root, _payloads: {"head": "a" * 40, "status": ""},
    )
    archive = tmp_path / "stage.tar.gz"
    built = build_stage_tar(root, archive, require_clean_source=True)
    stage_path = f"stages/{built['manifest_sha256']}/stage.tar.gz"
    remote = LocalPublisher(tmp_path / "remote")
    remote.put_file(archive, stage_path)
    package = pod._prepare_stage_bootstrap(
        stage_path, "org/staging", tmp_path / "bootstrap", publisher=remote
    )
    assert (package / "publish.py").read_text() == "PIN = 'stage'\n"
    assert (package / "pod_entry.sh").is_file()


def test_controller_rejects_stage_whose_path_names_another_manifest(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "src/ouro_jlens").mkdir(parents=True)
    (root / "artifacts/jlens/data").mkdir(parents=True)
    (root / "utilities/tests/unit").mkdir(parents=True)
    (root / "src/ouro_jlens/publish.py").write_text("PIN = 'stage'\n")
    (root / "src/ouro_jlens/pod_entry.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (root / "artifacts/jlens/data/wikitext_prompts.json").write_text("[]\n")
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    monkeypatch.setattr(
        publish_module,
        "_git_pin",
        lambda _root, _payloads: {"head": "a" * 40, "status": ""},
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
    (root / "artifacts/jlens/data/wikitext_prompts.json").write_text("[]\n")
    (root / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    archive = tmp_path / "stage.tar.gz"
    build_stage_tar(root, archive)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(extracted)
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
    (root / "artifacts/jlens/data/wikitext_prompts.json").write_text("[]\n")
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
        "run", "--run-id", "r1", "--results", "org/results", "--stage-path",
        "stages/" + "a" * 64 + "/stage.tar.gz", "--state-root", str(tmp_path),
        "--sync-root", str(tmp_path / "sync"), "--receipt-root", str(tmp_path / "receipts"),
    ])

    class Supervisor:
        def __init__(self):
            self.terminated = 0

        def terminate_verified(self):
            self.terminated += 1

    supervisor = Supervisor()
    monkeypatch.setattr(
        pod,
        "_prepare_stage_bootstrap",
        lambda *_args, **_kwargs: Path("src/ouro_jlens"),
    )
    monkeypatch.setattr(pod, "_create_lease", lambda _args: ({
        "run_id": "r1", "ssh_ip": "127.0.0.1", "ssh_port": 2222,
        "monitor_unit": None,
    }, supervisor))
    commands = []
    def run(command, check=False):
        commands.append(command)
        return subprocess.CompletedProcess(command, 17 if "bash" in command else 0)

    monkeypatch.setattr(pod.subprocess, "run", run)
    class Publisher:
        def read_file(self, _path):
            from ouro_jlens.publish import MissingRemote
            raise MissingRemote("index")

    monkeypatch.setattr(pod, "HfPublisher", lambda repo: Publisher())
    monkeypatch.setattr(pod, "sync_and_verify", lambda *a, **k: [])
    assert pod.run_job(args) == 17
    assert supervisor.terminated == 1
    assert [command[0] for command in commands] == ["ssh", "scp", "ssh"]
    assert any(item.startswith("STAGE_PATH=stages/") for item in commands[2])
    assert any(item == "STAGING=Vykos/ouro-jlens-staging" for item in commands[2])


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
raise SystemExit(0)
"""
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update({
        "PYTHON": str(fake_python),
        "FAKE_LOG": str(fake_log),
        "RUN_ID": "r1",
        "RESULTS": "org/results",
        "SHARD_SIZE": "25",
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
