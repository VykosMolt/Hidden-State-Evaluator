"""Focused offline checks for the paid JLens shell boundary.

These tests intentionally exercise only the shell-owned contract.  They do
not contact RunPod/Hugging Face, download a stage, or require a CUDA device.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ouro_jlens.publish import RUNTIME_IMAGE


ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "src/ouro_jlens/pod_entry.sh"
RUN = ROOT / "src/ouro_jlens/run_b300.sh"


def _fake_command(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(f"#!/usr/bin/env bash\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _entry_environment(tmp_path: Path, *, gpu_names: str = "NVIDIA B300 SXM6 AC") -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gpu_marker = tmp_path / "nvidia-smi.called"
    hf_marker = tmp_path / "stage-download.called"
    _fake_command(
        bin_dir,
        "nvidia-smi",
        f"printf '%b\\n' {gpu_names!r}; touch {gpu_marker!s}",
    )
    bootstrap_python = _fake_command(
        bin_dir,
        "bootstrap-python",
        "if [[ ${1:-} == -m && ${2:-} == ouro_jlens.publish "
        "&& ${3:-} == stage-download ]]; then\n"
        f"  touch {hf_marker!s}\n"
        "  exit 42\n"
        "fi\n"
        f"exec {sys.executable!r} \"$@\"",
    )
    manifest = "a" * 64
    env = {
        **os.environ,
        "PYTHON": str(bootstrap_python),
        "PYTHONPATH": str(ROOT / "src"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "RESULTS": "org/results",
        "STAGING": "org/staging",
        "STAGE_PATH": f"stages/{manifest}/stage.tar.gz",
        "RUN_ID": "run-1",
        "JLENS_ATTEMPT_ID": "attempt-1",
        "JLENS_HF_TOKEN_BOOTSTRAP": "hf_test_token",
        "WORK": str(tmp_path / "work"),
        "JLENS_IMAGE_DIGEST": RUNTIME_IMAGE,
        "JLENS_LAUNCH_NONCE": "N" * 32,
        "EXPECTED_STAGE_MANIFEST_SHA256": manifest,
        "EXPECTED_STAGE_SOURCE_HEAD": "b" * 40,
        "JLENS_WORKER_DEADLINE_EPOCH": str(int(time.time()) + 3600),
    }
    return env, gpu_marker, hf_marker


def test_shell_scripts_are_parseable_and_bind_every_launch_fact() -> None:
    for script in (ENTRY, RUN):
        result = subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=False)
        assert result.returncode == 0

    entry = ENTRY.read_text(encoding="utf-8")
    run = RUN.read_text(encoding="utf-8")
    assert "export PYTHONDONTWRITEBYTECODE=1" in entry
    assert entry.index("export PYTHONDONTWRITEBYTECODE=1") < entry.index('"$PY"')
    for name in (
        "JLENS_LAUNCH_NONCE",
        "EXPECTED_STAGE_MANIFEST_SHA256",
        "EXPECTED_STAGE_SOURCE_HEAD",
        "JLENS_WORKER_DEADLINE_EPOCH",
    ):
        assert f"{name}=${{{name}:?" in entry
        assert f"{name}=${{{name}:?" in run
        assert name in entry.split("export", 1)[1]
        assert name in run.split("export", 1)[1]
    assert "stage_path_digest\" != \"$EXPECTED_STAGE_MANIFEST_SHA256" in entry
    assert "STAGE_MANIFEST_SHA256\" != \"$EXPECTED_STAGE_MANIFEST_SHA256" in entry
    assert "stage_source_head\" != \"$EXPECTED_STAGE_SOURCE_HEAD" in entry
    assert "nvidia-smi" in entry and "--query-gpu=name" in entry
    stage_download = "if run_stage_download_with_token; then"
    assert entry.index("--query-gpu=name") < entry.index(stage_download)
    assert "timeout --kill-after=2s 30s nvidia-smi" in entry
    assert "timeout --signal=TERM --kill-after=\"${JLENS_TIMEOUT_GRACE_SECONDS}s\"" in entry
    assert 'env HF_TOKEN="$(<' not in entry
    assert "hf download" not in entry
    assert "ouro_jlens.publish stage-download" in entry
    assert '--token-file "$JLENS_HF_TOKEN_FILE"' in entry
    assert entry.index(stage_download) < entry.index("run_with_deadline bash src/ouro_jlens/setup_b300.sh")
    assert "deadline_remaining" in entry
    assert "GNU coreutils" in entry
    assert "os.statvfs" in entry
    assert "80 * 1024 * 1024 * 1024" in entry
    assert entry.rindex("if ! check_free_space") < entry.index("run_with_deadline bash src/ouro_jlens/setup_b300.sh")

    publish_shard = run[run.index("publish_shard() {") : run.index("\n}\n\nverify_eval_receipts", run.index("publish_shard() {"))]
    assert "discard_completed_checkpoint_pair" in publish_shard
    assert 'publish_file "$out.ckpt"' not in publish_shard
    recovery = run[run.index("publish_interrupted_checkpoints() {") : run.index("\non_exit()", run.index("publish_interrupted_checkpoints() {"))]
    assert 'publish_file "$checkpoint"' in recovery
    assert 'publish_file "$sidecar"' in recovery
    on_exit = run.index("on_exit() {")
    assert on_exit < run.index('exec >>"$exit_log" 2>&1') < run.index("sync_log", on_exit)
    assert "main computation log ends at this cutover" in run
    assert run.count("--checkpoints ''") == 2
    for field in ("attempt_id", "launch_nonce", "stage_manifest_sha256", "stage_source_head", "worker_deadline_epoch"):
        assert f'"{field}"' in run


@pytest.mark.parametrize("deadline", [None, "not-a-number", str(int(time.time()) - 1)])
def test_entry_rejects_missing_malformed_or_past_deadline_before_external_work(
    tmp_path: Path, deadline: str | None
) -> None:
    env, gpu_marker, hf_marker = _entry_environment(tmp_path)
    if deadline is None:
        env.pop("JLENS_WORKER_DEADLINE_EPOCH", None)
    else:
        env["JLENS_WORKER_DEADLINE_EPOCH"] = deadline
    result = subprocess.run(
        ["bash", str(ENTRY)], cwd=ROOT, env=env, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert not gpu_marker.exists()
    assert not hf_marker.exists()
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("JLENS_LAUNCH_NONCE", "unsafe/nonce"),
        ("EXPECTED_STAGE_MANIFEST_SHA256", "A" * 64),
        ("EXPECTED_STAGE_SOURCE_HEAD", "c" * 41),
    ],
)
def test_entry_rejects_invalid_launch_binding_before_gpu_or_hf(
    tmp_path: Path, field: str, value: str
) -> None:
    env, gpu_marker, hf_marker = _entry_environment(tmp_path)
    env[field] = value
    result = subprocess.run(
        ["bash", str(ENTRY)], cwd=ROOT, env=env, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert not gpu_marker.exists()
    assert not hf_marker.exists()


@pytest.mark.parametrize(
    "gpu_names",
    ["NVIDIA H100 80GB HBM3", "NVIDIA B300 SXM6 AC\nNVIDIA A100-SXM4-80GB", "NVIDIA B3000 SXM6 AC"],
)
def test_entry_rejects_non_exactly_one_word_boundary_b300_before_stage_download(
    tmp_path: Path, gpu_names: str
) -> None:
    env, gpu_marker, hf_marker = _entry_environment(tmp_path, gpu_names=gpu_names)
    result = subprocess.run(
        ["bash", str(ENTRY)], cwd=ROOT, env=env, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert gpu_marker.exists()
    assert not hf_marker.exists()


def test_entry_allows_one_word_boundary_b300_and_reaches_stage_download(tmp_path: Path) -> None:
    env, gpu_marker, hf_marker = _entry_environment(tmp_path)
    result = subprocess.run(
        ["bash", str(ENTRY)], cwd=ROOT, env=env, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode == 42
    assert gpu_marker.exists()
    assert hf_marker.exists()


def test_completed_checkpoint_pair_is_removed_without_upload(tmp_path: Path) -> None:
    script = RUN.read_text(encoding="utf-8")
    start = script.index("discard_completed_checkpoint_pair() {")
    end = script.index("\n}\n\nquarantine_orphan", start) + 2
    function = script[start:end]
    output = tmp_path / "exit3_shard_0000_0008.pt"
    output.write_bytes(b"lens")
    checkpoint = Path(f"{output}.ckpt")
    sidecar = Path(f"{checkpoint}.json")
    checkpoint.write_bytes(b"checkpoint")
    sidecar.write_text("sealed\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", "-c", function + "\ndiscard_completed_checkpoint_pair \"$1\"", "test", str(output)],
        check=False,
    )
    assert result.returncode == 0
    assert output.exists()
    assert not checkpoint.exists()
    assert not sidecar.exists()


def test_unpaired_completed_checkpoint_is_not_silently_deleted(tmp_path: Path) -> None:
    script = RUN.read_text(encoding="utf-8")
    start = script.index("discard_completed_checkpoint_pair() {")
    end = script.index("\n}\n\nquarantine_orphan", start) + 2
    function = script[start:end]
    output = tmp_path / "exit3_shard_0000_0008.pt"
    output.write_bytes(b"lens")
    checkpoint = Path(f"{output}.ckpt")
    checkpoint.write_bytes(b"checkpoint")
    result = subprocess.run(
        ["bash", "-c", function + "\ndiscard_completed_checkpoint_pair \"$1\"", "test", str(output)],
        check=False,
    )
    assert result.returncode == 2
    assert checkpoint.exists()


def test_free_space_preflight_fails_closed_without_setup_or_external_work(tmp_path: Path) -> None:
    script = ENTRY.read_text(encoding="utf-8")
    start = script.index("check_free_space() {")
    end = script.index("\n}\n\n# Query the physical lease", start) + 2
    function = script[start:end]
    result = subprocess.run(
        ["bash", "-c", function + "\ncheck_free_space"],
        cwd=ROOT,
        env={
            **os.environ,
            "PY": sys.executable,
            "WORK": str(tmp_path),
            "JLENS_MIN_FREE_BYTES": str(10**30),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "insufficient free space" in result.stderr
