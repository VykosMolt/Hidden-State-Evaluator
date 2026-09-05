"""Offline tests for the protected JLens Hugging Face credential channel."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ouro_jlens import publish
from ouro_jlens.publish import HfPublisher, PublishError, read_hf_token_file


ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "src/ouro_jlens/pod_entry.sh"
SETUP = ROOT / "src/ouro_jlens/setup_b300.sh"
STAGE_UPLOAD = ROOT / "src/ouro_jlens/stage_upload.sh"
TOKEN = "hf_test_token_123"
BOOTSTRAP = "hf_bootstrap_secret_456"
MANIFEST = "a" * 64


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _write_token(path: Path, token: str = TOKEN, mode: int = 0o600) -> Path:
    path.write_text(token, encoding="ascii")
    path.chmod(mode)
    return path


def test_hf_token_file_requires_exact_private_regular_file(tmp_path: Path) -> None:
    token_path = _write_token(tmp_path / "token")
    assert read_hf_token_file(token_path) == TOKEN
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600

    token_path.chmod(0o640)
    with pytest.raises(PublishError, match="0600"):
        read_hf_token_file(token_path)

    token_path.chmod(0o600)
    token_path.write_text(f"{TOKEN}\n", encoding="ascii")
    with pytest.raises(PublishError, match="malformed"):
        read_hf_token_file(token_path)

    target = tmp_path / "real-token"
    _write_token(target)
    link = tmp_path / "linked-token"
    link.symlink_to(target)
    with pytest.raises(PublishError):
        read_hf_token_file(link)


def test_stdlib_stage_download_needs_no_hf_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_path = _write_token(tmp_path / "token")
    payload = b"verified-stage-bytes"
    captured: dict[str, object] = {}

    class Response:
        status = 200
        headers = {"Content-Length": str(len(payload))}

        def __init__(self) -> None:
            self._sent = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size: int) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return payload

    def urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(publish.urllib.request, "urlopen", urlopen)
    destination = tmp_path / "nested" / "stage.tar.gz"
    result = publish.download_hf_file_stdlib(
        "org/staging",
        f"stages/{MANIFEST}/stage.tar.gz",
        destination,
        token_file=token_path,
    )
    assert destination.read_bytes() == payload
    assert result == {
        "path": str(destination),
        "size": len(payload),
        "sha256": publish.sha256_bytes(payload),
    }
    request = captured["request"]
    assert request.full_url == (
        f"https://huggingface.co/org/staging/resolve/main/stages/{MANIFEST}/"
        "stage.tar.gz?download=true"
    )
    assert request.get_header("Authorization") == f"Bearer {TOKEN}"
    assert captured["timeout"] == 60


def test_publisher_child_alone_receives_token_and_parent_env_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setenv("HF_TOKEN", "ambient-parent-secret")
    monkeypatch.setenv("JLENS_HF_TOKEN_BOOTSTRAP", BOOTSTRAP)

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    publisher = HfPublisher("org/results", token=TOKEN, runner=runner)
    publisher._run(["hf", "whoami"])

    assert len(calls) == 1
    command, kwargs = calls[0]
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["HF_TOKEN"] == TOKEN
    assert "JLENS_HF_TOKEN_BOOTSTRAP" not in environment
    assert TOKEN not in command
    assert BOOTSTRAP not in command
    assert os.environ["HF_TOKEN"] == "ambient-parent-secret"
    assert os.environ["JLENS_HF_TOKEN_BOOTSTRAP"] == BOOTSTRAP


def test_publisher_scrubs_ambient_credentials_without_explicit_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("HF_TOKEN", "ambient-parent-secret")
    monkeypatch.setenv("JLENS_HF_TOKEN_BOOTSTRAP", BOOTSTRAP)
    HfPublisher("org/results", runner=runner)._run(["hf", "download", "public"])
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert "HF_TOKEN" not in environment
    assert "JLENS_HF_TOKEN_BOOTSTRAP" not in environment


def test_publisher_does_not_retain_runner_secret_diagnostics() -> None:
    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise RuntimeError(f"unexpected credential {TOKEN}")

    with pytest.raises(PublishError) as error:
        HfPublisher("org/results", token=TOKEN, runner=runner)._run(["hf", "whoami"])
    assert TOKEN not in str(error.value)


def test_entry_scopes_bootstrap_to_stage_download_and_deletes_file_on_exit(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gpu_marker = tmp_path / "gpu-called"
    hf_ok = tmp_path / "hf-token-ok"
    argv_leak = tmp_path / "token-in-process-arguments"
    bootstrap_absent = tmp_path / "bootstrap-absent"
    token_path_marker = tmp_path / "token-path"
    _write_executable(
        bin_dir / "nvidia-smi",
        f"printf '%s\\n' 'NVIDIA B300 SXM6 AC'; touch {gpu_marker!s}",
    )
    bootstrap_python = bin_dir / "bootstrap-python"
    _write_executable(
        bootstrap_python,
        "if [[ ${1:-} == -m && ${2:-} == ouro_jlens.publish "
        "&& ${3:-} == stage-download ]]; then\n"
        "  token_file=''\n"
        "  while (($#)); do\n"
        "    if [[ $1 == --token-file ]]; then token_file=$2; shift 2; else shift; fi\n"
        "  done\n"
        f"  if [[ $(<\"$token_file\") == {TOKEN!r} ]]; then touch {hf_ok!s}; fi\n"
        "  { tr '\\\\0' '\\\\n' < \"/proc/$$/cmdline\"; "
        "tr '\\\\0' '\\\\n' < \"/proc/$PPID/cmdline\"; } "
        f"| grep -Fq {TOKEN!r} && touch {argv_leak!s} || true\n"
        f"  if [[ -z \"${{JLENS_HF_TOKEN_BOOTSTRAP+x}}\" ]]; then touch {bootstrap_absent!s}; fi\n"
        f"  printf '%s' \"$token_file\" > {token_path_marker!s}\n"
        "  exit 42\n"
        "fi\n"
        f"exec {sys.executable!r} \"$@\"",
    )
    environment = {
        **os.environ,
        "PYTHON": str(bootstrap_python),
        "PYTHONPATH": str(ROOT / "src"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "RESULTS": "org/results",
        "STAGING": "org/staging",
        "STAGE_PATH": f"stages/{MANIFEST}/stage.tar.gz",
        "RUN_ID": "run-1",
        "JLENS_ATTEMPT_ID": "attempt-1",
        "JLENS_HF_TOKEN_BOOTSTRAP": TOKEN,
        "HF_TOKEN": "ambient-parent-secret",
        "WORK": str(tmp_path / "work"),
        "JLENS_IMAGE_DIGEST": (
            "runpod/pytorch@sha256:bbe1496e2215cca3d25a5e5cd291d31ea86603e4577a81eb40096787a50e5303"
        ),
        "JLENS_LAUNCH_NONCE": "N" * 32,
        "EXPECTED_STAGE_MANIFEST_SHA256": MANIFEST,
        "EXPECTED_STAGE_SOURCE_HEAD": "b" * 40,
        "JLENS_WORKER_DEADLINE_EPOCH": "4102444800",
    }
    result = subprocess.run(
        ["bash", str(ENTRY)], cwd=ROOT, env=environment,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 42
    assert gpu_marker.exists()
    assert hf_ok.exists()
    assert not argv_leak.exists()
    assert bootstrap_absent.exists()
    assert token_path_marker.exists()
    assert not Path(token_path_marker.read_text(encoding="ascii")).exists()
    assert TOKEN not in result.stdout
    assert TOKEN not in result.stderr


def test_entry_installs_cleanup_before_post_creation_token_checks(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    token_tmp = tmp_path / "token-tmp"
    bin_dir.mkdir()
    token_tmp.mkdir()
    _write_executable(bin_dir / "stat", "printf '%s\\n' 400")
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "TMPDIR": str(token_tmp),
        "JLENS_HF_TOKEN_BOOTSTRAP": TOKEN,
    }
    result = subprocess.run(
        ["bash", str(ENTRY)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert not list(token_tmp.iterdir())
    assert TOKEN not in result.stdout and TOKEN not in result.stderr


def test_setup_hostile_children_never_receive_credential_variables(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src/ouro_jlens").mkdir(parents=True)
    (repo / "src/ouro_jlens/runtime.lock.json").write_text("{}\n", encoding="utf-8")
    (repo / "src/ouro_jlens/setup_b300.sh").write_bytes(SETUP.read_bytes())
    (repo / "src/ouro_jlens/setup_b300.sh").chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    leak_marker = tmp_path / "credential-leaked"
    python_fake = bin_dir / "python-fake"
    _write_executable(
        python_fake,
        f"if [[ -n \"${{HF_TOKEN:-}}\" || -n \"${{JLENS_HF_TOKEN_BOOTSTRAP:-}}\" ]]; then touch {leak_marker!s}; fi\n"
        "if [[ \"${1:-}\" == '-m' && \"${2:-}\" == 'ouro_jlens.publish' ]]; then\n"
        f"  if [[ \"${{3:-}}\" == 'stage-verify' ]]; then printf '%s\\n' '{{\"manifest_sha256\":\"{MANIFEST}\"}}'; fi\n"
        "fi\n"
        "exit 0",
    )
    git_fake = bin_dir / "git"
    _write_executable(
        git_fake,
        f"if [[ -n \"${{HF_TOKEN:-}}\" || -n \"${{JLENS_HF_TOKEN_BOOTSTRAP:-}}\" ]]; then touch {leak_marker!s}; fi\n"
        "if [[ \"${1:-}\" == 'clone' ]]; then mkdir -p \"${4}/.git\"; fi\n"
        "if [[ \"${1:-}\" == '-C' && \"${3:-}\" == 'rev-parse' ]]; then printf '%s\\n' '581d398613e5602a5af361e1c34d3a92ea82ba8e'; fi\n"
        "exit 0",
    )
    hf_fake = bin_dir / "hf"
    _write_executable(
        hf_fake,
        f"if [[ -n \"${{HF_TOKEN:-}}\" || -n \"${{JLENS_HF_TOKEN_BOOTSTRAP:-}}\" ]]; then touch {leak_marker!s}; fi\n"
        "cache=''\n"
        "while (($#)); do if [[ $1 == '--cache-dir' ]]; then cache=$2; shift 2; else shift; fi; done\n"
        "mkdir -p \"$cache/models--ByteDance--Ouro-2.6B/snapshots/1ed04250da1a9936042725d302e81c8fa2ab5abd\"\n"
        "touch \"$cache/models--ByteDance--Ouro-2.6B/snapshots/1ed04250da1a9936042725d302e81c8fa2ab5abd/model.safetensors\"",
    )
    environment = {
        **os.environ,
        "PYTHON": str(python_fake),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path / "home"),
        "JLENS_IMAGE_DIGEST": (
            "runpod/pytorch@sha256:bbe1496e2215cca3d25a5e5cd291d31ea86603e4577a81eb40096787a50e5303"
        ),
        "HF_TOKEN": TOKEN,
        "JLENS_HF_TOKEN_BOOTSTRAP": BOOTSTRAP,
    }
    result = subprocess.run(
        ["bash", str(repo / "src/ouro_jlens/setup_b300.sh")],
        cwd=repo, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not leak_marker.exists()


def _stage_upload_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "src/ouro_jlens").mkdir(parents=True)
    (repo / "src/ouro_jlens/stage_upload.sh").write_bytes(STAGE_UPLOAD.read_bytes())
    (repo / "src/ouro_jlens/stage_upload.sh").chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    publish_marker = tmp_path / "stage-publish-called"
    python_fake = bin_dir / "python-fake"
    _write_executable(
        python_fake,
        "if [[ \"${1:-}\" == '-m' && \"${2:-}\" == 'ouro_jlens.publish' ]]; then\n"
        f"  if [[ \"${{3:-}}\" == 'stage-create' ]]; then printf '%s\\n' '{{\"archive\":\"tmp.tar.gz\",\"manifest_sha256\":\"{MANIFEST}\"}}'; fi\n"
        f"  if [[ \"${{3:-}}\" == 'stage-publish' ]]; then touch {publish_marker!s}; fi\n"
        "elif [[ \"${1:-}\" == '-c' ]]; then\n"
        f"  printf '%s\\n' '{MANIFEST}'\n"
        "fi\n"
        "exit 0",
    )
    return repo, bin_dir, publish_marker


def test_stage_upload_dry_run_does_not_read_or_require_token_file(tmp_path: Path) -> None:
    repo, bin_dir, publish_marker = _stage_upload_fixture(tmp_path)
    environment = {
        **os.environ,
        "PYTHON": str(bin_dir / "python-fake"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "DRY_RUN": "1",
        "HF_TOKEN": TOKEN,
        "JLENS_HF_TOKEN_BOOTSTRAP": BOOTSTRAP,
    }
    result = subprocess.run(
        ["bash", str(repo / "src/ouro_jlens/stage_upload.sh")],
        cwd=repo, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not publish_marker.exists()
    assert "DRY RUN" in result.stdout


def test_stage_upload_requires_private_token_before_stage_publish(tmp_path: Path) -> None:
    repo, bin_dir, publish_marker = _stage_upload_fixture(tmp_path)
    environment = {
        **os.environ,
        "PYTHON": str(bin_dir / "python-fake"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "HF_TOKEN": TOKEN,
        "JLENS_HF_TOKEN_BOOTSTRAP": BOOTSTRAP,
    }
    result = subprocess.run(
        ["bash", str(repo / "src/ouro_jlens/stage_upload.sh")],
        cwd=repo, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert not publish_marker.exists()
    assert TOKEN not in result.stdout
    assert TOKEN not in result.stderr
