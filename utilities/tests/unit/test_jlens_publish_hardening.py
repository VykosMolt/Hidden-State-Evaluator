"""Focused regression tests for the immutable JLens publisher contract."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouro_jlens import publish as publish_module
from ouro_jlens.publish import (
    HfPublisher,
    ImmutableConflict,
    LocalPublisher,
    MissingRemote,
    PublishError,
    StageVerificationError,
    _confirmed_digest_fd,
    build_stage_tar,
    extract_stage_tar,
    publish_file,
    sync_and_verify,
    verify_stage_root,
    verify_stage_tar,
)


class _ParentConflict(Exception):
    response = SimpleNamespace(status_code=412)


class _ConflictApi:
    def __init__(self) -> None:
        self.head = "a" * 40
        self.commit_calls: list[dict[str, object]] = []

    def repo_info(self, **_kwargs):
        return SimpleNamespace(sha=self.head)

    def create_commit(self, **kwargs):
        operation = list(kwargs["operations"])[0]
        self.commit_calls.append(kwargs)
        # Consume the descriptor-anchored path as Xet does, but never mutate
        # the fake repository after the deliberate parent conflict.
        with open(operation.path_or_fileobj, "rb") as handle:
            handle.read()
        raise _ParentConflict()


def _publisher_with_target_schedule(api: _ConflictApi, schedule: list[bytes | None]):
    publisher = HfPublisher("org/results", api=api)
    reads = iter(schedule)

    def read_file(
        _remote_path: str,
        *,
        expected_size: int | None = None,
        revision: str | None = None,
    ) -> bytes:
        assert expected_size is not None
        if revision is not None:
            assert revision == api.head
        try:
            value = next(reads)
        except StopIteration:
            value = None
        if value is None:
            raise MissingRemote("target is not visible")
        return value

    publisher.read_file = read_file  # type: ignore[method-assign]
    return publisher


def test_parent_412_missing_target_never_submits_second_mutation(tmp_path, monkeypatch):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"immutable payload")
    api = _ConflictApi()
    publisher = _publisher_with_target_schedule(api, [None] * 20)
    monkeypatch.setattr(publish_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(PublishError, match="without a second mutation"):
        publisher.put_file(source, "r1/artifacts/payload.bin")
    assert len(api.commit_calls) == 1


def test_parent_412_reconciliation_is_read_only_and_classifies_target(tmp_path, monkeypatch):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"immutable payload")
    expected = source.read_bytes()
    monkeypatch.setattr(publish_module.time, "sleep", lambda _seconds: None)

    matching_api = _ConflictApi()
    matching = _publisher_with_target_schedule(matching_api, [None, None, expected])
    matching.put_file(source, "r1/receipts/payload.receipt.json")
    assert len(matching_api.commit_calls) == 1

    different_api = _ConflictApi()
    different = _publisher_with_target_schedule(different_api, [None, b"other"])
    with pytest.raises(ImmutableConflict):
        different.put_file(source, "stages/stage.tar.gz")
    assert len(different_api.commit_calls) == 1


def test_confirmed_digest_accepts_ctime_only_retry_but_not_digest_change(tmp_path, monkeypatch):
    path = tmp_path / "payload"
    path.write_bytes(b"stable")
    fd = os.open(path, os.O_RDONLY)
    try:
        real_digest = publish_module._digest_fd
        calls = 0

        def ctime_race(descriptor, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise PublishError("regular file changed while being hashed")
            return real_digest(descriptor, **kwargs)

        monkeypatch.setattr(publish_module, "_digest_fd", ctime_race)
        assert _confirmed_digest_fd(fd)[1] == hashlib.sha256(b"stable").hexdigest()
    finally:
        os.close(fd)

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    first_fd = os.open(first, os.O_RDONLY)
    second_fd = os.open(second, os.O_RDONLY)
    try:
        values = iter([(5, "a" * 64), (6, "b" * 64)])
        monkeypatch.setattr(
            publish_module,
            "_digest_fd",
            lambda _descriptor, **_kwargs: next(values),
        )
        with pytest.raises(PublishError, match="changed between"):
            _confirmed_digest_fd(first_fd, attempts=2, retry_delay=0)
    finally:
        os.close(first_fd)
        os.close(second_fd)


def test_sync_skips_download_for_matching_installed_receipt(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"already present")

    class CountingPublisher(LocalPublisher):
        def __init__(self, root):
            super().__init__(root)
            self.downloads = 0

        def _download_file_to_fd(self, remote_path, destination_fd, **kwargs):
            self.downloads += 1
            return super()._download_file_to_fd(remote_path, destination_fd, **kwargs)

    publisher = CountingPublisher(tmp_path / "remote")
    receipt = publish_file(
        publisher,
        source,
        run_id="r1",
        relative_path="payload.bin",
        kind="artifact",
    )
    local_root = tmp_path / "synced"
    destination = local_root / receipt.relative_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())
    publisher.downloads = 0

    assert sync_and_verify(
        publisher,
        run_id="r1",
        local_root=local_root,
        relative_paths=[receipt.relative_path],
    ) == [receipt]
    assert publisher.downloads == 0


def test_sync_resource_preflight_rejects_total_before_download(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload larger than one byte")

    class CountingPublisher(LocalPublisher):
        def __init__(self, root):
            super().__init__(root)
            self.downloads = 0

        def _download_file_to_fd(self, remote_path, destination_fd, **kwargs):
            self.downloads += 1
            return super()._download_file_to_fd(remote_path, destination_fd, **kwargs)

    publisher = CountingPublisher(tmp_path / "remote")
    receipt = publish_file(
        publisher,
        source,
        run_id="r1",
        relative_path="payload.bin",
        kind="artifact",
    )
    publisher.downloads = 0
    monkeypatch.setattr(publish_module, "MAX_SYNC_TOTAL_BYTES", receipt.size - 1)

    with pytest.raises(PublishError, match="receipt bytes"):
        sync_and_verify(
            publisher,
            run_id="r1",
            local_root=tmp_path / "synced",
            relative_paths=[receipt.relative_path],
        )
    assert publisher.downloads == 0


def test_sync_resource_preflight_rejects_insufficient_free_space(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")

    class CountingPublisher(LocalPublisher):
        def __init__(self, root):
            super().__init__(root)
            self.downloads = 0

        def _download_file_to_fd(self, remote_path, destination_fd, **kwargs):
            self.downloads += 1
            return super()._download_file_to_fd(remote_path, destination_fd, **kwargs)

    publisher = CountingPublisher(tmp_path / "remote")
    receipt = publish_file(
        publisher,
        source,
        run_id="r1",
        relative_path="payload.bin",
        kind="artifact",
    )
    publisher.downloads = 0
    monkeypatch.setattr(
        publish_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1, used=1, free=0),
    )

    with pytest.raises(PublishError, match="insufficient free space"):
        sync_and_verify(
            publisher,
            run_id="r1",
            local_root=tmp_path / "synced",
            relative_paths=[receipt.relative_path],
        )
    assert publisher.downloads == 0


@pytest.mark.parametrize(
    ("limit_name", "paths", "message"),
    [
        ("MAX_SYNC_RECEIPTS", ["one", "two"], "receipt count"),
        ("MAX_SYNC_RELATIVE_PATH_BYTES", ["long-path"], "path exceeds"),
    ],
)
def test_sync_resource_preflight_rejects_bounded_path_listing(
    tmp_path, monkeypatch, limit_name, paths, message
):
    class NoReadPublisher:
        def __init__(self):
            self.reads = 0

        def read_file(self, _remote_path):
            self.reads += 1
            raise AssertionError("receipt reads must not occur after path preflight failure")

    if limit_name == "MAX_SYNC_RECEIPTS":
        monkeypatch.setattr(publish_module, limit_name, 1)
    else:
        monkeypatch.setattr(publish_module, limit_name, 3)
    publisher = NoReadPublisher()

    with pytest.raises(PublishError, match=message):
        sync_and_verify(
            publisher,
            run_id="r1",
            local_root=tmp_path / "synced",
            relative_paths=paths,
        )
    assert publisher.reads == 0


def test_descriptor_download_preserves_legacy_download_file_publishers(tmp_path):
    calls: list[int | None] = []

    class DownloadOnlyPublisher:
        def download_file(self, remote_path, destination, *, expected_size=None):
            assert remote_path == "r1/artifacts/payload.bin"
            calls.append(expected_size)
            destination.write_bytes(b"legacy payload")

    temporary = publish_module._exclusive_temp_bytes(
        b"", directory=tmp_path, prefix="destination-"
    )
    try:
        publish_module._download_remote_expected(
            DownloadOnlyPublisher(),
            "r1/artifacts/payload.bin",
            temporary,
            len(b"legacy payload"),
        )
        assert publish_module._read_fd(temporary.fd) == b"legacy payload"
    finally:
        temporary.cleanup()
    assert calls == [len(b"legacy payload")]


def test_hf_descriptor_download_timeout_scales_with_known_size(tmp_path):
    calls: list[dict[str, object]] = []

    def runner(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    publisher = HfPublisher("org/results", runner=runner, timeout=120)
    publisher._entry_definitively_missing = lambda *_args, **_kwargs: False
    fd = os.open(tmp_path / "destination", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(PublishError, match="no regular target"):
            publisher._download_file_to_fd(
                "large.bin", fd, expected_size=256 * 1024 * 1024
            )
    finally:
        os.close(fd)
    assert calls and calls[0]["timeout"] >= 60 + 256


def test_hf_typed_entry_missing_is_exposed_as_missing_remote(tmp_path, monkeypatch):
    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Error: File not found in repository.",
        )

    publisher = HfPublisher("org/results", runner=runner)
    monkeypatch.setattr(
        publisher,
        "_entry_definitively_missing",
        lambda *_args, **_kwargs: True,
    )
    with pytest.raises(MissingRemote):
        publisher.read_file("absent.bin")
    with pytest.raises(MissingRemote):
        publisher.download_file("absent.bin", tmp_path / "downloaded.bin")


@pytest.mark.parametrize(("entries", "expected"), [([], True), ([object()], False)])
def test_hf_missing_classifier_requires_empty_revision_aware_path_response(
    entries,
    expected,
):
    calls = []

    class Api:
        def repo_info(self, **kwargs):
            calls.append(("repo_info", kwargs))
            return SimpleNamespace(sha="a" * 40)

        def get_paths_info(self, **kwargs):
            calls.append(("get_paths_info", kwargs))
            return entries

    publisher = HfPublisher(
        "org/results",
        token="hf_test_token_123",
        api=Api(),
    )
    assert publisher._entry_definitively_missing(
        "immutable.bin",
        revision="a" * 40,
    ) is expected
    assert calls == [
        ("repo_info", {
            "repo_id": "org/results",
            "repo_type": "model",
            "revision": "a" * 40,
            "token": "hf_test_token_123",
        }),
        ("get_paths_info", {
            "repo_id": "org/results",
            "paths": ["immutable.bin"],
            "repo_type": "model",
            "revision": "a" * 40,
            "token": "hf_test_token_123",
        }),
    ]


def test_hf_missing_classifier_fails_closed_on_path_api_error():
    class Api:
        def repo_info(self, **_kwargs):
            raise RuntimeError("revision, repository, auth, or transport failure")

        def get_paths_info(self, **_kwargs):
            raise AssertionError("path query must not follow an unresolved revision")

    publisher = HfPublisher("org/results", api=Api())
    assert not publisher._entry_definitively_missing(
        "immutable.bin",
        revision="a" * 40,
    )


def test_hf_missing_classifier_rejects_mismatched_resolved_commit():
    class Api:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="b" * 40)

        def get_paths_info(self, **_kwargs):
            raise AssertionError("path query must not use a mismatched commit")

    publisher = HfPublisher("org/results", api=Api())
    assert not publisher._entry_definitively_missing(
        "immutable.bin",
        revision="a" * 40,
    )


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Revision Not Found for url: https://huggingface.invalid/private",
        "Repository Not Found: HTTP 404",
        "404 Client Error: Not Found",
    ],
)
def test_hf_ambiguous_cli_not_found_never_authorizes_mutation(
    tmp_path,
    monkeypatch,
    diagnostic,
):
    source = tmp_path / "replacement.bin"
    source.write_bytes(b"replacement")

    class Api:
        def __init__(self):
            self.mutations = 0

        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="a" * 40)

        def create_commit(self, **_kwargs):
            self.mutations += 1

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=diagnostic)

    api = Api()
    publisher = HfPublisher("org/results", runner=runner, api=api)
    monkeypatch.setattr(
        publisher,
        "_entry_definitively_missing",
        lambda *_args, **_kwargs: False,
    )
    with pytest.raises(PublishError, match="could not inspect"):
        publisher.put_file(source, "immutable.bin")
    assert api.mutations == 0


def test_hf_success_without_downloaded_file_never_authorizes_mutation(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "replacement.bin"
    source.write_bytes(b"replacement")

    class Api:
        def __init__(self):
            self.mutations = 0

        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="a" * 40)

        def create_commit(self, **_kwargs):
            self.mutations += 1

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    api = Api()
    publisher = HfPublisher("org/results", runner=runner, api=api)
    monkeypatch.setattr(
        publisher,
        "_entry_definitively_missing",
        lambda *_args, **_kwargs: False,
    )
    with pytest.raises(PublishError, match="could not inspect"):
        publisher.put_file(source, "immutable.bin")
    assert api.mutations == 0


def test_hf_explicit_token_is_private_to_api_and_cli(monkeypatch):
    token = "hf_test_token_123"
    api_tokens: list[str | None] = []

    class FakeHfApi:
        def __init__(self, *, token=None):
            self.token = token
            api_tokens.append(token)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeHfApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(command, **kwargs):
        calls.append((list(command), kwargs))
        return subprocess.CompletedProcess(
            command,
            1 if command[-1] == "failure" else 0,
            stdout="",
            stderr=f"token={token}" if command[-1] == "failure" else "",
        )

    publisher = HfPublisher("org/results", token=token, runner=runner)
    assert publisher._api_client().token == token
    publisher._run(["hf", "whoami"])
    assert api_tokens == [token]
    assert calls and calls[0][0] == ["hf", "whoami"]
    environment = calls[0][1]["env"]
    assert isinstance(environment, dict)
    assert environment["HF_TOKEN"] == token
    assert token not in calls[0][0]
    with pytest.raises(PublishError) as error:
        publisher._run(["hf", "download", "failure"])
    assert token not in str(error.value)


@pytest.mark.parametrize("token", ["", "hf_", "hf bad token", "token-without-prefix"])
def test_hf_token_format_is_rejected(token):
    with pytest.raises(ValueError, match="token"):
        HfPublisher("org/results", token=token)


def test_hf_owned_temporary_upload_is_read_from_byte_zero(monkeypatch, tmp_path):
    uploaded: list[bytes] = []

    class FakeOperation:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeApi:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="a" * 40)

        def create_commit(self, **kwargs):
            operation = list(kwargs["operations"])[0]
            with open(operation.path_or_fileobj, "rb") as handle:
                uploaded.append(handle.read())

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.CommitOperationAdd = FakeOperation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    publisher = HfPublisher("org/results", api=FakeApi())
    monkeypatch.setattr(
        publisher,
        "_target_state",
        lambda *_args, **_kwargs: "missing",
    )
    temporary = publish_module._exclusive_temp_bytes(
        b"owned payload", directory=tmp_path, prefix="owned-"
    )
    try:
        publisher.put_file(temporary, "r1/artifacts/payload.bin")
    finally:
        temporary.cleanup()
    assert uploaded == [b"owned payload"]


def test_hf_upload_uses_private_readonly_snapshot_inode(monkeypatch, tmp_path):
    source = tmp_path / "mutable-source.bin"
    source.write_bytes(b"expected")
    observed: dict[str, object] = {}

    class FakeOperation:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class FakeApi:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="a" * 40)

        def create_commit(self, **kwargs):
            operation = list(kwargs["operations"])[0]
            path = Path(operation.path_or_fileobj)
            observed["mode"] = stat.S_IMODE(path.stat().st_mode)
            try:
                with path.open("r+b") as handle:
                    handle.write(b"wrong___")
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.seek(0)
                    observed["bytes"] = handle.read()
            except PermissionError:
                observed["write_blocked"] = True
                observed["bytes"] = path.read_bytes()

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.CommitOperationAdd = FakeOperation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    publisher = HfPublisher("org/results", api=FakeApi())
    monkeypatch.setattr(
        publisher,
        "_target_state",
        lambda *_args, **_kwargs: "missing",
    )
    publisher.put_file(source, "immutable.bin")
    assert observed == {
        "mode": 0o400,
        "write_blocked": True,
        "bytes": b"expected",
    }
    assert source.read_bytes() == b"expected"


def _tar_with_members(path: Path, members: list[tuple[str, bytes, int]]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, payload, mode in members:
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_stage_archive_rejects_special_modes_and_expansion_ceilings(tmp_path, monkeypatch):
    special = tmp_path / "special.tar"
    _tar_with_members(special, [("payload", b"x", 0o4755)])
    with pytest.raises(StageVerificationError, match="special permission"):
        verify_stage_tar(special)

    members = tmp_path / "members.tar"
    _tar_with_members(members, [("one", b"1", 0o644), ("two", b"2", 0o644)])
    monkeypatch.setattr(publish_module, "MAX_STAGE_ARCHIVE_MEMBERS", 1)
    with pytest.raises(StageVerificationError, match="too many members"):
        verify_stage_tar(members)

    expanded = tmp_path / "expanded.tar"
    _tar_with_members(expanded, [("payload", b"123", 0o644)])
    monkeypatch.setattr(publish_module, "MAX_STAGE_ARCHIVE_MEMBERS", 100_000)
    monkeypatch.setattr(publish_module, "MAX_STAGE_EXPANDED_BYTES", 2)
    with pytest.raises(StageVerificationError, match="expanded size"):
        verify_stage_tar(expanded)


def test_stage_extraction_normalizes_regular_and_directory_modes(tmp_path):
    archive_path = tmp_path / "modes.tar"
    with tarfile.open(archive_path, "w") as archive:
        directory = tarfile.TarInfo("nested/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o700
        archive.addfile(directory)
        payload = tarfile.TarInfo("nested/payload")
        payload.mode = 0o600
        payload.size = 1
        archive.addfile(payload, io.BytesIO(b"x"))
    destination = tmp_path / "extracted"
    destination.mkdir()
    with tarfile.open(archive_path) as archive:
        publish_module._safe_extract(archive, destination)
    assert (destination / "nested").stat().st_mode & 0o777 == 0o755
    assert (destination / "nested/payload").stat().st_mode & 0o777 == 0o644


def test_stage_archive_snapshot_rejects_path_substitution(tmp_path, monkeypatch):
    archive = tmp_path / "archive.tar"
    _tar_with_members(archive, [("payload", b"x", 0o644)])
    original_copy = publish_module._copy_fd

    def copy_then_replace(source_fd, destination_fd):
        result = original_copy(source_fd, destination_fd)
        archive.unlink()
        archive.write_bytes(b"replacement")
        return result

    monkeypatch.setattr(publish_module, "_copy_fd", copy_then_replace)
    with pytest.raises(StageVerificationError, match="path changed"):
        verify_stage_tar(archive)


def test_stage_verification_rejects_unmanifested_non_python_file(tmp_path):
    source = tmp_path / "source"
    (source / "src/ouro_jlens").mkdir(parents=True)
    (source / "artifacts/jlens/data").mkdir(parents=True)
    (source / "utilities/tests/unit").mkdir(parents=True)
    (source / "src/ouro_jlens/example.py").write_text("x = 1\n")
    repo_root = Path(__file__).resolve().parents[3]
    for relative in (
        publish_module.FITTING_CORPUS_GENERATOR_RELATIVE,
        publish_module.FITTING_CORPUS_RELATIVE,
        publish_module.FITTING_CORPUS_PROVENANCE_RELATIVE,
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo_root / relative).read_bytes())
    (source / "utilities/tests/unit/test_ouro_jlens.py").write_text("def test_ok(): pass\n")
    archive = tmp_path / "stage.tar.gz"
    build_stage_tar(source, archive)
    installed = tmp_path / "installed"
    extract_stage_tar(archive, installed)
    (installed / "unmanifested.txt").write_text("not in manifest\n")
    with pytest.raises(StageVerificationError, match="unlisted stage payload"):
        verify_stage_root(installed)


def test_status_binds_launch_and_stage_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("JLENS_ATTEMPT_ID", "attempt-1")
    monkeypatch.setenv("JLENS_LAUNCH_NONCE", "nonce-1-" + "n" * 25)
    monkeypatch.setenv("EXPECTED_STAGE_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("EXPECTED_STAGE_SOURCE_HEAD", "b" * 40)
    monkeypatch.setenv("JLENS_WORKER_DEADLINE_EPOCH", str(int(publish_module.time.time()) + 3600))
    args = SimpleNamespace(
        repo=None,
        local_publisher=tmp_path / "remote",
        run_id="r1",
        local_root=tmp_path / "local",
        state="running",
        stage="setup",
        exit_code=0,
        kind="heartbeat",
        receipt_root=tmp_path / "receipts",
    )
    publish_module._cmd_status(args)
    publisher = LocalPublisher(tmp_path / "remote")
    status_files = publisher.list_files("r1/artifacts/status/")
    assert len(status_files) == 1
    payload = json.loads(publisher.read_file(status_files[0]))
    assert payload["launch_nonce"] == "nonce-1-" + "n" * 25
    assert payload["stage_manifest_sha256"] == "a" * 64
    assert payload["stage_source_head"] == "b" * 40
    assert payload["worker_deadline_epoch"] > int(publish_module.time.time())


def test_hf_deferred_payload_metadata_is_pinned_to_one_exact_revision():
    receipt_a = publish_module.Receipt(
        1, "r1", "lens/a.pt", "r1/artifacts/lens/a.pt", "merged_lens",
        "1" * 64, 10,
    )
    receipt_b = publish_module.Receipt(
        1, "r1", "lens/b.pt.ckpt", "r1/artifacts/lens/b.pt.ckpt",
        "checkpoint", "2" * 64, 20,
    )
    calls = []

    class Api:
        def repo_info(self, **kwargs):
            calls.append(("head", kwargs))
            return SimpleNamespace(sha="a" * 40)

        def get_paths_info(self, **kwargs):
            calls.append(("paths", kwargs))
            return [
                SimpleNamespace(
                    path="r1/artifacts/lens/b.pt.ckpt", size=20,
                    lfs=SimpleNamespace(sha256="2" * 64),
                ),
                SimpleNamespace(
                    path="r1/artifacts/lens/a.pt", size=10,
                    lfs=SimpleNamespace(sha256="1" * 64),
                ),
            ]

    evidence = HfPublisher(
        "org/results", token="hf_test_token_123", api=Api()
    ).verify_remote_receipts_metadata([receipt_b, receipt_a])
    assert evidence == {"revision": "a" * 40, "count": 2}
    assert calls[1] == ("paths", {
        "repo_id": "org/results",
        "paths": [
            "r1/artifacts/lens/a.pt", "r1/artifacts/lens/b.pt.ckpt"
        ],
        "repo_type": "model",
        "revision": "a" * 40,
        "token": "hf_test_token_123",
    })


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (None, "missing"),
        (SimpleNamespace(
            path="r1/artifacts/lens/a.pt", size=11,
            lfs=SimpleNamespace(sha256="1" * 64),
        ), "differs"),
        (SimpleNamespace(
            path="r1/artifacts/lens/a.pt", size=10,
            lfs=SimpleNamespace(sha256="2" * 64),
        ), "differs"),
        (SimpleNamespace(
            path="r1/artifacts/lens/a.pt", size=10, lfs=None,
        ), "omitted"),
    ],
)
def test_hf_deferred_payload_metadata_fails_closed(entry, message):
    receipt = publish_module.Receipt(
        1, "r1", "lens/a.pt", "r1/artifacts/lens/a.pt", "merged_lens",
        "1" * 64, 10,
    )

    class Api:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="a" * 40)

        def get_paths_info(self, **_kwargs):
            return [] if entry is None else [entry]

    with pytest.raises(PublishError, match=message):
        HfPublisher("org/results", api=Api()).verify_remote_receipts_metadata(
            [receipt]
        )


@pytest.mark.parametrize("response", ["duplicate", "unrequested", "error"])
def test_hf_deferred_payload_metadata_rejects_inexact_or_failed_api(response):
    receipt = publish_module.Receipt(
        1, "r1", "lens/a.pt", "r1/artifacts/lens/a.pt", "merged_lens",
        "1" * 64, 10,
    )
    exact = SimpleNamespace(
        path=receipt.remote_path, size=receipt.size,
        lfs=SimpleNamespace(sha256=receipt.sha256),
    )

    class Api:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="a" * 40)

        def get_paths_info(self, **_kwargs):
            if response == "error":
                raise RuntimeError("auth, revision, transport, or server failure")
            if response == "duplicate":
                return [exact, exact]
            return [SimpleNamespace(
                path="r1/artifacts/lens/unrequested.pt", size=10,
                lfs=SimpleNamespace(sha256="1" * 64),
            )]

    with pytest.raises(PublishError):
        HfPublisher("org/results", api=Api()).verify_remote_receipts_metadata(
            [receipt]
        )


def test_local_deferred_payload_metadata_rehashes_current_bytes(tmp_path):
    publisher = LocalPublisher(tmp_path / "remote")
    source = tmp_path / "tensor.pt"
    source.write_bytes(b"tensor")
    receipt = publish_file(
        publisher, source, run_id="r1", relative_path="lens/a.pt",
        kind="merged_lens",
    )
    assert publisher.verify_remote_receipts_metadata([receipt]) == {
        "revision": "local", "count": 1,
    }
    remote_path = tmp_path / "remote" / receipt.remote_path
    remote_path.chmod(0o600)
    remote_path.write_bytes(b"poison")
    with pytest.raises(ImmutableConflict, match="differs"):
        publisher.verify_remote_receipts_metadata([receipt])
