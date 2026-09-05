"""Immutable run artifacts and stage-package verification for JLens.

The paid run is deliberately coupled to this small, boring protocol.  Every
file is addressed below a caller supplied ``run_id`` and is accompanied by a
receipt containing its SHA-256 digest.  A publisher must acknowledge the
receipt after it has been written; an existing path can only be reused when
its receipt and bytes are identical.

The module keeps Hugging Face optional for local tests and offline dry runs.
``LocalPublisher`` is filesystem-backed, while ``HfPublisher`` uses the
official ``huggingface_hub`` commit API for parent-OID compare-and-swap writes
and the CLI only for remote reads/downloads.  The shell entry points use the
commands at the bottom of this file.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import gzip
import hashlib
import importlib.metadata as importlib_metadata
import inspect
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Protocol


SCHEMA_VERSION = 1
MODEL_REPOSITORY = "ByteDance/Ouro-2.6B"
MODEL_REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"
JLENS_REPOSITORY = "https://github.com/anthropics/jacobian-lens.git"
JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
TARGET_PYTHON = "3.12"
RUNTIME_DEPENDENCIES = {
    "transformers": "4.54.1",
    "numpy": "2.4.3",
    "scikit-learn": "1.8.0",
    "threadpoolctl": "3.6.0",
    "matplotlib": "3.10.8",
    "safetensors": "0.7.0",
    "accelerate": "1.13.0",
    "huggingface_hub": "0.36.2",
}
# The pinned Ouro image still requires transformers 4.54.1.  The exact
# jacobian-lens revision declares ``transformers>=5.5``; setup installs jlens
# with ``--no-deps`` and verifies this deliberate, documented compatibility
# override after installation.  Keep both sides explicit so a future pin
# change cannot turn an accidental resolver conflict into a different run.
JLENS_TRANSFORMERS_REQUIREMENT = "transformers>=5.5"
JLENS_TRANSFORMERS_OVERRIDE = RUNTIME_DEPENDENCIES["transformers"]
RUNTIME_IMAGE = (
    "runpod/pytorch@sha256:"
    "bbe1496e2215cca3d25a5e5cd291d31ea86603e4577a81eb40096787a50e5303"
)
# Compatibility name consumed by the controller's canonical-image lookup.
JLENS_IMAGE_DIGEST = RUNTIME_IMAGE
RUNTIME_LOCK_RELATIVE = "src/ouro_jlens/runtime.lock.json"
# This digest is a trust anchor for the committed lock itself.  A verifier
# must not accept a lock whose metadata was edited together with its expected
# values: package versions and installed-file aggregates are meaningful only
# when the lock bytes are the reviewed artifact.
RUNTIME_LOCK_SHA256 = "e47cb6fc32c345326291e79adb80c25512c3417cb05db3941f5ff1871723e571"
FITTING_CORPUS_SHA256 = "972bd0234a9530a46c77dfcc68f429dba81c708e6205e787accf0ed01f9bce7d"
FITTING_CORPUS_PROVENANCE_SHA256 = (
    "38ee1cca019664de4f028971b0cb7d02d532829f0a9151e43d1897b97030d86e"
)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
LAUNCH_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
WORKER_DEADLINE_RE = re.compile(r"^[0-9]{1,20}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RECEIPT_INDEX_RE = re.compile(r"^receipt-index\.[0-9a-f]{64}\.json$")
FITTING_CORPUS_RELATIVE = "artifacts/jlens/data/wikitext_prompts_b08601e.json"
FITTING_CORPUS_PROVENANCE_RELATIVE = (
    "artifacts/jlens/data/wikitext_prompts_b08601e.provenance.json"
)
FITTING_CORPUS_GENERATOR_RELATIVE = "src/ouro_jlens/fetch_wikitext.py"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_DATASETS_VERSION = "4.8.5"
HF_REPO_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$"
)
HF_TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9][A-Za-z0-9_-]{2,}$")
# A token is intentionally read from a descriptor-backed, permission-checked
# file rather than accepted on a command line.  The upper bound is generous
# for current HF tokens while preventing a malformed path from becoming an
# unbounded memory read.
MAX_HF_TOKEN_BYTES = 4096
GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

# Stage archives are generated from a small source tree and prompt corpus.
# These ceilings are deliberately far above the reviewed stage size while
# bounding tar-bomb expansion before any member data is materialized.
MAX_STAGE_ARCHIVE_MEMBERS = 100_000
MAX_STAGE_EXPANDED_BYTES = 4 * 1024 * 1024 * 1024
SAFE_STAGE_FILE_MODE = 0o644
SAFE_STAGE_EXECUTABLE_MODE = 0o755
SAFE_STAGE_DIRECTORY_MODE = 0o755
STAGE_SPECIAL_MODE_BITS = stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX

# Controller-side synchronization is bounded before any remote payload is
# downloaded.  The expected run is well below these ceilings; they primarily
# prevent a malformed listing or index from exhausting the local disk.
MAX_SYNC_RECEIPTS = 100_000
MAX_SYNC_TOTAL_BYTES = 100 * 1024 * 1024 * 1024
MAX_SYNC_RELATIVE_PATH_BYTES = 4096
MAX_SYNC_PATH_BYTES = 16 * 1024 * 1024


class PublishError(RuntimeError):
    """An artifact could not be published or acknowledged."""


class ImmutableConflict(PublishError):
    """A run path already contains different bytes or metadata."""


class MissingRemote(PublishError):
    """A requested remote path does not exist."""


class StageVerificationError(PublishError):
    """A stage archive or its pinned inputs are not internally consistent."""


class _RemoteCommandFailure(PublishError):
    """Sanitized private command error; raw remote diagnostics are discarded."""


def verify_runtime_compatibility() -> dict[str, str | bool]:
    """Verify the explicit jlens/transformers compatibility override.

    The staged image pins an older transformers release for the Ouro model,
    while the pinned jlens package metadata asks for a newer major release.
    This check is intentionally run *after* the ``--no-deps`` install and
    fails closed if either side changes or the metadata cannot be inspected.
    """

    try:
        requirements = importlib_metadata.requires("jlens") or []
        installed = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError as exc:
        raise StageVerificationError("jlens/transformers metadata is unavailable") from exc
    declared = [item for item in requirements if item.split(";", 1)[0].strip().lower()
                == JLENS_TRANSFORMERS_REQUIREMENT]
    if declared != [JLENS_TRANSFORMERS_REQUIREMENT]:
        raise StageVerificationError(
            "pinned jlens metadata did not declare transformers>=5.5 exactly"
        )
    if installed != JLENS_TRANSFORMERS_OVERRIDE:
        raise StageVerificationError(
            f"compatibility override requires transformers=={JLENS_TRANSFORMERS_OVERRIDE}, "
            f"found {installed}"
        )
    return {
        "jlens_requirement": JLENS_TRANSFORMERS_REQUIREMENT,
        "transformers_runtime": installed,
        "compatibility_override": True,
    }


def _normalise_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_file_aggregate(dist: importlib_metadata.Distribution) -> dict[str, int | str]:
    """Hash every installed file named by one distribution's RECORD.

    The lock is intentionally stronger than a version pin.  A package can
    have the expected version while its wheel bytes, generated extension, or
    metadata differ, so setup verifies a deterministic aggregate of all
    readable installed distribution files before downloading the model.
    """

    rows: list[tuple[str, str, int]] = []
    for item in dist.files or ():
        relative = str(item).replace("\\", "/")
        # Bytecode caches are generated as a side effect of whichever import
        # happens first and contain timestamp/cache metadata.  They are not
        # wheel payload bytes; excluding them keeps a lock reproducible while
        # retaining every source, extension, and distribution-metadata file.
        if relative.endswith(".pyc") or "/__pycache__/" in f"/{relative}/":
            continue
        path = Path(dist.locate_file(item))
        # Some image distributions record an optional shared-object link or
        # a platform-specific path that is absent in this environment.  The
        # lock generator excludes those entries; the aggregate's count and
        # digest still fail closed if a regular locked file disappears.
        if path.is_symlink() or not path.is_file():
            continue
        size, digest = _path_digest(path, label="runtime package file")
        rows.append((relative, digest, size))
    rows.sort()
    material = "".join(
        f"{relative}  {digest}  {size}\n"
        for relative, digest, size in rows
    ).encode()
    return {
        "files_sha256": sha256_bytes(material),
        "file_count": len(rows),
        "bytes": sum(size for _relative, _digest, size in rows),
    }


def _runtime_lock_path(lock_path: str | Path | None) -> Path:
    return Path(lock_path) if lock_path is not None else Path(__file__).with_name("runtime.lock.json")


def verify_runtime_lock(
    lock_path: str | Path | None = None,
    *,
    image_digest: str | None = None,
    jlens_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the exact image/runtime lock and installed package bytes.

    ``image_digest`` defaults to the setup environment's explicit
    ``JLENS_IMAGE_DIGEST`` value.  Requiring that value, rather than merely
    trusting the lock's text, prevents a package-perfect run in a different
    container from being mistaken for the reviewed image.
    """

    path = _runtime_lock_path(lock_path)
    try:
        lock_bytes = _read_regular_file(path)
    except PublishError as exc:
        raise StageVerificationError("runtime lock is unavailable") from exc
    if sha256_bytes(lock_bytes) != RUNTIME_LOCK_SHA256:
        raise StageVerificationError("runtime lock digest does not match the committed lock")
    try:
        lock = json.loads(lock_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StageVerificationError("runtime lock is not valid JSON") from exc
    if not isinstance(lock, dict) or lock.get("schema") != "ouro_jlens.runtime_lock.v1":
        raise StageVerificationError("runtime lock schema is not supported")
    if lock.get("image") != RUNTIME_IMAGE:
        raise StageVerificationError("runtime lock image is not the approved digest")
    supplied_image = image_digest if image_digest is not None else os.environ.get("JLENS_IMAGE_DIGEST")
    if supplied_image != RUNTIME_IMAGE:
        raise StageVerificationError("JLENS_IMAGE_DIGEST does not match the approved image")

    python_pin = lock.get("python")
    if (not isinstance(python_pin, dict)
            or python_pin.get("implementation") != "CPython"
            or python_pin.get("version") != "3.12.3"):
        raise StageVerificationError("runtime lock Python identity is malformed")
    if (sys.implementation.name != "cpython"
            or ".".join(map(str, sys.version_info[:3])) != python_pin["version"]):
        raise StageVerificationError("runtime Python does not match the locked image")

    torch_pin = lock.get("torch")
    if not isinstance(torch_pin, dict):
        raise StageVerificationError("runtime lock Torch identity is malformed")
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        raise StageVerificationError("locked Torch runtime is unavailable") from exc
    if torch.__version__ != torch_pin.get("version"):
        raise StageVerificationError("runtime Torch version does not match the lock")
    if torch.version.cuda != torch_pin.get("cuda"):
        raise StageVerificationError("runtime Torch CUDA version does not match the lock")
    if bool(torch.backends.cuda.is_built()) != bool(torch_pin.get("cuda_built")):
        raise StageVerificationError("runtime Torch CUDA build does not match the lock")
    for name in ("torchvision", "torchaudio", "triton"):
        try:
            found = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as exc:
            raise StageVerificationError("locked Torch companion package is unavailable") from exc
        if found != torch_pin.get(name):
            raise StageVerificationError("runtime Torch companion package does not match the lock")

    jlens_pin = lock.get("jlens")
    if (not isinstance(jlens_pin, dict)
            or jlens_pin.get("repository") != JLENS_REPOSITORY
            or jlens_pin.get("revision") != JLENS_REVISION
            or jlens_pin.get("transformers_requirement") != JLENS_TRANSFORMERS_REQUIREMENT
            or jlens_pin.get("install") != "editable-no-deps"):
        raise StageVerificationError("runtime lock jlens identity is malformed")
    checkout = Path(jlens_dir) if jlens_dir is not None else Path.home() / "jacobian-lens"
    if checkout.is_symlink() or not (checkout / ".git").is_dir():
        raise StageVerificationError("locked jlens checkout is unavailable")
    try:
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        dirty = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StageVerificationError("could not inspect locked jlens checkout") from exc
    if (revision.returncode != 0 or revision.stdout.strip() != JLENS_REVISION
            or dirty.returncode != 0 or dirty.stdout.strip()):
        raise StageVerificationError("jlens checkout does not match the locked revision")

    expected_packages = lock.get("packages")
    if not isinstance(expected_packages, dict) or not expected_packages:
        raise StageVerificationError("runtime lock package map is malformed")
    actual_packages: dict[str, list[importlib_metadata.Distribution]] = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata.get("Name")
        if not isinstance(name, str) or not name:
            raise StageVerificationError("installed runtime package has no name")
        normalized = _normalise_distribution_name(name)
        actual_packages.setdefault(normalized, []).append(dist)
    expected_by_name = {
        _normalise_distribution_name(name): value
        for name, value in expected_packages.items()
        if isinstance(name, str)
    }
    if set(actual_packages) != set(expected_by_name):
        raise StageVerificationError("installed runtime package set does not match the lock")
    for normalized, distributions in sorted(actual_packages.items()):
        expected = expected_by_name[normalized]
        if not isinstance(expected, dict):
            raise StageVerificationError("runtime lock package entry is malformed")
        for dist in distributions:
            if dist.version != expected.get("version"):
                raise StageVerificationError("installed runtime package version does not match the lock")
            aggregate = _distribution_file_aggregate(dist)
            if any(aggregate.get(key) != expected.get(key)
                   for key in ("files_sha256", "file_count", "bytes")):
                raise StageVerificationError("installed runtime package bytes do not match the lock")
    compatibility = verify_runtime_compatibility()
    for name, version in RUNTIME_DEPENDENCIES.items():
        normalized = _normalise_distribution_name(name)
        if any(dist.version != version for dist in actual_packages[normalized]):
            raise StageVerificationError("direct runtime dependency does not match the lock")
    return {
        "image": RUNTIME_IMAGE,
        "python": python_pin["version"],
        "packages": len(expected_packages),
        "jlens_revision": JLENS_REVISION,
        "compatibility_override": compatibility["compatibility_override"],
    }


def validate_run_id(run_id: str) -> str:
    """Validate the path component used to isolate one run.

    In particular, slash, dot-dot, and empty IDs are rejected.  This is
    intentionally stricter than a generic filesystem helper because a run ID
    is also an externally visible immutable namespace.
    """

    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run_id {run_id!r}")
    if run_id in {".", ".."}:
        raise ValueError(f"invalid run_id {run_id!r}")
    return run_id


def safe_relative(path: str | Path) -> str:
    """Return a normalized, non-escaping POSIX relative path."""

    raw = str(path).replace("\\", "/")
    p = PurePosixPath(raw)
    raw_parts = raw.split("/")
    if (not raw or p.is_absolute() or raw in {".", ".."}
            or any(part in {"", ".", ".."} for part in raw_parts)):
        raise ValueError(f"unsafe relative path {path!r}")
    parts = p.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe relative path {path!r}")
    return p.as_posix()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    fd = _open_regular(Path(path), label="file")
    try:
        return _digest_fd(fd, chunk_size=chunk_size)[1]
    finally:
        os.close(fd)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def payload_remote_path(run_id: str, relative: str | Path) -> str:
    return f"{validate_run_id(run_id)}/artifacts/{safe_relative(relative)}"


def receipt_remote_path(run_id: str, relative: str | Path) -> str:
    return f"{validate_run_id(run_id)}/receipts/{safe_relative(relative)}.receipt.json"


def is_receipt_index(relative: str | Path) -> bool:
    """Return whether a relative artifact path names an index generation."""

    try:
        normalized = safe_relative(relative)
    except ValueError:
        return False
    name = PurePosixPath(normalized).name
    return normalized == "receipt-index.json" or bool(RECEIPT_INDEX_RE.fullmatch(name))


def receipt_index_relative_path(data: bytes) -> str:
    """Address an index by the digest of its canonical cumulative contents."""

    return f"receipt-index.{sha256_bytes(data)}.json"


@dataclass
class _OwnedTempFile:
    """A named temporary inode kept open by its owner until publication."""

    path: Path
    fd: int
    unlink_on_cleanup: bool = True
    sealed_readonly: bool = False
    _closed: bool = False

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def close(self) -> None:
        if not self._closed:
            os.close(self.fd)
            self._closed = True

    def cleanup(self) -> None:
        """Unlink only while the name still identifies this inode, then close."""

        if self._closed:
            return
        try:
            if self.unlink_on_cleanup:
                _unlink_owned_path(self.fd, self.path)
        finally:
            self.close()

    def __enter__(self) -> "_OwnedTempFile":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.cleanup()


def _identity_from_stat(value: os.stat_result) -> tuple[int, int, bool]:
    return (value.st_dev, value.st_ino, stat.S_ISREG(value.st_mode))


def _fd_identity(fd: int) -> tuple[int, int, bool]:
    return _identity_from_stat(os.fstat(fd))


def _path_identity(path: Path) -> tuple[int, int, bool]:
    return _identity_from_stat(os.stat(path, follow_symlinks=False))


def _assert_owned_path(fd: int, path: Path, *, label: str) -> None:
    """Require that ``path`` still names the regular inode held by ``fd``."""

    try:
        fd_identity = _fd_identity(fd)
        path_identity = _path_identity(path)
    except OSError as exc:
        raise PublishError(f"{label} temporary path is unavailable") from exc
    if not fd_identity[2]:
        raise PublishError(f"{label} temporary is not a regular file")
    if path_identity != fd_identity:
        raise PublishError(f"{label} temporary path no longer names its owned inode")


def _unlink_owned_path(fd: int, path: Path) -> bool:
    """Best-effort cleanup that never unlinks a replacement pathname."""

    try:
        _assert_owned_path(fd, path, label="cleanup")
    except (OSError, PublishError):
        return False
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        return False
    return True


def _write_fd(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:  # pragma: no cover - defensive OS failure guard
            raise OSError("short write to temporary file")
        view = view[written:]


def _content_stat(value: os.stat_result) -> tuple[int, int, int, int]:
    """Return inode metadata that changes when file contents are mutated.

    ``ctime`` is intentionally omitted here. Linking a temporary hard link
    and unlinking that name changes the shared inode's ctime without changing
    its bytes. Callers that observe that metadata-only transition perform a
    bounded second descriptor read and compare the actual bytes/digest.
    """

    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _full_content_stat(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (*_content_stat(value), value.st_ctime_ns)


def _read_fd_once(
    fd: int, *, chunk_size: int
) -> tuple[bytes, os.stat_result, os.stat_result]:
    """Read one descriptor and return bytes plus before/after metadata."""

    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise PublishError("file descriptor does not reference a regular file")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(fd)
    return b"".join(chunks), before, after


def _read_fd(fd: int, *, chunk_size: int = 1024 * 1024) -> bytes:
    """Read one open descriptor without reopening its pathname.

    A hard-link/unlink race can change only ctime while a descriptor is being
    read. In that narrow case, require one additional identical descriptor
    read before accepting the bytes. A size, mtime, inode, or byte change is
    still rejected immediately.
    """

    original_offset = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        result, before, after = _read_fd_once(fd, chunk_size=chunk_size)
        if _full_content_stat(before) == _full_content_stat(after):
            return result
        if _content_stat(before) != _content_stat(after):
            raise PublishError("regular file changed while being read")

        # ctime-only changes are expected when another writer unlinks its
        # temporary hard-link name. Confirm identical bytes from a later
        # stable descriptor read; this does not turn a content mutation into
        # an accepted read.
        previous = result
        for _attempt in range(4):
            current, retry_before, retry_after = _read_fd_once(
                fd, chunk_size=chunk_size
            )
            if _content_stat(retry_before) != _content_stat(retry_after):
                raise PublishError("regular file changed while being read")
            if current != previous:
                raise PublishError("regular file changed while being read")
            if _full_content_stat(retry_before) == _full_content_stat(retry_after):
                return current
            previous = current
        raise PublishError("regular file did not stabilize while being read")
    finally:
        os.lseek(fd, original_offset, os.SEEK_SET)


def _digest_fd_once(
    fd: int, *, chunk_size: int
) -> tuple[int, str, os.stat_result, os.stat_result]:
    """Hash one descriptor and return size/digest plus before/after metadata."""

    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise PublishError("file descriptor does not reference a regular file")
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(fd, chunk_size)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
    after = os.fstat(fd)
    return size, digest.hexdigest(), before, after


def _digest_fd(fd: int, *, chunk_size: int = 1024 * 1024) -> tuple[int, str]:
    """Return size and digest from one stable open regular-file descriptor.

    See :func:`_read_fd` for why a ctime-only transition receives a bounded
    identical-digest confirmation rather than failing the hard-link race.
    """

    original_offset = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        size, digest, before, after = _digest_fd_once(fd, chunk_size=chunk_size)
        if _full_content_stat(before) == _full_content_stat(after):
            return size, digest
        if _content_stat(before) != _content_stat(after):
            raise PublishError("regular file changed while being hashed")

        previous = (size, digest)
        for _attempt in range(4):
            retry_size, retry_digest, retry_before, retry_after = _digest_fd_once(
                fd, chunk_size=chunk_size
            )
            if _content_stat(retry_before) != _content_stat(retry_after):
                raise PublishError("regular file changed while being hashed")
            if (retry_size, retry_digest) != previous:
                raise PublishError("regular file changed while being hashed")
            if _full_content_stat(retry_before) == _full_content_stat(retry_after):
                return retry_size, retry_digest
            previous = (retry_size, retry_digest)
        raise PublishError("regular file did not stabilize while being hashed")
    finally:
        os.lseek(fd, original_offset, os.SEEK_SET)


def _confirmed_digest_fd(
    fd: int, *, attempts: int = 8, retry_delay: float = 0.001
) -> tuple[int, str]:
    """Return two consecutive stable, identical digests.

    Installing an immutable file uses a temporary hard link.  Unlinking that
    temporary name changes the shared inode's ctime without changing its
    bytes, so a concurrent loser can legitimately observe one strict
    :func:`_digest_fd` failure.  Requiring two subsequent stable, identical
    reads distinguishes that metadata-only race from a destination whose
    contents are still changing.
    """

    previous: tuple[int, str] | None = None
    last_error: PublishError | None = None
    for attempt in range(attempts):
        try:
            current = _digest_fd(fd)
        except PublishError as exc:
            previous = None
            last_error = exc
            if attempt + 1 < attempts and retry_delay > 0:
                time.sleep(retry_delay)
            continue
        if previous == current:
            return current
        previous = current
        # Yield to the winning installer so its final temporary-name unlink
        # and directory fsync can settle the shared inode's ctime.  The wait
        # is bounded, and two identical full-file digests are still required;
        # this never converts a changing destination into an accepted one.
        if attempt + 1 < attempts and retry_delay > 0:
            time.sleep(retry_delay)
    if last_error is not None:
        raise PublishError("immutable destination did not stabilize while being hashed") from last_error
    raise PublishError("immutable destination changed between confirmed digest reads")


def _open_regular(path: Path, *, label: str) -> int:
    _reject_symlink_ancestors(path)
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PublishError(f"{label} is unavailable: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PublishError(f"{label} is not a regular file: {path}")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _copy_fd(source_fd: int, destination_fd: int) -> tuple[int, str]:
    """Copy and hash bytes from one stable source fd into another fd."""

    before = os.fstat(source_fd)
    if not stat.S_ISREG(before.st_mode):
        raise PublishError("immutable source is not a regular file")
    os.lseek(source_fd, 0, os.SEEK_SET)
    os.lseek(destination_fd, 0, os.SEEK_SET)
    os.ftruncate(destination_fd, 0)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(source_fd, 1024 * 1024)
        if not chunk:
            break
        _write_fd(destination_fd, chunk)
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(source_fd)
    before_identity = (before.st_dev, before.st_ino, before.st_size,
                       before.st_mtime_ns, before.st_ctime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size,
                      after.st_mtime_ns, after.st_ctime_ns)
    if size != before.st_size or stat.S_IMODE(before.st_mode) != stat.S_IMODE(after.st_mode):
        raise PublishError("immutable source changed while being copied")
    if before_identity != after_identity:
        # A temporary hard-link install/unlink may update only ctime after
        # the source bytes have been copied. Confirm the source digest from a
        # later descriptor read; any actual byte/size/mtime change remains a
        # hard failure.
        if _content_stat(before) != _content_stat(after):
            raise PublishError("immutable source changed while being copied")
        confirmed_size, confirmed_digest = _digest_fd(source_fd)
        if (confirmed_size, confirmed_digest) != (size, digest.hexdigest()):
            raise PublishError("immutable source changed while being copied")
    return size, digest.hexdigest()


def _path_digest(
    path: Path, *, label: str, chunk_size: int = 1024 * 1024
) -> tuple[int, str]:
    """Hash a regular path through one descriptor, never by reopen-after-stat."""

    fd = _open_regular(path, label=label)
    try:
        return _digest_fd(fd, chunk_size=chunk_size)
    finally:
        os.close(fd)

def _atomic_write(path: Path, data: bytes) -> None:
    """Replace a regular destination with a complete, fsynced file.

    This helper is for the few deliberately mutable ``latest``-style files.
    Immutable protocol files use :func:`_write_immutable` instead.  Both
    paths use ``mkstemp`` so an attacker cannot preplant a predictable temp
    symlink, and both reject links before any bytes are written.
    """

    _reject_symlink_ancestors(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ImmutableConflict(f"atomic destination is not a regular file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(path)
    temporary = _exclusive_temp_bytes(
        data, directory=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        _replace_owned(temporary, path)
    finally:
        temporary.cleanup()


def _replace_owned(temporary: _OwnedTempFile, destination: Path) -> None:
    """Replace a destination from an fd-anchored temporary inode."""

    _assert_owned_path(temporary.fd, temporary.path, label="atomic")
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ImmutableConflict(f"atomic destination is not a regular file: {destination}")
    _reject_symlink_ancestors(destination)
    os.replace(temporary.path, destination)
    _fsync_directory(destination.parent)


def _write_immutable(path: Path, data: bytes) -> None:
    """Create a local protocol file without replacing an existing path.

    The temporary inode stays open through the hard-link operation.  Its
    pathname is checked against the descriptor immediately before linking and
    is only removed if it still names that same inode.
    """

    _reject_symlink_ancestors(path)
    if path.is_symlink():
        raise ImmutableConflict(f"immutable path is a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(path)
    temporary = _exclusive_temp_bytes(
        data,
        directory=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        _link_immutable(
            temporary.path,
            path,
            source_fd=temporary.fd,
            source_digest=_digest_fd(temporary.fd)[1],
        )
        _fsync_directory(path.parent)
    finally:
        temporary.cleanup()


def _read_regular_file(path: Path) -> bytes:
    """Read one regular file without following a destination symlink."""

    fd = _open_regular(path, label="file")
    try:
        return _read_fd(fd)
    finally:
        os.close(fd)


def read_hf_token_file(path: str | Path) -> str:
    """Read one tightly permissioned HF token without following links.

    The caller supplies only a pathname.  The returned value is suitable for
    ``HfPublisher(token=...)``; it is never included in an exception or a
    command argument by this module.  Exact mode ``0600`` is required so a
    token accidentally copied with broader permissions cannot silently enter
    a paid run.
    """

    try:
        token_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise PublishError("HF token file path is malformed") from exc
    if token_path.is_symlink():
        raise PublishError("HF token file path is unsafe")
    try:
        _reject_symlink_ancestors(token_path)
    except PublishError:
        raise PublishError("HF token file path is unsafe") from None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(token_path, flags)
    except OSError:
        raise PublishError("HF token file is unavailable") from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PublishError("HF token file is not a regular file")
        if stat.S_IMODE(before.st_mode) != 0o600:
            raise PublishError("HF token file must have mode 0600")
        if before.st_size <= 0 or before.st_size > MAX_HF_TOKEN_BYTES:
            raise PublishError("HF token file content is malformed")
        raw = _read_fd(fd)
        after = os.fstat(fd)
        if (not stat.S_ISREG(after.st_mode)
                or stat.S_IMODE(after.st_mode) != 0o600
                or _full_content_stat(before) != _full_content_stat(after)):
            raise PublishError("HF token file changed while being read")
    finally:
        os.close(fd)
    try:
        token = raw.decode("ascii")
    except UnicodeDecodeError:
        raise PublishError("HF token file content is malformed") from None
    if HF_TOKEN_RE.fullmatch(token) is None:
        raise PublishError("HF token file content is malformed")
    return token


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability for local immutable installs."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive_temp_bytes(
    data: bytes, *, directory: Path, prefix: str, suffix: str = ""
) -> _OwnedTempFile:
    """Write complete bytes to an exclusive temporary inode and keep its fd."""

    _reject_symlink_ancestors(directory / "placeholder")
    directory.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(directory / "placeholder")
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(directory))
    path = Path(name)
    temporary = _OwnedTempFile(path, fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PublishError(f"temporary path is not a regular file: {path}")
        _write_fd(fd, data)
        os.fsync(fd)
        _fsync_directory(directory)
        return temporary
    except BaseException:
        temporary.cleanup()
        raise


def _seal_owned_temp_readonly(temporary: _OwnedTempFile) -> _OwnedTempFile:
    """Drop the only writable descriptor and retain an exact read-only inode."""

    _assert_owned_path(temporary.fd, temporary.path, label="seal")
    os.fsync(temporary.fd)
    os.fchmod(temporary.fd, 0o400)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        readonly_fd = os.open(temporary.path, flags)
    except OSError:
        raise PublishError("could not reopen upload snapshot read-only") from None
    try:
        if _fd_identity(readonly_fd) != _fd_identity(temporary.fd):
            raise PublishError("upload snapshot path changed while being sealed")
        if stat.S_IMODE(os.fstat(readonly_fd).st_mode) != 0o400:
            raise PublishError("upload snapshot is not read-only")
        if fcntl.fcntl(readonly_fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
            raise PublishError("upload snapshot descriptor is writable")
    except BaseException:
        os.close(readonly_fd)
        raise
    os.close(temporary.fd)
    temporary.fd = readonly_fd
    temporary.sealed_readonly = True
    return temporary


def _snapshot_upload_source(
    source_fd: int,
    source_path: Path,
) -> _OwnedTempFile:
    """Copy one stable source into a private read-only upload inode."""

    _assert_owned_path(source_fd, source_path, label="upload source")
    snapshot = _exclusive_temp_bytes(
        b"",
        # Keep the transient copy on the source filesystem. The paid worker
        # preflights free space there; /tmp may be a much smaller tmpfs.
        directory=source_path.parent,
        prefix="jlens-upload-snapshot-",
        suffix=".tmp",
    )
    try:
        copied_size, copied_digest = _copy_fd(source_fd, snapshot.fd)
        confirmed_size, confirmed_digest = _digest_fd(source_fd)
        if (confirmed_size, confirmed_digest) != (copied_size, copied_digest):
            raise PublishError("upload source changed while being snapshotted")
        _assert_owned_path(source_fd, source_path, label="upload source")
        _seal_owned_temp_readonly(snapshot)
        if _digest_fd(snapshot.fd) != (copied_size, copied_digest):
            raise PublishError("sealed upload snapshot differs from its source")
        return snapshot
    except BaseException:
        snapshot.cleanup()
        raise


def _reject_symlink_ancestors(path: Path) -> None:
    """Reject a writable path routed through an existing symlink directory."""

    current = Path(path).parent
    while current != current.parent:
        if current.is_symlink():
            raise PublishError(f"writable path has a symlinked parent: {current}")
        current = current.parent


def _install_immutable(source: Path | _OwnedTempFile, destination: Path) -> None:
    """Install ``source`` without replacing an existing destination."""

    if isinstance(source, _OwnedTempFile):
        _install_owned_immutable(source, destination)
        return
    source = Path(source)
    destination = Path(destination)
    _reject_symlink_ancestors(destination)
    if destination.is_symlink():
        raise ImmutableConflict(f"immutable path is a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(destination)
    source_fd = _open_regular(source, label="immutable source")
    temporary = _exclusive_temp_bytes(
        b"",
        directory=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        _copy_fd(source_fd, temporary.fd)
        os.fsync(temporary.fd)
        _link_immutable(
            temporary.path,
            destination,
            source_fd=temporary.fd,
            source_digest=_digest_fd(temporary.fd)[1],
        )
        _fsync_directory(destination.parent)
    finally:
        os.close(source_fd)
        temporary.cleanup()


def _install_owned_immutable(source: _OwnedTempFile, destination: Path) -> None:
    """Link an already-complete owned inode into an immutable destination."""

    destination = Path(destination)
    _reject_symlink_ancestors(destination)
    if destination.is_symlink():
        raise ImmutableConflict(f"immutable path is a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(destination)
    _size, source_digest = _digest_fd(source.fd)
    _link_immutable(
        source.path,
        destination,
        source_fd=source.fd,
        source_digest=source_digest,
    )
    _fsync_directory(destination.parent)


def _link_immutable(
    source: Path,
    destination: Path,
    *,
    source_for_comparison: Path | None = None,
    source_fd: int | None = None,
    source_digest: str | None = None,
) -> None:
    """Link a complete file without replacing an existing destination."""

    source = Path(source)
    destination = Path(destination)
    local_source_fd = False
    if source_fd is None:
        source_fd = _open_regular(source, label="immutable source")
        local_source_fd = True
    try:
        _assert_owned_path(source_fd, source, label="immutable source")
        _reject_symlink_ancestors(destination)
        if destination.is_symlink():
            raise ImmutableConflict(f"immutable path is a symlink: {destination}")
        try:
            # Refuse source symlinks even if the name changes after the
            # identity check.  The post-link check below rejects a replacement
            # regular file without consuming its bytes.
            os.link(source, destination, follow_symlinks=False)
        except FileExistsError as exc:
            if destination.is_symlink():
                raise ImmutableConflict(f"immutable path is a symlink: {destination}") from exc
            if not destination.is_file():
                raise ImmutableConflict(f"immutable path is not a regular file: {destination}") from exc
            destination_fd = _open_regular(destination, label="immutable destination")
            try:
                destination_size, destination_digest = _confirmed_digest_fd(destination_fd)
            finally:
                os.close(destination_fd)
            if source_digest is None:
                source_size, source_digest = _digest_fd(source_fd)
            else:
                source_size = os.fstat(source_fd).st_size
            if destination_size != source_size or destination_digest != source_digest:
                raise ImmutableConflict(f"immutable path already differs: {destination}") from exc
            return
        linked_identity = _path_identity(destination)
        owned_identity = _fd_identity(source_fd)
        if linked_identity != owned_identity:
            # Remove only the destination inode just observed.  This leaves an
            # attacker's original regular file untouched even if it was linked
            # during the race.
            _unlink_identity(destination, linked_identity)
            raise PublishError("immutable source path changed during link")
    finally:
        if local_source_fd:
            os.close(source_fd)


def _unlink_identity(path: Path, identity: tuple[int, int, bool]) -> bool:
    """Unlink ``path`` only when its current inode is ``identity``."""

    try:
        if _path_identity(path) != identity:
            return False
        path.unlink()
    except (FileNotFoundError, OSError):
        return False
    return True


class Publisher(Protocol):
    """Publisher interface used by publication and receipt discovery."""

    def put_file(self, source: Path, remote_path: str) -> None: ...

    def read_file(self, remote_path: str) -> bytes: ...

    def download_file(
        self,
        remote_path: str,
        destination: Path,
        *,
        expected_size: int | None = None,
    ) -> None: ...

    def list_files(self, prefix: str = "") -> list[str]: ...

    def verify_remote_receipts_metadata(
        self, receipts: Iterable[Any]
    ) -> Mapping[str, Any]: ...


def _remote_receipt_identity(receipt: Any) -> tuple[str, int, str]:
    """Return one receipt's validated remote byte identity."""

    try:
        remote_path = getattr(receipt, "remote_path")
        size = getattr(receipt, "size")
        digest = getattr(receipt, "sha256")
    except AttributeError as exc:
        raise PublishError("remote custody receipt is malformed") from exc
    try:
        remote_path = safe_relative(remote_path)
    except (TypeError, ValueError) as exc:
        raise PublishError("remote custody receipt has an unsafe path") from exc
    if (isinstance(size, bool) or not isinstance(size, int) or size < 0
            or not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None):
        raise PublishError("remote custody receipt has an invalid byte identity")
    return remote_path, size, digest


class LocalPublisher:
    """Filesystem-backed publisher for tests, recovery, and dry runs."""

    def __init__(self, root: str | Path):
        raw_root = Path(root)
        _reject_symlink_ancestors(raw_root)
        if raw_root.is_symlink():
            raise PublishError(f"publisher root is a symlink: {raw_root}")
        self.root = raw_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, remote_path: str) -> Path:
        rel = safe_relative(remote_path)
        lexical = self.root / rel
        _reject_symlink_ancestors(lexical)
        if lexical.is_symlink():
            raise PublishError(f"remote path is a symlink: {remote_path}")
        path = lexical.resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError(f"remote path escapes publisher root: {remote_path!r}")
        return path

    def put_file(self, source: Path | _OwnedTempFile, remote_path: str) -> None:
        destination = self._path(remote_path)
        if isinstance(source, _OwnedTempFile):
            _assert_owned_path(source.fd, source.path, label="upload")
            _install_owned_immutable(source, destination)
            return
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            raise PublishError(f"source is not a regular file: {source}")
        _install_immutable(source, destination)

    def read_file(self, remote_path: str) -> bytes:
        path = self._path(remote_path)
        try:
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
            return _read_regular_file(path)
        except FileNotFoundError as exc:
            raise MissingRemote(remote_path) from exc

    def download_file(
        self,
        remote_path: str,
        destination: Path,
        *,
        expected_size: int | None = None,
    ) -> None:
        source = self._path(remote_path)
        try:
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(source)
            if expected_size is not None and (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
            ):
                raise ValueError("expected_size must be a non-negative integer")
            if expected_size is not None and source.stat().st_size != expected_size:
                raise ImmutableConflict(f"remote payload size differs: {remote_path}")
            _install_immutable(source, Path(destination))
        except FileNotFoundError as exc:
            raise MissingRemote(remote_path) from exc

    def _download_file_to_fd(
        self,
        remote_path: str,
        destination_fd: int,
        *,
        expected_size: int | None = None,
    ) -> None:
        """Copy a remote file directly into an already-owned descriptor."""

        source = self._path(remote_path)
        try:
            source_fd = _open_regular(source, label="remote payload")
        except PublishError as first_error:
            # A concurrent immutable writer may create the path between the
            # failed open and any later existence check. Retry the descriptor
            # open once when the name has appeared; if it is still absent,
            # report MissingRemote so the caller can attempt its CAS write.
            # A symlink, non-regular file, or persistent open failure remains
            # a hard publication error.
            if not source.exists() and not source.is_symlink():
                raise MissingRemote(remote_path) from first_error
            try:
                source_fd = _open_regular(source, label="remote payload")
            except PublishError as retry_error:
                if not source.exists() and not source.is_symlink():
                    raise MissingRemote(remote_path) from retry_error
                raise
        try:
            if expected_size is not None and (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
            ):
                raise ValueError("expected_size must be a non-negative integer")
            if expected_size is not None and source.stat().st_size != expected_size:
                raise ImmutableConflict(f"remote payload size differs: {remote_path}")
            _copy_fd(source_fd, destination_fd)
            os.fsync(destination_fd)
        finally:
            os.close(source_fd)

    def list_files(self, prefix: str = "") -> list[str]:
        if prefix:
            raw_prefix = str(prefix).replace("\\", "/")
            if raw_prefix.startswith("/"):
                raise PublishError("invalid repository listing prefix")
            try:
                normalized_prefix = safe_relative(raw_prefix.rstrip("/"))
            except ValueError as exc:
                raise PublishError("invalid repository listing prefix") from exc
            base = self._path(normalized_prefix)
        else:
            base = self.root
        if not base.exists():
            return []
        files: list[str] = []
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                raise PublishError(f"publisher tree contains a symlink: {path}")
            if path.is_file():
                files.append(path.relative_to(self.root).as_posix())
        return files

    def verify_remote_receipts_metadata(
        self, receipts: Iterable[Any]
    ) -> Mapping[str, Any]:
        """Prove deferred local-publisher payloads still match their receipts."""

        identities: dict[str, tuple[int, str]] = {}
        for receipt in receipts:
            remote_path, size, digest = _remote_receipt_identity(receipt)
            if remote_path in identities:
                raise PublishError("remote custody receipt paths are duplicated")
            identities[remote_path] = (size, digest)
        for remote_path, expected in sorted(identities.items()):
            try:
                observed = _path_digest(
                    self._path(remote_path), label="deferred remote payload"
                )
            except (OSError, PublishError) as exc:
                raise PublishError(
                    f"could not verify deferred remote payload: {remote_path}"
                ) from exc
            if observed != expected:
                raise ImmutableConflict(
                    f"deferred remote payload differs from its receipt: {remote_path}"
                )
        return {"revision": "local", "count": len(identities)}


class HfPublisher:
    """Publisher backed by Hugging Face's parent-commit CAS API.

    Payload publication cannot use ``hf upload`` after a local head read: two
    workers could observe one head and both overwrite an immutable path.  The
    official ``HfApi.create_commit(..., parent_commit=...)`` operation binds
    the mutation to the exact head returned by ``repo_info``.  Only an HTTP
    412 (the API's proven parent-head conflict) is reconciled only through
    bounded read-only target checks; no failed commit is blindly retried for
    the same immutable path.
    """

    _CAS_ATTEMPTS = 1
    _RECONCILE_ATTEMPTS = 8
    _RECONCILE_DELAY = 0.25

    def __init__(
        self,
        repository: str,
        token: str | None = None,
        *,
        runner=None,
        timeout: int = 120,
        api=None,
    ):
        if not isinstance(repository, str) or not HF_REPO_RE.fullmatch(repository):
            raise ValueError("invalid Hugging Face repository")
        if token is not None and (
            not isinstance(token, str) or HF_TOKEN_RE.fullmatch(token) is None
        ):
            raise ValueError("invalid Hugging Face token")
        self.repository = repository
        self.runner = runner or subprocess.run
        self.timeout = timeout
        self.token = token
        # Keep the dependency lazy for local CPU tests and offline stage
        # checks.  Tests may inject a fake API, while real callers get the
        # official ``huggingface_hub.HfApi`` only for CAS/listing operations.
        self._api = api

    def _run(
        self,
        argv: list[str],
        *,
        pass_fds: tuple[int, ...] = (),
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        kwargs: dict[str, Any] = {
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": self.timeout if timeout is None else timeout,
        }
        if pass_fds:
            # ``/proc/self/fd/N`` is used only with a descriptor that remains
            # open for the child.  Injected test runners retain the old call
            # shape unless an fd-backed upload is explicitly exercised.
            kwargs["pass_fds"] = pass_fds
        # Always scrub credential variables copied from the controller.  An
        # unauthenticated/public publisher must not accidentally hand an
        # ambient secret to a child, and an authenticated publisher gets its
        # token only in this private environment copy.
        child_environment = os.environ.copy()
        child_environment.pop("HF_TOKEN", None)
        child_environment.pop("JLENS_HF_TOKEN_BOOTSTRAP", None)
        if self.token is not None:
            # Keep the credential out of argv and do not mutate the caller's
            # process environment.  The private copy is passed only to the
            # child that needs to authenticate the HF CLI.
            child_environment["HF_TOKEN"] = self.token
        kwargs["env"] = child_environment
        try:
            result = self.runner(argv, **kwargs)
        except (OSError, subprocess.SubprocessError) as exc:
            raise _RemoteCommandFailure(f"publisher command failed: {argv[0]}") from None
        except TypeError as exc:
            raise PublishError("publisher runner cannot preserve an owned upload descriptor") from None
        except Exception:
            # Do not let an injected runner or a client wrapper propagate a
            # diagnostic that may contain the credential.
            raise _RemoteCommandFailure(f"publisher command failed: {argv[0]}") from None
        returncode = getattr(result, "returncode", None)
        if returncode != 0:
            error = _RemoteCommandFailure(
                f"publisher command returned {returncode if returncode is not None else 'unknown'}: {argv[0]}"
            )
            raise error
        return result

    def _entry_definitively_missing(
        self,
        remote_path: str,
        *,
        revision: str | None = None,
    ) -> bool:
        """Classify absence only from the revision-aware paths API.

        CLI diagnostics such as HTTP 404 or ``not found`` are ambiguous:
        they can denote a missing repository or revision as well as a missing
        file.  The Hub metadata resolver can likewise report an entry error
        while a newly-created revision has not propagated to that resolver.
        The control-plane paths API both resolves the exact revision and
        returns an empty list for an absent requested path.  Every malformed
        response, authentication, repository, revision, transport, and server
        failure remains unknown/fail-closed.
        """

        try:
            token: str | bool = self.token if self.token is not None else False
            api = self._api_client()
            requested_revision = revision or "main"
            info = api.repo_info(
                repo_id=self.repository,
                repo_type="model",
                revision=requested_revision,
                token=token,
            )
            resolved_revision = getattr(info, "sha", None) or getattr(info, "oid", None)
            if (
                not isinstance(resolved_revision, str)
                or GIT_OID_RE.fullmatch(resolved_revision) is None
                or (
                    revision is not None
                    and GIT_OID_RE.fullmatch(revision) is not None
                    and resolved_revision != revision
                )
            ):
                return False
            entries = api.get_paths_info(
                repo_id=self.repository,
                paths=[remote_path],
                repo_type="model",
                revision=resolved_revision,
                token=token,
            )
        except Exception:
            return False
        if not isinstance(entries, list):
            return False
        if not entries:
            return True
        # A non-empty but mismatched response violates the exact-path request
        # contract and therefore cannot establish absence.
        return False

    def _transfer_timeout(self, expected_size: int | None) -> float:
        """Return a bounded timeout sized for at least 1 MiB/s plus overhead."""

        if expected_size is None:
            return float(self.timeout)
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError("expected_size must be a non-negative integer")
        # Keep the historical timeout as the floor for small transfers, but
        # avoid expiring a known-large model/result transfer at 120 seconds.
        seconds_at_one_mib_per_second = (expected_size + (1024 * 1024) - 1) // (1024 * 1024)
        return max(float(self.timeout), float(60 + seconds_at_one_mib_per_second))

    def _api_client(self):
        api = self._api
        if api is not None:
            return api
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]
        except (ImportError, ModuleNotFoundError):
            raise PublishError("huggingface_hub is required for CAS publication") from None
        try:
            api = HfApi(token=self.token)
        except Exception:
            raise PublishError("could not initialize huggingface_hub API") from None
        self._api = api
        return api

    def ensure_repository(self) -> None:
        """Create the staging namespace if needed, without writing an artifact."""

        try:
            self._api_client().create_repo(
                repo_id=self.repository,
                repo_type="model",
                private=True,
                exist_ok=True,
            )
        except Exception:
            raise PublishError("could not initialize Hugging Face repository") from None

    def _repository_head(self) -> str:
        """Read the current main-branch commit OID through the official API."""

        try:
            info = self._api_client().repo_info(
                repo_id=self.repository,
                revision="main",
                repo_type="model",
                token=self.token if self.token is not None else False,
            )
        except Exception:
            raise PublishError("could not read Hugging Face repository head") from None
        if isinstance(info, Mapping):
            oid = info.get("sha") or info.get("oid")
        else:
            oid = getattr(info, "sha", None) or getattr(info, "oid", None)
        # A commit parent must be an actual Git object ID, not an arbitrary
        # response field or a branch name.  Keep malformed server responses
        # from turning into an unbound mutation.
        if not isinstance(oid, str) or GIT_OID_RE.fullmatch(oid) is None:
            raise PublishError("Hugging Face repository head is malformed")
        return oid

    @staticmethod
    def _is_parent_conflict(error: BaseException) -> bool:
        """Classify only the official create_commit precondition failure."""

        response = getattr(error, "response", None)
        response_status = (
            response.get("status_code") if isinstance(response, Mapping)
            else getattr(response, "status_code", None)
        )
        return response_status == 412 or getattr(error, "status_code", None) == 412

    def _target_state(
        self,
        remote_path: str,
        size: int,
        digest: str,
        *,
        revision: str | None = None,
    ) -> str:
        """Return matching/different/missing/unknown without exposing errors.

        When ``revision`` is supplied, the read is pinned to the exact parent
        OID later used by ``create_commit``.  A main-branch read performed
        before or after that snapshot would leave an overwrite gap.
        """

        try:
            reader = self.read_file
            try:
                parameters = inspect.signature(reader).parameters
            except (TypeError, ValueError):
                parameters = {}
            accepts_size = (
                "expected_size" in parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            )
            accepts_revision = (
                "revision" in parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            )
            if revision is not None and not accepts_revision:
                return "unknown"
            kwargs: dict[str, Any] = {}
            if accepts_size:
                kwargs["expected_size"] = size
            if revision is not None:
                kwargs["revision"] = revision
            data = reader(remote_path, **kwargs)
        except MissingRemote:
            return "missing"
        except ImmutableConflict:
            # A descriptor-aware expected-size check may classify a remote
            # object as conflicting before its bytes are copied; that is still
            # definitive evidence that the immutable target differs.
            return "different"
        except Exception:
            # A transport/authentication/parse failure does not prove that the
            # target is absent.  Treating it as missing would permit a blind
            # retry after an unknown mutation outcome.
            return "unknown"
        if not isinstance(data, bytes):
            return "unknown"
        if len(data) == size and sha256_bytes(data) == digest:
            return "matching"
        return "different"

    def _reconcile_target(
        self, remote_path: str, size: int, digest: str
    ) -> str:
        """Reconcile one attempted mutation using read-only target reads.

        This method deliberately never reads the branch head and never calls
        ``create_commit``. It is used after a parent-commit conflict or an
        otherwise uncertain mutation outcome, where a missing target is not
        evidence that another mutation is safe.
        """

        state = "unknown"
        for attempt in range(self._RECONCILE_ATTEMPTS):
            state = self._target_state(remote_path, size, digest)
            if state in {"matching", "different"}:
                return state
            if attempt + 1 < self._RECONCILE_ATTEMPTS:
                time.sleep(self._RECONCILE_DELAY)
        return state

    def _create_commit(
        self,
        source_fd: int,
        remote_path: str,
        parent: str,
    ) -> None:
        """Submit one descriptor-anchored operation with an exact parent."""

        try:
            from huggingface_hub import CommitOperationAdd  # type: ignore[import-not-found]
        except (ImportError, ModuleNotFoundError):
            raise PublishError("huggingface_hub is required for CAS publication") from None
        # ``hf_xet`` accelerates only path-backed operations.  Anchor that
        # pathname to a duplicate of the already-owned inode through procfs:
        # replacing ``source_path`` cannot redirect the upload, while the
        # string path keeps the Xet transfer path enabled.  Fail closed on a
        # platform without a usable descriptor filesystem instead of falling
        # back to the much slower BinaryIO/HTTP uploader.
        try:
            upload_fd = os.dup(source_fd)
        except OSError:
            raise PublishError("could not duplicate the owned upload descriptor") from None
        try:
            descriptor_path = f"/proc/self/fd/{upload_fd}"
            if not os.path.isfile(descriptor_path):
                raise PublishError("descriptor-backed Xet upload path is unavailable")
            operation = CommitOperationAdd(
                path_in_repo=remote_path,
                path_or_fileobj=descriptor_path,
            )
            self._api_client().create_commit(
                repo_id=self.repository,
                operations=[operation],
                commit_message="ouro-jlens: publish immutable artifact",
                repo_type="model",
                revision="main",
                parent_commit=parent,
            )
        finally:
            os.close(upload_fd)

    def put_file(self, source: Path | _OwnedTempFile, remote_path: str) -> None:
        remote_path = safe_relative(remote_path)
        owned = isinstance(source, _OwnedTempFile)
        source_path = source.path if owned else Path(source)
        source_fd = source.fd if owned else _open_regular(source_path, label="upload source")
        upload_snapshot: _OwnedTempFile | None = None
        try:
            if (
                owned
                and source.sealed_readonly
                and stat.S_IMODE(os.fstat(source_fd).st_mode) == 0o400
                and fcntl.fcntl(source_fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
            ):
                upload_source = source
                _assert_owned_path(
                    upload_source.fd,
                    upload_source.path,
                    label="sealed upload",
                )
            else:
                upload_snapshot = _snapshot_upload_source(source_fd, source_path)
                upload_source = upload_snapshot
            expected_size, expected_digest = _digest_fd(upload_source.fd)
            # Snapshot the branch first, then inspect the immutable target at
            # that exact commit. A rival write after this snapshot forces the
            # parent-commit precondition to fail instead of being overwritten.
            parent = self._repository_head()
            initial_state = self._target_state(
                remote_path,
                expected_size,
                expected_digest,
                revision=parent,
            )
            try:
                _assert_owned_path(
                    upload_source.fd,
                    upload_source.path,
                    label="sealed upload",
                )
                initial_size, initial_digest = _digest_fd(upload_source.fd)
            except PublishError:
                raise PublishError("upload source changed while being published") from None
            if (initial_size, initial_digest) != (expected_size, expected_digest):
                raise PublishError("upload source changed while being published")
            if initial_state == "matching":
                return
            if initial_state == "different":
                raise ImmutableConflict(
                    f"immutable remote path already differs: {remote_path}"
                )
            if initial_state == "unknown":
                raise PublishError("could not inspect remote immutable target")
            _assert_owned_path(
                upload_source.fd,
                upload_source.path,
                label="sealed upload",
            )
            current_size, current_digest = _digest_fd(upload_source.fd)
            if (current_size, current_digest) != (expected_size, expected_digest):
                raise PublishError("upload source changed while being published")
            try:
                self._create_commit(upload_source.fd, remote_path, parent)
            except Exception as exc:
                # Every failed mutation is reconciled with bounded read-only
                # target reads. In particular, a 412 parent conflict must not
                # trigger another mutation merely because the target is still
                # missing from a stale listing/read.
                state = self._reconcile_target(
                    remote_path, expected_size, expected_digest
                )
                try:
                    _assert_owned_path(
                        upload_source.fd,
                        upload_source.path,
                        label="sealed upload",
                    )
                    reconciled_size, reconciled_digest = _digest_fd(upload_source.fd)
                except PublishError:
                    raise PublishError(
                        "upload source changed while being published"
                    ) from None
                if (reconciled_size, reconciled_digest) != (expected_size, expected_digest):
                    raise PublishError(
                        "upload source changed while being published"
                    ) from None
                if state == "matching":
                    return
                if state == "different":
                    raise ImmutableConflict(
                        f"immutable remote path already differs: {remote_path}"
                    ) from None
                if self._is_parent_conflict(exc):
                    raise PublishError(
                        "remote parent conflict could not be resolved without a second mutation"
                    ) from None
                raise PublishError(
                    "remote commit outcome could not be reconciled"
                ) from None
            # A successful API response is a known commit. Verify that the
            # caller's descriptor remained the exact inode/content it
            # submitted before accepting the mutation.
            _assert_owned_path(
                upload_source.fd,
                upload_source.path,
                label="sealed upload",
            )
            final_size, final_digest = _digest_fd(upload_source.fd)
            if (final_size, final_digest) != (expected_size, expected_digest):
                raise PublishError("upload source changed while being published")
            return
        finally:
            if upload_snapshot is not None:
                upload_snapshot.cleanup()
            if not owned:
                os.close(source_fd)

    def read_file(
        self,
        remote_path: str,
        *,
        expected_size: int | None = None,
        revision: str | None = None,
    ) -> bytes:
        temporary = _exclusive_temp_bytes(
            b"", directory=Path(tempfile.gettempdir()), prefix="jlens-hf-read-", suffix=".tmp"
        )
        try:
            self._download_file_to_fd(
                remote_path,
                temporary.fd,
                expected_size=expected_size,
                revision=revision,
            )
            return _read_fd(temporary.fd)
        finally:
            temporary.cleanup()

    def download_file(
        self,
        remote_path: str,
        destination: Path,
        *,
        expected_size: int | None = None,
    ) -> None:
        remote_path = safe_relative(remote_path)
        destination = Path(destination)
        with tempfile.TemporaryDirectory(prefix="jlens-hf-get-") as tmp:
            try:
                self._run(
                    ["hf", "download", self.repository, remote_path, "--local-dir", tmp],
                    timeout=self._transfer_timeout(expected_size),
                )
            except _RemoteCommandFailure as exc:
                if self._entry_definitively_missing(remote_path):
                    raise MissingRemote(remote_path) from exc
                raise
            candidates = [Path(tmp) / remote_path, Path(tmp) / Path(remote_path).name]
            for candidate in candidates:
                if candidate.is_file():
                    if expected_size is not None and candidate.stat().st_size != expected_size:
                        raise ImmutableConflict(f"remote payload size differs: {remote_path}")
                    _install_immutable(candidate, destination)
                    return
            if self._entry_definitively_missing(remote_path):
                raise MissingRemote(remote_path)
            raise PublishError("publisher download produced no regular target")

    def _download_file_to_fd(
        self,
        remote_path: str,
        destination_fd: int,
        *,
        expected_size: int | None = None,
        revision: str | None = None,
    ) -> None:
        """Download through a private CLI directory, then copy one stable fd."""

        remote_path = safe_relative(remote_path)
        if revision is not None and (
            not isinstance(revision, str) or GIT_OID_RE.fullmatch(revision) is None
        ):
            raise PublishError("Hugging Face download revision is malformed")
        with tempfile.TemporaryDirectory(prefix="jlens-hf-get-") as tmp:
            try:
                command = ["hf", "download", self.repository, remote_path]
                if revision is not None:
                    command.extend(["--revision", revision])
                command.extend(["--local-dir", tmp])
                self._run(
                    command,
                    timeout=self._transfer_timeout(expected_size),
                )
            except _RemoteCommandFailure as exc:
                if self._entry_definitively_missing(
                    remote_path,
                    revision=revision,
                ):
                    raise MissingRemote(remote_path) from exc
                raise
            candidates = [Path(tmp) / remote_path, Path(tmp) / Path(remote_path).name]
            for candidate in candidates:
                if candidate.is_file() and not candidate.is_symlink():
                    if expected_size is not None and candidate.stat().st_size != expected_size:
                        raise ImmutableConflict(f"remote payload size differs: {remote_path}")
                    source_fd = _open_regular(candidate, label="remote payload")
                    try:
                        _copy_fd(source_fd, destination_fd)
                        os.fsync(destination_fd)
                    finally:
                        os.close(source_fd)
                    return
            if self._entry_definitively_missing(
                remote_path,
                revision=revision,
            ):
                raise MissingRemote(remote_path)
            raise PublishError("publisher download produced no regular target")

    def list_files(self, prefix: str = "") -> list[str]:
        """List regular repository files below ``prefix``.

        Receipt discovery cannot safely infer a complete run from a terminal
        index alone.  ``hf``'s CLI has no stable machine-readable listing
        contract, so enumeration uses the official lazy Python API.  Any
        import, authentication, transport, pagination, or malformed-response
        failure is raised as ``PublishError``; returning an empty list would
        falsely turn an unavailable repository into a successful empty run.
        """

        raw_prefix = str(prefix).replace("\\", "/")
        if raw_prefix.startswith("/"):
            raise PublishError("invalid repository listing prefix")
        clean_prefix = raw_prefix.rstrip("/")
        if clean_prefix:
            try:
                clean_prefix = safe_relative(clean_prefix)
            except ValueError as exc:
                raise PublishError("invalid repository listing prefix") from exc
        api = self._api
        if api is None:
            try:
                from huggingface_hub import HfApi  # type: ignore[import-not-found]
            except (ImportError, ModuleNotFoundError) as exc:
                raise PublishError("huggingface_hub is required to enumerate repository files") from exc
            try:
                api = HfApi(token=self.token)
            except Exception as exc:
                raise PublishError("could not initialize huggingface_hub API") from exc
        try:
            tree = api.list_repo_tree(
                repo_id=self.repository,
                path_in_repo=clean_prefix or None,
                recursive=True,
            )
            items = list(tree)
        except Exception as exc:
            raise PublishError("could not enumerate Hugging Face repository files") from exc
        files: list[str] = []
        expected_prefix = f"{clean_prefix}/" if clean_prefix else ""
        for item in items:
            if isinstance(item, str):
                name = item
                item_type = "file"
                class_name = "file"
            else:
                name = None
                item_type = getattr(item, "type", None)
                class_name = type(item).__name__.lower()
            if isinstance(item, dict):
                item_type = item.get("type", item_type)
                class_name = str(item.get("type", class_name)).lower()
            if item_type in {"directory", "folder", "tree"} or "folder" in class_name:
                continue
            # Current huggingface_hub RepoFile objects expose ``path``;
            # older releases and test doubles commonly use ``rfilename``.
            if name is None:
                name = getattr(item, "path", None)
                if name is None:
                    name = getattr(item, "rfilename", None)
                if name is None and isinstance(item, dict):
                    name = item.get("path", item.get("rfilename"))
            if name is None:
                raise PublishError("Hugging Face listing contained an unnamed item")
            if not isinstance(name, str):
                raise PublishError("Hugging Face listing contained a malformed file name")
            try:
                name = safe_relative(name)
            except ValueError as exc:
                raise PublishError("Hugging Face listing contained an unsafe file name") from exc
            if expected_prefix and not name.startswith(expected_prefix):
                raise PublishError("Hugging Face listing escaped its requested prefix")
            files.append(name)
        return sorted(set(files))

    def verify_remote_receipts_metadata(
        self, receipts: Iterable[Any]
    ) -> Mapping[str, Any]:
        """Verify deferred Xet/LFS payload identities at one exact repo commit.

        Receipts are historical acknowledgements.  Before the controller seals
        an application-only extraction, this method proves that every deferred
        large payload is still present with the receipt's size and LFS SHA-256,
        without downloading the tensor bytes.
        """

        identities: dict[str, tuple[int, str]] = {}
        for receipt in receipts:
            remote_path, size, digest = _remote_receipt_identity(receipt)
            if remote_path in identities:
                raise PublishError("remote custody receipt paths are duplicated")
            identities[remote_path] = (size, digest)
        revision = self._repository_head()
        token: str | bool = self.token if self.token is not None else False
        observed: dict[str, tuple[int, str]] = {}
        api = self._api_client()
        requested = sorted(identities)
        try:
            for start in range(0, len(requested), 500):
                batch = requested[start:start + 500]
                entries = api.get_paths_info(
                    repo_id=self.repository,
                    paths=batch,
                    repo_type="model",
                    revision=revision,
                    token=token,
                )
                if not isinstance(entries, list):
                    raise PublishError("Hugging Face path metadata response is malformed")
                for entry in entries:
                    if isinstance(entry, Mapping):
                        path = entry.get("path", entry.get("rfilename"))
                        size = entry.get("size")
                        lfs = entry.get("lfs")
                    else:
                        path = getattr(entry, "path", None)
                        if path is None:
                            path = getattr(entry, "rfilename", None)
                        size = getattr(entry, "size", None)
                        lfs = getattr(entry, "lfs", None)
                    try:
                        path = safe_relative(path)
                    except (TypeError, ValueError) as exc:
                        raise PublishError(
                            "Hugging Face path metadata contained an unsafe path"
                        ) from exc
                    if path not in identities or path in observed:
                        raise PublishError(
                            "Hugging Face path metadata did not match the exact request"
                        )
                    lfs_digest = (
                        lfs.get("sha256") if isinstance(lfs, Mapping)
                        else getattr(lfs, "sha256", None)
                    )
                    if (isinstance(size, bool) or not isinstance(size, int) or size < 0
                            or not isinstance(lfs_digest, str)
                            or SHA256_RE.fullmatch(lfs_digest) is None):
                        raise PublishError(
                            "Hugging Face path metadata omitted an exact LFS byte identity"
                        )
                    observed[path] = (size, lfs_digest)
        except PublishError:
            raise
        except Exception:
            raise PublishError("could not verify deferred Hugging Face payload metadata") from None
        missing = set(identities) - set(observed)
        if missing:
            raise MissingRemote(
                f"deferred remote payload metadata is missing: {sorted(missing)[0]}"
            )
        for path, expected in identities.items():
            if observed[path] != expected:
                raise ImmutableConflict(
                    f"deferred remote payload differs from its receipt: {path}"
                )
        return {"revision": revision, "count": len(identities)}


@dataclass(frozen=True)
class Receipt:
    schema_version: int
    run_id: str
    relative_path: str
    remote_path: str
    kind: str
    sha256: str
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "relative_path": self.relative_path,
            "remote_path": self.remote_path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size": self.size,
        }

    @classmethod
    def from_bytes(cls, data: bytes) -> "Receipt":
        try:
            obj = json.loads(data)
            if not isinstance(obj, dict):
                raise TypeError("receipt is not an object")
            if (not isinstance(obj.get("schema_version"), int)
                    or isinstance(obj.get("schema_version"), bool)
                    or not isinstance(obj.get("run_id"), str)
                    or not isinstance(obj.get("relative_path"), str)
                    or not isinstance(obj.get("remote_path"), str)
                    or not isinstance(obj.get("kind"), str)
                    or not isinstance(obj.get("sha256"), str)
                    or not isinstance(obj.get("size"), int)
                    or isinstance(obj.get("size"), bool)):
                raise TypeError("receipt fields have invalid types")
            receipt = cls(
                schema_version=obj["schema_version"],
                run_id=obj["run_id"],
                relative_path=safe_relative(obj["relative_path"]),
                remote_path=safe_relative(obj["remote_path"]),
                kind=obj["kind"],
                sha256=obj["sha256"],
                size=obj["size"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PublishError("malformed artifact receipt") from exc
        if receipt.schema_version != SCHEMA_VERSION or not SHA256_RE.fullmatch(receipt.sha256):
            raise PublishError("unsupported or malformed artifact receipt")
        validate_run_id(receipt.run_id)
        expected = payload_remote_path(receipt.run_id, receipt.relative_path)
        if receipt.remote_path != expected or receipt.size < 0 or not receipt.kind:
            raise PublishError("receipt path or size does not match its run")
        return receipt


def make_receipt(source: str | Path, run_id: str, relative_path: str | Path, kind: str) -> Receipt:
    source = Path(source)
    source_fd = _open_regular(source, label="receipt source")
    try:
        return _make_receipt_from_fd(source, source_fd, run_id, relative_path, kind)
    finally:
        os.close(source_fd)


def _make_receipt_from_fd(
    source: Path, source_fd: int, run_id: str, relative_path: str | Path, kind: str
) -> Receipt:
    relative = safe_relative(relative_path)
    validate_run_id(run_id)
    if not kind or any(c in kind for c in "\r\n"):
        raise ValueError("invalid artifact kind")
    size, digest = _digest_fd(source_fd)
    return Receipt(
        SCHEMA_VERSION,
        run_id,
        relative,
        payload_remote_path(run_id, relative),
        kind,
        digest,
        size,
    )


def _receipt_root_path(receipt_root: str | Path | None, receipt: Receipt) -> Path | None:
    if receipt_root is None:
        return None
    return Path(receipt_root) / f"{receipt.relative_path}.receipt.json"


def _check_ack(receipt: Receipt, data: bytes) -> None:
    acknowledged = Receipt.from_bytes(data)
    if acknowledged != receipt:
        raise ImmutableConflict(f"publisher acknowledged a different receipt for {receipt.relative_path}")


def _download_remote(
    publisher: Publisher,
    remote_path: str,
    destination: _OwnedTempFile,
    *,
    expected_size: int | None = None,
) -> None:
    """Download into a descriptor whose pathname remains owned by the caller.

    Built-in publishers provide a descriptor-aware path.  Older injected
    publishers are handled through ``read_file`` and a direct descriptor write;
    this keeps their public API working without reopening a temporary name.
    """

    download_to_fd = getattr(publisher, "_download_file_to_fd", None)
    if callable(download_to_fd):
        # Preserve injected/older publishers whose descriptor helper accepts
        # only the original two positional arguments. Built-in publishers
        # advertise ``expected_size`` and use it to derive a transfer timeout.
        try:
            parameters = inspect.signature(download_to_fd).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_size = (
            "expected_size" in parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        if expected_size is not None and accepts_size:
            download_to_fd(remote_path, destination.fd, expected_size=expected_size)
        else:
            download_to_fd(remote_path, destination.fd)
        return
    download_file = getattr(publisher, "download_file", None)
    if callable(download_file):
        # Preserve the original public Publisher API without handing a
        # possibly replaceable pathname directly to the caller.  Download to
        # an absent path in a private directory, then copy that inode through
        # one no-follow descriptor into the owned destination.
        try:
            parameters = inspect.signature(download_file).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_size = (
            "expected_size" in parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        with tempfile.TemporaryDirectory(prefix="jlens-download-") as tmp:
            candidate = Path(tmp) / "payload"
            if expected_size is not None and accepts_size:
                download_file(remote_path, candidate, expected_size=expected_size)
            else:
                download_file(remote_path, candidate)
            source_fd = _open_regular(candidate, label="downloaded payload")
            try:
                if expected_size is not None and os.fstat(source_fd).st_size != expected_size:
                    raise ImmutableConflict(f"remote payload size differs: {remote_path}")
                _copy_fd(source_fd, destination.fd)
                os.fsync(destination.fd)
            finally:
                os.close(source_fd)
        return
    data = publisher.read_file(remote_path)
    os.lseek(destination.fd, 0, os.SEEK_SET)
    os.ftruncate(destination.fd, 0)
    _write_fd(destination.fd, data)
    os.fsync(destination.fd)


def _download_remote_expected(
    publisher: Publisher,
    remote_path: str,
    destination: _OwnedTempFile,
    expected_size: int,
) -> None:
    """Call the descriptor downloader with size when its API supports it.

    The indirection also keeps tests and older injected callers that
    monkeypatch the pre-size ``_download_remote`` helper source-compatible.
    """

    try:
        parameters = inspect.signature(_download_remote).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_size = (
        "expected_size" in parameters
        or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    )
    if accepts_size:
        _download_remote(
            publisher, remote_path, destination, expected_size=expected_size
        )
    else:
        _download_remote(publisher, remote_path, destination)


def _verify_remote_payload(
    publisher: Publisher,
    receipt: Receipt,
    *,
    attempts: int = 1,
    retry_seconds: float = 0.25,
) -> None:
    last_error: PublishError | None = None
    for attempt in range(attempts):
        destination = _exclusive_temp_bytes(
            b"", directory=Path(tempfile.gettempdir()), prefix="jlens-verify-", suffix=".tmp"
        )
        try:
            _download_remote_expected(
                publisher,
                receipt.remote_path,
                destination,
                receipt.size,
            )
            size, digest = _digest_fd(destination.fd)
            if size != receipt.size or digest != receipt.sha256:
                raise ImmutableConflict(f"remote payload differs: {receipt.remote_path}")
            return
        except ImmutableConflict:
            raise
        except FileNotFoundError as exc:
            last_error = PublishError(f"remote payload is missing: {receipt.remote_path}")
            last_error.__cause__ = exc
        except PublishError as exc:
            last_error = exc
        finally:
            destination.cleanup()
        if attempt + 1 < attempts:
            time.sleep(retry_seconds)
    if last_error is not None:
        raise last_error
    raise PublishError(f"remote payload could not be verified: {receipt.remote_path}")


def _read_remote_ack(
    publisher: Publisher,
    remote_path: str,
    *,
    attempts: int = 8,
    retry_seconds: float = 0.25,
) -> bytes:
    last_error: PublishError | None = None
    for attempt in range(attempts):
        try:
            return publisher.read_file(remote_path)
        except PublishError as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(retry_seconds)
    raise PublishError(f"remote upload was not acknowledged: {remote_path}") from last_error


def publish_file(
    publisher: Publisher,
    source: str | Path,
    *,
    run_id: str,
    relative_path: str | Path,
    kind: str,
    receipt_root: str | Path | None = None,
) -> Receipt:
    """Publish one file, then read back and verify its immutable receipt.

    This function is idempotent across a crash between payload and receipt
    upload.  If the remote receipt already exists, the payload is never
    overwritten; it is accepted only when its digest exactly matches.
    """

    source = Path(source)
    source_fd = _open_regular(source, label="receipt source")
    stable_source: _OwnedTempFile | None = None
    try:
        stable_source = _snapshot_upload_source(source_fd, source)
        receipt = _make_receipt_from_fd(
            stable_source.path,
            stable_source.fd,
            run_id,
            relative_path,
            kind,
        )
        receipt_path = receipt_remote_path(run_id, receipt.relative_path)
        try:
            existing_data = publisher.read_file(receipt_path)
        except MissingRemote:
            existing_data = None
        if existing_data is not None:
            existing = Receipt.from_bytes(existing_data)
            if existing != receipt:
                raise ImmutableConflict(f"existing receipt differs: {receipt_path}")
            # The ACK is the receipt read itself, but validate the payload too.
            try:
                _verify_remote_payload(publisher, receipt, attempts=8)
            except MissingRemote as exc:
                raise PublishError(f"receipt exists but payload is missing: {receipt.remote_path}") from exc
        else:
            # A crash can happen after payload upload but before receipt upload.
            # Never overwrite that orphaned payload: bind it by digest and only
            # create the missing receipt if it is exactly the same artifact.
            try:
                _verify_remote_payload(publisher, receipt)
            except MissingRemote:
                _assert_owned_path(
                    stable_source.fd,
                    stable_source.path,
                    label="sealed upload",
                )
                publisher.put_file(stable_source, receipt.remote_path)
                # Bind the receipt to the bytes actually present remotely before
                # publishing the receipt itself.  An upload that returned success
                # but stored truncated or changed bytes is never acknowledged.
                _verify_remote_payload(publisher, receipt, attempts=8)
            # An orphaned payload already matching the receipt needs no upload.
            receipt_tmp = _exclusive_temp_bytes(
                canonical_json(receipt.as_dict()),
                directory=Path(tempfile.gettempdir()),
                prefix="jlens-receipt-",
                suffix=".json",
            )
            try:
                _assert_owned_path(receipt_tmp.fd, receipt_tmp.path, label="receipt upload")
                publisher.put_file(receipt_tmp, receipt_path)
            finally:
                receipt_tmp.cleanup()
            # A successful upload is not an acknowledgment until read back.
            ack_data = _read_remote_ack(publisher, receipt_path)
            _check_ack(receipt, ack_data)

        local_receipt = _receipt_root_path(receipt_root, receipt)
        if local_receipt is not None:
            _write_immutable(local_receipt, canonical_json(receipt.as_dict()))
        return receipt
    finally:
        os.close(source_fd)
        if stable_source is not None:
            stable_source.cleanup()


def verify_local_receipt(source: str | Path, receipt_path: str | Path, *, run_id: str | None = None) -> Receipt:
    source = Path(source)
    receipt = Receipt.from_bytes(_read_regular_file(Path(receipt_path)))
    if run_id is not None and receipt.run_id != validate_run_id(run_id):
        raise PublishError("receipt belongs to another run")
    size, digest = _path_digest(source, label="receipt payload")
    if size != receipt.size or digest != receipt.sha256:
        raise PublishError(f"receipt digest mismatch: {source}")
    return receipt


def restore_file(
    publisher: Publisher,
    *,
    run_id: str,
    relative_path: str | Path,
    destination: str | Path,
) -> Receipt | None:
    """Restore one previously acknowledged payload after a worker restart.

    ``None`` means the receipt is not present yet.  Any other failure is
    raised, so a transient/network error cannot be mistaken for a missing
    shard and trigger an unsafe recomputation.
    """

    run_id = validate_run_id(run_id)
    relative = safe_relative(relative_path)
    try:
        receipt = Receipt.from_bytes(publisher.read_file(receipt_remote_path(run_id, relative)))
    except MissingRemote:
        return None
    destination = Path(destination)
    _reject_symlink_ancestors(destination)
    if destination.is_symlink():
        raise PublishError(f"restore destination is a symlink: {destination}")
    if destination.exists():
        size, digest = _path_digest(destination, label="restore destination")
        if size != receipt.size or digest != receipt.sha256:
            raise ImmutableConflict(f"restore destination differs: {destination}")
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(destination)
    temporary = _exclusive_temp_bytes(
        b"", directory=destination.parent, prefix=".jlens-restore-", suffix=".tmp"
    )
    try:
        _download_remote_expected(
            publisher,
            receipt.remote_path,
            temporary,
            receipt.size,
        )
        size, digest = _digest_fd(temporary.fd)
        if size != receipt.size or digest != receipt.sha256:
            raise PublishError(f"remote digest mismatch: {receipt.remote_path}")
        _install_owned_immutable(temporary, destination)
    finally:
        temporary.cleanup()
    return receipt


def write_sidecar(
    source: str | Path,
    output: str | Path,
    *,
    run_id: str,
    relative_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Create a digest-bound sidecar when a worker crashed before metadata.

    Existing sidecars are immutable: a different document is never silently
    replaced.  This is intentionally small and records only facts available
    after a restart; it does not invent model or fit results.
    """

    source = Path(source)
    output = Path(output)
    _reject_symlink_ancestors(output)
    receipt = make_receipt(source, run_id, relative_path, "artifact")
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "relative_path": receipt.relative_path,
        "sha256": receipt.sha256,
        "size": receipt.size,
    }
    if metadata:
        reserved = {"schema_version", "run_id", "relative_path", "sha256", "size"}
        overlap = sorted(reserved.intersection(metadata))
        if overlap:
            raise ValueError(f"sidecar metadata cannot override {', '.join(overlap)}")
        value.update(dict(metadata))
    data = canonical_json(value)
    # Sidecars are immutable protocol facts.  Always use the exclusive
    # installer, including when the destination appears to exist: its
    # FileExists path compares the installed inode without a check-then-act
    # race, and accepts an existing byte-identical document.
    _write_immutable(output, data)
    return output


def publish_tree(
    publisher: Publisher,
    root: str | Path,
    *,
    run_id: str,
    relative_prefix: str = "",
    kind: str = "artifact",
    receipt_root: str | Path | None = None,
) -> list[Receipt]:
    """Publish every regular file below ``root``.

    There is deliberately no extension, size, checkpoint, or shard
    exclusion.  Caller-owned receipt caches live outside ``root``.
    """

    root = Path(root)
    _reject_symlink_ancestors(root)
    if root.is_symlink() or not root.is_dir():
        raise PublishError(f"artifact tree is missing: {root}")
    prefix = safe_relative(relative_prefix) if relative_prefix else ""
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PublishError(f"artifact tree contains a symlink: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        files.append((path, f"{prefix}/{rel}" if prefix else rel))
    receipts = []
    for path, relative in sorted(files, key=lambda item: item[1]):
        receipts.append(publish_file(publisher, path, run_id=run_id, relative_path=relative,
                                     kind=kind, receipt_root=receipt_root))
    return receipts


def _bounded_sync_relative(path: str | Path) -> str:
    """Normalize one sync path while enforcing resource-facing limits."""

    if not isinstance(path, (str, Path)):
        raise PublishError("sync path is not a string or path")
    try:
        relative = safe_relative(path)
    except (TypeError, ValueError) as exc:
        raise PublishError("publisher returned an unsafe sync path") from exc
    if len(relative.encode("utf-8")) > MAX_SYNC_RELATIVE_PATH_BYTES:
        raise PublishError("sync path exceeds the maximum permitted length")
    return relative


def _bounded_sync_paths(paths: Iterable[str | Path]) -> list[str]:
    """Return unique sorted sync paths without materializing an unbounded set."""

    try:
        iterator = iter(paths)
    except TypeError as exc:
        raise PublishError("sync paths are not iterable") from exc
    result: set[str] = set()
    path_bytes = 0
    try:
        for raw_path in iterator:
            relative = _bounded_sync_relative(raw_path)
            if relative in result:
                continue
            if len(result) >= MAX_SYNC_RECEIPTS:
                raise PublishError("sync receipt count exceeds the permitted limit")
            path_bytes += len(relative.encode("utf-8"))
            if path_bytes > MAX_SYNC_PATH_BYTES:
                raise PublishError("sync path metadata exceeds the permitted limit")
            result.add(relative)
    except PublishError:
        raise
    except Exception as exc:
        raise PublishError("sync path enumeration failed") from exc
    return sorted(result)


def sync_and_verify(
    publisher: Publisher,
    *,
    run_id: str,
    local_root: str | Path,
    relative_paths: Iterable[str] | None = None,
    receipt_root: str | Path | None = None,
) -> list[Receipt]:
    """Fetch acknowledged artifacts and verify every digest locally.

    With ``relative_paths=None``, the publisher enumerates the complete direct
    receipt set and unions every versioned index generation before syncing;
    this is the controller-facing recovery interface.  A missing or malformed
    receipt is a hard failure.
    """

    run_id = validate_run_id(run_id)
    if relative_paths is None:
        relative_paths = discover_receipt_paths(publisher, run_id)
    else:
        # Keep callers written against the old singleton index API working:
        # an explicit ``receipt-index.json`` means the newest versioned
        # generation when the legacy path is absent.  New callers should pass
        # ``None`` and let discovery collect the complete union.
        requested = _bounded_sync_paths(relative_paths)
        if "receipt-index.json" in requested:
            try:
                publisher.read_file(receipt_remote_path(run_id, "receipt-index.json"))
            except MissingRemote:
                discovered = discover_receipt_paths(publisher, run_id)
                indexes = [path for path in discovered if is_receipt_index(path)]
                versioned = [path for path in indexes if path != "receipt-index.json"]
                if versioned:
                    requested = [
                        (sorted(versioned)[-1] if path == "receipt-index.json" else path)
                        for path in requested
                    ]
                else:
                    requested = [path for path in requested if path != "receipt-index.json"]
        relative_paths = requested
    local_root = Path(local_root)
    _reject_symlink_ancestors(local_root / "placeholder")
    if local_root.is_symlink():
        raise PublishError(f"local sync root is a symlink: {local_root}")
    local_root.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(local_root / "placeholder")
    local_receipt_root: Path | None = None
    if receipt_root is not None:
        local_receipt_root = Path(receipt_root)
        _reject_symlink_ancestors(local_receipt_root / "placeholder")
        if local_receipt_root.is_symlink():
            raise PublishError(f"local receipt root is a symlink: {local_receipt_root}")
        receipt_absolute = Path(os.path.abspath(local_receipt_root))
        payload_absolute = Path(os.path.abspath(local_root))
        if (receipt_absolute == payload_absolute or receipt_absolute in payload_absolute.parents
                or payload_absolute in receipt_absolute.parents):
            raise PublishError("local receipt root and payload root must not overlap")
    # Read and validate every receipt before starting any transfer.  This is
    # both a bounded-resource preflight and an identity check: a malformed or
    # oversized remote listing must fail before a partial local convergence.
    requested = _bounded_sync_paths(relative_paths)
    planned: list[tuple[str, Receipt, Path, bool]] = []
    total_declared = 0
    total_missing = 0
    largest_missing = 0
    for relative in requested:
        receipt_path = receipt_remote_path(run_id, relative)
        receipt = Receipt.from_bytes(publisher.read_file(receipt_path))
        if receipt.run_id != run_id or receipt.relative_path != relative:
            raise PublishError(f"receipt identity mismatch: {receipt_path}")
        if receipt.size > MAX_SYNC_TOTAL_BYTES - total_declared:
            raise PublishError("sync receipt bytes exceed the permitted limit")
        total_declared += receipt.size
        destination = local_root / relative
        _reject_symlink_ancestors(destination)
        if destination.is_symlink():
            raise PublishError(f"local destination is a symlink: {destination}")
        if destination.exists():
            local_size, local_digest = _path_digest(destination, label="local destination")
            if local_size != receipt.size or local_digest != receipt.sha256:
                raise ImmutableConflict(f"local destination differs: {destination}")
            planned.append((relative, receipt, destination, False))
            continue
        total_missing += receipt.size
        largest_missing = max(largest_missing, receipt.size)
        planned.append((relative, receipt, destination, True))

    if total_missing:
        required_free = total_missing + largest_missing
        try:
            available_free = shutil.disk_usage(local_root).free
        except OSError as exc:
            raise PublishError("could not determine free space before sync") from exc
        if available_free < required_free:
            raise PublishError(
                "insufficient free space for sync: "
                f"need {required_free} bytes, have {available_free}"
            )

    receipts: list[Receipt] = []
    for relative, receipt, destination, needs_download in planned:
        if not needs_download:
            if local_receipt_root is not None:
                local_receipt = local_receipt_root / f"{relative}.receipt.json"
                _write_immutable(local_receipt, canonical_json(receipt.as_dict()))
            receipts.append(receipt)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_ancestors(destination)
        temporary = _exclusive_temp_bytes(
            b"", directory=destination.parent, prefix=".jlens-sync-", suffix=".tmp"
        )
        try:
            _download_remote_expected(
                publisher,
                receipt.remote_path,
                temporary,
                receipt.size,
            )
            size, digest = _digest_fd(temporary.fd)
            if size != receipt.size or digest != receipt.sha256:
                raise PublishError(f"remote digest mismatch: {receipt.remote_path}")
            _install_owned_immutable(temporary, destination)
        finally:
            temporary.cleanup()
        if local_receipt_root is not None:
            local_receipt = local_receipt_root / f"{relative}.receipt.json"
            _write_immutable(local_receipt, canonical_json(receipt.as_dict()))
        receipts.append(receipt)
    return receipts


def _remote_receipt_relatives(publisher: Publisher, run_id: str) -> list[str]:
    """Enumerate and validate all receipt paths exposed by a publisher."""

    list_files = getattr(publisher, "list_files", None)
    if not callable(list_files):
        raise PublishError("publisher cannot enumerate receipts; relative_paths are required")
    prefix = f"{validate_run_id(run_id)}/receipts/"
    try:
        files = list_files(prefix)
        if isinstance(files, Mapping):
            raise PublishError("publisher returned a malformed receipt listing")
    except PublishError:
        raise
    except Exception as exc:
        raise PublishError("publisher receipt enumeration failed") from exc
    if isinstance(files, (str, bytes)):
        raise PublishError("publisher returned a malformed receipt listing")
    relatives: list[str] = []
    listing_count = 0
    try:
        iterator = iter(files)
        for path in iterator:
            listing_count += 1
            if listing_count > MAX_SYNC_RECEIPTS:
                raise PublishError("sync receipt listing exceeds the permitted limit")
            if not isinstance(path, str):
                raise PublishError("publisher returned a malformed receipt path")
            try:
                normalized = safe_relative(path)
            except ValueError as exc:
                raise PublishError("publisher returned an unsafe receipt path") from exc
            if not normalized.startswith(prefix) or not normalized.endswith(".receipt.json"):
                continue
            relative = normalized[len(prefix) : -len(".receipt.json")]
            relatives.append(_bounded_sync_relative(relative))
    except PublishError:
        raise
    except Exception as exc:
        raise PublishError("publisher receipt enumeration failed") from exc
    return _bounded_sync_paths(relatives)


def discover_receipt_paths(publisher: Publisher, run_id: str) -> list[str]:
    """Return the complete acknowledged-artifact union for one run.

    Every receipt generation is retained.  Versioned indexes are parsed and
    unioned with the directly enumerated receipt set, so a missing terminal
    index cannot hide earlier artifacts and a partial index cannot silently
    drop a receipt.  Malformed or unavailable index data is a hard failure.
    """

    run_id = validate_run_id(run_id)
    receipt_relatives = _remote_receipt_relatives(publisher, run_id)
    union = set(receipt_relatives)
    union_path_bytes = sum(len(path.encode("utf-8")) for path in union)
    for relative in receipt_relatives:
        if not is_receipt_index(relative):
            continue
        try:
            receipt = Receipt.from_bytes(
                publisher.read_file(receipt_remote_path(run_id, relative))
            )
            if (receipt.run_id != run_id or receipt.relative_path != relative
                    or receipt.kind != "receipt_index"):
                raise PublishError(f"receipt identity mismatch: {relative}")
            index_data = publisher.read_file(receipt.remote_path)
            if (relative != "receipt-index.json"
                    and relative != receipt_index_relative_path(index_data)):
                raise PublishError(f"receipt index content is not bound to its path: {relative}")
            for indexed_relative in read_index(index_data, run_id=run_id):
                if indexed_relative in union:
                    continue
                if len(union) >= MAX_SYNC_RECEIPTS:
                    raise PublishError("sync receipt count exceeds the permitted limit")
                union_path_bytes += len(indexed_relative.encode("utf-8"))
                if union_path_bytes > MAX_SYNC_PATH_BYTES:
                    raise PublishError("sync path metadata exceeds the permitted limit")
                union.add(indexed_relative)
        except MissingRemote as exc:
            raise PublishError(f"receipt index generation is incomplete: {relative}") from exc
    return _bounded_sync_paths(union)


def publish_index(
    publisher: Publisher,
    *,
    run_id: str,
    receipt_root: str | Path,
    local_root: str | Path,
) -> Receipt:
    """Publish one immutable cumulative index generation.

    The relative path includes the SHA-256 of the canonical contents, so a
    resumed run can publish a larger index without conflicting with an older
    generation.  The index itself is not included in its ``relative_paths``;
    its receipt is still discoverable directly from the receipt tree.
    """

    run_id = validate_run_id(run_id)
    receipt_root = Path(receipt_root)
    _reject_symlink_ancestors(receipt_root / "placeholder")
    if receipt_root.is_symlink():
        raise PublishError(f"receipt root is a symlink: {receipt_root}")
    suffix = ".receipt.json"
    relatives: list[str] = []
    if receipt_root.is_dir():
        for path in sorted(receipt_root.rglob(f"*{suffix}")):
            relative_receipt = path.relative_to(receipt_root).as_posix()
            _reject_symlink_ancestors(path)
            if path.is_symlink():
                raise PublishError(f"local receipt is a symlink: {path}")
            if not path.is_file():
                continue
            relative = relative_receipt[:-len(suffix)]
            try:
                receipt = Receipt.from_bytes(_read_regular_file(path))
            except (OSError, PublishError) as exc:
                raise PublishError(f"cannot index malformed local receipt: {path}") from exc
            if receipt.run_id != run_id or receipt.relative_path != relative:
                raise PublishError(f"local receipt identity mismatch: {path}")
            if is_receipt_index(relative):
                if receipt.kind != "receipt_index":
                    raise PublishError(f"local receipt index kind is malformed: {path}")
                continue
            relatives.append(relative)
    relatives.sort()
    payload = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "relative_paths": relatives}
    data = canonical_json(payload)
    index_relative = receipt_index_relative_path(data)
    local_root = Path(local_root)
    _reject_symlink_ancestors(local_root / "placeholder")
    if local_root.is_symlink():
        raise PublishError(f"index local root is a symlink: {local_root}")
    local_root.mkdir(parents=True, exist_ok=True)
    index = local_root / index_relative
    _write_immutable(index, data)
    return publish_file(publisher, index, run_id=run_id, relative_path=index_relative,
                        kind="receipt_index", receipt_root=receipt_root)


def read_index(data: bytes, *, run_id: str) -> list[str]:
    """Validate a terminal receipt index and return its relative paths."""

    run_id = validate_run_id(run_id)
    try:
        value = json.loads(data)
        if value.get("schema_version") != SCHEMA_VERSION or value.get("run_id") != run_id:
            raise ValueError("index identity mismatch")
        paths = value["relative_paths"]
        if not isinstance(paths, list):
            raise ValueError("index paths are not a list")
        result = _bounded_sync_paths(paths)
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PublishError("malformed receipt index") from exc
    if any(is_receipt_index(path) for path in result):
        raise PublishError("receipt index must not contain itself or another generation")
    return result


def _stage_payload_paths(root: Path) -> list[Path]:
    """Return the complete auditable source/input surface for a paid run.

    The runtime only needs the package and prompt corpus, but the stage also
    carries the canonical documentation and every JLens-specific unit test
    when they exist.  That keeps the exact contract and its executable checks
    beside the code that will run, without making synthetic test repositories
    manufacture files they do not have.
    """

    unit_tests = root / "utilities" / "tests" / "unit"
    optional_tests = sorted(
        {
            *unit_tests.glob("test_jlens*.py"),
            *unit_tests.glob("test_peer*_jlens*.py"),
            *(path for path in (unit_tests / "test_peer1_index_map.py",) if path.is_file()),
        },
        key=lambda path: path.name,
    ) if unit_tests.is_dir() else []
    roots = [
        root / "src" / "ouro_jlens",
        root / FITTING_CORPUS_RELATIVE,
        root / FITTING_CORPUS_PROVENANCE_RELATIVE,
        unit_tests / "test_ouro_jlens.py",
        *optional_tests,
    ]
    docs = root / "docs" / "jlens"
    if docs.is_dir():
        roots.append(docs)
    files: list[Path] = []
    for item in roots:
        if item.is_symlink():
            raise StageVerificationError(f"stage source is a symlink: {item}")
        if item.is_dir():
            for path in item.rglob("*"):
                # ``docs/jlens/watch`` is a live append-only operator channel,
                # not source or experimental evidence. It must remain writable
                # during a lease without changing the reviewed stage.
                if docs in path.parents and path.relative_to(docs).parts[:1] == ("watch",):
                    continue
                if path.is_symlink():
                    raise StageVerificationError(f"stage source contains a symlink: {path}")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    files.append(path)
        elif item.is_file():
            files.append(item)
        else:
            raise StageVerificationError(f"required stage input is missing: {item}")
    return sorted(set(files), key=lambda p: p.relative_to(root).as_posix())


def _git_pin(root: Path, payloads: Iterable[Path]) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", "-C", str(root), *args], check=False,
                                    capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    # Bind only repository-owned source, tests, and docs.  The prompt corpus is
    # an ignored artifact input whose bytes are bound by MANIFEST.sha256.
    tracked_payloads = sorted({
        path.relative_to(root).as_posix()
        for path in payloads
        if "artifacts" not in path.relative_to(root).parts
    })
    status = run("status", "--porcelain", "--untracked-files=all", "--", *tracked_payloads)
    head = run("rev-parse", "HEAD")
    committed_payload_sha256: dict[str, str] = {}
    if head:
        for relative in tracked_payloads:
            try:
                result = subprocess.run(
                    ["git", "-C", str(root), "show", f"{head}:{relative}"],
                    check=False,
                    capture_output=True,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if result.returncode == 0:
                committed_payload_sha256[relative] = sha256_bytes(result.stdout)
    return {
        "head": head,
        "status": status,
        "committed_payload_sha256": committed_payload_sha256,
    }


def _read_stable_file(path: Path) -> tuple[bytes, int]:
    """Read a regular source file from one inode and detect mid-read changes."""

    _reject_symlink_ancestors(path)
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise StageVerificationError(f"cannot read stage source: {path}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise StageVerificationError(f"stage source is not a regular file: {path}")
        try:
            data = _read_fd(fd)
        except PublishError as exc:
            raise StageVerificationError(f"stage source changed while being read: {path}") from exc
        after = os.fstat(fd)
        identity_before = (before.st_dev, before.st_ino, before.st_size,
                           before.st_mtime_ns, before.st_ctime_ns,
                           stat.S_IMODE(before.st_mode))
        identity_after = (after.st_dev, after.st_ino, after.st_size,
                          after.st_mtime_ns, after.st_ctime_ns,
                          stat.S_IMODE(after.st_mode))
        if identity_before != identity_after:
            raise StageVerificationError(f"stage source changed while being read: {path}")
        try:
            path_identity = _path_identity(path)
        except OSError as exc:
            raise StageVerificationError(f"stage source changed while being read: {path}") from exc
        if path_identity != _fd_identity(fd):
            raise StageVerificationError(f"stage source path changed while being read: {path}")
        return data, _safe_stage_mode(before.st_mode, directory=False, label=str(path))
    finally:
        os.close(fd)


def _safe_stage_mode(mode: int, *, directory: bool, label: str) -> int:
    """Reject special permission bits and return a deterministic safe mode."""

    if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0:
        raise StageVerificationError(f"stage member mode is malformed: {label}")
    if mode & STAGE_SPECIAL_MODE_BITS:
        raise StageVerificationError(f"stage member has special permission bits: {label}")
    permissions = stat.S_IMODE(mode)
    if directory:
        return SAFE_STAGE_DIRECTORY_MODE
    return SAFE_STAGE_EXECUTABLE_MODE if permissions & 0o111 else SAFE_STAGE_FILE_MODE


def _verify_stage_path_mode(path: Path, mode: int) -> None:
    """Require extracted stage entries to use the deterministic safe mode."""

    is_directory = stat.S_ISDIR(mode)
    if not (is_directory or stat.S_ISREG(mode)):
        raise StageVerificationError(f"stage path is not a regular file or directory: {path}")
    expected = _safe_stage_mode(mode, directory=is_directory, label=str(path))
    if stat.S_IMODE(mode) != expected:
        raise StageVerificationError(
            f"stage path mode is not deterministic: {path} "
            f"({stat.S_IMODE(mode):04o}, expected {expected:04o})"
        )


def _snapshot_regular_file(
    path: str | Path,
    *,
    directory: str | Path | None = None,
    prefix: str = "jlens-stage-snapshot-",
) -> _OwnedTempFile:
    """Copy one no-follow source fd into an owned, mutation-checked snapshot."""

    source_path = Path(path)
    source_fd = _open_regular(source_path, label="stage archive")
    try:
        try:
            source_identity = _path_identity(source_path)
        except OSError as exc:
            raise StageVerificationError(f"stage archive is unavailable: {source_path}") from exc
        if source_identity != _fd_identity(source_fd):
            raise StageVerificationError(f"stage archive path changed while being opened: {source_path}")
        snapshot_dir = Path(directory) if directory is not None else Path(tempfile.gettempdir())
        snapshot = _exclusive_temp_bytes(
            b"", directory=snapshot_dir, prefix=prefix, suffix=".tmp"
        )
        try:
            try:
                copied_size, copied_digest = _copy_fd(source_fd, snapshot.fd)
            except PublishError as exc:
                raise StageVerificationError(
                    f"stage archive changed while being snapshotted: {source_path}"
                ) from exc
            # A source can be changed while the first copy is in flight and
            # then have its timestamps restored.  Confirm the bytes from the
            # same no-follow descriptor before releasing the source fd; this
            # catches that case without reopening a possibly substituted
            # pathname.  Metadata-only hard-link/unlink churn is handled by
            # _digest_fd's bounded ctime confirmation.
            confirmed_size, confirmed_digest = _digest_fd(source_fd)
            if (confirmed_size, confirmed_digest) != (copied_size, copied_digest):
                raise StageVerificationError(
                    f"stage archive changed while being snapshotted: {source_path}"
                )
            os.lseek(snapshot.fd, 0, os.SEEK_SET)
            try:
                _assert_owned_path(source_fd, source_path, label="stage archive")
            except PublishError as exc:
                raise StageVerificationError(
                    f"stage archive path changed while being snapshotted: {source_path}"
                ) from exc
            return snapshot
        except BaseException:
            snapshot.cleanup()
            raise
    finally:
        os.close(source_fd)


def build_stage_tar(
    root: str | Path,
    output: str | Path,
    *,
    require_clean_source: bool = False,
) -> dict[str, Any]:
    """Build a stage archive with a payload manifest and pinned inputs."""

    raw_root = Path(root)
    _reject_symlink_ancestors(raw_root)
    if raw_root.is_symlink():
        raise StageVerificationError(f"stage root is a symlink: {raw_root}")
    root = raw_root.resolve()
    output = Path(output)
    _reject_symlink_ancestors(output)
    if output.is_symlink():
        raise StageVerificationError(f"stage output is a symlink: {output}")
    # Reject a locally forged or stale fitting corpus before snapshotting any
    # bytes.  Archive verification repeats this check after extraction, but a
    # producer must not be able to emit a self-consistent manifest around an
    # unpinned corpus in the first place.
    _verify_fitting_corpus(root)
    payloads = _stage_payload_paths(root)
    raw_source_pin = _git_pin(root, payloads)
    source_pin = {
        "head": raw_source_pin.get("head"),
        "status": raw_source_pin.get("status"),
    }
    committed_payload_sha256 = raw_source_pin.get("committed_payload_sha256")
    if require_clean_source:
        if source_pin["head"] is None:
            raise StageVerificationError("paid stage requires a git commit identity")
        if source_pin["status"] is None:
            raise StageVerificationError("paid stage could not verify scoped source status")
        if source_pin["status"]:
            raise StageVerificationError("paid stage source/tests/docs are not clean and committed")
        if not isinstance(committed_payload_sha256, dict):
            raise StageVerificationError("paid stage could not verify committed HEAD payload bytes")
        expected_committed = {
            path.relative_to(root).as_posix()
            for path in payloads
            if "artifacts" not in path.relative_to(root).parts
        }
        if set(committed_payload_sha256) != expected_committed:
            raise StageVerificationError("paid stage committed HEAD payload map is incomplete")
    snapshots: list[tuple[Path, str, bytes, int]] = []
    entries = []
    for path in payloads:
        rel = path.relative_to(root).as_posix()
        data, mode = _read_stable_file(path)
        digest = sha256_bytes(data)
        if (require_clean_source and "artifacts" not in PurePosixPath(rel).parts):
            if (not isinstance(committed_payload_sha256, dict)
                    or committed_payload_sha256.get(rel) != digest):
                raise StageVerificationError(
                    f"stage source bytes do not match committed HEAD blob: {rel}"
                )
        snapshots.append((path, rel, data, mode))
        entries.append({"path": rel, "sha256": digest, "size": len(data)})
    manifest_text = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries)
    manifest_digest = sha256_bytes(manifest_text.encode())
    pinned = {
        "schema_version": SCHEMA_VERSION,
        "payload_manifest_sha256": manifest_digest,
        "source": source_pin,
        "source_policy": "clean_git_payload" if require_clean_source else "snapshot_manifest_only",
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "jacobian_lens": {"repository": JLENS_REPOSITORY, "revision": JLENS_REVISION},
        "python": TARGET_PYTHON,
        "dependencies": RUNTIME_DEPENDENCIES,
    }
    if isinstance(committed_payload_sha256, dict):
        expected_committed = {
            path.relative_to(root).as_posix()
            for path in payloads
            if "artifacts" not in path.relative_to(root).parts
        }
        # Snapshot stages may intentionally include uncommitted files. Do not
        # publish a partial HEAD map that a verifier could mistake for a
        # complete binding; clean paid stages were required to be complete
        # above and therefore retain the map.
        if set(committed_payload_sha256) == expected_committed:
            pinned["committed_payload_sha256"] = dict(sorted(committed_payload_sha256.items()))
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(output)
    # Use an explicit gzip stream with a fixed mtime.  Archive payloads come
    # from the stable, HEAD-checked snapshots above, not a second path read
    # after the manifest was computed.  That closes the source archive TOCTOU.
    temporary = _exclusive_temp_bytes(
        b"", directory=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(temporary.fd, "wb", closefd=False) as raw_output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for _path, rel, data, mode in snapshots:
                        info = tarfile.TarInfo(rel)
                        info.mode = mode
                        info.size = len(data)
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        archive.addfile(info, io.BytesIO(data))
                    for name, data in (
                        ("MANIFEST.sha256", manifest_text.encode()),
                        ("PINNED_INPUTS.json", canonical_json(pinned)),
                    ):
                        info = tarfile.TarInfo(name)
                        info.mode = 0o644
                        info.size = len(data)
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        archive.addfile(info, io.BytesIO(data))
            raw_output.flush()
            os.fsync(raw_output.fileno())
        after_source_pin = _git_pin(root, payloads)
        after_identity = {
            "head": after_source_pin.get("head"),
            "status": after_source_pin.get("status"),
            "committed_payload_sha256": after_source_pin.get("committed_payload_sha256"),
        }
        before_identity = {
            "head": raw_source_pin.get("head"),
            "status": raw_source_pin.get("status"),
            "committed_payload_sha256": raw_source_pin.get("committed_payload_sha256"),
        }
        if after_identity != before_identity:
            raise StageVerificationError("stage source HEAD/status changed while archive was built")
        _replace_owned(temporary, output)
    finally:
        temporary.cleanup()
    return {"archive": str(output), "payload_count": len(entries), "manifest_sha256": manifest_digest,
            "pinned_inputs": pinned}


def _archive_member_name(member: tarfile.TarInfo) -> str:
    """Return one canonical relative name for an archive member."""

    raw_name = member.name
    if not isinstance(raw_name, str) or "\x00" in raw_name:
        raise StageVerificationError(f"unsafe stage archive path: {raw_name!r}")
    trimmed_name = raw_name[:-1] if member.isdir() and raw_name.endswith("/") else raw_name
    try:
        return safe_relative(trimmed_name)
    except ValueError as exc:
        # A root directory entry is harmless, but only the conventional root
        # spellings are accepted.  In particular, "/" must not be silently
        # normalized to the extraction root.
        if member.isdir() and raw_name in {"", ".", "./"}:
            return ""
        raise StageVerificationError(f"unsafe stage archive path: {member.name}") from exc


def _archive_target(destination: Path, member_name: str, member: tarfile.TarInfo) -> Path:
    """Validate one member target against the private extraction root."""

    target = destination if not member_name else destination / member_name
    try:
        _reject_symlink_ancestors(target)
    except PublishError as exc:
        raise StageVerificationError(f"stage archive path traverses a link: {member.name}") from exc
    if target.is_symlink():
        raise StageVerificationError(f"stage archive target is an existing link: {member.name}")
    resolved = target.resolve(strict=False)
    if destination != resolved and destination not in resolved.parents:
        raise StageVerificationError(f"stage archive path escapes root: {member.name}")
    return target


def _ensure_extract_parent(destination: Path, target: Path, created_dirs: set[str]) -> None:
    """Create missing parent directories without following an archive link."""

    relative_parent = target.parent.relative_to(destination)
    current = destination
    components: list[str] = []
    for component in relative_parent.parts:
        components.append(component)
        current = current / component
        try:
            _reject_symlink_ancestors(current)
        except PublishError as exc:
            raise StageVerificationError(f"stage archive path traverses a link: {current}") from exc
        if current.is_symlink():
            raise StageVerificationError(f"stage archive path traverses a link: {current}")
        try:
            current.mkdir(mode=SAFE_STAGE_DIRECTORY_MODE)
            # mkdir is affected by umask; bind the extracted directory to the
            # deterministic stage mode explicitly.
            os.chmod(current, SAFE_STAGE_DIRECTORY_MODE, follow_symlinks=False)
        except FileExistsError as exc:
            if current.is_symlink() or not current.is_dir():
                raise StageVerificationError(f"stage archive target collision: {current}") from exc
            relative = "/".join(components)
            if relative not in created_dirs:
                raise StageVerificationError(f"stage archive target collision: {current}") from exc
        except OSError as exc:
            raise StageVerificationError(f"cannot create stage archive directory: {current}") from exc
        else:
            created_dirs.add("/".join(components))


def _extract_regular_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    target: Path,
) -> None:
    """Materialize one regular member into a new staging inode."""

    if member.size < 0:
        raise StageVerificationError(f"negative stage archive member size: {member.name}")
    safe_mode = _safe_stage_mode(member.mode, directory=False, label=member.name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags, 0o600)
    except FileExistsError as exc:
        raise StageVerificationError(f"stage archive target collision: {member.name}") from exc
    except OSError as exc:
        raise StageVerificationError(f"cannot create stage archive file: {member.name}") from exc
    try:
        source = archive.extractfile(member)
        if source is None:
            raise StageVerificationError(f"stage archive member has no file data: {member.name}")
        try:
            remaining = member.size
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise StageVerificationError(
                        f"stage archive member is truncated: {member.name}"
                    )
                if len(chunk) > remaining:
                    raise StageVerificationError(
                        f"stage archive member is oversized: {member.name}"
                    )
                _write_fd(fd, chunk)
                remaining -= len(chunk)
            # TarFile's ExFileObject is bounded to the member size.  Reading a
            # byte nevertheless catches a custom/invalid file object that
            # exposes more data than the declared member.
            if source.read(1):
                raise StageVerificationError(f"stage archive member is oversized: {member.name}")
        finally:
            source.close()
        os.fchmod(fd, safe_mode)
        os.fsync(fd)
    except BaseException:
        raise
    finally:
        os.close(fd)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically install a directory while refusing a destination collision.

    Linux's ``renameat2(..., RENAME_NOREPLACE)`` is the only primitive that
    combines an atomic directory rename with a no-clobber guarantee.  The
    paid runtime is Linux; fail closed on a platform/kernel without that
    primitive instead of falling back to a check-then-rename race.
    """

    if not (sys.platform.startswith("linux") and os.name == "posix"):
        raise StageVerificationError("atomic no-replace directory install is unavailable")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2")
    except (AttributeError, OSError) as exc:
        raise StageVerificationError("atomic no-replace directory install is unavailable") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if source.parent != destination.parent:
        raise StageVerificationError("stage source and destination are not siblings")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(destination.parent, flags)
    except OSError as exc:
        raise StageVerificationError("stage destination parent is unavailable") from exc
    try:
        result = renameat2(
            parent_fd,
            os.fsencode(source.name),
            parent_fd,
            os.fsencode(destination.name),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise StageVerificationError(f"stage destination already exists: {destination}")
        raise OSError(error, os.strerror(error), os.fspath(destination))
    finally:
        os.close(parent_fd)


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Validate and materialize an archive into an empty private directory.

    No archive helper is allowed to recursively walk the destination.  Every
    member is classified first, then copied into an exclusive inode while its
    parent chain is checked.  The caller owns cleanup of this staging root if
    a later member or verification step fails.
    """

    raw_destination = Path(destination)
    if raw_destination.is_symlink():
        raise StageVerificationError(f"stage extraction root is a symlink: {raw_destination}")
    destination = raw_destination.resolve(strict=False)
    try:
        _reject_symlink_ancestors(destination)
    except PublishError as exc:
        raise StageVerificationError(f"stage extraction root has a symlinked parent: {destination}") from exc
    if destination.is_symlink() or not destination.is_dir():
        raise StageVerificationError(f"stage extraction root is not a directory: {destination}")
    try:
        if next(destination.iterdir(), None) is not None:
            raise StageVerificationError("stage extraction root must be empty")
    except OSError as exc:
        raise StageVerificationError("cannot inspect stage extraction root") from exc

    plans: list[tuple[tarfile.TarInfo, str]] = []
    seen_names: set[str] = set()
    try:
        members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise StageVerificationError("cannot enumerate stage archive") from exc
    if len(members) > MAX_STAGE_ARCHIVE_MEMBERS:
        raise StageVerificationError(
            f"stage archive has too many members: {len(members)} > "
            f"{MAX_STAGE_ARCHIVE_MEMBERS}"
        )
    expanded_bytes = 0
    for member in members:
        member_name = _archive_member_name(member)
        if member_name in seen_names:
            raise StageVerificationError(f"duplicate stage archive member: {member.name}")
        seen_names.add(member_name)
        if member.issym() or member.islnk():
            raise StageVerificationError(f"links are not allowed in stage archive: {member.name}")
        if not (member.isdir() or member.isfile()):
            raise StageVerificationError(f"special archive member is not allowed: {member.name}")
        if member.isfile() and (
            isinstance(member.size, bool) or not isinstance(member.size, int)
        ):
            raise StageVerificationError(f"malformed stage archive member size: {member.name}")
        if member.isfile() and member.size < 0:
            raise StageVerificationError(f"negative stage archive member size: {member.name}")
        if member.isfile():
            if member.size > MAX_STAGE_EXPANDED_BYTES - expanded_bytes:
                raise StageVerificationError(
                    f"stage archive expanded size exceeds {MAX_STAGE_EXPANDED_BYTES} bytes"
                )
            expanded_bytes += member.size
        _safe_stage_mode(member.mode, directory=member.isdir(), label=member.name)
        _archive_target(destination, member_name, member)
        plans.append((member, member_name))

    created_dirs: set[str] = set()
    directory_modes: dict[Path, int] = {}
    for member, member_name in plans:
        target = destination if not member_name else destination / member_name
        if member.isdir():
            if member_name:
                _ensure_extract_parent(destination, target, created_dirs)
                if target.is_symlink() or (target.exists() and not target.is_dir()):
                    raise StageVerificationError(f"stage archive target collision: {member.name}")
                if not target.exists():
                    try:
                        target.mkdir(mode=0o700)
                    except FileExistsError as exc:
                        raise StageVerificationError(
                            f"stage archive target collision: {member.name}"
                        ) from exc
                    except OSError as exc:
                        raise StageVerificationError(
                            f"cannot create stage archive directory: {member.name}"
                        ) from exc
                    created_dirs.add(member_name)
                elif member_name not in created_dirs:
                    raise StageVerificationError(f"stage archive target collision: {member.name}")
            directory_modes[target] = _safe_stage_mode(
                member.mode, directory=True, label=member.name
            )
            continue
        _ensure_extract_parent(destination, target, created_dirs)
        _extract_regular_member(archive, member, target)

    # Apply directory modes only after every file is present, so a restrictive
    # archive mode cannot prevent a later member from being materialized.
    for directory, mode in directory_modes.items():
        try:
            os.chmod(directory, mode, follow_symlinks=False)
            _fsync_directory(directory)
        except OSError as exc:
            raise StageVerificationError(f"cannot finalize stage directory mode: {directory}") from exc
    _fsync_directory(destination)


def _can_shadow_python(relative: str) -> bool:
    """Return whether an unlisted file could alter staged Python execution.

    ``stage-extract`` is intentionally allowed to reuse a writable run
    workspace for data and results.  It must not, however, accept an
    unlisted source/import file: ``pod_entry.sh`` puts ``<workspace>/src`` at
    the front of ``PYTHONPATH`` and an extra module there would silently win
    over the manifest-pinned implementation.  Reject the import-related
    suffixes everywhere, and reject every unlisted file below a ``src`` tree.
    """

    path = PurePosixPath(relative)
    suffix = path.suffix.lower()
    if suffix in {".py", ".pyi", ".pyc", ".pyo", ".so", ".pyd", ".pth"}:
        return True
    return "src" in path.parts


def _verify_fitting_corpus(root: Path) -> None:
    """Verify the exact pinned corpus and its producing source inside a stage."""

    corpus = root / FITTING_CORPUS_RELATIVE
    provenance = root / FITTING_CORPUS_PROVENANCE_RELATIVE
    generator = root / FITTING_CORPUS_GENERATOR_RELATIVE
    for label, path in (("corpus", corpus), ("corpus provenance", provenance),
                        ("corpus generator", generator)):
        if path.is_symlink() or not path.is_file():
            raise StageVerificationError(f"stage fitting {label} is missing or linked")
    # The provenance document is not a trust anchor by itself: an attacker can
    # rewrite both it and the prompt bytes while keeping their self-reported
    # digests internally consistent.  Compare both immutable inputs with the
    # reviewed constants before parsing any claims from them.
    if sha256_file(corpus) != FITTING_CORPUS_SHA256:
        raise StageVerificationError("stage fitting corpus bytes are not the trusted corpus")
    if sha256_file(provenance) != FITTING_CORPUS_PROVENANCE_SHA256:
        raise StageVerificationError("stage fitting corpus provenance is not trusted")
    try:
        prompts = json.loads(_read_regular_file(corpus))
        document = json.loads(_read_regular_file(provenance))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageVerificationError("stage fitting corpus provenance is unreadable") from exc
    source = {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "revision": WIKITEXT_REVISION,
        "minimum_characters": 600,
        "requested_prompts": 1200,
    }
    if (not isinstance(document, dict)
            or document.get("schema_version") != 1
            or document.get("status") != "FRESH_PINNED_DATASET_REVISION"
            or document.get("source") != source
            or document.get("datasets_version") != WIKITEXT_DATASETS_VERSION):
        raise StageVerificationError("stage fitting corpus source identity is not pinned")
    if (not isinstance(prompts, list) or len(prompts) != 1200
            or any(not isinstance(value, str) or len(value.strip()) < 600 for value in prompts)):
        raise StageVerificationError("stage fitting corpus does not satisfy the frozen selection rule")
    expected_output = {
        "path": "wikitext_prompts",
        "size": corpus.stat().st_size,
        "sha256": sha256_file(corpus),
    }
    expected_generator = {
        "path": FITTING_CORPUS_GENERATOR_RELATIVE,
        "size": generator.stat().st_size,
        "sha256": sha256_file(generator),
    }
    if document.get("output") != expected_output:
        raise StageVerificationError("stage fitting corpus byte identity mismatch")
    if document.get("generator") != expected_generator:
        raise StageVerificationError("stage fitting corpus generator identity mismatch")


def verify_stage_root(root: str | Path, *, allow_extra: bool = False) -> dict[str, Any]:
    raw_root = Path(root)
    _reject_symlink_ancestors(raw_root)
    if raw_root.is_symlink():
        raise StageVerificationError(f"stage root is a symlink: {raw_root}")
    root = raw_root.resolve()
    try:
        root_stat = os.stat(root, follow_symlinks=False)
    except OSError as exc:
        raise StageVerificationError(f"stage root is unavailable: {root}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise StageVerificationError(f"stage root is not a directory: {root}")
    # The private extraction root is created with a restrictive mode by the
    # transaction owner; reject special bits without requiring that root's
    # exact mode to match the payload-entry normalization below.
    _safe_stage_mode(root_stat.st_mode, directory=True, label=str(root))
    manifest_path, pinned_path = root / "MANIFEST.sha256", root / "PINNED_INPUTS.json"
    if (not manifest_path.is_file() or not pinned_path.is_file()
            or manifest_path.is_symlink() or pinned_path.is_symlink()):
        raise StageVerificationError("stage is missing MANIFEST.sha256 or PINNED_INPUTS.json")
    entries = []
    seen: set[str] = set()
    try:
        manifest_lines = manifest_path.read_text().splitlines()
    except (OSError, UnicodeError) as exc:
        raise StageVerificationError("cannot read stage manifest") from exc
    for line in manifest_lines:
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError as exc:
            raise StageVerificationError(f"malformed manifest line: {line!r}") from exc
        rel = safe_relative(rel)
        if rel in seen:
            raise StageVerificationError(f"duplicate manifest path: {rel}")
        seen.add(rel)
        if not SHA256_RE.fullmatch(digest):
            raise StageVerificationError(f"malformed digest for {rel}")
        path = root / rel
        resolved = path.resolve()
        if root != resolved and root not in resolved.parents:
            raise StageVerificationError(f"stage payload escapes root: {rel}")
        current = path.parent
        while current != root and current != current.parent:
            if current.is_symlink():
                raise StageVerificationError(f"stage payload traverses a link: {rel}")
            current = current.parent
        try:
            path_stat = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise StageVerificationError(f"stage payload is unavailable: {rel}") from exc
        _verify_stage_path_mode(path, path_stat.st_mode)
        if stat.S_ISREG(path_stat.st_mode) and path_stat.st_nlink != 1:
            raise StageVerificationError(f"stage payload is linked: {rel}")
        if not stat.S_ISREG(path_stat.st_mode) or path.is_symlink() or sha256_file(path) != digest:
            raise StageVerificationError(f"stage payload digest mismatch: {rel}")
        entries.append({"path": rel, "sha256": digest, "size": path_stat.st_size})
    try:
        pinned = json.loads(pinned_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageVerificationError("malformed PINNED_INPUTS.json") from exc
    if not isinstance(pinned, dict):
        raise StageVerificationError("PINNED_INPUTS.json must contain an object")
    if pinned.get("schema_version") != SCHEMA_VERSION:
        raise StageVerificationError("unsupported stage schema")
    manifest_digest = sha256_file(manifest_path)
    if pinned.get("payload_manifest_sha256") != manifest_digest:
        raise StageVerificationError("pinned manifest digest mismatch")
    if pinned.get("model") != {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION}:
        raise StageVerificationError("model repository/revision is not pinned to the approved input")
    if pinned.get("jacobian_lens") != {"repository": JLENS_REPOSITORY, "revision": JLENS_REVISION}:
        raise StageVerificationError("jacobian-lens repository/revision is not pinned")
    if pinned.get("python") != TARGET_PYTHON:
        raise StageVerificationError("stage Python version is not pinned to the target image")
    if pinned.get("dependencies") != RUNTIME_DEPENDENCIES:
        raise StageVerificationError("runtime dependencies are not pinned")
    _verify_fitting_corpus(root)
    source = pinned.get("source")
    if not isinstance(source, dict) or set(source) != {"head", "status"}:
        raise StageVerificationError("stage source pin is malformed")
    source_policy = pinned.get("source_policy")
    if source_policy not in {"clean_git_payload", "snapshot_manifest_only"}:
        raise StageVerificationError("stage source policy is malformed")
    if (source["head"] is not None and
            (not isinstance(source["head"], str)
             or GIT_OID_RE.fullmatch(source["head"]) is None)):
        raise StageVerificationError("stage source head is malformed")
    if source["status"] is not None and not isinstance(source["status"], str):
        raise StageVerificationError("stage source status is malformed")
    if source_policy == "clean_git_payload" and (source["head"] is None or source["status"] != ""):
        raise StageVerificationError("clean stage is not bound to a clean git commit")
    committed = pinned.get("committed_payload_sha256")
    if source_policy == "clean_git_payload" and not isinstance(committed, dict):
        raise StageVerificationError("clean stage is missing committed HEAD payload bytes")
    if committed is not None:
        if not isinstance(committed, dict):
            raise StageVerificationError("committed payload pin is malformed")
        expected_tracked = {
            entry["path"] for entry in entries
            if "artifacts" not in PurePosixPath(entry["path"]).parts
        }
        if set(committed) != expected_tracked:
            raise StageVerificationError("committed payload pin is incomplete")
        if any(not isinstance(path, str) or not SHA256_RE.fullmatch(str(digest))
               for path, digest in committed.items()):
            raise StageVerificationError("committed payload pin contains a malformed digest")
        if source_policy == "clean_git_payload":
            manifest_digests = {entry["path"]: entry["sha256"] for entry in entries}
            if any(committed[path] != manifest_digests[path] for path in expected_tracked):
                raise StageVerificationError(
                    "clean stage payload does not match its committed HEAD blob"
                )
    allowed = seen | {"MANIFEST.sha256", "PINNED_INPUTS.json"}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise StageVerificationError(f"links are not allowed in stage root: {relative}")
        try:
            path_stat = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise StageVerificationError(f"stage path is unavailable: {relative}") from exc
        _verify_stage_path_mode(path, path_stat.st_mode)
        if stat.S_ISREG(path_stat.st_mode) and path_stat.st_nlink != 1:
            raise StageVerificationError(f"stage path is linked: {relative}")
        if stat.S_ISREG(path_stat.st_mode) and relative not in allowed:
            if not allow_extra or _can_shadow_python(relative):
                raise StageVerificationError(f"unlisted stage payload: {relative}")
    return {"payload_count": len(entries), "manifest_sha256": manifest_digest, "pinned_inputs": pinned,
            "entries": entries}


def verify_stage_tar(archive_path: str | Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jlens-stage-verify-") as tmp:
        temporary_root = Path(tmp)
        snapshot = _snapshot_regular_file(
            archive_path, directory=temporary_root, prefix="archive-"
        )
        try:
            destination = temporary_root / "stage"
            destination.mkdir()
            try:
                with os.fdopen(os.dup(snapshot.fd), "rb") as raw_archive:
                    with tarfile.open(fileobj=raw_archive, mode="r:*") as archive:
                        _safe_extract(archive, destination)
            except (OSError, ValueError, tarfile.TarError) as exc:
                raise StageVerificationError(f"cannot read stage archive: {archive_path}") from exc
            return verify_stage_root(destination, allow_extra=False)
        finally:
            snapshot.cleanup()


def extract_stage_tar(archive_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Verify and install one archive transactionally into an absent root."""

    raw_destination = Path(destination)
    try:
        _reject_symlink_ancestors(raw_destination)
    except PublishError as exc:
        raise StageVerificationError(f"stage destination has a symlinked parent: {raw_destination}") from exc
    if raw_destination.is_symlink() or raw_destination.exists():
        raise StageVerificationError(f"stage destination already exists: {raw_destination}")
    destination = raw_destination.resolve(strict=False)
    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_ancestors(parent / "placeholder")
    except OSError as exc:
        raise StageVerificationError(f"cannot create stage destination parent: {parent}") from exc
    except PublishError as exc:
        raise StageVerificationError(f"stage destination has a symlinked parent: {parent}") from exc
    if destination.is_symlink() or destination.exists():
        raise StageVerificationError(f"stage destination already exists: {destination}")

    try:
        with tempfile.TemporaryDirectory(prefix="jlens-stage-snapshot-") as snapshot_tmp:
            snapshot = _snapshot_regular_file(
                archive_path, directory=Path(snapshot_tmp), prefix="archive-"
            )
            try:
                with tempfile.TemporaryDirectory(
                    prefix=f".{destination.name}.jlens-stage-", dir=str(parent)
                ) as temporary:
                    staging = Path(temporary)
                    with os.fdopen(os.dup(snapshot.fd), "rb") as raw_archive:
                        with tarfile.open(fileobj=raw_archive, mode="r:*") as archive:
                            _safe_extract(archive, staging)
                    result = verify_stage_root(staging, allow_extra=False)
                    if destination.is_symlink() or destination.exists():
                        raise StageVerificationError(f"stage destination already exists: {destination}")
                    _rename_directory_noreplace(staging, destination)
                    _fsync_directory(parent)
                    return result
            finally:
                snapshot.cleanup()
    except StageVerificationError:
        raise
    except (OSError, ValueError, tarfile.TarError, shutil.Error) as exc:
        raise StageVerificationError(f"cannot extract stage archive: {archive_path}") from exc


def _publisher_from_args(args: argparse.Namespace) -> Publisher:
    if args.local_publisher:
        # A local publisher never needs the credential.  Keep accepting the
        # common command-line option so callers can share one invocation shape
        # without turning a local operation into an HF dependency.
        return LocalPublisher(args.local_publisher)
    if not args.repo:
        raise PublishError("--repo or --local-publisher is required")
    token_file = getattr(args, "token_file", None)
    if not token_file:
        raise PublishError("--token-file is required for Hugging Face publishers")
    return HfPublisher(args.repo, token=read_hf_token_file(token_file))


def _cmd_publish_file(args: argparse.Namespace) -> int:
    publisher = _publisher_from_args(args)
    receipt = publish_file(publisher, args.file, run_id=args.run_id, relative_path=args.relative,
                           kind=args.kind, receipt_root=args.receipt_root)
    print(json.dumps(receipt.as_dict(), sort_keys=True))
    return 0


def _cmd_publish_tree(args: argparse.Namespace) -> int:
    publisher = _publisher_from_args(args)
    receipts = publish_tree(publisher, args.root, run_id=args.run_id, relative_prefix=args.prefix,
                            kind=args.kind, receipt_root=args.receipt_root)
    print(json.dumps({"count": len(receipts), "run_id": args.run_id}, sort_keys=True))
    return 0


def _cmd_restore_file(args: argparse.Namespace) -> int:
    publisher = _publisher_from_args(args)
    receipt = restore_file(publisher, run_id=args.run_id, relative_path=args.relative,
                           destination=args.file)
    print("RESTORED" if receipt is not None else "ABSENT")
    return 0


def _cmd_sidecar(args: argparse.Namespace) -> int:
    metadata: dict[str, Any] = {}
    if args.target_ut is not None:
        metadata["target_ut"] = args.target_ut
    if args.start is not None:
        metadata["start"] = args.start
    if args.end is not None:
        metadata["end"] = args.end
    if args.n_prompts is not None:
        metadata["n_prompts"] = args.n_prompts
    output = write_sidecar(args.file, args.output, run_id=args.run_id, relative_path=args.relative,
                           metadata=metadata)
    print(str(output))
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    publisher = _publisher_from_args(args)
    receipts = sync_and_verify(publisher, run_id=args.run_id, local_root=args.root,
                               relative_paths=args.relative, receipt_root=args.receipt_root)
    print(json.dumps({"count": len(receipts), "run_id": args.run_id}, sort_keys=True))
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    publisher = _publisher_from_args(args)
    receipt = publish_index(publisher, run_id=args.run_id, receipt_root=args.receipt_root_path,
                            local_root=args.local_root)
    print(json.dumps(receipt.as_dict(), sort_keys=True))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.local_root)
    validate_run_id(args.run_id)
    attempt_id = os.environ.get("JLENS_ATTEMPT_ID")
    if not isinstance(attempt_id, str):
        raise PublishError("JLENS_ATTEMPT_ID is required and malformed")
    try:
        attempt_id = validate_run_id(attempt_id)
    except ValueError as exc:
        raise PublishError("JLENS_ATTEMPT_ID is required and malformed") from exc
    launch_nonce = os.environ.get("JLENS_LAUNCH_NONCE")
    if not isinstance(launch_nonce, str) or LAUNCH_NONCE_RE.fullmatch(launch_nonce) is None:
        raise PublishError("JLENS_LAUNCH_NONCE is required and malformed")
    expected_stage_digest = os.environ.get("EXPECTED_STAGE_MANIFEST_SHA256")
    if not isinstance(expected_stage_digest, str) or not SHA256_RE.fullmatch(expected_stage_digest):
        raise PublishError("EXPECTED_STAGE_MANIFEST_SHA256 is required and malformed")
    expected_stage_head = os.environ.get("EXPECTED_STAGE_SOURCE_HEAD")
    if not isinstance(expected_stage_head, str) or not GIT_OID_RE.fullmatch(expected_stage_head):
        raise PublishError("EXPECTED_STAGE_SOURCE_HEAD is required and malformed")
    worker_deadline_raw = os.environ.get("JLENS_WORKER_DEADLINE_EPOCH")
    if not isinstance(worker_deadline_raw, str) or WORKER_DEADLINE_RE.fullmatch(worker_deadline_raw) is None:
        raise PublishError("JLENS_WORKER_DEADLINE_EPOCH is required and malformed")
    worker_deadline_epoch = int(worker_deadline_raw)
    if worker_deadline_epoch <= int(time.time()):
        raise PublishError("JLENS_WORKER_DEADLINE_EPOCH is already past")
    publisher = _publisher_from_args(args)
    _reject_symlink_ancestors(root / "placeholder")
    if root.is_symlink():
        raise PublishError(f"status root is a symlink: {root}")
    status_dir = root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    nonce = f"{time.time_ns():020d}"
    path = status_dir / f"{stamp}_{nonce}.json"
    payload = {"schema_version": SCHEMA_VERSION, "run_id": args.run_id, "state": args.state,
               "stage": args.stage, "exit_code": args.exit_code, "recorded_at": time.time(),
               "launch_nonce": launch_nonce,
               "stage_manifest_sha256": expected_stage_digest,
               "stage_source_head": expected_stage_head,
               "worker_deadline_epoch": worker_deadline_epoch,
               # Retain the historical names for offline consumers; the
               # canonical fields above are what the paid success gate reads.
               "expected_stage_manifest_sha256": expected_stage_digest,
               "expected_stage_source_head": expected_stage_head}
    payload["attempt_id"] = attempt_id
    stage_digest = os.environ.get("STAGE_MANIFEST_SHA256")
    if stage_digest:
        if not SHA256_RE.fullmatch(stage_digest):
            raise PublishError("STAGE_MANIFEST_SHA256 is malformed")
        if stage_digest != expected_stage_digest:
            raise PublishError("STAGE_MANIFEST_SHA256 does not match expected stage manifest")
    _write_immutable(path, canonical_json(payload))
    rel = f"status/{path.name}"
    publish_file(publisher, path, run_id=args.run_id, relative_path=rel, kind=args.kind,
                 receipt_root=args.receipt_root)
    print(str(path))
    return 0


def _cmd_stage_create(args: argparse.Namespace) -> int:
    result = build_stage_tar(args.root, args.output, require_clean_source=True)
    print(json.dumps({"archive": result["archive"], "payload_count": result["payload_count"],
                      "manifest_sha256": result["manifest_sha256"]}, sort_keys=True))
    return 0


def _cmd_stage_verify(args: argparse.Namespace) -> int:
    result = (verify_stage_tar(args.archive) if args.archive
              else verify_stage_root(args.root, allow_extra=args.allow_extra))
    print(json.dumps({"payload_count": result["payload_count"], "manifest_sha256": result["manifest_sha256"]},
                     sort_keys=True))
    return 0


def _cmd_stage_extract(args: argparse.Namespace) -> int:
    """Safely extract and verify a stage archive into a workspace directory."""

    result = extract_stage_tar(args.archive, args.destination)
    print(json.dumps({"payload_count": result["payload_count"], "manifest_sha256": result["manifest_sha256"]},
                     sort_keys=True))
    return 0


def _cmd_stage_publish(args: argparse.Namespace) -> int:
    """Publish and remotely re-verify one digest-addressed stage archive."""

    archive = Path(args.archive)
    local = verify_stage_tar(archive)
    manifest_digest = local["manifest_sha256"]
    remote = safe_relative(args.remote)
    if remote.count(manifest_digest) != 1:
        raise StageVerificationError(
            "stage remote path must contain exactly its verified manifest digest"
        )
    if not args.token_file:
        raise PublishError("--token-file is required for Hugging Face publishers")
    publisher = HfPublisher(args.repo, token=read_hf_token_file(args.token_file))
    publisher.ensure_repository()
    publisher.put_file(archive, remote)
    with tempfile.TemporaryDirectory(prefix="jlens-stage-remote-") as tmp:
        remote_archive = Path(tmp) / "stage.tar.gz"
        local_size, local_digest = _path_digest(archive, label="local stage archive")
        downloader = publisher.download_file
        try:
            parameters = inspect.signature(downloader).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_size = (
            "expected_size" in parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        if accepts_size:
            downloader(remote, remote_archive, expected_size=local_size)
        else:
            # Preserve compatibility with injected publishers implementing
            # the original two-argument download_file method.
            downloader(remote, remote_archive)
        remote_size, remote_digest = _path_digest(remote_archive, label="remote stage archive")
        if (remote_size, remote_digest) != (local_size, local_digest):
            raise ImmutableConflict("remote stage archive differs from the local archive")
        verified_remote = verify_stage_tar(remote_archive)
        if verified_remote["manifest_sha256"] != manifest_digest:
            raise StageVerificationError("remote stage manifest differs from the local archive")
    print(json.dumps({
        "archive_sha256": local_digest,
        "manifest_sha256": manifest_digest,
        "remote_path": remote,
    }, sort_keys=True))
    return 0


def _cmd_runtime_verify(_args: argparse.Namespace) -> int:
    print(json.dumps(verify_runtime_lock(), sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def publisher_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repo")
        command.add_argument("--local-publisher")
        command.add_argument(
            "--token-file",
            help="0600 local file containing the HF token (never the token itself)",
        )
        command.add_argument("--run-id", required=True)
        command.add_argument("--receipt-root")

    p = sub.add_parser("publish-file")
    publisher_options(p)
    p.add_argument("--file", required=True)
    p.add_argument("--relative", required=True)
    p.add_argument("--kind", required=True)
    p.set_defaults(fn=_cmd_publish_file)

    p = sub.add_parser("publish-tree")
    publisher_options(p)
    p.add_argument("--root", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--kind", default="artifact")
    p.set_defaults(fn=_cmd_publish_tree)

    p = sub.add_parser("restore-file")
    publisher_options(p)
    p.add_argument("--file", required=True)
    p.add_argument("--relative", required=True)
    p.set_defaults(fn=_cmd_restore_file)

    p = sub.add_parser("sidecar")
    p.add_argument("--run-id", required=True)
    p.add_argument("--file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--relative", required=True)
    p.add_argument("--target-ut", type=int)
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--n-prompts", type=int)
    p.set_defaults(fn=_cmd_sidecar)

    p = sub.add_parser("sync")
    publisher_options(p)
    p.add_argument("--root", required=True)
    p.add_argument("--relative", action="append")
    p.set_defaults(fn=_cmd_sync)

    p = sub.add_parser("index")
    publisher_options(p)
    p.add_argument("--local-root", required=True)
    p.add_argument("--receipt-root-path", required=True)
    p.set_defaults(fn=_cmd_index)

    p = sub.add_parser("status")
    publisher_options(p)
    p.add_argument("--local-root", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--exit-code", type=int)
    p.add_argument("--kind", default="heartbeat")
    p.set_defaults(fn=_cmd_status)

    p = sub.add_parser("stage-create")
    p.add_argument("--root", default=".")
    p.add_argument("--output", required=True)
    p.set_defaults(fn=_cmd_stage_create)

    p = sub.add_parser("stage-verify")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--root")
    group.add_argument("--archive")
    p.add_argument("--allow-extra", action="store_true",
                   help="allow pre-existing workspace files when verifying a staged root")
    p.set_defaults(fn=_cmd_stage_verify)

    p = sub.add_parser("stage-extract")
    p.add_argument("--archive", required=True)
    p.add_argument("--destination", required=True)
    p.set_defaults(fn=_cmd_stage_extract)

    p = sub.add_parser("stage-publish")
    p.add_argument("--repo", required=True)
    p.add_argument(
        "--token-file",
        help="0600 local file containing the HF token (never the token itself)",
    )
    p.add_argument("--archive", required=True)
    p.add_argument("--remote", required=True)
    p.set_defaults(fn=_cmd_stage_publish)

    p = sub.add_parser("runtime-verify")
    p.set_defaults(fn=_cmd_runtime_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (PublishError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
