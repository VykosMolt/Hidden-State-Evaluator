"""Immutable run artifacts and stage-package verification for JLens.

The paid run is deliberately coupled to this small, boring protocol.  Every
file is addressed below a caller supplied ``run_id`` and is accompanied by a
receipt containing its SHA-256 digest.  A publisher must acknowledge the
receipt after it has been written; an existing path can only be reused when
its receipt and bytes are identical.

The module has no dependency on Hugging Face.  ``LocalPublisher`` is used by
tests and offline dry runs, while ``HfPublisher`` shells out to the ``hf``
CLI only when a real remote run explicitly asks for it.  The shell entry
points use the CLI at the bottom of this file.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Protocol


SCHEMA_VERSION = 1
MODEL_REPOSITORY = "ByteDance/Ouro-2.6B"
MODEL_REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"
JLENS_REPOSITORY = "https://github.com/anthropics/jacobian-lens.git"
JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
TARGET_PYTHON = "3.11"
RUNTIME_DEPENDENCIES = {
    "transformers": "4.54.1",
    "numpy": "2.4.3",
    "scikit-learn": "1.8.0",
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
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HF_REPO_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$"
)


class PublishError(RuntimeError):
    """An artifact could not be published or acknowledged."""


class ImmutableConflict(PublishError):
    """A run path already contains different bytes or metadata."""


class MissingRemote(PublishError):
    """A requested remote path does not exist."""


class StageVerificationError(PublishError):
    """A stage archive or its pinned inputs are not internally consistent."""


class _RemoteCommandFailure(PublishError):
    """Private command error retaining enough detail to classify a 404."""


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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def payload_remote_path(run_id: str, relative: str | Path) -> str:
    return f"{validate_run_id(run_id)}/artifacts/{safe_relative(relative)}"


def receipt_remote_path(run_id: str, relative: str | Path) -> str:
    return f"{validate_run_id(run_id)}/receipts/{safe_relative(relative)}.receipt.json"


def _atomic_write(path: Path, data: bytes) -> None:
    _reject_symlink_ancestors(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _write_immutable(path: Path, data: bytes) -> None:
    """Create a local protocol file without replacing an existing path."""

    _reject_symlink_ancestors(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        if path.is_symlink() or not path.is_file():
            raise ImmutableConflict(f"immutable path is not a regular file: {path}") from exc
        try:
            existing = path.read_bytes()
        except OSError as read_exc:
            raise ImmutableConflict(f"cannot read immutable path: {path}") from read_exc
        if existing != data:
            raise ImmutableConflict(f"immutable path already differs: {path}")
        return
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _reject_symlink_ancestors(path: Path) -> None:
    """Reject a writable path routed through an existing symlink directory."""

    current = Path(path).parent
    while current != current.parent:
        if current.is_symlink():
            raise PublishError(f"writable path has a symlinked parent: {current}")
        current = current.parent


def _install_immutable(source: Path, destination: Path) -> None:
    """Install ``source`` without replacing an existing destination."""

    if source.is_symlink() or not source.is_file():
        raise PublishError(f"immutable source is not a regular file: {source}")
    # Never let an immutable artifact be routed through an existing link.  A
    # resolved destination can otherwise hide a symlink that points back into
    # the publisher root and make an apparently harmless retry overwrite a
    # different object.
    _reject_symlink_ancestors(destination)
    if destination.is_symlink():
        raise ImmutableConflict(f"immutable path is a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, tmp)
        _link_immutable(tmp, destination, source_for_comparison=source)
    finally:
        tmp.unlink(missing_ok=True)


def _link_immutable(source: Path, destination: Path, *, source_for_comparison: Path | None = None) -> None:
    """Link a complete temporary file without replacing a destination."""

    if source.is_symlink() or not source.is_file():
        raise PublishError(f"immutable source is not a regular file: {source}")
    _reject_symlink_ancestors(destination)
    if destination.is_symlink():
        raise ImmutableConflict(f"immutable path is a symlink: {destination}")
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        if destination.is_symlink():
            raise ImmutableConflict(f"immutable path is a symlink: {destination}") from exc
        comparison = source_for_comparison or source
        if sha256_file(destination) != sha256_file(comparison):
            raise ImmutableConflict(f"immutable path already differs: {destination}") from exc


class Publisher(Protocol):
    """Minimal publisher interface used by ``publish_file``."""

    def put_file(self, source: Path, remote_path: str) -> None: ...

    def read_file(self, remote_path: str) -> bytes: ...

    def download_file(self, remote_path: str, destination: Path) -> None: ...


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

    def put_file(self, source: Path, remote_path: str) -> None:
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            raise PublishError(f"source is not a regular file: {source}")
        _install_immutable(source, self._path(remote_path))

    def read_file(self, remote_path: str) -> bytes:
        path = self._path(remote_path)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise MissingRemote(remote_path) from exc

    def download_file(self, remote_path: str, destination: Path) -> None:
        source = self._path(remote_path)
        try:
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(source)
            _install_immutable(source, Path(destination))
        except FileNotFoundError as exc:
            raise MissingRemote(remote_path) from exc

    def list_files(self, prefix: str = "") -> list[str]:
        base = self._path(prefix.rstrip("/")) if prefix else self.root
        if not base.exists():
            return []
        files: list[str] = []
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                raise PublishError(f"publisher tree contains a symlink: {path}")
            if path.is_file():
                files.append(path.relative_to(self.root).as_posix())
        return files


class HfPublisher:
    """Publisher backed by the Hugging Face CLI.

    ``hf upload`` is intentionally invoked with an argument list, never a
    shell command.  The CLI's output is captured so tokens or signed URLs
    cannot enter the run log.  A remote receipt is read back after upload,
    making a successful return from ``hf upload`` insufficient as an ACK.
    """

    def __init__(self, repository: str, *, runner=None, timeout: int = 120):
        if not isinstance(repository, str) or not HF_REPO_RE.fullmatch(repository):
            raise ValueError("invalid Hugging Face repository")
        self.repository = repository
        self.runner = runner or subprocess.run
        self.timeout = timeout

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            result = self.runner(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise _RemoteCommandFailure(f"publisher command failed: {argv[0]}") from exc
        returncode = getattr(result, "returncode", None)
        if returncode != 0:
            detail = (getattr(result, "stderr", "") or "").lower()
            error = _RemoteCommandFailure(
                f"publisher command returned {returncode if returncode is not None else 'unknown'}: {argv[0]}"
            )
            # Keep server diagnostics available for classifying a missing path,
            # but never put command stderr (which may contain credentials or
            # signed URLs) in the exception rendered by a run log.
            error.detail = detail
            raise error
        return result

    def put_file(self, source: Path, remote_path: str) -> None:
        if not Path(source).is_file() or Path(source).is_symlink():
            raise PublishError(f"source is not a regular file: {source}")
        self._run(["hf", "upload", self.repository, str(source), safe_relative(remote_path)])

    def read_file(self, remote_path: str) -> bytes:
        with tempfile.NamedTemporaryFile(prefix="jlens-hf-read-", delete=False) as tmp:
            destination = Path(tmp.name)
        destination.unlink(missing_ok=True)
        try:
            self.download_file(remote_path, destination)
            return destination.read_bytes()
        finally:
            destination.unlink(missing_ok=True)

    def download_file(self, remote_path: str, destination: Path) -> None:
        remote_path = safe_relative(remote_path)
        destination = Path(destination)
        with tempfile.TemporaryDirectory(prefix="jlens-hf-get-") as tmp:
            try:
                self._run(["hf", "download", self.repository, remote_path, "--local-dir", tmp])
            except _RemoteCommandFailure as exc:
                detail = getattr(exc, "detail", "")
                if any(marker in detail for marker in ("404", "not found", "does not exist", "cannot find")):
                    raise MissingRemote(remote_path) from exc
                raise
            candidates = [Path(tmp) / remote_path, Path(tmp) / Path(remote_path).name]
            for candidate in candidates:
                if candidate.is_file():
                    _install_immutable(candidate, destination)
                    return
            raise MissingRemote(remote_path)


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
    relative = safe_relative(relative_path)
    validate_run_id(run_id)
    if not kind or any(c in kind for c in "\r\n"):
        raise ValueError("invalid artifact kind")
    if not source.is_file() or source.is_symlink():
        raise PublishError(f"source is not a regular file: {source}")
    return Receipt(SCHEMA_VERSION, run_id, relative, payload_remote_path(run_id, relative), kind,
                   sha256_file(source), source.stat().st_size)


def _receipt_root_path(receipt_root: str | Path | None, receipt: Receipt) -> Path | None:
    if receipt_root is None:
        return None
    return Path(receipt_root) / f"{receipt.relative_path}.receipt.json"


def _check_ack(receipt: Receipt, data: bytes) -> None:
    acknowledged = Receipt.from_bytes(data)
    if acknowledged != receipt:
        raise ImmutableConflict(f"publisher acknowledged a different receipt for {receipt.relative_path}")


def _download_remote(publisher: Publisher, remote_path: str, destination: Path) -> None:
    """Download to a file without requiring large payloads in memory.

    Older injected fakes may only implement ``read_file``; retaining that
    fallback keeps the protocol easy to test while real publishers use their
    streaming-to-disk implementation for multi-gigabyte lens shards.
    """

    download = getattr(publisher, "download_file", None)
    if callable(download):
        download(remote_path, destination)
        return
    data = publisher.read_file(remote_path)
    destination.write_bytes(data)


def _verify_remote_payload(publisher: Publisher, receipt: Receipt) -> None:
    with tempfile.NamedTemporaryFile(prefix="jlens-verify-", delete=False) as tmp:
        destination = Path(tmp.name)
    destination.unlink(missing_ok=True)
    try:
        _download_remote(publisher, receipt.remote_path, destination)
        if (destination.stat().st_size != receipt.size
                or sha256_file(destination) != receipt.sha256):
            raise ImmutableConflict(f"remote payload differs: {receipt.remote_path}")
    except FileNotFoundError as exc:
        raise PublishError(f"remote payload is missing: {receipt.remote_path}") from exc
    finally:
        destination.unlink(missing_ok=True)


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
    receipt = make_receipt(source, run_id, relative_path, kind)
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
            _verify_remote_payload(publisher, receipt)
        except MissingRemote as exc:
            raise PublishError(f"receipt exists but payload is missing: {receipt.remote_path}") from exc
    else:
        # A crash can happen after payload upload but before receipt upload.
        # Never overwrite that orphaned payload: bind it by digest and only
        # create the missing receipt if it is exactly the same artifact.
        try:
            _verify_remote_payload(publisher, receipt)
        except MissingRemote:
            publisher.put_file(source, receipt.remote_path)
            # Bind the receipt to the bytes actually present remotely before
            # publishing the receipt itself.  An upload that returned success
            # but stored truncated or changed bytes is never acknowledged.
            _verify_remote_payload(publisher, receipt)
        else:
            # The orphaned payload already matches the receipt we are about to
            # publish, so no upload is necessary.
            pass
        with tempfile.NamedTemporaryFile(prefix="jlens-receipt-", suffix=".json", delete=False) as tmp:
            receipt_tmp = Path(tmp.name)
            tmp.write(canonical_json(receipt.as_dict()))
        try:
            publisher.put_file(receipt_tmp, receipt_path)
        finally:
            receipt_tmp.unlink(missing_ok=True)
        # A successful upload is not an acknowledgment until read back.
        try:
            ack_data = publisher.read_file(receipt_path)
        except MissingRemote as exc:
            raise PublishError(f"receipt upload was not acknowledged: {receipt_path}") from exc
        _check_ack(receipt, ack_data)

    local_receipt = _receipt_root_path(receipt_root, receipt)
    if local_receipt is not None:
        _write_immutable(local_receipt, canonical_json(receipt.as_dict()))
    return receipt


def verify_local_receipt(source: str | Path, receipt_path: str | Path, *, run_id: str | None = None) -> Receipt:
    source = Path(source)
    receipt = Receipt.from_bytes(Path(receipt_path).read_bytes())
    if run_id is not None and receipt.run_id != validate_run_id(run_id):
        raise PublishError("receipt belongs to another run")
    if not source.is_file() or source.is_symlink():
        raise PublishError(f"receipt payload is missing: {source}")
    if source.stat().st_size != receipt.size or sha256_file(source) != receipt.sha256:
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
        if destination.stat().st_size != receipt.size or sha256_file(destination) != receipt.sha256:
            raise ImmutableConflict(f"restore destination differs: {destination}")
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="jlens-restore-", dir=str(destination.parent), delete=False) as tmp:
        temporary = Path(tmp.name)
    temporary.unlink(missing_ok=True)
    try:
        _download_remote(publisher, receipt.remote_path, temporary)
        if temporary.stat().st_size != receipt.size or sha256_file(temporary) != receipt.sha256:
            raise PublishError(f"remote digest mismatch: {receipt.remote_path}")
        _install_immutable(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
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
    if output.exists():
        if output.is_symlink() or output.read_bytes() != data:
            raise ImmutableConflict(f"sidecar already differs: {output}")
    else:
        _atomic_write(output, data)
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
    if not root.is_dir():
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


def sync_and_verify(
    publisher: Publisher,
    *,
    run_id: str,
    local_root: str | Path,
    relative_paths: Iterable[str] | None = None,
) -> list[Receipt]:
    """Fetch acknowledged artifacts and verify every digest locally.

    ``relative_paths`` is required for publishers that cannot enumerate a
    repository cheaply (HF); ``LocalPublisher`` can enumerate its receipts
    when omitted.  A missing or malformed receipt is a hard failure.
    """

    run_id = validate_run_id(run_id)
    if relative_paths is None:
        if not hasattr(publisher, "list_files"):
            raise PublishError("relative_paths are required for this publisher")
        prefix = f"{run_id}/receipts/"
        files = getattr(publisher, "list_files")(prefix)
        relative_paths = []
        for path in files:
            if not path.startswith(prefix) or not path.endswith(".receipt.json"):
                continue
            rel = path[len(prefix) : -len(".receipt.json")]
            relative_paths.append(rel)
    local_root = Path(local_root)
    receipts: list[Receipt] = []
    for relative in sorted({safe_relative(x) for x in relative_paths}):
        receipt_path = receipt_remote_path(run_id, relative)
        receipt = Receipt.from_bytes(publisher.read_file(receipt_path))
        if receipt.run_id != run_id or receipt.relative_path != relative:
            raise PublishError(f"receipt identity mismatch: {receipt_path}")
        destination = local_root / relative
        _reject_symlink_ancestors(destination)
        if destination.is_symlink():
            raise PublishError(f"local destination is a symlink: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="jlens-sync-", dir=str(destination.parent), delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            tmp_path.unlink(missing_ok=True)
            _download_remote(publisher, receipt.remote_path, tmp_path)
            if tmp_path.stat().st_size != receipt.size or sha256_file(tmp_path) != receipt.sha256:
                raise PublishError(f"remote digest mismatch: {receipt.remote_path}")
            if destination.exists():
                if destination.stat().st_size != receipt.size or sha256_file(destination) != receipt.sha256:
                    raise ImmutableConflict(f"local destination differs: {destination}")
                tmp_path.unlink(missing_ok=True)
            else:
                _link_immutable(tmp_path, destination, source_for_comparison=tmp_path)
                tmp_path.unlink(missing_ok=True)
        finally:
            tmp_path.unlink(missing_ok=True)
        receipts.append(receipt)
    return receipts


def publish_index(
    publisher: Publisher,
    *,
    run_id: str,
    receipt_root: str | Path,
    local_root: str | Path,
) -> Receipt:
    """Publish one terminal index that lets a controller discover all ACKs.

    The index is written only after the run has acknowledged every payload.
    Its own receipt is deliberately not included in ``relative_paths`` to
    avoid a circular document; a controller adds ``receipt-index.json`` when
    synchronizing.
    """

    run_id = validate_run_id(run_id)
    receipt_root = Path(receipt_root)
    suffix = ".receipt.json"
    relatives: list[str] = []
    if receipt_root.is_dir():
        for path in sorted(receipt_root.rglob(f"*{suffix}")):
            relative_receipt = path.relative_to(receipt_root).as_posix()
            if path.is_symlink():
                raise PublishError(f"local receipt is a symlink: {path}")
            if (not path.is_file()
                    or relative_receipt == f"receipt-index.json{suffix}"):
                continue
            relative = relative_receipt[:-len(suffix)]
            try:
                receipt = Receipt.from_bytes(path.read_bytes())
            except (OSError, PublishError) as exc:
                raise PublishError(f"cannot index malformed local receipt: {path}") from exc
            if receipt.run_id != run_id or receipt.relative_path != relative:
                raise PublishError(f"local receipt identity mismatch: {path}")
            relatives.append(relative)
    relatives.sort()
    payload = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "relative_paths": relatives}
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    index = local_root / "receipt-index.json"
    _write_immutable(index, canonical_json(payload))
    return publish_file(publisher, index, run_id=run_id, relative_path="receipt-index.json",
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
        result = sorted({safe_relative(x) for x in paths})
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PublishError("malformed receipt index") from exc
    if "receipt-index.json" in result:
        raise PublishError("receipt index must not contain itself")
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
    optional_tests = [
        unit_tests / name
        for name in (
            "test_jlens_integrity.py",
            "test_jlens_rental.py",
            "test_jlens_reports.py",
            "test_peer1_index_map.py",
            "test_peer3_jlens_audit.py",
        )
        if (unit_tests / name).is_file()
    ]
    roots = [
        root / "src" / "ouro_jlens",
        root / "artifacts" / "jlens" / "data" / "wikitext_prompts.json",
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
    return {"head": run("rev-parse", "HEAD"), "status": status}


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
    payloads = _stage_payload_paths(root)
    source_pin = _git_pin(root, payloads)
    if require_clean_source:
        if source_pin["head"] is None:
            raise StageVerificationError("paid stage requires a git commit identity")
        if source_pin["status"] is None:
            raise StageVerificationError("paid stage could not verify scoped source status")
        if source_pin["status"]:
            raise StageVerificationError("paid stage source/tests/docs are not clean and committed")
    entries = []
    for path in payloads:
        rel = path.relative_to(root).as_posix()
        entries.append({"path": rel, "sha256": sha256_file(path), "size": path.stat().st_size})
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jlens-stage-") as tmp:
        tmp_path = Path(tmp)
        manifest_path = tmp_path / "MANIFEST.sha256"
        pinned_path = tmp_path / "PINNED_INPUTS.json"
        manifest_path.write_text(manifest_text)
        pinned_path.write_bytes(canonical_json(pinned))
        # Use an explicit gzip stream with a fixed mtime.  The manifest pins
        # payload bytes, and the archive itself should be reproducible too so
        # retries cannot silently produce different bytes at one stage path.
        fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=str(output.parent))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as raw_output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                        for path in payloads:
                            rel = path.relative_to(root).as_posix()
                            info = archive.gettarinfo(str(path), arcname=rel)
                            info.uid = info.gid = 0
                            info.uname = info.gname = ""
                            info.mtime = 0
                            with path.open("rb") as handle:
                                archive.addfile(info, handle)
                        for path in (manifest_path, pinned_path):
                            info = archive.gettarinfo(str(path), arcname=path.name)
                            info.uid = info.gid = 0
                            info.uname = info.gname = ""
                            info.mtime = 0
                            with path.open("rb") as handle:
                                archive.addfile(info, handle)
                raw_output.flush()
                os.fsync(raw_output.fileno())
            os.replace(temporary, output)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise
    return {"archive": str(output), "payload_count": len(entries), "manifest_sha256": manifest_digest,
            "pinned_inputs": pinned}


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    seen_names: set[str] = set()
    for member in archive.getmembers():
        # Duplicate members make the resulting tree depend on extraction
        # order.  Refuse them so the manifest describes one unambiguous tree.
        raw_name = member.name
        if member.isdir() and raw_name.endswith("/"):
            raw_name = raw_name[:-1]
        try:
            member_name = safe_relative(raw_name)
        except ValueError as exc:
            if raw_name in {".", ""} and member.isdir():
                member_name = ""
            else:
                raise StageVerificationError(f"unsafe stage archive path: {member.name}") from exc
        if member_name in seen_names:
            raise StageVerificationError(f"duplicate stage archive member: {member.name}")
        seen_names.add(member_name)
        lexical_target = destination / member.name
        if lexical_target.is_symlink():
            raise StageVerificationError(f"stage archive target is an existing link: {member.name}")
        target = lexical_target.resolve()
        if destination != target and destination not in target.parents:
            raise StageVerificationError(f"stage archive path escapes root: {member.name}")
        current = lexical_target.parent
        while current != destination and current != current.parent:
            if current.is_symlink():
                raise StageVerificationError(f"stage archive path traverses a link: {member.name}")
            current = current.parent
        if member.issym() or member.islnk():
            raise StageVerificationError(f"links are not allowed in stage archive: {member.name}")
        if not (member.isdir() or member.isfile()):
            raise StageVerificationError(f"special archive member is not allowed: {member.name}")
    archive.extractall(destination)


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


def verify_stage_root(root: str | Path, *, allow_extra: bool = False) -> dict[str, Any]:
    raw_root = Path(root)
    _reject_symlink_ancestors(raw_root)
    if raw_root.is_symlink():
        raise StageVerificationError(f"stage root is a symlink: {raw_root}")
    root = raw_root.resolve()
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
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise StageVerificationError(f"stage payload digest mismatch: {rel}")
        entries.append({"path": rel, "sha256": digest, "size": path.stat().st_size})
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
    source = pinned.get("source")
    if not isinstance(source, dict) or set(source) != {"head", "status"}:
        raise StageVerificationError("stage source pin is malformed")
    source_policy = pinned.get("source_policy")
    if source_policy not in {"clean_git_payload", "snapshot_manifest_only"}:
        raise StageVerificationError("stage source policy is malformed")
    if (source["head"] is not None and
            (not isinstance(source["head"], str)
             or not re.fullmatch(r"[0-9a-f]{40,64}", source["head"]))):
        raise StageVerificationError("stage source head is malformed")
    if source["status"] is not None and not isinstance(source["status"], str):
        raise StageVerificationError("stage source status is malformed")
    if source_policy == "clean_git_payload" and (source["head"] is None or source["status"] != ""):
        raise StageVerificationError("clean stage is not bound to a clean git commit")
    allowed = seen | {"MANIFEST.sha256", "PINNED_INPUTS.json"}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise StageVerificationError(f"links are not allowed in stage root: {relative}")
        if path.is_file() and relative not in allowed:
            if not allow_extra or _can_shadow_python(relative):
                raise StageVerificationError(f"unlisted stage payload: {relative}")
    return {"payload_count": len(entries), "manifest_sha256": manifest_digest, "pinned_inputs": pinned,
            "entries": entries}


def verify_stage_tar(archive_path: str | Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jlens-stage-verify-") as tmp:
        destination = Path(tmp) / "stage"
        destination.mkdir()
        try:
            with tarfile.open(archive_path, "r:*") as archive:
                _safe_extract(archive, destination)
        except (OSError, ValueError, tarfile.TarError) as exc:
            raise StageVerificationError(f"cannot read stage archive: {archive_path}") from exc
        return verify_stage_root(destination)


def extract_stage_tar(archive_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Extract one verified archive into a fresh or restartable workspace."""

    destination = Path(destination)
    _reject_symlink_ancestors(destination)
    if destination.is_symlink():
        raise StageVerificationError(f"stage destination is a symlink: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            _safe_extract(archive, destination)
        return verify_stage_root(destination, allow_extra=True)
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise StageVerificationError(f"cannot extract stage archive: {archive_path}") from exc


def _publisher_from_args(args: argparse.Namespace) -> Publisher:
    if args.local_publisher:
        return LocalPublisher(args.local_publisher)
    if not args.repo:
        raise PublishError("--repo or --local-publisher is required")
    return HfPublisher(args.repo)


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
                               relative_paths=args.relative)
    print(json.dumps({"count": len(receipts), "run_id": args.run_id}, sort_keys=True))
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    publisher = _publisher_from_args(args)
    receipt = publish_index(publisher, run_id=args.run_id, receipt_root=args.receipt_root_path,
                            local_root=args.local_root)
    print(json.dumps(receipt.as_dict(), sort_keys=True))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    publisher = _publisher_from_args(args)
    root = Path(args.local_root)
    validate_run_id(args.run_id)
    status_dir = root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    nonce = f"{time.time_ns():020d}"
    path = status_dir / f"{stamp}_{nonce}.json"
    payload = {"schema_version": SCHEMA_VERSION, "run_id": args.run_id, "state": args.state,
               "stage": args.stage, "exit_code": args.exit_code, "recorded_at": time.time()}
    stage_digest = os.environ.get("STAGE_MANIFEST_SHA256")
    if stage_digest:
        if not SHA256_RE.fullmatch(stage_digest):
            raise PublishError("STAGE_MANIFEST_SHA256 is malformed")
        payload["stage_manifest_sha256"] = stage_digest
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


def _cmd_runtime_verify(_args: argparse.Namespace) -> int:
    print(json.dumps(verify_runtime_compatibility(), sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def publisher_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repo")
        command.add_argument("--local-publisher")
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
