"""Small, dependency-light helpers for durable JLens evidence.

The fitting and evaluation commands write large binary files and JSON
sidecars.  A successful process must leave either the complete file or no
file that can be mistaken for a complete result.  This module centralises the
few rules needed by those commands:

* all replacement writes happen in the destination directory and use
  ``os.replace``;
* hashes are SHA-256 over the bytes that are actually on disk;
* JSON digests use one canonical representation so a prompt slice has a
  stable identity across processes.

The module intentionally has no model or cloud dependencies.  It is also
used by lightweight tests and by report/probe code that only needs to record
file identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of *data*."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# A short alias is useful at call sites which deal in records rather than
# arbitrary byte strings.  Keep the public name explicit as well.
hash_file = sha256_file


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with deterministic separators, ordering, and UTF-8."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash the canonical JSON representation of *value*."""

    return sha256_bytes(canonical_json_bytes(value))


def prompt_slice_sha256(prompts: list[str] | tuple[str, ...]) -> str:
    """Hash an ordered prompt slice, including its boundaries implicitly."""

    return sha256_json(list(prompts))


def aggregate_sha256(values: Mapping[str, str] | list[str] | tuple[str, ...]) -> str:
    """Hash a deterministic collection of already-computed digests.

    Mappings are sorted by key; sequences retain their order.  This is used
    for source/generator manifests where the individual file records remain
    useful to a reviewer but a compact identity is convenient in a sidecar.
    """

    if isinstance(values, Mapping):
        payload: Any = {str(k): str(v) for k, v in sorted(values.items(), key=lambda item: str(item[0]))}
    else:
        payload = [str(v) for v in values]
    return sha256_json(payload)


def file_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the byte identity of an existing file.

    ``path`` is preserved as supplied (rather than silently making an
    absolute path) so sidecars remain readable when a project is relocated.
    The file is opened and hashed before the record is returned; a missing or
    unreadable file therefore fails the producing command instead of creating
    an unverifiable sidecar.
    """

    candidate = Path(path)
    stat = candidate.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "sha256": sha256_file(candidate),
    }


def _temporary_path(path: Path, suffix: str = ".tmp") -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=str(path.parent))
    return fd, Path(raw)


def _replace_complete(temp: Path, destination: Path) -> None:
    """Flush a temporary file and atomically install it at *destination*."""

    # The caller has already closed the descriptor.  Re-open only to issue a
    # best-effort fsync; on filesystems where fsync is unavailable the atomic
    # replacement still provides the important no-partial-file guarantee.
    try:
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass
    os.replace(temp, destination)
    try:
        fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Atomically write bytes to *path*.

    The temporary file is removed on every failure.  ``os.replace`` is used
    only after the complete payload has been written and flushed.
    """

    destination = Path(path)
    fd, temporary = _temporary_path(destination)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_complete(temporary, destination)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(
    path: str | os.PathLike[str], value: str, *, encoding: str = "utf-8"
) -> None:
    """Atomically write text to *path* using *encoding*."""

    atomic_write_bytes(path, value.encode(encoding))


def atomic_write_json(
    path: str | os.PathLike[str], payload: Any, *, indent: int | None = 1
) -> None:
    """Atomically write a JSON document encoded as UTF-8."""

    text = json.dumps(
        payload,
        indent=indent,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def atomic_savez(path: str | os.PathLike[str], **arrays: Any) -> None:
    """Atomically write a compressed NumPy ``.npz`` archive.

    NumPy appends ``.npz`` when given a filename without that suffix, so a
    named temporary file is opened explicitly to make the replacement path
    unambiguous.
    """

    import numpy as np

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = _temporary_path(destination, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_complete(temporary, destination)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def atomic_torch_save(obj: Any, path: str | os.PathLike[str]) -> None:
    """Atomically serialize a PyTorch object without importing torch eagerly."""

    import torch

    destination = Path(path)
    fd, temporary = _temporary_path(destination, suffix=".pt")
    os.close(fd)
    try:
        torch.save(obj, str(temporary))
        _replace_complete(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def source_manifest(paths: list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...]) -> dict[str, Any]:
    """Record source files and a stable aggregate identity."""

    records = [file_record(path) for path in paths]
    by_path = {record["path"]: record["sha256"] for record in records}
    return {
        "files": records,
        "sha256": aggregate_sha256(by_path),
    }


__all__ = [
    "SCHEMA_VERSION",
    "aggregate_sha256",
    "atomic_savez",
    "atomic_torch_save",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "canonical_json_bytes",
    "file_record",
    "hash_file",
    "prompt_slice_sha256",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "source_manifest",
]
