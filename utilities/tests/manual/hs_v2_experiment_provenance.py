"""Shared, dependency-light provenance helpers for HS-v2 manual experiments.

The experiment launchers in this directory are deliberately runnable as plain
scripts.  This module therefore avoids package-relative imports and keeps the
captured record JSON-native so that a summary remains readable without the
original Python environment.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def utc_now() -> str:
    """Return an unambiguous, sortable UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def jsonable(value: Any) -> Any:
    """Recursively convert configs and CLI values into strict JSON values."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((jsonable(item) for item in value), key=repr)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_manifest(
    paths: Iterable[Path],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Hash an exact, ordered set of files, including an aggregate digest."""

    root = project_root.resolve()
    candidates = sorted({Path(path).expanduser().resolve() for path in paths})
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"manifest input is not a file: {path}")
        try:
            name = str(path.relative_to(root))
        except ValueError:
            name = str(path)
        digest = _sha256_file(path)
        size = path.stat().st_size
        files.append({"path": name, "bytes": size, "sha256": digest})
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
    return {
        "algorithm": "sha256",
        "aggregate_sha256": aggregate.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def source_manifest(project_root: Path, script_path: Path) -> dict[str, Any]:
    """Hash all HS-v2 Python sources plus the exact experiment launcher."""

    root = project_root.resolve()
    helper = Path(__file__).resolve()
    paths = list((root / "src" / "hunter_seeker_v2").rglob("*.py"))
    paths.extend((helper, Path(script_path).resolve()))
    return sha256_manifest(paths, project_root=root)


def _command(project_root: Path, arguments: Sequence[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            tuple(arguments),
            cwd=project_root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return {
            "command": list(arguments),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": list(arguments),
        "returncode": result.returncode,
        "stdout": result.stdout.rstrip("\n"),
        "stderr": result.stderr.rstrip("\n"),
    }


def git_state(project_root: Path) -> dict[str, Any]:
    """Record both repository identity and dirty state, including untracked V2."""

    head = _command(project_root, ("git", "rev-parse", "HEAD"))
    status = _command(
        project_root,
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
    )
    status_text = str(status["stdout"])
    return {
        "head": head["stdout"] if head["returncode"] == 0 else None,
        "head_command": head,
        "dirty": bool(status_text) if status["returncode"] == 0 else None,
        "status_entry_count": len(status_text.splitlines()) if status_text else 0,
        "status_porcelain": status,
        "status_sha256": hashlib.sha256(status_text.encode("utf-8")).hexdigest(),
    }


def runtime_environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for distribution in ("numpy", "torch", "arc-agi", "arcengine"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "package_versions": versions,
    }


@dataclass(slots=True)
class ProvenanceCapture:
    """Immutable-at-start evidence plus a monotonic duration clock."""

    base: dict[str, Any]
    _started_monotonic: float

    def finish(
        self,
        *,
        effective_agent_configs: Mapping[str, Any],
        ordering: Mapping[str, Any],
    ) -> dict[str, Any]:
        finished_at = utc_now()
        duration = max(0.0, time.perf_counter() - self._started_monotonic)
        return {
            **self.base,
            "timing": {
                "started_at_utc": self.base["started_at_utc"],
                "completed_at_utc": finished_at,
                "duration_seconds": round(duration, 6),
            },
            "effective_agent_configs": jsonable(effective_agent_configs),
            "ordering": jsonable(ordering),
        }


def start_provenance(
    *,
    args: Any,
    project_root: Path,
    script_path: Path,
    input_paths: Iterable[Path] = (),
) -> ProvenanceCapture:
    """Capture reproducibility evidence before the first live episode."""

    root = project_root.resolve()
    started_at = utc_now()
    started_monotonic = time.perf_counter()
    return ProvenanceCapture(
        base={
            "schema_version": 1,
            "started_at_utc": started_at,
            "arguments": jsonable(vars(args)),
            "argv": [str(Path(script_path).resolve()), *sys.argv[1:]],
            "working_directory": str(Path.cwd().resolve()),
            "environment": runtime_environment(),
            "git": git_state(root),
            "source_manifest": source_manifest(root, script_path),
            "input_manifest": sha256_manifest(input_paths, project_root=root),
        },
        _started_monotonic=started_monotonic,
    )


def prepare_summary_target(
    out_root: Path,
    *,
    overwrite: bool,
) -> Path:
    """Validate overwrite intent before expensive environment interaction."""

    root = Path(out_root).expanduser()
    target = root / "summary.json"
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing experiment summary: {target}; "
            "pass --overwrite to replace it explicitly"
        )
    root.mkdir(parents=True, exist_ok=True)
    return target


def write_summary(target: Path, payload: Any, *, overwrite: bool) -> None:
    """Write strict JSON, with exclusive creation unless overwrite was explicit."""

    encoded = json.dumps(
        jsonable(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if not overwrite:
        with target.open("x", encoding="utf-8") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as destination:
        temporary = Path(destination.name)
        destination.write(encoded)
        destination.flush()
        os.fsync(destination.fileno())
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="hs-v2-provenance-") as directory:
        root = Path(directory)
        sample = root / "sample.bin"
        sample.write_bytes(b"abc")
        manifest = sha256_manifest((sample,), project_root=root)
        assert manifest["file_count"] == 1
        assert manifest["files"][0]["sha256"] == hashlib.sha256(b"abc").hexdigest()
        target = prepare_summary_target(root / "out", overwrite=False)
        write_summary(target, {"ok": True}, overwrite=False)
        try:
            prepare_summary_target(root / "out", overwrite=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("exclusive summary guard did not reject overwrite")


if __name__ == "__main__":
    _self_test()
    print("hs_v2_experiment_provenance self-test passed")
