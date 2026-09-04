"""Fit and merge recurrent Jacobian-lens shards with integrity sidecars.

The binary lens is useful only together with the inputs and code that made
it. Every completed ``.pt`` therefore has a JSON sidecar containing the
prompt slice, model/jlens identity, source hashes, configuration, and the
hash of the bytes on disk. A merge refuses to consume anything that cannot
be verified against that contract.

Examples::

    python src/ouro_jlens/fit_lens.py fit --target-ut 3 --start 0 --end 100 \
        --dim-batch 8 --out artifacts/jlens/lens/exit3/shard_000_100.pt
    python src/ouro_jlens/fit_lens.py merge --out lens.pt shard_*.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import jlens

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouro_jlens.evidence import (  # noqa: E402
    SCHEMA_VERSION,
    aggregate_sha256,
    atomic_write_json,
    file_record,
    prompt_slice_sha256,
    sha256_file,
    sha256_json,
    source_manifest,
)
from ouro_jlens.recurrent import OURO_REVISION, PROJECT_ROOT, load_ouro  # noqa: E402

try:
    from jlens.fitting import SKIP_FIRST_N_POSITIONS
except (ImportError, AttributeError):  # pragma: no cover - only old jlens installs
    SKIP_FIRST_N_POSITIONS = 16

DEFAULT_PROMPTS = PROJECT_ROOT / "artifacts" / "jlens" / "data" / "wikitext_prompts.json"
LOG = logging.getLogger(__name__)


class IntegrityError(ValueError):
    """Raised when a lens or sidecar cannot be tied to its declared inputs."""


def sidecar_path(path: str | os.PathLike[str]) -> Path:
    """Return the metadata path paired with a binary lens path."""

    return Path(path).with_suffix(".json")


def checkpoint_sidecar_path(path: str | os.PathLike[str]) -> Path:
    """Return the metadata path paired with a resumable ``.ckpt`` file."""

    return Path(f"{path}.json")


def jlens_commit() -> str:
    """Return the installed jlens checkout revision."""

    try:
        repo = Path(jlens.__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "UNKNOWN"
    revision = result.stdout.strip()
    return revision or "UNKNOWN"


def _load_prompt_file(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read prompts file {path}: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("prompts")
    if not isinstance(payload, list) or not all(isinstance(prompt, str) for prompt in payload):
        raise IntegrityError(f"prompts file {path} must contain a JSON list of strings")
    return payload


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrityError(f"{name} must be an integer, got {value!r}")
    return value


def _validate_slice(start: Any, end: Any, n_prompts: int) -> tuple[int, int]:
    start, end = _integer(start, "start"), _integer(end, "end")
    if start < 0 or end <= start or end > n_prompts:
        raise IntegrityError(
            f"invalid prompt range [{start}, {end}) for {n_prompts} prompts; "
            "require 0 <= start < end <= len(prompts)"
        )
    return start, end


def _resolved_target(model: Any, target_ut: Any) -> tuple[int, int, list[int]]:
    target_ut = _integer(target_ut, "target_ut")
    n_ut = getattr(model, "n_ut", None)
    if not isinstance(n_ut, int) or isinstance(n_ut, bool) or n_ut <= 0:
        raise IntegrityError("loaded model does not expose a positive integer n_ut")
    if not 0 <= target_ut < n_ut:
        raise IntegrityError(f"target_ut={target_ut} out of range [0, {n_ut})")
    try:
        target_virtual = int(model.exit_index(target_ut))
    except (AssertionError, IndexError, TypeError, ValueError) as exc:
        raise IntegrityError(f"cannot resolve target_ut={target_ut}: {exc}") from exc
    n_layers = getattr(model, "n_layers", None)
    if not isinstance(n_layers, int) or target_virtual <= 0 or target_virtual >= n_layers:
        raise IntegrityError(
            f"resolved target virtual layer {target_virtual} is invalid for n_layers={n_layers}"
        )
    return target_ut, target_virtual, list(range(target_virtual))


def _path_if_exists(path: Path) -> Path:
    if not path.is_file():
        raise IntegrityError(f"required source file is missing: {path}")
    return path


def _source_identity() -> dict[str, Any]:
    """Hash the generator and the implementation it delegates to."""

    paths: list[Path] = [
        _path_if_exists(Path(__file__).resolve()),
        _path_if_exists(Path(__file__).with_name("recurrent.py").resolve()),
        _path_if_exists(Path(__file__).with_name("evidence.py").resolve()),
    ]
    fitting_file = getattr(getattr(jlens, "fitting", None), "__file__", None)
    if fitting_file:
        paths.append(_path_if_exists(Path(fitting_file).resolve()))
    manifest = source_manifest(paths)
    generator_record = manifest["files"][0]
    return {
        "source_files": manifest["files"],
        "source_sha256": manifest["sha256"],
        "generator_file": generator_record,
        "generator_sha256": generator_record["sha256"],
    }


def _prompt_identity(prompt_path: Path, prompts: list[str], start: int, end: int) -> dict[str, Any]:
    record = file_record(prompt_path)
    selected = prompts[start:end]
    digest = prompt_slice_sha256(selected)
    return {
        "prompt_file": record,
        "prompt_file_sha256": record["sha256"],
        "prompt_file_count": len(prompts),
        "prompt_slice": {
            "start": start,
            "end": end,
            "count": len(selected),
            "sha256": digest,
        },
        "prompt_slice_sha256": digest,
    }


def _base_metadata(
    model: Any,
    *,
    target_ut: int,
    target_virtual: int,
    source_layers: list[int],
    prompt_path: Path,
    prompts: list[str],
    start: int,
    end: int,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
    checkpoint_every: int | None,
) -> dict[str, Any]:
    n_physical = getattr(model, "n_physical", None)
    n_ut = getattr(model, "n_ut", None)
    n_layers = getattr(model, "n_layers", None)
    d_model = getattr(model, "d_model", None)
    if not all(
        isinstance(v, int) and not isinstance(v, bool)
        for v in (n_physical, n_ut, n_layers, d_model)
    ):
        raise IntegrityError(
            "loaded model must expose integer n_physical, n_ut, n_layers, and d_model"
        )
    if min(n_physical, n_ut, n_layers, d_model) <= 0:
        raise IntegrityError("loaded model dimensions must all be positive")
    source = _source_identity()
    prompts_identity = _prompt_identity(prompt_path, prompts, start, end)
    model_revision = str(getattr(model, "model_revision", OURO_REVISION))
    snapshot_path = getattr(model, "snapshot_path", None)
    if snapshot_path is None and model_revision == OURO_REVISION:
        snapshot_path = PROJECT_ROOT / "artifacts" / "hf_cache" / "hub" / (
            "models--ByteDance--Ouro-2.6B"
        ) / "snapshots" / OURO_REVISION
    model_files: list[dict[str, Any]] = []
    if snapshot_path is not None:
        snapshot = Path(snapshot_path)
        for name in ("model.safetensors", "config.json", "modeling_ouro.py", "tokenizer.json"):
            candidate = snapshot / name
            if candidate.is_file():
                model_files.append(file_record(candidate))
    model_bytes_status = "HASH_BOUND" if model_files else "REVISION_AND_SHAPE_ONLY"
    model_snapshot_sha256 = aggregate_sha256(
        {record["path"]: record["sha256"] for record in model_files}
        if model_files else {"revision": model_revision, "shape": sha256_json({
            "n_physical": n_physical, "n_ut": n_ut, "n_layers": n_layers, "d_model": d_model,
        })}
    )
    config = {
        "target_ut": target_ut,
        "target_virtual": target_virtual,
        "source_layers": source_layers,
        "n_physical": n_physical,
        "n_ut": n_ut,
        "n_layers": n_layers,
        "d_model": d_model,
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "skip_first": skip_first,
        "checkpoint_every": checkpoint_every,
        "virtual_index": "ut * n_physical + layer",
        "bos_prepended": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "fit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **config,
        "config": config,
        "model_revision": model_revision,
        "model": {
            "revision": model_revision,
            "n_physical": n_physical,
            "n_ut": n_ut,
            "n_layers": n_layers,
            "d_model": d_model,
            "bytes_status": model_bytes_status,
            "files": model_files,
            "aggregate_sha256": model_snapshot_sha256,
        },
        "model_snapshot_sha256": model_snapshot_sha256,
        "jlens_commit": jlens_commit(),
        **prompts_identity,
        **source,
        "prompts": str(prompt_path),
        "start": start,
        "end": end,
        "n_requested": end - start,
        "seconds": None,
    }


# These fields define compatibility between independent shards. Runtime and
# output-only fields are deliberately absent.
IDENTITY_FIELDS = (
    "target_ut",
    "target_virtual",
    "source_layers",
    "n_physical",
    "n_ut",
    "n_layers",
    "d_model",
    "dim_batch",
    "max_seq_len",
    "skip_first",
    "checkpoint_every",
    "virtual_index",
    "bos_prepended",
    "model_revision",
    "model_snapshot_sha256",
    "jlens_commit",
    "prompt_file_sha256",
    "prompt_file_count",
    "source_sha256",
    "generator_sha256",
)


def identity_projection(meta: dict[str, Any]) -> dict[str, Any]:
    """Return the fields that must agree for shards/lenses to be combined."""

    return {field: meta.get(field) for field in IDENTITY_FIELDS}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid sidecar {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrityError(f"sidecar {path} must contain a JSON object")
    return payload


def _verify_binary(path: Path, meta: dict[str, Any]) -> None:
    if not path.is_file():
        raise IntegrityError(f"binary output is missing: {path}")
    output = meta.get("output")
    declared = meta.get("output_sha256")
    if not isinstance(output, dict) or not output.get("path"):
        raise IntegrityError(f"sidecar for {path} has no output file record")
    if isinstance(output, dict):
        declared = output.get("sha256", declared)
    if not isinstance(declared, str) or len(declared) != 64:
        raise IntegrityError(f"sidecar for {path} has no output SHA-256")
    actual = sha256_file(path)
    if actual != declared:
        raise IntegrityError(
            f"output hash mismatch for {path}: expected {declared}, got {actual}"
        )
    if output.get("size") is not None:
        if int(output["size"]) != path.stat().st_size:
            raise IntegrityError(f"output size mismatch for {path}")


def validate_sidecar(path: str | os.PathLike[str], *, kind: str = "fit") -> dict[str, Any]:
    """Load and verify a completed lens and its sidecar."""

    binary = Path(path)
    metadata_path = sidecar_path(binary)
    if not metadata_path.is_file():
        raise IntegrityError(f"missing sidecar for {binary}: {metadata_path}")
    meta = _read_json(metadata_path)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise IntegrityError(f"unsupported sidecar schema in {metadata_path}")
    if meta.get("kind") != kind:
        raise IntegrityError(
            f"sidecar {metadata_path} has kind={meta.get('kind')!r}, expected {kind!r}"
        )
    required = (
        "source_layers",
        "target_ut",
        "target_virtual",
        "model",
        "model_revision",
        "model_snapshot_sha256",
        "jlens_commit",
        "prompt_file",
        "prompt_slice",
        "prompt_file_sha256",
        "prompt_slice_sha256",
        "start",
        "end",
        "n_requested",
        "n_fitted",
        "source_files",
        "source_sha256",
        "generator_file",
        "generator_sha256",
        "output",
        "output_sha256",
    )
    missing = [field for field in required if field not in meta]
    if missing:
        raise IntegrityError(f"sidecar {metadata_path} is missing required fields: {missing}")
    if not isinstance(meta.get("source_layers"), list):
        raise IntegrityError(f"sidecar {metadata_path} has invalid source_layers")
    model_record = meta.get("model")
    if not isinstance(model_record, dict) or meta.get("model_revision") != model_record.get("revision"):
        raise IntegrityError(f"model identity mismatch in {metadata_path}")
    model_files = model_record.get("files", [])
    if not isinstance(model_files, list):
        raise IntegrityError(f"invalid model file manifest in {metadata_path}")
    if model_record.get("bytes_status") == "HASH_BOUND":
        if not model_files:
            raise IntegrityError(f"hash-bound model has no file records in {metadata_path}")
        model_digests: dict[str, str] = {}
        for record in model_files:
            if not isinstance(record, dict) or not record.get("path"):
                raise IntegrityError(f"invalid model file record in {metadata_path}")
            actual = file_record(Path(record["path"]))
            if actual != record:
                raise IntegrityError(f"model file hash/size mismatch: {record.get('path')}")
            model_digests[record["path"]] = record["sha256"]
        if aggregate_sha256(model_digests) != model_record.get("aggregate_sha256"):
            raise IntegrityError(f"model aggregate digest mismatch in {metadata_path}")
    if meta.get("model_snapshot_sha256") != model_record.get("aggregate_sha256"):
        raise IntegrityError(f"model snapshot digest field mismatch in {metadata_path}")
    _verify_binary(binary, meta)
    # Verify the prompt/source records still describe the bytes consumed by
    # this process. This catches a modified source file even when the output
    # itself was not modified.
    prompt_file = meta.get("prompt_file")
    if not isinstance(prompt_file, dict) or not prompt_file.get("path"):
        raise IntegrityError(f"invalid prompt file record for {binary}")
    prompt_path = Path(prompt_file["path"])
    if not prompt_path.is_file():
        raise IntegrityError(f"prompt file is missing: {prompt_path}")
    prompt_record = file_record(prompt_path)
    if prompt_record != prompt_file:
        raise IntegrityError(f"prompt file hash/size mismatch for {binary}")
    if meta.get("prompt_file_sha256") != prompt_record["sha256"]:
        raise IntegrityError(f"prompt file digest field mismatch for {binary}")
    prompts = _load_prompt_file(prompt_path)
    if meta.get("prompt_file_count") != len(prompts):
        raise IntegrityError(f"prompt file count mismatch for {binary}")
    prompt_slice = meta.get("prompt_slice")
    if not isinstance(prompt_slice, dict):
        raise IntegrityError(f"invalid prompt slice record for {binary}")
    start, end = prompt_slice.get("start"), prompt_slice.get("end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end <= start
        or end > len(prompts)
        or prompt_slice.get("count") != end - start
    ):
        raise IntegrityError(f"invalid prompt slice bounds for {binary}")
    slice_digest = prompt_slice_sha256(prompts[start:end])
    if prompt_slice.get("sha256") != slice_digest or meta.get("prompt_slice_sha256") != slice_digest:
        raise IntegrityError(f"prompt slice hash mismatch for {binary}")
    source_records = meta.get("source_files", [])
    if not isinstance(source_records, list) or not source_records:
        raise IntegrityError(f"invalid source manifest in {metadata_path}")
    source_digests: dict[str, str] = {}
    for record in source_records:
        if not isinstance(record, dict) or not record.get("path"):
            raise IntegrityError(f"invalid source record in {metadata_path}")
        source_path = Path(record["path"])
        if not source_path.is_file():
            raise IntegrityError(f"source file is missing: {source_path}")
        actual = file_record(source_path)
        if actual != record:
            raise IntegrityError(f"source file hash/size mismatch: {source_path}")
        source_digests[record["path"]] = record["sha256"]
    if meta.get("source_sha256") != aggregate_sha256(source_digests):
        raise IntegrityError(f"source manifest digest mismatch for {binary}")
    generator = meta.get("generator_file")
    if (
        not isinstance(generator, dict)
        or meta.get("generator_sha256") != generator.get("sha256")
        or generator not in source_records
    ):
        raise IntegrityError(f"generator identity mismatch for {binary}")
    return meta


def _atomic_lens_save(lens: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        lens.save(str(temporary))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_fit_args(args: Any) -> tuple[Path, int, int, int, int, int]:
    prompt_path = Path(getattr(args, "prompts", DEFAULT_PROMPTS))
    start = _integer(getattr(args, "start", 0), "start")
    end = _integer(getattr(args, "end", 100), "end")
    dim_batch = _integer(getattr(args, "dim_batch", 8), "dim_batch")
    max_seq_len = _integer(getattr(args, "max_seq_len", 128), "max_seq_len")
    skip_first = _integer(getattr(args, "skip_first", SKIP_FIRST_N_POSITIONS), "skip_first")
    checkpoint_every = getattr(args, "checkpoint_every", None)
    if checkpoint_every is not None:
        checkpoint_every = _integer(checkpoint_every, "checkpoint_every")
        if checkpoint_every <= 0:
            raise IntegrityError(f"checkpoint_every must be > 0 or None, got {checkpoint_every}")
    if dim_batch <= 0:
        raise IntegrityError(f"dim_batch must be > 0, got {dim_batch}")
    if max_seq_len <= 1:
        raise IntegrityError(f"max_seq_len must be > 1, got {max_seq_len}")
    if skip_first < 0:
        raise IntegrityError(f"skip_first must be >= 0, got {skip_first}")
    return prompt_path, start, end, dim_batch, max_seq_len, skip_first


def _checkpoint_contract(meta: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(meta)
    result["kind"] = "fit_checkpoint"
    result.pop("output", None)
    result.pop("output_sha256", None)
    return result


def _prepare_checkpoint(checkpoint: Path, expected: dict[str, Any]) -> None:
    metadata_path = checkpoint_sidecar_path(checkpoint)
    if checkpoint.exists():
        if not metadata_path.is_file():
            raise IntegrityError(f"checkpoint exists without sidecar: {checkpoint}")
        found = _read_json(metadata_path)
        if found.get("schema_version") != SCHEMA_VERSION or found.get("kind") != "fit_checkpoint":
            raise IntegrityError(f"invalid checkpoint sidecar: {metadata_path}")
        if identity_projection(found) != identity_projection(expected):
            raise IntegrityError(f"checkpoint configuration/model/prompt mismatch: {checkpoint}")
        if found.get("start") != expected.get("start") or found.get("end") != expected.get("end"):
            raise IntegrityError(f"checkpoint prompt range mismatch: {checkpoint}")
        checkpoint_record = found.get("checkpoint")
        if checkpoint_record is None:
            raise IntegrityError(
                f"checkpoint was never sealed with a digest and cannot be resumed: {checkpoint}"
            )
        if not isinstance(checkpoint_record, dict):
            raise IntegrityError(f"invalid checkpoint record: {metadata_path}")
        actual = file_record(checkpoint)
        if actual != checkpoint_record:
            raise IntegrityError(f"checkpoint hash/size mismatch: {checkpoint}")
    elif metadata_path.exists():
        raise IntegrityError(f"stale checkpoint sidecar without checkpoint: {metadata_path}")
    else:
        atomic_write_json(metadata_path, _checkpoint_contract(expected))


def fit(args: Any) -> Any:
    """Fit one prompt range and atomically publish a verified lens."""

    out = Path(getattr(args, "out"))
    prompt_path, start, end, dim_batch, max_seq_len, skip_first = _validate_fit_args(args)
    prompts = _load_prompt_file(prompt_path)
    start, end = _validate_slice(start, end, len(prompts))

    # Validate target and construct the identity before running the expensive
    # fit. This also ensures that an existing output can be checked without
    # silently overwriting it.
    model = load_ouro()
    target_ut, target_virtual, source_layers = _resolved_target(model, getattr(args, "target_ut"))
    expected = _base_metadata(
        model,
        target_ut=target_ut,
        target_virtual=target_virtual,
        source_layers=source_layers,
        prompt_path=prompt_path,
        prompts=prompts,
        start=start,
        end=end,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        checkpoint_every=getattr(args, "checkpoint_every", None),
    )

    # A complete matching output is idempotent. Any partial or stale output
    # is rejected rather than replaced, because replacement could hide a
    # failed prior run from a reviewer.
    if out.exists():
        found = validate_sidecar(out, kind="fit")
        if (
            identity_projection(found) != identity_projection(expected)
            or found.get("start") != start
            or found.get("end") != end
        ):
            raise IntegrityError(f"existing output has incompatible sidecar: {out}")
        LOG.info("verified existing lens %s; nothing to do", out)
        return None
    if sidecar_path(out).exists():
        raise IntegrityError(f"stale sidecar without output: {sidecar_path(out)}")

    checkpoint = Path(getattr(args, "checkpoint", None) or f"{out}.ckpt")
    _prepare_checkpoint(checkpoint, expected)
    t0 = time.perf_counter()
    try:
        lens = jlens.fit(
            model,
            prompts[start:end],
            source_layers=source_layers,
            target_layer=target_virtual,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
            checkpoint_path=str(checkpoint),
            checkpoint_every=getattr(args, "checkpoint_every", None),
            resume=True,
        )
    except BaseException:
        # A checkpoint is resumable only when this wrapper observed a complete
        # checkpoint file and sealed its bytes.  An uncatchable process/pod
        # loss leaves the pre-run sidecar unsealed and the next run fails
        # closed instead of trusting a possibly partial file.
        if checkpoint.is_file():
            checkpoint_meta = _checkpoint_contract(expected)
            checkpoint_meta["checkpoint"] = file_record(checkpoint)
            checkpoint_meta["status"] = "INTERRUPTED_RESUMABLE"
            atomic_write_json(checkpoint_sidecar_path(checkpoint), checkpoint_meta)
        raise
    elapsed = round(time.perf_counter() - t0, 3)
    if list(getattr(lens, "source_layers", [])) != source_layers:
        raise IntegrityError("jlens.fit returned unexpected source_layers")
    if int(getattr(lens, "d_model", -1)) != int(expected["d_model"]):
        raise IntegrityError("jlens.fit returned unexpected d_model")

    _atomic_lens_save(lens, out)
    metadata = copy.deepcopy(expected)
    metadata.update(
        {
            "seconds": elapsed,
            "n_prompts": int(getattr(lens, "n_prompts", 0)),
            "n_fitted": int(getattr(lens, "n_prompts", 0)),
        }
    )
    output = file_record(out)
    metadata["output"] = output
    metadata["output_sha256"] = output["sha256"]
    atomic_write_json(sidecar_path(out), metadata)
    if checkpoint.is_file():
        checkpoint_meta = _checkpoint_contract(expected)
        checkpoint_meta["checkpoint"] = file_record(checkpoint)
        atomic_write_json(checkpoint_sidecar_path(checkpoint), checkpoint_meta)
    LOG.info("saved %s (%d fitted prompts)", out, metadata["n_fitted"])
    return lens


def _range_from_meta(meta: dict[str, Any], path: Path) -> tuple[int, int]:
    start, end = meta.get("start"), meta.get("end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        prompt_slice = meta.get("prompt_slice", {})
        start, end = prompt_slice.get("start"), prompt_slice.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        raise IntegrityError(f"invalid prompt range in {path}")
    declared = meta.get("n_requested")
    if declared is not None and declared != end - start:
        raise IntegrityError(f"prompt count/range mismatch in {path}")
    return start, end


def _validate_lens_shape(lens: Any, meta: dict[str, Any], path: Path) -> None:
    expected_layers = meta.get("source_layers")
    if not isinstance(expected_layers, list) or list(getattr(lens, "source_layers", [])) != expected_layers:
        raise IntegrityError(f"lens source_layers disagree with sidecar: {path}")
    d_model = int(meta.get("d_model", -1))
    if int(getattr(lens, "d_model", -2)) != d_model:
        raise IntegrityError(f"lens d_model disagrees with sidecar: {path}")
    n_prompts = int(getattr(lens, "n_prompts", -1))
    declared = meta.get("n_fitted", meta.get("n_prompts"))
    if declared is not None and n_prompts != int(declared):
        raise IntegrityError(f"lens n_prompts disagrees with sidecar: {path}")
    if n_prompts <= 0:
        raise IntegrityError(f"lens contains no fitted prompts: {path}")
    for layer in expected_layers:
        matrix = getattr(lens, "jacobians", {}).get(layer)
        if matrix is None or tuple(matrix.shape) != (d_model, d_model):
            raise IntegrityError(f"invalid Jacobian shape at layer {layer} in {path}")


def _merged_prompt_identity(first: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    prompt_file = first.get("prompt_file", {})
    path = Path(prompt_file.get("path", first.get("prompts", "")))
    result: dict[str, Any] = {"start": start, "end": end, "count": end - start}
    if path.is_file():
        prompts = _load_prompt_file(path)
        if end > len(prompts):
            raise IntegrityError(f"merged range [{start}, {end}) exceeds prompt file length")
        result["sha256"] = prompt_slice_sha256(prompts[start:end])
    else:
        # Keep merge usable after a verified artifact is relocated, while
        # retaining an identity that cannot be confused with a raw assertion.
        result["sha256"] = sha256_json(
            {
                "prompt_file_sha256": first.get("prompt_file_sha256"),
                "start": start,
                "end": end,
            }
        )
    return result


def merge(args: Any) -> Any:
    """Verify contiguous non-overlapping shards and publish their weighted mean."""

    raw_shards = [Path(p) for p in getattr(args, "shards", [])]
    if not raw_shards:
        raise IntegrityError("merge requires at least one shard")
    canonical = [p.resolve() for p in raw_shards]
    if len(set(canonical)) != len(canonical):
        raise IntegrityError("duplicate shard path")
    out = Path(getattr(args, "out"))
    if out.resolve() in set(canonical):
        raise IntegrityError("merge output must not overwrite an input shard")

    entries: list[tuple[int, int, Path, dict[str, Any], Any]] = []
    for path in raw_shards:
        meta = validate_sidecar(path, kind="fit")
        target_virtual = meta.get("target_virtual")
        if not isinstance(target_virtual, int) or target_virtual <= 0:
            raise IntegrityError(f"invalid target_virtual in {path}")
        if meta.get("source_layers") != list(range(target_virtual)):
            raise IntegrityError(f"source_layers are not exact contiguous range in {path}")
        lens = jlens.JacobianLens.load(str(path))
        _validate_lens_shape(lens, meta, path)
        start, end = _range_from_meta(meta, path)
        entries.append((start, end, path, meta, lens))

    entries.sort(key=lambda entry: (entry[0], entry[1], str(entry[2])))
    baseline = entries[0][3]
    for _, _, path, meta, _ in entries[1:]:
        if identity_projection(meta) != identity_projection(baseline):
            raise IntegrityError(f"shard configuration/model/prompt/source mismatch: {path}")
    for previous, current in zip(entries, entries[1:]):
        if current[0] < previous[1]:
            raise IntegrityError(f"overlapping shard ranges: {previous[2]} and {current[2]}")
        if current[0] > previous[1]:
            raise IntegrityError(f"gap in shard ranges: {previous[1]}..{current[0]}")

    start, end = entries[0][0], entries[-1][1]
    if out.exists():
        if not sidecar_path(out).is_file():
            raise IntegrityError(f"merge output exists without sidecar: {out}")
        found = validate_sidecar(out, kind="merged")
        if (
            identity_projection(found) != identity_projection(baseline)
            or found.get("start") != start
            or found.get("end") != end
        ):
            raise IntegrityError(f"existing merge output has incompatible sidecar: {out}")
        LOG.info("verified existing merged lens %s; nothing to do", out)
        return None
    if sidecar_path(out).exists():
        raise IntegrityError(f"stale merge sidecar without output: {sidecar_path(out)}")

    merged = jlens.JacobianLens.merge([entry[4] for entry in entries])
    _atomic_lens_save(merged, out)
    metadata = copy.deepcopy(baseline)
    metadata["kind"] = "merged"
    metadata["start"], metadata["end"] = start, end
    metadata["n_requested"] = end - start
    metadata["prompt_slice"] = _merged_prompt_identity(baseline, start, end)
    metadata["prompt_slice_sha256"] = metadata["prompt_slice"]["sha256"]
    metadata["n_prompts"] = int(merged.n_prompts)
    metadata["n_fitted"] = int(merged.n_prompts)
    metadata["shards"] = [str(entry[2]) for entry in entries]
    metadata["shard_records"] = [
        {
            "path": str(entry[2]),
            "start": entry[0],
            "end": entry[1],
            "output_sha256": entry[3].get(
                "output_sha256", entry[3].get("output", {}).get("sha256")
            ),
            "sidecar_sha256": sha256_file(sidecar_path(entry[2])),
        }
        for entry in entries
    ]
    output = file_record(out)
    metadata["output"] = output
    metadata["output_sha256"] = output["sha256"]
    atomic_write_json(sidecar_path(out), metadata)
    LOG.info("merged %d shards, %d fitted prompts -> %s", len(entries), merged.n_prompts, out)
    return merged


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--target-ut", type=int, required=True)
    fit_parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    fit_parser.add_argument("--start", type=int, default=0)
    fit_parser.add_argument("--end", type=int, default=100)
    fit_parser.add_argument("--dim-batch", type=int, default=8)
    fit_parser.add_argument("--max-seq-len", type=int, default=128)
    fit_parser.add_argument("--skip-first", type=int, default=SKIP_FIRST_N_POSITIONS)
    fit_parser.add_argument("--checkpoint-every", type=int, default=None)
    fit_parser.add_argument("--checkpoint")
    fit_parser.add_argument("--out", required=True)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--out", required=True)
    merge_parser.add_argument("shards", nargs="+")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        fit(args) if args.cmd == "fit" else merge(args)
    except (IntegrityError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    main()
