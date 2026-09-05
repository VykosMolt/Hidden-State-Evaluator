"""Build the canonical, machine-readable JLens result from retained artifacts.

The report is an evidence roll-up, not an approval switch.  In particular, a
boolean in a sidecar, a filename containing ``n80``, or a completed-looking
directory cannot promote a claim.  Optional artifacts are retained when they
are malformed or old, but their lineage is carried forward as a degraded
state.  This lets the report describe the useful arithmetic in the historical
evaluation while keeping current-generation claims closed until the producer
contract is actually verified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ouro_jlens import analyze
from ouro_jlens.evidence import aggregate_sha256, atomic_write_json, file_record, sha256_json
from ouro_jlens.recurrent import OURO_REVISION, OURO_SNAPSHOT, model_snapshot_files
from ouro_jlens.validate import (
    _valid_image_digest,
    approved_b300_image_digest,
    derive_milestone_passes,
    derive_validation_rollup,
    paid_validation_accepted,
)


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINEAGE_UNFROZEN = "LOCAL_OBSERVATIONAL_ARITHMETIC_REPRODUCIBLE_LINEAGE_UNFROZEN"
CURRENT_GENERATION = "CURRENT_GENERATION_EVIDENCE_VERIFIED"
VALIDATION_MILESTONES = (
    "m1_noninterference",
    "m2_exit_equality",
    "m3_recurrent_identity",
    "m4_distinct_vjps",
    "m5_stock_consistency",
)
VALIDATION_PROVENANCE_FIELDS = {
    "backend",
    "created_at",
    "cuda",
    "jlens_commit",
    "model_revision",
    "model_shape",
    "model_bytes",
    "platform",
    "package_versions",
    "python",
    "status",
    "source_files",
    "source_sha256",
    "torch",
    "jlens_source",
    "image_digest",
}
VALIDATION_PACKAGE_VERSIONS = frozenset({
    "torch",
    "transformers",
    "jlens",
    "numpy",
    "safetensors",
    "accelerate",
    "huggingface-hub",
})
PROBE_RUNTIME_VERSIONS = frozenset({
    "python",
    "numpy",
    "torch",
    "scikit-learn",
    "threadpoolctl",
    "jlens",
    "transformers",
    "safetensors",
    "accelerate",
    "huggingface-hub",
})
CHECKPOINT_RUNTIME_VERSIONS = frozenset({
    "python",
    "numpy",
    "torch",
    "jlens",
    "transformers",
    "safetensors",
    "accelerate",
    "huggingface-hub",
})
EVALUATION_PROVENANCE_FIELDS = {
    "schema_version",
    "created_at",
    "validation",
    "model",
    "model_snapshot_sha256",
    "jlens_commit",
    "config",
    "inputs",
    "lens_inputs",
    "source_files",
    "source_sha256",
    "outputs",
    "item_count",
    "correct_count",
    "item_metadata_sha256",
}


# These are templates only.  Evidence-backed statuses are assigned by
# ``_derive_claim_statuses`` below, so adding a row here cannot create a new
# supported claim by accident.
CLAIMS = {
    "instrumentation": {
        "status": "PENDING_FRESH_VALIDATION",
        "claim": "A complete current-source M1-M5 validation artifact is required before instrumentation is supported.",
    },
    "multihop_relative_deficit": {
        "status": "NOT_ESTABLISHED",
        "claim": "The retained multihop table is descriptive only after its input identity and fit size are verified.",
    },
    "arithmetic_relative_deficit": {
        "status": "NOT_ESTABLISHED",
        "claim": "The retained arithmetic table is descriptive only after its input identity and fit size are verified.",
    },
    "local_eventual_convergence": {
        "status": "NOT_ESTABLISHED",
        "claim": "The local/eventual comparison is not established without a structurally valid retained comparison.",
    },
    "lens_free_exit_agreement": {
        "status": "NOT_ESTABLISHED",
        "claim": "Lens-free exit agreement is descriptive only on verified retained populations.",
    },
    "large_fit_robustness": {
        "status": "INCONCLUSIVE",
        "claim": "Nested fits do not establish behavior at n=1000 or fixed-n variability.",
    },
    "cross_loop_association": {
        "status": "INCONCLUSIVE",
        "claim": "Cross-loop matrices are descriptive and do not identify a causal mechanism.",
    },
    "transport_mechanism": {
        "status": "INCONCLUSIVE",
        "claim": "Norm/scatter patterns do not establish prompt-specific rewriting.",
    },
    "supervised_probe": {
        "status": "INCONCLUSIVE",
        "claim": "The arithmetic probe is not supported without a complete, current provenance chain.",
    },
    "probe_familywide_inference": {
        "status": "INCONCLUSIVE",
        "claim": "Probe intervals are pointwise and do not establish a multiplicity-adjusted family-wide result.",
    },
    "checkpoint_comparison": {
        "status": "INCONCLUSIVE",
        "claim": "Checkpoint trends are not established without a structurally valid retained table.",
    },
    "estimator_independence": {
        "status": "INCONCLUSIVE",
        "claim": "The matched 2x2 position-by-reduction control has not been run.",
    },
    "architecture_cause": {
        "status": "PLAUSIBLE_HYPOTHESIS",
        "claim": "Shared-head recurrent supervision is a plausible contributor, not a demonstrated cause.",
    },
    "b300_validation": {
        "status": "NOT_RETAINED",
        "claim": "No current hash-bound validation artifact records execution on a B300 device.",
    },
    "submission_readiness": {
        "status": "NOT_ESTABLISHED",
        "claim": "The local observational result is not a completed mechanism study.",
    },
}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))


def _record_shape(record: object) -> bool:
    """Whether *record* has the complete byte-record shape.

    A path-only or hash-only record is not provenance.  ``bool`` is rejected
    for sizes because JSON's boolean subtype of integer otherwise creates a
    surprisingly easy one-field forgery.
    """

    return (
        isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and bool(record["path"])
        and isinstance(record.get("size"), int)
        and not isinstance(record["size"], bool)
        and record["size"] >= 0
        and _is_sha256(record.get("sha256"))
    )


def _record_hashes(value: object) -> dict[str, str]:
    """Collect path-to-digest bindings from nested complete file records."""

    found: dict[str, str] = {}
    if isinstance(value, dict):
        if _record_shape(value):
            found[value["path"]] = value["sha256"]
        for child in value.values():
            found.update(_record_hashes(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_record_hashes(child))
    return dict(sorted(found.items()))


def _path_candidates(path: str | Path) -> list[Path]:
    candidate = Path(path)
    if candidate.is_absolute():
        return [candidate]
    return [PROJECT_ROOT / candidate, candidate]


def _record_matches(record: object, expected: Path | None = None) -> bool:
    """Validate a record and, when possible, its current bytes."""

    if not _record_shape(record):
        return False
    declared = Path(record["path"])
    if expected is not None:
        try:
            # Producers use either an absolute path, a project-relative path,
            # or a path relative to the output/test bundle root.  Accept only
            # those explicit identities; never infer a fit from a filename.
            expected_resolved = expected.resolve()
            possible = {
                declared.resolve(),
                (PROJECT_ROOT / declared).resolve(),
                (expected.parent / declared).resolve(),
            }
            if expected_resolved not in possible:
                return False
        except OSError:
            return False
        candidates = [expected]
    else:
        candidates = _path_candidates(declared)
    for candidate in candidates:
        try:
            actual = file_record(candidate)
        except (OSError, ValueError):
            continue
        if actual["size"] == record["size"] and actual["sha256"] == record["sha256"]:
            return True
    return False


def _artifact_lens_record_matches(
    record: object,
    *,
    root: Path = PROJECT_ROOT,
    artifact_root: Path | None,
    expected: Path | None = None,
) -> Path | None:
    """Resolve a synchronized lens record without remapping other provenance."""

    try:
        from ouro_jlens.evaluate import _record_matches_artifact

        return _record_matches_artifact(
            record,
            root=root,
            label="evaluation lens artifact",
            artifact_root=artifact_root,
            expected=expected,
        )
    except Exception:
        return None


def _canonical_artifact_record(path: Path, artifact_root: Path | None = None) -> dict[str, Any]:
    """Record a result byte with its stable ``artifacts/jlens`` logical name."""

    if artifact_root is None:
        return file_record(path)
    from ouro_jlens.evaluate import _artifact_root
    from ouro_jlens.fit_lens import reject_symlink_path

    root = _artifact_root(artifact_root)
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        return file_record(path)
    # ``_artifact_root``/the lexical walk rejects links before the logical
    # name is reconstructed.
    reject_symlink_path(candidate)
    record = file_record(candidate)
    record["path"] = f"artifacts/jlens/{relative}"
    return record


def _record_points_to(record: object, expected: Path) -> bool:
    """Check that a complete record names the intended path, not a copy."""

    if not _record_shape(record):
        return False
    try:
        declared = Path(record["path"])
        expected_resolved = expected.resolve()
        return expected_resolved in {
            declared.resolve(),
            (PROJECT_ROOT / declared).resolve(),
            (expected.parent / declared).resolve(),
        }
    except (OSError, TypeError, ValueError):
        return False


def _probe_runtime_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == PROBE_RUNTIME_VERSIONS
        and all(isinstance(key, str) and isinstance(version, str) and version for key, version in value.items())
        and all(version != "NOT_INSTALLED" for version in value.values())
    )


def _declared_records(value: object) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(value, list):
        return [], ["source_files is not a list"]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, record in enumerate(value):
        if not _record_shape(record):
            errors.append(f"source_files[{index}] is not a complete file record")
        else:
            records.append(record)
    return records, errors


def _source_records_valid(
    payload: Mapping[str, Any], *, expected_names: set[str] | None = None
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    records, errors = _declared_records(payload.get("source_files"))
    if not records:
        errors.append("source_files is empty")
    aggregate = aggregate_sha256({record["path"]: record["sha256"] for record in records}) if records else None
    if payload.get("source_sha256") != aggregate:
        errors.append("source_sha256 does not match source_files")
    if expected_names is not None:
        names = {Path(record["path"]).name for record in records}
        paths = [record["path"] for record in records]
        if len(records) != len(expected_names) or len(set(paths)) != len(paths):
            errors.append("source_files contains duplicate or extra records")
        if names != expected_names:
            errors.append(f"source_files names {sorted(names)} != {sorted(expected_names)}")
    for record in records:
        if not _record_matches(record):
            errors.append(f"source file bytes are unavailable or changed: {record.get('path')}")
        if expected_names is not None:
            # Basename equality alone permits a forged ``/tmp/validate.py``
            # record.  Bind each producer to the actual module under this
            # checkout, while still accepting the relocatable project-relative
            # spelling emitted by current producers.
            name = Path(record["path"]).name
            expected = PROJECT_ROOT / "src" / "ouro_jlens" / name
            try:
                declared = Path(record["path"])
                resolved = declared.resolve() if declared.is_absolute() else (PROJECT_ROOT / declared).resolve()
                if resolved != expected.resolve():
                    errors.append(f"source file path is not current module: {record.get('path')}")
            except OSError:
                errors.append(f"source file path cannot be resolved: {record.get('path')}")
    return not errors, errors, records


def _logical_records_valid(
    value: object, *, root: Path, prefix: str, label: str
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    """Validate a prefixed logical record list against a concrete root."""

    errors: list[str] = []
    if not isinstance(value, dict) or not isinstance(value.get("files"), list) or not value["files"]:
        return False, [f"{label} file manifest is missing"], []
    records = value["files"]
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not _record_shape(record):
            errors.append(f"{label}[{index}] is not a complete file record")
            continue
        logical = str(record["path"])
        expected_prefix = f"{prefix}/"
        if not logical.startswith(expected_prefix) or logical in seen:
            errors.append(f"{label}[{index}] logical path is invalid or duplicated")
            continue
        seen.add(logical)
        relative = Path(logical[len(expected_prefix):])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"{label}[{index}] escapes its declared root")
            continue
        expected = root / relative
        try:
            actual = file_record(expected)
        except (OSError, ValueError):
            errors.append(f"{label}[{index}] bytes are unavailable: {logical}")
            continue
        if record["size"] != actual["size"] or record["sha256"] != actual["sha256"]:
            errors.append(f"{label}[{index}] bytes have changed: {logical}")
    aggregate = aggregate_sha256({record["path"]: record["sha256"] for record in records if _record_shape(record)})
    if value.get("aggregate_sha256") != aggregate:
        errors.append(f"{label} aggregate digest does not match its records")
    return not errors, errors, [record for record in records if _record_shape(record)]


def _record_matches_logical(record: object, expected: Path, logical_path: str) -> bool:
    """Match a current producer's logical path and the bytes it names."""

    if not _record_shape(record) or record.get("path") != logical_path:
        return False
    try:
        actual = file_record(expected)
    except (OSError, ValueError):
        return False
    return record["size"] == actual["size"] and record["sha256"] == actual["sha256"]


def _probe_source_records_valid(
    records: object,
    source_sha256: object,
    *,
    names: tuple[str, ...] = (
        "probe_cv.py",
        "probe.py",
        "evaluate.py",
        "evaldata.py",
        "recurrent.py",
        "evidence.py",
    ),
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    declared, errors = _declared_records(records)
    expected_project = {f"src/ouro_jlens/{name}" for name in names}
    project_records = {
        record["path"]: record
        for record in declared
        if record.get("path") in expected_project
    }
    dependency_records = {
        record["path"]: record
        for record in declared
        if isinstance(record.get("path"), str)
        and record["path"].startswith("dependency/jlens/")
    }
    if set(project_records) != expected_project:
        errors.append("probe project source manifest is incomplete")
    for logical, record in project_records.items():
        if not _record_matches_logical(record, PROJECT_ROOT / logical, logical):
            errors.append(f"probe source record is missing or changed: {logical}")
    try:
        from ouro_jlens.probe import _jlens_package_root, _jlens_source_paths

        jlens_root = _jlens_package_root()
        expected_dependency = {
            f"dependency/jlens/{path.relative_to(jlens_root).as_posix()}"
            for path in _jlens_source_paths()
        }
    except (ImportError, OSError, ValueError):
        jlens_root = None
        expected_dependency = set()
        errors.append("probe installed jlens source is unavailable")
    if set(dependency_records) != expected_dependency:
        errors.append("probe dependency source manifest is incomplete")
    if jlens_root is not None:
        for logical, record in dependency_records.items():
            relative = Path(logical[len("dependency/jlens/"):])
            if not _record_matches_logical(record, jlens_root / relative, logical):
                errors.append(f"probe dependency source is missing or changed: {logical}")
    expected_count = len(expected_project) + len(expected_dependency)
    if len(declared) != expected_count or len({record["path"] for record in declared}) != len(declared):
        errors.append("probe source manifest contains duplicate or extra records")
    if aggregate_sha256({record["path"]: record["sha256"] for record in declared}) != source_sha256:
        errors.append("probe source aggregate does not match source records")
    return not errors, declared, errors


def _probe_model_records_valid(records: object) -> tuple[bool, list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not isinstance(records, list) or not records:
        return False, [], ["probe model manifest is missing"]
    returned: list[dict[str, Any]] = []
    declared_paths: set[str] = set()
    for record in records:
        if not _record_shape(record) or not isinstance(record.get("path"), str) or not record["path"].startswith("model_snapshot/"):
            errors.append("probe model record is malformed")
            continue
        relative = Path(record["path"][len("model_snapshot/"):])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append("probe model record escapes the snapshot")
            continue
        if record["path"] in declared_paths:
            errors.append("probe model manifest contains duplicate paths")
        declared_paths.add(record["path"])
        if not _record_matches_logical(record, OURO_SNAPSHOT / relative, record["path"]):
            errors.append(f"probe model bytes are unavailable or changed: {record['path']}")
        else:
            returned.append(record)
    try:
        expected_paths = {
            f"model_snapshot/{path.relative_to(OURO_SNAPSHOT).as_posix()}"
            for path in model_snapshot_files(OURO_SNAPSHOT)
        }
    except (OSError, ValueError):
        expected_paths = set()
        errors.append("probe current model snapshot is unavailable")
    if declared_paths != expected_paths:
        errors.append("probe model manifest does not cover the current snapshot")
    return not errors, returned, errors


def _validation_info(payload: object) -> dict[str, Any]:
    """Validate the full current validator schema without trusting booleans."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return {"verified": False, "status": "FAILED_OR_INCOMPLETE_CURRENT_VALIDATION", "errors": ["validation is not an object"]}
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported validation schema")
    if not isinstance(payload.get("numerical_pass"), bool):
        errors.append("numerical_pass is missing or not boolean")
    elif payload["numerical_pass"] is not True:
        errors.append("numerical_pass is false")
    if payload.get("pass") is not True:
        errors.append("top-level pass is not true")
    if not isinstance(payload.get("bit_exact"), bool):
        errors.append("bit_exact is missing or not boolean")
    missing = [name for name in VALIDATION_MILESTONES if name not in payload]
    if missing:
        errors.append(f"missing validation milestones: {missing}")
    for name in VALIDATION_MILESTONES:
        milestone = payload.get(name)
        if not isinstance(milestone, dict):
            errors.append(f"{name} is missing or not an object")
        elif milestone.get("pass") is not True:
            errors.append(f"{name}.pass is not true")

    # Recompute the complete rollup from primitive measurements.  Every
    # stored gate is checked for exact agreement, including intentionally
    # non-numerical fields such as status and the bit-exact comparison list.
    try:
        expected_rollup = derive_validation_rollup(payload)
    except (KeyError, TypeError, ValueError, IndexError):
        expected_rollup = None
        errors.append("current validator rollup is unavailable")
    if expected_rollup is not None:
        expected_milestones = expected_rollup["milestone_passes"]
        for name, expected in expected_milestones.items():
            milestone = payload.get(name)
            if not isinstance(milestone, dict) or type(milestone.get("pass")) is not bool:
                errors.append(f"{name}.pass is missing or not boolean")
            elif milestone["pass"] != expected:
                errors.append(f"{name}.pass disagrees with primitive measurements")
        for field in ("numerical_pass", "pass", "status", "bit_exact"):
            if payload.get(field) != expected_rollup[field]:
                errors.append(f"validation {field} disagrees with primitive rollup")

    required = payload.get("required_bit_exact_comparisons")
    comparisons = payload.get("bit_exact_comparisons")
    if not isinstance(required, list) or not required:
        errors.append("required_bit_exact_comparisons is missing or empty")
    if not isinstance(comparisons, list) or not comparisons:
        errors.append("bit_exact_comparisons is missing or empty")
    if isinstance(required, list) and isinstance(comparisons, list):
        required_paths = [item for item in required if isinstance(item, str)]
        comparison_paths = [item.get("path") for item in comparisons if isinstance(item, dict)]
        if len(required_paths) != len(required) or len(comparison_paths) != len(comparisons):
            errors.append("comparison path entries are malformed")
        if len(set(required_paths)) != len(required_paths):
            errors.append("required comparison paths are duplicated")
        if len(set(comparison_paths)) != len(comparison_paths):
            errors.append("comparison paths are duplicated")
        if set(required_paths) != set(comparison_paths):
            errors.append("required and recorded comparison sets differ")
        for comparison in comparisons:
            if not isinstance(comparison, dict) or not isinstance(comparison.get("equal"), bool) or not isinstance(comparison.get("close"), bool):
                errors.append("comparison lacks boolean equal/close fields")
        if expected_rollup is not None:
            expected_paths = expected_rollup["required_bit_exact_comparisons"]
            if required_paths != expected_paths:
                errors.append("comparison paths do not match the current validator rollup")
            if comparisons != expected_rollup["bit_exact_comparisons"]:
                errors.append("bit_exact_comparisons disagree with primitive measurements")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("validation provenance is missing")
        source_records: list[dict[str, Any]] = []
    else:
        missing_fields = VALIDATION_PROVENANCE_FIELDS - set(provenance)
        if missing_fields:
            errors.append(f"validation provenance missing fields: {sorted(missing_fields)}")
        source_ok, source_errors, source_records = _source_records_valid(
            provenance, expected_names={"validate.py", "recurrent.py", "evidence.py"}
        )
        if not source_ok:
            errors.extend(source_errors)
        if not isinstance(provenance.get("jlens_commit"), str) or not provenance.get("jlens_commit"):
            errors.append("validation jlens_commit is missing")
        package_versions = provenance.get("package_versions")
        if (
            not isinstance(package_versions, dict)
            or not package_versions
            or set(package_versions) != VALIDATION_PACKAGE_VERSIONS
            or not all(isinstance(name, str) and name and isinstance(version, str) and version for name, version in package_versions.items())
            or any(version == "NOT_INSTALLED" for version in package_versions.values())
        ):
            errors.append("validation package_versions are missing or malformed")
        if provenance.get("model_revision") != OURO_REVISION:
            errors.append("validation model revision is not the current Ouro revision")
        if provenance.get("status") != "HASH_BOUND":
            errors.append("validation provenance is not HASH_BOUND")
        image_digest = provenance.get("image_digest")
        expected_image_digest = os.environ.get("JLENS_IMAGE_DIGEST")
        if image_digest is None and expected_image_digest is None:
            pass
        elif not _valid_image_digest(image_digest):
            errors.append("validation image_digest is missing or malformed")
        elif not _valid_image_digest(expected_image_digest):
            errors.append("JLENS_IMAGE_DIGEST is missing or malformed in the verifier environment")
        elif image_digest != expected_image_digest:
            errors.append("validation image_digest does not match JLENS_IMAGE_DIGEST")
        for field in ("created_at", "platform", "python", "torch"):
            if not isinstance(provenance.get(field), str) or not provenance.get(field):
                errors.append(f"validation provenance {field} is missing")
        for field in ("cuda", "backend"):
            if not isinstance(provenance.get(field), dict):
                errors.append(f"validation provenance {field} is missing")
        cuda = provenance.get("cuda")
        if isinstance(cuda, dict):
            available = cuda.get("available")
            count = cuda.get("device_count")
            devices = cuda.get("devices")
            if type(available) is not bool or type(count) is not int or count < 0 or not isinstance(devices, list):
                errors.append("validation CUDA device record is malformed")
            elif (available is False and (count != 0 or devices)) or (available is True and count != len(devices)):
                errors.append("validation CUDA availability and device records disagree")
            elif len({device.get("index") for device in devices if isinstance(device, dict)}) != len(devices):
                errors.append("validation CUDA device records are duplicated or malformed")
            else:
                for device in devices:
                    if (
                        not isinstance(device, dict)
                        or type(device.get("index")) is not int
                        or device["index"] < 0
                        or not isinstance(device.get("name"), str)
                        or not device["name"]
                        or not isinstance(device.get("capability"), list)
                        or len(device["capability"]) != 2
                        or any(type(value) is not int or value < 0 for value in device["capability"])
                        or type(device.get("total_memory")) is not int
                        or device["total_memory"] <= 0
                    ):
                        errors.append("validation CUDA device record is malformed")
                        break
        model_bytes = provenance.get("model_bytes")
        model_ok, model_errors, _ = _logical_records_valid(
            model_bytes,
            root=OURO_SNAPSHOT,
            prefix="model_snapshot",
            label="validation model_bytes",
        )
        if not isinstance(model_bytes, dict) or model_bytes.get("status") != "HASH_BOUND":
            errors.append("validation model_bytes are not HASH_BOUND")
        if not model_ok:
            errors.extend(model_errors)
        elif not any(Path(record["path"]).name.endswith(".safetensors") for record in model_bytes.get("files", [])):
            errors.append("validation model_bytes omit safetensors weights")
        if isinstance(model_bytes, dict) and isinstance(model_bytes.get("files"), list):
            try:
                expected_model_paths = {
                    f"model_snapshot/{path.relative_to(OURO_SNAPSHOT).as_posix()}"
                    for path in model_snapshot_files(OURO_SNAPSHOT)
                }
            except (OSError, ValueError):
                expected_model_paths = set()
                errors.append("current model snapshot file set is unavailable")
            declared_model_paths = {
                record.get("path") for record in model_bytes["files"] if isinstance(record, dict)
            }
            if declared_model_paths != expected_model_paths:
                errors.append("validation model_bytes do not cover the current snapshot")
        try:
            import jlens

            jlens_root = Path(jlens.__file__).resolve().parent
        except (ImportError, OSError):
            jlens_root = None
        jlens_ok, jlens_errors, _ = _logical_records_valid(
            provenance.get("jlens_source"),
            root=jlens_root if jlens_root is not None else PROJECT_ROOT,
            prefix="jlens",
            label="validation jlens_source",
        )
        if not jlens_ok:
            errors.extend(jlens_errors)
        jlens_records = provenance.get("jlens_source", {}).get("files") if isinstance(provenance.get("jlens_source"), dict) else None
        if jlens_root is not None and isinstance(jlens_records, list):
            expected_jlens_paths = {
                f"jlens/{path.relative_to(jlens_root).as_posix()}"
                for path in jlens_root.rglob("*.py")
                if path.is_file()
            }
            declared_jlens_paths = {
                record.get("path") for record in jlens_records if isinstance(record, dict)
            }
            if declared_jlens_paths != expected_jlens_paths:
                errors.append("validation jlens_source does not cover the installed package")
        shape = provenance.get("model_shape")
        if not isinstance(shape, dict) or any(
            not isinstance(shape.get(key), int) or isinstance(shape.get(key), bool) or shape[key] <= 0
            for key in ("d_model", "n_layers", "n_physical", "n_ut")
        ):
            errors.append("validation model_shape is incomplete")
    model = payload.get("model")
    if not isinstance(model, dict) or any(
        not isinstance(model.get(key), int) or isinstance(model.get(key), bool) or model[key] <= 0
        for key in ("d_model", "n_layers", "n_physical", "n_ut")
    ):
        errors.append("validation model shape is incomplete")
    if not isinstance(payload.get("prompt_tokens"), int) or isinstance(payload.get("prompt_tokens"), bool) or payload["prompt_tokens"] <= 0:
        errors.append("validation prompt_tokens is missing")
    if not isinstance(payload.get("report_file"), str) or not payload["report_file"]:
        errors.append("validation report_file is missing")
    if isinstance(model, dict) and isinstance(provenance, dict):
        provenance_shape = provenance.get("model_shape")
        if isinstance(provenance_shape, dict) and {
            key: provenance_shape.get(key)
            for key in ("d_model", "n_layers", "n_physical", "n_ut")
        } != {
            key: model.get(key)
            for key in ("d_model", "n_layers", "n_physical", "n_ut")
        }:
            errors.append("validation model and provenance shapes differ")
    verified = not errors
    return {
        "verified": verified,
        "status": "SUPPORTED_CURRENT_VALIDATION" if verified else "FAILED_OR_INCOMPLETE_CURRENT_VALIDATION",
        "errors": errors,
        "source_files": source_records,
        "comparison_count": len(comparisons) if isinstance(comparisons, list) else 0,
        "bit_exact": payload.get("bit_exact") is True,
        "paid_accepted": bool(
            verified
            and paid_validation_accepted(payload)
        ),
    }


def _evaluation_fit_prompts(
    provenance: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
    artifact_root: Path | None = None,
) -> int | None:
    """Extract fit size only from an authenticated evaluation provenance chain."""

    for key in ("fit_prompts", "n_prompts"):
        value = provenance.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    lens_inputs = provenance.get("lens_inputs")
    if not isinstance(lens_inputs, list):
        return None
    identity_counts: list[int] = []
    for entry in lens_inputs:
        if not isinstance(entry, dict):
            continue
        identity = entry.get("identity")
        if isinstance(identity, dict):
            for key in ("fit_prompts", "n_prompts"):
                value = identity.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    if key == "n_prompts":
                        identity_counts.append(value)
                    else:
                        return value
        sidecar = entry.get("sidecar")
        sidecar_path = _artifact_lens_record_matches(
            sidecar,
            root=root,
            artifact_root=artifact_root,
        )
        if not _record_shape(sidecar) or sidecar_path is None:
            continue
        try:
            sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(sidecar_payload, dict):
            value = sidecar_payload.get("n_prompts")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                identity_counts.append(value)
    if identity_counts and len(set(identity_counts)) == 1:
        return identity_counts[0]
    return None


def _evaluation_info(
    eval_dir: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Check the complete current ``evaluate.py`` producer record.

    The evaluator's schema deliberately stores logical paths for output and
    model/lens records.  Every logical record is resolved against its known
    producer root and checked against current bytes before it can supply the
    fit size or promote the main table.
    """

    eval_dir = Path(eval_dir)
    path = eval_dir / "provenance.json"
    if eval_dir.is_symlink() or path.is_symlink():
        return {
            "verified": False,
            "status": "INVALID_EVALUATION_PROVENANCE",
            "errors": ["evaluation directory or provenance is linked"],
            "fit_prompts": None,
        }
    if not path.is_file():
        return {
            "verified": False,
            "status": LINEAGE_UNFROZEN,
            "errors": ["evaluation provenance is missing"],
            "fit_prompts": None,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"verified": False, "status": "INVALID_EVALUATION_PROVENANCE", "errors": [str(exc)], "fit_prompts": None}
    errors: list[str] = []
    if not isinstance(payload, dict):
        errors.append("evaluation provenance is not an object")
        return {"verified": False, "status": "INVALID_EVALUATION_PROVENANCE", "errors": errors, "fit_prompts": None}
    try:
        from ouro_jlens.evaluate import _check_worker_deadline

        _check_worker_deadline(payload)
    except Exception as exc:
        errors.append(f"evaluation worker deadline binding is invalid: {exc}")
    missing = EVALUATION_PROVENANCE_FIELDS - set(payload)
    if missing:
        errors.append(f"evaluation provenance missing fields: {sorted(missing)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported evaluation provenance schema")
    if not isinstance(payload.get("created_at"), str) or not payload.get("created_at"):
        errors.append("evaluation provenance created_at is missing")
    if not isinstance(payload.get("jlens_commit"), str) or not payload.get("jlens_commit"):
        errors.append("evaluation provenance jlens_commit is missing")
    else:
        try:
            from ouro_jlens.fit_lens import jlens_commit

            current_jlens_commit = jlens_commit()
        except (ImportError, OSError, ValueError):
            current_jlens_commit = "UNKNOWN"
        if current_jlens_commit not in {"UNKNOWN", payload.get("jlens_commit")}:
            errors.append("evaluation provenance jlens_commit is stale")

    validation = payload.get("validation")
    if not isinstance(validation, dict) or validation.get("kind") not in {"project", "test_bundle"}:
        errors.append("evaluation validation root identity is missing")
    lens_validation_root = (
        PROJECT_ROOT
        if isinstance(validation, dict) and validation.get("kind") == "project"
        else eval_dir.parent
    )

    config = payload.get("config")
    if not isinstance(config, dict):
        errors.append("evaluation config is missing")
    else:
        tasks = config.get("tasks")
        if (
            not isinstance(tasks, list)
            or not tasks
            or not all(isinstance(task, str) and task for task in tasks)
            or len(set(tasks)) != len(tasks)
        ):
            errors.append("evaluation config tasks are missing")
        if config.get("prompt_policy") not in {"identical", "nested"}:
            errors.append("evaluation prompt policy is missing or invalid")
        for key in ("position", "max_intermediates", "max_names", "greedy_steps"):
            value = config.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 and key != "position":
                errors.append(f"evaluation config {key} is missing or invalid")

    model = payload.get("model")
    model_files: list[dict[str, Any]] = []
    if not isinstance(model, dict):
        errors.append("evaluation model identity is missing")
    else:
        if model.get("revision") != OURO_REVISION:
            errors.append("evaluation model identity is stale")
        if model.get("bytes_status") != "HASH_BOUND":
            errors.append("evaluation model bytes are not hash-bound")
        for key in ("n_physical", "n_ut", "n_layers", "d_model"):
            if not isinstance(model.get(key), int) or isinstance(model.get(key), bool) or model[key] <= 0:
                errors.append(f"evaluation model {key} is missing or invalid")
        if not isinstance(model.get("files"), list) or not model["files"]:
            errors.append("evaluation model file manifest is missing")
        else:
            model_files = [record for record in model["files"] if isinstance(record, dict)]
            if len(model_files) != len(model["files"]):
                errors.append("evaluation model file manifest is malformed")
            try:
                expected_model_relatives = {
                    path.relative_to(OURO_SNAPSHOT).as_posix()
                    for path in model_snapshot_files(OURO_SNAPSHOT)
                }
            except (OSError, ValueError):
                expected_model_relatives = set()
                errors.append("current model snapshot file set is unavailable")
            matched_model_relatives: set[str] = set()
            for record in model_files:
                if not _record_matches(record):
                    errors.append(f"evaluation model bytes are unavailable or changed: {record.get('path')}")
                matches = [
                    relative for relative in expected_model_relatives
                    if _record_points_to(record, OURO_SNAPSHOT / relative)
                ]
                if len(matches) != 1:
                    errors.append(f"evaluation model path is not the current snapshot: {record.get('path')}")
                else:
                    matched_model_relatives.add(matches[0])
            if matched_model_relatives != expected_model_relatives:
                errors.append("evaluation model file set does not cover the current snapshot")
            model_aggregate = aggregate_sha256({record["path"]: record["sha256"] for record in model_files if _record_shape(record)}) if all(_record_shape(record) for record in model_files) else None
            if model.get("aggregate_sha256") != model_aggregate:
                errors.append("evaluation model aggregate digest does not match model files")
            if payload.get("model_snapshot_sha256") != model_aggregate:
                errors.append("evaluation model snapshot digest does not match model files")

    outputs = payload.get("outputs")
    expected_outputs = {
        "arrays": eval_dir / "arrays.npz",
        "items": eval_dir / "items.json",
        "task_names": eval_dir / "task_names.json",
    }
    if not isinstance(outputs, dict):
        errors.append("evaluation outputs are missing")
    else:
        for name, expected in expected_outputs.items():
            if not _record_matches(outputs.get(name), expected):
                errors.append(f"evaluation output {name} is missing or changed")
    source_ok, source_errors, source_records = _source_records_valid(
        payload, expected_names={"evaluate.py", "evaldata.py", "evidence.py", "fit_lens.py", "recurrent.py"}
    )
    if not source_ok:
        errors.extend(source_errors)
    inputs = payload.get("inputs")
    evaluation_files = inputs.get("evaluation_files") if isinstance(inputs, dict) else None
    if not isinstance(inputs, dict) or not isinstance(evaluation_files, list) or not evaluation_files:
        errors.append("evaluation input records are missing")
    else:
        if inputs.get("evaluation_input_sha256") != sha256_json(evaluation_files):
            errors.append("evaluation input aggregate does not match evaluation_files")
        for record in evaluation_files:
            if not _record_shape(record) or not _record_matches(record):
                errors.append(f"evaluation input bytes are unavailable or changed: {record.get('path') if isinstance(record, dict) else record}")
        if isinstance(config, dict) and isinstance(config.get("tasks"), list) and len(evaluation_files) == len(config["tasks"]):
            try:
                from ouro_jlens.evaldata import JLENS_DATA

                for task, record in zip(config["tasks"], evaluation_files, strict=True):
                    expected = JLENS_DATA / f"lens-eval-{task}.json"
                    if not _record_points_to(record, expected):
                        errors.append(f"evaluation input path is not the declared task source: {task}")
            except (ImportError, OSError, TypeError, ValueError):
                errors.append("evaluation task source files are unavailable")
    lens_inputs = payload.get("lens_inputs")
    if not isinstance(lens_inputs, list) or not lens_inputs:
        errors.append("lens input records are missing")
    else:
        seen_targets: set[int] = set()
        for entry in lens_inputs:
            if not isinstance(entry, dict) or not isinstance(entry.get("target_ut"), int) or isinstance(entry.get("target_ut"), bool) or not isinstance(entry.get("identity"), dict):
                errors.append("lens input identity is incomplete")
                continue
            target_ut = entry["target_ut"]
            if target_ut in seen_targets:
                errors.append("lens input targets are duplicated")
            seen_targets.add(target_ut)
            binary, sidecar = entry.get("binary"), entry.get("sidecar")
            binary_path = _artifact_lens_record_matches(
                binary,
                root=lens_validation_root,
                artifact_root=artifact_root,
            )
            sidecar_path_value = _artifact_lens_record_matches(
                sidecar,
                root=lens_validation_root,
                artifact_root=artifact_root,
            )
            if binary_path is None or sidecar_path_value is None:
                errors.append("lens input binary/sidecar is missing or changed")
            elif sidecar_path_value != binary_path.with_suffix(".json"):
                errors.append("lens input sidecar does not match binary")
            identity = entry["identity"]
            required_identity = (
                "target_ut", "target_virtual", "source_layers", "model_revision",
                "jlens_commit", "prompt_file_sha256", "prompt_slice_sha256",
                "prompt_slice", "n_requested", "n_fitted", "n_prompts",
                "prompt_provenance_sha256", "prompt_source_revision",
                "source_sha256", "generator_sha256",
            )
            if not set(required_identity) <= set(identity):
                errors.append("lens input identity fields are incomplete")
            for key in (
                "prompt_file_sha256", "prompt_slice_sha256", "prompt_provenance_sha256",
                "source_sha256", "generator_sha256",
            ):
                if not _is_sha256(identity.get(key)):
                    errors.append(f"lens input {key} is missing or malformed")
            prompt_slice = identity.get("prompt_slice")
            if (
                not isinstance(prompt_slice, dict)
                or any(
                    not isinstance(prompt_slice.get(key), int)
                    or isinstance(prompt_slice.get(key), bool)
                    for key in ("start", "end", "count")
                )
                or prompt_slice.get("start", 0) < 0
                or prompt_slice.get("end", 0) <= prompt_slice.get("start", 0)
                or prompt_slice.get("count") != prompt_slice.get("end", 0) - prompt_slice.get("start", 0)
            ):
                errors.append("lens input prompt slice identity is malformed")
            for key in ("n_requested", "n_fitted", "n_prompts"):
                value = identity.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    errors.append(f"lens input {key} is missing or malformed")
            if (
                isinstance(prompt_slice, dict)
                and isinstance(prompt_slice.get("count"), int)
                and all(isinstance(identity.get(key), int) and not isinstance(identity.get(key), bool)
                        for key in ("n_requested", "n_fitted", "n_prompts"))
                and any(identity[key] != prompt_slice["count"] for key in ("n_requested", "n_fitted", "n_prompts"))
            ):
                errors.append("lens input counts disagree with prompt slice")
            sidecar_payload = None
            if binary_path is not None:
                try:
                    sidecar_payload = json.loads(sidecar_path_value.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    errors.append("lens input sidecar payload is unavailable")
            if isinstance(sidecar_payload, dict):
                for key in required_identity:
                    if sidecar_payload.get(key) != identity.get(key):
                        errors.append(f"lens input identity disagrees with sidecar: {key}")
                # The evaluation record is not allowed to replace a complete
                # fit sidecar with a copied binary plus a few matching fields.
                # Re-run the producer's sidecar gate so prompt/model/source
                # provenance and the generator identity are checked too.
                if binary_path is not None:
                    try:
                        from ouro_jlens.evaluate import validate_lens_sidecar

                        validate_lens_sidecar(
                            binary_path,
                            kind=sidecar_payload.get("kind"),
                            validation_root=lens_validation_root,
                            artifact_root=artifact_root,
                        )
                    except (ImportError, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
                        errors.append(f"lens input sidecar fails its producer validation: {exc}")
            expected_target_virtual = (
                target_ut * model.get("n_physical", 0) + model.get("n_physical", 0) - 1
                if isinstance(model, dict) and isinstance(model.get("n_physical"), int)
                else None
            )
            if (
                identity.get("target_ut") != target_ut
                or identity.get("target_virtual") != expected_target_virtual
                or identity.get("model_revision") != OURO_REVISION
                or identity.get("jlens_commit") != payload.get("jlens_commit")
                or not isinstance(model, dict)
                or not isinstance(model.get("n_ut"), int)
                or isinstance(model.get("n_ut"), bool)
                or not 0 <= target_ut < model.get("n_ut", 0)
            ):
                errors.append("lens input target/model identity is stale")
            source_layers = identity.get("source_layers")
            target_virtual = identity.get("target_virtual")
            if not isinstance(target_virtual, int) or isinstance(target_virtual, bool) or not isinstance(source_layers, list) or source_layers != list(range(target_virtual)):
                errors.append("lens input source layer identity is invalid")
        if isinstance(model, dict) and isinstance(model.get("n_ut"), int) and model["n_ut"] - 1 not in seen_targets:
            errors.append("lens inputs omit the eventual-exit lens")
    if isinstance(config, dict) and isinstance(lens_inputs, list) and lens_inputs:
        identities = [entry.get("identity", {}) for entry in lens_inputs if isinstance(entry, dict)]
        prompt_files = {identity.get("prompt_file_sha256") for identity in identities}
        if len(prompt_files) != 1:
            errors.append("lens inputs use different prompt files")
        fit_values = [identity.get("n_prompts") for identity in identities]
        if (
            len(fit_values) != len(lens_inputs)
            or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in fit_values)
        ):
            errors.append("lens inputs use different or invalid fit counts")
        elif config.get("prompt_policy") == "identical" and len(set(fit_values)) != 1:
            errors.append("identical prompt policy has different fit counts")
        if config.get("prompt_policy") == "identical":
            prompt_slices = {identity.get("prompt_slice_sha256") for identity in identities}
            if len(prompt_slices) != 1:
                errors.append("identical prompt policy has different prompt slices")
    if not isinstance(payload.get("item_count"), int) or isinstance(payload.get("item_count"), bool) or payload["item_count"] <= 0:
        errors.append("evaluation item_count is missing")
    elif (eval_dir / "items.json").is_file():
        try:
            items = json.loads((eval_dir / "items.json").read_text(encoding="utf-8"))
            if not isinstance(items, list) or payload["item_count"] != len(items):
                errors.append("evaluation item_count does not match items.json")
            if isinstance(items, list) and any(not isinstance(item, dict) or not isinstance(item.get("correct"), bool) for item in items):
                errors.append("evaluation item correctness records are incomplete")
            if isinstance(items, list) and payload.get("correct_count") != sum(item.get("correct") is True for item in items if isinstance(item, dict)):
                errors.append("evaluation correct_count does not match items.json")
            if isinstance(items, list) and payload.get("item_metadata_sha256") != sha256_json(items):
                errors.append("evaluation item_metadata_sha256 does not match items.json")
        except (OSError, json.JSONDecodeError, TypeError):
            errors.append("evaluation items.json cannot be read")
    if artifact_root is not None:
        try:
            from ouro_jlens.evaluate import validate_evaluation_provenance

            validate_evaluation_provenance(
                eval_dir,
                artifact_root=artifact_root,
            )
        except Exception as exc:
            errors.append(f"evaluation relocation validation failed: {exc}")
    fit_prompts = _evaluation_fit_prompts(
        payload,
        root=lens_validation_root,
        artifact_root=artifact_root,
    )
    return {
        "verified": not errors,
        "status": CURRENT_GENERATION if not errors else "INVALID_EVALUATION_PROVENANCE",
        "errors": errors,
        "fit_prompts": fit_prompts if not errors else None,
        "source_files": source_records,
        "record": _canonical_artifact_record(path, artifact_root),
        "payload": payload,
    }


def _checkpoint_rows(payload: object) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("checkpoints")
    if isinstance(rows, dict):
        return {str(name): row for name, row in rows.items() if isinstance(row, dict)}
    return {
        str(name): row for name, row in payload.items()
        if name in {"base", "thinking", "rltt"} and isinstance(row, dict)
    }


def _trend(values: list[list[float]]) -> str:
    if len(values) < 2:
        return "not_measured"
    flat = np.asarray(values, dtype=float)
    if not np.isfinite(flat).all():
        return "invalid"
    differences = np.diff(flat, axis=0)
    if np.all(differences <= 0):
        return "decreases"
    if np.all(differences >= 0):
        return "increases"
    return "mixed"


def _runtime_versions_valid(value: object, expected: frozenset[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == expected
        and all(
            isinstance(name, str)
            and isinstance(version, str)
            and version
            and version != "NOT_INSTALLED"
            for name, version in value.items()
        )
    )


def _checkpoint_source_records_valid(
    payload: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    """Validate the mixed project/dependency source manifest emitted by checkpoints."""

    records, errors = _declared_records(payload.get("source_files"))
    if not records:
        errors.append("checkpoint source manifest is empty")
        return False, records, errors
    expected_project = {
        "src/ouro_jlens/checkpoints.py",
        "src/ouro_jlens/evaluate.py",
        "src/ouro_jlens/evaldata.py",
        "src/ouro_jlens/recurrent.py",
        "src/ouro_jlens/evidence.py",
    }
    project_records = [record for record in records if record["path"] in expected_project]
    dependency_records = [
        record for record in records
        if isinstance(record.get("path"), str)
        and record["path"].startswith("dependency/jlens/")
    ]
    if {record["path"] for record in project_records} != expected_project:
        errors.append("checkpoint project source set is incomplete")
    if len(project_records) + len(dependency_records) != len(records):
        errors.append("checkpoint source manifest contains unknown records")
    for record in project_records:
        if not _record_matches(record, PROJECT_ROOT / Path(record["path"])):
            errors.append(f"checkpoint project source is missing or changed: {record['path']}")
    try:
        import jlens

        jlens_root = Path(jlens.__file__).resolve().parent
        expected_dependency = {
            f"dependency/jlens/{path.relative_to(jlens_root).as_posix()}"
            for path in jlens_root.rglob("*.py")
            if path.is_file()
        }
    except (ImportError, OSError, ValueError):
        jlens_root = None
        expected_dependency = set()
        errors.append("checkpoint installed jlens source is unavailable")
    if {record["path"] for record in dependency_records} != expected_dependency:
        errors.append("checkpoint dependency source set is incomplete")
    if (
        len({record["path"] for record in records}) != len(records)
        or len(records) != len(expected_project) + len(expected_dependency)
    ):
        errors.append("checkpoint source manifest contains duplicate or extra records")
    if jlens_root is not None:
        for record in dependency_records:
            relative = Path(record["path"][len("dependency/jlens/"):])
            if not _record_matches_logical(record, jlens_root / relative, record["path"]):
                errors.append(f"checkpoint dependency source is missing or changed: {record['path']}")
    aggregate = aggregate_sha256({record["path"]: record["sha256"] for record in records})
    if payload.get("source_sha256") != aggregate:
        errors.append("checkpoint source aggregate does not match source records")
    return not errors, records, errors


def _checkpoint_info(payload: object) -> dict[str, Any]:
    rows = _checkpoint_rows(payload)
    errors: list[str] = []
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        errors.append("unsupported checkpoint schema")
    if not isinstance(payload, dict) or payload.get("status") != "REQUESTED_CHECKPOINT_MEASUREMENTS_COMPLETE_UNFROZEN":
        errors.append("checkpoint run is not marked COMPLETE")
    if isinstance(payload, dict) and payload.get("missing") != []:
        errors.append("checkpoint run has missing checkpoints")
    requested = payload.get("requested_checkpoints") if isinstance(payload, dict) else None
    if (
        not isinstance(requested, list)
        or not all(isinstance(item, str) for item in requested)
        or set(requested) != {"base", "thinking", "rltt"}
    ):
        errors.append("checkpoint requested group set is incomplete")
    if isinstance(payload, dict) and (
        not isinstance(payload.get("interpretation_limit"), str)
        or "heterogeneous named checkpoints" not in payload["interpretation_limit"]
        or "not an ordered or monotonic training trajectory" not in payload["interpretation_limit"]
    ):
        errors.append("checkpoint interpretation limit is missing")
    if not isinstance(payload, dict) or not isinstance(payload.get("checkpoints"), dict):
        errors.append("checkpoint rows are missing from the current schema")
    if set(rows) != {"base", "thinking", "rltt"}:
        errors.append("checkpoint group must contain exactly base, thinking, and rltt")
    source_ok, source_records, source_errors = _checkpoint_source_records_valid(
        payload if isinstance(payload, dict) else {},
    )
    if not source_ok:
        errors.extend(source_errors)
    if isinstance(payload, dict):
        if not _runtime_versions_valid(payload.get("runtime_versions"), CHECKPOINT_RUNTIME_VERSIONS):
            errors.append("checkpoint runtime provenance is missing or incomplete")
        try:
            import jlens

            jlens_root = Path(jlens.__file__).resolve().parent
        except (ImportError, OSError):
            jlens_root = None
        jlens_manifest = payload.get("jlens_source")
        jlens_ok, jlens_errors, jlens_records = _logical_records_valid(
            jlens_manifest,
            root=jlens_root if jlens_root is not None else PROJECT_ROOT,
            prefix="dependency/jlens",
            label="checkpoint jlens source",
        )
        if not jlens_ok:
            errors.extend(jlens_errors)
        if jlens_root is not None and isinstance(jlens_manifest, dict):
            expected_jlens_paths = {
                f"dependency/jlens/{path.relative_to(jlens_root).as_posix()}"
                for path in jlens_root.rglob("*.py")
                if path.is_file()
            }
            declared_jlens_paths = {
                record.get("path")
                for record in jlens_manifest.get("files", [])
                if isinstance(record, dict)
            }
            if declared_jlens_paths != expected_jlens_paths:
                errors.append("checkpoint jlens source does not cover the installed package")
            if {record.get("path") for record in jlens_records} != {
                record.get("path")
                for record in source_records
                if record.get("path", "").startswith("dependency/jlens/")
            }:
                errors.append("checkpoint source and jlens manifests differ")
    tokenizer_manifest = payload.get("input_tokenizer") if isinstance(payload, dict) else None
    tokenizer_ok, tokenizer_errors, tokenizer_records = _logical_records_valid(
        tokenizer_manifest,
        root=OURO_SNAPSHOT,
        prefix="input_tokenizer",
        label="checkpoint tokenizer input",
    )
    if not tokenizer_ok:
        errors.extend(tokenizer_errors)
    else:
        expected_tokenizer_paths = {
            f"input_tokenizer/{path.relative_to(OURO_SNAPSHOT).as_posix()}"
            for path in OURO_SNAPSHOT.rglob("*")
            if path.is_file()
        }
        if {record["path"] for record in tokenizer_records} != expected_tokenizer_paths:
            errors.append("checkpoint tokenizer input manifest does not cover the base snapshot")
    checkpoint_roots: dict[str, Path | None] = {
        "base": OURO_SNAPSHOT,
        "thinking": next(iter(sorted((PROJECT_ROOT / "artifacts/hf_cache/hub/models--ByteDance--Ouro-2.6B-Thinking/snapshots").glob("*"))), None),
        "rltt": PROJECT_ROOT / "models/ouro_rltt_local",
    }
    metric_values: dict[str, dict[str, list[list[float]]]] = {"kl": {}, "js": {}, "agreement": {}}
    for task in ("multihop", "order-ops"):
        for checkpoint in ("base", "thinking", "rltt"):
            row = rows.get(checkpoint, {})
            task_row = row.get(task) if isinstance(row, dict) else None
            if not isinstance(task_row, dict):
                errors.append(f"missing checkpoint row {checkpoint}.{task}")
                continue
            for key, target in (
                ("kl_exit_k_to_exit_last_mean", "kl"),
                ("js_to_final_mean", "js"),
                ("exit_top1_equals_final", "agreement"),
            ):
                values = task_row.get(key)
                if not isinstance(values, list) or len(values) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in values):
                    errors.append(f"invalid checkpoint metric {checkpoint}.{task}.{key}")
                else:
                    metric_values[target].setdefault(task, []).append([float(v) for v in values])
            if not isinstance(row.get("checkpoint_id"), str) or row.get("checkpoint_id") != checkpoint:
                errors.append(f"checkpoint id is missing or changed: {checkpoint}")
            if not isinstance(row.get("model_revision"), str) or not row["model_revision"]:
                errors.append(f"checkpoint model revision is missing: {checkpoint}")
            checkpoint_files = row.get("checkpoint_files")
            root = checkpoint_roots.get(checkpoint)
            if root is None or not isinstance(checkpoint_files, list) or not checkpoint_files:
                errors.append(f"checkpoint file manifest is missing: {checkpoint}")
            else:
                declared_checkpoint_paths: set[str] = set()
                for record in checkpoint_files:
                    if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"].startswith("checkpoint/"):
                        errors.append(f"checkpoint file record is malformed: {checkpoint}")
                        continue
                    relative = Path(record["path"][len("checkpoint/"):])
                    if record["path"] in declared_checkpoint_paths:
                        errors.append(f"checkpoint file manifest contains duplicate paths: {checkpoint}")
                    declared_checkpoint_paths.add(record["path"])
                    if relative.is_absolute() or ".." in relative.parts or not _record_matches_logical(record, root / relative, record["path"]):
                        errors.append(f"checkpoint model bytes are unavailable or changed: {checkpoint}/{relative}")
                try:
                    expected_checkpoint_paths = {
                        f"checkpoint/{candidate.relative_to(root).as_posix()}"
                        for candidate in root.rglob("*")
                        if candidate.is_file()
                    }
                except (OSError, ValueError):
                    expected_checkpoint_paths = set()
                    errors.append(f"checkpoint file inventory is unavailable: {checkpoint}")
                if declared_checkpoint_paths != expected_checkpoint_paths:
                    errors.append(f"checkpoint file manifest does not cover the complete checkpoint: {checkpoint}")
                names = {Path(record.get("path", "")).name for record in checkpoint_files if isinstance(record, dict)}
                if not {"config.json", "modeling_ouro.py", "tokenizer.json"} <= names or not any(name.endswith(".safetensors") for name in names):
                    errors.append(f"checkpoint file manifest omits required model bytes: {checkpoint}")
                expected_aggregate = aggregate_sha256({
                    record["path"]: record["sha256"]
                    for record in checkpoint_files
                    if _record_shape(record)
                }) if all(_record_shape(record) for record in checkpoint_files) else None
                if row.get("model_files_sha256") != expected_aggregate:
                    errors.append(f"checkpoint model file aggregate is missing or changed: {checkpoint}")
    trends = {
        metric: {task: _trend(values) for task, values in tasks.items()}
        for metric, tasks in metric_values.items()
    }
    valid = not errors
    return {
        "verified": valid,
        "errors": errors,
        "rows": rows,
        "trends": trends,
        "status": "SUPPORTED_LOCAL_UNFROZEN" if valid else "INCONCLUSIVE",
        "source_files": source_records,
    }


def checkpoint_claim_text(info: Mapping[str, Any]) -> str:
    """Describe the measured checkpoint trends without collapsing metrics."""

    trends = info.get("trends", {})
    kl = trends.get("kl", {}) if isinstance(trends, dict) else {}
    js = trends.get("js", {}) if isinstance(trends, dict) else {}
    agreement = trends.get("agreement", {}) if isinstance(trends, dict) else {}
    distribution = ", ".join(
        f"{task} KL {kl.get(task, 'not_measured')} and JS {js.get(task, 'not_measured')}"
        for task in ("multihop", "order-ops")
    )
    categorical = ", ".join(
        f"{task} top-1 agreement {agreement.get(task, 'not_measured')}"
        for task in ("multihop", "order-ops")
    )
    return (
        "Across the retained base, Thinking, and RLTT rows, distributional trends are "
        f"{distribution}; categorical trends are {categorical}. "
        "The metrics are heterogeneous, so no blanket agreement-improvement claim is made."
    )


def _fit_size_valid(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == SCHEMA_VERSION
        and isinstance(payload.get("fit_sizes"), list)
        and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in payload["fit_sizes"])
        and isinstance(payload.get("inputs"), dict)
        and all(_record_shape(record) for record in payload["inputs"].values())
        and isinstance(payload.get("populations"), dict)
        and set(payload["populations"]) >= {"multihop", "order_ops_numeric"}
    )


def _transport_valid(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("status") == "MODELED_ASSOCIATION_ONLY"
        and isinstance(payload.get("inputs"), list)
        and all(_record_shape(record) for record in payload["inputs"])
        and isinstance(payload.get("per_loop"), list)
        and len(payload["per_loop"]) == 4
    )


def _probe_valid(payload: object) -> dict[str, Any]:
    """Validate the summary and its complete current probe provenance chain."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return {"verified": False, "errors": ["probe summary is not an object"], "status": "INCONCLUSIVE"}
    accepted_statuses = {
        "LOCAL_CURRENT_SOURCE_HASH_BOUND_UNFROZEN",
        "LOCAL_CURRENT_SOURCE_MIXED_PRE_CUSTODY_LENS_UNFROZEN",
    }
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported probe summary schema")
    if payload.get("status") not in accepted_statuses:
        errors.append("probe summary is derived, degraded, or not current-source validated")
    if payload.get("input_provenance_status") != "VALIDATED_CURRENT_SOURCE_MODEL_AND_SCORE_BYTES":
        errors.append("probe input provenance has not been validated")
    if payload.get("lens_score_status") not in {
        "FRESH_CURRENT_SOURCE_HASH_BOUND_LENS",
        "FRESH_CURRENT_SOURCE_RETAINED_PRE_CUSTODY_LENS",
    }:
        errors.append("probe lens score status is not current-source validated")
    population = payload.get("population")
    expected_population = {
        "total_prompts": 648,
        "eligible_prompts": 576,
        "excluded_untrainable_prompts": 72,
        "total_unordered_pair_clusters": 45,
        "eligible_unordered_pair_clusters": 39,
    }
    if not isinstance(population, dict) or any(population.get(k) != v for k, v in expected_population.items()):
        errors.append("probe population does not match the frozen 648/576/39 design")
    expected_readouts = {"supervised_probe", "logit_lens", "eventual_exit_jacobian_lens"}
    readouts = payload.get("readouts")
    if not isinstance(readouts, dict) or set(readouts) != expected_readouts:
        errors.append("probe readout set is incomplete")
    expected_contrasts = {
        "supervised_probe_minus_logit_lens",
        "eventual_exit_jacobian_lens_minus_logit_lens",
        "eventual_exit_jacobian_lens_minus_supervised_probe",
    }
    contrasts = payload.get("contrasts")
    if not isinstance(contrasts, dict) or set(contrasts) != expected_contrasts:
        errors.append("probe comparison set is incomplete")
    for name, readout in readouts.items() if isinstance(readouts, dict) else ():
        if not isinstance(readout, dict) or not isinstance(readout.get("per_loop"), list) or len(readout["per_loop"]) != 4 or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) for value in readout["per_loop"]):
            errors.append(f"probe readout {name} is malformed")
        if not isinstance(readout, dict) or not isinstance(readout.get("ci95"), list) or len(readout["ci95"]) != 4:
            errors.append(f"probe readout {name} intervals are malformed")
        selected = readout.get("selected_physical_layer_by_heldout_fold") if isinstance(readout, dict) else None
        if not isinstance(selected, list) or len(selected) != 4 or any(not isinstance(row, list) or len(row) != 5 for row in selected):
            errors.append(f"probe readout {name} layer selection is malformed")
    for name, contrast in contrasts.items() if isinstance(contrasts, dict) else ():
        if not isinstance(contrast, dict) or not isinstance(contrast.get("per_loop"), list) or len(contrast["per_loop"]) != 4 or not isinstance(contrast.get("ci95"), list) or len(contrast["ci95"]) != 4:
            errors.append(f"probe contrast {name} is malformed")
    bootstrap = payload.get("bootstrap")
    if not isinstance(bootstrap, dict) or not isinstance(bootstrap.get("draws"), int) or isinstance(bootstrap.get("draws"), bool) or bootstrap.get("draws", 0) <= 0:
        errors.append("probe bootstrap contract is incomplete")
    if not _record_matches_logical(
        payload.get("input"),
        PROJECT_ROOT / "artifacts/jlens/probe/cv_all648/arrays.npz",
        "probe_arrays",
    ):
        errors.append("probe summary input record is missing or changed")

    design = None
    provenance = None
    cache_provenance = None
    summary_arrays = PROJECT_ROOT / "artifacts/jlens/probe/cv_all648/arrays.npz"
    design_path = PROJECT_ROOT / "artifacts/jlens/probe/cv_all648/design.json"
    provenance_path = PROJECT_ROOT / "artifacts/jlens/probe/cv_all648/lens_all648.provenance.json"
    cache_path = PROJECT_ROOT / "artifacts/jlens/probe/n80_v2/gpu_cache.npz"
    cache_provenance_path = cache_path.with_suffix(".provenance.json")
    try:
        design = json.loads(design_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        cache_provenance = json.loads(cache_provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"probe design/provenance chain is unavailable: {exc}")

    source_records: list[dict[str, Any]] = []
    if not isinstance(cache_provenance, dict):
        errors.append("probe hidden-cache provenance is missing")
    else:
        if cache_provenance.get("schema_version") != 2 or cache_provenance.get("status") != "FRESH_CURRENT_SOURCE_AND_MODEL":
            errors.append("probe hidden-cache provenance schema/status is stale")
        if not _record_matches_logical(cache_provenance.get("output"), cache_path, "gpu_cache"):
            errors.append("probe hidden-cache output record is missing or changed")
        if cache_provenance.get("design") != {"prompts": 648, "virtual_locations": 192, "hidden_width": 2048}:
            errors.append("probe hidden-cache design is malformed")
        model_ok, _, model_errors = _probe_model_records_valid(cache_provenance.get("model_files"))
        if not model_ok:
            errors.extend(model_errors)
        source_ok, source_records, source_errors = _probe_source_records_valid(
            cache_provenance.get("source_files"), cache_provenance.get("source_sha256")
        )
        if not source_ok:
            errors.extend(source_errors)
        if not _probe_runtime_valid(cache_provenance.get("runtime_versions")):
            errors.append("probe hidden-cache runtime provenance is missing")

    if not isinstance(provenance, dict):
        errors.append("probe lens provenance is missing")
    else:
        lens_status = provenance.get("status")
        if provenance.get("schema_version") != 2 or lens_status not in {
            "FRESH_CURRENT_SOURCE_HASH_BOUND_LENS",
            "FRESH_CURRENT_SOURCE_RETAINED_PRE_CUSTODY_LENS",
        }:
            errors.append("probe lens provenance schema/status is stale")
        if payload.get("lens_score_status") != lens_status:
            errors.append("probe summary/lens provenance statuses differ")
        if provenance.get("lens_lineage_status") not in {
            "HASH_BOUND", "RETAINED_PRE_CUSTODY_EXACT_BYTES_ONLY",
        }:
            errors.append("probe lens lineage status is missing")
        if not _record_matches_logical(provenance.get("output"), PROJECT_ROOT / "artifacts/jlens/probe/cv_all648/lens_all648.npz", "lens_scores"):
            errors.append("probe lens output record is missing or changed")
        expected_lens_inputs = {
            "gpu_cache": (cache_path, "gpu_cache"),
            "gpu_cache_provenance": (cache_provenance_path, "gpu_cache_provenance"),
            "lens": (PROJECT_ROOT / "artifacts/jlens/lens/exit3/exit3_n80.pt", "lens"),
            "lens_sidecar": (PROJECT_ROOT / "artifacts/jlens/lens/exit3/exit3_n80.json", "lens_sidecar"),
        }
        lens_inputs = provenance.get("inputs")
        if not isinstance(lens_inputs, dict) or set(lens_inputs) != set(expected_lens_inputs):
            errors.append("probe lens input manifest is incomplete")
        else:
            for name, (expected, logical) in expected_lens_inputs.items():
                if not _record_matches_logical(lens_inputs.get(name), expected, logical):
                    errors.append(f"probe lens input {name} is missing or changed")
        score_source_ok, score_source_records, score_source_errors = _probe_source_records_valid(
            provenance.get("source_files"), provenance.get("source_sha256"),
            names=(
                "probe_cv.py", "probe.py", "evaluate.py", "evaldata.py",
                "recurrent.py", "evidence.py", "fit_lens.py",
            ),
        )
        if not score_source_ok:
            errors.extend(score_source_errors)
        if not _probe_runtime_valid(provenance.get("runtime_versions")):
            errors.append("probe lens runtime provenance is missing")
        if (
            isinstance(cache_provenance, dict)
            and provenance.get("model_files") != cache_provenance.get("model_files")
        ):
            errors.append("probe lens/model manifests differ")
        if (
            isinstance(cache_provenance, dict)
            and _probe_runtime_valid(cache_provenance.get("runtime_versions"))
            and provenance.get("runtime_versions") != cache_provenance.get("runtime_versions")
        ):
            errors.append("probe lens/cache runtime provenance differs")

    if not isinstance(design, dict):
        errors.append("probe design record is missing")
    else:
        if design.get("schema_version") != 2 or design.get("status") != "FRESH_CURRENT_SOURCE_INNER_SELECTED_OUTER_SCORED":
            errors.append("probe design schema/status is stale")
        if not isinstance(design.get("seed"), int) or isinstance(design.get("seed"), bool):
            errors.append("probe design seed is missing")
        design_body = design.get("design")
        if not isinstance(design_body, dict):
            errors.append("probe design body is missing")
            design_body = {}
        if design_body.get("fold_assignment_unit") != "unordered_operand_pair":
            errors.append("probe design fold assignment unit is invalid")
        fold_sizes = design_body.get("fold_sizes")
        if not isinstance(fold_sizes, list) or len(fold_sizes) != 5 or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in fold_sizes) or sum(fold_sizes) != 648:
            errors.append("probe design fold sizes are malformed")
        if not isinstance(design_body.get("validation_pairs_by_fold"), list) or len(design_body["validation_pairs_by_fold"]) != 5:
            errors.append("probe design validation pairs are malformed")
        if not isinstance(design_body.get("selection_ancestry"), str) or "outer fold" not in design_body.get("selection_ancestry", ""):
            errors.append("probe design selection ancestry is missing")
        expected_design_inputs = {
            "gpu_cache": (cache_path, "gpu_cache"),
            "gpu_cache_provenance": (cache_provenance_path, "gpu_cache_provenance"),
            "lens_scores": (PROJECT_ROOT / "artifacts/jlens/probe/cv_all648/lens_all648.npz", "lens_scores"),
            "lens_score_provenance": (provenance_path, "lens_score_provenance"),
        }
        if not isinstance(design.get("inputs"), dict) or set(design["inputs"]) != set(expected_design_inputs):
            errors.append("probe design input records are incomplete")
        else:
            for name, (expected, logical) in expected_design_inputs.items():
                if not _record_matches_logical(design["inputs"].get(name), expected, logical):
                    errors.append(f"probe design input {name} is missing or changed")
        design_source_ok, _, design_source_errors = _probe_source_records_valid(
            design.get("source_files"), design.get("source_sha256"),
            names=("probe_cv.py", "probe.py", "probe_report.py", "evidence.py"),
        )
        if not design_source_ok:
            errors.extend(design_source_errors)
        if not _probe_runtime_valid(design.get("runtime_versions")):
            errors.append("probe design runtime provenance is missing")
        if (
            isinstance(cache_provenance, dict)
            and _probe_runtime_valid(cache_provenance.get("runtime_versions"))
            and design.get("runtime_versions") != cache_provenance.get("runtime_versions")
        ):
            errors.append("probe design/cache runtime provenance differs")
        if not _record_matches_logical(design.get("output"), summary_arrays, "probe_arrays"):
            errors.append("probe design output record is missing or changed")

    upstream = payload.get("upstream_evidence")
    required_upstream = {
        "hidden_cache", "hidden_cache_provenance", "lens", "lens_sidecar",
        "lens_scores", "lens_score_source_sha256", "model_files", "cpu_arrays",
        "cpu_design_provenance", "cpu_source_sha256",
    }
    if not isinstance(upstream, dict) or not required_upstream <= set(upstream):
        errors.append("probe upstream evidence bindings are incomplete")
    else:
        upstream_expected = {
            "hidden_cache": (cache_path, "gpu_cache"),
            "hidden_cache_provenance": (cache_provenance_path, "gpu_cache_provenance"),
            "lens": (PROJECT_ROOT / "artifacts/jlens/lens/exit3/exit3_n80.pt", "lens"),
            "lens_sidecar": (PROJECT_ROOT / "artifacts/jlens/lens/exit3/exit3_n80.json", "lens_sidecar"),
            "lens_scores": (PROJECT_ROOT / "artifacts/jlens/probe/cv_all648/lens_all648.npz", "lens_scores"),
            "cpu_arrays": (summary_arrays, "probe_arrays"),
            "cpu_design_provenance": (design_path, "probe_cpu_design_provenance"),
        }
        for name, (expected, logical) in upstream_expected.items():
            if not _record_matches_logical(upstream.get(name), expected, logical):
                errors.append(f"probe upstream binding {name} is missing or changed")
        if isinstance(provenance, dict) and upstream.get("lens_score_source_sha256") != provenance.get("source_sha256"):
            errors.append("probe upstream source digest differs from lens provenance")
        if isinstance(cache_provenance, dict) and upstream.get("model_files") != cache_provenance.get("model_files"):
            errors.append("probe upstream model manifest differs from cache provenance")
        if isinstance(design, dict):
            if upstream.get("cpu_source_sha256") != design.get("source_sha256"):
                errors.append("probe upstream CPU source digest differs from design")

    try:
        with np.load(summary_arrays, allow_pickle=False) as arrays:
            required_arrays = {"probe_rank", "chosen_C", "selection_accuracy", "folds", "labels", "correct", "jl_cand", "ll_cand"}
            if not required_arrays <= set(arrays.files):
                errors.append("probe arrays are incomplete")
            for name in ("probe_rank", "jl_cand", "ll_cand"):
                if name in arrays and (arrays[name].shape != (648, 192) or not np.issubdtype(arrays[name].dtype, np.integer) or arrays[name].min() < 0):
                    errors.append(f"probe array {name} is malformed")
            if "selection_accuracy" in arrays and (arrays["selection_accuracy"].shape != (5, 192) or not np.isfinite(arrays["selection_accuracy"]).all()):
                errors.append("probe selection accuracy is malformed")
            if "folds" in arrays and arrays["folds"].shape != (648,):
                errors.append("probe fold array is malformed")
    except (OSError, ValueError):
        errors.append("probe arrays are unavailable")

    return {
        "verified": not errors,
        "errors": errors,
        "status": "SUPPORTED_LOCAL_UNFROZEN" if not errors else "INCONCLUSIVE_PROBE_DEGRADED_OR_UNVERIFIED",
        "summary_status": payload.get("status"),
        "design": design,
        "provenance": provenance,
        "source_files": score_source_records if isinstance(provenance, dict) else source_records,
    }


def _generator_record(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        relative = candidate.resolve().relative_to(PROJECT_ROOT.resolve())
        # Hash the resolved file explicitly.  ``file_record(relative)`` would
        # depend on the caller's cwd and could silently fail for a valid
        # report built from another directory.
        record = file_record(candidate)
        record["path"] = str(relative)
        return record
    except (OSError, ValueError):
        try:
            return file_record(candidate)
        except OSError:
            return None


def _generator_set(report: Mapping[str, Any], paths: Iterable[str | Path] = (), records: Iterable[object] = ()) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    def canonical(record: object) -> dict[str, Any] | None:
        if not _record_shape(record):
            return None
        normalized = dict(record)
        candidate = Path(record["path"])
        try:
            normalized["path"] = str(candidate.resolve().relative_to(PROJECT_ROOT.resolve()))
        except (OSError, ValueError):
            normalized["path"] = str(candidate)
        return normalized

    base = report.get("generator")
    normalized_base = canonical(base)
    if normalized_base is not None:
        found[normalized_base["path"]] = normalized_base
    for path in paths:
        record = _generator_record(path)
        if record is not None:
            found[record["path"]] = record
    for record in records:
        normalized = canonical(record)
        if normalized is not None:
            found[normalized["path"]] = normalized
    return [found[path] for path in sorted(found)]


def _claim_defaults(report: dict) -> None:
    generator_hash = report["generator"]["sha256"]
    for record in report["claims"].values():
        generator_set = [report["generator"]]
        record.update({
            "population": None,
            "estimand": None,
            "value": None,
            "uncertainty": None,
            "input_hashes": {},
            "generator_sha256": generator_hash,
            "generator_set": generator_set,
            "generator_set_sha256": aggregate_sha256(
                {item["path"]: item["sha256"] for item in generator_set}
            ),
        })


def _main_state(
    main: object,
    *,
    artifact_root: Path | None = None,
) -> tuple[str, list[str]]:
    if not isinstance(main, dict):
        return "INCONCLUSIVE_MAIN_EVIDENCE_INVALID", ["main is not an object"]
    errors: list[str] = []
    expected_inputs = {"arrays.npz", "items.json", "task_names.json"}
    inputs = main.get("input")
    if not isinstance(inputs, dict) or not expected_inputs <= set(inputs) or not all(_record_shape(record) for record in inputs.values()):
        errors.append("main input records are incomplete")
    elif artifact_root is None and not all(_record_matches(record) for record in inputs.values()):
        errors.append("main input bytes are unavailable or changed")
    elif artifact_root is not None and not all(
        _artifact_lens_record_matches(record, artifact_root=artifact_root) is not None
        for record in inputs.values()
    ):
        errors.append("main input bytes are unavailable or changed")
    populations = main.get("populations")
    if not isinstance(populations, dict) or set(populations) != {"multihop", "order_ops_numeric"}:
        errors.append("main population set is incomplete")
    else:
        for name in populations:
            row = populations[name]
            paired = row.get("paired_j_minus_logit") if isinstance(row, dict) else None
            if not isinstance(row, dict) or not all(isinstance(row.get(k), int) and not isinstance(row.get(k), bool) and row[k] >= 0 for k in ("n_raw_items", "n_clean_items", "n_clean_slots")):
                errors.append(f"main population {name} is malformed")
            effects = paired.get("excess_pass10_per_loop") if isinstance(paired, dict) else None
            intervals = paired.get("ci95_unadjusted") if isinstance(paired, dict) else None
            valid_effects = (
                isinstance(effects, list)
                and len(effects) == 4
                and all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) for value in effects)
            )
            valid_intervals = (
                isinstance(intervals, list)
                and len(intervals) == 4
                and all(
                    isinstance(interval, list)
                    and len(interval) == 2
                    and all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) for value in interval)
                    for interval in intervals
                )
            )
            if not valid_effects or not valid_intervals:
                errors.append(f"main population {name} paired statistic is malformed")
    fit_prompts = main.get("fit_prompts")
    if fit_prompts is not None and (not isinstance(fit_prompts, int) or isinstance(fit_prompts, bool) or fit_prompts <= 0):
        errors.append("main fit_prompts is malformed")
    if errors:
        return "INCONCLUSIVE_MAIN_EVIDENCE_INVALID", errors
    if fit_prompts is not None and fit_prompts != 80:
        return "INCONCLUSIVE_NON_80_MAIN_INPUT", [f"main fit_prompts={fit_prompts}, expected 80"]
    lineage = main.get("lineage")
    lineage_verified = _main_lineage_is_current(lineage, artifact_root=artifact_root)
    if not lineage_verified:
        return (
            LINEAGE_UNFROZEN,
            ["main evaluation provenance is missing or unverified"],
        )
    return "SUPPORTED_LOCAL_UNFROZEN", []


def _main_lineage_is_current(
    lineage: object,
    *,
    artifact_root: Path | None = None,
) -> bool:
    """Re-open the producer record before accepting a current fit size."""

    if not isinstance(lineage, dict) or lineage.get("verified") is not True or lineage.get("status") != CURRENT_GENERATION:
        return False
    record = lineage.get("record")
    if not _record_shape(record):
        return False
    if artifact_root is None:
        if not _record_matches(record):
            return False
    else:
        if _artifact_lens_record_matches(record, artifact_root=artifact_root) is None:
            return False
    # ``main_readout`` exposes the provenance record but intentionally omits
    # its large payload from the canonical JSON.  Re-open the record by path so
    # a caller cannot promote a fabricated ``verified`` boolean or fit field.
    if artifact_root is None:
        candidates = _path_candidates(record["path"])
    else:
        candidate = _artifact_lens_record_matches(record, artifact_root=artifact_root)
        candidates = [candidate] if candidate is not None else []
    for candidate in candidates:
        if candidate.name != "provenance.json" or not candidate.is_file():
            continue
        try:
            checked = (
                _evaluation_info(candidate.parent, artifact_root=artifact_root)
                if artifact_root is not None
                else _evaluation_info(candidate.parent)
            )
        except (AttributeError, IndexError, OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if checked.get("verified") is True and checked.get("status") == CURRENT_GENERATION:
            return True
    return False


def classify_loop_effect(effect: object, interval: object) -> str:
    """Classify one effect using its unadjusted confidence interval.

    A directional label is emitted only when the finite point estimate and
    the corresponding interval endpoint agree strictly.  Zero, malformed,
    and crossing intervals remain ``inconclusive``.
    """

    if (
        isinstance(effect, (int, float))
        and not isinstance(effect, bool)
        and np.isfinite(effect)
        and isinstance(interval, (list, tuple))
        and len(interval) == 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) for value in interval)
    ):
        lower, upper = float(interval[0]), float(interval[1])
        point = float(effect)
        if lower > upper:
            return "inconclusive"
        if point < 0 and upper < 0:
            return "below"
        if point > 0 and lower > 0:
            return "above"
    return "inconclusive"


def derive_loop_decisions(
    effects: object, intervals: object, *, count: int = 4
) -> list[str]:
    """Return one data-derived direction for every requested loop."""

    if not isinstance(effects, list) or not isinstance(intervals, list):
        return ["inconclusive"] * count
    return [
        classify_loop_effect(
            effects[index] if index < len(effects) else None,
            intervals[index] if index < len(intervals) else None,
        )
        for index in range(count)
    ]


def _loop_effect_evidence(effects: object, intervals: object, *, count: int = 4) -> list[dict[str, Any]]:
    decisions = derive_loop_decisions(effects, intervals, count=count)
    rows: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        effect = effects[index] if isinstance(effects, list) and index < len(effects) else None
        interval = intervals[index] if isinstance(intervals, list) and index < len(intervals) else None
        rows.append({
            "loop": index + 1,
            "effect": effect if isinstance(effect, (int, float)) and not isinstance(effect, bool) and np.isfinite(effect) else None,
            "ci95_unadjusted": list(interval) if isinstance(interval, (list, tuple)) and len(interval) == 2 and all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) for value in interval) else None,
            "decision": decision,
        })
    return rows


def _loop_wording(decisions: list[str]) -> str:
    labels = {
        "below": "below zero",
        "above": "above zero",
        "inconclusive": "inconclusive",
    }
    groups: list[str] = []
    for decision in ("below", "above", "inconclusive"):
        loops = [str(index + 1) for index, value in enumerate(decisions) if value == decision]
        if loops:
            groups.append(f"{labels[decision]} at loops {', '.join(loops)}")
    return "; ".join(groups) if groups else "inconclusive"


def _effect_claim_status(
    lineage_status: str,
    decisions: list[str],
    *,
    arithmetic: bool = False,
) -> str:
    """Combine dynamic effect direction with the existing lineage downgrade."""

    if lineage_status == LINEAGE_UNFROZEN:
        return LINEAGE_UNFROZEN
    resolved = {decision for decision in decisions if decision != "inconclusive"}
    if not resolved:
        return "INCONCLUSIVE_NO_RESOLVED_LOOPS"
    if resolved == {"below"}:
        return "SUPPORTED_LOCAL_UNFROZEN"
    if resolved == {"above"}:
        return "SUPPORTED_ABOVE_LOCAL_UNFROZEN"
    return "SUPPORTED_MIXED_LOCAL_UNFROZEN"


def _derive_claim_statuses(
    report: dict,
    *,
    artifact_root: Path | None = None,
) -> None:
    claims = report["claims"]
    main_status, main_errors = _main_state(
        report.get("main"),
        artifact_root=artifact_root,
    )
    main = report.get("main") if isinstance(report.get("main"), dict) else {}
    main_lineage = main.get("lineage", {}) if isinstance(main, dict) else {}
    if main_status in {"SUPPORTED_LOCAL_UNFROZEN", LINEAGE_UNFROZEN}:
        main_hashes = _record_hashes(main.get("input"))
        fit = main.get("fit_prompts")
        fit_verified = main_status == "SUPPORTED_LOCAL_UNFROZEN" and fit == 80
        fit_wording = "the verified n=80 fit" if fit_verified else "the retained evaluation (fit size not authenticated)"
        effect_specs = (
            ("multihop_relative_deficit", "multihop", "task-defined multihop intermediate-token readout", False),
            ("arithmetic_relative_deficit", "order_ops_numeric", "task-defined arithmetic intermediate-token readout", True),
        )
        for claim_name, public_name, population_wording, arithmetic in effect_specs:
            population = main["populations"][public_name]
            paired = population["paired_j_minus_logit"]
            effects = paired["excess_pass10_per_loop"]
            intervals = paired["ci95_unadjusted"]
            decisions = derive_loop_decisions(effects, intervals)
            evidence_rows = _loop_effect_evidence(effects, intervals)
            effect_status = _effect_claim_status(main_status, decisions, arithmetic=arithmetic)
            # A lineage downgrade remains the externally visible status for
            # retained historical fits, while ``effect_status`` preserves the
            # data-derived direction for consumers that need it.
            claims[claim_name].update({
                "status": effect_status,
                "effect_status": _effect_claim_status("SUPPORTED_LOCAL_UNFROZEN", decisions, arithmetic=arithmetic),
                "loop_decisions": decisions,
                "loop_evidence": evidence_rows,
                "claim": (
                    f"On {fit_wording}, the paired J-minus-logit effect for {population_wording} "
                    f"is {_loop_wording(decisions)}; this is local arithmetic with "
                    f"{'current' if fit_verified else 'unfrozen'} evaluation lineage."
                ),
            })
        claims["lens_free_exit_agreement"].update({
            "status": main_status,
            "claim": "Item-weighted exit agreement is descriptive on the retained 93/55-stimulus populations; it is not current-generation evidence.",
        })
        for name, public in (("multihop_relative_deficit", "multihop"), ("arithmetic_relative_deficit", "order_ops_numeric")):
            population = main["populations"][public]
            claims[name]["population"] = {key: population[key] for key in ("n_raw_items", "n_clean_items", "n_clean_slots")}
            claims[name]["estimand"] = main.get("estimand")
            paired = population["paired_j_minus_logit"]
            claims[name]["value"] = paired["excess_pass10_per_loop"]
            claims[name]["uncertainty"] = {"ci95_unadjusted": paired["ci95_unadjusted"]}
            claims[name]["input_hashes"] = main_hashes
        claims["lens_free_exit_agreement"].update({
            "population": {
                "multihop": main["populations"]["multihop"]["n_raw_items"],
                "order_ops": main["populations"]["order_ops_numeric"]["n_raw_items"],
            },
            "estimand": "fraction of unique items whose exit-k top token equals the final-exit top token",
            "value": main.get("lens_free_exits"),
            "input_hashes": main_hashes,
        })
    else:
        reason = "; ".join(main_errors)
        for name in ("multihop_relative_deficit", "arithmetic_relative_deficit", "lens_free_exit_agreement", "cross_loop_association"):
            claims[name]["status"] = main_status
            claims[name]["claim"] = f"Not established: verified main evidence is unavailable ({reason})."
        for name, public in (
            ("multihop_relative_deficit", "multihop"),
            ("arithmetic_relative_deficit", "order_ops_numeric"),
        ):
            population = main.get("populations", {}).get(public, {}) if isinstance(main.get("populations"), dict) else {}
            paired = population.get("paired_j_minus_logit", {}) if isinstance(population, dict) else {}
            effects = paired.get("excess_pass10_per_loop") if isinstance(paired, dict) else None
            intervals = paired.get("ci95_unadjusted") if isinstance(paired, dict) else None
            decisions = derive_loop_decisions(effects, intervals)
            claims[name].update({
                "effect_status": "INCONCLUSIVE_MALFORMED_EFFECT_EVIDENCE",
                "loop_decisions": decisions,
                "loop_evidence": _loop_effect_evidence(effects, intervals),
                "claim": (
                    f"Not established: verified main evidence is unavailable ({reason}); "
                    f"retained loop classifications are {_loop_wording(decisions)}."
                ),
            })
    claims["instrumentation"]["lineage_status"] = report.get("validation", {}).get("lineage", {}).get("status") if isinstance(report.get("validation"), dict) else "NOT_RETAINED"
    validation = report.get("validation")
    if isinstance(validation, dict):
        info = validation.get("lineage", {})
        if isinstance(info, dict) and info.get("verified"):
            claims["instrumentation"].update({
                "status": "SUPPORTED_CURRENT_VALIDATION",
                "claim": "Current-source validation passes the complete M1-M5 numerical gate; comparison count, source identities, and bit-exact status are recorded separately.",
                "population": "current source, fixed local Ouro snapshot",
                "estimand": "M1-M5 mechanical validation gate",
                "value": {"numerical_pass": validation.get("numerical_pass"), "bit_exact": validation.get("bit_exact")},
                "input_hashes": _record_hashes(validation),
            })
            cuda = validation.get("provenance", {}).get("cuda")
            devices = cuda.get("devices") if isinstance(cuda, dict) else None
            b300_devices = (
                isinstance(devices, list)
                and bool(devices)
                and all(
                    isinstance(device, dict)
                    and isinstance(device.get("name"), str)
                    and re.search(r"\bB300\b", device["name"], re.IGNORECASE)
                    for device in devices
                )
            )
            if b300_devices and info.get("paid_accepted") is True:
                claims["b300_validation"].update({
                    "status": "SUPPORTED_CURRENT_B300_VALIDATION",
                    "claim": (
                        "Current-source validation passes the complete M1-M5 numerical gate "
                        "on the recorded B300 device set."
                    ),
                    "population": "current source and fixed Ouro snapshot on the recorded B300 device set",
                    "estimand": "M1-M5 mechanical validation gate",
                    "value": {
                        "devices": devices,
                        "numerical_pass": validation.get("numerical_pass"),
                        "bit_exact": validation.get("bit_exact"),
                    },
                    "input_hashes": _record_hashes(validation),
                })
            else:
                approved_image = approved_b300_image_digest()
                claims["b300_validation"].update({
                    "status": "NOT_CURRENT_B300_VALIDATION",
                    "claim": (
                        "The current validation artifact is hash-bound but does not satisfy both "
                        "the exclusively-B300 and controller-approved immutable-image gates."
                    ),
                    "population": "recorded validation device set",
                    "estimand": "M1-M5 mechanical validation gate on B300 hardware",
                    "value": {
                        "devices": devices,
                        "image_digest": validation.get("provenance", {}).get("image_digest"),
                        "approved_image_digest": approved_image,
                    },
                    "input_hashes": _record_hashes(validation),
                })
        else:
            claims["instrumentation"]["status"] = "FAILED_OR_INCOMPLETE_CURRENT_VALIDATION"
            claims["instrumentation"]["claim"] = "Instrumentation remains open because the supplied validation artifact is missing, incomplete, stale, or failed a full M1-M5/provenance check."

    local = report.get("local_vs_eventual")
    if isinstance(local, dict) and isinstance(local.get("rows"), list) and local["rows"]:
        local_status = local.get("status")
        if local_status not in {
            "REFUTED_PRE_FINAL",
            "OBSERVED_PRE_FINAL_MONOTONIC_DECREASE",
        }:
            local_status = "INCONCLUSIVE_LOCAL_EVENTUAL_EVIDENCE"
        local_lineage = local.get("lineage") if isinstance(local.get("lineage"), dict) else {}
        current_wording = (
            "a current hash-verified evaluation"
            if local_lineage.get("verified") is True
            else "a retained historical comparison with unfrozen evaluation lineage"
        )
        pattern = (
            "decreases strictly"
            if local_status == "OBSERVED_PRE_FINAL_MONOTONIC_DECREASE"
            else "does not decrease strictly"
        )
        claims["local_eventual_convergence"].update({
            "status": local_status,
            "claim": (
                f"Mean KL from the local-exit readout to the eventual-exit readout {pattern} "
                f"over loops 1-3 on {current_wording}."
            ),
            "population": local.get("population"),
            "estimand": "mean KL from each local-exit readout to the eventual-exit readout at pre-final loops",
            "value": local["rows"],
            "input_hashes": _record_hashes(local),
        })
    else:
        claims["local_eventual_convergence"]["status"] = "INCONCLUSIVE_LOCAL_EVENTUAL_EVIDENCE"

    fit_size = report.get("fit_size")
    if isinstance(fit_size, dict) and _fit_size_valid(fit_size):
        multihop = fit_size["populations"].get("multihop", {})
        claims["large_fit_robustness"].update({
            "status": "INCONCLUSIVE",
            "population": {"n_items": multihop.get("n_items"), "nested_fit_sizes": fit_size.get("fit_sizes")},
            "estimand": "n=8 to n=80 change in item-mean log10(best-rank+1); positive is improvement",
            "value": multihop.get("n80_minus_n8_mean_log10_rank_improvement"),
            "uncertainty": {"ci95_unadjusted": multihop.get("improvement_ci95")},
            "input_hashes": _record_hashes(fit_size),
        })
    else:
        claims["large_fit_robustness"]["status"] = "INCONCLUSIVE_MISSING_OR_INVALID_FIT_SIZE_EVIDENCE"

    cross = report.get("cross_loop")
    cross_lineage_verified = (
        isinstance(cross, dict)
        and cross.get("lineage_verified") is True
        and cross.get("lineage_status") == CURRENT_GENERATION
        and isinstance(cross.get("fits"), dict)
        and set(cross["fits"]) == {"32", "80"}
        and all(
            isinstance(cross["fits"].get(str(size)), dict)
            and cross["fits"][str(size)].get("fit_prompts") == size
            and isinstance(cross["fits"][str(size)].get("lineage"), dict)
            and cross["fits"][str(size)]["lineage"].get("verified") is True
            and cross["fits"][str(size)]["lineage"].get("status") == CURRENT_GENERATION
            and isinstance(cross["fits"][str(size)].get("multihop"), dict)
            and isinstance(cross["fits"][str(size)].get("order_ops_numeric"), dict)
            for size in (32, 80)
        )
    )
    if main_status in {
        "SUPPORTED_LOCAL_UNFROZEN",
        LINEAGE_UNFROZEN,
    } and isinstance(cross, dict) and isinstance(cross.get("fits"), dict) and cross["fits"]:
        cross_status = (
            "SUPPORTED_ASSOCIATION_MULTIHOP"
            if main_status == "SUPPORTED_LOCAL_UNFROZEN" and cross_lineage_verified
            else LINEAGE_UNFROZEN
        )
        claims["cross_loop_association"].update({
            "status": cross_status,
            "lineage_status": cross.get("lineage_status", LINEAGE_UNFROZEN),
            "lineage_verified": cross_lineage_verified,
            "population": "retained n=32 and n=80 clean multihop and arithmetic items",
            "estimand": "any-layer excess-hit@10 matrix indexed by fit loop and state loop",
            "value": cross["fits"],
            "input_hashes": _record_hashes(cross),
        })
    transport = report.get("transport")
    if isinstance(transport, dict) and _transport_valid(transport):
        claims["transport_mechanism"].update({
            "status": "INCONCLUSIVE",
            "population": {"sample_sizes": transport.get("sample_sizes"), "n_sources": transport.get("n_sources")},
            "estimand": transport.get("model"),
            "value": {"per_loop": transport.get("per_loop"), "negative_sigma_squared_total": transport.get("negative_sigma_squared_total")},
            "input_hashes": _record_hashes(transport),
        })
    probe = report.get("probe")
    if isinstance(probe, dict):
        probe_info = probe.get("lineage", {})
        probe_fields = {
            "population": probe.get("population"),
            "estimand": probe.get("estimand"),
            "value": {"readouts": probe.get("readouts"), "baselines": probe.get("baselines")},
            "uncertainty": {"contrasts": probe.get("contrasts"), "bootstrap": probe.get("bootstrap")},
            "input_hashes": _record_hashes(probe),
        }
        claims["supervised_probe"].update(probe_fields)
        claims["probe_familywide_inference"].update(probe_fields)
        if isinstance(probe_info, dict) and probe_info.get("verified"):
            claims["supervised_probe"]["status"] = "SUPPORTED_LOCAL_UNFROZEN"
            claims["probe_familywide_inference"]["status"] = "INCONCLUSIVE_UNADJUSTED_INTERVALS"
        else:
            claims["supervised_probe"]["status"] = "INCONCLUSIVE_PROBE_DEGRADED_OR_UNVERIFIED"
            claims["probe_familywide_inference"]["status"] = "INCONCLUSIVE_PROBE_DEGRADED_OR_UNVERIFIED"
    checkpoints = report.get("checkpoints")
    if isinstance(checkpoints, dict):
        checkpoint_info = checkpoints.get("lineage", {})
        if isinstance(checkpoint_info, dict) and checkpoint_info.get("verified"):
            measured = _checkpoint_rows(checkpoints)
            claims["checkpoint_comparison"].update({
                "status": "SUPPORTED_LOCAL_UNFROZEN",
                "claim": checkpoint_claim_text(checkpoint_info),
                "population": "retained base, Thinking, and RLTT checkpoint evaluation stimuli",
                "estimand": "exit-to-final KL, JS, entropy, final-token rank, and top-1 agreement",
                "value": measured,
                "input_hashes": _record_hashes(checkpoints),
            })
        else:
            claims["checkpoint_comparison"]["status"] = "INCONCLUSIVE_CHECKPOINT_EVIDENCE_UNVERIFIED"


def _complete_claim_records(
    report: dict,
    *,
    artifact_root: Path | None = None,
) -> None:
    """Attach values, input hashes, and the complete upstream generator set."""

    _claim_defaults(report)
    _derive_claim_statuses(report, artifact_root=artifact_root)
    claims = report["claims"]
    source_root = Path(__file__).resolve().parent
    analyze_path = source_root / "analyze.py"
    generator_paths: dict[str, tuple[Path, ...]] = {
        "multihop_relative_deficit": (analyze_path, source_root / "evaluate.py", source_root / "evaldata.py"),
        "arithmetic_relative_deficit": (analyze_path, source_root / "evaluate.py", source_root / "evaldata.py"),
        "lens_free_exit_agreement": (analyze_path, source_root / "evaluate.py", source_root / "evaldata.py"),
        "local_eventual_convergence": (analyze_path,),
        "large_fit_robustness": (analyze_path, source_root / "fitsize_report.py"),
        "cross_loop_association": (analyze_path, source_root / "evaluate.py"),
        "transport_mechanism": (source_root / "transport_report.py",),
        "supervised_probe": (source_root / "probe_report.py", source_root / "probe_cv.py", source_root / "probe.py", source_root / "evaluate.py"),
        "probe_familywide_inference": (source_root / "probe_report.py", source_root / "probe_cv.py", source_root / "probe.py", source_root / "evaluate.py"),
        "checkpoint_comparison": (source_root / "checkpoints.py",),
        "instrumentation": (source_root / "validate.py", source_root / "recurrent.py", source_root / "evidence.py"),
        "estimator_independence": (source_root / "evaluate.py",),
        "architecture_cause": (source_root / "recurrent.py",),
        "b300_validation": (source_root / "run_b300.sh", source_root / "setup_b300.sh"),
        "submission_readiness": (source_root / "manifest.py", source_root / "verify_artifacts.py"),
    }
    evidence_names = {
        "instrumentation": "validation",
        "local_eventual_convergence": "local_vs_eventual",
        "large_fit_robustness": "fit_size",
        "cross_loop_association": "cross_loop",
        "transport_mechanism": "transport",
        "supervised_probe": "probe",
        "probe_familywide_inference": "probe",
        "checkpoint_comparison": "checkpoints",
    }
    for name, paths in generator_paths.items():
        records = []
        evidence = report.get(evidence_names.get(name, "main"))
        if isinstance(evidence, dict):
            info = evidence.get("lineage")
            if isinstance(info, dict):
                records.extend(info.get("source_files", []))
                provenance = info.get("provenance")
                if isinstance(provenance, dict):
                    records.extend(provenance.get("source_files", []))
                design = info.get("design")
                if isinstance(design, dict):
                    records.extend(design.get("source_files", []))
        claims[name]["generator_set"] = _generator_set(report, paths, records)
        claims[name]["generator_sha256"] = report["generator"]["sha256"]
        claims[name]["generator_set_sha256"] = aggregate_sha256({record["path"]: record["sha256"] for record in claims[name]["generator_set"]})


def _legacy_eval(eval_dir: Path) -> analyze.Eval:
    """Load a retained pre-custody evaluation without promoting its lineage.

    The current ``analyze.Eval`` intentionally rejects the padded slot
    metadata used by the historical directories.  Their arrays remain useful
    for local arithmetic, so construct the same analysis state without the
    current-generation structure/provenance gate; the caller still records
    ``LINEAGE_UNFROZEN`` from ``_evaluation_info``.
    """

    ev = analyze.Eval.__new__(analyze.Eval)
    ev.provenance = None
    with np.load(eval_dir / "arrays.npz", allow_pickle=False) as values:
        ev.arrays = {key: values[key] for key in values.files}
    ev.items = json.loads((eval_dir / "items.json").read_text())
    for item in ev.items:
        if "continuation" in item and "target" in item:
            item["correct"] = analyze.boundary_prefix_match(item["continuation"], item["target"])
    ev.task_names = json.loads((eval_dir / "task_names.json").read_text())
    n, max_names = ev.arrays["jlens_exit3_allrank"].shape[:2]
    ev.own = np.zeros((n, max_names), bool)
    ev.valid = np.zeros((n, max_names), bool)
    ev.is_op = np.zeros((n, max_names), bool)
    ev.slot_mask = analyze.Eval.groups(ev)
    for i, item in enumerate(ev.items):
        names = ev.task_names[item["task"]]
        ev.valid[i, : len(names)] = True
        ev.is_op[i, : len(names)] = [name in analyze.OPERATIONS for name in names]
        for index in item["own_index"]:
            if index >= 0:
                ev.own[i, index] = True
    return ev


def _load_eval(
    eval_dir: Path,
    *,
    artifact_root: Path | None = None,
) -> analyze.Eval:
    """Load current evaluations strictly and old ones for observation only."""

    lineage = (
        _evaluation_info(eval_dir, artifact_root=artifact_root)
        if artifact_root is not None
        else _evaluation_info(eval_dir)
    )
    try:
        # ``analyze.Eval`` has no relocation parameter.  When the report has
        # already run the full relocation-aware evaluator gate above, avoid a
        # second default-root validation that would incorrectly reopen the
        # project artifact tree.
        return analyze.Eval(
            eval_dir,
            verify_provenance=(artifact_root is None and lineage.get("verified") is True),
        )
    except ValueError:
        if lineage.get("verified") is True:
            raise
        return _legacy_eval(eval_dir)


def _ci(values: np.ndarray) -> list[list[float]]:
    # ``analyze.boot_ci`` is scalar-oriented; calculate vector intervals here.
    rng = np.random.default_rng(analyze.BOOT_SEED)
    draws = np.stack([
        values[rng.integers(0, len(values), len(values))].mean(0)
        for _ in range(analyze.N_BOOT)
    ])
    return np.percentile(draws, [2.5, 97.5], axis=0).T.round(3).tolist()


def main_readout(
    eval_dir: Path,
    *,
    artifact_root: Path | None = None,
) -> tuple[dict, analyze.Eval]:
    lineage = (
        _evaluation_info(eval_dir, artifact_root=artifact_root)
        if artifact_root is not None
        else _evaluation_info(eval_dir)
    )
    # Historical evaluations may intentionally lack the current provenance
    # wrapper.  Read their retained arrays for descriptive arithmetic, then
    # let ``_evaluation_info`` independently keep their lineage unfrozen.
    ev = (
        _load_eval(eval_dir, artifact_root=artifact_root)
        if artifact_root is not None
        else _load_eval(eval_dir)
    )
    input_records = {
        name: _canonical_artifact_record(eval_dir / name, artifact_root)
        for name in ("arrays.npz", "items.json", "task_names.json")
    }
    if lineage.get("record"):
        input_records["provenance"] = lineage["record"]
    out = {
        "input": input_records,
        # This value is filled only from a verified producer chain.  The old
        # ``fitsize_n80`` directory name is intentionally not evidence.
        "fit_prompts": lineage.get("fit_prompts"),
        "fit_prompts_status": "VERIFIED_FROM_PROVENANCE" if lineage.get("verified") else "NOT_AUTHENTICATED",
        "lineage": {key: value for key, value in lineage.items() if key not in {"payload"}},
        "estimand": (
            "item-mean of own-name hit@10 at any layer in a loop minus the mean of the "
            "same any-layer statistic over matched control names"
        ),
        "bootstrap": {"unit": "unique_item", "draws": analyze.N_BOOT, "seed": analyze.BOOT_SEED},
        "populations": {},
    }
    task_specs = (("multihop", "multihop"), ("order_ops_numeric", "order-ops numeric"))
    for public_name, mask_name in task_specs:
        mask = ev.slot_mask[mask_name]
        ii, _ = ev.own_slots(mask)
        raw_task = "order-ops" if mask_name.startswith("order-ops") else mask_name
        raw_count = sum(item["task"] == raw_task for item in ev.items)
        population = {
            "n_raw_items": raw_count,
            "n_clean_items": int(len(np.unique(ii))),
            "n_clean_slots": int(len(ii)),
            "n_clean_model_correct_items": int(len(np.unique(ev.own_slots(
                ev.slot_mask[f"{mask_name} (model correct)"]
            )[0]))),
            "readouts": {},
        }
        item_values = {}
        for label, key in (
            ("eventual_exit_jacobian_lens", "jlens_exit3_allrank"),
            ("logit_lens", "logitlens_allrank"),
        ):
            values = analyze.any_layer_items(ev.scores(ev.arrays[key], mask))
            item_values[label] = values
            population["readouts"][label] = {
                metric: array.mean(0).round(3).tolist()
                for metric, array in values.items()
            }
        delta = item_values["eventual_exit_jacobian_lens"]["excess_pass10"] - item_values["logit_lens"]["excess_pass10"]
        population["paired_j_minus_logit"] = {
            "excess_pass10_per_loop": delta.mean(0).round(3).tolist(),
            "ci95_unadjusted": _ci(delta),
        }
        out["populations"][public_name] = population
    out["corrected_model_accuracy"] = {
        "n_correct_all_items": int(sum(item["correct"] for item in ev.items)),
        "n_items": len(ev.items),
        "definition": "boundary-aware prefix match recomputed from retained continuation and target",
    }
    out["lens_free_exits"] = analyze.model_exits(ev)
    return out, ev


def local_eventual(
    local_eval_dir: Path,
    *,
    artifact_root: Path | None = None,
) -> dict:
    ev = (
        _load_eval(local_eval_dir, artifact_root=artifact_root)
        if artifact_root is not None
        else _load_eval(local_eval_dir)
    )
    arrays = ev.arrays
    rows = []
    mean_kl = []
    for loop in range(analyze.N_UT - 1):
        block = slice(loop * analyze.N_LAYER, (loop + 1) * analyze.N_LAYER)
        local_key = f"jlens_exit{loop}"
        if f"{local_key}_allrank" not in arrays:
            raise ValueError(f"missing {local_key} in {local_eval_dir}")
        local_values = analyze.loc_maps(ev.scores(
            arrays[f"{local_key}_allrank"], ev.slot_mask["multihop"]
        ))["excess"][loop]
        eventual_values = analyze.loc_maps(ev.scores(
            arrays["jlens_exit3_allrank"], ev.slot_mask["multihop"]
        ))["excess"][loop]
        loop_mean_kl = float(
            arrays[f"{local_key}_kl_to_eventual_readout"][:, block].mean()
        )
        if not np.isfinite(loop_mean_kl):
            raise ValueError(f"non-finite local/eventual KL at loop {loop + 1}")
        mean_kl.append(loop_mean_kl)
        rows.append({
            "loop": loop + 1,
            "mean_kl_local_to_eventual": round(loop_mean_kl, 3),
            "top1_agreement_last8_layers": round(float((
                arrays[f"{local_key}_top1"][:, block][:, -8:]
                == arrays["jlens_exit3_top1"][:, block][:, -8:]
            ).mean()), 3),
            "multihop_mean_layer_excess_local": round(float(local_values.mean()), 3),
            "multihop_mean_layer_excess_eventual": round(float(eventual_values.mean()), 3),
            "local_minus_eventual": round(float((local_values - eventual_values).mean()), 3),
        })
    lineage = (
        _evaluation_info(local_eval_dir, artifact_root=artifact_root)
        if artifact_root is not None
        else _evaluation_info(local_eval_dir)
    )
    lens_counts: dict[int, int] = {}
    payload = lineage.get("payload") if isinstance(lineage, dict) else None
    lens_inputs = payload.get("lens_inputs") if isinstance(payload, dict) else None
    if lineage.get("verified") is True and isinstance(lens_inputs, list):
        for entry in lens_inputs:
            identity = entry.get("identity") if isinstance(entry, dict) else None
            target = entry.get("target_ut") if isinstance(entry, dict) else None
            count = identity.get("n_prompts") if isinstance(identity, dict) else None
            if type(target) is int and type(count) is int and 0 <= target < analyze.N_UT and count > 0:
                lens_counts[target] = count
    if set(lens_counts) == set(range(analyze.N_UT)):
        if len(set(lens_counts.values())) == 1:
            count = next(iter(lens_counts.values()))
            fit_note = f"all local and eventual lenses use the same authenticated n={count} prompt prefix"
        else:
            fit_note = "authenticated fit prompts by target: " + ", ".join(
                f"exit {target} n={lens_counts[target]}" for target in sorted(lens_counts)
            )
    else:
        fit_note = "lens fit counts are not authenticated by current evaluation provenance"
    strictly_decreasing = bool(np.all(np.diff(np.asarray(mean_kl, dtype=float)) < 0))
    return {
        "status": (
            "OBSERVED_PRE_FINAL_MONOTONIC_DECREASE"
            if strictly_decreasing
            else "REFUTED_PRE_FINAL"
        ),
        "fit_note": fit_note,
        "fit_prompts_by_target": {str(key): value for key, value in sorted(lens_counts.items())},
        "strictly_decreasing_mean_kl": strictly_decreasing,
        "population": {
            "task": "multihop",
            "n_items": sum(item.get("task") == "multihop" for item in ev.items),
        },
        "rows": rows,
        "input": _canonical_artifact_record(local_eval_dir / "arrays.npz", artifact_root),
        "lineage": {key: value for key, value in lineage.items() if key not in {"payload"}},
    }


def cross_loop_by_fit_size(
    eval_dirs: dict[int, Path],
    *,
    artifact_root: Path | None = None,
) -> dict:
    result = {
        "status": LINEAGE_UNFROZEN,
        "interpretation": (
            "Rows index the loop where J was fitted and columns the loop producing the state. "
            "A row-dominated matrix is descriptive and does not identify remaining horizon as a cause."
        ),
        "fits": {},
    }
    for n, directory in sorted(eval_dirs.items()):
        lineage = (
            _evaluation_info(directory, artifact_root=artifact_root)
            if artifact_root is not None
            else _evaluation_info(directory)
        )
        try:
            input_record = _canonical_artifact_record(directory / "arrays.npz", artifact_root)
        except (OSError, ValueError):
            input_record = {
                "path": str(directory / "arrays.npz"),
                "status": "UNAVAILABLE",
            }
        row = {
            "input": input_record,
            "fit_prompts": lineage.get("fit_prompts"),
            "lineage": {key: value for key, value in lineage.items() if key not in {"payload"}},
        }
        try:
            ev = (
                _load_eval(directory, artifact_root=artifact_root)
                if artifact_root is not None
                else _load_eval(directory)
            )
            row.update({
                "multihop": analyze.cross_loop_summary(ev, ev.slot_mask["multihop"]),
                "order_ops_numeric": analyze.cross_loop_summary(
                    ev, ev.slot_mask["order-ops numeric"]
                ),
            })
        except Exception as exc:
            prior_errors = row["lineage"].get("errors", [])
            if not isinstance(prior_errors, list):
                prior_errors = []
            row["lineage"]["errors"] = [*prior_errors, f"cross-loop arrays unavailable: {exc}"]
            row["status"] = "INCONCLUSIVE_CROSS_LOOP_INPUT_INVALID"
        result["fits"][str(n)] = row
    exact_current = (
        set(result["fits"]) == {"32", "80"}
        and all(
            isinstance(entry.get("lineage"), dict)
            and entry["lineage"].get("verified") is True
            and entry["lineage"].get("status") == CURRENT_GENERATION
            and entry.get("fit_prompts") == int(size)
            and isinstance(entry.get("multihop"), dict)
            and isinstance(entry.get("order_ops_numeric"), dict)
            for size, entry in ((32, result["fits"]["32"]), (80, result["fits"]["80"]))
        )
    )
    result["lineage_status"] = CURRENT_GENERATION if exact_current else LINEAGE_UNFROZEN
    result["status"] = result["lineage_status"]
    result["lineage_verified"] = exact_current
    return result


def _optional_payload(
    path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        return {"lineage": {"verified": False, "status": "NOT_RETAINED", "errors": [f"missing artifact: {path}"]}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"lineage": {"verified": False, "status": "INVALID_ARTIFACT", "errors": [str(exc)]}, "_record": _canonical_artifact_record(path, artifact_root)}
    if not isinstance(payload, dict):
        payload = {"value": payload}
    payload["_record"] = _canonical_artifact_record(path, artifact_root)
    return payload


def build_report(eval_dir: Path, local_eval_dir: Path, *, validation: Path | None = None,
                 fit_size: Path | None = None, transport: Path | None = None,
                 probe: Path | None = None,
                 checkpoints: Path | None = None,
                 artifact_root: Path | None = None) -> dict:
    main, main_ev = (
        main_readout(eval_dir, artifact_root=artifact_root)
        if artifact_root is not None
        else main_readout(eval_dir)
    )
    claims = json.loads(json.dumps(CLAIMS))
    report_generator = _generator_record(Path(__file__).resolve())
    if report_generator is None:
        raise FileNotFoundError(f"report generator is unavailable: {__file__}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "overall_status": main.get("lineage", {}).get("status", LINEAGE_UNFROZEN),
        "scope": (
            "Fixed local Ouro-2.6B snapshot and retained stimuli; task-defined intermediate-token readout "
            "under a prompt-averaged Jacobian-lens estimator"
        ),
        "main": main,
        "local_vs_eventual": (
            local_eventual(local_eval_dir, artifact_root=artifact_root)
            if artifact_root is not None
            else local_eventual(local_eval_dir)
        ),
        "claims": claims,
        "generator": report_generator,
    }
    n32 = eval_dir.parent / "fitsize_n32"
    if main_ev is not None:
        report["cross_loop"] = (
            cross_loop_by_fit_size(
                {32: n32, 80: eval_dir},
                artifact_root=artifact_root,
            )
            if artifact_root is not None
            else cross_loop_by_fit_size({32: n32, 80: eval_dir})
        )
    optional = {
        "validation": validation,
        "fit_size": fit_size,
        "transport": transport,
        "probe": probe,
        "checkpoints": checkpoints,
    }
    for name, path in optional.items():
        if path is None:
            continue
        payload = _optional_payload(path, artifact_root=artifact_root)
        if name == "validation":
            info = _validation_info(payload)
            payload["lineage"] = info
        elif name == "probe":
            info = _probe_valid(payload)
            payload["lineage"] = info
        elif name == "checkpoints":
            info = _checkpoint_info(payload)
            payload["lineage"] = info
        report[name] = payload
    _complete_claim_records(report, artifact_root=artifact_root)
    main_status, _main_errors = _main_state(
        report.get("main"),
        artifact_root=artifact_root,
    )
    if main_status == LINEAGE_UNFROZEN:
        report["overall_status"] = LINEAGE_UNFROZEN
    elif main_status == "SUPPORTED_LOCAL_UNFROZEN":
        report["overall_status"] = CURRENT_GENERATION
    else:
        report["overall_status"] = main_status
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="artifacts/jlens/eval/fitsize_n80")
    parser.add_argument("--local-eval", default="artifacts/jlens/eval/round1_exit3x32")
    parser.add_argument("--validation")
    parser.add_argument("--fit-size", default="artifacts/jlens/final/fit_size.json")
    parser.add_argument("--transport")
    parser.add_argument("--probe")
    parser.add_argument("--checkpoints", default="artifacts/jlens/checkpoints/exit_divergence.json")
    parser.add_argument("--artifact-root")
    parser.add_argument("--out", default="artifacts/jlens/final/analysis.json")
    parser.add_argument("--claims-out", default="artifacts/jlens/final/claim_status.json")
    args = parser.parse_args()
    report = build_report(
        Path(args.eval), Path(args.local_eval),
        validation=Path(args.validation) if args.validation else None,
        fit_size=Path(args.fit_size) if args.fit_size else None,
        transport=Path(args.transport) if args.transport else None,
        probe=Path(args.probe) if args.probe else None,
        checkpoints=Path(args.checkpoints) if args.checkpoints else None,
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
    )
    atomic_write_json(Path(args.out), report)
    atomic_write_json(Path(args.claims_out), report["claims"])
    print(json.dumps({"overall_status": report["overall_status"], "claims": report["claims"]}, indent=2))


if __name__ == "__main__":
    main()
