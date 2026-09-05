"""Build and verify the deterministic custody manifest for the JLens study.

The manifest is a byte inventory, not a scientific approval.  It binds the
current code, fixed model snapshot, retained raw inputs, and canonical reports
while preserving the fact that most empirical artifacts predate this custody
pipeline and remain scientifically unfrozen.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Iterable

from ouro_jlens.evidence import aggregate_sha256, atomic_write_json, file_record, sha256_json
from ouro_jlens.recurrent import OURO_REVISION, OURO_SNAPSHOT, PROJECT_ROOT, model_snapshot_files

SCHEMA_VERSION = 1

# A custody manifest is useful only when it has the whole contract surface.
# Keep both sets explicit so verification cannot silently accept a hand-made
# subset that happens to contain a few plausible files.
REQUIRED_GROUPS = frozenset((
    "source",
    "tests",
    "documentation",
    "model_snapshot",
    "retained_raw_inputs",
    "derived_outputs",
    "current_validation",
    "historical_evidence",
    "external_jlens_source",
))
REQUIRED_TOP_LEVEL_KEYS = frozenset((
    "schema_version",
    "status",
    "scientific_effect",
    "project_head",
    "model_revision",
    "jlens_revision",
    "environment",
    "hardware",
    "reproduction_contract",
    "groups",
    "required_groups",
    "required_paths",
    "inventory_sha256",
    "aggregate_sha256",
    "current_validation_status",
    "probe_score_provenance",
))

# A custody manifest that silently omits a load-bearing input is worse than no
# manifest: it gives a false impression of closure.  Keep the minimum accepted
# surface explicit and let ``build`` fail if any member is absent.
REQUIRED_PATHS = (
    "src/ouro_jlens/analyze.py",
    "src/ouro_jlens/evaluate.py",
    "src/ouro_jlens/fit_lens.py",
    "src/ouro_jlens/manifest.py",
    "src/ouro_jlens/probe_cv.py",
    "src/ouro_jlens/probe_report.py",
    "src/ouro_jlens/report.py",
    "src/ouro_jlens/transport_report.py",
    "src/ouro_jlens/validate.py",
    "src/ouro_jlens/verify_artifacts.py",
    "utilities/tests/unit/test_jlens_integrity.py",
    "utilities/tests/unit/test_jlens_rental.py",
    "utilities/tests/unit/test_jlens_reports.py",
    "utilities/tests/unit/test_ouro_jlens.py",
    "utilities/tests/unit/test_peer1_index_map.py",
    "utilities/tests/unit/test_peer3_jlens_audit.py",
    "docs/jlens/README.md",
    "docs/jlens/METHODS.md",
    "docs/jlens/RESULTS.md",
    "docs/jlens/REPRODUCE.md",
    "docs/jlens/HANDOFF.md",
    "docs/jlens/INCIDENT_2026-09-04.md",
    "docs/jlens/AUDIT_INDEX.md",
    "artifacts/jlens/data/wikitext_prompts.json",
    "artifacts/jlens/eval/fitsize_n80/arrays.npz",
    "artifacts/jlens/lens/exit3/exit3_n80.pt",
    "artifacts/jlens/lens/exit3/exit3_n80.json",
    "artifacts/jlens/probe/cv_all648/arrays.npz",
    "artifacts/jlens/probe/cv_all648/design.json",
    "artifacts/jlens/probe/cv_all648/summary.json",
    "artifacts/jlens/final/fit_size.json",
    "artifacts/jlens/final/transport.json",
    "artifacts/jlens/final/analysis.json",
    "artifacts/jlens/final/claim_status.json",
    "artifacts/jlens/final/verification.json",
    "artifacts/jlens/validation/milestones.json",
)

TRACKED_GROUPS = frozenset(("source", "tests", "documentation"))
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


def _relative(path: Path) -> Path:
    # Keep repository symlink paths (not their cache/blob targets) in the
    # manifest.  File hashing still follows the symlink and binds target bytes.
    resolved = path.absolute()
    try:
        return resolved.relative_to(PROJECT_ROOT.absolute())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes project root: {path}") from exc


def _files_under(path: Path, patterns: tuple[str, ...] = ("*",)) -> list[Path]:
    if not path.exists():
        return []
    found: set[Path] = set()
    for pattern in patterns:
        found.update(candidate for candidate in path.rglob(pattern) if candidate.is_file())
    return sorted(found)


def _existing(paths: Iterable[Path]) -> list[Path]:
    return sorted({_relative(path) for path in paths if path.is_file()})


def inventory() -> dict[str, list[Path]]:
    source = _files_under(PROJECT_ROOT / "src" / "ouro_jlens", ("*.py", "*.sh"))
    unit_tests = PROJECT_ROOT / "utilities" / "tests" / "unit"
    tests = sorted({
        *unit_tests.glob("test_jlens*.py"),
        *(unit_tests / name for name in (
            "test_ouro_jlens.py", "test_peer1_index_map.py", "test_peer3_jlens_audit.py"
        )),
    })
    docs = _files_under(PROJECT_ROOT / "docs" / "jlens", ("*.md", "SHA256SUMS"))
    model = _files_under(OURO_SNAPSHOT)

    raw: list[Path] = [PROJECT_ROOT / "artifacts/jlens/data/wikitext_prompts.json"]
    # Evaluation provenance is load-bearing even when the numerical arrays
    # themselves are retained.  Older runs have no such file; that absence is
    # deliberately visible in the inventory rather than inferred from a
    # directory name.
    raw.extend(_files_under(
        PROJECT_ROOT / "artifacts/jlens/eval",
        ("arrays.npz", "items.json", "task_names.json", "provenance.json"),
    ))
    # Bind every retained fitted lens and sidecar, not merely the members used
    # by the headline table.  This prevents an apparently valid manifest from
    # being reused after a shard or comparison fit changes.
    raw.extend(_files_under(PROJECT_ROOT / "artifacts/jlens/lens", ("*.pt", "*.json")))
    raw.extend(PROJECT_ROOT / path for path in (
        "artifacts/jlens/probe/n80_v2/gpu_cache.npz",
        "artifacts/jlens/probe/n80_v2/gpu_cache.provenance.json",
        "artifacts/jlens/probe/cv_all648/lens_all648.npz",
        "artifacts/jlens/probe/cv_all648/lens_all648.provenance.json",
        "artifacts/jlens/checkpoints/exit_divergence.json",
    ))
    # The current probe is a chain: design -> lens-score provenance -> cache
    # provenance.  Bind every provenance document in the current-generation
    # directories so a stripped/recomputed summary cannot look complete.
    for probe_root in (
        PROJECT_ROOT / "artifacts/jlens/probe/n80_v2",
        PROJECT_ROOT / "artifacts/jlens/probe/cv_all648",
    ):
        raw.extend(_files_under(probe_root, ("*.provenance.json",)))
    derived = [PROJECT_ROOT / path for path in (
        "artifacts/jlens/probe/cv_all648/arrays.npz",
        "artifacts/jlens/probe/cv_all648/design.json",
        "artifacts/jlens/probe/cv_all648/summary.json",
        "artifacts/jlens/final/fit_size.json",
        "artifacts/jlens/final/transport.json",
        "artifacts/jlens/final/analysis.json",
        "artifacts/jlens/final/claim_status.json",
        "artifacts/jlens/final/verification.json",
    )]
    validation = sorted((PROJECT_ROOT / "artifacts/jlens/validation").glob("milestones_*.json"))
    validation.extend(
        PROJECT_ROOT / "artifacts/jlens/validation" / name
        for name in ("milestones.json", "latest.json")
        if (PROJECT_ROOT / "artifacts/jlens/validation" / name).is_file()
    )
    validation = sorted(set(validation))
    history = _files_under(PROJECT_ROOT / "artifacts/jlens/probe/history")
    return {
        "source": _existing(source),
        "tests": _existing(tests),
        "documentation": _existing(docs),
        "model_snapshot": _existing(model),
        "retained_raw_inputs": _existing(raw),
        "derived_outputs": _existing(derived),
        "current_validation": _existing(validation),
        "historical_evidence": _existing(history),
    }


def _jlens_identity() -> tuple[list[Path], str]:
    try:
        import jlens

        package = Path(jlens.__file__).resolve().parent
        repo = package.parent
        files = sorted(path for path in package.rglob("*.py") if path.is_file())
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return files, result.stdout.strip()
    except (ImportError, OSError, subprocess.CalledProcessError):
        return [], "UNKNOWN"


def _git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "ls-files", "--error-unmatch", str(path)],
        capture_output=True, check=False,
    )
    return result.returncode == 0


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in (
        "numpy", "scipy", "scikit-learn", "threadpoolctl", "torch", "transformers", "jlens",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED_AS_DISTRIBUTION"
    return versions


def _hardware() -> dict[str, object]:
    out: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        out.update({
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        })
    except (ImportError, RuntimeError, OSError) as exc:
        out["cuda_inventory_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _record_matches(
    record: object,
    expected: Path,
    *,
    logical_path: str | None = None,
) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        if (
            not isinstance(record.get("path"), str)
            or not isinstance(record.get("size"), int)
            or isinstance(record.get("size"), bool)
            or record["size"] < 0
            or not isinstance(record.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", record["sha256"])
        ):
            return False
        if logical_path is not None and record["path"] != logical_path:
            return False
        declared = Path(record["path"])
        actual = file_record(expected)
        path_ok = (
            declared.resolve() == expected.resolve()
            or (PROJECT_ROOT / declared).resolve() == expected.resolve()
            or logical_path is not None
        )
        return (
            path_ok
            and record.get("size") == actual["size"]
            and record.get("sha256") == actual["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _probe_score_provenance_status() -> str:
    """Classify the retained GPU score cache from its complete identity chain."""

    score = PROJECT_ROOT / "artifacts/jlens/probe/cv_all648/lens_all648.npz"
    provenance = score.with_suffix(".provenance.json")
    if not score.is_file() or not provenance.is_file() or score.is_symlink() or provenance.is_symlink():
        return "NOT_RETAINED"
    try:
        document = json.loads(provenance.read_text(encoding="utf-8"))
        accepted_statuses = {
            "FRESH_CURRENT_SOURCE_HASH_BOUND_LENS",
            "FRESH_CURRENT_SOURCE_RETAINED_PRE_CUSTODY_LENS",
        }
        if document.get("schema_version") != 2 or document.get("status") not in accepted_statuses:
            return "INVALID_OR_STALE_PROVENANCE"
        if document.get("lens_lineage_status") not in {
            "HASH_BOUND", "RETAINED_PRE_CUSTODY_EXACT_BYTES_ONLY",
        }:
            return "INVALID_OR_STALE_PROVENANCE"
        expected_lens_status = (
            "FRESH_CURRENT_SOURCE_HASH_BOUND_LENS"
            if document.get("lens_lineage_status") == "HASH_BOUND"
            else "FRESH_CURRENT_SOURCE_RETAINED_PRE_CUSTODY_LENS"
        )
        if document.get("status") != expected_lens_status:
            return "INVALID_OR_STALE_PROVENANCE"
        inputs = document.get("inputs")
        expected_inputs = {
            "gpu_cache": (PROJECT_ROOT / "artifacts/jlens/probe/n80_v2/gpu_cache.npz", "gpu_cache"),
            "gpu_cache_provenance": (
                PROJECT_ROOT / "artifacts/jlens/probe/n80_v2/gpu_cache.provenance.json",
                "gpu_cache_provenance",
            ),
            "lens": (PROJECT_ROOT / "artifacts/jlens/lens/exit3/exit3_n80.pt", "lens"),
            "lens_sidecar": (PROJECT_ROOT / "artifacts/jlens/lens/exit3/exit3_n80.json", "lens_sidecar"),
        }
        if not isinstance(inputs, dict) or set(inputs) != set(expected_inputs):
            return "INVALID_OR_STALE_PROVENANCE"
        if not _record_matches(document.get("output"), score, logical_path="lens_scores"):
            return "INVALID_OR_STALE_PROVENANCE"
        if any(not _record_matches(inputs.get(name), expected, logical_path=logical)
               for name, (expected, logical) in expected_inputs.items()):
            return "INVALID_OR_STALE_PROVENANCE"

        cache_provenance_path = expected_inputs["gpu_cache_provenance"][0]
        if not cache_provenance_path.is_file() or cache_provenance_path.is_symlink():
            return "INVALID_OR_STALE_PROVENANCE"
        cache_document = json.loads(cache_provenance_path.read_text(encoding="utf-8"))
        if cache_document.get("schema_version") != 2 or cache_document.get("status") != "FRESH_CURRENT_SOURCE_AND_MODEL":
            return "INVALID_OR_STALE_PROVENANCE"
        if not _record_matches(cache_document.get("output"), expected_inputs["gpu_cache"][0], logical_path="gpu_cache"):
            return "INVALID_OR_STALE_PROVENANCE"
        if cache_document.get("design") != {"prompts": 648, "virtual_locations": 192, "hidden_width": 2048}:
            return "INVALID_OR_STALE_PROVENANCE"
        model_files = cache_document.get("model_files")
        if not isinstance(model_files, list) or not model_files:
            return "INVALID_OR_STALE_PROVENANCE"
        model_flat = {}
        for record in model_files:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"].startswith("model_snapshot/"):
                return "INVALID_OR_STALE_PROVENANCE"
            model_path = OURO_SNAPSHOT / Path(record["path"]).relative_to("model_snapshot")
            if not _record_matches(record, model_path, logical_path=record["path"]):
                return "INVALID_OR_STALE_PROVENANCE"
            model_flat[record["path"]] = record["sha256"]
        try:
            expected_model_paths = {
                f"model_snapshot/{path.relative_to(OURO_SNAPSHOT).as_posix()}"
                for path in model_snapshot_files(OURO_SNAPSHOT)
            }
        except (OSError, ValueError):
            return "INVALID_OR_STALE_PROVENANCE"
        if len(model_flat) != len(model_files) or set(model_flat) != expected_model_paths:
            return "INVALID_OR_STALE_PROVENANCE"
        if (
            not isinstance(cache_document.get("runtime_versions"), dict)
            or set(cache_document["runtime_versions"]) != PROBE_RUNTIME_VERSIONS
            or not all(isinstance(value, str) and value for value in cache_document["runtime_versions"].values())
            or any(value == "NOT_INSTALLED" for value in cache_document["runtime_versions"].values())
        ):
            return "INVALID_OR_STALE_PROVENANCE"

        from ouro_jlens.report import _probe_source_records_valid

        cache_source_names = ("probe_cv.py", "probe.py", "evaluate.py", "evaldata.py", "recurrent.py", "evidence.py")
        score_source_names = (*cache_source_names, "fit_lens.py")
        source_records = cache_document.get("source_files")
        score_source_records = document.get("source_files")
        for records, expected_names, expected_digest in (
            (source_records, cache_source_names, cache_document.get("source_sha256")),
            (score_source_records, score_source_names, document.get("source_sha256")),
        ):
            valid_sources, _, _ = _probe_source_records_valid(
                records, expected_digest, names=expected_names,
            )
            if not valid_sources:
                return "INVALID_OR_STALE_PROVENANCE"
        if (
            document.get("model_files") != model_files
            or not isinstance(document.get("runtime_versions"), dict)
            or document.get("runtime_versions") != cache_document.get("runtime_versions")
        ):
            return "INVALID_OR_STALE_PROVENANCE"
        design_path = PROJECT_ROOT / "artifacts/jlens/probe/cv_all648/design.json"
        if not design_path.is_file() or design_path.is_symlink():
            return "INVALID_OR_STALE_PROVENANCE"
        design = json.loads(design_path.read_text(encoding="utf-8"))
        if (
            design.get("schema_version") != 2
            or design.get("status") != "FRESH_CURRENT_SOURCE_INNER_SELECTED_OUTER_SCORED"
            or not isinstance(design.get("seed"), int)
            or isinstance(design.get("seed"), bool)
            or not isinstance(design.get("design"), dict)
            or design["design"].get("fold_assignment_unit") != "unordered_operand_pair"
            or not isinstance(design["design"].get("fold_sizes"), list)
            or len(design["design"]["fold_sizes"]) != 5
            or sum(design["design"]["fold_sizes"]) != 648
            or not isinstance(design["design"].get("validation_pairs_by_fold"), list)
            or len(design["design"]["validation_pairs_by_fold"]) != 5
            or not isinstance(design["design"].get("selection_ancestry"), str)
            or "outer fold" not in design["design"].get("selection_ancestry", "")
        ):
            return "INVALID_OR_STALE_PROVENANCE"
        design_inputs = design.get("inputs")
        if not isinstance(design_inputs, dict) or set(design_inputs) != {
            "gpu_cache", "gpu_cache_provenance", "lens_scores", "lens_score_provenance",
        }:
            return "INVALID_OR_STALE_PROVENANCE"
        for name, expected, logical in (
            ("gpu_cache", expected_inputs["gpu_cache"][0], "gpu_cache"),
            ("gpu_cache_provenance", cache_provenance_path, "gpu_cache_provenance"),
            ("lens_scores", score, "lens_scores"),
            ("lens_score_provenance", provenance, "lens_score_provenance"),
        ):
            if not _record_matches(design_inputs.get(name), expected, logical_path=logical):
                return "INVALID_OR_STALE_PROVENANCE"
        if not _record_matches(design.get("output"), score.with_name("arrays.npz"), logical_path="probe_arrays"):
            return "INVALID_OR_STALE_PROVENANCE"
        if (
            not isinstance(design.get("runtime_versions"), dict)
            or set(design["runtime_versions"]) != PROBE_RUNTIME_VERSIONS
            or not all(isinstance(value, str) and value for value in design["runtime_versions"].values())
            or any(value == "NOT_INSTALLED" for value in design["runtime_versions"].values())
            or design.get("runtime_versions") != cache_document.get("runtime_versions")
        ):
            return "INVALID_OR_STALE_PROVENANCE"
        design_source_names = ("probe_cv.py", "probe.py", "probe_report.py", "evidence.py")
        design_sources = design.get("source_files")
        valid_design_sources, _, _ = _probe_source_records_valid(
            design_sources, design.get("source_sha256"), names=design_source_names,
        )
        if not valid_design_sources:
            return "INVALID_OR_STALE_PROVENANCE"
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return "INVALID_OR_STALE_PROVENANCE"
    return str(document["status"])


def _inventory_paths(groups: dict[str, list[Path]]) -> dict[str, list[str]]:
    return {
        group: sorted(str(path) for path in paths)
        for group, paths in sorted(groups.items())
    }


def _assert_required_paths() -> None:
    missing = [path for path in REQUIRED_PATHS if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required manifest inputs are missing: {missing}")


def build_manifest() -> dict:
    _assert_required_paths()
    groups = inventory()
    if set(groups) != REQUIRED_GROUPS - {"external_jlens_source"}:
        raise RuntimeError(
            f"manifest inventory groups changed: {sorted(groups)} != {sorted(REQUIRED_GROUPS)}"
        )
    jlens_files, jlens_revision = _jlens_identity()
    if not jlens_files or jlens_revision == "UNKNOWN":
        raise RuntimeError("external jlens source or revision is unavailable")
    groups["external_jlens_source"] = jlens_files
    records: dict[str, list[dict]] = {}
    for group, paths in groups.items():
        records[group] = []
        for path in paths:
            record = file_record(path)
            record["tracked"] = _tracked(path)
            records[group].append(record)
    flat = {record["path"]: record["sha256"] for rows in records.values() for record in rows}
    untracked = {
        group: [record["path"] for record in records[group] if not record["tracked"]]
        for group in TRACKED_GROUPS
    }
    untracked = {group: paths for group, paths in untracked.items() if paths}
    if untracked:
        raise RuntimeError(f"accepted source/tests/docs must be tracked before manifest build: {untracked}")
    current_validation = records["current_validation"]
    latest_validation = PROJECT_ROOT / "artifacts/jlens/validation/milestones.json"
    validation_status = "NOT_ESTABLISHED"
    if latest_validation.is_file():
        try:
            validation_status = _manifest_current_validation_status()
        except (OSError, json.JSONDecodeError):
            validation_status = "INVALID"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "LOCAL_CUSTODY_BOUND_PREEXISTING_EMPIRICAL_ARTIFACTS_UNFROZEN",
        "scientific_effect": "NONE; byte identity is not semantic acceptance",
        "project_head": _git_head(),
        "model_revision": OURO_REVISION,
        "jlens_revision": jlens_revision,
        "environment": _versions(),
        "hardware": _hardware(),
        "required_groups": sorted(REQUIRED_GROUPS),
        "required_paths": list(REQUIRED_PATHS),
        "reproduction_contract": {
            "seeds": {"analysis_bootstrap": 0, "probe_cv": 0, "probe_bootstrap": 991},
            "commands": [
                "PYTHONPATH=src venv/bin/python -m ouro_jlens.verify_artifacts",
                "PYTHONPATH=src venv/bin/python -m ouro_jlens.report --validation artifacts/jlens/validation/milestones.json --transport artifacts/jlens/final/transport.json --probe artifacts/jlens/probe/cv_all648/summary.json",
                "PYTHONPATH=src venv/bin/python -m ouro_jlens.manifest verify",
            ],
        },
        "groups": records,
        "inventory_sha256": sha256_json(_inventory_paths(groups)),
        "aggregate_sha256": sha256_json(flat),
        "current_validation_status": validation_status if current_validation else "NOT_ESTABLISHED",
        "probe_score_provenance": _probe_score_provenance_status(),
    }


def _manifest_record_shape(record: object) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and bool(record["path"])
        and isinstance(record.get("size"), int)
        and not isinstance(record["size"], bool)
        and record["size"] >= 0
        and isinstance(record.get("sha256"), str)
        and len(record["sha256"]) == 64
        and all(character in "0123456789abcdefABCDEF" for character in record["sha256"])
        and isinstance(record.get("tracked"), bool)
    )


def _manifest_current_validation_status() -> str:
    latest = PROJECT_ROOT / "artifacts/jlens/validation/milestones.json"
    if not latest.is_file():
        return "NOT_ESTABLISHED"
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "INVALID"
    if not isinstance(payload, dict):
        return "INVALID"
    # Keep the manifest's status gate identical to the canonical report's
    # evidence validator.  Presence of model/JLens records is insufficient:
    # the helper also rehashes their current bytes and checks every M1-M5 and
    # comparison binding before this manifest can advertise a pass.
    try:
        from ouro_jlens.report import _validation_info

        validated = _validation_info(payload)
    except (AttributeError, ImportError, IndexError, KeyError, OSError, TypeError, ValueError):
        return "FAILED_OR_INCOMPLETE_VALIDATION"
    return (
        "NUMERICAL_AND_PROVENANCE_PASS"
        if validated.get("verified") is True
        else "FAILED_OR_INCOMPLETE_VALIDATION"
    )


def verify_manifest(path: Path) -> dict:
    """Verify a complete current manifest against bytes and live inventory.

    The old verifier accepted a one-record ``{"groups": {"raw": ...}}``
    document.  That is a generic checksum, not a JLens custody manifest, so
    this verifier fails closed on missing contract fields, groups, required
    paths, provenance records, or tracked source/tests/docs.
    """

    mismatches: list[dict[str, object]] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "manifest": {"path": str(path), "error": str(exc)},
            "files_checked": 0,
            "mismatches": [{"error": f"cannot read manifest: {exc}"}],
        }
    if not isinstance(manifest, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "manifest": {"path": str(path)},
            "files_checked": 0,
            "mismatches": [{"error": "manifest is not an object"}],
        }
    missing_top = sorted(REQUIRED_TOP_LEVEL_KEYS - set(manifest))
    if missing_top:
        mismatches.append({"field": "top_level", "error": "missing required fields", "missing": missing_top})
    extra_top = sorted(set(manifest) - REQUIRED_TOP_LEVEL_KEYS)
    if extra_top:
        mismatches.append({"field": "top_level", "error": "unknown fields", "extra": extra_top})
    if manifest.get("schema_version") != SCHEMA_VERSION:
        mismatches.append({"field": "schema_version", "expected": SCHEMA_VERSION, "actual": manifest.get("schema_version")})
    if manifest.get("required_groups") != sorted(REQUIRED_GROUPS):
        mismatches.append({"field": "required_groups", "expected": sorted(REQUIRED_GROUPS), "actual": manifest.get("required_groups")})
    if manifest.get("required_paths") != list(REQUIRED_PATHS):
        mismatches.append({"field": "required_paths", "expected": list(REQUIRED_PATHS), "actual": manifest.get("required_paths")})

    groups = manifest.get("groups")
    if not isinstance(groups, dict):
        mismatches.append({"field": "groups", "error": "groups must be an object"})
        groups = {}
    if set(groups) != REQUIRED_GROUPS:
        mismatches.append({"field": "groups", "expected": sorted(REQUIRED_GROUPS), "actual": sorted(groups)})

    # Recompute the path inventory before looking at declared hashes.  This
    # catches omitted evaluation provenance and current probe chain files even
    # if a caller recomputes aggregate_sha256 over the stripped manifest.
    current_groups: dict[str, list[Path]] = {}
    try:
        current_groups = inventory()
        jlens_files, jlens_revision = _jlens_identity()
        current_groups["external_jlens_source"] = jlens_files
    except (ImportError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        mismatches.append({"field": "inventory", "error": str(exc)})
        jlens_revision = "UNKNOWN"
    if set(current_groups) != REQUIRED_GROUPS:
        mismatches.append({"field": "current_inventory_groups", "expected": sorted(REQUIRED_GROUPS), "actual": sorted(current_groups)})
    current_paths = _inventory_paths(current_groups) if current_groups else {}
    if manifest.get("inventory_sha256") is not None:
        current_inventory = sha256_json(current_paths)
        if current_inventory != manifest.get("inventory_sha256"):
            mismatches.append({"field": "inventory_sha256", "expected": manifest.get("inventory_sha256"), "actual": current_inventory})
    if manifest.get("jlens_revision") != jlens_revision:
        mismatches.append({"field": "jlens_revision", "expected": manifest.get("jlens_revision"), "actual": jlens_revision})

    flat: dict[str, str] = {}
    seen: set[str] = set()
    for group in sorted(REQUIRED_GROUPS):
        declared_records = groups.get(group)
        if not isinstance(declared_records, list):
            mismatches.append({"group": group, "error": "not a list"})
            continue
        declared_paths: list[str] = []
        for index, declared in enumerate(declared_records):
            if not _manifest_record_shape(declared):
                mismatches.append({"group": group, "index": index, "error": "incomplete file record"})
                continue
            if set(declared) != {"path", "size", "sha256", "tracked"}:
                mismatches.append({"group": group, "index": index, "error": "unknown file-record fields"})
            raw_path = str(declared["path"])
            rel = Path(raw_path)
            if raw_path in seen:
                mismatches.append({"path": raw_path, "error": "duplicate manifest path"})
                continue
            seen.add(raw_path)
            declared_paths.append(raw_path)
            if rel.is_absolute() and group != "external_jlens_source":
                mismatches.append({"path": raw_path, "error": "absolute path outside external source group"})
                continue
            if not rel.is_absolute() and ".." in rel.parts:
                mismatches.append({"path": raw_path, "error": "path escapes project root"})
                continue
            candidate = rel if rel.is_absolute() else PROJECT_ROOT / rel
            try:
                raw_actual = file_record(candidate)
                actual = {"path": raw_path, "size": raw_actual["size"], "sha256": raw_actual["sha256"]}
            except (OSError, ValueError) as exc:
                mismatches.append({"path": raw_path, "error": str(exc)})
                continue
            flat[raw_path] = actual["sha256"]
            if actual["size"] != declared["size"] or actual["sha256"] != declared["sha256"]:
                mismatches.append({"path": raw_path, "declared": declared, "actual": actual})
            tracked = _tracked(candidate)
            if tracked != declared["tracked"]:
                mismatches.append({"path": raw_path, "field": "tracked", "expected": declared["tracked"], "actual": tracked})
            if group in TRACKED_GROUPS and (not declared["tracked"] or not tracked):
                mismatches.append({"path": raw_path, "group": group, "error": "source/tests/docs must be tracked"})
        expected_paths = current_paths.get(group, [])
        if sorted(declared_paths) != sorted(expected_paths):
            mismatches.append({"group": group, "field": "inventory_paths", "expected": sorted(expected_paths), "actual": sorted(declared_paths)})

    required_present = {
        str(record.get("path"))
        for rows in groups.values() if isinstance(rows, list)
        for record in rows if isinstance(record, dict)
    }
    missing_required = [required for required in REQUIRED_PATHS if required not in required_present]
    if missing_required:
        mismatches.append({"field": "REQUIRED_PATHS", "error": "required paths are absent from manifest", "missing": missing_required})
    missing_live_required = [required for required in REQUIRED_PATHS if not (PROJECT_ROOT / required).is_file()]
    if missing_live_required:
        mismatches.append({"field": "REQUIRED_PATHS", "error": "required paths are absent from current tree", "missing": missing_live_required})

    try:
        current_head = _git_head()
        if manifest.get("project_head") != current_head:
            mismatches.append({"field": "project_head", "expected": manifest.get("project_head"), "actual": current_head})
    except (OSError, subprocess.CalledProcessError) as exc:
        mismatches.append({"field": "project_head", "error": str(exc)})
    if manifest.get("model_revision") != OURO_REVISION:
        mismatches.append({"field": "model_revision", "expected": OURO_REVISION, "actual": manifest.get("model_revision")})
    current_environment = _versions()
    if manifest.get("environment") != current_environment:
        mismatches.append({"field": "environment", "expected": manifest.get("environment"), "actual": current_environment})
    reproduction = manifest.get("reproduction_contract")
    if not isinstance(reproduction, dict) or set(reproduction) != {"seeds", "commands"} or not isinstance(reproduction.get("seeds"), dict) or set(reproduction["seeds"]) != {"analysis_bootstrap", "probe_cv", "probe_bootstrap"} or not isinstance(reproduction.get("commands"), list) or not reproduction["commands"]:
        mismatches.append({"field": "reproduction_contract", "error": "complete seeds and commands are required"})
    if manifest.get("status") != "LOCAL_CUSTODY_BOUND_PREEXISTING_EMPIRICAL_ARTIFACTS_UNFROZEN":
        mismatches.append({"field": "status", "error": "unexpected custody status", "actual": manifest.get("status")})
    if manifest.get("scientific_effect") != "NONE; byte identity is not semantic acceptance":
        mismatches.append({"field": "scientific_effect", "error": "unexpected scientific effect", "actual": manifest.get("scientific_effect")})
    if manifest.get("current_validation_status") != _manifest_current_validation_status():
        mismatches.append({"field": "current_validation_status", "expected": manifest.get("current_validation_status"), "actual": _manifest_current_validation_status()})
    probe_status = _probe_score_provenance_status()
    if manifest.get("probe_score_provenance") != probe_status:
        mismatches.append({"field": "probe_score_provenance", "expected": manifest.get("probe_score_provenance"), "actual": probe_status})
    aggregate = sha256_json(flat)
    if aggregate != manifest.get("aggregate_sha256"):
        mismatches.append({"field": "aggregate_sha256", "expected": manifest.get("aggregate_sha256"), "actual": aggregate})
    try:
        manifest_record = file_record(path)
    except OSError:
        manifest_record = {"path": str(path)}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not mismatches else "FAIL",
        "manifest": manifest_record,
        "files_checked": len(flat),
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--manifest", default="artifacts/jlens/MANIFEST.json")
    parser.add_argument("--verification-out")
    args = parser.parse_args()
    path = Path(args.manifest)
    if args.command == "build":
        atomic_write_json(path, build_manifest())
        print(json.dumps({"status": "BUILT", "manifest": file_record(path)}, indent=2))
        return 0
    result = verify_manifest(path)
    if args.verification_out:
        atomic_write_json(Path(args.verification_out), result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
