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
import subprocess
from pathlib import Path
from typing import Iterable

from ouro_jlens.evidence import aggregate_sha256, atomic_write_json, file_record, sha256_json
from ouro_jlens.recurrent import OURO_REVISION, OURO_SNAPSHOT, PROJECT_ROOT

SCHEMA_VERSION = 1

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
    raw.extend(_files_under(
        PROJECT_ROOT / "artifacts/jlens/eval", ("arrays.npz", "items.json", "task_names.json")
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
    if validation:
        validation.extend(PROJECT_ROOT / "artifacts/jlens/validation" / name
                          for name in ("milestones.json", "latest.json"))
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
    for distribution in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "jlens"):
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


def _record_matches(record: object, expected: Path) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        declared = Path(record["path"])
        actual = file_record(expected)
        return (
            declared.resolve() == expected.resolve()
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
        inputs = document["inputs"]
        source_records = document["source_files"]
        expected_sources = [
            PROJECT_ROOT / f"src/ouro_jlens/{name}"
            for name in ("probe_cv.py", "probe.py", "evaluate.py", "recurrent.py")
        ]
        if (
            document.get("schema_version") != 1
            or document.get("status") != "FRESH_CURRENT_SOURCE"
            or not _record_matches(document.get("output"), score)
            or not isinstance(inputs, dict)
            or not _record_matches(
                inputs.get("gpu_cache"),
                PROJECT_ROOT / "artifacts/jlens/probe/n80_v2/gpu_cache.npz",
            )
            or not _record_matches(
                inputs.get("lens"),
                PROJECT_ROOT / "artifacts/jlens/lens/exit3/exit3_n80.pt",
            )
            or not isinstance(source_records, list)
            or len(source_records) != len(expected_sources)
            or not all(
                _record_matches(record, expected)
                for record, expected in zip(source_records, expected_sources, strict=True)
            )
            or document.get("source_sha256") != aggregate_sha256(
                {str(record["path"]): str(record["sha256"]) for record in source_records}
            )
        ):
            return "INVALID_OR_STALE_PROVENANCE"
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return "INVALID_OR_STALE_PROVENANCE"
    return "FRESH_CURRENT_SOURCE_AND_MANIFESTED_INPUTS"


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
            latest = json.loads(latest_validation.read_text(encoding="utf-8"))
            validation_status = "PASS" if latest.get("numerical_pass") is True else "FAIL"
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


def verify_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema")
    mismatches = []
    flat: dict[str, str] = {}
    seen: set[str] = set()
    for group, records in manifest.get("groups", {}).items():
        if not isinstance(records, list):
            mismatches.append({"group": group, "error": "not a list"})
            continue
        for declared in records:
            rel = Path(declared["path"])
            if str(rel) in seen:
                mismatches.append({"path": str(rel), "error": "duplicate manifest path"})
                continue
            seen.add(str(rel))
            if rel.is_absolute() and group != "external_jlens_source":
                mismatches.append({"path": str(rel), "error": "absolute path outside external source group"})
                continue
            if not rel.is_absolute() and (rel.is_absolute() or ".." in rel.parts):
                mismatches.append({"path": str(rel), "error": "path escapes project root"})
                continue
            try:
                candidate = rel if rel.is_absolute() else PROJECT_ROOT / rel
                raw_actual = file_record(candidate)
                actual = {"path": str(rel), "size": raw_actual["size"],
                          "sha256": raw_actual["sha256"]}
            except (OSError, ValueError) as exc:
                mismatches.append({"path": str(rel), "error": str(exc)})
                continue
            flat[str(rel)] = actual["sha256"]
            if actual["size"] != declared.get("size") or actual["sha256"] != declared.get("sha256"):
                mismatches.append({"path": str(rel), "declared": declared, "actual": actual})
            if "tracked" in declared and bool(declared["tracked"]) != _tracked(candidate):
                mismatches.append({
                    "path": str(rel), "field": "tracked", "expected": declared["tracked"],
                    "actual": _tracked(candidate),
                })
    if manifest.get("inventory_sha256") is not None:
        try:
            current_groups = inventory()
            jlens_files, jlens_revision = _jlens_identity()
            current_groups["external_jlens_source"] = jlens_files
            current_inventory = sha256_json(_inventory_paths(current_groups))
            if current_inventory != manifest["inventory_sha256"]:
                mismatches.append({
                    "field": "inventory_sha256", "expected": manifest["inventory_sha256"],
                    "actual": current_inventory,
                })
            if jlens_revision != manifest.get("jlens_revision"):
                mismatches.append({
                    "field": "jlens_revision", "expected": manifest.get("jlens_revision"),
                    "actual": jlens_revision,
                })
        except (ImportError, OSError, subprocess.CalledProcessError, ValueError) as exc:
            mismatches.append({"field": "inventory_sha256", "error": str(exc)})
    if manifest.get("project_head") is not None:
        current_head = _git_head()
        if current_head != manifest["project_head"]:
            mismatches.append({
                "field": "project_head", "expected": manifest["project_head"],
                "actual": current_head,
            })
    if manifest.get("environment") is not None and _versions() != manifest["environment"]:
        mismatches.append({
            "field": "environment", "expected": manifest["environment"], "actual": _versions(),
        })
    aggregate = sha256_json(flat)
    if aggregate != manifest.get("aggregate_sha256"):
        mismatches.append({"field": "aggregate_sha256", "expected": manifest.get("aggregate_sha256"),
                           "actual": aggregate})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not mismatches else "FAIL",
        "manifest": file_record(path),
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
