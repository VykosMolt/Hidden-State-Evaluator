"""Fail-closed verifier for the complete paid B300 JLens run.

The local arithmetic verifier intentionally covers only retained arrays and
does not establish custody or current lineage.  This module is the paid-run
gate: it checks the current validation artifact, every evaluation/lens input
used by the canonical report, a direct transport recomputation, and exact
regeneration of both canonical JSON products.  Probe artifacts are outside
this contract and are deliberately not imported or required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ouro_jlens import report as report_module
from ouro_jlens import evaluate as evaluate_module
from ouro_jlens import transport_report, verify_artifacts
from ouro_jlens.evidence import file_record
from ouro_jlens.fit_lens import IntegrityError
from ouro_jlens.publish import (
    _write_immutable,
    canonical_json,
    validate_run_id,
)
from ouro_jlens.recurrent import OURO_REVISION, PROJECT_ROOT
from ouro_jlens.validate import (
    _valid_image_digest,
    approved_b300_image_digest,
    paid_validation_accepted,
)


class PaidVerificationError(ValueError):
    """The paid artifacts do not satisfy the current acceptance contract."""


EXPECTED_MODEL_SHAPE = {
    "n_physical": 48,
    "n_ut": 4,
    "n_layers": 192,
    "d_model": 2048,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaidVerificationError(message)


def _has_link_component(path: Path) -> bool:
    return any(candidate.is_symlink() for candidate in (path, *path.parents))


def _regular(path: Path, label: str) -> Path:
    _require(not _has_link_component(path) and path.is_file(), f"{label} is missing or linked: {path}")
    return path


def _exact_directory(path: Path, expected: Path, label: str) -> Path:
    """Require one lexical directory path with no symlink component."""

    _require(
        path.is_dir() and not _has_link_component(path),
        f"{label} is missing or linked: {path}",
    )
    _require(
        Path(os.path.abspath(path)) == Path(os.path.abspath(expected)),
        f"{label} is not the current run directory",
    )
    return path


def _json(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaidVerificationError(f"{label} is unavailable or invalid: {path}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _image_digest() -> str:
    value = os.environ.get("JLENS_IMAGE_DIGEST")
    _require(
        _valid_image_digest(value),
        "JLENS_IMAGE_DIGEST must be a full immutable image@sha256 reference",
    )
    approved = approved_b300_image_digest()
    _require(
        approved is not None,
        "controller-approved canonical B300 image identity is unavailable",
    )
    _require(
        value == approved,
        "JLENS_IMAGE_DIGEST does not match the controller-approved B300 image",
    )
    return value


def _run_id() -> str:
    value = os.environ.get("RUN_ID")
    _require(isinstance(value, str) and bool(value), "RUN_ID is required for paid verification")
    try:
        return validate_run_id(value)
    except ValueError as exc:
        raise PaidVerificationError("RUN_ID is not a valid immutable artifact lineage") from exc


def _attempt_id() -> str | None:
    """Return the optional lease-attempt binding used by controller replay."""

    value = os.environ.get("JLENS_ATTEMPT_ID")
    if value is None:
        return None
    try:
        return validate_run_id(value)
    except (TypeError, ValueError) as exc:
        raise PaidVerificationError(
            "JLENS_ATTEMPT_ID is not a valid immutable lease attempt"
        ) from exc


# Keep this identical to the controller/remote shell launch contract.  A
# short or shell-significant token must not be able to produce a paid verdict.
_SAFE_NONCE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWER_GIT_HEAD = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _launch_bindings() -> dict[str, str | int]:
    """Require the controller's immutable launch/stage identities."""

    nonce = os.environ.get("JLENS_LAUNCH_NONCE")
    manifest = os.environ.get("EXPECTED_STAGE_MANIFEST_SHA256")
    source_head = os.environ.get("EXPECTED_STAGE_SOURCE_HEAD")
    _require(isinstance(nonce, str) and _SAFE_NONCE.fullmatch(nonce) is not None,
             "JLENS_LAUNCH_NONCE is missing or not a safe token")
    _require(isinstance(manifest, str) and _LOWER_SHA256.fullmatch(manifest) is not None,
             "EXPECTED_STAGE_MANIFEST_SHA256 is missing or not a lowercase SHA-256 digest")
    _require(isinstance(source_head, str) and _LOWER_GIT_HEAD.fullmatch(source_head) is not None,
             "EXPECTED_STAGE_SOURCE_HEAD is missing or not a lowercase 40/64-hex git object ID")
    try:
        worker_deadline = evaluate_module._worker_deadline_epoch()
    except Exception as exc:
        raise PaidVerificationError(str(exc)) from exc
    _require(
        worker_deadline is not None,
        "JLENS_WORKER_DEADLINE_EPOCH is missing",
    )
    return {
        "launch_nonce": nonce,
        "stage_manifest_sha256": manifest,
        "stage_source_head": source_head,
        "worker_deadline_epoch": worker_deadline,
    }


def _validation_gate(path: Path, image_digest: str) -> dict[str, Any]:
    payload = _json(path, "validation report")
    info = report_module._validation_info(payload)
    _require(info.get("verified") is True, "current validation artifact is not verified")
    _require(paid_validation_accepted(payload), "paid validation requires bit-exact equal/close comparisons")
    provenance = payload.get("provenance")
    _require(isinstance(provenance, dict), "validation provenance is missing")
    _require(provenance.get("image_digest") == image_digest, "validation image identity differs from controller identity")
    _require(
        {
            key: payload.get("model", {}).get(key)
            for key in EXPECTED_MODEL_SHAPE
        } == EXPECTED_MODEL_SHAPE
        and {
            key: provenance.get("model_shape", {}).get(key)
            for key in EXPECTED_MODEL_SHAPE
        } == EXPECTED_MODEL_SHAPE,
        "validation model shape is not the current Ouro contract",
    )
    cuda = provenance.get("cuda")
    _require(isinstance(cuda, dict) and cuda.get("available") is True, "paid validation did not run with CUDA available")
    devices = cuda.get("devices")
    _require(
        isinstance(devices, list)
        and devices
        and cuda.get("device_count") == len(devices)
        and all(
            isinstance(device, dict)
            and isinstance(device.get("name"), str)
            and bool(re.search(r"\bB300\b", device["name"], re.IGNORECASE))
            for device in devices
        ),
        "validation device records do not establish an exclusively B300 run",
    )
    return {
        "status": "SUPPORTED_CURRENT_B300_VALIDATION",
        "image_digest": image_digest,
        "devices": devices,
        "validation_status": info.get("status"),
        "bit_exact": payload.get("bit_exact"),
    }


def _lens_paths(lens_root: Path, *, artifact_root: Path | None = None) -> dict[int, Path]:
    prefixes = {n: lens_root / f"exit3_n{n}.pt" for n in (8, 32, 56, 80)}
    for n, path in prefixes.items():
        _regular(path, f"prefix n={n} lens")
        _regular(path.with_suffix(".json"), f"prefix n={n} sidecar")
    # The canonical evaluations consume every target-ut lens, not only the
    # eventual-exit prefixes used for the transport fit.  Resolve all four
    # merged lens/sidecar pairs before accepting evaluation provenance so a
    # valid evaluation cannot hide a missing or replaced local lens.
    for target in range(4):
        path = lens_root / f"exit{target}.pt"
        _regular(path, f"target-ut={target} lens")
        _regular(path.with_suffix(".json"), f"target-ut={target} sidecar")
        try:
            metadata = evaluate_module.validate_lens_sidecar(
                path,
                kind="merged",
                artifact_root=artifact_root,
            )
            raw_shards = metadata.get("shards")
            _require(isinstance(raw_shards, list) and raw_shards, f"target-ut={target} shard seal is missing")
            shard_paths: list[Path] = []
            for raw in raw_shards:
                _require(isinstance(raw, str) and raw, f"target-ut={target} shard seal path is malformed")
                candidate = Path(raw)
                if not candidate.is_absolute():
                    # Project-root sidecars use ``artifacts/jlens/...``;
                    # explicit test bundles commonly use a path relative to
                    # their validation root.  Resolve both forms, but never
                    # allow a valid external shard to satisfy this run.
                    candidates = (
                        PROJECT_ROOT / candidate,
                        lens_root.parent.parent / candidate,
                        lens_root.parent / candidate,
                        path.parent / candidate,
                        Path.cwd() / candidate,
                    )
                    candidate = next((value for value in candidates if value.is_file()), candidates[0])
                candidate_absolute = Path(os.path.abspath(candidate))
                lens_absolute = Path(os.path.abspath(lens_root))
                _require(
                    candidate_absolute.parent == lens_absolute
                    and candidate_absolute.name.startswith(f"exit{target}_shard_")
                    and candidate_absolute.suffix == ".pt",
                    f"target-ut={target} shard seal escapes the current lens root",
                )
                _regular(candidate_absolute, f"target-ut={target} sealed shard")
                shard_paths.append(candidate_absolute)
            evaluate_module.validate_merged_lens_shards(
                path,
                shard_paths,
                artifact_root=artifact_root,
            )
        except (IntegrityError, OSError, ValueError, TypeError, KeyError) as exc:
            raise PaidVerificationError(f"target-ut={target} merged lens is not sealed to current shards: {path}") from exc
    return prefixes


def _validate_prefix_lenses(
    lens_root: Path,
    *,
    artifact_root: Path | None = None,
) -> tuple[list[tuple[int, Path]], list[dict[str, Any]]]:
    prefixes = _lens_paths(lens_root, artifact_root=artifact_root)
    shard_paths = [
        lens_root / "exit3_shard_0000_0008.pt",
        lens_root / "exit3_shard_0008_0032.pt",
        lens_root / "exit3_shard_0032_0056.pt",
        lens_root / "exit3_shard_0056_0080.pt",
    ]
    for shard in shard_paths:
        _regular(shard, "eventual-exit shard")
        _regular(shard.with_suffix(".json"), "eventual-exit shard sidecar")
        try:
            evaluate_module.validate_lens_sidecar(
                shard,
                kind="fit",
                artifact_root=artifact_root,
            )
        except (IntegrityError, OSError, ValueError, TypeError, KeyError) as exc:
            raise PaidVerificationError(f"eventual-exit shard is not current: {shard}") from exc
    records: list[dict[str, Any]] = []
    specs: list[tuple[int, Path]] = []
    for index, n in enumerate((8, 32, 56, 80)):
        path = prefixes[n]
        try:
            metadata = evaluate_module.validate_lens_sidecar(
                path,
                kind="merged",
                artifact_root=artifact_root,
            )
            evaluate_module.validate_merged_lens_shards(
                path,
                shard_paths[: index + 1],
                artifact_root=artifact_root,
            )
        except (IntegrityError, OSError, ValueError, TypeError, KeyError) as exc:
            raise PaidVerificationError(f"prefix n={n} lens is not sealed to current shards: {path}") from exc
        _require(
            metadata.get("n_prompts") == n
            and metadata.get("n_requested") == n
            and metadata.get("n_fitted") == n
            and metadata.get("start") == 0
            and metadata.get("end") == n,
            f"prefix n={n} sidecar fit range is not exact",
        )
        specs.append((n, path))
        records.append({
            "n_prompts": n,
            "binary": file_record(path) | {"path": f"lens_n{n}"},
            "sidecar": file_record(path.with_suffix(".json")) | {"path": f"lens_n{n}_sidecar"},
        })
    return specs, records


def _evaluation_counts(provenance: dict[str, Any], expected: int) -> None:
    entries = provenance.get("lens_inputs")
    _require(isinstance(entries, list) and entries, "evaluation lens lineage is missing")
    counts: list[int] = []
    for entry in entries:
        identity = entry.get("identity") if isinstance(entry, dict) else None
        count = identity.get("n_prompts") if isinstance(identity, dict) else None
        _require(type(count) is int and count == expected, "evaluation fit count does not match its exact target")
        counts.append(count)
    _require(counts and all(count == expected for count in counts), "evaluation lens fit counts are inconsistent")


def _validate_evaluation(
    eval_dir: Path,
    *,
    expected_lenses: list[tuple[int, Path]],
    expected_count: int,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    _require(eval_dir.is_dir() and not eval_dir.is_symlink(), f"evaluation directory is missing or linked: {eval_dir}")
    for path in eval_dir.rglob("*"):
        _require(not path.is_symlink(), f"evaluation tree contains a link: {path}")
    try:
        from ouro_jlens.evaluate import validate_evaluation_provenance

        provenance = validate_evaluation_provenance(
            eval_dir,
            artifact_root=artifact_root,
            lens_paths=expected_lenses,
            expected_prompt_policy="identical",
        )
    except Exception as exc:
        raise PaidVerificationError(f"evaluation lineage is not current: {eval_dir}") from exc
    _require(isinstance(provenance, dict), f"evaluation provenance is malformed: {eval_dir}")
    _evaluation_counts(provenance, expected_count)
    model = provenance.get("model")
    _require(
        isinstance(model, dict)
        and model.get("revision") == OURO_REVISION
        and {key: model.get(key) for key in EXPECTED_MODEL_SHAPE} == EXPECTED_MODEL_SHAPE,
        f"evaluation model identity is not the current Ouro shape: {eval_dir}",
    )
    return provenance


def _validate_evaluation_set(
    eval_root: Path,
    lens_root: Path,
    n: int,
    *,
    artifact_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    prefixes = {size: lens_root / f"exit3_n{size}.pt" for size in (8, 32, 56, 80)}
    expected: dict[str, tuple[Path, list[tuple[int, Path]], int]] = {
        **{
            f"fitsize_n{size}": (eval_root / f"fitsize_n{size}", [(3, prefixes[size])], size)
            for size in (8, 32, 56, 80)
        },
        f"n{n}_exit3": (eval_root / f"n{n}_exit3", [(3, lens_root / "exit3.pt")], n),
        f"n{n}_allexits": (
            eval_root / f"n{n}_allexits",
            [(ut, lens_root / f"exit{ut}.pt") for ut in (3, 2, 1, 0)],
            n,
        ),
        f"n{n}_allexits_pos-2": (
            eval_root / f"n{n}_allexits_pos-2",
            [(ut, lens_root / f"exit{ut}.pt") for ut in (3, 2, 1, 0)],
            n,
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for tag, (directory, lenses, count) in expected.items():
        result[tag] = _validate_evaluation(
            directory,
            expected_lenses=lenses,
            expected_count=count,
            artifact_root=artifact_root,
        )
    return result


def _finite_list(value: object, length: int, label: str) -> None:
    _require(
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) and np.isfinite(item) for item in value),
        f"{label} has the wrong shape or non-finite values",
    )


def _validate_analysis_shape(analysis: dict[str, Any], claims: dict[str, Any]) -> None:
    _require(analysis.get("schema_version") == report_module.SCHEMA_VERSION, "canonical analysis schema changed")
    _require(analysis.get("overall_status") == report_module.CURRENT_GENERATION, "canonical analysis is not current-generation")
    main = analysis.get("main")
    _require(isinstance(main, dict), "canonical analysis main section is missing")
    _require(main.get("fit_prompts") == 80 and main.get("fit_prompts_status") == "VERIFIED_FROM_PROVENANCE", "canonical main fit n=80 is not authenticated")
    populations = main.get("populations")
    _require(isinstance(populations, dict) and set(populations) == {"multihop", "order_ops_numeric"}, "canonical population set is incomplete")
    for public, population in populations.items():
        _require(isinstance(population, dict), f"canonical population is malformed: {public}")
        readouts = population.get("readouts")
        _require(isinstance(readouts, dict) and {"eventual_exit_jacobian_lens", "logit_lens"} <= set(readouts), f"canonical readouts are incomplete: {public}")
        for readout_name, readout in readouts.items():
            _require(isinstance(readout, dict), f"canonical readout is malformed: {public}/{readout_name}")
            for metric, values in readout.items():
                _finite_list(values, 4, f"canonical readout {public}/{readout_name}/{metric}")
        paired = population.get("paired_j_minus_logit")
        _require(isinstance(paired, dict), f"canonical paired statistic is missing: {public}")
        _finite_list(paired.get("excess_pass10_per_loop"), 4, f"canonical effects {public}")
        intervals = paired.get("ci95_unadjusted")
        _require(
            isinstance(intervals, list)
            and len(intervals) == 4
            and all(isinstance(interval, list) and len(interval) == 2 for interval in intervals),
            f"canonical confidence intervals are malformed: {public}",
        )
        for interval in intervals:
            _finite_list(interval, 2, f"canonical confidence interval {public}")
    accuracy = main.get("corrected_model_accuracy")
    _require(isinstance(accuracy, dict) and type(accuracy.get("n_items")) is int and accuracy["n_items"] == 148, "canonical corrected accuracy shape is invalid")
    exits = main.get("lens_free_exits")
    _require(isinstance(exits, dict) and {"multihop", "order-ops numeric"} <= set(exits), "canonical exit report is incomplete")
    for name in ("multihop", "order-ops numeric"):
        row = exits[name]
        _require(isinstance(row, dict), f"canonical exit row is malformed: {name}")
        _finite_list(row.get("exit_top1_equals_final_top1_all_items"), 4, f"canonical exit agreement {name}")
    local = analysis.get("local_vs_eventual")
    _require(isinstance(local, dict) and isinstance(local.get("rows"), list) and len(local["rows"]) == 3, "canonical local/eventual shape is invalid")
    for row in local["rows"]:
        _require(isinstance(row, dict) and type(row.get("loop")) is int, "canonical local/eventual row is malformed")
    transport = analysis.get("transport")
    _require(isinstance(transport, dict), "canonical transport section is missing")
    _require(transport.get("sample_sizes") == [8, 32, 56, 80], "canonical transport fit-size set is not exact")
    _require(isinstance(analysis.get("validation"), dict), "canonical validation section is missing")
    _require(isinstance(claims, dict) and set(claims) == set(report_module.CLAIMS), "canonical claim set changed")
    for name in ("multihop_relative_deficit", "arithmetic_relative_deficit"):
        claim = claims[name]
        _require(isinstance(claim, dict), f"canonical claim is malformed: {name}")
        decisions = claim.get("loop_decisions")
        _require(
            isinstance(decisions, list)
            and len(decisions) == 4
            and all(value in {"below", "above", "inconclusive"} for value in decisions),
            f"canonical loop decisions are malformed: {name}",
        )
    _require(claims.get("instrumentation", {}).get("status") == "SUPPORTED_CURRENT_VALIDATION", "instrumentation claim is not current")
    _require(claims.get("b300_validation", {}).get("status") == "SUPPORTED_CURRENT_B300_VALIDATION", "B300 claim is not current")


def _verify_transport_report(
    transport_path: Path,
    specs: list[tuple[int, Path]],
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    transport = _json(transport_path, "transport report")
    _require(transport.get("sample_sizes") == [8, 32, 56, 80], "transport report does not contain exact required fit sizes")
    try:
        if artifact_root is None:
            sizes, norms, records = transport_report.lens_squared_norms(specs)
        else:
            with evaluate_module.relocated_lens_validation(artifact_root):
                sizes, norms, records = transport_report.lens_squared_norms(specs)
        expected = transport_report.summarize(sizes, norms)
    except Exception as exc:
        raise PaidVerificationError("transport direct recomputation failed") from exc
    expected.update({"schema_version": 1, "status": "MODELED_ASSOCIATION_ONLY", "inputs": records})
    _require(transport == expected, "transport report does not exactly match direct lens recomputation")
    return {
        "sample_sizes": sizes.tolist(),
        "n_sources": int(norms.shape[1]),
        "status": transport.get("status"),
    }


def _verify_fit_size_report(
    path: Path,
    eval_root: Path,
    *,
    artifact_root: Path | None = None,
) -> None:
    payload = _json(path, "fit-size report")
    try:
        from ouro_jlens import fitsize_report

        if artifact_root is None:
            expected = fitsize_report.build_report(eval_root, [8, 32, 56, 80])
        else:
            with evaluate_module.relocated_evaluation_validation(artifact_root):
                expected = fitsize_report.build_report(eval_root, [8, 32, 56, 80])
    except Exception as exc:
        raise PaidVerificationError("fit-size report could not be regenerated") from exc
    _require(payload == expected, "fit-size report does not exactly match current evaluations")
    marker = path.with_name("fit_size.status.json")
    marker_payload = _json(marker, "fit-size completion marker")
    _require(marker_payload.get("status") == "COMPLETE_CURRENT_PROVENANCE", "fit-size completion status is not current")


def verify_paid(
    *,
    main: Path,
    local_eval: Path,
    eval_root: Path,
    lens_root: Path,
    validation: Path,
    transport: Path,
    analysis: Path,
    claims: Path,
    fit_size: Path | None = None,
    checkpoints: Path | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Run all paid acceptance checks and return an immutable verdict payload."""

    run_id = _run_id()
    attempt_id = _attempt_id()
    bindings = _launch_bindings()
    image_digest = _image_digest()
    approved_image_digest = approved_b300_image_digest()
    normalized_artifact_root = (
        evaluate_module._artifact_root(artifact_root)
        if artifact_root is not None
        else None
    )
    validation_result = _validation_gate(validation, image_digest)
    _require(eval_root.is_dir() and not _has_link_component(eval_root), "evaluation root is missing or linked")
    _require(lens_root.is_dir() and not _has_link_component(lens_root), "lens root is missing or linked")
    match = re.fullmatch(r"n([1-9][0-9]*)", lens_root.name)
    _require(match is not None, f"lens root does not bind a positive run size: {lens_root}")
    n = int(match.group(1))
    _require(n >= 80, "paid B300 run requires N >= 80")
    prefix_specs, prefix_records = _validate_prefix_lenses(
        lens_root,
        artifact_root=normalized_artifact_root,
    )
    evaluation_lineage = _validate_evaluation_set(
        eval_root,
        lens_root,
        n,
        artifact_root=normalized_artifact_root,
    )
    expected_local = eval_root / f"n{n}_allexits"
    _exact_directory(local_eval, expected_local, "local evaluation path")
    _exact_directory(main, eval_root / "fitsize_n80", "main evaluation path")
    analysis_payload = _json(analysis, "canonical analysis")
    claims_payload = _json(claims, "canonical claims")
    _validate_analysis_shape(analysis_payload, claims_payload)
    try:
        observed_main = verify_artifacts.verify_main(main, analysis_payload)
    except Exception as exc:
        raise PaidVerificationError("main n=80 arrays do not match canonical analysis") from exc
    _verify_transport_report(
        transport,
        prefix_specs,
        artifact_root=normalized_artifact_root,
    )
    fit_size_path = fit_size or Path(transport).with_name("fit_size.json")
    _verify_fit_size_report(
        fit_size_path,
        eval_root,
        artifact_root=normalized_artifact_root,
    )
    try:
        regenerated = report_module.build_report(
            main,
            local_eval,
            validation=validation,
            fit_size=fit_size_path,
            transport=transport,
            checkpoints=checkpoints,
            artifact_root=normalized_artifact_root,
        )
    except Exception as exc:
        raise PaidVerificationError("canonical report could not be regenerated from current inputs") from exc
    _require(regenerated == analysis_payload, "canonical analysis is not an exact regeneration")
    _require(regenerated.get("claims") == claims_payload, "canonical claims are not an exact regeneration")
    main_lineage = evaluation_lineage.get("fitsize_n80", {})
    _require(isinstance(main_lineage, dict), "n=80 evaluation lineage is missing")
    result = {
        "schema_version": 1,
        "run_id": run_id,
        **bindings,
        "status": "PASS",
        "accepted": True,
        "scope": "paid B300 JLens validation, current evaluation/lens lineage, canonical regeneration, and direct transport recomputation",
        "image_digest": image_digest,
        "controller_approved_image_digest": approved_image_digest,
        "validation_status": validation_result["status"],
        "b300_validation_status": validation_result["status"],
        "canonical_report_status": analysis_payload.get("overall_status"),
        "evaluation_lineage_status": report_module.CURRENT_GENERATION,
        "main": observed_main,
        "evaluations": {
            tag: {
                "fit_prompts": provenance.get("lens_inputs", [{}])[0].get("identity", {}).get("n_prompts"),
                "status": report_module.CURRENT_GENERATION,
                "provenance": report_module._canonical_artifact_record(
                    eval_root / tag / "provenance.json",
                    normalized_artifact_root,
                ),
            }
            for tag, provenance in sorted(evaluation_lineage.items())
        },
        "lens_lineage": {"prefixes": prefix_records},
        "transport": {
            "status": "MODELED_ASSOCIATION_ONLY",
            "sample_sizes": [8, 32, 56, 80],
        },
        "required_reports": {
            "analysis": report_module._canonical_artifact_record(
                analysis,
                normalized_artifact_root,
            ),
            "claims": report_module._canonical_artifact_record(
                claims,
                normalized_artifact_root,
            ),
            "fit_size": report_module._canonical_artifact_record(
                fit_size_path,
                normalized_artifact_root,
            ),
            "transport": report_module._canonical_artifact_record(
                transport,
                normalized_artifact_root,
            ),
            "validation": report_module._canonical_artifact_record(
                validation,
                normalized_artifact_root,
            ),
        },
    }
    if attempt_id is not None:
        result["attempt_id"] = attempt_id
    return result


def _invalidate_output(path: Path) -> None:
    """Invalidate a prior verdict before any new paid verification attempt."""

    verify_artifacts._invalidate_output(Path(path))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", default="artifacts/jlens/eval/fitsize_n80")
    parser.add_argument("--local-eval", default="artifacts/jlens/eval/n100_allexits")
    parser.add_argument("--eval-root", default="artifacts/jlens/eval")
    parser.add_argument("--lens-root", default="artifacts/jlens/lens/n100")
    parser.add_argument("--validation", default="artifacts/jlens/validation/milestones.json")
    parser.add_argument("--transport", default="artifacts/jlens/final/transport.json")
    parser.add_argument("--analysis", default="artifacts/jlens/final/analysis.json")
    parser.add_argument("--claims", "--claims-out", dest="claims", default="artifacts/jlens/final/claim_status.json")
    parser.add_argument("--fit-size", default="artifacts/jlens/final/fit_size.json")
    parser.add_argument("--checkpoints", default="artifacts/jlens/checkpoints/exit_divergence.json")
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--out", default="artifacts/jlens/final/verification.json")
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = Path(args.out)
    _invalidate_output(output)
    try:
        result = verify_paid(
            main=Path(args.main),
            local_eval=Path(args.local_eval),
            eval_root=Path(args.eval_root),
            lens_root=Path(args.lens_root),
            validation=Path(args.validation),
            transport=Path(args.transport),
            analysis=Path(args.analysis),
            claims=Path(args.claims),
            fit_size=Path(args.fit_size),
            checkpoints=Path(args.checkpoints) if args.checkpoints else None,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
        _write_immutable(output, canonical_json(result))
    except Exception as exc:
        print(f"PAID VERIFICATION FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
