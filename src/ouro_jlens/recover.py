"""Offline post-termination extraction for one completed JLens lease.

This module deliberately has no RunPod dependency beyond importing the
controller's validation helpers.  Recovery authenticates one durable lease
state, uses the explicitly recorded Hugging Face token only for stage/result
reads, and asks :func:`ouro_jlens.pod._sync_terminated_run` to perform the
receipt, inventory, paid-gate, and staged replay checks.

The state contract is intentionally nested.  The live lease state contains
many operational fields that may grow over time, while ``extraction_contract``
is the small, exact set of values that is allowed to authorize a no-cost
recovery.  Unknown fields in that mapping are rejected so a recovery cannot
silently begin trusting a new, unreviewed binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ouro_jlens import pod
from ouro_jlens.publish import HfPublisher, PublishError, _read_regular_file, canonical_json, safe_relative


class RecoveryError(pod.SafetyError):
    """The durable recovery contract is incomplete or does not verify."""


# The contract is deliberately exact.  ``provider_terminate_after`` and
# ``worker_deadline_epoch`` are top-level lease fields because the controller
# computes them at deployment time; every other extraction binding belongs in
# this mapping.
EXTRACTION_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "attempt_id",
        "artifact_run_id",
        "launch_nonce",
        "image",
        "stage_manifest_sha256",
        "stage_source_head",
        "staging_repository",
        "stage_path",
        "results_repository",
        "n_prompts",
        "hf_token_path",
        "hf_token_sha256",
        "sync_root",
        "receipt_root",
        "results_preflight",
    }
)
RESULTS_PREFLIGHT_KEYS = frozenset({"schema_version", "remote_path", "size", "sha256"})
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_TOKEN_BYTES = 4096


def _fail(message: str) -> None:
    raise RecoveryError(message)


def _read_fd_bounded(fd: int, *, limit: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, limit - total + 1))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            _fail(f"{label} exceeds the bounded size limit")
        chunks.append(chunk)


def _open_exact_regular(path: Path, *, label: str, mode: int | None = None) -> int:
    """Open one no-follow regular inode after checking its path ancestry."""

    try:
        pod._reject_state_symlinks(path)
    except pod.SafetyError as exc:
        raise RecoveryError(f"{label} path is symlinked") from exc
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RecoveryError(f"{label} is unavailable") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            _fail(f"{label} is not one regular unlinked file")
        if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
            _fail(f"{label} permissions are not exact")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _explicit_state_path(args: argparse.Namespace) -> Path:
    """Resolve exactly one caller-supplied state path; never choose a default."""

    candidates: list[Path] = []
    for name in ("state_file", "state"):
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is None:
            continue
        if not isinstance(value, (str, Path)):
            _fail("recovery state path is malformed")
        candidates.append(Path(value).expanduser())
    if len(candidates) != 1:
        _fail("recovery requires exactly one explicit lease state path")
    path = candidates[0]
    if path.is_symlink():
        _fail("recovery state path is a symlink")
    # A relative CLI path is still explicit.  Resolve it once before opening
    # and then use that exact absolute path for all checks; the contract's
    # output roots remain absolute so they cannot be reinterpreted by a later
    # working-directory change.
    return Path(os.path.abspath(path))


def _read_state(path: Path) -> dict[str, Any]:
    # The state contract requires a regular, non-symlink lease document.  The
    # token is the secret that has the strict 0600 requirement; do not add a
    # filesystem-mode dependency to older already-terminated state files.
    fd = _open_exact_regular(path, label="lease state")
    try:
        raw = _read_fd_bounded(fd, limit=MAX_STATE_BYTES, label="lease state")
    finally:
        os.close(fd)
    try:
        state = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RecoveryError("lease state is malformed JSON") from exc
    if not isinstance(state, dict):
        _fail("lease state is not an object")
    return state


def _absolute_contract_path(value: Any, label: str, *, directory: bool = False) -> Path:
    if (not isinstance(value, str) or not value or "\x00" in value
            or "\r" in value or "\n" in value):
        _fail(f"{label} is malformed")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        _fail(f"{label} must be an absolute normalized path")
    try:
        pod._reject_state_symlinks(path / "placeholder")
    except pod.SafetyError as exc:
        raise RecoveryError(f"{label} has a symlinked ancestor") from exc
    if path.is_symlink():
        _fail(f"{label} is a symlink")
    if path.exists() and (
        (directory and not path.is_dir()) or (not directory and not path.is_file())
    ):
        _fail(f"{label} has the wrong filesystem type")
    return path


def _validate_state_and_contract(state: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if state.get("schema_version") != 2:
        _fail("lease state schema is not the supported version")
    if state.get("status") != "terminated":
        _fail("recovery requires a terminated lease state")
    if state.get("termination_verified") is not True:
        _fail("recovery requires verified termination")
    if state.get("deployment_started") is not True:
        _fail("recovery requires a deployed lease")

    try:
        attempt_id = pod.validate_run_id(state["attempt_id"])
        if pod.validate_run_id(state["run_id"]) != attempt_id:
            _fail("lease run and attempt identities disagree")
        if pod.validate_run_id(state["lease_attempt_id"]) != attempt_id:
            _fail("lease attempt alias disagrees with the attempt")
        artifact_run_id = pod.validate_run_id(state["artifact_run_id"])
        if pod.validate_run_id(state["artifact_lineage_id"]) != artifact_run_id:
            _fail("artifact lineage alias disagrees with the artifact run")
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoveryError("lease identity is malformed") from exc
    if state.get("pod_name") != pod._name_for(attempt_id):
        _fail("lease pod name is not bound to the attempt")

    contract = state.get("extraction_contract")
    if not isinstance(contract, Mapping) or set(contract) != EXTRACTION_CONTRACT_KEYS:
        _fail("extraction contract fields are not exact")
    contract = dict(contract)
    if contract.get("schema_version") != 1:
        _fail("extraction contract schema is not supported")
    try:
        if pod.validate_run_id(contract.get("attempt_id")) != attempt_id:
            _fail("extraction attempt is not bound to lease attempt")
        if pod.validate_run_id(contract.get("artifact_run_id")) != artifact_run_id:
            _fail("extraction artifact is not bound to lease lineage")
        nonce = pod._validate_launch_nonce(contract.get("launch_nonce"))
        image = pod._validate_paid_image(contract.get("image"))
        stage_manifest = pod._validate_stage_manifest_digest(contract.get("stage_manifest_sha256"))
        stage_source = pod._validate_source_head(contract.get("stage_source_head"))
        staging_repository = pod._validate_hf_repo(contract.get("staging_repository"), "staging repository")
        results_repository = pod._validate_hf_repo(contract.get("results_repository"), "results repository")
    except (KeyError, TypeError, ValueError, pod.SafetyError) as exc:
        if isinstance(exc, RecoveryError):
            raise
        raise RecoveryError("extraction contract identity is malformed") from exc
    if staging_repository == results_repository:
        _fail("staging and results repositories must be different")
    try:
        stage_path = pod._validate_stage_path(contract.get("stage_path"))
    except (TypeError, ValueError, pod.SafetyError) as exc:
        raise RecoveryError("extraction stage path is malformed") from exc
    n_prompts = contract.get("n_prompts")
    try:
        n_prompts = pod._positive_int(n_prompts, "n_prompts")
    except pod.SafetyError as exc:
        raise RecoveryError("extraction prompt count is malformed") from exc
    if n_prompts < 80:
        _fail("paid recovery requires at least 80 prompts")
    token_path = _absolute_contract_path(contract.get("hf_token_path"), "HF token path")
    token_fingerprint = contract.get("hf_token_sha256")
    if not isinstance(token_fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", token_fingerprint) is None:
        _fail("HF token fingerprint is malformed")
    sync_root = _absolute_contract_path(contract.get("sync_root"), "sync root", directory=True)
    receipt_root = _absolute_contract_path(contract.get("receipt_root"), "receipt root", directory=True)
    if sync_root == receipt_root or sync_root in receipt_root.parents or receipt_root in sync_root.parents:
        _fail("sync and receipt roots must not overlap")

    preflight = contract.get("results_preflight")
    if not isinstance(preflight, Mapping) or set(preflight) != RESULTS_PREFLIGHT_KEYS:
        _fail("results preflight fields are not exact")
    preflight = dict(preflight)
    expected_preflight, _ = _preflight_identity(
        repository=results_repository,
        attempt_id=attempt_id,
        artifact_run_id=artifact_run_id,
        launch_nonce=nonce,
        stage_manifest_sha256=stage_manifest,
        stage_source_head=stage_source,
        hf_token_sha256=token_fingerprint,
    )
    if preflight != expected_preflight:
        _fail("results preflight is not bound to the extraction contract")

    try:
        top_nonce = pod._validate_launch_nonce(state["launch_nonce"])
        top_image = pod._validate_paid_image(state["image"])
        top_manifest = pod._validate_stage_manifest_digest(state["stage_manifest_sha256"])
        top_source = pod._validate_source_head(state["stage_source_head"])
    except (KeyError, TypeError, ValueError, pod.SafetyError) as exc:
        raise RecoveryError("top-level lease stage bindings are malformed") from exc
    if (top_nonce != nonce or top_image != image or top_manifest != stage_manifest
            or top_source != stage_source):
        _fail("top-level lease bindings disagree with extraction contract")
    if "hf_token_sha256" in state and state["hf_token_sha256"] != token_fingerprint:
        _fail("top-level HF token fingerprint disagrees with extraction contract")
    if "results_preflight" in state and state["results_preflight"] != preflight:
        _fail("top-level results preflight disagrees with extraction contract")

    provider_deadline = state.get("provider_terminate_after")
    if (not isinstance(provider_deadline, str)
            or pod.PROVIDER_DATETIME_RE.fullmatch(provider_deadline) is None):
        _fail("provider termination deadline is missing or malformed")
    worker_deadline = state.get("worker_deadline_epoch")
    if isinstance(worker_deadline, bool) or not isinstance(worker_deadline, int) or worker_deadline <= 0:
        _fail("worker deadline is missing or malformed")
    try:
        provider_epoch = int(datetime.strptime(
            provider_deadline, "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc).timestamp())
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecoveryError("provider termination deadline is malformed") from exc
    expected_worker = provider_epoch - pod.WORKER_DEADLINE_MARGIN_SECONDS
    if worker_deadline != expected_worker:
        _fail("worker deadline is not bound to provider termination")

    pod_id = state.get("pod_id")
    machine_id = state.get("machine_id")
    if (not isinstance(pod_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", pod_id) is None
            or not isinstance(machine_id, str) or re.fullmatch(r"[A-Za-z0-9_.:-]+", machine_id) is None):
        _fail("provider lease identity is malformed")
    if state.get("gpu") != pod.DEFAULT_GPU or state.get("provider_gpu_display_name") != pod.DEFAULT_GPU:
        _fail("provider lease is not bound to the exact B300 identity")
    provider_lease = {
        "requested_gpu": pod.DEFAULT_GPU,
        "pod_id": pod_id,
        "machine_id": machine_id,
        "pod_name": state["pod_name"],
        "image": image,
        "gpu_display_name": pod.DEFAULT_GPU,
        "provider_terminate_after": provider_deadline,
    }
    normalized = {
        "attempt_id": attempt_id,
        "artifact_run_id": artifact_run_id,
        "launch_nonce": nonce,
        "image": image,
        "stage_manifest_sha256": stage_manifest,
        "stage_source_head": stage_source,
        "staging_repository": staging_repository,
        "stage_path": stage_path,
        "results_repository": results_repository,
        "n_prompts": n_prompts,
        "hf_token_path": token_path,
        "hf_token_sha256": token_fingerprint,
        "sync_root": sync_root,
        "receipt_root": receipt_root,
        "results_preflight": preflight,
        "worker_deadline_epoch": worker_deadline,
        "provider_lease": provider_lease,
    }
    return normalized, dict(state)


def _preflight_identity(
    *,
    repository: str,
    attempt_id: str,
    artifact_run_id: str,
    launch_nonce: str,
    stage_manifest_sha256: str,
    stage_source_head: str,
    hf_token_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    payload = {
        "schema_version": 1,
        "purpose": "jlens-results-authority-preflight",
        "repository": repository,
        "attempt_id": attempt_id,
        "artifact_run_id": artifact_run_id,
        "launch_nonce": launch_nonce,
        "stage_manifest_sha256": stage_manifest_sha256,
        "stage_source_head": stage_source_head,
        "hf_token_sha256": hf_token_sha256,
    }
    data = canonical_json(payload)
    digest = hashlib.sha256(data).hexdigest()
    remote_path = f"_jlens_preflight/{artifact_run_id}/{attempt_id}/{digest}.json"
    return {"schema_version": 1, "remote_path": remote_path, "size": len(data), "sha256": digest}, data


def _read_hf_token(path: Path, expected_fingerprint: str) -> str:
    fd = _open_exact_regular(path, label="HF token", mode=0o600)
    try:
        raw = _read_fd_bounded(fd, limit=MAX_TOKEN_BYTES, label="HF token")
    finally:
        os.close(fd)
    try:
        token = raw.decode("utf-8").strip()
    except UnicodeError as exc:
        raise RecoveryError("HF token is malformed") from exc
    if pod.HF_TOKEN_RE.fullmatch(token) is None:
        _fail("HF token is malformed")
    if hashlib.sha256(token.encode("utf-8")).hexdigest() != expected_fingerprint:
        _fail("HF token fingerprint does not match the lease")
    return token


def _verify_results_preflight(publisher: HfPublisher, contract: Mapping[str, Any], token_data: bytes) -> None:
    preflight = contract["results_preflight"]
    remote_path = preflight["remote_path"]
    try:
        safe_relative(remote_path)
    except (TypeError, ValueError) as exc:
        raise RecoveryError("results preflight path is unsafe") from exc
    with tempfile.TemporaryDirectory(prefix="jlens-recovery-preflight-") as temporary:
        destination = Path(temporary) / "preflight.json"
        try:
            publisher.download_file(remote_path, destination, expected_size=preflight["size"])
            observed = _read_regular_file(destination)
        except (OSError, ValueError, TypeError, PublishError) as exc:
            raise RecoveryError("results preflight readback failed") from exc
    if observed != token_data:
        _fail("results preflight readback bytes differ")


def recover_terminated_run(args: argparse.Namespace) -> int:
    """Recover one terminated run without touching the RunPod API."""

    state_path = _explicit_state_path(args)
    state = _read_state(state_path)
    normalized, _ = _validate_state_and_contract(state)
    token = _read_hf_token(normalized["hf_token_path"], normalized["hf_token_sha256"])

    # This publisher is intentionally constructed with the explicit token
    # loaded from the state contract.  No RunPod credential or client is read
    # anywhere in this module.
    stage_publisher = HfPublisher(normalized["staging_repository"], token=token)
    try:
        with tempfile.TemporaryDirectory(prefix="jlens-recovery-stage-") as temporary:
            bootstrap = pod._prepare_stage_bootstrap(
                normalized["stage_path"],
                normalized["staging_repository"],
                Path(temporary),
                publisher=stage_publisher,
            )
            if (bootstrap.manifest_sha256 != normalized["stage_manifest_sha256"]
                    or bootstrap.source_head != normalized["stage_source_head"]):
                _fail("downloaded stage bindings differ from the lease")
            try:
                verified = pod.verify_stage_root(bootstrap.root, allow_extra=False)
            except (OSError, ValueError, PublishError) as exc:
                raise RecoveryError("downloaded stage root could not be verified") from exc
            if (verified.get("manifest_sha256") != normalized["stage_manifest_sha256"]
                    or verified.get("pinned_inputs", {}).get("source", {}).get("head")
                    != normalized["stage_source_head"]):
                _fail("downloaded stage root bindings differ from the lease")

            results_publisher = HfPublisher(normalized["results_repository"], token=token)
            _, preflight_data = _preflight_identity(
                repository=normalized["results_repository"],
                attempt_id=normalized["attempt_id"],
                artifact_run_id=normalized["artifact_run_id"],
                launch_nonce=normalized["launch_nonce"],
                stage_manifest_sha256=normalized["stage_manifest_sha256"],
                stage_source_head=normalized["stage_source_head"],
                hf_token_sha256=normalized["hf_token_sha256"],
            )
            _verify_results_preflight(results_publisher, normalized, preflight_data)
            try:
                pod._sync_terminated_run(
                    results_publisher,
                    run_id=normalized["artifact_run_id"],
                    attempt_id=normalized["attempt_id"],
                    local_root=normalized["sync_root"],
                    receipt_root=normalized["receipt_root"],
                    n_prompts=normalized["n_prompts"],
                    image=normalized["image"],
                    launch_nonce=normalized["launch_nonce"],
                    stage_manifest_sha256=normalized["stage_manifest_sha256"],
                    stage_source_head=normalized["stage_source_head"],
                    worker_deadline_epoch=normalized["worker_deadline_epoch"],
                    replay_source_root=bootstrap.root,
                    compute_exit_code=0,
                    termination_verified=True,
                    provider_lease=normalized["provider_lease"],
                    full_replay=True,
                )
            except (OSError, ValueError, TypeError, PublishError, pod.SafetyError) as exc:
                raise RecoveryError("terminated-run extraction did not pass the paid gates") from exc
    except RecoveryError:
        raise
    except (OSError, ValueError, TypeError, PublishError, pod.SafetyError) as exc:
        raise RecoveryError("terminated-run recovery failed") from exc
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", dest="state_file", required=True, type=Path)
    parsed = parser.parse_args(argv)
    try:
        return recover_terminated_run(parsed)
    except (RecoveryError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
