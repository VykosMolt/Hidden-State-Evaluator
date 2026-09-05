"""Fail-closed RunPod lease controller for the JLens experiment.

The only paid code path is :func:`run_job`; it does not issue a deployment
mutation until a durable pending lease and a disconnect-surviving user
monitor have been installed.  The monitor owns the lease after creation and
continuously checks the balance floor, accrued spend, runtime, and exact pod
identity.  Every exit path attempts termination and verifies that the pod
disappeared from the account listing.

The default CLI is intentionally conservative.  ``create --dry-run`` is
entirely offline; no credential, API, SSH, HF, or systemd call is made.

  python src/ouro_jlens/pod.py create --dry-run
  python src/ouro_jlens/pod.py run --image <name>@sha256:<64>
  python src/ouro_jlens/pod.py monitor --state artifacts/jlens/runs/<run_id>/state.json
  python src/ouro_jlens/pod.py status
  python src/ouro_jlens/pod.py destroy <pod-id>
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from ouro_jlens.publish import (
    HfPublisher,
    HF_REPO_RE,
    PublishError,
    Receipt,
    RUNTIME_IMAGE,
    _path_digest,
    _read_regular_file,
    _write_immutable,
    canonical_json,
    discover_receipt_paths,
    extract_stage_tar,
    is_receipt_index,
    read_hf_token_file,
    read_index,
    receipt_remote_path,
    safe_relative,
    sync_and_verify,
    validate_run_id,
    verify_stage_root,
    verify_stage_tar,
)
API = "https://api.runpod.io/graphql"
KEY_FILE = Path.home() / "Documents" / "Credentials" / "fidelio.txt"
KEY_SELECTOR_ENV = "RUNPOD_API_KEY_SHA256"
PUBKEY = Path.home() / ".ssh" / "id_ed25519_dsv4_b300.pub"
NAME = "ouro-jlens"
NAME_PREFIX = f"{NAME}-"
# A paid run must supply a digest-addressed image.  Keeping the old mutable
# tag out of the controller prevents an omitted CLI value from silently
# selecting a different host image after a registry update.
IMAGE: str | None = None
IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
def _canonical_b300_image() -> str | None:
    """Return the one exact image identity owned by ``publish.py``."""

    return RUNTIME_IMAGE if IMAGE_RE.fullmatch(RUNTIME_IMAGE) else None


# Keep the public compatibility constant useful to callers and tests while
# sourcing it from the publisher rather than copying a mutable image tag or
# digest into the lease controller.
IMAGE = _canonical_b300_image()
TORCH_VERSION = "2.8.0+cu128"
EXPECTED_TORCH_VERSION = TORCH_VERSION
TORCH_VERSION_ENV = "JLENS_EXPECTED_TORCH_VERSION"
PUBLIC_KEY_RE = re.compile(
    r"(?:ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp256|ecdsa-sha2-nistp384|"
    r"ecdsa-sha2-nistp521) [A-Za-z0-9+/]+={0,3}(?: [^\r\n]*)?"
)
HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9][A-Za-z0-9_-]{2,}")
DEFAULT_STATE_ROOT = Path("artifacts/jlens/runs")
DEFAULT_GPU = "NVIDIA B300 SXM6 AC"
DEFAULT_DISK_GB = 150
DEFAULT_MIN_BALANCE = 5.0
DEFAULT_MAX_SPEND = 30.0
DEFAULT_MAX_RUNTIME = 12_600.0
DEFAULT_POLL_SECONDS = 15.0
DEFAULT_MAX_API_FAILURES = 3
DEFAULT_PENDING_TIMEOUT = 180.0
DEFAULT_TERMINATION_ATTEMPTS = 12
DEFAULT_TERMINATION_POLL_SECONDS = 5.0
# These are non-negotiable paid-account boundaries, not merely CLI defaults.
# Callers may choose a higher reserve or lower spend/runtime, but no code path
# may relax them for a B300 lease.
HARD_MIN_BALANCE = 5.0
HARD_MAX_SPEND = 30.0
HARD_MAX_RUNTIME = 12_600.0
TERMINATION_ABSENCE_CONFIRMATIONS = 3
WORKER_DEADLINE_MARGIN_SECONDS = 900
MIN_END_TO_END_WORKER_SECONDS = 10_800.0
WORKER_STARTUP_RESERVE_SECONDS = 300.0
MAX_RUNPOD_CREDENTIAL_BYTES = 64 * 1024
MONITOR_READY_MAX_AGE = 30.0
MONITOR_BUNDLE_FILES = ("__init__.py", "pod.py", "publish.py")
MONITOR_BUNDLE_ROOT = "monitor-bundle"
SYSTEMD_USER_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
PROVIDER_DATETIME_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
LAUNCH_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class SafetyError(RuntimeError):
    """A safety invariant prevents continuing."""


class CredentialSelectionError(SafetyError):
    """The durable credential file does not identify one exact key."""


class APIUncertain(SafetyError):
    """The API did not provide a trustworthy answer."""


class IdentityMismatch(SafetyError):
    """The observed pod is not the exact leased pod."""


class TerminationUnverified(SafetyError):
    """Termination was attempted but absence could not be verified."""


def _default_run_id() -> str:
    # UUID4 is opaque and gives every mutation a unique pod name.  It is also
    # safe as a path component and remains stable across monitor restarts.
    return uuid.uuid4().hex


def api_key(key_file: Path = KEY_FILE) -> str:
    """Load a deployment key without ever printing it."""

    env_key = os.environ.get("RUNPOD_API_KEY")
    if env_key is not None:
        if not re.fullmatch(r"rpa_[A-Za-z0-9]+", env_key):
            raise SafetyError("RUNPOD_API_KEY has an invalid format")
        selector = os.environ.get(KEY_SELECTOR_ENV)
        if selector is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", selector):
                raise SafetyError(f"{KEY_SELECTOR_ENV} has an invalid format")
            fingerprint = hashlib.sha256(env_key.encode("utf-8")).hexdigest()
            if fingerprint != selector:
                raise SafetyError("RUNPOD_API_KEY does not match the selected credential")
        return env_key
    selector = os.environ.get(KEY_SELECTOR_ENV)
    if selector is not None and not re.fullmatch(r"[0-9a-f]{64}", selector):
        raise SafetyError(f"{KEY_SELECTOR_ENV} has an invalid format")
    return _select_file_key(key_file, selector)


def _select_file_key(key_file: Path = KEY_FILE, selector: str | None = None) -> str:
    """Select one exact credential from the durable credential file.

    The detached monitor deliberately uses this path even when the launching
    shell had a ``RUNPOD_API_KEY`` environment variable.  Only the resulting
    fingerprint crosses the lease boundary.
    """

    key_file = Path(key_file)
    text = _read_private_credential_file(
        key_file,
        name="RunPod credential file",
        max_bytes=MAX_RUNPOD_CREDENTIAL_BYTES,
    )
    # Repeated copies of the same credential are not an identity ambiguity.
    # Deduplicate exact values while preserving the file's order; genuinely
    # different credentials still require an explicit fingerprint selector.
    keys = list(dict.fromkeys(
        re.findall(r"(?<![A-Za-z0-9_])rpa_[A-Za-z0-9]+(?![A-Za-z0-9_])", text)
    ))
    if selector is not None:
        matches = [
            key for key in keys
            if hashlib.sha256(key.encode("utf-8")).hexdigest() == selector
        ]
        if len(matches) != 1:
            raise SafetyError("RunPod key selector does not identify exactly one credential")
        return matches[0]
    if len(keys) != 1:
        raise CredentialSelectionError(
            f"credential file must contain exactly one RunPod API key unless an exact SHA-256 selector is supplied: {key_file}"
        )
    return keys[0]


def _read_private_credential_file(
    path: Path,
    *,
    name: str,
    max_bytes: int,
) -> str:
    """Read one exact-0600 regular credential through a stable descriptor."""

    path = Path(path)
    try:
        _reject_state_symlinks(path)
    except SafetyError:
        raise SafetyError(f"cannot read {name}: unsafe path") from None
    flags = (
        os.O_RDONLY
        | os.O_NONBLOCK
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError:
        raise SafetyError(f"cannot read {name}: {path}") from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SafetyError(f"{name} is not a regular file")
        if stat.S_IMODE(before.st_mode) != 0o600:
            raise SafetyError(f"{name} must have mode 0600")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise SafetyError(f"{name} has invalid size")
        chunks: list[bytes] = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(fd, min(4096, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if total > max_bytes or total != before.st_size:
            raise SafetyError(f"{name} has invalid size")
        if identity_after != identity_before:
            raise SafetyError(f"{name} changed while being read")
    except OSError:
        raise SafetyError(f"cannot read {name}: {path}") from None
    finally:
        os.close(fd)
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError:
        raise SafetyError(f"{name} is not valid UTF-8") from None


def _activate_key_selector(args: argparse.Namespace) -> str | None:
    """Activate one non-secret selector and bind it to an env credential.

    A selector supplied by the CLI cannot override a selector already present
    in the environment.  When the shell supplies the key directly, derive the
    selector from that exact value and reject every explicit mismatch before a
    client can issue a request.
    """

    explicit = getattr(args, "runpod_key_sha256", None)
    environment = os.environ.get(KEY_SELECTOR_ENV)
    for value, label in ((explicit, "RunPod key selector"),
                         (environment, f"{KEY_SELECTOR_ENV}")):
        if value is not None and (
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        ):
            raise SafetyError(f"{label} must be a lowercase SHA-256 digest")
    if explicit is not None and environment is not None and explicit != environment:
        raise SafetyError("explicit RunPod key selector conflicts with the environment selector")

    env_key = os.environ.get("RUNPOD_API_KEY")
    derived = None
    if env_key is not None:
        if not re.fullmatch(r"rpa_[A-Za-z0-9]+", env_key):
            raise SafetyError("RUNPOD_API_KEY has an invalid format")
        derived = hashlib.sha256(env_key.encode("utf-8")).hexdigest()
        requested = explicit if explicit is not None else environment
        if requested is not None and requested != derived:
            raise SafetyError("RUNPOD_API_KEY does not match the selected credential")

    selector = derived or explicit or environment
    if selector is not None:
        os.environ[KEY_SELECTOR_ENV] = selector
        setattr(args, "runpod_key_sha256", selector)
    return selector


def _credential_fingerprint(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"rpa_[A-Za-z0-9]+", value):
        raise SafetyError("RunPod credential has an invalid format")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _foreground_credential_fingerprint(args: argparse.Namespace) -> str | None:
    """Resolve and persist the exact credential identity used before deploy."""

    selector = _activate_key_selector(args)
    env_key = os.environ.get("RUNPOD_API_KEY")
    if env_key is not None:
        fingerprint = _credential_fingerprint(env_key)
        # A shell-only credential cannot establish a disconnect-safe monitor:
        # the detached process must be able to resolve this same key from the
        # durable credential file.
        _select_file_key(KEY_FILE, fingerprint)
    elif selector is not None:
        _select_file_key(KEY_FILE, selector)
        fingerprint = selector
    else:
        try:
            fingerprint = _credential_fingerprint(_select_file_key(KEY_FILE))
        except CredentialSelectionError:
            # Leave an ambiguous file unresolved rather than guessing.  A
            # real RunPodClient calls api_key() before any mutation and fails
            # closed; the creator also cannot accept a READY acknowledgement
            # without a concrete fingerprint.
            fingerprint = None
    if fingerprint is not None:
        setattr(args, "runpod_key_sha256", fingerprint)
    return fingerprint


def gql(query: str, variables: Mapping[str, Any] | None = None, *,
        key_file: Path = KEY_FILE, credential_fingerprint: str | None = None,
        file_only: bool = False) -> dict[str, Any]:
    """Perform one API request, converting all uncertainty to APIUncertain."""

    payload: dict[str, Any] = {"query": query}
    if variables is not None:
        payload["variables"] = dict(variables)
    request = urllib.request.Request(
        API,
        json.dumps(payload).encode(),
        {
            "content-type": "application/json",
            "Authorization": f"Bearer {_api_key_for_request(key_file, credential_fingerprint, file_only)}",
            "User-Agent": "curl/8.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            out = json.load(response)
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise APIUncertain("RunPod API request failed") from exc
    if not isinstance(out, dict) or out.get("errors"):
        raise APIUncertain("RunPod API returned an error")
    data = out.get("data")
    if not isinstance(data, dict):
        raise APIUncertain("RunPod API returned no data")
    return data


def _api_key_for_request(key_file: Path, fingerprint: str | None,
                         file_only: bool) -> str:
    """Resolve the request credential without crossing identity boundaries."""

    if file_only:
        if fingerprint is None:
            raise SafetyError("detached monitor has no credential fingerprint")
        if (not isinstance(fingerprint, str)
                or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)):
            raise SafetyError("detached monitor credential fingerprint is malformed")
        return _select_file_key(key_file, fingerprint)
    return api_key(key_file)


class RunPodClient:
    """Small injectable API facade; tests can provide a fake ``gql`` callable."""

    def __init__(self, gql_fn: Callable[..., dict[str, Any]] | None = None,
                 *, credential_fingerprint: str | None = None,
                 key_file: Path = KEY_FILE, file_only: bool = False):
        self.credential_fingerprint = credential_fingerprint
        self.key_file = Path(key_file)
        self.file_only = bool(file_only or credential_fingerprint is not None)
        if gql_fn is not None:
            self.gql = gql_fn
        else:
            self.gql = lambda query, variables=None: gql(
                query, variables, key_file=self.key_file,
                credential_fingerprint=self.credential_fingerprint,
                file_only=self.file_only,
            )
        self._deployment_guard: object | None = None

    def _arm_deployment(self, guard: object) -> None:
        """Issue a one-shot mutation capability to the lease creator."""

        if self._deployment_guard is not None:
            raise SafetyError("a deployment mutation is already armed")
        self._deployment_guard = guard

    def balance_details(self) -> dict[str, Any]:
        try:
            myself = self.gql("{ myself { clientBalance currentSpendPerHr } }")["myself"]
            balance = float(myself["clientBalance"])
            spend_rate = float(myself.get("currentSpendPerHr") or 0)
            if not math.isfinite(balance) or not math.isfinite(spend_rate) or spend_rate < 0:
                raise ValueError("non-finite balance response")
            return {
                "clientBalance": balance,
                "currentSpendPerHr": spend_rate,
            }
        except (KeyError, TypeError, ValueError, OSError, TimeoutError, APIUncertain) as exc:
            if isinstance(exc, APIUncertain):
                raise
            raise APIUncertain("RunPod balance response was malformed") from exc

    def balance(self) -> float:
        return self.balance_details()["clientBalance"]

    def pods(self) -> list[dict[str, Any]]:
        # These fields are part of the listing shape already used by the
        # original controller.  machineId is included where available so a
        # returned object cannot silently be rebound by name alone.
        query = """{ myself { pods { id name machineId desiredStatus costPerHr
            machine { gpuDisplayName }
            runtime { uptimeInSeconds ports { ip publicPort privatePort isIpPublic } } } } }"""
        try:
            pods = self.gql(query)["myself"]["pods"]
            if not isinstance(pods, list):
                raise TypeError("pods is not a list")
            return [dict(p) for p in pods]
        except APIUncertain:
            raise
        except (KeyError, TypeError, ValueError, OSError, TimeoutError) as exc:
            raise APIUncertain("RunPod pod listing was malformed") from exc

    def gpu_offer(self, gpu: str) -> dict[str, Any]:
        """Return the exact current one-GPU secure-cloud offer."""

        gpu = _validate_paid_gpu(gpu)
        query = """query GPUOffer($id: String!) {
          gpuTypes(input: {id: $id}) {
            id displayName memoryInGb secureCloud communityCloud
            lowestPrice(input: {gpuCount: 1, secureCloud: true}) {
              stockStatus uninterruptablePrice availableGpuCounts
            }
          }
        }"""
        try:
            offers = self.gql(query, {"id": gpu})["gpuTypes"]
            if not isinstance(offers, list) or len(offers) != 1:
                raise TypeError("exact GPU offer is not unique")
            offer = dict(offers[0])
            price = dict(offer["lowestPrice"])
            rate = float(price["uninterruptablePrice"])
            stock = price["stockStatus"]
            if (offer.get("id") != gpu or offer.get("displayName") != "B300"
                    or offer.get("memoryInGb") != 288
                    or offer.get("secureCloud") is not True
                    or not isinstance(stock, str)
                    or stock.strip().lower() in {"", "none"}
                    or not math.isfinite(rate) or rate <= 0):
                raise ValueError("exact secure B300 offer is unavailable")
            price["uninterruptablePrice"] = rate
            offer["lowestPrice"] = price
            return offer
        except APIUncertain:
            raise
        except (KeyError, TypeError, ValueError, OSError, TimeoutError) as exc:
            raise APIUncertain("RunPod secure B300 offer was malformed or unavailable") from exc

    def deploy(self, *, name: str, gpu: str, disk: int, public_key: str, hf_token: str,
               image: str | None = None, torch_version: str = TORCH_VERSION,
               image_ref: str | None = None, terminate_after: str | None = None,
               _guard: object | None = None) -> dict[str, Any]:
        """Create exactly one uniquely named pod after supervision is armed."""

        if self._deployment_guard is None or _guard is not self._deployment_guard:
            raise SafetyError("deployment requires an armed supervised lease")
        if image is None:
            image = image_ref
        elif image_ref is not None and image_ref != image:
            raise SafetyError("image and image_ref disagree")
        image = _validate_paid_image(image)
        if (not isinstance(name, str)
                or not re.fullmatch(rf"{re.escape(NAME_PREFIX)}[A-Za-z0-9][A-Za-z0-9_.-]{{0,127}}", name)):
            raise SafetyError("deployment name is not a safe JLens lease name")
        _validate_paid_gpu(gpu)
        _positive_int(disk, "disk")
        if not isinstance(terminate_after, str) or PROVIDER_DATETIME_RE.fullmatch(
            terminate_after
        ) is None:
            raise SafetyError("deployment requires an exact provider termination deadline")
        if (not isinstance(public_key, str) or not PUBLIC_KEY_RE.fullmatch(public_key.strip())):
            raise SafetyError("public key is empty or malformed")
        if (not isinstance(hf_token, str) or not HF_TOKEN_RE.fullmatch(hf_token)):
            raise SafetyError("HF token is empty or malformed")
        if (not isinstance(torch_version, str)
                or not re.fullmatch(
                    r"[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:\+cu[0-9]+)?", torch_version
                )):
            raise SafetyError("expected Torch version is malformed")
        if torch_version != TORCH_VERSION:
            raise SafetyError("expected Torch version does not match the pinned controller constant")
        # Consume the capability before the network mutation.  A caller must
        # establish a fresh durable lease and monitor before any retry.
        self._deployment_guard = None
        # The values are sent as GraphQL variables.  They never enter a
        # query string, logs, exception text, or dry-run output.
        query = """mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id machineId costPerHr desiredStatus name machine { gpuDisplayName }
          }
        }"""
        variables = {
            "input": {
                # The repository token is never exposed to a community host.
                "cloudType": "SECURE",
                "gpuCount": 1,
                "gpuTypeId": gpu,
                "name": name,
                "imageName": image,
                "containerDiskInGb": disk,
                "volumeInGb": 0,
                "minVcpuCount": 16,
                "minMemoryInGb": 64,
                "ports": "22/tcp",
                # Provider-enforced deletion survives controller reboot,
                # suspend, network loss, and local monitor failure.
                "terminateAfter": terminate_after,
                "env": [{"key": "PUBLIC_KEY", "value": public_key},
                        # pod_entry immediately moves this one-use bootstrap
                        # secret into a protected file and unsets it before
                        # any dependency or project process is started.
                        {"key": "JLENS_HF_TOKEN_BOOTSTRAP", "value": hf_token},
                        {"key": "TORCH_VERSION", "value": torch_version},
                        {"key": "JLENS_TORCH_VERSION", "value": torch_version},
                        {"key": TORCH_VERSION_ENV, "value": torch_version}],
            }
        }
        try:
            raw = self.gql(query, variables)
            pod = raw.get("podFindAndDeployOnDemand")
            if not isinstance(pod, dict) or not pod.get("id"):
                raise APIUncertain("deploy response did not contain a pod identity")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", str(pod["id"])):
                raise APIUncertain("deploy response contained an invalid pod identity")
            return dict(pod)
        except APIUncertain:
            raise
        except (AttributeError, TypeError, KeyError, ValueError, OSError, TimeoutError) as exc:
            raise APIUncertain("deploy response was malformed") from exc

    def terminate(self, pod_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
            raise SafetyError("invalid pod id")
        try:
            # Pod IDs are restricted above, so this inline value cannot
            # inject GraphQL syntax and avoids relying on a provider-specific
            # ID-vs-String variable type.
            result = self.gql(f"mutation {{ podTerminate(input: {{podId: {json.dumps(pod_id)}}}) }}")
            # Some API versions return a Boolean, others return a small object;
            # either is accepted only when no GraphQL error was reported.
            if "podTerminate" not in result or result["podTerminate"] in (None, False):
                raise APIUncertain("terminate response did not acknowledge the request")
        except APIUncertain:
            raise
        except (TypeError, KeyError, ValueError, OSError, TimeoutError) as exc:
            raise APIUncertain("terminate response was malformed") from exc


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _reject_state_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_parent(path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_parent(path: Path) -> None:
    """Make a completed lease-state directory entry durable before mutation."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path.parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise SafetyError(f"cannot durably persist lease state directory: {path.parent}") from exc


def _create_initial_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one lease state file without ever replacing an existing path.

    Lease state is the durable capability boundary for a paid attempt.  The
    first write therefore uses ``O_EXCL`` (and ``O_NOFOLLOW`` where available)
    instead of the update helper above.  Later monitor updates may use the
    atomic replacement helper, but a run ID can never be re-opened by
    replacing a terminated or otherwise stale state file.
    """

    path = Path(path)
    _reject_state_symlinks(path)
    if path.exists() or path.is_symlink():
        raise SafetyError(f"lease state already exists: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SafetyError(f"cannot create lease state directory: {path.parent}") from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise SafetyError(f"lease state already exists: {path}") from exc
    except OSError as exc:
        raise SafetyError(f"cannot create lease state: {path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent(path)
    except BaseException:
        # A failed initial serialization is still not allowed to leave a
        # misleading partial state that a later attempt could replace.
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _reject_state_symlinks(path: Path) -> None:
    """Keep durable lease state on the requested filesystem path."""

    path = Path(path)
    if path.is_symlink():
        raise SafetyError(f"lease state path is a symlink: {path}")
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise SafetyError(f"lease state has a symlinked parent: {current}")
        current = current.parent


def _bundle_digest(files: Mapping[str, str]) -> str:
    payload = json.dumps({"files": dict(sorted(files.items()))},
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SafetyError(f"cannot read monitor bundle file: {path}") from exc
    return digest.hexdigest()


def _monitor_bundle_metadata(root: Path, files: Mapping[str, str]) -> dict[str, Any]:
    digest = _bundle_digest(files)
    relative_root = f"{MONITOR_BUNDLE_ROOT}/{digest}"
    return {"root": relative_root, "digest": digest,
            "files": dict(sorted(files.items()))}


def _verify_monitor_bundle(state_file: Path, state: Mapping[str, Any]) -> Path:
    """Verify the content-addressed monitor bundle before every launch."""

    raw = state.get("monitor_bundle")
    if not isinstance(raw, Mapping):
        raise SafetyError("lease state has no immutable monitor bundle")
    root_value = raw.get("root")
    digest = raw.get("digest")
    files = raw.get("files")
    if (not isinstance(root_value, str) or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or root_value != f"{MONITOR_BUNDLE_ROOT}/{digest}"
            or not isinstance(files, Mapping)):
        raise SafetyError("monitor bundle metadata is malformed")
    expected_files = {
        f"ouro_jlens/{name}" for name in MONITOR_BUNDLE_FILES
    }
    if set(files) != expected_files:
        raise SafetyError("monitor bundle does not contain the required package")
    normalized: dict[str, str] = {}
    for relative, expected in files.items():
        if (not isinstance(relative, str) or not isinstance(expected, str)
                or not re.fullmatch(r"ouro_jlens/[A-Za-z0-9_.-]+", relative)
                or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise SafetyError("monitor bundle file metadata is malformed")
        normalized[relative] = expected
    if _bundle_digest(normalized) != digest:
        raise SafetyError("monitor bundle content address is invalid")

    state_file = Path(state_file)
    _reject_state_symlinks(state_file)
    bundle_root = state_file.parent / root_value
    _reject_state_symlinks(bundle_root)
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise SafetyError("monitor bundle is missing or not a directory")
    try:
        bundle_root.resolve().relative_to(state_file.parent.resolve())
    except ValueError as exc:
        raise SafetyError("monitor bundle escapes the lease state directory") from exc
    for relative, expected in normalized.items():
        path = bundle_root / relative
        _reject_state_symlinks(path)
        if path.is_symlink() or not path.is_file() or _sha256_path(path) != expected:
            raise SafetyError("monitor bundle hash verification failed")
    return bundle_root


def _install_monitor_bundle(state_file: Path, *, source: Path | None = None) -> dict[str, Any]:
    """Copy the monitor package into a local content-addressed bundle."""

    state_file = Path(state_file)
    _reject_state_symlinks(state_file)
    state_dir = state_file.parent
    state_dir.mkdir(parents=True, exist_ok=True)
    _reject_state_symlinks(state_dir)
    package = Path(source) if source is not None else Path(__file__).resolve().parent
    package = package.expanduser()
    if package.is_symlink() or not package.is_dir():
        raise SafetyError("monitor bundle source is not a regular package directory")
    files: dict[str, bytes] = {}
    for name in MONITOR_BUNDLE_FILES:
        path = package / name
        _reject_state_symlinks(path)
        if path.is_symlink() or not path.is_file():
            raise SafetyError(f"monitor bundle source is missing {name}")
        try:
            files[f"ouro_jlens/{name}"] = path.read_bytes()
        except OSError as exc:
            raise SafetyError(f"cannot read monitor bundle source: {name}") from exc
    hashes = {relative: hashlib.sha256(content).hexdigest()
              for relative, content in files.items()}
    metadata = _monitor_bundle_metadata(state_dir, hashes)
    bundle_root = state_dir / metadata["root"]
    bundle_parent = bundle_root.parent
    bundle_parent.mkdir(parents=True, exist_ok=True)
    _reject_state_symlinks(bundle_parent)
    _reject_state_symlinks(bundle_root)
    if bundle_root.exists():
        _verify_monitor_bundle(state_file, {"monitor_bundle": metadata})
        return metadata

    temporary = Path(tempfile.mkdtemp(prefix=".monitor-bundle-", dir=str(state_dir)))
    try:
        target = temporary / "ouro_jlens"
        target.mkdir()
        for relative, content in files.items():
            destination = temporary / relative
            destination.write_bytes(content)
        try:
            os.replace(temporary, bundle_root)
        except FileExistsError:
            # Another creator may have installed the same content address in
            # the meantime.  Verify its bytes instead of replacing it.
            _verify_monitor_bundle(state_file, {"monitor_bundle": metadata})
            shutil.rmtree(temporary, ignore_errors=True)
            return metadata
        _fsync_parent(bundle_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _verify_monitor_bundle(state_file, {"monitor_bundle": metadata})
    return metadata


def _read_json(path: Path) -> dict[str, Any]:
    _reject_state_symlinks(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SafetyError(f"cannot read lease state: {path}") from exc
    if not isinstance(value, dict):
        raise SafetyError(f"lease state is not an object: {path}")
    return value


def state_path(run_id: str, state_root: str | Path = DEFAULT_STATE_ROOT) -> Path:
    return Path(state_root) / validate_run_id(run_id) / "state.json"


def _name_for(run_id: str) -> str:
    return f"{NAME_PREFIX}{validate_run_id(run_id)}"


def _finite_number(value: Any, name: str, *, minimum: float | None = None,
                   strict_minimum: bool = False) -> float:
    """Validate one numeric CLI value before any lease mutation."""

    if isinstance(value, bool):
        raise SafetyError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SafetyError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise SafetyError(f"{name} must be finite")
    if minimum is not None and (number <= minimum if strict_minimum else number < minimum):
        comparator = ">" if strict_minimum else ">="
        raise SafetyError(f"{name} must be {comparator}{minimum}")
    return number


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SafetyError(f"{name} must be a positive integer")
    if value < minimum:
        raise SafetyError(f"{name} must be a positive integer")
    return value


def _validate_image_reference(value: Any) -> str:
    if not isinstance(value, str) or not IMAGE_RE.fullmatch(value):
        raise SafetyError("paid runs require an immutable image reference name@sha256:<64>")
    return value


def _validate_paid_image(value: Any) -> str:
    """Require a paid run to use the publisher's canonical B300 image."""

    image = _validate_image_reference(value)
    approved = _canonical_b300_image()
    if approved is None:
        raise SafetyError("publisher-approved canonical B300 image is unavailable")
    if image != approved:
        raise SafetyError("paid runs require the publisher-approved canonical B300 image")
    return approved


def _validate_gpu(value: Any) -> str:
    if (not isinstance(value, str) or not value.strip()
            or any(char in value for char in "\r\n\x00")):
        raise SafetyError("GPU type is empty or malformed")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._+:/-]*", value):
        raise SafetyError("GPU type contains unsafe characters")
    return value


def _validate_paid_gpu(value: Any) -> str:
    """Require the one provider GPU identity covered by the paid contract."""

    gpu = _validate_gpu(value)
    if gpu != DEFAULT_GPU:
        raise SafetyError(f"paid runs require exact GPU type {DEFAULT_GPU!r}")
    return gpu


def _require_assigned_b300(pod: Mapping[str, Any]) -> str:
    machine = pod.get("machine")
    name = machine.get("gpuDisplayName") if isinstance(machine, Mapping) else None
    # RunPod's pod listing uses aliases such as ``B300`` or ``NVIDIA B300``
    # for the exact SKU requested by the deploy mutation.  This API field is
    # corroborating metadata; pod_entry.sh checks the physical device with
    # nvidia-smi before downloading any paid-workload bytes.
    if (not isinstance(name, str)
            or re.search(r"(?i)(?<![A-Za-z0-9_])B300(?![A-Za-z0-9_])", name) is None):
        raise IdentityMismatch(
            f"provider did not identify the assigned GPU as B300: {name!r}"
        )
    # Lease/recovery records use the exact requested SKU as their canonical
    # identity; the raw provider alias is not an independently stable API.
    return DEFAULT_GPU


def _optional_assigned_b300(pod: Mapping[str, Any]) -> str | None:
    """Validate an early assignment when present; allow provider pending null."""

    machine = pod.get("machine")
    if machine is None:
        return None
    if not isinstance(machine, Mapping):
        raise IdentityMismatch("provider returned a malformed machine assignment")
    name = machine.get("gpuDisplayName")
    if name is None:
        return None
    return _require_assigned_b300(pod)


def _validate_launch_nonce(value: Any) -> str:
    if not isinstance(value, str) or LAUNCH_NONCE_RE.fullmatch(value) is None:
        raise SafetyError("launch nonce is missing or malformed")
    return value


def _validate_stage_manifest_digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SafetyError("stage manifest digest is missing or malformed")
    return value


def _validate_source_head(value: Any) -> str:
    if not isinstance(value, str) or GIT_OID_RE.fullmatch(value) is None:
        raise SafetyError("stage source HEAD is missing or malformed")
    return value


def _provider_termination_deadline(now: float, max_runtime: float) -> str:
    """Return the absolute RunPod TTL in its GraphQL ``DateTime`` format."""

    now = _finite_number(now, "provider deadline clock", minimum=0)
    max_runtime = _finite_number(
        max_runtime, "provider maximum runtime", minimum=0, strict_minimum=True
    )
    deadline = datetime.fromtimestamp(now, timezone.utc) + timedelta(seconds=max_runtime)
    value = deadline.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if PROVIDER_DATETIME_RE.fullmatch(value) is None:
        raise SafetyError("provider termination deadline could not be encoded")
    return value


def _worker_deadline_epoch(provider_deadline: str, *, now: float) -> int:
    """Reserve a fixed publication window before the provider hard-kill."""

    if (not isinstance(provider_deadline, str)
            or PROVIDER_DATETIME_RE.fullmatch(provider_deadline) is None):
        raise SafetyError("provider termination deadline is malformed")
    now = _finite_number(now, "worker deadline clock", minimum=0)
    deadline = int(datetime.strptime(
        provider_deadline, "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc).timestamp()) - WORKER_DEADLINE_MARGIN_SECONDS
    if deadline <= now + 60:
        raise SafetyError("provider runtime leaves no bounded artifact-publication window")
    return deadline


def _validate_offer_budget(offer: Mapping[str, Any], config: "SafetyConfig") -> float:
    """Require the quoted full-runtime cost to fit the hard local spend cap."""

    try:
        rate = float(offer["lowestPrice"]["uninterruptablePrice"])
    except (KeyError, TypeError, ValueError) as exc:
        raise APIUncertain("RunPod secure B300 offer omitted its hourly price") from exc
    projected = rate * config.max_runtime / 3600.0
    if not math.isfinite(projected) or projected > config.max_spend:
        raise SafetyError(
            f"quoted full-runtime cost ${projected:.2f} exceeds the ${config.max_spend:.2f} cap"
        )
    return projected


def _validate_paid_workload_window(config: "SafetyConfig") -> None:
    """Reject a paid TTL that cannot cover the declared end-to-end workload."""

    usable = config.max_runtime - WORKER_DEADLINE_MARGIN_SECONDS
    if usable < MIN_END_TO_END_WORKER_SECONDS:
        raise SafetyError(
            "paid runtime leaves less than the 10800-second end-to-end worker floor"
        )


def _require_remaining_worker_window(
    worker_deadline_epoch: int,
    *,
    reserve_seconds: float = 0.0,
    now: float | None = None,
) -> float:
    """Fail before remote startup when provisioning consumed the run budget."""

    if (isinstance(worker_deadline_epoch, bool)
            or not isinstance(worker_deadline_epoch, int)
            or worker_deadline_epoch <= 0):
        raise SafetyError("lease worker deadline is missing or malformed")
    reserve = _finite_number(
        reserve_seconds, "worker startup reserve", minimum=0
    )
    current = time.time() if now is None else _finite_number(
        now, "worker-window clock", minimum=0
    )
    remaining = worker_deadline_epoch - current
    if remaining < MIN_END_TO_END_WORKER_SECONDS + reserve:
        raise SafetyError(
            "provisioning left less than the required end-to-end worker window"
        )
    return remaining


def _validate_path_argument(value: Any, name: str, *, allow_missing: bool = False) -> Path:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
    else:
        raise SafetyError(f"{name} is not a valid path")
    _reject_state_symlinks(path)
    if path.exists() and not path.is_file() and not path.is_dir():
        raise SafetyError(f"{name} is not a regular file or directory")
    if not allow_missing and not path.exists():
        raise SafetyError(f"{name} does not exist: {path}")
    return path


def _validate_secret_file(path_value: Any, name: str, pattern: re.Pattern[str]) -> tuple[Path, str]:
    path = _validate_path_argument(path_value, name)
    if path.is_symlink() or not path.is_file():
        raise SafetyError(f"{name} is not a regular file")
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise SafetyError(f"cannot read {name}") from exc
    if not pattern.fullmatch(value):
        raise SafetyError(f"{name} is empty or malformed")
    return path, value


def _validate_public_key_file(path_value: Any) -> tuple[Path, str]:
    # Public keys are passed to cloud-init as one line.  Restrict the key
    # algorithms and base64 body, while allowing the usual optional comment.
    return _validate_secret_file(path_value, "public key", PUBLIC_KEY_RE)


def _validate_hf_token_file(path_value: Any) -> tuple[Path, str]:
    try:
        path = Path(path_value).expanduser()
    except (TypeError, ValueError):
        raise SafetyError("HF token file path is malformed") from None
    try:
        token = read_hf_token_file(path)
    except PublishError:
        raise SafetyError("HF token file must be a stable regular 0600 file with valid content") from None
    return path, token


def _validate_identity_file(path_value: Any) -> Path:
    path = _validate_path_argument(path_value, "identity file")
    if path.is_symlink() or not path.is_file():
        raise SafetyError("identity file is not a regular file")
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise SafetyError("identity file is not readable") from exc
    return path


def _validate_ssh_key_pair(identity_file: Path, public_key: str) -> None:
    """Prove the launch public key belongs to the controller identity."""

    identity_file = _validate_identity_file(identity_file)
    try:
        mode = identity_file.stat().st_mode & 0o777
    except OSError as exc:
        raise SafetyError("cannot inspect SSH identity permissions") from exc
    if mode & 0o077:
        raise SafetyError("SSH identity file must not be accessible to group or other users")
    try:
        result = subprocess.run(
            ["ssh-keygen", "-y", "-f", str(identity_file)],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError("could not derive the SSH public key from the identity") from exc
    if result.returncode != 0:
        raise SafetyError("SSH identity is unreadable or invalid")
    derived = (result.stdout or "").strip().split()
    requested = public_key.strip().split()
    if len(derived) < 2 or len(requested) < 2 or derived[:2] != requested[:2]:
        raise SafetyError("SSH identity does not match the launch public key")


def _validate_ssh_port(value: Any, name: str = "SSH port") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise SafetyError(f"{name} is invalid")
    return value


def _validate_ssh_host(value: Any) -> str:
    if (not isinstance(value, str) or not value
            or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value)):
        raise SafetyError("SSH host is invalid")
    return value


def _validate_machine_id(value: Any) -> str:
    if (not isinstance(value, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,255}", value) is None):
        raise SafetyError("RunPod machine identity is missing or invalid")
    return value


def _validate_run_dimensions(args: argparse.Namespace) -> None:
    """Validate values interpolated into the remote command."""

    if hasattr(args, "n_prompts"):
        _positive_int(getattr(args, "n_prompts"), "n_prompts")
    if hasattr(args, "dim_batch"):
        _positive_int(getattr(args, "dim_batch"), "dim_batch")


def _resolve_run_ids(args: argparse.Namespace) -> tuple[str, str]:
    """Return ``(lease_attempt_id, artifact_lineage_id)``.

    The parser retains the explicit names for durable-state compatibility,
    but the paid path currently requires both identities to match. Attempt-
    variant validation/report bytes are not yet safely namespaced for reuse.
    """

    requested_run = getattr(args, "run_id", None)
    requested_attempt = getattr(args, "attempt_id", None)
    requested_lineage = getattr(args, "artifact_run_id", None)
    if requested_attempt is not None:
        attempt_id = validate_run_id(requested_attempt)
        lineage_value = requested_lineage if requested_lineage is not None else requested_run
        lineage_id = validate_run_id(lineage_value if lineage_value is not None else attempt_id)
    else:
        attempt_id = validate_run_id(requested_run if requested_run is not None else _default_run_id())
        lineage_id = validate_run_id(requested_lineage if requested_lineage is not None else attempt_id)
    return attempt_id, lineage_id


def _validate_state_roots(args: argparse.Namespace, attempt_id: str) -> Path:
    raw_root = getattr(args, "state_root", DEFAULT_STATE_ROOT)
    if not isinstance(raw_root, (str, Path)):
        raise SafetyError("state root is not a valid path")
    root = Path(raw_root).expanduser()
    _reject_state_symlinks(root)
    if root.exists() and not root.is_dir():
        raise SafetyError("state root is not a directory")
    state_file = state_path(attempt_id, root)
    _reject_state_symlinks(state_file)
    return state_file


def _assert_new_state_path(state_file: Path) -> None:
    _reject_state_symlinks(state_file)
    if state_file.exists() or state_file.is_symlink():
        raise SafetyError(f"lease state already exists: {state_file}")


def _validate_lease_args(args: argparse.Namespace, *, attempt_id: str,
                         artifact_run_id: str) -> dict[str, Any]:
    """Validate all paid-lease inputs before writing state or arming deploy."""

    if artifact_run_id != attempt_id:
        raise SafetyError(
            "paid same-lineage retry is disabled; use one fresh attempt/artifact run ID"
        )
    if getattr(args, "allow_second", False):
        raise SafetyError("--allow-second is disabled; every lease attempt needs a unique pod name")
    _validate_run_dimensions(args)
    image = _validate_image_reference(getattr(args, "image", None))
    gpu = _validate_paid_gpu(getattr(args, "gpu", None))
    disk = _positive_int(getattr(args, "disk", None), "disk", minimum=1)
    identity_file = _validate_identity_file(getattr(args, "identity_file", None))
    pubkey_path, public_key = _validate_public_key_file(getattr(args, "pubkey", None))
    hf_token_path, hf_token = _validate_hf_token_file(getattr(args, "hf_token", None))
    hf_token_sha256 = hashlib.sha256(hf_token.encode("utf-8")).hexdigest()
    expected_hf_token_sha256 = getattr(args, "expected_hf_token_sha256", None)
    if (expected_hf_token_sha256 is not None
            and expected_hf_token_sha256 != hf_token_sha256):
        raise SafetyError("HF token changed after the controller preflight")
    ssh_timeout = _finite_number(getattr(args, "ssh_timeout", 900), "ssh_timeout",
                                 minimum=0, strict_minimum=True)
    ssh_connect_timeout = _finite_number(
        getattr(args, "ssh_connect_timeout", 30), "ssh_connect_timeout",
        minimum=0, strict_minimum=True,
    )
    if ssh_connect_timeout < 1:
        raise SafetyError("ssh_connect_timeout must be at least one second")
    state_file = _validate_state_roots(args, attempt_id)
    # _lease_config performs the complete finite/range validation for the
    # supervisor limits.  Reconstructing it here ensures no state write can
    # precede a malformed timing value.
    config = _lease_config(args)
    _validate_paid_workload_window(config)
    launch_nonce = getattr(args, "launch_nonce", None)
    if launch_nonce is None:
        launch_nonce = secrets.token_urlsafe(48)
        args.launch_nonce = launch_nonce
    launch_nonce = _validate_launch_nonce(launch_nonce)
    stage_manifest_sha256 = getattr(args, "expected_stage_manifest_sha256", None)
    stage_source_head = getattr(args, "expected_stage_source_head", None)
    if getattr(args, "stage_path", None) is not None:
        stage_manifest_sha256 = _validate_stage_manifest_digest(stage_manifest_sha256)
        stage_source_head = _validate_source_head(stage_source_head)
    return {
        "attempt_id": attempt_id,
        "artifact_run_id": artifact_run_id,
        "run_id": attempt_id,
        "state_file": state_file,
        "image": image,
        "gpu": gpu,
        "disk": disk,
        "identity_file": identity_file,
        "pubkey_path": pubkey_path,
        "public_key": public_key,
        "hf_token_path": hf_token_path,
        "hf_token": hf_token,
        "hf_token_sha256": hf_token_sha256,
        "ssh_timeout": ssh_timeout,
        "ssh_connect_timeout": ssh_connect_timeout,
        "config": config,
        "launch_nonce": launch_nonce,
        "stage_manifest_sha256": stage_manifest_sha256,
        "stage_source_head": stage_source_head,
    }


@dataclass
class SafetyConfig:
    min_balance: float = DEFAULT_MIN_BALANCE
    max_spend: float = DEFAULT_MAX_SPEND
    max_runtime: float = DEFAULT_MAX_RUNTIME
    poll_seconds: float = DEFAULT_POLL_SECONDS
    max_api_failures: int = DEFAULT_MAX_API_FAILURES
    pending_timeout: float = DEFAULT_PENDING_TIMEOUT
    termination_attempts: int = DEFAULT_TERMINATION_ATTEMPTS
    termination_poll_seconds: float = DEFAULT_TERMINATION_POLL_SECONDS

    def __post_init__(self) -> None:
        try:
            numeric = tuple(float(value) for value in
                            (self.min_balance, self.max_spend, self.max_runtime,
                             self.poll_seconds, self.pending_timeout,
                             self.termination_poll_seconds))
        except (TypeError, ValueError) as exc:
            raise ValueError("safety limits must be finite numbers") from exc
        (self.min_balance, self.max_spend, self.max_runtime,
         self.poll_seconds, self.pending_timeout,
         self.termination_poll_seconds) = numeric
        if (not all(math.isfinite(value) for value in numeric)
                or self.min_balance < 0 or self.max_spend <= 0 or self.max_runtime <= 0):
            raise ValueError("safety limits must be positive (balance floor may be zero)")
        if self.min_balance < HARD_MIN_BALANCE:
            raise ValueError(f"balance floor cannot be below ${HARD_MIN_BALANCE:.2f}")
        if self.max_spend > HARD_MAX_SPEND:
            raise ValueError(f"maximum spend cannot exceed ${HARD_MAX_SPEND:.2f}")
        if self.max_runtime > HARD_MAX_RUNTIME:
            raise ValueError(
                f"maximum runtime cannot exceed {int(HARD_MAX_RUNTIME)} seconds"
            )
        if (isinstance(self.max_api_failures, bool)
                or not isinstance(self.max_api_failures, int)
                or self.poll_seconds <= 0 or self.max_api_failures < 1
                or self.pending_timeout <= 0):
            raise ValueError("invalid supervisor timing")
        if (isinstance(self.termination_attempts, bool)
                or not isinstance(self.termination_attempts, int)
                or self.termination_attempts < TERMINATION_ABSENCE_CONFIRMATIONS
                or self.termination_poll_seconds < 0):
            raise ValueError("invalid termination policy")


@dataclass
class LeaseSupervisor:
    client: RunPodClient
    state_file: Path
    config: SafetyConfig = field(default_factory=SafetyConfig)
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time
    monotonic: Callable[[], float] = time.monotonic
    output: Callable[[str], None] = print

    def __post_init__(self) -> None:
        self.state_file = Path(self.state_file)
        self._consecutive_api_failures = 0

    def _state(self) -> dict[str, Any]:
        state = _read_json(self.state_file)
        try:
            run_id = validate_run_id(state["run_id"])
            attempt_id = validate_run_id(state.get("attempt_id", run_id))
            if attempt_id != run_id:
                raise ValueError("lease attempt identity is not bound to run identity")
            lease_alias = validate_run_id(state.get("lease_attempt_id", attempt_id))
            if lease_alias != attempt_id:
                raise ValueError("lease attempt alias is not bound to run identity")
            if state["pod_name"] != _name_for(attempt_id):
                raise ValueError("pod name is not bound to the run identity")
            artifact_run_id = validate_run_id(state.get("artifact_run_id", run_id))
            lineage_alias = validate_run_id(state.get("artifact_lineage_id", artifact_run_id))
            if lineage_alias != artifact_run_id:
                raise ValueError("artifact lineage alias is not bound to the run identity")
            _validate_launch_nonce(state["launch_nonce"])
            if state.get("stage_manifest_sha256") is not None:
                _validate_stage_manifest_digest(state["stage_manifest_sha256"])
            if state.get("stage_source_head") is not None:
                _validate_source_head(state["stage_source_head"])
            if state.get("deployment_started") is True:
                deadline = state.get("provider_terminate_after")
                if not isinstance(deadline, str) or PROVIDER_DATETIME_RE.fullmatch(deadline) is None:
                    raise ValueError("provider termination deadline is missing")
                expected_worker = int(datetime.strptime(
                    deadline, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc).timestamp()) - WORKER_DEADLINE_MARGIN_SECONDS
                worker_deadline = state.get("worker_deadline_epoch")
                if (isinstance(worker_deadline, bool)
                        or not isinstance(worker_deadline, int)
                        or worker_deadline != expected_worker):
                    raise ValueError("worker deadline is not bound to provider termination")
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetyError("lease identity is malformed") from exc
        return state

    def _fallback_identity(self) -> dict[str, Any]:
        """Recover only the exact run name when durable state is unreadable.

        A monitor must still attempt cleanup after a truncated state write or
        a restart.  We intentionally do not recover a pod id from malformed
        JSON: the run-bound name is the only identity that can be reconstructed
        without trusting partial state.
        """

        _reject_state_symlinks(self.state_file)
        candidate: str | None = None
        try:
            raw = json.loads(self.state_file.read_text())
            if isinstance(raw, dict):
                value = raw.get("run_id")
                if isinstance(value, str):
                    candidate = value
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        if candidate is None:
            value = self.state_file.parent.name
            try:
                candidate = validate_run_id(value)
            except ValueError as exc:
                raise SafetyError("cannot recover an exact run identity for cleanup") from exc
        else:
            candidate = validate_run_id(candidate)
        return {"run_id": candidate, "pod_name": _name_for(candidate), "status": "unknown"}

    @staticmethod
    def _merge_nonterminal_state(
        current: Mapping[str, Any], incoming: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Merge a stale-safe lease update while holding the state lock.

        The creator and detached monitor legitimately write disjoint fields.
        Replacing the document with either side's older snapshot can erase a
        pod binding, SSH endpoint, or monitor-unit identity.  Preserve fields
        the incoming snapshot never observed, reject changes to write-once
        lease identities, and make lifecycle/cost evidence monotonic.
        """

        write_once = {
            "schema_version", "run_id", "attempt_id", "lease_attempt_id",
            "artifact_run_id", "artifact_lineage_id", "pod_name", "pod_id",
            "machine_id", "launch_nonce", "monitor_nonce",
            "runpod_key_sha256", "stage_manifest_sha256", "stage_source_head",
            "provider_terminate_after", "worker_deadline_epoch", "image", "gpu",
            "monitor_unit", "ssh_ip", "ssh_port", "provider_gpu_display_name",
        }
        for field in write_once:
            before = current.get(field)
            after = incoming.get(field)
            if before is not None and after is not None and before != after:
                raise IdentityMismatch(
                    f"concurrent state update changed write-once lease field: {field}"
                )

        merged = dict(current)
        for field, value in incoming.items():
            # A stale snapshot commonly contains the pre-binding null.  Null
            # is absence for write-once identities, never an instruction to
            # erase a value another writer has already bound.
            if field in write_once and current.get(field) is not None and value is None:
                continue
            merged[field] = value

        status_rank = {"unknown": 0, "pending": 1, "active": 2, "terminated": 3}
        old_status = current.get("status")
        new_status = incoming.get("status")
        if old_status in status_rank and new_status in status_rank:
            merged["status"] = (
                old_status
                if status_rank[old_status] > status_rank[new_status]
                else new_status
            )
        if "deployment_started" in current or "deployment_started" in incoming:
            merged["deployment_started"] = bool(
                current.get("deployment_started") or incoming.get("deployment_started")
            )
        if "termination_verified" in current or "termination_verified" in incoming:
            merged["termination_verified"] = bool(
                current.get("termination_verified") or incoming.get("termination_verified")
            )

        for field in (
            "bound_at", "monitor_heartbeat_at", "last_heartbeat_at",
            "last_api_error_at", "termination_attempted_at", "terminated_at",
            "estimated_spend", "elapsed", "api_failures",
        ):
            values = [value for value in (current.get(field), incoming.get(field))
                      if isinstance(value, (int, float)) and not isinstance(value, bool)
                      and math.isfinite(float(value))]
            if values:
                merged[field] = max(values)
        balances = [value for value in (current.get("balance"), incoming.get("balance"))
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value))]
        if balances:
            # A stale lower balance is conservative; it cannot hide a floor
            # breach by replacing a newer low observation with an older high.
            merged["balance"] = min(balances)
        protected = {
            str(value)
            for source in (current, incoming)
            for value in (source.get("protected_pod_ids") or [])
        }
        if protected:
            merged["protected_pod_ids"] = sorted(protected)

        monitor_rank = {
            "arming": 0, "READY": 1, "monitoring": 1, "healthy": 2,
            "terminating": 3, "terminated": 4,
        }
        old_monitor = current.get("monitor_status")
        new_monitor = incoming.get("monitor_status")
        if old_monitor in monitor_rank and new_monitor in monitor_rank:
            merged["monitor_status"] = (
                old_monitor
                if monitor_rank[old_monitor] > monitor_rank[new_monitor]
                else new_monitor
            )
        if merged.get("termination_verified") is True \
                and merged.get("status") == "terminated":
            merged.pop("termination_error", None)
        return merged

    def _save(self, state: Mapping[str, Any]) -> None:
        _reject_state_symlinks(self.state_file)
        lock_path = self.state_file.with_name(f".{self.state_file.name}.lock")
        _reject_state_symlinks(lock_path)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            lock_fd = os.open(lock_path, flags, 0o600)
        except OSError:
            raise SafetyError("cannot lock durable lease state") from None
        try:
            lock_stat = os.fstat(lock_fd)
            if (not stat.S_ISREG(lock_stat.st_mode)
                    or stat.S_IMODE(lock_stat.st_mode) != 0o600):
                raise SafetyError("durable lease-state lock is unsafe")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if not self.state_file.exists() or not self.state_file.is_file():
                raise SafetyError(f"lease state disappeared: {self.state_file}")
            current = _read_json(self.state_file)
            if (current.get("status") == "terminated"
                    and current.get("termination_verified") is True):
                identity_fields = (
                    "run_id", "artifact_run_id", "pod_name", "pod_id",
                    "machine_id", "launch_nonce",
                )
                if any(
                    current.get(field) != state.get(field)
                    for field in identity_fields
                    if current.get(field) is not None or state.get(field) is not None
                ):
                    raise IdentityMismatch(
                        "stale state write does not match the terminated lease"
                    )
                # Terminal verified state is monotonic. A detached monitor
                # holding an older heartbeat can never resurrect the lease.
                return
            merged = self._merge_nonterminal_state(current, state)
            _atomic_json(self.state_file, merged)
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

    @staticmethod
    def _exact_pod(state: Mapping[str, Any], pods: list[Mapping[str, Any]]) -> dict[str, Any] | None:
        pod_id = state.get("pod_id")
        pod_name = state.get("pod_name")
        matches = [p for p in pods if p.get("id") == pod_id]
        if len(matches) > 1:
            raise IdentityMismatch("multiple pods returned for the leased id")
        if not matches:
            return None
        pod = dict(matches[0])
        if pod.get("name") != pod_name:
            raise IdentityMismatch("pod id resolved to a different name")
        expected_machine = state.get("machine_id")
        observed_machine = pod.get("machineId")
        if expected_machine is not None:
            try:
                expected_machine = _validate_machine_id(expected_machine)
                observed_machine = _validate_machine_id(observed_machine)
            except SafetyError as exc:
                raise IdentityMismatch(
                    "pod id resolved without a valid bound machine"
                ) from exc
            if observed_machine != expected_machine:
                raise IdentityMismatch("pod id resolved to a different machine")
        return pod

    @staticmethod
    def _named_pods(state: Mapping[str, Any], pods: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        name = state.get("pod_name")
        return [dict(p) for p in pods if p.get("name") == name]

    def _discover_pending(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if state.get("deployment_started") is not True:
            return None
        pods = self.client.pods()
        matches = self._named_pods(state, pods)
        if len(matches) > 1:
            # A unique run name should never have two pods.  Treat this as an
            # identity breach; the caller's finally path will attempt every
            # exact-name id and verify both disappear.
            raise IdentityMismatch("more than one pod has the pending lease name")
        if not matches:
            return None
        pod = matches[0]
        if not pod.get("id"):
            raise IdentityMismatch("discovered pod has no id")
        protected = {str(value) for value in (state.get("protected_pod_ids") or [])}
        if str(pod["id"]) in protected:
            raise IdentityMismatch("pending lease resolved to a pre-existing pod")
        gpu_display_name = _optional_assigned_b300(pod)
        state.update({"pod_id": pod["id"],
                      "cost_per_hr": float(pod.get("costPerHr") or 0), "status": "active",
                      "bound_at": self.clock(),
                      "provider_gpu_display_name": gpu_display_name})
        discovered_machine = pod.get("machineId")
        if discovered_machine is not None:
            state["machine_id"] = _validate_machine_id(discovered_machine)
        self._save(state)
        return pod

    def _terminate_exact_name(self, state: Mapping[str, Any]) -> bool:
        """Best-effort cleanup for a create response lost to API uncertainty."""

        run_id = state.get("run_id")
        pod_name = state.get("pod_name")
        if not isinstance(run_id, str) or not isinstance(pod_name, str):
            raise SafetyError("cannot terminate without a run-bound pod name")
        if pod_name != _name_for(run_id):
            raise IdentityMismatch("refusing to terminate an unbound pod name")
        # A monitor can be started before deploy is armed.  If that launch
        # fails, it has no ownership of a same-name pod that may appear from a
        # concurrent actor; never use the pending name as a termination key.
        if state.get("deployment_started") is not True:
            return True
        protected = {
            str(value) for value in (state.get("protected_pod_ids") or [])
            if isinstance(value, (str, int))
        }
        last_error: BaseException | None = None
        consecutive_absences = 0
        for attempt in range(self.config.termination_attempts):
            try:
                pods = self.client.pods()
                named = self._named_pods(state, pods)
                protected_matches = [p for p in named if str(p.get("id")) in protected]
                matches = [p for p in named if str(p.get("id")) not in protected]
                if protected_matches and not matches:
                    # The only exact-name object is known to predate this
                    # attempt.  It is not ours to terminate.
                    raise IdentityMismatch("exact lease name belongs to a pre-existing pod")
                if len(matches) > 1:
                    consecutive_absences = 0
                    # A lost create response can race with duplicate
                    # visibility. Attempt every exact-name identity before
                    # deciding that cleanup is unverified.
                    failures = []
                    for pod in matches:
                        pod_id = pod.get("id")
                        if not pod_id:
                            failures.append(IdentityMismatch("exact-name pod has no id"))
                            continue
                        try:
                            self._terminate_verified(state, pod_id=pod_id, require_name=True)
                        except SafetyError as exc:
                            failures.append(exc)
                    if failures:
                        last_error = failures[-1]
                elif matches:
                    consecutive_absences = 0
                    pod_id = matches[0].get("id")
                    if not pod_id:
                        last_error = IdentityMismatch("exact-name pod has no id")
                    else:
                        try:
                            self._terminate_verified(state, pod_id=pod_id, require_name=True)
                        except SafetyError as exc:
                            last_error = exc
                else:
                    if protected_matches:
                        raise IdentityMismatch("pre-existing pod remains under the lease name")
                    consecutive_absences += 1
                    if consecutive_absences >= TERMINATION_ABSENCE_CONFIRMATIONS:
                        return True
            except (APIUncertain, OSError, TimeoutError) as exc:
                last_error = exc
                consecutive_absences = 0
            if attempt + 1 < self.config.termination_attempts and self.config.termination_poll_seconds:
                self.sleep(self.config.termination_poll_seconds)
        raise TerminationUnverified("could not verify termination of the exact lease name") from last_error

    def _terminate_verified(self, state: Mapping[str, Any], *, pod_id: str | None = None,
                            require_name: bool = True) -> None:
        target = pod_id or state.get("pod_id")
        if not target:
            return
        if require_name:
            try:
                initial_matches = [p for p in self.client.pods() if p.get("id") == target]
                if len(initial_matches) > 1:
                    raise IdentityMismatch("multiple pods returned for the termination id")
            except (APIUncertain, OSError, TimeoutError) as exc:
                raise APIUncertain("cannot verify termination identity before mutation") from exc
            if initial_matches and initial_matches[0].get("name") != state.get("pod_name"):
                raise IdentityMismatch("refusing to terminate a pod with a different identity")
        last_error: BaseException | None = None
        consecutive_absences = 0
        for attempt in range(self.config.termination_attempts):
            try:
                self.client.terminate(str(target))
            except (APIUncertain, OSError, TimeoutError) as exc:
                last_error = exc
            except SafetyError:
                raise
            try:
                pods = self.client.pods()
                exact = [p for p in pods if p.get("id") == target]
                if len(exact) > 1:
                    raise IdentityMismatch("multiple pods returned for the termination id")
                if exact:
                    consecutive_absences = 0
                    if require_name and any(p.get("name") != state.get("pod_name") for p in exact):
                        raise IdentityMismatch("termination lookup found a different pod identity")
                else:
                    if require_name and any(p.get("name") == state.get("pod_name") for p in pods):
                        raise IdentityMismatch("termination id disappeared while another same-name pod remained")
                    consecutive_absences += 1
                    if consecutive_absences >= TERMINATION_ABSENCE_CONFIRMATIONS:
                        self.output(
                            f"terminated and verified pod {target} "
                            f"({consecutive_absences} consecutive absences)"
                        )
                        return
            except (APIUncertain, OSError, TimeoutError) as exc:
                last_error = exc
                consecutive_absences = 0
            if attempt + 1 < self.config.termination_attempts and self.config.termination_poll_seconds:
                self.sleep(self.config.termination_poll_seconds)
        raise TerminationUnverified(f"could not verify termination of pod {target}") from last_error

    def terminate_verified(self) -> None:
        state_error: BaseException | None = None
        try:
            state = self._state()
        except BaseException as exc:
            state_error = exc
            state = self._fallback_identity()
        verified = False
        try:
            if state.get("pod_id"):
                self._terminate_verified(state)
            else:
                self._terminate_exact_name(state)
            verified = True
        finally:
            # Do not claim success when the API never proved absence.  The
            # state is still durable so a later recovery process can retry.
            state = dict(state)
            state["termination_attempted_at"] = self.clock()
            state["termination_verified"] = verified
            if verified:
                state["status"] = "terminated"
            try:
                self._save(state)
            except OSError:
                pass
        if state_error is not None:
            raise state_error

    def _authenticated_ready(self, state: dict[str, Any]) -> dict[str, Any]:
        """Perform the pre-deployment monitor handshake and acknowledge it."""

        nonce = state.get("monitor_nonce")
        fingerprint = state.get("runpod_key_sha256")
        if (not isinstance(nonce, str) or not nonce
                or any(char in nonce for char in "\r\n\x00")
                or not isinstance(fingerprint, str)
                or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)):
            raise SafetyError("monitor handshake identity is missing or malformed")
        if state.get("status") != "pending" \
                or state.get("termination_verified") is True \
                or state.get("termination_attempted_at") is not None \
                or state.get("terminated_at") is not None \
                or any(key in state for key in (
                    "termination_error", "monitor_error", "error")):
            raise SafetyError("monitor handshake cannot acknowledge a terminating lease")
        if state.get("deployment_started") is not False:
            raise SafetyError("monitor handshake requires a pending deployment")
        if isinstance(self.client, RunPodClient):
            client_fingerprint = self.client.credential_fingerprint
            if client_fingerprint != fingerprint:
                raise SafetyError("monitor credential does not match the lease fingerprint")

        try:
            balance = self.client.balance()
            if isinstance(balance, bool):
                raise ValueError("boolean balance")
            balance = float(balance)
            if not math.isfinite(balance):
                raise ValueError("non-finite balance")
            pods = self.client.pods()
        except (APIUncertain, OSError, TimeoutError):
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise APIUncertain("monitor handshake response was malformed") from exc
        if (not isinstance(pods, list)
                or any(not isinstance(item, Mapping) for item in pods)):
            raise APIUncertain("monitor handshake pod listing was malformed")
        pod_ids: list[str] = []
        for item in pods:
            pod_id = item.get("id")
            name = item.get("name")
            if (isinstance(pod_id, bool) or not isinstance(pod_id, (str, int))
                    or not str(pod_id) or not isinstance(name, str) or not name):
                raise APIUncertain("monitor handshake pod identity was malformed")
            pod_ids.append(str(pod_id))
        if len(pod_ids) != len(set(pod_ids)):
            raise IdentityMismatch("monitor handshake returned duplicate pod identities")

        protected = {str(value) for value in (state.get("protected_pod_ids") or [])}
        for item in pods:
            name = str(item.get("name", ""))
            if name.startswith(NAME_PREFIX) and str(item.get("id")) not in protected:
                raise SafetyError("a JLens pod appeared before monitor readiness")

        # Do not acknowledge a stale snapshot after the creator or a cleanup
        # path has advanced the durable lease.  Preserve fields (such as the
        # returned systemd unit) written while the API checks were in flight.
        latest = self._state()
        if (latest.get("monitor_nonce") != nonce
                or latest.get("runpod_key_sha256") != fingerprint
                or latest.get("status") != "pending"
                or latest.get("deployment_started") is not False
                or latest.get("termination_verified") is True
                or latest.get("termination_attempted_at") is not None
                or latest.get("terminated_at") is not None
                or any(key in latest for key in (
                    "termination_error", "monitor_error", "error"))):
            raise SafetyError("lease changed while monitor handshake was in flight")
        heartbeat = self.clock()
        if not isinstance(heartbeat, (int, float)) or isinstance(heartbeat, bool) \
                or not math.isfinite(float(heartbeat)):
            raise SafetyError("monitor heartbeat is not finite")
        acknowledged = dict(latest)
        acknowledged.update({
            "monitor_status": "READY",
            "monitor_ready_nonce": nonce,
            "monitor_ready_fingerprint": fingerprint,
            "monitor_heartbeat_at": float(heartbeat),
            "last_heartbeat_at": float(heartbeat),
            "monitor_balance": balance,
        })
        # _save uses a temp file + fsync + replace, so the acknowledgement is
        # one atomic state transition and cannot expose a partial READY pair.
        self._save(acknowledged)
        return acknowledged

    def monitor(self, *, once: bool = False) -> None:
        """Run the monitor until a budget/runtime breach or remote completion.

        The monitor intentionally does not infer completion from SSH status.
        A separate supervised ``run`` command (or an operator) calls
        ``terminate_verified`` after the remote job exits; this process owns
        the hard spending limits in the meantime.
        """

        state_error: BaseException | None = None
        try:
            state = self._state()
        except BaseException as exc:
            state_error = exc
            state = self._fallback_identity()
        # A systemd restart after a completed cleanup must be a no-op.  In
        # particular, do this before arming the ``finally`` cleanup scope so
        # Restart=on-failure cannot turn a verified termination into a fresh
        # API/termination loop.
        if state_error is None and state.get("status") == "terminated" \
                and state.get("termination_verified") is True:
            return
        start_mono = self.monotonic()
        state = dict(state)
        state.setdefault("monitor_started_at", self.clock())
        handshake_required = state.get("monitor_nonce") is not None
        if not handshake_required:
            state["monitor_status"] = "monitoring"
        # Initialize a conservative fallback before entering the cleanup
        # scope. A malformed persisted timestamp then still reaches the
        # termination path instead of failing before ``finally`` is armed.
        state_bound_at = self.clock()
        try:
            if state_error is not None:
                raise state_error
            # Keep the first persistence operation inside the cleanup scope:
            # even a full disk or a state-write race must not skip the
            # termination attempt for an already-bound pod.
            state_bound_at = float(state.get("bound_at", state.get("created_at", state_bound_at)))
            if not math.isfinite(state_bound_at):
                raise SafetyError("lease binding timestamp is not finite")
            if handshake_required:
                state = self._authenticated_ready(state)
            else:
                self._save(state)
            while True:
                state = self._state()
                if not state.get("pod_id"):
                    created = float(state.get("created_at", self.clock()))
                    if self.clock() - created > self.config.pending_timeout:
                        raise SafetyError("pending lease expired without an exact pod identity")
                    try:
                        pod = self._discover_pending(state)
                        self._consecutive_api_failures = 0
                    except (APIUncertain, OSError, TimeoutError) as exc:
                        self._consecutive_api_failures += 1
                        if self._consecutive_api_failures >= self.config.max_api_failures:
                            raise APIUncertain("repeated API uncertainty while binding lease") from exc
                        self.sleep(self.config.poll_seconds)
                        continue
                    if pod is None:
                        if once:
                            return
                        self.sleep(self.config.poll_seconds)
                        continue
                    state = self._state()
                try:
                    balance = self.client.balance()
                    pods = self.client.pods()
                    pod = self._exact_pod(state, pods)
                    if pod is None:
                        raise IdentityMismatch("leased pod is absent or has changed identity")
                    self._consecutive_api_failures = 0
                except (APIUncertain, OSError, TimeoutError) as exc:
                    self._consecutive_api_failures += 1
                    state = dict(state)
                    state["last_api_error_at"] = self.clock()
                    state["api_failures"] = self._consecutive_api_failures
                    self._save(state)
                    if self._consecutive_api_failures >= self.config.max_api_failures:
                        raise APIUncertain("repeated API uncertainty while supervising lease") from exc
                    self.sleep(self.config.poll_seconds)
                    continue

                # Wall-clock binding time survives a monitor restart; the
                # monotonic component protects the current process from clock
                # corrections.  A restart must not reset the paid runtime.
                elapsed = max(0.0, self.monotonic() - start_mono, self.clock() - state_bound_at)
                uptime = float((pod.get("runtime") or {}).get("uptimeInSeconds") or 0)
                cost = float(pod.get("costPerHr") or state.get("cost_per_hr") or 0)
                if not math.isfinite(uptime) or uptime < 0 or not math.isfinite(cost) or cost <= 0:
                    raise SafetyError("pod cost is absent; refusing to supervise unknown spend")
                balance_before = float(state.get("balance_before", balance))
                if not math.isfinite(balance_before):
                    raise SafetyError("starting balance is not finite")
                balance_spend = max(0.0, balance_before - balance)
                estimated_spend = max(cost * max(elapsed, uptime) / 3600.0, balance_spend)
                if balance < self.config.min_balance:
                    raise SafetyError(f"balance floor breached: ${balance:.2f} < ${self.config.min_balance:.2f}")
                desired = str(pod.get("desiredStatus", "")).upper()
                if desired in {"TERMINATED", "EXITED", "FAILED", "CANCELLED"}:
                    raise SafetyError(f"leased pod entered terminal state {desired}")
                if estimated_spend >= self.config.max_spend:
                    raise SafetyError(f"max spend breached: ${estimated_spend:.2f} >= ${self.config.max_spend:.2f}")
                if max(elapsed, uptime) >= self.config.max_runtime:
                    raise SafetyError("maximum runtime breached")
                state = dict(state)
                state.update({"last_heartbeat_at": self.clock(), "balance": balance,
                              "estimated_spend": estimated_spend, "elapsed": elapsed,
                              "monitor_status": "READY" if handshake_required else "healthy"})
                self._save(state)
                if once:
                    return
                self.sleep(self.config.poll_seconds)
        finally:
            # A monitor that exits for *any* reason owns cleanup.  No caller
            # may mistake an exception, disconnect, or API timeout for a safe
            # lease release.
            cleanup_state = dict(state)
            try:
                # Prefer the newest durable identity, but retain the last
                # known-good in-memory identity if the file was concurrently
                # truncated or otherwise became unreadable.
                cleanup_state = self._state()
            except BaseException:
                cleanup_state = dict(state)
            cleanup_state = dict(cleanup_state)
            cleanup_state["monitor_status"] = "terminating"
            try:
                self._save(cleanup_state)
            except BaseException:
                # Persistence failure cannot justify skipping cleanup.
                pass
            termination_error: BaseException | None = None
            try:
                if cleanup_state.get("pod_id"):
                    self._terminate_verified(cleanup_state)
                else:
                    self._terminate_exact_name(cleanup_state)
            except BaseException as exc:
                termination_error = exc
            finally:
                cleanup_state["termination_verified"] = termination_error is None
                if termination_error is None:
                    cleanup_state["status"] = "terminated"
                    cleanup_state.pop("termination_error", None)
                else:
                    cleanup_state["termination_error"] = str(termination_error)
                cleanup_state["terminated_at"] = self.clock()
                try:
                    self._save(cleanup_state)
                except BaseException as save_error:
                    if termination_error is None:
                        termination_error = save_error
            if termination_error is not None:
                raise termination_error


def systemd_user_preflight(*, runner: Callable[..., Any] | None = None) -> None:
    """Fail closed unless a user systemd manager can host the monitor."""

    runner = runner or subprocess.run
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not user or any(c in user for c in "\r\n"):
        raise SafetyError("current user is unknown; refusing paid launch")
    commands = (["systemctl", "--user", "show-environment"],
                ["systemctl", "--user", "--version"],
                ["loginctl", "show-user", user, "--property=Linger", "--value"])
    for index, command in enumerate(commands):
        try:
            result = runner(command, check=False, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            raise SafetyError("user systemd is unavailable; refusing paid launch") from exc
        if getattr(result, "returncode", None) != 0:
            raise SafetyError("user systemd preflight failed; refusing paid launch")
        if index == 2 and (getattr(result, "stdout", "") or "").strip().lower() != "yes":
            raise SafetyError("user lingering is disabled; refusing disconnect-vulnerable launch")


def verify_systemd_monitor_active(unit: str, *, runner: Callable[..., Any] | None = None) -> None:
    """Require an already-running monitor immediately before deployment."""

    if not isinstance(unit, str) or not re.fullmatch(r"[A-Za-z0-9_.:@-]{1,255}", unit):
        raise SafetyError("monitor unit identity is malformed")
    runner = runner or subprocess.run
    try:
        result = runner(["systemctl", "--user", "is-active", unit], check=False,
                        capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError("could not verify the lease monitor is active") from exc
    if (getattr(result, "returncode", None) != 0
            or (getattr(result, "stdout", "") or "").strip() != "active"):
        raise SafetyError("lease monitor is not active immediately before deployment")


def _require_monitor_ready(state: Mapping[str, Any], *, nonce: str,
                           fingerprint: str | None, now: float | None = None) -> None:
    """Accept only the authenticated, fresh READY acknowledgement."""

    if (not isinstance(nonce, str) or not nonce
            or any(char in nonce for char in "\r\n\x00")):
        raise SafetyError("lease has no monitor nonce for readiness")
    if (not isinstance(fingerprint, str)
            or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)):
        raise SafetyError("lease has no credential fingerprint for monitor readiness")
    if state.get("monitor_status") != "READY":
        raise SafetyError("disconnect-safe monitor did not provide an authenticated READY state")
    if state.get("monitor_nonce") != nonce \
            or state.get("runpod_key_sha256") != fingerprint:
        raise SafetyError("monitor READY identity does not match the lease")
    if state.get("monitor_ready_nonce") != nonce:
        raise SafetyError("monitor READY nonce is missing or inconsistent")
    if state.get("monitor_ready_fingerprint") != fingerprint:
        raise SafetyError("monitor READY fingerprint is missing or inconsistent")
    if state.get("status") != "pending" or state.get("deployment_started") is not False:
        raise SafetyError("monitor READY state is not a pending lease")
    if (state.get("termination_verified") is True
            or state.get("termination_attempted_at") is not None
            or state.get("terminated_at") is not None
            or any(key in state for key in (
                "termination_error", "monitor_error", "error"))):
        raise SafetyError("monitor READY state contains termination/error information")
    heartbeat_values = [state.get(name) for name in ("monitor_heartbeat_at", "last_heartbeat_at")
                        if state.get(name) is not None]
    if not heartbeat_values:
        raise SafetyError("monitor READY state has no heartbeat")
    try:
        heartbeat = float(heartbeat_values[0])
        if any(float(value) != heartbeat for value in heartbeat_values[1:]):
            raise ValueError("heartbeat fields disagree")
        current = time.time() if now is None else float(now)
        created = float(state.get("created_at", heartbeat))
    except (TypeError, ValueError) as exc:
        raise SafetyError("monitor READY heartbeat is malformed") from exc
    if (not math.isfinite(heartbeat) or not math.isfinite(current)
            or not math.isfinite(created)
            or heartbeat < created - 1.0
            or heartbeat > current + 1.0
            or current - heartbeat > MONITOR_READY_MAX_AGE):
        raise SafetyError("monitor READY heartbeat is stale")


def _systemd_safe_absolute(path: str | Path, label: str) -> str:
    value = os.path.abspath(os.fspath(path))
    if re.fullmatch(r"/[A-Za-z0-9_./:@+-]+", value) is None:
        raise SafetyError(f"{label} path is not safe for a persistent systemd unit")
    return value


def _monitor_unit_path(unit: str, unit_dir: Path | None = None) -> Path:
    if not isinstance(unit, str) or re.fullmatch(r"ouro-jlens-monitor-[A-Za-z0-9_.-]+\.service", unit) is None:
        raise SafetyError("monitor unit identity is malformed")
    directory = Path(unit_dir) if unit_dir is not None else SYSTEMD_USER_UNIT_DIR
    return directory / unit


def start_systemd_monitor(state_file: Path, *, python: str = sys.executable,
                          script: Path | None = None, runner: Callable[..., Any] | None = None,
                          unit_dir: Path | None = None) -> str:
    """Install, enable, and start a reboot-persistent per-attempt monitor."""

    state_file = Path(state_file)
    _reject_state_symlinks(state_file)
    state_file = state_file.absolute()
    state = _read_json(state_file)
    run_id = validate_run_id(str(state["run_id"]))
    runner = runner or subprocess.run
    if "monitor_bundle" in state:
        bundle_root = _verify_monitor_bundle(state_file, state)
    else:
        source = Path(script).resolve().parent if script is not None else Path(__file__).resolve().parent
        metadata = _install_monitor_bundle(state_file, source=source)
        state = dict(state)
        state["monitor_bundle"] = metadata
        _atomic_json(state_file, state)
        bundle_root = _verify_monitor_bundle(state_file, state)
    systemd_user_preflight(runner=runner)
    unit = f"ouro-jlens-monitor-{run_id}.service"
    unit_path = _monitor_unit_path(unit, unit_dir)
    python_path = _systemd_safe_absolute(python, "Python executable")
    bundle_path = _systemd_safe_absolute(bundle_root, "monitor bundle")
    script_path = _systemd_safe_absolute(bundle_root / "ouro_jlens" / "pod.py", "monitor script")
    state_path_value = _systemd_safe_absolute(state_file, "lease state")
    unit_data = (
        "[Unit]\n"
        f"Description=Ouro JLens lease monitor {run_id}\n"
        "Wants=network-online.target\n"
        "After=network-online.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        "UnsetEnvironment=RUNPOD_API_KEY RUNPOD_API_KEY_SHA256\n"
        f"Environment=PYTHONPATH={bundle_path}\n"
        f"ExecStart={python_path} {script_path} monitor --state {state_path_value}\n"
        "Restart=on-failure\n"
        "RestartSec=5s\n"
        "TimeoutStopSec=30s\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    ).encode("utf-8")
    try:
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        _write_immutable(unit_path, unit_data)
    except (OSError, PublishError) as exc:
        raise SafetyError("could not install reboot-persistent lease monitor") from exc
    try:
        reload_result = runner(["systemctl", "--user", "daemon-reload"], check=False,
                               capture_output=True, text=True, timeout=30)
        result = runner(["systemctl", "--user", "enable", "--now", unit], check=False,
                        capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        if unit_path.is_file() and not unit_path.is_symlink():
            unit_path.unlink()
        raise SafetyError("could not start reboot-persistent monitor") from exc
    if (getattr(reload_result, "returncode", None) != 0
            or getattr(result, "returncode", None) != 0):
        if unit_path.is_file() and not unit_path.is_symlink():
            unit_path.unlink()
        raise SafetyError("systemd did not enable the reboot-persistent monitor")
    try:
        enabled = runner(["systemctl", "--user", "is-enabled", unit], check=False,
                         capture_output=True, text=True, timeout=10)
        check = runner(["systemctl", "--user", "show", unit,
                        "--property=ActiveState", "--value"], check=False,
                       capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError("could not verify the disconnect-safe monitor") from exc
    if (getattr(enabled, "returncode", None) != 0
            or (getattr(enabled, "stdout", "") or "").strip() != "enabled"
            or getattr(check, "returncode", None) != 0
            or (getattr(check, "stdout", "") or "").strip() not in {"active", "activating"}):
        try:
            runner(["systemctl", "--user", "disable", "--now", unit], check=False,
                   capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass
        if unit_path.is_file() and not unit_path.is_symlink():
            unit_path.unlink()
        raise SafetyError("disconnect-safe monitor is not active")
    return unit


def wait_for_ssh(client: RunPodClient, pod_id: str, *, timeout: float = 900,
                 poll_seconds: float = 15, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 output: Callable[[str], None] = print) -> dict[str, Any]:
    """Wait for SSH while preserving the lease's exact identity."""

    if not isinstance(pod_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
        raise SafetyError("invalid pod id")
    timeout = _finite_number(timeout, "ssh timeout", minimum=0, strict_minimum=True)
    poll_seconds = _finite_number(poll_seconds, "SSH poll interval", minimum=0, strict_minimum=True)
    started = clock()
    while clock() - started < timeout:
        pods = client.pods()
        pod = next((p for p in pods if p.get("id") == pod_id), None)
        if pod is None:
            raise IdentityMismatch("leased pod disappeared while waiting for SSH")
        raw_ports = ((pod.get("runtime") or {}).get("ports") or [])
        if not isinstance(raw_ports, list):
            raise SafetyError("RunPod returned malformed runtime ports")
        ports = [x for x in raw_ports
                 if isinstance(x, Mapping) and x.get("privatePort") == 22 and x.get("isIpPublic")]
        output(f"  {int(clock() - started):>4}s status={pod.get('desiredStatus')} ssh={'yes' if ports else 'no'}")
        if ports:
            endpoint = ports[0]
            host = endpoint.get("ip")
            try:
                public_port = int(endpoint.get("publicPort"))
            except (TypeError, ValueError) as exc:
                raise SafetyError("RunPod returned an invalid SSH port") from exc
            try:
                host = _validate_ssh_host(host)
                public_port = _validate_ssh_port(public_port)
            except SafetyError as exc:
                raise SafetyError("RunPod returned an unsafe SSH endpoint") from exc
            return {"pod": pod, "port": {**endpoint, "ip": host, "publicPort": public_port}}
        sleep(poll_seconds)
    raise SafetyError("pod did not expose SSH within the safety timeout")


# Compatibility helpers retain the old read/inspect surface while routing all
# API work through the fail-closed client.  They intentionally do not create
# or terminate anything implicitly.
def balance() -> float:
    return RunPodClient().balance()


def _pods() -> list[dict[str, Any]]:
    return RunPodClient().pods()


def _runtime(pod_id: str) -> dict[str, Any] | None:
    return next((pod for pod in _pods() if pod.get("id") == pod_id), None)


def wait(pod_id: str, timeout: float = 900) -> None:
    ready = wait_for_ssh(RunPodClient(), pod_id, timeout=timeout)
    port = ready["port"]
    pod = ready["pod"]
    path = Path("artifacts/jlens/pod.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, {"id": pod_id, "ip": port["ip"], "port": port["publicPort"],
                         "costPerHr": pod.get("costPerHr"), "name": pod.get("name")})


def _terminate(pod_id: str) -> None:
    client = RunPodClient()
    pod = _pod_for_destroy(client, pod_id)
    config = SafetyConfig(min_balance=HARD_MIN_BALANCE, max_spend=DEFAULT_MAX_SPEND,
                          max_runtime=DEFAULT_MAX_RUNTIME, termination_poll_seconds=0)
    state = {"run_id": "destroy", "pod_name": pod.get("name"), "pod_id": pod["id"],
             "machine_id": pod.get("machineId")}
    LeaseSupervisor(client, Path("/dev/null"), config)._terminate_verified(state, pod_id=pod_id)


def _read_launch_secret(path: Path, name: str) -> str:
    try:
        value = path.read_text().strip()
    except OSError as exc:
        raise SafetyError(f"cannot read {name} credential") from exc
    if not value or any(c in value for c in "\r\n"):
        raise SafetyError(f"{name} credential is empty or malformed")
    return value


def _validate_stage_path(value: str) -> str:
    """Require the run to name the locally verified, digest-addressed stage."""

    try:
        path = safe_relative(value)
    except ValueError as exc:
        raise SafetyError("stage path is not a safe repository-relative path") from exc
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", path):
        raise SafetyError("stage path contains shell-significant characters")
    if len(re.findall(r"[0-9a-f]{64}", path)) != 1:
        raise SafetyError("stage path must contain exactly one SHA-256 manifest digest")
    return path


@dataclass(frozen=True)
class StageBootstrap:
    package: Path
    root: Path
    manifest_sha256: str
    source_head: str
    committed_payload_sha256: Mapping[str, str]


def _preflight_results_repository(
    publisher: Any,
    *,
    repository: str,
    attempt_id: str,
    artifact_run_id: str,
    launch_nonce: str,
    stage_manifest_sha256: str,
    stage_source_head: str,
    hf_token_sha256: str,
    attempts: int = 8,
    retry_seconds: float = 0.5,
) -> dict[str, Any]:
    """Prove write, readback, and listing authority before any paid mutation."""

    repository = _validate_hf_repo(repository, "results repository")
    attempt_id = validate_run_id(attempt_id)
    artifact_run_id = validate_run_id(artifact_run_id)
    launch_nonce = _validate_launch_nonce(launch_nonce)
    stage_manifest_sha256 = _validate_stage_manifest_digest(stage_manifest_sha256)
    stage_source_head = _validate_source_head(stage_source_head)
    if (not isinstance(hf_token_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", hf_token_sha256) is None):
        raise SafetyError("HF token fingerprint is malformed")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise SafetyError("results preflight attempts are invalid")
    retry_seconds = _finite_number(
        retry_seconds, "results preflight retry delay", minimum=0
    )
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
    remote_path = (
        f"_jlens_preflight/{artifact_run_id}/{attempt_id}/{digest}.json"
    )
    try:
        publisher.ensure_repository()
        with tempfile.TemporaryDirectory(prefix="jlens-results-preflight-") as temporary:
            source = Path(temporary) / "canary.json"
            downloaded = Path(temporary) / "readback.json"
            _write_immutable(source, data)
            publisher.put_file(source, remote_path)
            publisher.download_file(remote_path, downloaded, expected_size=len(data))
            if _read_regular_file(downloaded) != data:
                raise PublishError("results preflight readback bytes differ")
        prefix = remote_path.rsplit("/", 1)[0] + "/"
        listed = False
        for number in range(attempts):
            paths = publisher.list_files(prefix)
            if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
                raise PublishError("results preflight listing is malformed")
            if remote_path in paths:
                listed = True
                break
            if number + 1 < attempts:
                time.sleep(min(retry_seconds * (2 ** number), 8.0))
        if not listed:
            raise PublishError("results preflight canary did not converge in listing")
    except (OSError, ValueError, TypeError, PublishError) as exc:
        raise SafetyError(
            "results repository write/read/list authority was not proven"
        ) from exc
    return {
        "schema_version": 1,
        "remote_path": remote_path,
        "size": len(data),
        "sha256": digest,
    }


def _prepare_stage_bootstrap(
    stage_path: str,
    staging_repository: str,
    destination: Path,
    *,
    publisher=None,
) -> StageBootstrap:
    """Materialize bootstrap code only from the immutable remote stage.

    This preflight happens before a lease is created.  Copying the controller's
    live checkout would let dirty or post-upload source decide how the paid
    archive is verified and extracted.
    """

    stage_path = _validate_stage_path(stage_path)
    staging_repository = _validate_hf_repo(staging_repository, "staging repository")
    expected_digest = re.findall(r"[0-9a-f]{64}", stage_path)[0]
    destination = Path(destination)
    if destination.is_symlink():
        raise SafetyError("bootstrap destination is a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "stage.tar.gz"
    remote = publisher or HfPublisher(staging_repository)
    try:
        remote.download_file(stage_path, archive)
        verified = verify_stage_tar(archive)
    except (PublishError, OSError, ValueError) as exc:
        raise SafetyError("could not download and verify the immutable stage before leasing") from exc
    if verified.get("manifest_sha256") != expected_digest:
        raise SafetyError("remote stage manifest does not match its digest-addressed path")
    source = verified.get("pinned_inputs", {}).get("source")
    committed = verified.get("pinned_inputs", {}).get("committed_payload_sha256")
    if (
        verified.get("pinned_inputs", {}).get("source_policy") != "clean_git_payload"
        or not isinstance(source, dict)
        or not isinstance(source.get("head"), str)
        or GIT_OID_RE.fullmatch(source["head"]) is None
        or source.get("status") != ""
        or not isinstance(committed, dict)
        or not committed
        or any(
            not isinstance(path, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for path, digest in committed.items()
        )
    ):
        raise SafetyError("remote stage is not bound to clean committed source")
    extracted = destination / "verified"
    try:
        extracted_result = extract_stage_tar(archive, extracted)
    except (PublishError, OSError, ValueError) as exc:
        raise SafetyError("could not safely extract the verified stage bootstrap") from exc
    if extracted_result.get("manifest_sha256") != expected_digest:
        raise SafetyError("extracted bootstrap differs from the verified stage")
    package = extracted / "src" / "ouro_jlens"
    entrypoint = package / "pod_entry.sh"
    if package.is_symlink() or entrypoint.is_symlink() or not entrypoint.is_file():
        raise SafetyError("verified stage does not contain a regular pod entrypoint")
    return StageBootstrap(
        package=package,
        root=extracted,
        manifest_sha256=expected_digest,
        source_head=source["head"],
        committed_payload_sha256=dict(committed),
    )


def _verify_controller_source_matches_stage(bootstrap: StageBootstrap) -> None:
    """Bind the live controller/replay source to the uploaded clean stage."""

    root = Path(__file__).resolve().parents[2]
    relative_paths: list[str] = []
    for raw_path, expected_digest in bootstrap.committed_payload_sha256.items():
        try:
            relative = safe_relative(raw_path)
        except ValueError as exc:
            raise SafetyError("stage committed payload map contains an unsafe path") from exc
        path = root / relative
        try:
            size, digest = _path_digest(path, label="controller stage-bound source")
        except (OSError, PublishError) as exc:
            raise SafetyError(f"stage-bound controller source is unavailable: {relative}") from exc
        if size < 0 or digest != expected_digest:
            raise SafetyError(f"controller source differs from uploaded stage: {relative}")
        relative_paths.append(relative)
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all",
             "--", *sorted(relative_paths)],
            check=False, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError("could not bind controller source to the uploaded stage") from exc
    if head.returncode != 0 or head.stdout.strip() != bootstrap.source_head:
        raise SafetyError("controller HEAD differs from uploaded stage source HEAD")
    if status.returncode != 0 or status.stdout.strip():
        raise SafetyError("controller source changed after the clean stage was uploaded")


def _validate_hf_repo(value: str, name: str) -> str:
    if not isinstance(value, str) or not HF_REPO_RE.fullmatch(value):
        raise SafetyError(f"{name} is not a valid Hugging Face repository")
    return value


def _validate_remote_script(value: str) -> str:
    try:
        path = safe_relative(value)
    except ValueError as exc:
        raise SafetyError("remote script path is not safe") from exc
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", path):
        raise SafetyError("remote script path contains shell-significant characters")
    return path


def _new_state(run_id: str, args: argparse.Namespace, *, artifact_run_id: str | None = None,
               image: str | None = None, credential_fingerprint: str | None = None,
               monitor_bundle: Mapping[str, Any] | None = None) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    if artifact_run_id is None:
        artifact_run_id = getattr(args, "artifact_run_id", None)
    if artifact_run_id is None:
        artifact_run_id = run_id
    artifact_run_id = validate_run_id(artifact_run_id)
    image = image if image is not None else getattr(args, "image", None)
    if image is not None:
        image = _validate_image_reference(image)
    fingerprint = (
        credential_fingerprint
        if credential_fingerprint is not None
        else getattr(args, "runpod_key_sha256", None)
    )
    if (fingerprint is not None
            and (not isinstance(fingerprint, str)
                 or not re.fullmatch(r"[0-9a-f]{64}", fingerprint))):
        raise SafetyError("lease credential identity must be a lowercase SHA-256 digest")
    state = {
        "schema_version": 2,
        "run_id": run_id,
        "attempt_id": run_id,
        "lease_attempt_id": run_id,
        "artifact_run_id": artifact_run_id,
        "artifact_lineage_id": artifact_run_id,
        "pod_name": _name_for(run_id),
        "gpu": getattr(args, "gpu", None),
        "image": image,
        "created_at": time.time(),
        "status": "pending",
        "min_balance": getattr(args, "min_balance", DEFAULT_MIN_BALANCE),
        "max_spend": getattr(args, "max_spend", DEFAULT_MAX_SPEND),
        "max_runtime": getattr(args, "max_runtime", DEFAULT_MAX_RUNTIME),
        "poll_seconds": getattr(args, "poll_seconds", DEFAULT_POLL_SECONDS),
        "max_api_failures": getattr(args, "max_api_failures", DEFAULT_MAX_API_FAILURES),
        "pending_timeout": getattr(args, "pending_timeout", DEFAULT_PENDING_TIMEOUT),
        "termination_attempts": getattr(args, "termination_attempts", DEFAULT_TERMINATION_ATTEMPTS),
        "termination_poll_seconds": getattr(args, "termination_poll_seconds", DEFAULT_TERMINATION_POLL_SECONDS),
        "monitor_status": "arming",
        "monitor_nonce": secrets.token_urlsafe(32),
        # This nonce is independent of the user-visible attempt ID.  Reusing
        # an ID or deleting a state directory can therefore never make an old
        # remote terminal record satisfy a new launch.
        "launch_nonce": _validate_launch_nonce(
            getattr(args, "launch_nonce", None) or secrets.token_urlsafe(48)
        ),
        "deployment_started": False,
        "torch_version": TORCH_VERSION,
        "expected_torch_version": TORCH_VERSION,
        # This is a digest of the credential, never the credential itself.
        # The detached monitor uses it to select the same key after a reboot
        # or shell disconnect when the credential file contains rotations.
        "runpod_key_sha256": fingerprint,
    }
    stage_manifest = getattr(args, "expected_stage_manifest_sha256", None)
    stage_source_head = getattr(args, "expected_stage_source_head", None)
    if stage_manifest is not None:
        state["stage_manifest_sha256"] = _validate_stage_manifest_digest(stage_manifest)
    if stage_source_head is not None:
        state["stage_source_head"] = _validate_source_head(stage_source_head)
    extraction_contract = getattr(args, "extraction_contract", None)
    if extraction_contract is not None:
        if not isinstance(extraction_contract, Mapping):
            raise SafetyError("extraction recovery contract is malformed")
        # The controller constructs and validates this mapping before lease
        # creation. Persist a detached plain dictionary so later no-cost
        # recovery never depends on mutable argparse state.
        state["extraction_contract"] = dict(extraction_contract)
    results_preflight = getattr(args, "results_preflight", None)
    if results_preflight is not None:
        if not isinstance(results_preflight, Mapping):
            raise SafetyError("results preflight evidence is malformed")
        state["results_preflight"] = dict(results_preflight)
    if monitor_bundle is not None:
        state["monitor_bundle"] = dict(monitor_bundle)
    return state




def _lease_config(args: argparse.Namespace) -> SafetyConfig:
    return SafetyConfig(
        min_balance=getattr(args, "min_balance", DEFAULT_MIN_BALANCE),
        max_spend=getattr(args, "max_spend", DEFAULT_MAX_SPEND),
        max_runtime=getattr(args, "max_runtime", DEFAULT_MAX_RUNTIME),
        poll_seconds=getattr(args, "poll_seconds", DEFAULT_POLL_SECONDS),
        max_api_failures=getattr(args, "max_api_failures", DEFAULT_MAX_API_FAILURES),
        pending_timeout=getattr(args, "pending_timeout", DEFAULT_PENDING_TIMEOUT),
        termination_attempts=getattr(args, "termination_attempts", DEFAULT_TERMINATION_ATTEMPTS),
        termination_poll_seconds=getattr(args, "termination_poll_seconds", DEFAULT_TERMINATION_POLL_SECONDS),
    )


def _create_lease(args: argparse.Namespace, *, client: RunPodClient | None = None,
                  monitor_launcher: Callable[[Path], str] | None = None) -> tuple[dict[str, Any], LeaseSupervisor]:
    monitor_launcher = monitor_launcher or start_systemd_monitor
    attempt_id, artifact_run_id = _resolve_run_ids(args)
    # Validate every local argument and credential before touching the account.
    # This keeps malformed requests out of even the read-only API preflight and
    # guarantees that no state or paid mutation precedes complete validation.
    validated = _validate_lease_args(
        args, attempt_id=attempt_id, artifact_run_id=artifact_run_id
    )
    lease_config = validated["config"]
    state_file = validated["state_file"]
    _assert_new_state_path(state_file)
    credential_fingerprint = _foreground_credential_fingerprint(args)
    production_client = client is None
    if production_client:
        _validate_ssh_key_pair(validated["identity_file"], validated["public_key"])
        client = RunPodClient(
            credential_fingerprint=credential_fingerprint,
            key_file=KEY_FILE,
            file_only=credential_fingerprint is not None,
        )
    elif isinstance(client, RunPodClient):
        if (client.credential_fingerprint is not None
                and credential_fingerprint is not None
                and client.credential_fingerprint != credential_fingerprint):
            raise SafetyError("client credential does not match the lease fingerprint")
        if credential_fingerprint is not None:
            client.credential_fingerprint = credential_fingerprint
            client.file_only = True

    # Read-only account checks happen before the first state mutation.  The
    # reserve includes the configured maximum spend: a floor-only check can
    # start a run that cannot afford its own worst-case lease.
    balance = client.balance()
    if isinstance(balance, bool):
        raise APIUncertain("RunPod balance response was malformed")
    try:
        balance = float(balance)
    except (TypeError, ValueError) as exc:
        raise APIUncertain("RunPod balance response was malformed") from exc
    if not math.isfinite(balance):
        raise APIUncertain("RunPod balance response was non-finite")
    required_balance = lease_config.min_balance + lease_config.max_spend
    if not math.isfinite(required_balance):
        raise SafetyError("balance reserve is non-finite")
    if balance < required_balance:
        raise SafetyError(
            f"balance ${balance:.2f} is below the ${required_balance:.2f} reserve "
            "(minimum balance plus maximum spend); not creating"
        )

    # Apply the publisher-owned exact image identity after the reserve check.
    # This preserves the useful no-mutation balance diagnostic while still
    # rejecting every non-canonical image before state creation or deploy.
    image = _validate_paid_image(validated["image"])
    validated["image"] = image

    state = _new_state(attempt_id, args, artifact_run_id=artifact_run_id,
                       image=validated["image"],
                       credential_fingerprint=credential_fingerprint)
    state.update({"balance_before": balance, "gpu": validated["gpu"],
                  "disk_gb": validated["disk"],
                  "identity_file": str(validated["identity_file"]),
                  "pubkey_path": str(validated["pubkey_path"]),
                  "hf_token_path": str(validated["hf_token_path"]),
                  "hf_token_sha256": validated["hf_token_sha256"],
                  "image": validated["image"],
                  "launch_nonce": validated["launch_nonce"],
                  "torch_version": TORCH_VERSION,
                  "expected_torch_version": TORCH_VERSION})
    if validated["stage_manifest_sha256"] is not None:
        state["stage_manifest_sha256"] = validated["stage_manifest_sha256"]
    if validated["stage_source_head"] is not None:
        state["stage_source_head"] = validated["stage_source_head"]
    live = client.pods()
    if not isinstance(live, list) or any(not isinstance(item, Mapping) for item in live):
        raise APIUncertain("RunPod pod listing was malformed")
    active_jlens = [
        p for p in live if str(p.get("name", "")).startswith(NAME_PREFIX)
    ]
    if active_jlens:
        raise SafetyError(
            f"a JLens pod already exists ({active_jlens[0].get('id')}); "
            "concurrent JLens leases are disabled"
        )
    provider_offer: dict[str, Any] | None = None
    if isinstance(client, RunPodClient):
        provider_offer = client.gpu_offer(validated["gpu"])
        _validate_offer_budget(provider_offer, lease_config)
        state["provider_offer"] = provider_offer
    state["protected_pod_ids"] = [str(p["id"]) for p in live
                                   if isinstance(p, Mapping) and p.get("id")]
    monitor_bundle = _install_monitor_bundle(state_file)
    state["monitor_bundle"] = monitor_bundle

    # This is the only initial state write.  It must fail on any reused path,
    # including a state whose previous run was already marked terminated.
    _create_initial_json(state_file, state)
    supervisor = LeaseSupervisor(client, state_file, lease_config)
    monitor_unit: str | None = None
    pod_id: str | None = None
    deployment_started = False
    try:
        monitor_unit = monitor_launcher(state_file)
        if (not isinstance(monitor_unit, str)
                or not re.fullmatch(r"[A-Za-z0-9_.:@-]{1,255}", monitor_unit)):
            raise SafetyError("monitor launcher did not return a valid systemd unit")
        # The monitor may have updated state between systemd activation and
        # this return.  Merge into its current document instead of replacing
        # that acknowledgement with the stale pre-launch copy.
        state = supervisor._state()
        _verify_monitor_bundle(state_file, state)
        if state.get("status") == "terminated" or state.get("termination_verified") is True:
            raise SafetyError("lease monitor terminated the pending attempt before deployment")
        state["monitor_unit"] = monitor_unit
        supervisor._save(state)
        # The built-in transient unit is asynchronous, so give it a bounded
        # window to complete the handshake. Alternate launchers must persist
        # the exact acknowledgement before returning; they must not turn a
        # deterministic failure into an unbounded wait.
        wait_for_ready = monitor_launcher is start_systemd_monitor
        deadline = time.monotonic() + 30.0
        while True:
            state = supervisor._state()
            status = state.get("status")
            if (isinstance(status, str) and status.lower() in
                    {"terminated", "terminating", "error"}
                    or state.get("termination_verified") is True
                    or state.get("termination_attempted_at") is not None
                    or state.get("terminated_at") is not None
                    or any(key in state for key in (
                        "termination_error", "monitor_error", "error"))):
                raise SafetyError("lease monitor reported termination or error before deployment")
            try:
                _require_monitor_ready(
                    state,
                    nonce=str(state.get("monitor_nonce")),
                    fingerprint=credential_fingerprint,
                )
                break
            except SafetyError:
                if not wait_for_ready:
                    raise
                if state.get("monitor_status") not in {"arming", None}:
                    raise
                if time.monotonic() >= deadline:
                    raise SafetyError("disconnect-safe monitor did not acknowledge lease ownership")
            time.sleep(0.05)
        # Recheck immediately before the paid mutation to narrow the balance
        # race.  No monitor or mutation occurs when this check is uncertain.
        balance = client.balance()
        if isinstance(balance, bool):
            raise APIUncertain("RunPod balance response was malformed")
        try:
            balance = float(balance)
        except (TypeError, ValueError) as exc:
            raise APIUncertain("RunPod balance response was malformed") from exc
        if not math.isfinite(balance):
            raise APIUncertain("RunPod balance response was non-finite")
        if balance < required_balance:
            raise SafetyError(f"balance ${balance:.2f} is below the ${required_balance:.2f} reserve; not creating")
        state["balance_before"] = balance
        supervisor._save(state)
        launch_kwargs = {
            "name": state["pod_name"],
            "gpu": validated["gpu"],
            "disk": validated["disk"],
            "public_key": validated["public_key"],
            "hf_token": validated["hf_token"],
            "image": validated["image"],
            "torch_version": TORCH_VERSION,
        }
        # Close the name-availability race after the monitor and final
        # balance check.  A pod that appears before the mutation is not ours;
        # reject the attempt rather than letting a later failure path scan by
        # name and terminate it.
        latest = client.pods()
        if not isinstance(latest, list) or any(not isinstance(item, Mapping) for item in latest):
            raise APIUncertain("RunPod pod listing was malformed before deployment")
        concurrent_jlens = [
            item for item in latest
            if str(item.get("name", "")).startswith(NAME_PREFIX)
        ]
        if concurrent_jlens:
            raise SafetyError(
                "a JLens pod appeared before deployment; concurrent JLens leases are disabled"
            )
        if isinstance(client, RunPodClient):
            # Re-read the exact secure offer immediately before mutation; a
            # stale preflight quote cannot authorize a newly unaffordable or
            # unavailable placement.
            provider_offer = client.gpu_offer(validated["gpu"])
            _validate_offer_budget(provider_offer, lease_config)
        state = supervisor._state()
        _verify_monitor_bundle(state_file, state)
        if state.get("monitor_status") == "terminated" or state.get("status") == "terminated" \
                or state.get("termination_verified") is True:
            raise SafetyError("lease monitor stopped owning the attempt before deployment")
        _require_monitor_ready(
            state,
            nonce=str(state.get("monitor_nonce")),
            fingerprint=credential_fingerprint,
        )
        protected = {str(value) for value in (state.get("protected_pod_ids") or [])}
        protected.update(str(item["id"]) for item in latest if item.get("id"))
        state["protected_pod_ids"] = sorted(protected)
        if provider_offer is not None:
            state["provider_offer"] = provider_offer
        supervisor._save(state)
        verify_systemd_monitor_active(str(monitor_unit))
        provider_terminate_after = _provider_termination_deadline(
            time.time(), lease_config.max_runtime
        )
        worker_deadline_epoch = _worker_deadline_epoch(
            provider_terminate_after, now=time.time()
        )
        state["provider_terminate_after"] = provider_terminate_after
        state["worker_deadline_epoch"] = worker_deadline_epoch
        launch_kwargs["terminate_after"] = provider_terminate_after
        state["deployment_started"] = True
        supervisor._save(state)
        deployment_started = True
        if isinstance(client, RunPodClient):
            client._arm_deployment(supervisor)
            pod = client.deploy(**launch_kwargs, _guard=supervisor)
        else:
            # Dependency-injected fakes keep the same observable call surface
            # for offline tests; production clients always require the guard
            # above.
            pod = client.deploy(**launch_kwargs)
        if not isinstance(pod, Mapping) or not pod.get("id"):
            raise APIUncertain("deploy response did not contain a pod identity")
        if str(pod["id"]) in set(state.get("protected_pod_ids") or []):
            raise IdentityMismatch("deploy response reused a pre-existing pod identity")
        if pod.get("name") is not None and pod.get("name") != state["pod_name"]:
            raise IdentityMismatch("deploy response was not bound to the requested pod name")
        pod_id = str(pod["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
            raise APIUncertain("deploy response contained an invalid pod identity")
        cost_per_hr = float(pod.get("costPerHr") or 0)
        if not math.isfinite(cost_per_hr) or cost_per_hr <= 0:
            raise SafetyError("deploy response omitted a valid pod cost")
        actual_full_runtime_cost = cost_per_hr * lease_config.max_runtime / 3600.0
        if (not math.isfinite(actual_full_runtime_cost)
                or actual_full_runtime_cost > lease_config.max_spend):
            raise SafetyError(
                "deployed pod price exceeds the maximum full-runtime spend cap"
            )
        gpu_display_name = _optional_assigned_b300(pod)
        deploy_machine = pod.get("machineId")
        if deploy_machine is not None:
            deploy_machine = _validate_machine_id(deploy_machine)
        state.update({"pod_id": pod_id,
                      "cost_per_hr": cost_per_hr, "status": "active",
                      "bound_at": time.time()})
        if gpu_display_name is not None:
            state["provider_gpu_display_name"] = gpu_display_name
        if deploy_machine is not None:
            state["machine_id"] = deploy_machine
        supervisor._save(state)
        # Wait is itself identity checked; on any error the finally path
        # invokes termination and refuses to hand out a billing pod.
        # Provider TTL starts before deployment.  Bound SSH provisioning by
        # the actual deadline, preserving both the measured worker floor and
        # a separate bounded bootstrap-transfer reserve.
        ssh_budget = (
            worker_deadline_epoch - time.time()
            - MIN_END_TO_END_WORKER_SECONDS
            - WORKER_STARTUP_RESERVE_SECONDS
        )
        if ssh_budget <= 0:
            raise SafetyError(
                "deployment left no bounded SSH provisioning window"
            )
        ready = wait_for_ssh(
            client,
            pod_id,
            timeout=min(validated["ssh_timeout"], ssh_budget),
            poll_seconds=lease_config.poll_seconds,
        )
        # RunPod may omit ``machine.gpuDisplayName`` from both the deploy
        # response and the ready-pod listing.  The mutation itself is bound to
        # the exact B300 SKU and the remote entrypoint checks the physical GPU
        # with nvidia-smi before downloading the stage or model.  Treat an API
        # GPU name as corroboration when present, not as a required field.
        # machineId remains mandatory at readiness so cleanup stays bound to
        # the concrete lease RunPod assigned.
        state = supervisor._state()
        LeaseSupervisor._exact_pod(state, [ready["pod"]])
        ready_machine = _validate_machine_id(ready["pod"].get("machineId"))
        ready_gpu_name = _optional_assigned_b300(ready["pod"])
        if (gpu_display_name is not None and ready_gpu_name is not None
                and ready_gpu_name != gpu_display_name):
            raise IdentityMismatch("provider GPU identity changed while waiting for SSH")
        _require_remaining_worker_window(
            worker_deadline_epoch,
            reserve_seconds=WORKER_STARTUP_RESERVE_SECONDS,
        )
        port = ready["port"]
        state.update({
            "machine_id": ready_machine,
            # This records the exact SKU RunPod accepted.  pod_entry.sh
            # independently attests the physical device before useful work.
            "provider_gpu_display_name": ready_gpu_name or gpu_display_name or DEFAULT_GPU,
            "ssh_ip": port["ip"],
            "ssh_port": port["publicPort"],
        })
        supervisor._save(state)
        print(f"run_id={attempt_id} pod={pod_id} monitor=armed")
        print(f"SSH READY:\n  ssh -i {validated['identity_file']} -o StrictHostKeyChecking=no -p "
              f"{port['publicPort']} root@{port['ip']}")
        return state, supervisor
    except BaseException:
        try:
            state = dict(_read_json(state_file))
        except BaseException:
            # Retain the identity assembled before the failed persistence;
            # cleanup still has the exact run name and any deploy response id.
            state = dict(state)
        if pod_id:
            state["pod_id"] = pod_id
            try:
                supervisor._save(state)
            except (OSError, SafetyError):
                pass
        # A failed monitor launch occurs before the deployment capability is
        # armed and owns no pod.  In particular, do not scan by name and kill
        # a pod that appeared concurrently.  Once deploy was attempted, the
        # supervisor may recover an API response lost after mutation, subject
        # to protected pre-existing identities in the state.
        if pod_id or deployment_started:
            try:
                supervisor.terminate_verified()
            except SafetyError as exc:
                raise TerminationUnverified("launch failed and termination was not verified") from exc
        if monitor_unit and monitor_launcher is start_systemd_monitor:
            try:
                stop_systemd_monitor(monitor_unit)
            except SafetyError as exc:
                raise TerminationUnverified("launch failed and monitor shutdown was not verified") from exc
        raise


def create(args: argparse.Namespace) -> None:
    """Handle the offline-only legacy command.

    Provisioning is intentionally available only through ``run``, where
    stage verification, supervision, remote execution, and teardown remain
    one controlled lifecycle.
    """

    if args.dry_run:
        # Dry-run validates only non-secret local values.  It intentionally
        # does not inspect credential paths or instantiate an API publisher.
        if getattr(args, "allow_second", False):
            raise SafetyError("--allow-second is disabled; every lease attempt needs a unique pod name")
        attempt_id, artifact_run_id = _resolve_run_ids(args)
        _validate_gpu(getattr(args, "gpu", DEFAULT_GPU))
        _positive_int(getattr(args, "disk", DEFAULT_DISK_GB), "disk")
        _lease_config(args)
        image = getattr(args, "image", None)
        if image is not None:
            _validate_image_reference(image)
        print("DRY RUN: no API, credential, pod, SSH, HF, or systemd action")
        print(json.dumps({"run_id": attempt_id, "artifact_run_id": artifact_run_id,
                          "pod_name": _name_for(attempt_id), "gpu": args.gpu,
                          "disk_gb": args.disk, "image": image,
                          "min_balance": args.min_balance,
                          "max_spend": args.max_spend, "max_runtime": args.max_runtime}, sort_keys=True))
        return
    raise SafetyError("provision-only create is disabled; use supervised run")


def stop_systemd_monitor(unit: str, *, runner: Callable[..., Any] | None = None,
                         unit_dir: Path | None = None) -> None:
    """Disable, stop, verify, and remove a persistent per-attempt monitor."""

    if not unit:
        return
    runner = runner or subprocess.run
    unit_path = _monitor_unit_path(unit, unit_dir)
    try:
        result = runner(["systemctl", "--user", "disable", "--now", unit], check=False,
                        capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError("could not stop the lease monitor") from exc
    if getattr(result, "returncode", None) != 0:
        raise SafetyError("systemd did not stop the lease monitor")
    try:
        check = runner(["systemctl", "--user", "is-active", unit], check=False,
                       capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError("could not verify the lease monitor stopped") from exc
    stopped_states = {"inactive", "failed", "unknown", "deactivating"}
    if (getattr(check, "returncode", None) == 0
            or (getattr(check, "stdout", "") or "").strip() not in stopped_states):
        raise SafetyError("lease monitor is still active")
    try:
        if unit_path.is_symlink() or (unit_path.exists() and not unit_path.is_file()):
            raise SafetyError("persistent monitor unit path is not a regular file")
        if unit_path.exists():
            unit_path.unlink()
        reload_result = runner(["systemctl", "--user", "daemon-reload"], check=False,
                               capture_output=True, text=True, timeout=30)
    except OSError as exc:
        raise SafetyError("could not remove the persistent monitor unit") from exc
    if getattr(reload_result, "returncode", None) != 0:
        raise SafetyError("systemd did not reload after monitor removal")


REQUIRED_SUCCESS_RECEIPT_KINDS = frozenset(
    {
        "shard", "sidecar", "merged_lens", "prefix_lens", "eval_output",
        "marker", "report", "verification", "validation", "expected_inventory",
        "run_log", "heartbeat", "receipt_index", "recovery_status",
    }
)
ALLOWED_SUCCESS_RECEIPT_KINDS = REQUIRED_SUCCESS_RECEIPT_KINDS | {
    "checkpoint", "checkpoint_sidecar",
}
SCIENTIFIC_INVENTORY_KINDS = frozenset(
    {"shard", "sidecar", "merged_lens", "prefix_lens", "eval_output",
     "report", "verification", "validation"}
)
EXPECTED_INVENTORY_RELATIVE = "inventory/expected-inventory.json"
EXPECTED_REPORT_PRODUCTS = frozenset({
    "eval/fitsize_convergence.json",
    "eval/fitsize_summary.md",
    "final/fit_size.json",
    "final/fit_size.status.json",
    "final/transport.json",
    "final/analysis.json",
    "final/claim_status.json",
    "final/verification.json",
})
_EXPECTED_INVENTORY_KEYS = frozenset({
    "schema_version", "run_id", "attempt_id", "launch_nonce",
    "stage_manifest_sha256", "stage_source_head", "worker_deadline_epoch",
    "n_prompts", "shard_size", "shard_intervals",
    "variable_shards", "lens", "prefix_lenses", "main_evaluations",
    "fit_size_evaluations", "report_products", "validation_products",
    "evaluation_tags",
})
_INVENTORY_RECORD_KEYS = frozenset({"path", "receipt", "kind", "size", "sha256"})


def _receipt_attribute(receipt: Any, name: str) -> Any:
    try:
        return getattr(receipt, name)
    except AttributeError as exc:
        raise PublishError(f"publisher returned a malformed receipt ({name} missing)") from exc


def _receipt_inventory(receipts: list[Any], *, run_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for receipt in receipts:
        schema_version = _receipt_attribute(receipt, "schema_version")
        receipt_run = _receipt_attribute(receipt, "run_id")
        relative = _receipt_attribute(receipt, "relative_path")
        remote_path = _receipt_attribute(receipt, "remote_path")
        kind = _receipt_attribute(receipt, "kind")
        digest = _receipt_attribute(receipt, "sha256")
        size = _receipt_attribute(receipt, "size")
        if schema_version != 1 or receipt_run != run_id:
            raise PublishError("publisher returned a receipt from another artifact lineage")
        if (not isinstance(relative, str) or not isinstance(remote_path, str)
                or not isinstance(kind, str)):
            raise PublishError("publisher returned a receipt with invalid path or kind")
        try:
            relative = safe_relative(relative)
            remote_path = safe_relative(remote_path)
        except ValueError as exc:
            raise PublishError("publisher returned an unsafe receipt path") from exc
        if remote_path != f"{run_id}/artifacts/{relative}":
            raise PublishError("publisher returned a receipt with a mismatched remote path")
        if kind not in ALLOWED_SUCCESS_RECEIPT_KINDS:
            raise PublishError(f"successful run contains an unsupported receipt kind: {kind}")
        if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise PublishError("publisher returned a receipt with invalid byte identity")
        if relative in result:
            raise PublishError("publisher returned duplicate receipt paths")
        result[relative] = receipt
    return result


def _verify_synced_tree_exact(
    local_root: Path,
    receipts: list[Any],
    *,
    run_id: str,
    allowed_receipts: list[Any] | None = None,
) -> None:
    """Reject missing, linked, special, unreceipted, or mismatched local bytes.

    The receipts argument is the required extraction set. allowed_receipts
    may additionally name payloads already custodied by an earlier failed
    attempt in the same immutable artifact lineage. Those retained payloads
    are accepted only after their bytes are rehashed against the current
    complete remote receipt union.
    """

    local_root = Path(local_root)
    expected_map = _receipt_inventory(receipts, run_id=run_id)
    allowed_map = (
        expected_map
        if allowed_receipts is None
        else _receipt_inventory(allowed_receipts, run_id=run_id)
    )
    expected = set(expected_map)
    allowed = set(allowed_map)
    if not expected <= allowed:
        raise PublishError("required extraction receipts exceed the allowed receipt inventory")
    if not local_root.exists() and not expected:
        return
    if local_root.is_symlink() or not local_root.is_dir():
        raise PublishError("synchronized artifact root is not a regular directory")
    observed: set[str] = set()
    try:
        for path in local_root.rglob("*"):
            if path.is_symlink():
                raise PublishError("synchronized artifact tree contains a symlink")
            if path.is_dir():
                continue
            stat_result = path.stat(follow_symlinks=False)
            if not path.is_file() or stat_result.st_nlink != 1:
                raise PublishError("synchronized artifact tree contains a special or linked file")
            try:
                relative = path.relative_to(local_root).as_posix()
                relative = safe_relative(relative)
            except ValueError as exc:
                raise PublishError("synchronized artifact escaped its root") from exc
            observed.add(relative)
    except OSError as exc:
        raise PublishError("could not enumerate the synchronized artifact tree") from exc
    if not expected <= observed or not observed <= allowed:
        missing = sorted(expected - observed)
        extra = sorted(observed - allowed)
        raise PublishError(
            f"synchronized artifact tree is not receipt-exact (missing={missing}, extra={extra})"
        )
    for relative in sorted(observed):
        receipt = allowed_map[relative]
        size, digest = _path_digest(
            local_root / relative,
            label="synchronized artifact payload",
        )
        if (
            size != _receipt_attribute(receipt, "size")
            or digest != _receipt_attribute(receipt, "sha256")
        ):
            raise PublishError(
                f"synchronized artifact differs from its receipt: {relative}"
            )


def _read_synced_json(local_root: Path, relative: str, label: str) -> dict[str, Any]:
    path = local_root / safe_relative(relative)
    try:
        payload = json.loads(_read_regular_file(path))
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise PublishError(f"{label} is missing, linked, or malformed") from exc
    if not isinstance(payload, dict):
        raise PublishError(f"{label} is not a JSON object")
    return payload


def _expected_intervals(target: int, n_prompts: int, shard_size: int) -> list[list[int]]:
    if target == 3:
        intervals = [[0, 8], [8, 32], [32, 56], [56, 80]]
        if n_prompts > 80:
            intervals.append([80, n_prompts])
        return intervals
    return [
        [start, min(start + shard_size, n_prompts)]
        for start in range(0, n_prompts, shard_size)
    ]


def _collect_inventory_records(value: Any, *, label: str = "inventory") -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    def visit(item: Any, location: str) -> None:
        if isinstance(item, dict) and set(item) == _INVENTORY_RECORD_KEYS:
            relative = item.get("path")
            receipt_path = item.get("receipt")
            kind = item.get("kind")
            digest = item.get("sha256")
            size = item.get("size")
            if (not isinstance(relative, str) or not isinstance(receipt_path, str)
                    or not isinstance(kind, str) or not isinstance(digest, str)
                    or not isinstance(size, int) or isinstance(size, bool) or size < 0):
                raise PublishError(f"{location} contains a malformed artifact record")
            try:
                relative = safe_relative(relative)
                receipt_path = safe_relative(receipt_path)
            except ValueError as exc:
                raise PublishError(f"{location} contains an unsafe artifact path") from exc
            if receipt_path != f"{relative}.receipt.json":
                raise PublishError(f"{location} receipt path does not bind its artifact")
            if kind not in SCIENTIFIC_INVENTORY_KINDS:
                raise PublishError(f"{location} uses an unsupported scientific kind")
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise PublishError(f"{location} contains a malformed artifact digest")
            if relative in records:
                if records[relative] != dict(item):
                    raise PublishError(f"expected inventory contradicts artifact path: {relative}")
            else:
                records[relative] = dict(item)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{location}[{index}]")

    visit(value, label)
    return records


def _validate_expected_inventory(
    payload: dict[str, Any],
    receipt_map: dict[str, Any],
    local_root: Path,
    *,
    run_id: str,
    n_prompts: int,
    attempt_id: str,
    launch_nonce: str,
    stage_manifest_sha256: str,
    stage_source_head: str,
    worker_deadline_epoch: int,
) -> tuple[set[str], set[str]]:
    def require_record(value: Any, relative: str, kind: str, label: str) -> None:
        if (not isinstance(value, dict) or set(value) != _INVENTORY_RECORD_KEYS
                or value.get("path") != relative
                or value.get("receipt") != f"{relative}.receipt.json"
                or value.get("kind") != kind):
            raise PublishError(f"{label} artifact identity is not exact")

    if set(payload) != _EXPECTED_INVENTORY_KEYS:
        raise PublishError("expected inventory schema fields are not exact")
    if (payload.get("schema_version") != 2 or payload.get("run_id") != run_id
            or payload.get("attempt_id") != attempt_id
            or payload.get("launch_nonce") != launch_nonce
            or payload.get("stage_manifest_sha256") != stage_manifest_sha256
            or payload.get("stage_source_head") != stage_source_head
            or payload.get("worker_deadline_epoch") != worker_deadline_epoch
            or payload.get("n_prompts") != n_prompts):
        raise PublishError("expected inventory does not match the requested artifact lineage")
    shard_size = payload.get("shard_size")
    if (not isinstance(shard_size, int) or isinstance(shard_size, bool)
            or shard_size <= 0):
        raise PublishError("expected inventory shard size is invalid")
    expected_intervals = {
        str(target): _expected_intervals(target, n_prompts, shard_size)
        for target in range(4)
    }
    if payload.get("shard_intervals") != expected_intervals:
        raise PublishError("expected inventory shard plan is not exact")
    lenses = payload.get("lens")
    if not isinstance(lenses, dict) or set(lenses) != set(expected_intervals):
        raise PublishError("expected inventory lens targets are incomplete")
    for target, intervals in expected_intervals.items():
        entry = lenses.get(target)
        if (not isinstance(entry, dict) or set(entry) != {"intervals", "shards", "merged"}
                or entry.get("intervals") != intervals
                or not isinstance(entry.get("shards"), list)
                or len(entry["shards"]) != len(intervals)
                or not isinstance(entry.get("merged"), dict)):
            raise PublishError(f"expected inventory lens target {target} is malformed")
        for pair, interval in zip(entry["shards"], intervals):
            if (not isinstance(pair, dict)
                    or set(pair) != {"payload", "sidecar", "target_ut", "start", "end"}
                    or pair.get("target_ut") != int(target)
                    or [pair.get("start"), pair.get("end")] != interval):
                raise PublishError(f"expected inventory shard target {target} is malformed")
            stem = f"lens/n{n_prompts}/exit{target}_shard_{interval[0]:04d}_{interval[1]:04d}"
            require_record(pair["payload"], f"{stem}.pt", "shard", f"target {target} shard")
            require_record(pair["sidecar"], f"{stem}.json", "sidecar", f"target {target} shard")
        merged_stem = f"lens/n{n_prompts}/exit{target}"
        merged = entry["merged"]
        if not isinstance(merged, dict) or set(merged) != {"payload", "sidecar"}:
            raise PublishError(f"expected inventory merged lens target {target} is malformed")
        require_record(merged["payload"], f"{merged_stem}.pt", "merged_lens",
                       f"target {target} merged lens")
        require_record(merged["sidecar"], f"{merged_stem}.json", "sidecar",
                       f"target {target} merged lens")
    if payload.get("variable_shards") != lenses["3"]["shards"]:
        raise PublishError("expected inventory variable shard plan disagrees with target 3")

    prefixes = payload.get("prefix_lenses")
    if (not isinstance(prefixes, list) or len(prefixes) != 4
            or [entry.get("n_prompts") if isinstance(entry, dict) else None for entry in prefixes]
            != [8, 32, 56, 80]
            or any(not isinstance(entry, dict)
                   or set(entry) != {"payload", "sidecar", "n_prompts", "sources"}
                   for entry in prefixes)):
        raise PublishError("expected inventory prefix lens set is not exact")
    target_three_sources = [
        f"exit3_shard_{start:04d}_{end:04d}.pt"
        for start, end in expected_intervals["3"][:4]
    ]
    for index, (entry, size) in enumerate(zip(prefixes, (8, 32, 56, 80)), start=1):
        stem = f"lens/n{n_prompts}/exit3_n{size}"
        require_record(entry["payload"], f"{stem}.pt", "prefix_lens", f"prefix n={size}")
        require_record(entry["sidecar"], f"{stem}.json", "sidecar", f"prefix n={size}")
        if entry.get("sources") != target_three_sources[:index]:
            raise PublishError(f"expected inventory prefix n={size} source seal is not exact")

    expected_main_tags = [
        f"n{n_prompts}_exit3", f"n{n_prompts}_allexits",
        f"n{n_prompts}_allexits_pos-2",
    ]
    main_evaluations = payload.get("main_evaluations")
    if (not isinstance(main_evaluations, list)
            or [entry.get("tag") if isinstance(entry, dict) else None
                for entry in main_evaluations] != expected_main_tags
            or any(not isinstance(entry, dict) or set(entry) != {"tag", "files"}
                   or not isinstance(entry.get("files"), list) or not entry["files"]
                   for entry in main_evaluations)):
        raise PublishError("expected inventory main evaluation set is not exact")
    expected_fit_tags = [f"fitsize_n{size}" for size in (8, 32, 56, 80)]
    fit_evaluations = payload.get("fit_size_evaluations")
    if (not isinstance(fit_evaluations, list)
            or [entry.get("tag") if isinstance(entry, dict) else None
                for entry in fit_evaluations] != expected_fit_tags
            or [entry.get("n_prompts") if isinstance(entry, dict) else None
                for entry in fit_evaluations] != [8, 32, 56, 80]
            or any(not isinstance(entry, dict)
                   or set(entry) != {"tag", "n_prompts", "files"}
                   or not isinstance(entry.get("files"), list) or not entry["files"]
                   for entry in fit_evaluations)):
        raise PublishError("expected inventory fit-size evaluation set is not exact")
    if payload.get("evaluation_tags") != expected_main_tags:
        raise PublishError("expected inventory evaluation tag declaration is not exact")
    required_eval_names = {
        "arrays.npz", "items.json", "task_names.json", "provenance.json",
        "summary.json", "summary.md",
    }
    for entry in [*main_evaluations, *fit_evaluations]:
        tag = entry["tag"]
        paths = {
            record.get("path") for record in entry["files"] if isinstance(record, dict)
        }
        if (not all(isinstance(path, str) and path.startswith(f"eval/{tag}/") for path in paths)
                or not {f"eval/{tag}/{name}" for name in required_eval_names} <= paths):
            raise PublishError(f"expected inventory evaluation {tag} file set is incomplete")
        for record in entry["files"]:
            if not isinstance(record, dict) or record.get("kind") != "eval_output":
                raise PublishError(f"expected inventory evaluation {tag} kind is invalid")

    report_records = payload.get("report_products")
    validation_records = payload.get("validation_products")
    if (not isinstance(report_records, list) or not isinstance(validation_records, list)
            or not validation_records):
        raise PublishError("expected inventory report or validation products are missing")
    if {entry.get("path") for entry in report_records if isinstance(entry, dict)} != EXPECTED_REPORT_PRODUCTS:
        raise PublishError("expected inventory report product set is not exact")
    for record in report_records:
        expected_kind = "verification" if record.get("path") == "final/verification.json" else "report"
        if not isinstance(record, dict) or record.get("kind") != expected_kind:
            raise PublishError("expected inventory report product kind is invalid")
    if not any(
        isinstance(entry, dict) and entry.get("path") == "validation/milestones.json"
        for entry in validation_records
    ):
        raise PublishError("expected inventory omits validation/milestones.json")
    if any(not isinstance(record, dict)
           or not isinstance(record.get("path"), str)
           or not record["path"].startswith("validation/")
           or record.get("kind") != "validation"
           for record in validation_records):
        raise PublishError("expected inventory validation product path or kind is invalid")

    records = _collect_inventory_records(payload)
    declared = set(records)
    observed_scientific = {
        relative for relative, receipt in receipt_map.items()
        if _receipt_attribute(receipt, "kind") in SCIENTIFIC_INVENTORY_KINDS
    }
    if declared != observed_scientific:
        missing = sorted(declared - observed_scientific)
        extra = sorted(observed_scientific - declared)
        raise PublishError(
            f"expected inventory and scientific receipts differ (missing={missing}, extra={extra})"
        )
    for relative, record in records.items():
        receipt = receipt_map[relative]
        if (_receipt_attribute(receipt, "kind") != record["kind"]
                or _receipt_attribute(receipt, "size") != record["size"]
                or _receipt_attribute(receipt, "sha256") != record["sha256"]):
            raise PublishError(f"expected inventory byte identity differs from receipt: {relative}")
    return declared, set(expected_main_tags + expected_fit_tags)


def _verify_success_inventory(
    receipts: list[Any],
    local_root: Path,
    *,
    run_id: str,
    n_prompts: int,
    attempt_id: str,
    launch_nonce: str,
    stage_manifest_sha256: str,
    stage_source_head: str,
    worker_deadline_epoch: int,
    required_local_paths: set[str] | None = None,
) -> None:
    """Check the policy inventory and terminal success marker after sync.

    The publisher owns receipt enumeration.  This check only interprets the
    verified local copies; it never substitutes a controller-side receipt
    cache for the remote union.
    """

    run_id = validate_run_id(run_id)
    attempt_id = validate_run_id(attempt_id)
    launch_nonce = _validate_launch_nonce(launch_nonce)
    stage_manifest_sha256 = _validate_stage_manifest_digest(stage_manifest_sha256)
    stage_source_head = _validate_source_head(stage_source_head)
    if (isinstance(worker_deadline_epoch, bool)
            or not isinstance(worker_deadline_epoch, int)
            or worker_deadline_epoch <= 0):
        raise PublishError("worker deadline binding is malformed")
    receipt_map = _receipt_inventory(receipts, run_id=run_id)
    local_paths = set(receipt_map) if required_local_paths is None else set(required_local_paths)
    if not local_paths <= set(receipt_map):
        raise PublishError("local extraction requests paths outside the remote receipt inventory")
    for relative in sorted(local_paths):
        receipt = receipt_map[relative]
        try:
            size, digest = _path_digest(local_root / relative, label="synced receipt payload")
        except OSError as exc:
            raise PublishError(f"synced receipt payload is unavailable: {relative}") from exc
        if (size != _receipt_attribute(receipt, "size")
                or digest != _receipt_attribute(receipt, "sha256")):
            raise PublishError(f"synced receipt payload differs from its receipt: {relative}")
    relatives = list(receipt_map)
    kinds = {_receipt_attribute(receipt, "kind") for receipt in receipts}
    terminal_records: list[tuple[float, str, dict[str, Any]]] = []
    index_receipts: list[Any] = []
    for receipt in receipts:
        relative = _receipt_attribute(receipt, "relative_path")
        kind = _receipt_attribute(receipt, "kind")
        if is_receipt_index(relative):
            index_receipts.append(receipt)
        if kind == "heartbeat" and relative.startswith("status/"):
            try:
                payload = _read_synced_json(local_root, relative, "heartbeat")
            except PublishError:
                raise
            recorded_at = payload.get("recorded_at")
            if (payload.get("run_id") != run_id or not isinstance(recorded_at, (int, float))
                    or isinstance(recorded_at, bool) or not math.isfinite(float(recorded_at))):
                raise PublishError("successful run contains a malformed heartbeat")
            if (payload.get("attempt_id") == attempt_id
                    and payload.get("launch_nonce") == launch_nonce
                    and payload.get("stage_manifest_sha256") == stage_manifest_sha256
                    and payload.get("stage_source_head") == stage_source_head
                    and payload.get("worker_deadline_epoch") == worker_deadline_epoch):
                terminal_records.append((float(recorded_at), relative, payload))
    missing = sorted(REQUIRED_SUCCESS_RECEIPT_KINDS - kinds)
    if missing:
        raise PublishError(f"successful run is missing receipt kinds: {', '.join(missing)}")
    if not terminal_records:
        raise PublishError("successful run has no heartbeat for the current lease attempt")
    terminal_records.sort(key=lambda item: (item[0], item[1]))
    terminal_relative = terminal_records[-1][1]
    terminal = terminal_records[-1][2]
    if (terminal.get("state") != "succeeded" or terminal.get("stage") != "complete"
            or terminal.get("exit_code") != 0
            or isinstance(terminal.get("exit_code"), bool)):
        raise PublishError("latest current-attempt heartbeat is not terminal success")

    inventory_receipt = receipt_map.get(EXPECTED_INVENTORY_RELATIVE)
    if (inventory_receipt is None
            or _receipt_attribute(inventory_receipt, "kind") != "expected_inventory"):
        raise PublishError("successful run has no exact expected-inventory receipt")
    expected_payload = _read_synced_json(
        local_root, EXPECTED_INVENTORY_RELATIVE, "expected inventory"
    )
    declared_paths, evaluation_tags = _validate_expected_inventory(
        expected_payload,
        receipt_map,
        local_root,
        run_id=run_id,
        n_prompts=n_prompts,
        attempt_id=attempt_id,
        launch_nonce=launch_nonce,
        stage_manifest_sha256=stage_manifest_sha256,
        stage_source_head=stage_source_head,
        worker_deadline_epoch=worker_deadline_epoch,
    )
    expected_markers = {f"markers/{tag}.complete" for tag in evaluation_tags}
    observed_markers = {
        relative for relative, receipt in receipt_map.items()
        if _receipt_attribute(receipt, "kind") == "marker"
    }
    if observed_markers != expected_markers:
        raise PublishError("evaluation completion-marker receipt set is not exact")
    for relative in expected_markers:
        try:
            marker = _read_regular_file(local_root / relative)
        except OSError as exc:
            raise PublishError(f"evaluation completion marker is unavailable: {relative}") from exc
        tag = relative[len("markers/"):-len(".complete")]
        if marker != f"{run_id}:{tag}\n".encode():
            raise PublishError(f"evaluation completion marker is malformed: {relative}")
    log_paths = {
        relative for relative, receipt in receipt_map.items()
        if _receipt_attribute(receipt, "kind") == "run_log"
    }
    if not log_paths or any(not path.startswith("logs/") for path in log_paths):
        raise PublishError("successful run has no valid acknowledged run log")
    recovery_relative = f"recovery/{attempt_id}.json"
    recovery_receipt = receipt_map.get(recovery_relative)
    if (recovery_receipt is None
            or _receipt_attribute(recovery_receipt, "kind") != "recovery_status"):
        raise PublishError("successful run has no current-attempt checkpoint recovery status")
    recovery = _read_synced_json(local_root, recovery_relative, "checkpoint recovery status")
    if (recovery.get("schema_version") != 1 or recovery.get("run_id") != run_id
            or recovery.get("attempt_id") != attempt_id or recovery.get("complete") is not True
            or recovery.get("launch_nonce") != launch_nonce
            or recovery.get("stage_manifest_sha256") != stage_manifest_sha256
            or recovery.get("stage_source_head") != stage_source_head
            or recovery.get("worker_deadline_epoch") != worker_deadline_epoch
            or not isinstance(recovery.get("recovered_checkpoint_pairs"), int)
            or isinstance(recovery.get("recovered_checkpoint_pairs"), bool)
            or recovery.get("recovered_checkpoint_pairs") < 0):
        raise PublishError("current-attempt checkpoint recovery status is incomplete")

    # The terminal index is itself enumerated and synchronized by the
    # publisher.  Direct enumeration remains authoritative for failed-run
    # recovery, but a successful run needs this policy-derived completeness
    # assertion as well.
    if not index_receipts:
        raise PublishError("successful run has no acknowledged receipt index")
    actual_paths = {path for path in relatives if not is_receipt_index(path)}
    mandatory_index_paths = (
        declared_paths | expected_markers
        | {EXPECTED_INVENTORY_RELATIVE, terminal_relative, recovery_relative}
    )
    complete_indexes = 0
    for index_receipt in index_receipts:
        index_relative = _receipt_attribute(index_receipt, "relative_path")
        try:
            index_path = local_root / safe_relative(index_relative)
            indexed_paths = set(read_index(
                index_path.read_bytes(), run_id=run_id,
            ))
        except (OSError, UnicodeError, ValueError, TypeError, PublishError) as exc:
            raise PublishError("successful run has a malformed receipt index") from exc
        if (mandatory_index_paths <= indexed_paths <= actual_paths
                and any(path in indexed_paths for path in log_paths)):
            complete_indexes += 1
        elif not indexed_paths.issubset(actual_paths):
            raise PublishError("receipt index names artifacts outside the publisher receipt inventory")
    if complete_indexes == 0:
        raise PublishError("no receipt index generation covers the exact successful-run inventory")


def _read_remote_receipt_inventory(publisher: Any, *, run_id: str) -> list[Receipt]:
    """Read the complete bounded receipt union without fetching large payloads."""

    run_id = validate_run_id(run_id)
    receipts: list[Receipt] = []
    for relative in discover_receipt_paths(publisher, run_id):
        receipt = Receipt.from_bytes(
            publisher.read_file(receipt_remote_path(run_id, relative))
        )
        if receipt.run_id != run_id or receipt.relative_path != relative:
            raise PublishError(f"receipt identity mismatch: {relative}")
        receipts.append(receipt)
    # The receipt parser and discovery bounds protect memory; this additional
    # identity pass rejects contradictory duplicate declarations explicitly.
    _receipt_inventory(receipts, run_id=run_id)
    return receipts


def _application_extraction_paths(receipts: list[Any], *, run_id: str) -> set[str]:
    """Return the small write-up/evaluation bundle, excluding large lens tensors.

    Every large tensor remains durably acknowledged in the remote receipt
    inventory and is recoverable later with ``pod.py recover``.  The immediate
    post-termination path retrieves all reports, evaluations, plots,
    validation, logs, status records, inventory, and lens sidecars, but does
    not block the application on downloading duplicated ``lens/*.pt`` files.
    """

    receipt_map = _receipt_inventory(receipts, run_id=run_id)
    selected = {
        relative
        for relative in receipt_map
        if not (relative.startswith("lens/") and relative.endswith((".pt", ".pt.ckpt")))
    }
    if not selected:
        raise PublishError("application extraction set is empty")
    return selected


def _verify_deferred_remote_payloads(
    publisher: Any,
    receipts: list[Any],
    local_paths: set[str],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Prove deferred lens bytes still exist at one exact remote revision."""

    receipt_map = _receipt_inventory(receipts, run_id=run_id)
    if not isinstance(local_paths, set) or not local_paths <= set(receipt_map):
        raise PublishError("application extraction paths do not match the receipt inventory")
    deferred = [
        receipt_map[relative]
        for relative in sorted(set(receipt_map) - local_paths)
    ]
    for receipt in deferred:
        relative = _receipt_attribute(receipt, "relative_path")
        if not (
            relative.startswith("lens/")
            and relative.endswith((".pt", ".pt.ckpt"))
        ):
            raise PublishError("application extraction deferred a non-lens payload")
    verifier = getattr(publisher, "verify_remote_receipts_metadata", None)
    if not callable(verifier):
        raise PublishError("publisher cannot verify deferred remote payload custody")
    try:
        evidence = verifier(deferred)
    except PublishError:
        raise
    except Exception:
        raise PublishError("publisher remote custody verification failed") from None
    if not isinstance(evidence, Mapping):
        raise PublishError("publisher returned malformed remote custody evidence")
    revision = evidence.get("revision")
    count = evidence.get("count")
    if (not isinstance(revision, str) or not revision
            or isinstance(count, bool) or not isinstance(count, int)
            or count != len(deferred)):
        raise PublishError("publisher returned inexact remote custody evidence")
    return {"revision": revision, "count": count}


def _retain_remote_receipts(
    receipts: list[Any], *, run_id: str, receipt_root: Path
) -> None:
    """Retain every remote receipt even when its large payload is deferred."""

    receipt_map = _receipt_inventory(receipts, run_id=run_id)
    for relative, receipt in sorted(receipt_map.items()):
        _write_immutable(
            Path(receipt_root) / f"{relative}.receipt.json",
            canonical_json({
                "schema_version": _receipt_attribute(receipt, "schema_version"),
                "run_id": run_id,
                "relative_path": relative,
                "remote_path": _receipt_attribute(receipt, "remote_path"),
                "kind": _receipt_attribute(receipt, "kind"),
                "sha256": _receipt_attribute(receipt, "sha256"),
                "size": _receipt_attribute(receipt, "size"),
            }),
        )


def _verify_extracted_paid_gate(
    local_root: Path,
    *,
    run_id: str,
    attempt_id: str,
    image: str,
    launch_nonce: str,
    stage_manifest_sha256: str,
    stage_source_head: str,
    worker_deadline_epoch: int,
) -> None:
    """Interpret the hash-verified paid verdict from the extracted byte set.

    The remote verifier performs the source/model-aware scientific replay in
    the pinned stage.  The controller independently verifies that the exact
    verdict and every input named by the expected inventory were retrieved,
    then re-evaluates the security-critical acceptance fields locally.
    """

    verification = _read_synced_json(
        local_root, "final/verification.json", "paid verification verdict"
    )
    if (verification.get("schema_version") != 1
            or verification.get("status") != "PASS"
            or verification.get("accepted") is not True
            or verification.get("run_id") != run_id
            or verification.get("attempt_id") != attempt_id
            or verification.get("launch_nonce") != launch_nonce
            or verification.get("stage_manifest_sha256") != stage_manifest_sha256
            or verification.get("stage_source_head") != stage_source_head
            or verification.get("worker_deadline_epoch") != worker_deadline_epoch
            or verification.get("image_digest") != image
            or verification.get("controller_approved_image_digest") != image
            or verification.get("validation_status") != "SUPPORTED_CURRENT_B300_VALIDATION"
            or verification.get("b300_validation_status") != "SUPPORTED_CURRENT_B300_VALIDATION"
            or verification.get("canonical_report_status") != "CURRENT_GENERATION_EVIDENCE_VERIFIED"
            or verification.get("evaluation_lineage_status") != "CURRENT_GENERATION_EVIDENCE_VERIFIED"):
        raise PublishError("extracted paid verification verdict is not exact PASS/current lineage")
    validation = _read_synced_json(
        local_root, "validation/milestones.json", "paid validation artifact"
    )
    provenance = validation.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("image_digest") != image:
        raise PublishError("extracted validation image identity is not exact")
    analysis = _read_synced_json(local_root, "final/analysis.json", "canonical analysis")
    claims = _read_synced_json(local_root, "final/claim_status.json", "canonical claims")
    instrumentation = claims.get("instrumentation")
    b300 = claims.get("b300_validation")
    if (analysis.get("overall_status") != "CURRENT_GENERATION_EVIDENCE_VERIFIED"
            or analysis.get("claims") != claims
            or not isinstance(instrumentation, dict)
            or instrumentation.get("status") != "SUPPORTED_CURRENT_VALIDATION"
            or not isinstance(b300, dict)
            or b300.get("status") != "SUPPORTED_CURRENT_B300_VALIDATION"):
        raise PublishError("extracted canonical report is not current paid B300 evidence")


def _run_local_paid_replay(
    local_root: Path,
    receipt_root: Path,
    *,
    source_root: Path,
    run_id: str,
    attempt_id: str,
    n_prompts: int,
    image: str,
    launch_nonce: str,
    stage_manifest_sha256: str,
    stage_source_head: str,
    worker_deadline_epoch: int,
) -> dict[str, Any]:
    """Rerun the complete paid verifier over only the synchronized artifacts."""

    project_root = Path(os.path.abspath(source_root))
    try:
        stage = verify_stage_root(project_root, allow_extra=False)
    except (OSError, ValueError, PublishError) as exc:
        raise PublishError("local paid replay source stage is no longer exact") from exc
    pinned = stage.get("pinned_inputs")
    if (stage.get("manifest_sha256") != stage_manifest_sha256
            or not isinstance(pinned, Mapping)
            or not isinstance(pinned.get("source"), Mapping)
            or pinned["source"].get("head") != stage_source_head
            or pinned["source"].get("status") != ""):
        raise PublishError("local paid replay source differs from the launched stage")
    local_root = Path(os.path.abspath(local_root))
    receipt_root = Path(os.path.abspath(receipt_root))
    replay_dir = receipt_root / "local-replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("HF_TOKEN", None)
    environment.pop("RUNPOD_API_KEY", None)
    # Do not append the controller's ambient PYTHONPATH: only the exact
    # verified stage may supply JLens code to the independent replay.
    environment["PYTHONPATH"] = str(project_root / "src")
    environment["PYTHONSAFEPATH"] = "1"
    environment.update({
        "RUN_ID": run_id,
        "JLENS_ATTEMPT_ID": attempt_id,
        "JLENS_IMAGE_DIGEST": image,
        "JLENS_LAUNCH_NONCE": launch_nonce,
        "EXPECTED_STAGE_MANIFEST_SHA256": stage_manifest_sha256,
        "EXPECTED_STAGE_SOURCE_HEAD": stage_source_head,
        "JLENS_WORKER_DEADLINE_EPOCH": str(worker_deadline_epoch),
    })
    with tempfile.TemporaryDirectory(prefix=".paid-replay-", dir=receipt_root) as temporary:
        verdict_path = Path(temporary) / "verification.json"
        command = [
            sys.executable, "-P", "-m", "ouro_jlens.verify_paid",
            "--artifact-root", str(local_root),
            "--main", str(local_root / "eval" / "fitsize_n80"),
            "--local-eval", str(local_root / "eval" / f"n{n_prompts}_allexits"),
            "--eval-root", str(local_root / "eval"),
            "--lens-root", str(local_root / "lens" / f"n{n_prompts}"),
            "--validation", str(local_root / "validation" / "milestones.json"),
            "--transport", str(local_root / "final" / "transport.json"),
            "--analysis", str(local_root / "final" / "analysis.json"),
            "--claims", str(local_root / "final" / "claim_status.json"),
            "--fit-size", str(local_root / "final" / "fit_size.json"),
            "--checkpoints", "",
            "--out", str(verdict_path),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise PublishError("local paid verifier replay could not execute") from exc
        if result.returncode != 0:
            raise PublishError("local paid verifier replay rejected synchronized artifacts")
        replay = _read_synced_json(
            Path(temporary), verdict_path.name, "local paid replay verdict"
        )
    if (replay.get("status") != "PASS" or replay.get("accepted") is not True
            or replay.get("run_id") != run_id
            or replay.get("attempt_id") != attempt_id
            or replay.get("launch_nonce") != launch_nonce
            or replay.get("stage_manifest_sha256") != stage_manifest_sha256
            or replay.get("stage_source_head") != stage_source_head
            or replay.get("worker_deadline_epoch") != worker_deadline_epoch
            or replay.get("image_digest") != image):
        raise PublishError("local paid verifier replay verdict is not exact")
    replay_data = canonical_json(replay)
    replay_digest = hashlib.sha256(replay_data).hexdigest()
    _write_immutable(replay_dir / f"{replay_digest}.json", replay_data)
    return replay


def _write_custody_snapshot(
    receipt_root: Path,
    receipts: list[Any],
    *,
    run_id: str,
    attempt_id: str,
    compute_exit_code: int,
    termination_verified: bool,
    complete: bool,
    error: str | None,
    launch_nonce: str,
    stage_manifest_sha256: str,
    stage_source_head: str,
    worker_deadline_epoch: int,
    local_paid_replay: Mapping[str, Any] | None = None,
    remote_payload_custody: Mapping[str, Any] | None = None,
    provider_lease: Mapping[str, Any] | None = None,
) -> Path:
    receipt_map = _receipt_inventory(receipts, run_id=run_id) if receipts else {}
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "launch_nonce": launch_nonce,
        "stage_manifest_sha256": stage_manifest_sha256,
        "stage_source_head": stage_source_head,
        "worker_deadline_epoch": worker_deadline_epoch,
        "compute_exit_code": compute_exit_code,
        "termination_verified": termination_verified,
        "complete": complete,
        "extraction_profile": (
            "full_replay" if local_paid_replay is not None else
            "application" if complete else "partial"
        ),
        "error": error,
        "provider_lease": dict(provider_lease) if provider_lease is not None else None,
        "local_paid_replay": (
            dict(local_paid_replay) if local_paid_replay is not None else None
        ),
        "remote_payload_custody": (
            dict(remote_payload_custody)
            if remote_payload_custody is not None else None
        ),
        "receipts": [
            {
                "schema_version": _receipt_attribute(receipt, "schema_version"),
                "run_id": run_id,
                "relative_path": relative,
                "remote_path": _receipt_attribute(receipt, "remote_path"),
                "kind": _receipt_attribute(receipt, "kind"),
                "sha256": _receipt_attribute(receipt, "sha256"),
                "size": _receipt_attribute(receipt, "size"),
            }
            for relative, receipt in sorted(receipt_map.items())
        ],
    }
    data = canonical_json(payload)
    digest = hashlib.sha256(data).hexdigest()
    destination = receipt_root / "snapshots" / f"{digest}.json"
    _write_immutable(destination, data)
    return destination


def _sync_terminated_run(
    publisher: Any,
    *,
    run_id: str,
    attempt_id: str,
    local_root: Path,
    receipt_root: Path,
    n_prompts: int,
    image: str,
    launch_nonce: str,
    stage_manifest_sha256: str,
    stage_source_head: str,
    worker_deadline_epoch: int,
    replay_source_root: Path,
    compute_exit_code: int,
    termination_verified: bool,
    attempts: int = 20,
    retry_seconds: float = 1.0,
    provider_lease: Mapping[str, Any] | None = None,
    full_replay: bool = True,
) -> list[Any]:
    """Recover a terminated run from bounded eventually-consistent listings.

    ``full_replay=False`` is the paid controller's fast application path: it
    validates the complete remote receipt inventory but downloads only the
    write-up/evaluation bundle.  The offline recovery command retains the
    default full tensor retrieval and independent local replay.
    """

    aggregate: dict[str, Any] = {}
    last_error: Exception | None = None
    complete = False
    local_paid_replay: dict[str, Any] | None = None
    remote_payload_custody: dict[str, Any] | None = None
    max_attempts = attempts if compute_exit_code == 0 and termination_verified else min(attempts, 4)
    for number in range(max_attempts):
        try:
            quick_success = (
                compute_exit_code == 0 and termination_verified and not full_replay
            )
            if quick_success:
                observed = _read_remote_receipt_inventory(publisher, run_id=run_id)
            else:
                observed = sync_and_verify(
                    publisher,
                    run_id=run_id,
                    local_root=local_root,
                    relative_paths=None,
                    receipt_root=receipt_root,
                )
            for receipt in observed:
                relative = _receipt_attribute(receipt, "relative_path")
                prior = aggregate.get(relative)
                if prior is not None and prior != receipt:
                    raise PublishError(f"remote receipt identity changed across listings: {relative}")
                aggregate[relative] = receipt
            receipts = list(aggregate.values())
            if quick_success:
                _retain_remote_receipts(
                    receipts, run_id=run_id, receipt_root=receipt_root
                )
                local_paths = _application_extraction_paths(receipts, run_id=run_id)
                synced = sync_and_verify(
                    publisher,
                    run_id=run_id,
                    local_root=local_root,
                    relative_paths=sorted(local_paths),
                    receipt_root=receipt_root,
                )
                if {
                    _receipt_attribute(item, "relative_path") for item in synced
                } != local_paths:
                    raise PublishError("application extraction receipt set is incomplete")
                remote_payload_custody = _verify_deferred_remote_payloads(
                    publisher,
                    receipts,
                    local_paths,
                    run_id=run_id,
                )
                exact_receipts = synced
            else:
                local_paths = {
                    _receipt_attribute(item, "relative_path") for item in receipts
                }
                exact_receipts = receipts
            _verify_synced_tree_exact(
                local_root,
                exact_receipts,
                run_id=run_id,
                allowed_receipts=(receipts if quick_success else None),
            )
            if compute_exit_code == 0 and termination_verified:
                _verify_success_inventory(
                    receipts,
                    local_root,
                    run_id=run_id,
                    n_prompts=n_prompts,
                    attempt_id=attempt_id,
                    launch_nonce=launch_nonce,
                    stage_manifest_sha256=stage_manifest_sha256,
                    stage_source_head=stage_source_head,
                    worker_deadline_epoch=worker_deadline_epoch,
                    required_local_paths=(local_paths if quick_success else None),
                )
                _verify_extracted_paid_gate(
                    local_root,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    image=image,
                    launch_nonce=launch_nonce,
                    stage_manifest_sha256=stage_manifest_sha256,
                    stage_source_head=stage_source_head,
                    worker_deadline_epoch=worker_deadline_epoch,
                )
                if full_replay:
                    replay = _run_local_paid_replay(
                        local_root,
                        receipt_root,
                        source_root=replay_source_root,
                        run_id=run_id,
                        attempt_id=attempt_id,
                        n_prompts=n_prompts,
                        image=image,
                        launch_nonce=launch_nonce,
                        stage_manifest_sha256=stage_manifest_sha256,
                        stage_source_head=stage_source_head,
                        worker_deadline_epoch=worker_deadline_epoch,
                    )
                    replay_data = canonical_json(replay)
                    replay_digest = hashlib.sha256(replay_data).hexdigest()
                    replay_path = receipt_root / "local-replay" / f"{replay_digest}.json"
                    replay_size, replay_file_digest = _path_digest(
                        replay_path, label="local paid replay verdict"
                    )
                    if replay_file_digest != replay_digest:
                        raise PublishError("local paid replay custody digest is inconsistent")
                    local_paid_replay = {
                        "path": str(replay_path),
                        "size": replay_size,
                        "sha256": replay_digest,
                        "status": replay.get("status"),
                        "accepted": replay.get("accepted"),
                    }
                complete = True
                last_error = None
                break
            # A failed or termination-uncertain run is intentionally only a
            # partial snapshot. Two identical observations reduce eventual-
            # consistency omissions without ever promoting it to complete.
            if number > 0 and {
                _receipt_attribute(item, "relative_path") for item in observed
            } == set(aggregate):
                break
        except (PublishError, OSError, ValueError, TypeError) as exc:
            last_error = exc
        if number + 1 < max_attempts:
            time.sleep(min(retry_seconds * (2 ** number), 30.0))
    receipts = list(aggregate.values())
    _write_custody_snapshot(
        receipt_root,
        receipts,
        run_id=run_id,
        attempt_id=attempt_id,
        compute_exit_code=compute_exit_code,
        termination_verified=termination_verified,
        complete=complete,
        error=str(last_error) if last_error is not None else (
            None if complete else "partial snapshot: computation or termination was not successful"
        ),
        launch_nonce=launch_nonce,
        stage_manifest_sha256=stage_manifest_sha256,
        stage_source_head=stage_source_head,
        worker_deadline_epoch=worker_deadline_epoch,
        local_paid_replay=local_paid_replay,
        remote_payload_custody=remote_payload_custody,
        provider_lease=provider_lease,
    )
    if compute_exit_code == 0 and termination_verified and not complete:
        if last_error is not None:
            raise PublishError(f"successful run extraction did not converge: {last_error}") from last_error
        raise PublishError("successful run extraction did not converge")
    return receipts


def run_job(args: argparse.Namespace) -> int:
    """Run the remote entry point and unconditionally release the lease.

    This is the end-to-end path intended for a real experiment.  The legacy
    ``create`` command is offline-only; a computation must be launched
    through this function so the pinned entrypoint is transferred to a fresh
    host and termination plus local receipt synchronization are coupled to
    its exit.
    """

    _validate_run_dimensions(args)
    attempt_id, planned_artifact_run_id = _resolve_run_ids(args)
    if planned_artifact_run_id != attempt_id:
        raise SafetyError(
            "paid same-lineage retry is disabled; use one fresh attempt/artifact run ID"
        )
    if getattr(args, "confirm_launch", None) != attempt_id:
        raise SafetyError("paid run requires --confirm-launch equal to the exact lease attempt ID")
    args.attempt_id = attempt_id
    args.artifact_run_id = planned_artifact_run_id
    _assert_new_state_path(_validate_state_roots(args, attempt_id))
    # Validate command-line values that are independent of the lease before
    # downloading the stage or touching the account.  The full credential
    # check remains in ``_create_lease`` so stage verification still precedes
    # the first lease mutation.
    image = _validate_paid_image(getattr(args, "image", None))
    args.image = image
    _validate_paid_gpu(getattr(args, "gpu", DEFAULT_GPU))
    _positive_int(getattr(args, "disk", DEFAULT_DISK_GB), "disk")
    _validate_paid_workload_window(_lease_config(args))
    _finite_number(getattr(args, "ssh_timeout", 900), "ssh_timeout",
                   minimum=0, strict_minimum=True)
    ssh_connect_timeout = _finite_number(
        getattr(args, "ssh_connect_timeout", 30), "ssh_connect_timeout",
        minimum=0, strict_minimum=True,
    )
    if ssh_connect_timeout < 1:
        raise SafetyError("ssh_connect_timeout must be at least one second")
    sync_root_arg = _validate_path_argument(
        (args.sync_root if getattr(args, "sync_root", None) is not None
         else Path("artifacts/jlens/retrieved") / planned_artifact_run_id),
        "sync root", allow_missing=True,
    )
    if sync_root_arg.exists() and not sync_root_arg.is_dir():
        raise SafetyError("sync root is not a directory")
    receipt_root = _validate_path_argument(
        (args.receipt_root if getattr(args, "receipt_root", None) is not None
         else sync_root_arg.with_name(f"{sync_root_arg.name}.receipts")),
        "receipt root",
        allow_missing=True,
    )
    if receipt_root.exists() and not receipt_root.is_dir():
        raise SafetyError("receipt root is not a directory")
    receipt_absolute = Path(os.path.abspath(receipt_root))
    sync_absolute = Path(os.path.abspath(sync_root_arg))
    if (receipt_absolute == sync_absolute or receipt_absolute in sync_absolute.parents
            or sync_absolute in receipt_absolute.parents):
        raise SafetyError("receipt root and synchronized payload root must not overlap")
    # Persist and use one absolute interpretation. Recovery may run from a
    # different working directory and must still address the identical roots.
    sync_root_arg = sync_absolute
    receipt_root = receipt_absolute
    stage_path = _validate_stage_path(args.stage_path)
    results_repository = _validate_hf_repo(args.results, "results repository")
    staging_repository = _validate_hf_repo(args.staging, "staging repository")
    _hf_token_path, hf_token = _validate_hf_token_file(args.hf_token)
    args.expected_hf_token_sha256 = hashlib.sha256(
        hf_token.encode("utf-8")
    ).hexdigest()
    remote_script = _validate_remote_script(args.remote_script)
    if remote_script != "src/ouro_jlens/pod_entry.sh":
        raise SafetyError("remote runs require the pinned stage entrypoint")
    bootstrap_context = tempfile.TemporaryDirectory(prefix="jlens-controller-stage-")
    try:
        bootstrap = _prepare_stage_bootstrap(
            stage_path,
            staging_repository,
            Path(bootstrap_context.name),
            publisher=HfPublisher(staging_repository, token=hf_token),
        )
        _verify_controller_source_matches_stage(bootstrap)
        args.expected_stage_manifest_sha256 = bootstrap.manifest_sha256
        args.expected_stage_source_head = bootstrap.source_head
        args.launch_nonce = secrets.token_urlsafe(48)
        results_publisher = HfPublisher(results_repository, token=hf_token)
        args.results_preflight = _preflight_results_repository(
            results_publisher,
            repository=results_repository,
            attempt_id=attempt_id,
            artifact_run_id=planned_artifact_run_id,
            launch_nonce=args.launch_nonce,
            stage_manifest_sha256=bootstrap.manifest_sha256,
            stage_source_head=bootstrap.source_head,
            hf_token_sha256=args.expected_hf_token_sha256,
        )
        args.extraction_contract = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "artifact_run_id": planned_artifact_run_id,
            "launch_nonce": args.launch_nonce,
            "image": image,
            "stage_manifest_sha256": bootstrap.manifest_sha256,
            "stage_source_head": bootstrap.source_head,
            "staging_repository": staging_repository,
            "stage_path": stage_path,
            "results_repository": results_repository,
            "n_prompts": args.n_prompts,
            "hf_token_path": str(Path(os.path.abspath(_hf_token_path))),
            "hf_token_sha256": args.expected_hf_token_sha256,
            "sync_root": str(sync_absolute),
            "receipt_root": str(receipt_absolute),
            "results_preflight": dict(args.results_preflight),
        }
        state, supervisor = _create_lease(args)
    except BaseException:
        bootstrap_context.cleanup()
        raise
    # Initialize the cleanup path only from values already validated before
    # creation.  Every field returned by the paid lease is revalidated inside
    # the try/finally below, so a malformed or incomplete response can never
    # escape without an exact termination attempt.
    run_id = attempt_id
    artifact_run_id = planned_artifact_run_id
    launch_nonce = _validate_launch_nonce(args.launch_nonce)
    stage_manifest_sha256 = bootstrap.manifest_sha256
    stage_source_head = bootstrap.source_head
    provider_terminate_after = ""
    provider_deadline_epoch = 0.0
    worker_deadline_epoch = 0
    bootstrap_package = bootstrap.package
    sync_root = sync_root_arg
    job_rc = 2
    compute_rc = 2
    sync_rc = 0
    termination_ok = False
    try:
        if not isinstance(state, Mapping):
            raise SafetyError("lease response is not a state mapping")
        if validate_run_id(state.get("run_id")) != run_id:
            raise SafetyError("lease attempt identity differs from the requested attempt")
        if validate_run_id(state.get("artifact_run_id", run_id)) != artifact_run_id:
            raise SafetyError("lease artifact lineage differs from the requested lineage")
        if _validate_launch_nonce(state.get("launch_nonce")) != launch_nonce:
            raise SafetyError("lease launch nonce differs from the pre-deployment nonce")
        if (_validate_stage_manifest_digest(state.get("stage_manifest_sha256"))
                != stage_manifest_sha256):
            raise SafetyError("lease stage manifest differs from the verified bootstrap")
        if _validate_source_head(state.get("stage_source_head")) != stage_source_head:
            raise SafetyError("lease source HEAD differs from the verified bootstrap")
        raw_provider_deadline = state.get("provider_terminate_after")
        if (not isinstance(raw_provider_deadline, str)
                or PROVIDER_DATETIME_RE.fullmatch(raw_provider_deadline) is None):
            raise SafetyError("lease has no provider-enforced termination deadline")
        provider_terminate_after = raw_provider_deadline
        provider_deadline_epoch = datetime.strptime(
            provider_terminate_after, "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc).timestamp()
        remaining_provider_seconds = provider_deadline_epoch - time.time()
        max_runtime = _finite_number(
            getattr(args, "max_runtime", DEFAULT_MAX_RUNTIME),
            "max_runtime", minimum=0, strict_minimum=True,
        )
        if remaining_provider_seconds <= 0:
            raise SafetyError("provider-enforced termination deadline has already elapsed")
        if remaining_provider_seconds > max_runtime + 60.0:
            raise SafetyError("provider-enforced termination deadline exceeds the local runtime cap")
        expected_worker_deadline = _worker_deadline_epoch(
            provider_terminate_after, now=time.time()
        )
        raw_worker_deadline = state.get("worker_deadline_epoch")
        if (isinstance(raw_worker_deadline, bool)
                or not isinstance(raw_worker_deadline, int)
                or raw_worker_deadline != expected_worker_deadline):
            raise SafetyError("lease worker deadline is missing or not bound to the provider TTL")
        worker_deadline_epoch = raw_worker_deadline
        _require_remaining_worker_window(
            worker_deadline_epoch,
            reserve_seconds=WORKER_STARTUP_RESERVE_SECONDS,
        )
        bootstrap_dir = f"/tmp/ouro-jlens-{run_id}"
        ssh_ip = _validate_ssh_host(state.get("ssh_ip"))
        ssh_port = _validate_ssh_port(state.get("ssh_port"))
        host = f"root@{ssh_ip}"
        identity_file = _validate_identity_file(args.identity_file)
        ssh_connect_timeout = _finite_number(args.ssh_connect_timeout, "ssh_connect_timeout",
                                             minimum=0, strict_minimum=True)
        ssh_options = [
            "-i", str(identity_file),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={int(ssh_connect_timeout)}", "-p",
            str(ssh_port), host,
        ]
        transfer_timeout = max(
            1.0,
            min(
                WORKER_STARTUP_RESERVE_SECONDS,
                worker_deadline_epoch - time.time() - MIN_END_TO_END_WORKER_SECONDS,
            ),
        )
        prepare = subprocess.run(
            ["ssh", *ssh_options, "mkdir", "-p", bootstrap_dir],
            check=False, timeout=transfer_timeout,
        )
        if int(prepare.returncode) != 0:
            job_rc = int(prepare.returncode)
        else:
            scp_command = [
                "scp", "-q", "-r", "-i", str(identity_file),
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-P", str(ssh_port), str(bootstrap_package), f"{host}:{bootstrap_dir}",
            ]
            remaining_transfer = (
                worker_deadline_epoch - time.time() - MIN_END_TO_END_WORKER_SECONDS
            )
            if remaining_transfer <= 0:
                raise SafetyError(
                    "bootstrap preparation exhausted the worker startup reserve"
                )
            bootstrap_result = subprocess.run(
                scp_command,
                check=False,
                timeout=max(1.0, min(remaining_transfer, transfer_timeout)),
            )
            if int(bootstrap_result.returncode) != 0:
                job_rc = int(bootstrap_result.returncode)
            else:
                _require_remaining_worker_window(worker_deadline_epoch)
                command = ["ssh", *ssh_options, "env", f"PYTHONPATH={bootstrap_dir}",
                           f"RUN_ID={artifact_run_id}", f"JLENS_ATTEMPT_ID={run_id}",
                           f"JLENS_LAUNCH_NONCE={launch_nonce}",
                           f"EXPECTED_STAGE_MANIFEST_SHA256={stage_manifest_sha256}",
                           f"EXPECTED_STAGE_SOURCE_HEAD={stage_source_head}",
                           f"JLENS_WORKER_DEADLINE_EPOCH={worker_deadline_epoch}",
                           f"RESULTS={results_repository}", f"STAGE_PATH={stage_path}",
                           f"STAGING={staging_repository}", f"N_PROMPTS={args.n_prompts}",
                           f"DIM_BATCH={args.dim_batch}", f"{TORCH_VERSION_ENV}={TORCH_VERSION}",
                           f"TORCH_VERSION={TORCH_VERSION}", f"JLENS_TORCH_VERSION={TORCH_VERSION}",
                           f"JLENS_IMAGE_DIGEST={image}", "bash",
                           f"{bootstrap_dir}/ouro_jlens/pod_entry.sh"]
                remote_timeout = max(1.0, provider_deadline_epoch - time.time())
                result = subprocess.run(command, check=False, timeout=remote_timeout)
                job_rc = int(result.returncode)
    except subprocess.TimeoutExpired as exc:
        print(f"ERROR: remote command exceeded provider lease deadline: {exc}", file=sys.stderr)
        job_rc = 124
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: remote command failed to start: {exc}", file=sys.stderr)
        job_rc = 127
    finally:
        compute_rc = job_rc
        try:
            supervisor.terminate_verified()
            termination_ok = True
        except SafetyError as exc:
            print(f"ERROR: lease termination was not verified: {exc}", file=sys.stderr)
            if job_rc == 0:
                job_rc = 2
        unit = state.get("monitor_unit")
        # Keep a disconnect-surviving monitor alive when termination was not
        # proven.  Stopping it here would turn an uncertainty into an
        # unmanaged paid pod; the unit can retry cleanup on its own.
        if unit and termination_ok:
            try:
                stop_systemd_monitor(str(unit))
            except SafetyError as exc:
                print(f"ERROR: monitor shutdown was not verified: {exc}", file=sys.stderr)
                if job_rc == 0:
                    job_rc = 2
        # Sync after termination so the controller retains evidence even when
        # the remote shell or pod disappears.  Failure is never hidden.
        try:
            publisher = HfPublisher(results_repository, token=hf_token)
            # The concurrent publisher is the authority for the union of
            # acknowledged receipts.  In particular, never fall back to the
            # controller's stale/local cache when its terminal index is
            # missing after a partial run.
            _sync_terminated_run(
                publisher,
                run_id=artifact_run_id,
                attempt_id=run_id,
                local_root=sync_root,
                receipt_root=receipt_root,
                n_prompts=args.n_prompts,
                image=image,
                launch_nonce=launch_nonce,
                stage_manifest_sha256=stage_manifest_sha256,
                stage_source_head=stage_source_head,
                worker_deadline_epoch=worker_deadline_epoch,
                replay_source_root=bootstrap.root,
                compute_exit_code=compute_rc,
                termination_verified=termination_ok,
                provider_lease={
                    "requested_gpu": args.gpu,
                    "pod_id": state.get("pod_id"),
                    "machine_id": state.get("machine_id"),
                    "pod_name": state.get("pod_name"),
                    "image": image,
                    "gpu_display_name": state.get("provider_gpu_display_name"),
                    "provider_terminate_after": provider_terminate_after,
                },
                # The application needs the verified reports/evaluations
                # immediately, not a blocking 27+ GiB replay download.  Every
                # tensor remains remotely acknowledged and ``recover`` later
                # performs the complete local replay without paid compute.
                full_replay=False,
            )
        except (PublishError, OSError, ValueError, TypeError) as exc:
            print(f"ERROR: local receipt synchronization failed: {exc}", file=sys.stderr)
            sync_rc = 2
        bootstrap_context.cleanup()
    if job_rc != 0:
        return job_rc
    return sync_rc


def monitor(args: argparse.Namespace) -> None:
    state_file = Path(args.state)
    state = _read_json(state_file)
    _verify_monitor_bundle(state_file, state)
    selector = state.get("runpod_key_sha256")
    nonce = state.get("monitor_nonce")
    if (not isinstance(selector, str) or not re.fullmatch(r"[0-9a-f]{64}", selector)
            or not isinstance(nonce, str) or not nonce):
        raise SafetyError("lease state has no authenticated monitor identity")
    # A detached monitor never trusts a shell-only key inherited from the
    # creator.  Resolve exactly the fingerprint committed to the state file.
    os.environ.pop("RUNPOD_API_KEY", None)
    os.environ[KEY_SELECTOR_ENV] = selector
    _select_file_key(KEY_FILE, selector)
    client = RunPodClient(credential_fingerprint=selector, key_file=KEY_FILE,
                          file_only=True)
    config = SafetyConfig(min_balance=float(state.get("min_balance", DEFAULT_MIN_BALANCE)),
                          max_spend=float(state.get("max_spend", DEFAULT_MAX_SPEND)),
                          max_runtime=float(state.get("max_runtime", DEFAULT_MAX_RUNTIME)),
                          poll_seconds=float(state.get("poll_seconds", args.poll_seconds)),
                          max_api_failures=int(state.get("max_api_failures", args.max_api_failures)),
                          pending_timeout=float(state.get("pending_timeout", args.pending_timeout)),
                          termination_attempts=int(state.get("termination_attempts", args.termination_attempts)),
                          termination_poll_seconds=float(state.get("termination_poll_seconds", args.termination_poll_seconds)))
    LeaseSupervisor(client, state_file, config).monitor(once=args.once)


def _pod_for_destroy(client: RunPodClient, pod_id: str) -> dict[str, Any]:
    pods = client.pods()
    matches = [p for p in pods if p.get("id") == pod_id]
    if len(matches) != 1:
        raise IdentityMismatch(f"expected one exact pod id, found {len(matches)}")
    if not str(matches[0].get("name", "")).startswith(NAME_PREFIX):
        raise IdentityMismatch("refusing to destroy a pod outside the JLens namespace")
    return matches[0]


def destroy(args: argparse.Namespace) -> None:
    client = RunPodClient()
    if args.pod_id == "all":
        targets = [p for p in client.pods() if str(p.get("name", "")).startswith(NAME_PREFIX)]
    else:
        targets = [_pod_for_destroy(client, args.pod_id)]
    config = SafetyConfig(min_balance=HARD_MIN_BALANCE, max_spend=DEFAULT_MAX_SPEND, max_runtime=DEFAULT_MAX_RUNTIME,
                         poll_seconds=args.poll_seconds, termination_attempts=args.termination_attempts,
                         termination_poll_seconds=args.termination_poll_seconds)
    for pod in targets:
        state = {"run_id": "destroy", "pod_name": pod.get("name"), "pod_id": pod["id"],
                 "machine_id": pod.get("machineId")}
        # _terminate_verified only needs an in-memory identity, so avoid
        # creating a fake persistent run namespace for the panic button.
        supervisor = LeaseSupervisor(client, Path("/dev/null"), config)
        supervisor._terminate_verified(state, pod_id=pod["id"])
    print(f"verified termination for {len(targets)} pod(s)")


def status(args: argparse.Namespace) -> None:
    d = RunPodClient().gql("{ myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr runtime { uptimeInSeconds } } } }")
    myself = d["myself"]
    print(f"balance ${float(myself['clientBalance']):.2f}  spend/hr ${myself.get('currentSpendPerHr', 0)}")
    for pod in myself.get("pods", []):
        if not str(pod.get("name", "")).startswith(NAME_PREFIX):
            continue
        runtime = pod.get("runtime") or {}
        uptime = float(runtime.get("uptimeInSeconds") or 0)
        rate = float(pod.get("costPerHr") or 0)
        print(f"  {pod.get('id')} {pod.get('name')} {pod.get('desiredStatus')} ${rate}/hr "
              f"up {int(uptime // 60)}m spent ~${rate * uptime / 3600:.2f}")


def ssh(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.pod_file).read_text())
    print(f"ssh -i {args.identity_file} -o StrictHostKeyChecking=no -p {data['port']} root@{data['ip']}")


def recover(args: argparse.Namespace) -> int:
    """Repeat exact post-termination extraction without RunPod access."""

    # Import lazily because recover.py intentionally reuses this module's
    # validators and custody gate. The command itself never selects a RunPod
    # credential or constructs a RunPod client.
    from ouro_jlens.recover import recover_terminated_run

    return recover_terminated_run(args)


def _add_safety_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-balance", type=float, default=DEFAULT_MIN_BALANCE)
    parser.add_argument("--max-spend", type=float, default=DEFAULT_MAX_SPEND)
    parser.add_argument("--max-runtime", type=float, default=DEFAULT_MAX_RUNTIME)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--max-api-failures", type=int, default=DEFAULT_MAX_API_FAILURES)
    parser.add_argument("--pending-timeout", type=float, default=DEFAULT_PENDING_TIMEOUT)
    parser.add_argument("--termination-attempts", type=int, default=DEFAULT_TERMINATION_ATTEMPTS)
    parser.add_argument("--termination-poll-seconds", type=float, default=DEFAULT_TERMINATION_POLL_SECONDS)


def _add_credential_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runpod-key-sha256",
        help="SHA-256 of the exact RunPod key to select when the credential file contains rotations",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="offline dry-run only; paid provisioning uses run")
    c.add_argument("--gpu", default=DEFAULT_GPU)
    c.add_argument("--disk", type=int, default=DEFAULT_DISK_GB)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--run-id")
    c.add_argument("--attempt-id", help="fresh lease identity for a resumable artifact lineage")
    c.add_argument("--artifact-run-id", "--artifact-lineage-id", dest="artifact_run_id",
                   help="resumable artifact lineage (defaults to run ID)")
    c.add_argument("--image", "--image-ref", dest="image",
                   help="immutable container image reference name@sha256:<64>")
    c.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    c.add_argument("--pubkey", type=Path, default=PUBKEY)
    c.add_argument("--hf-token", type=Path, default=Path.home() / "Documents" / "Credentials" / "hf-token.txt")
    c.add_argument("--identity-file", default="~/.ssh/id_ed25519_dsv4_b300")
    c.add_argument("--ssh-timeout", type=float, default=900)
    _add_safety_options(c)
    _add_credential_selector(c)
    c.set_defaults(fn=create)

    r = sub.add_parser("run", help="provision, run remotely, sync receipts, and release the lease")
    r.add_argument("--gpu", default=DEFAULT_GPU)
    r.add_argument("--disk", type=int, default=DEFAULT_DISK_GB)
    r.add_argument("--run-id")
    r.add_argument("--attempt-id", help="fresh lease identity for a resumable artifact lineage")
    r.add_argument("--artifact-run-id", "--artifact-lineage-id", dest="artifact_run_id",
                   help="resumable artifact lineage (defaults to run ID)")
    r.add_argument("--confirm-launch",
                   help="explicit paid-launch interlock; must equal the lease attempt ID")
    r.add_argument("--image", "--image-ref", dest="image",
                   help="immutable container image reference name@sha256:<64>")
    r.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    r.add_argument("--pubkey", type=Path, default=PUBKEY)
    r.add_argument("--hf-token", type=Path, default=Path.home() / "Documents" / "Credentials" / "hf-token.txt")
    r.add_argument("--identity-file", default="~/.ssh/id_ed25519_dsv4_b300")
    r.add_argument("--ssh-timeout", type=float, default=900)
    r.add_argument("--ssh-connect-timeout", type=float, default=30)
    r.add_argument("--remote-script", default="src/ouro_jlens/pod_entry.sh")
    r.add_argument("--staging", default="Vykos/ouro-jlens-staging",
                   help="private HF repository containing the stage archive")
    r.add_argument("--stage-path", required=True,
                   help="digest-addressed stage path emitted by stage_upload.sh")
    r.add_argument("--results", required=True, help="HF result repository")
    r.add_argument("--n-prompts", type=int, default=100)
    r.add_argument("--dim-batch", type=int, default=32)
    r.add_argument("--receipt-root", type=Path)
    r.add_argument("--sync-root", type=Path,
                   help="local payload root (defaults to a run-lineage-specific directory)")
    _add_safety_options(r)
    _add_credential_selector(r)
    r.set_defaults(fn=run_job)

    m = sub.add_parser("monitor")
    m.add_argument("--state", type=Path, required=True)
    m.add_argument("--once", action="store_true")
    _add_safety_options(m)
    m.set_defaults(fn=monitor)

    s = sub.add_parser("status")
    _add_credential_selector(s)
    s.set_defaults(fn=status)

    x = sub.add_parser(
        "recover",
        help="repeat verified artifact extraction for one terminated lease (no RunPod access)",
    )
    x.add_argument("--state", dest="state_file", type=Path, required=True)
    x.set_defaults(fn=recover)

    h = sub.add_parser("ssh")
    h.add_argument("--pod-file", type=Path, default=Path("artifacts/jlens/pod.json"))
    h.add_argument("--identity-file", default="~/.ssh/id_ed25519_dsv4_b300")
    h.set_defaults(fn=ssh)

    d = sub.add_parser("destroy")
    d.add_argument("pod_id", help='exact pod id, or "all" for JLens-named pods')
    _add_safety_options(d)
    _add_credential_selector(d)
    d.set_defaults(fn=destroy)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if getattr(args, "cmd", None) in {"run", "monitor", "status", "destroy"}:
            _activate_key_selector(args)
        result = args.fn(args)
        return int(result or 0)
    except (SafetyError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
