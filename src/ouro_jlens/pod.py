"""Fail-closed RunPod lease controller for the JLens experiment.

The only code path that contains a create mutation is :func:`create`, and it
does not issue that mutation until a durable pending lease and a disconnect-
surviving user monitor have been installed.  The monitor owns the lease after
creation and continuously checks the balance floor, accrued spend, runtime,
and exact pod identity.  Every exit path attempts termination and verifies
that the pod disappeared from the account listing.

The default CLI is intentionally conservative.  ``create --dry-run`` is
entirely offline; no credential, API, SSH, HF, or systemd call is made.

  python src/ouro_jlens/pod.py create --dry-run
  python src/ouro_jlens/pod.py create
  python src/ouro_jlens/pod.py monitor --state artifacts/jlens/runs/<run_id>/state.json
  python src/ouro_jlens/pod.py status
  python src/ouro_jlens/pod.py destroy <pod-id>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ouro_jlens.publish import (
    HfPublisher,
    HF_REPO_RE,
    MissingRemote,
    PublishError,
    extract_stage_tar,
    payload_remote_path,
    read_index,
    safe_relative,
    sync_and_verify,
    validate_run_id,
    verify_stage_tar,
)


API = "https://api.runpod.io/graphql"
KEY_FILE = Path.home() / "Documents" / "Credentials" / "fidelio.txt"
PUBKEY = Path.home() / ".ssh" / "id_ed25519_dsv4_b300.pub"
NAME = "ouro-jlens"
NAME_PREFIX = f"{NAME}-"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_STATE_ROOT = Path("artifacts/jlens/runs")
DEFAULT_GPU = "NVIDIA B300 SXM6 AC"
DEFAULT_DISK_GB = 100
DEFAULT_MIN_BALANCE = 20.0
DEFAULT_MAX_SPEND = 20.0
DEFAULT_MAX_RUNTIME = 2 * 60 * 60
DEFAULT_POLL_SECONDS = 15.0
DEFAULT_MAX_API_FAILURES = 3
DEFAULT_PENDING_TIMEOUT = 180.0
DEFAULT_TERMINATION_ATTEMPTS = 12
DEFAULT_TERMINATION_POLL_SECONDS = 5.0


class SafetyError(RuntimeError):
    """A safety invariant prevents continuing."""


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
    if env_key:
        if not re.fullmatch(r"rpa_[A-Za-z0-9]+", env_key):
            raise SafetyError("RUNPOD_API_KEY has an invalid format")
        return env_key
    try:
        text = key_file.read_text()
    except OSError as exc:
        raise SafetyError(f"cannot read RunPod credential file: {key_file}") from exc
    keys = re.findall(r"(?<![A-Za-z0-9_])rpa_[A-Za-z0-9]+(?![A-Za-z0-9_])", text)
    if len(keys) != 1:
        raise SafetyError(f"credential file must contain exactly one RunPod API key: {key_file}")
    return keys[0]


def gql(query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Perform one API request, converting all uncertainty to APIUncertain."""

    payload: dict[str, Any] = {"query": query}
    if variables is not None:
        payload["variables"] = dict(variables)
    request = urllib.request.Request(
        API,
        json.dumps(payload).encode(),
        {
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key()}",
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


class RunPodClient:
    """Small injectable API facade; tests can provide a fake ``gql`` callable."""

    def __init__(self, gql_fn: Callable[..., dict[str, Any]] | None = None):
        self.gql = gql_fn or gql
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

    def deploy(self, *, name: str, gpu: str, disk: int, public_key: str, hf_token: str,
               _guard: object | None = None) -> dict[str, Any]:
        """Create exactly one uniquely named pod after supervision is armed."""

        if self._deployment_guard is None or _guard is not self._deployment_guard:
            raise SafetyError("deployment requires an armed supervised lease")
        # Consume the capability before the network mutation.  A caller must
        # establish a fresh durable lease and monitor before any retry.
        self._deployment_guard = None
        # The values are sent as GraphQL variables.  They never enter a
        # query string, logs, exception text, or dry-run output.
        query = """mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) { id machineId costPerHr desiredStatus name }
        }"""
        variables = {
            "input": {
                "cloudType": "ALL",
                "gpuCount": 1,
                "gpuTypeId": gpu,
                "name": name,
                "imageName": IMAGE,
                "containerDiskInGb": disk,
                "volumeInGb": 0,
                "minVcpuCount": 16,
                "minMemoryInGb": 64,
                "ports": "22/tcp",
                "env": [{"key": "PUBLIC_KEY", "value": public_key},
                        {"key": "HF_TOKEN", "value": hf_token}],
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
    except BaseException:
        try:
            os.unlink(tmp_name)
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
        if (not all(math.isfinite(float(value)) for value in
                     (self.min_balance, self.max_spend, self.max_runtime,
                      self.poll_seconds, self.pending_timeout, self.termination_poll_seconds))
                or self.min_balance < 0 or self.max_spend <= 0 or self.max_runtime <= 0):
            raise ValueError("safety limits must be positive (balance floor may be zero)")
        if self.poll_seconds <= 0 or self.max_api_failures < 1 or self.pending_timeout <= 0:
            raise ValueError("invalid supervisor timing")
        if self.termination_attempts < 1 or self.termination_poll_seconds < 0:
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
            run_id = validate_run_id(str(state["run_id"]))
            if str(state["pod_name"]) != _name_for(run_id):
                raise ValueError("pod name is not bound to the run identity")
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

    def _save(self, state: Mapping[str, Any]) -> None:
        _atomic_json(self.state_file, state)

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
        if expected_machine and observed_machine != expected_machine:
            raise IdentityMismatch("pod id resolved to a different machine")
        return pod

    @staticmethod
    def _named_pods(state: Mapping[str, Any], pods: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        name = state.get("pod_name")
        return [dict(p) for p in pods if p.get("name") == name]

    def _discover_pending(self, state: dict[str, Any]) -> dict[str, Any] | None:
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
        state.update({"pod_id": pod["id"], "machine_id": pod.get("machineId"),
                      "cost_per_hr": float(pod.get("costPerHr") or 0), "status": "active",
                      "bound_at": self.clock()})
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
        last_error: BaseException | None = None
        for attempt in range(self.config.termination_attempts):
            try:
                pods = self.client.pods()
                matches = self._named_pods(state, pods)
                if len(matches) > 1:
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
                    pod_id = matches[0].get("id")
                    if not pod_id:
                        last_error = IdentityMismatch("exact-name pod has no id")
                    else:
                        try:
                            self._terminate_verified(state, pod_id=pod_id, require_name=True)
                        except SafetyError as exc:
                            last_error = exc
                else:
                    # Require one final successful absence observation after
                    # the possible create response has settled. This closes
                    # the eventual-consistency gap where a pod appears after
                    # an uncertain mutation response.
                    if attempt + 1 >= self.config.termination_attempts:
                        return True
            except (APIUncertain, OSError, TimeoutError) as exc:
                last_error = exc
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
            except (APIUncertain, OSError, TimeoutError):
                initial_matches = []
            if initial_matches and initial_matches[0].get("name") != state.get("pod_name"):
                raise IdentityMismatch("refusing to terminate a pod with a different identity")
        last_error: BaseException | None = None
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
                    if require_name and any(p.get("name") != state.get("pod_name") for p in exact):
                        raise IdentityMismatch("termination lookup found a different pod identity")
                else:
                    if require_name and any(p.get("name") == state.get("pod_name") for p in pods):
                        raise IdentityMismatch("termination id disappeared while another same-name pod remained")
                    self.output(f"terminated and verified pod {target}")
                    return
            except (APIUncertain, OSError, TimeoutError) as exc:
                last_error = exc
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
                              "monitor_status": "healthy"})
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
                ["systemd-run", "--user", "--version"],
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


def start_systemd_monitor(state_file: Path, *, python: str = sys.executable,
                          script: Path | None = None, runner: Callable[..., Any] | None = None) -> str:
    """Start a persistent monitor unit and return its unit name."""

    state_file = Path(state_file).resolve()
    state = _read_json(state_file)
    run_id = validate_run_id(str(state["run_id"]))
    runner = runner or subprocess.run
    systemd_user_preflight(runner=runner)
    unit = f"ouro-jlens-monitor-{run_id}.service"
    script = script or Path(__file__).resolve()
    command = ["systemd-run", "--user", "--unit", unit, "--collect",
               "--property=Restart=on-failure", "--property=RestartSec=5s",
               "--property=TimeoutStopSec=30s", python, str(script), "monitor",
               "--state", str(state_file)]
    try:
        result = runner(command, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError("could not start disconnect-safe monitor") from exc
    if getattr(result, "returncode", None) != 0:
        raise SafetyError("systemd did not accept the disconnect-safe monitor")
    try:
        check = runner(["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
                       check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError("could not verify the disconnect-safe monitor") from exc
    if (getattr(check, "returncode", None) != 0
            or (getattr(check, "stdout", "") or "").strip() not in {"active", "activating"}):
        try:
            runner(["systemctl", "--user", "stop", unit], check=False,
                   capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass
        raise SafetyError("disconnect-safe monitor is not active")
    return unit


def wait_for_ssh(client: RunPodClient, pod_id: str, *, timeout: float = 900,
                 poll_seconds: float = 15, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 output: Callable[[str], None] = print) -> dict[str, Any]:
    """Wait for SSH while preserving the lease's exact identity."""

    started = clock()
    while clock() - started < timeout:
        pods = client.pods()
        pod = next((p for p in pods if p.get("id") == pod_id), None)
        if pod is None:
            raise IdentityMismatch("leased pod disappeared while waiting for SSH")
        ports = [x for x in ((pod.get("runtime") or {}).get("ports") or [])
                 if x.get("privatePort") == 22 and x.get("isIpPublic")]
        output(f"  {int(clock() - started):>4}s status={pod.get('desiredStatus')} ssh={'yes' if ports else 'no'}")
        if ports:
            endpoint = ports[0]
            host = endpoint.get("ip")
            try:
                public_port = int(endpoint.get("publicPort"))
            except (TypeError, ValueError) as exc:
                raise SafetyError("RunPod returned an invalid SSH port") from exc
            if (not isinstance(host, str) or
                    not re.fullmatch(r"[A-Za-z0-9_.:-]+", host) or
                    not 1 <= public_port <= 65535):
                raise SafetyError("RunPod returned an unsafe SSH endpoint")
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
    config = SafetyConfig(min_balance=0, max_spend=DEFAULT_MAX_SPEND,
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


def _prepare_stage_bootstrap(
    stage_path: str,
    staging_repository: str,
    destination: Path,
    *,
    publisher=None,
) -> Path:
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
    if (
        verified.get("pinned_inputs", {}).get("source_policy") != "clean_git_payload"
        or not isinstance(source, dict)
        or not isinstance(source.get("head"), str)
        or source.get("status") != ""
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
    return package


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


def _new_state(run_id: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": validate_run_id(run_id),
        "pod_name": _name_for(run_id),
        "gpu": args.gpu,
        "image": IMAGE,
        "created_at": time.time(),
        "status": "pending",
        "min_balance": args.min_balance,
        "max_spend": args.max_spend,
        "max_runtime": args.max_runtime,
        "poll_seconds": args.poll_seconds,
        "max_api_failures": args.max_api_failures,
        "pending_timeout": args.pending_timeout,
        "termination_attempts": args.termination_attempts,
        "termination_poll_seconds": args.termination_poll_seconds,
        "monitor_status": "arming",
    }


def _lease_config(args: argparse.Namespace) -> SafetyConfig:
    return SafetyConfig(min_balance=args.min_balance, max_spend=args.max_spend,
                        max_runtime=args.max_runtime, poll_seconds=args.poll_seconds,
                        max_api_failures=args.max_api_failures, pending_timeout=args.pending_timeout,
                        termination_attempts=args.termination_attempts,
                        termination_poll_seconds=args.termination_poll_seconds)


def _create_lease(args: argparse.Namespace, *, client: RunPodClient | None = None,
                  monitor_launcher: Callable[[Path], str] | None = None) -> tuple[dict[str, Any], LeaseSupervisor]:
    client = client or RunPodClient()
    monitor_launcher = monitor_launcher or start_systemd_monitor
    run_id = validate_run_id(args.run_id or _default_run_id())
    state_file = state_path(run_id, args.state_root)
    state = _new_state(run_id, args)
    lease_config = _lease_config(args)

    # Preflight all read-only checks before arming the lease.  In particular,
    # a negative balance is a hard no-create condition.
    balance = client.balance()
    if balance < 0 or balance < lease_config.min_balance:
        raise SafetyError(f"balance ${balance:.2f} is below the ${lease_config.min_balance:.2f} floor; not creating")
    state["balance_before"] = balance
    live = [p for p in client.pods() if str(p.get("name", "")).startswith(NAME_PREFIX)]
    if live and not args.allow_second:
        raise SafetyError(f"a JLens pod already exists ({live[0].get('id')}); destroy it first")

    _atomic_json(state_file, state)
    supervisor = LeaseSupervisor(client, state_file, lease_config)
    monitor_unit: str | None = None
    pod_id: str | None = None
    try:
        monitor_unit = monitor_launcher(state_file)
        state["monitor_unit"] = monitor_unit
        _atomic_json(state_file, state)
        # Recheck immediately before the paid mutation to narrow the balance
        # race.  No monitor or mutation occurs when this check is uncertain.
        balance = client.balance()
        if balance < 0 or balance < lease_config.min_balance:
            raise SafetyError(f"balance ${balance:.2f} is below the floor; not creating")
        state["balance_before"] = balance
        _atomic_json(state_file, state)
        launch_kwargs = {
            "name": state["pod_name"],
            "gpu": args.gpu,
            "disk": args.disk,
            "public_key": _read_launch_secret(args.pubkey, "public key"),
            "hf_token": _read_launch_secret(args.hf_token, "HF token"),
        }
        if isinstance(client, RunPodClient):
            client._arm_deployment(supervisor)
            pod = client.deploy(**launch_kwargs, _guard=supervisor)
        else:
            # Dependency-injected fakes keep the same observable call surface
            # for offline tests; production clients always require the guard
            # above.
            pod = client.deploy(**launch_kwargs)
        pod_id = str(pod["id"])
        cost_per_hr = float(pod.get("costPerHr") or 0)
        if not math.isfinite(cost_per_hr) or cost_per_hr <= 0:
            raise SafetyError("deploy response omitted a valid pod cost")
        state.update({"pod_id": pod_id, "machine_id": pod.get("machineId"),
                      "cost_per_hr": cost_per_hr, "status": "active",
                      "bound_at": time.time()})
        _atomic_json(state_file, state)
        # Wait is itself identity checked; on any error the finally path
        # invokes termination and refuses to hand out a billing pod.
        ready = wait_for_ssh(client, pod_id, timeout=args.ssh_timeout,
                             poll_seconds=args.poll_seconds)
        LeaseSupervisor._exact_pod(state, [ready["pod"]])
        port = ready["port"]
        state.update({"ssh_ip": port["ip"], "ssh_port": port["publicPort"]})
        _atomic_json(state_file, state)
        print(f"run_id={run_id} pod={pod_id} monitor=armed")
        print(f"SSH READY:\n  ssh -i {args.identity_file} -o StrictHostKeyChecking=no -p "
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
                _atomic_json(state_file, state)
            except OSError:
                pass
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
    if args.dry_run:
        run_id = validate_run_id(args.run_id or "dry-run")
        print("DRY RUN: no API, credential, pod, SSH, HF, or systemd action")
        print(json.dumps({"run_id": run_id, "pod_name": _name_for(run_id), "gpu": args.gpu,
                          "disk_gb": args.disk, "min_balance": args.min_balance,
                          "max_spend": args.max_spend, "max_runtime": args.max_runtime}, sort_keys=True))
        return
    _create_lease(args)


def stop_systemd_monitor(unit: str, *, runner: Callable[..., Any] | None = None) -> None:
    """Stop and verify the transient monitor after the lease is released."""

    if not unit:
        return
    runner = runner or subprocess.run
    try:
        result = runner(["systemctl", "--user", "stop", unit], check=False,
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


def _receipt_relatives(receipt_root: Path) -> list[str]:
    suffix = ".receipt.json"
    if not receipt_root.is_dir():
        return []
    relatives = []
    for path in sorted(receipt_root.rglob(f"*{suffix}")):
        if path.is_symlink():
            raise PublishError(f"local receipt is a symlink: {path}")
        if path.is_file():
            relatives.append(path.relative_to(receipt_root).as_posix()[:-len(suffix)])
    return relatives


def run_job(args: argparse.Namespace) -> int:
    """Run the remote entry point and unconditionally release the lease.

    This is the end-to-end path intended for a real experiment.  The
    provision-only ``create`` command remains useful for manual inspection,
    but a successful computation should be launched through this function so
    the pinned entrypoint is transferred to a fresh host and termination plus
    local receipt synchronization are coupled to its exit.
    """

    stage_path = _validate_stage_path(args.stage_path)
    results_repository = _validate_hf_repo(args.results, "results repository")
    staging_repository = _validate_hf_repo(args.staging, "staging repository")
    remote_script = _validate_remote_script(args.remote_script)
    if remote_script != "src/ouro_jlens/pod_entry.sh":
        raise SafetyError("remote runs require the pinned stage entrypoint")
    bootstrap_context = tempfile.TemporaryDirectory(prefix="jlens-controller-stage-")
    try:
        bootstrap_package = _prepare_stage_bootstrap(
            stage_path, staging_repository, Path(bootstrap_context.name)
        )
        state, supervisor = _create_lease(args)
    except BaseException:
        bootstrap_context.cleanup()
        raise
    run_id = state["run_id"]
    receipt_root = Path(args.receipt_root or Path(args.state_root) / run_id / "receipts")
    job_rc = 2
    sync_rc = 0
    termination_ok = False
    try:
        bootstrap_dir = f"/tmp/ouro-jlens-{run_id}"
        host = f"root@{state['ssh_ip']}"
        ssh_options = [
            "-i", str(Path(args.identity_file).expanduser()),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={int(args.ssh_connect_timeout)}", "-p",
            str(state["ssh_port"]), host,
        ]
        prepare = subprocess.run(["ssh", *ssh_options, "mkdir", "-p", bootstrap_dir], check=False)
        if int(prepare.returncode) != 0:
            job_rc = int(prepare.returncode)
        else:
            scp_command = [
                "scp", "-q", "-r", "-i", str(Path(args.identity_file).expanduser()),
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "-P", str(state["ssh_port"]), str(bootstrap_package), f"{host}:{bootstrap_dir}",
            ]
            bootstrap = subprocess.run(scp_command, check=False)
            if int(bootstrap.returncode) != 0:
                job_rc = int(bootstrap.returncode)
            else:
                command = ["ssh", *ssh_options, "env", f"PYTHONPATH={bootstrap_dir}",
                           f"RUN_ID={run_id}", f"RESULTS={results_repository}", f"STAGE_PATH={stage_path}",
                           f"STAGING={staging_repository}", f"N_PROMPTS={args.n_prompts}",
                           f"DIM_BATCH={args.dim_batch}", "bash", f"{bootstrap_dir}/ouro_jlens/pod_entry.sh"]
                result = subprocess.run(command, check=False)
                job_rc = int(result.returncode)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: remote command failed to start: {exc}", file=sys.stderr)
        job_rc = 127
    finally:
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
            publisher = HfPublisher(results_repository)
            try:
                paths = read_index(publisher.read_file(payload_remote_path(run_id, "receipt-index.json")),
                                   run_id=run_id)
                paths.append("receipt-index.json")
            except MissingRemote:
                if job_rc == 0:
                    raise PublishError("successful remote run has no terminal receipt index")
                paths = _receipt_relatives(receipt_root)
            receipts = sync_and_verify(publisher, run_id=run_id, local_root=args.sync_root,
                                       relative_paths=paths)
            if job_rc == 0:
                kinds = {receipt.kind for receipt in receipts}
                required = {"shard", "sidecar", "merged_lens", "eval_output", "heartbeat"}
                missing = sorted(required - kinds)
                if missing:
                    raise PublishError(f"successful run is missing receipt kinds: {', '.join(missing)}")
        except (PublishError, OSError, ValueError) as exc:
            print(f"ERROR: local receipt synchronization failed: {exc}", file=sys.stderr)
            sync_rc = 2
        bootstrap_context.cleanup()
    if job_rc != 0:
        return job_rc
    return sync_rc


def monitor(args: argparse.Namespace) -> None:
    state = _read_json(Path(args.state))
    config = SafetyConfig(min_balance=float(state.get("min_balance", DEFAULT_MIN_BALANCE)),
                          max_spend=float(state.get("max_spend", DEFAULT_MAX_SPEND)),
                          max_runtime=float(state.get("max_runtime", DEFAULT_MAX_RUNTIME)),
                          poll_seconds=float(state.get("poll_seconds", args.poll_seconds)),
                          max_api_failures=int(state.get("max_api_failures", args.max_api_failures)),
                          pending_timeout=float(state.get("pending_timeout", args.pending_timeout)),
                          termination_attempts=int(state.get("termination_attempts", args.termination_attempts)),
                          termination_poll_seconds=float(state.get("termination_poll_seconds", args.termination_poll_seconds)))
    LeaseSupervisor(RunPodClient(), Path(args.state), config).monitor(once=args.once)


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
    config = SafetyConfig(min_balance=0, max_spend=DEFAULT_MAX_SPEND, max_runtime=DEFAULT_MAX_RUNTIME,
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


def _add_safety_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-balance", type=float, default=DEFAULT_MIN_BALANCE)
    parser.add_argument("--max-spend", type=float, default=DEFAULT_MAX_SPEND)
    parser.add_argument("--max-runtime", type=float, default=DEFAULT_MAX_RUNTIME)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--max-api-failures", type=int, default=DEFAULT_MAX_API_FAILURES)
    parser.add_argument("--pending-timeout", type=float, default=DEFAULT_PENDING_TIMEOUT)
    parser.add_argument("--termination-attempts", type=int, default=DEFAULT_TERMINATION_ATTEMPTS)
    parser.add_argument("--termination-poll-seconds", type=float, default=DEFAULT_TERMINATION_POLL_SECONDS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="create only under a supervised lease")
    c.add_argument("--gpu", default=DEFAULT_GPU)
    c.add_argument("--disk", type=int, default=DEFAULT_DISK_GB)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--allow-second", action="store_true")
    c.add_argument("--run-id")
    c.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    c.add_argument("--pubkey", type=Path, default=PUBKEY)
    c.add_argument("--hf-token", type=Path, default=Path.home() / "Documents" / "Credentials" / "hf-token.txt")
    c.add_argument("--identity-file", default="~/.ssh/id_ed25519_dsv4_b300")
    c.add_argument("--ssh-timeout", type=float, default=900)
    _add_safety_options(c)
    c.set_defaults(fn=create)

    r = sub.add_parser("run", help="provision, run remotely, sync receipts, and release the lease")
    r.add_argument("--gpu", default=DEFAULT_GPU)
    r.add_argument("--disk", type=int, default=DEFAULT_DISK_GB)
    r.add_argument("--allow-second", action="store_true")
    r.add_argument("--run-id")
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
    r.add_argument("--sync-root", type=Path, default=Path("artifacts/jlens/retrieved"))
    _add_safety_options(r)
    r.set_defaults(fn=run_job)

    m = sub.add_parser("monitor")
    m.add_argument("--state", type=Path, required=True)
    m.add_argument("--once", action="store_true")
    _add_safety_options(m)
    m.set_defaults(fn=monitor)

    s = sub.add_parser("status")
    s.set_defaults(fn=status)

    h = sub.add_parser("ssh")
    h.add_argument("--pod-file", type=Path, default=Path("artifacts/jlens/pod.json"))
    h.add_argument("--identity-file", default="~/.ssh/id_ed25519_dsv4_b300")
    h.set_defaults(fn=ssh)

    d = sub.add_parser("destroy")
    d.add_argument("pod_id", help='exact pod id, or "all" for JLens-named pods')
    _add_safety_options(d)
    d.set_defaults(fn=destroy)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = args.fn(args)
        return int(result or 0)
    except (SafetyError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
