"""Combined O1 + Foundation Learner session supervisor (contract §13, §22).

The O1 calibration owns the accelerator first and absolutely.  This state
machine runs the O1 pod-side entry as an OPAQUE configured subprocess, waits
for its declared completion artefacts, verifies and transfers them, closes the
O1 process, and only then reloads a pristine Ouro checkpoint and runs the FL
ladder::

    START_SESSION -> RUN_O1_CALIBRATION -> O1_HALT_OR_COMPLETE ->
    VERIFY_O1_RECORDS -> TRANSFER_O1_RECORDS -> CLOSE_O1_PROCESS ->
    RELOAD_PRISTINE_OURO -> COMPUTE_REMAINING_AUTHORIZED_TIME ->
    RUN_FL_LADDER -> CHECKPOINT_AND_VERIFY -> TRANSFER_FL_ARTIFACTS ->
    TERMINATE_ACCELERATOR

It NEVER modifies the sealed O1 runner: the O1 entry, the O1 transfer, and the
termination are configured commands, and no O1 module is imported.

**Custody of O1 records vs FL isolation.**  ``VERIFY_O1_RECORDS`` must hash
files that live under the very roots the FL isolation guard refuses.  That is
not a loophole in the guard: the FL guard still refuses those paths for all FL
work, and the verification runs through a separate, deliberately crippled
:class:`O1RecordCustodian` that

* only accepts paths under the DECLARED O1 roots,
* only ever returns SHA-256 digests and path/size metadata, and
* has no method that returns file CONTENT to any caller.

Hashing bytes is not reading outcomes: a digest carries no calibration result,
no difficulty selection, and no analysis.  Every path the custodian touched is
recorded in the journal, so the claim is auditable rather than asserted.

**UNRESOLVED refusal.**  Any configuration value beginning with ``UNRESOLVED``
refuses a real run, exactly like the O1 session config.  A DRESS REHEARSAL may
proceed with unresolved values, and then every artefact it writes is loudly
labelled ``DRESS_REHEARSAL``.

**Closing out costs money.**  :meth:`SessionSupervisor.run` wraps the whole
state machine in ``try/finally``: whatever happens — a failed state, an
unexpected exception, a keyboard interrupt — the supervisor still attempts the
FL transfer (best effort) and ALWAYS attempts ``TERMINATE_ACCELERATOR``, and
always writes ``SESSION_FINAL_STATUS.json``.  A rented accelerator that keeps
billing after a crash is a real failure mode, not a hypothetical one (review
finding R-C2).

**Resume rebuilds state.**  On resume the supervisor reconstructs
``state_results`` from the journal's ``STATE_COMPLETED`` payloads, so a session
that crashed after ``COMPUTE_REMAINING_AUTHORIZED_TIME`` still knows its FL
allowance instead of refusing ``RUN_FL_LADDER``.  The resumed allowance is
taken from THIS pod's allowance (``O1_SESSION_AUTHORIZED_SECONDS``, the
remaining allocation net of earlier pods' spend, capped by the config) minus
this process's elapsed time.  The eviction-to-reacquisition interval is
unbilled dead time and is not charged against a replacement rental; a restart
on the SAME pod (``RUNPOD_POD_ID`` unchanged) does charge the gap, because the
pod billed for it.
"""
from __future__ import annotations

import argparse
import calendar
import errno
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..ecology.base import sha256_file, sha256_tree
from . import o1_isolation, result_verifier
from ..eviction import EvictedBySignal
from .redaction import redact
from .scheduler import MonotonicClock, Scheduler
from .stage_definitions import StageContext

__all__ = [
    "SupervisorError",
    "NoStageAdmitted",
    "SessionConfigError",
    "STATES",
    "SESSION_CONFIG_SCHEMA",
    "JOURNAL_NAME",
    "JOURNAL_SCHEMA",
    "REHEARSAL_LABEL",
    "SessionConfig",
    "O1RecordCustodian",
    "SessionSupervisor",
    "main",
]

SESSION_CONFIG_SCHEMA = "flb200.session_config.v1"
JOURNAL_NAME = "FL_SESSION_JOURNAL.jsonl"
JOURNAL_SCHEMA = "flb200.session_journal.v1"
RECEIPT_SCHEMA = "flb200.o1_close_receipt.v1"
FINAL_STATUS_SCHEMA = "flb200.session_final_status.v1"
REHEARSAL_LABEL = "DRESS_REHEARSAL"

STATES: tuple[str, ...] = (
    "START_SESSION",
    "RUN_O1_CALIBRATION",
    "O1_HALT_OR_COMPLETE",
    "VERIFY_O1_RECORDS",
    "TRANSFER_O1_RECORDS",
    "CLOSE_O1_PROCESS",
    "RELOAD_PRISTINE_OURO",
    "COMPUTE_REMAINING_AUTHORIZED_TIME",
    "RUN_FL_LADDER",
    "CHECKPOINT_AND_VERIFY",
    "TRANSFER_FL_ARTIFACTS",
    "TERMINATE_ACCELERATOR",
)

#: FL may not start before this state has completed AND left its receipt.
FL_PREREQUISITE_STATE = "TRANSFER_O1_RECORDS"
O1_TRANSFER_RECEIPT = "O1_TRANSFER_RECEIPT.json"
O1_CLOSE_RECEIPT = "O1_CLOSE_RECEIPT.json"

#: Bound on the configured termination command.  Generous, because a
#: slow provider API is not a reason to give up on stopping the bill,
#: but finite, because waiting forever costs money either way.
TERMINATE_TIMEOUT_SECONDS = 900.0
TRANSFER_TIMEOUT_SECONDS = 900.0
#: Charged against the pod allowance before this supervisor's clock starts
#: (image pull + container start + fetches + gates); the entry script may
#: stamp O1_POD_ENTRY_EPOCH so the measured container uptime is used when
#: larger.
PROVISIONING_ALLOWANCE_SECONDS = 1200.0

#: Below this the O1 phase cannot reach even its own affordability
#: gate, so spawning it only converts budget exhaustion into a
#: timeout that reads as a deterministic defect.
MIN_O1_PHASE_SECONDS = 300.0
#: Least FL time a real combined session must leave after the O1 timeout:
#: the frozen final-transfer reserve (1200 s) plus one minimal stage.
FL_MINIMUM_SECONDS_DEFAULT = 1200.0 + 1800.0
#: Per-acquisition allowance the O1 zero-touch launcher sets on the pod:
#: the REMAINING session allocation net of earlier pods' spend.
POD_AUTHORIZED_SECONDS_ENV = "O1_SESSION_AUTHORIZED_SECONDS"
#: RunPod sets this in every pod; it distinguishes a same-pod process
#: restart (billed gap) from a replacement pod (unbilled gap).
POD_ID_ENV = "RUNPOD_POD_ID"
#: Where the image bakes the FL package; never a forbidden root.
FL_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CLOSE_O1_TIMEOUT_SECONDS = 900.0


class SupervisorError(RuntimeError):
    """A supervisor invariant refused."""


class BudgetDependentAbort(SupervisorError):
    """An abort whose outcome depends on THIS pod's allowance or throughput.

    Reported with a marker suffix the off-pod driver recognises, so it stops
    the session WITHOUT condemning the deployment: a replacement pod restores
    the pre-calibration phase from the durable store and therefore has more
    time, and an operator who raises the budget has changed the very input
    the refusal was a function of.
    """


class TransientO1Failure(SupervisorError):
    """The O1 child refused for a reason a replacement pod may not repeat:
    the session aborts WITHOUT the deterministic marker."""


class NoStageAdmitted(SupervisorError):
    """The ladder ran but no stage was admitted; the session is not COMPLETE."""


class SessionConfigError(SupervisorError):
    """The session configuration is missing, malformed, or UNRESOLVED."""


# --------------------------------------------------------------------------
# session configuration
# --------------------------------------------------------------------------

_REQUIRED_CONFIG_FIELDS = (
    "session_id",
    "o1_entry_command",
    "o1_completion_markers",
    "o1_hash_manifests",
    "o1_transfer_command",
    "o1_roots",
    "checkpoint_dir",
    "checkpoint_tree_sha256",
    "pregen_root",
    "fl_out_dir",
    "session_authorized_seconds",
)


def _unresolved_values(obj: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, str):
        if obj.startswith("UNRESOLVED"):
            found.append(path or "<root>")
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            found.extend(_unresolved_values(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(_unresolved_values(value, f"{path}[{index}]"))
    return found


@dataclass
class SessionConfig:
    payload: dict
    path: str | None = None

    @staticmethod
    def load(path: str, *, guard: o1_isolation.IsolationGuard | None = None
             ) -> "SessionConfig":
        guard = guard or o1_isolation.default_guard()
        resolved = guard.guard(path, o1_isolation.MODE_READ)
        if not os.path.isfile(resolved):
            raise SessionConfigError(f"session config not found: {path!r}")
        with open(resolved, encoding="utf-8") as fh:
            payload = json.load(fh)
        return SessionConfig(payload=payload, path=resolved)

    def validate(self) -> dict:
        payload = self.payload
        if payload.get("schema") != SESSION_CONFIG_SCHEMA:
            raise SessionConfigError(
                f"schema {payload.get('schema')!r} != {SESSION_CONFIG_SCHEMA!r}")
        missing = [k for k in _REQUIRED_CONFIG_FIELDS if k not in payload]
        if missing:
            raise SessionConfigError(f"missing required fields {missing}")
        rehearsal = bool(payload.get("rehearsal", False))
        if not rehearsal:
            # Without these a real session would strand its results on a pod
            # that is still burning money; a rehearsal has neither concern.
            for key in ("fl_transfer_command", "terminate_command"):
                if not payload.get(key):
                    raise SessionConfigError(
                        f"a real session requires {key!r}: without it the FL "
                        "artefacts never leave the accelerator / the "
                        "accelerator is never terminated")
        unresolved = _unresolved_values(payload)
        if unresolved and not rehearsal:
            raise SessionConfigError(
                f"session config carries unresolved template fields "
                f"{unresolved}; a real session refuses to start. (The O1 "
                "pod entrypoint is a recorded, operator-bound gap: "
                "'o1_entry_command' must be resolved by the operator.)")
        if not rehearsal:
            # The O1 phase must be BOUNDED or FL is structurally unfunded:
            # O1's own watchdog is built from the whole per-pod allowance, so
            # with no o1_timeout_seconds it may legitimately consume all of
            # it and every FL stage is then refused as unaffordable — after
            # the rental was paid for.
            o1_timeout = payload.get("o1_timeout_seconds")
            authorized = payload.get("session_authorized_seconds")
            try:
                o1_timeout = None if o1_timeout is None else float(o1_timeout)
                authorized = None if authorized is None else float(authorized)
            except (TypeError, ValueError):
                o1_timeout = authorized = None
            if not o1_timeout or o1_timeout <= 0:
                raise SessionConfigError(
                    "a real combined session requires a positive "
                    "'o1_timeout_seconds': without it the O1 phase may consume "
                    "the whole allowance and the FL ladder cannot run")
            reserve = float(payload.get("fl_minimum_seconds")
                            or FL_MINIMUM_SECONDS_DEFAULT)
            if authorized is not None and o1_timeout + reserve >= authorized:
                raise SessionConfigError(
                    f"o1_timeout_seconds ({o1_timeout:.0f}) + the FL minimum "
                    f"({reserve:.0f}) is not below session_authorized_seconds "
                    f"({authorized:.0f}); the FL half of this session could "
                    f"never be admitted")
        if unresolved and rehearsal:
            payload.setdefault("_unresolved_fields", unresolved)
        return payload

    @property
    def rehearsal(self) -> bool:
        return bool(self.payload.get("rehearsal", False))

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


# --------------------------------------------------------------------------
# O1 record custody (hashes only, never content)
# --------------------------------------------------------------------------

class O1RecordCustodian:
    """Hash-only access to O1 artefacts, restricted to the declared O1 roots.

    This class exists so that ``VERIFY_O1_RECORDS`` can do its job without
    weakening the FL isolation guard.  It is deliberately crippled: there is no
    method that returns file content, and every accepted path must lie under a
    declared O1 root (paths outside are refused, so it cannot become a general
    file reader).  The only structured data it parses is the O1 manifest's own
    path/digest listing.
    """

    def __init__(self, o1_roots: Sequence[str]):
        if not o1_roots:
            raise SupervisorError(
                "the O1 record custodian requires the declared O1 roots")
        self.roots = sorted({os.path.realpath(os.path.abspath(str(r)))
                             for r in o1_roots})
        self.touched: list[str] = []

    def _accept(self, path: str) -> str:
        resolved = os.path.realpath(os.path.abspath(str(path)))
        for root in self.roots:
            if resolved == root or resolved.startswith(root.rstrip(os.sep) + os.sep):
                self.touched.append(resolved)
                return resolved
        raise SupervisorError(
            f"O1 custodian refuses {path!r}: it is not under a declared O1 "
            f"root {self.roots}; the custodian is not a general file reader")

    def digest(self, path: str) -> str:
        """SHA-256 of one O1 file.  A digest is not an outcome."""
        return sha256_file(self._accept(path))

    def exists(self, path: str) -> bool:
        return os.path.exists(self._accept(path))

    def manifest_entries(self, manifest_path: str) -> list[dict]:
        """Extract ``(path, sha256)`` pairs from an O1 hash manifest.

        Tolerates the O1 ``o1b200.transfer_manifest.v1`` shape, a plain
        ``{"files": {rel: {"sha256": ...}}}`` mapping, and the two-space
        ``SHA256SUMS`` text format.  Nothing but paths and digests is read.
        """
        resolved = self._accept(manifest_path)
        base = os.path.dirname(resolved)
        entries: list[dict] = []
        with open(resolved, "rb") as fh:
            raw = fh.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            for line in raw.decode("utf-8", errors="strict").splitlines():
                if not line.strip():
                    continue
                digest, rel = line.split("  ", 1)
                entries.append({"path": os.path.join(base, rel),
                                "sha256": digest})
            return entries
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        if isinstance(artifacts, dict):
            for name, spec in sorted(artifacts.items()):
                if not isinstance(spec, dict):
                    continue
                path = spec.get("path")
                digest = spec.get("sha256")
                if path and digest:
                    entries.append({
                        "name": name,
                        "path": path if os.path.isabs(path) else os.path.join(base, path),
                        "sha256": digest,
                        "kind": spec.get("kind", "file"),
                    })
        files = payload.get("files") if isinstance(payload, dict) else None
        if isinstance(files, dict):
            for rel, spec in sorted(files.items()):
                digest = spec.get("sha256") if isinstance(spec, dict) else spec
                if digest:
                    entries.append({"path": os.path.join(base, rel),
                                    "sha256": digest})
        elif isinstance(files, list):
            for spec in files:
                if isinstance(spec, dict) and spec.get("path") and spec.get("sha256"):
                    rel = spec["path"]
                    entries.append({
                        "path": rel if os.path.isabs(rel) else os.path.join(base, rel),
                        "sha256": spec["sha256"]})
        if not entries:
            raise SupervisorError(
                f"{manifest_path}: no path/digest pairs found; the supervisor "
                "cannot verify O1 records without them")
        return entries

    def verify_manifest(self, manifest_path: str) -> dict:
        """Recompute every listed digest.  File-level hashing only."""
        entries = self.manifest_entries(manifest_path)   # refuses if empty
        checked = 0
        missing: list[str] = []
        mismatched: list[str] = []
        for entry in entries:
            path = entry["path"]
            if entry.get("kind") == "reference":
                continue
            try:
                resolved = self._accept(path)
            except SupervisorError:
                missing.append(path)
                continue
            if not os.path.exists(resolved):
                missing.append(path)
                continue
            digest = (sha256_tree(resolved) if os.path.isdir(resolved)
                      else sha256_file(resolved))
            if digest != entry["sha256"]:
                mismatched.append(path)
            checked += 1
        return {
            "manifest": os.path.basename(manifest_path),
            "entries": len(entries),
            "checked": checked,
            "missing": missing,
            "mismatched": mismatched,
            "ok": not missing and not mismatched,
            "method": ("recomputed SHA-256 over file bytes / directory trees; "
                       "no scientific content was parsed"),
        }


# --------------------------------------------------------------------------
# the supervisor
# --------------------------------------------------------------------------

@dataclass
class SessionSupervisor:
    config: SessionConfig
    out_dir: str
    guard: o1_isolation.IsolationGuard = field(
        default_factory=o1_isolation.default_guard)
    clock: Any = field(default_factory=MonotonicClock)
    runner: Callable[..., Any] = subprocess.run
    ladder_runner: Callable[[Scheduler, StageContext], dict] | None = None
    context_factory: Callable[["SessionSupervisor"], StageContext] | None = None
    journal_path: str = field(init=False)
    payload: dict = field(init=False)
    state_results: dict[str, Any] = field(default_factory=dict, init=False)
    completed: list[str] = field(default_factory=list, init=False)
    determinism: dict | None = field(default=None, init=False)
    close_out: dict = field(default_factory=dict, init=False)
    _t0: float | None = field(default=None, init=False)
    _provisioning_seconds: float | None = field(default=None, init=False)
    _records: int = field(default=0, init=False)
    torn_journal_tail: dict | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.payload = self.config.validate()
        self.guard.makedirs(self.out_dir)
        self.journal_path = os.path.join(self.out_dir, JOURNAL_NAME)
        extra_roots = list(self.payload.get("o1_roots", [])) + list(
            self.payload.get("o1_forbidden_roots_extra", []))
        self.guard.add_forbidden_roots(extra_roots)
        self.custodian = O1RecordCustodian(self.payload["o1_roots"])
        # Off-pod durability under INTERRUPTIBLE capacity: restore the
        # mirrored journal/checkpoints BEFORE reading the journal, so a
        # fresh pod after eviction resumes exactly like a same-pod restart
        # (completed states skip; training resumes from the last valid
        # atomic checkpoint; post-checkpoint work reruns).  Local files are
        # never overwritten by the restore.
        self.durability = None
        self.durability_events: list[dict] = []
        destination = self.payload.get("fl_durable_destination")
        if destination and not str(destination).startswith("UNRESOLVED"):
            from .durability import FlDurableMirror

            def _mirror_event(event, **fields):
                # Buffered during construction (no journal yet) and while a
                # journal append is itself in flight (the push that emitted
                # this event); otherwise written to the journal AT ONCE.
                # Buffering until the next supervisor journal call lost every
                # mirror failure raised during RUN_FL_LADDER on eviction.
                self.durability_events.append({"event": str(event), **fields})
                # only once the constructor has asserted the session and
                # repaired a torn tail: an earlier drain appended BEFORE the
                # repair, burying the torn record mid-file (refused as
                # corruption) and restarting the index at 0
                if getattr(self, "_journal_ready", False) and not getattr(
                        self, "_journal_in_flight", False):
                    try:
                        self._drain_durability_events()
                    except Exception:  # noqa: BLE001 - stays buffered
                        pass

            self.durability = FlDurableMirror(
                str(destination), self.out_dir, on_event=_mirror_event,
                guard=self.guard.guard,
                session_id=str(self.payload.get("session_id") or ""))
            try:
                self.durability.restore_all()
            except Exception as exc:  # noqa: BLE001
                # a store outage must DEGRADE (start fresh; restore never
                # overwrites local files anyway), not refuse the session
                self.durability_events.append(
                    {"event": "FL_DURABILITY_RESTORE_FAILED",
                     "error": str(exc)[:300]})
        self._assert_journal_session()
        # the index is a global sequence: initialise it from what exists
        # BEFORE the repair journals its own record
        self._records = len(self.read_journal())
        self._repair_torn_tail()
        existing = self.read_journal()
        self._records = len(existing)
        self.completed = [r["state"] for r in existing
                          if r.get("event") == "STATE_COMPLETED"]
        self._journal_ready = True

    # ---------------- journal ----------------

    @property
    def rehearsal(self) -> bool:
        return self.config.rehearsal

    def label(self) -> str:
        return REHEARSAL_LABEL if self.rehearsal else str(
            self.payload.get("label", "FL_B200_SESSION"))

    def journal(self, event: str, state: str, payload: dict | None = None) -> dict:
        record = {
            "schema": JOURNAL_SCHEMA,
            "index": self._records,
            "event": str(event),
            "state": str(state),
            "session_id": self.payload.get("session_id"),
            "label": self.label(),
            "rehearsal": self.rehearsal,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "monotonic": round(self.clock.monotonic(), 6),
            **_redact_payload(payload or {}),
        }
        self.guard.append_line(
            self.journal_path,
            json.dumps(record, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False))
        self._records += 1
        if self.durability is not None:
            self._journal_in_flight = True
            try:
                self.durability.push_file(self.journal_path)
            finally:
                self._journal_in_flight = False
            self._drain_durability_events()
        return record

    def _repair_torn_tail(self) -> None:
        """Physically drop a torn trailing record before anything appends.

        Once the resumed session journals again, the torn line would sit
        in the MIDDLE of the file and every later read would refuse it as
        corruption.  The torn bytes are preserved in a ``.torn`` sidecar.
        """
        if not self.torn_journal_tail:
            return
        path = self.guard.guard(self.journal_path, o1_isolation.MODE_WRITE)
        with open(path, "rb") as fh:
            raw = fh.read()
        body = raw.rstrip(b"\n")
        cut = body.rfind(b"\n")
        keep, torn = (body[:cut + 1], body[cut + 1:]) if cut >= 0 else (b"", body)
        sidecar = path + ".torn"
        with open(self.guard.guard(sidecar, o1_isolation.MODE_WRITE),
                  "ab") as fh:
            fh.write(torn + b"\n")
        tmp = path + ".repair"
        with open(tmp, "wb") as fh:
            fh.write(keep)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        self.torn_journal_tail["sidecar"] = os.path.basename(sidecar)
        self.journal("JOURNAL_TORN_TAIL_DROPPED", "START_SESSION",
                     dict(self.torn_journal_tail))

    def _push_durable(self, path: str) -> None:
        """Mirror one session file if a durable destination is configured."""
        if self.durability is None or not os.path.isfile(path):
            return
        self.durability.push_file(path)
        self._drain_durability_events()

    def _assert_journal_session(self) -> None:
        """Refuse a restored journal that belongs to a different session."""
        expected = self.payload.get("session_id")
        for record in self.read_journal():
            got = record.get("session_id")
            if got is not None and got != expected:
                raise SupervisorError(
                    f"REFUSED: restored journal session_id {got!r} does not "
                    f"match this config's session_id {expected!r}; a foreign "
                    "or stale session must not be resumed")

    def _drain_durability_events(self) -> None:
        """Journal buffered mirror events (failures included).

        Without this, a journal mirror that fails for the whole session
        leaves no record, and after eviction the next pod resumes from a
        stale journal believing it was durable.
        """
        pending, self.durability_events = self.durability_events, []
        try:
            self._drain_entries(pending)
        except BaseException:
            # keep whatever was not written; it really does stay buffered
            self.durability_events = pending[self._drained:] + self.durability_events
            raise

    def _drain_entries(self, pending: list) -> None:
        self._drained = 0
        for entry in pending:
            # never mutate the buffered entry: on a failure it is re-buffered
            # with its event name intact, and only the UNWRITTEN suffix is
            event = entry.get("event", "FL_DURABILITY_EVENT")
            fields = {k: v for k, v in entry.items() if k != "event"}
            record = {
                "schema": JOURNAL_SCHEMA, "index": self._records,
                "event": event, "state": "DURABILITY",
                "session_id": self.payload.get("session_id"),
                "label": self.label(), "rehearsal": self.rehearsal,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "monotonic": round(self.clock.monotonic(), 6), **fields,
            }
            self.guard.append_line(
                self.journal_path,
                json.dumps(record, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False))
            self._records += 1
            self._drained += 1

    def read_journal(self) -> list[dict]:
        """Parse the journal; a torn TRAILING record is tolerated.

        Eviction can land mid-``write``: the last line is then a partial
        JSON document.  Treating that as corruption would crash the
        supervisor in its constructor — before ``run``'s ``finally`` can
        reach ``_emergency_close`` — and strand a billing pod.  The torn
        tail is dropped and recorded (``self.torn_journal_tail``); the
        state it belonged to simply re-runs.  A torn record in the MIDDLE
        of the file is genuine corruption and still refuses.
        """
        path = self.guard.guard(self.journal_path, o1_isolation.MODE_READ)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().split("\n") if ln.strip()]
        out: list[dict] = []
        for index, line in enumerate(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if index == len(lines) - 1:
                    self.torn_journal_tail = {
                        "line_index": index, "bytes": len(line),
                        "error": str(exc)[:200]}
                    break
                raise SupervisorError(
                    f"REFUSED: journal record {index} of {len(lines)} is "
                    f"unparseable ({exc}); a torn record that is not the "
                    f"tail is corruption, not an interrupted write") from exc
        return out

    # ---------------- resume ----------------

    @staticmethod
    def _utc_seconds(stamp: str | None) -> float | None:
        if not stamp:
            return None
        try:
            return float(calendar.timegm(
                time.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ")))
        except (ValueError, TypeError):
            return None

    def rebuild_state_results(self) -> dict:
        """Reconstruct ``state_results`` from the journal (review R-C2).

        Without this a resumed session skips ``COMPUTE_REMAINING_AUTHORIZED_TIME``
        as "already completed" and then refuses ``RUN_FL_LADDER`` for want of an
        FL allowance — i.e. a crash anywhere in the middle silently cost the
        whole FL half of the session.

        On a REPLACEMENT pod the allowance is this pod's remaining allocation
        (``O1_SESSION_AUTHORIZED_SECONDS``, already net of earlier pods'
        spend) minus this process's elapsed time: eviction leaves a
        wall-clock gap during which nothing was billed, and charging it would
        zero the replacement before any FL stage could run.  On the SAME pod
        (``RUNPOD_POD_ID`` unchanged) the gap WAS billed, so the journalled
        figure minus the gap applies.  This is an operational budget
        decision, not a scientific path (contract §20).
        """
        restored: dict[str, Any] = {}
        stamps: dict[str, Any] = {}
        for record in self.read_journal():
            if record.get("event") != "STATE_COMPLETED":
                continue
            state = str(record.get("state"))
            if "result" in record:
                restored[state] = record["result"]
                stamps[state] = record.get("utc")
        self.state_results.update(restored)
        # Reuse the frozen provisioning charge ONLY on the SAME pod.  On the
        # same pod, re-measuring would read a container uptime that already
        # contains the O1 phase -- the double charge that starved the ladder.
        # On a REPLACEMENT pod the journalled figure describes a DIFFERENT
        # container: the new pod paid its own cold image pull, 5 GB
        # checkpoint fetch and pregen fetch, and inheriting the old (smaller)
        # charge over-grants the ladder time that does not exist.  A fresh
        # O1_POD_ENTRY_EPOCH is stamped per container, so re-measuring is
        # both correct and cheap there.
        start = restored.get("START_SESSION")
        started_pod = None
        compute_prior = restored.get("COMPUTE_REMAINING_AUTHORIZED_TIME")
        if isinstance(compute_prior, Mapping):
            started_pod = compute_prior.get("pod_id")
        this_pod_id = os.environ.get(POD_ID_ENV) or None
        same_pod_now = bool(started_pod) and started_pod == this_pod_id
        if (isinstance(start, Mapping) and self._provisioning_seconds is None
                and same_pod_now):
            try:
                value = float(start["provisioning_seconds"])
            except (KeyError, TypeError, ValueError):
                value = None
            if value is not None:
                # Clamp: the journal is restored from the durable mirror, so
                # a corrupt or absurd value must not drive the O1 bound
                # negative (instant kill) or the FL allowance to zero.
                # Ceiling is the allowance NET of the FL reserve: clamping
                # to the raw allowance still admitted a corrupt value that
                # drives usable to -reserve.
                reserve = float(self.payload.get("fl_minimum_seconds")
                                or FL_MINIMUM_SECONDS_DEFAULT)
                pod_limit = self._raw_pod_limit_seconds()
                ceiling = max(0.0, pod_limit - reserve) if pod_limit else value
                self._provisioning_seconds = max(0.0, min(value, ceiling))
        report = {"restored_states": sorted(restored),
                  "available_foundation_learner_seconds": None}
        compute = restored.get("COMPUTE_REMAINING_AUTHORIZED_TIME")
        if isinstance(compute, Mapping):
            journalled = float(
                compute.get("available_foundation_learner_seconds") or 0.0)
            recorded_at = self._utc_seconds(
                stamps.get("COMPUTE_REMAINING_AUTHORIZED_TIME"))
            gap = 0.0
            if recorded_at is not None:
                gap = max(0.0, time.time() - recorded_at)
            authorized, source = self._pod_authorized_seconds()
            elapsed = self._elapsed()
            recorded_pod = compute.get("pod_id")
            this_pod = os.environ.get(POD_ID_ENV) or None
            same_pod = bool(recorded_pod) and recorded_pod == this_pod
            if same_pod:
                # a process restart on the SAME pod: the pod billed for the
                # whole gap, so the journalled figure minus the gap is exact
                available = max(0.0, min(journalled - gap,
                                         authorized - elapsed))
                rule = ("same pod restarted: the journalled FL allowance "
                        "minus the wall-clock gap (the pod billed for it), "
                        "capped by this pod's allowance")
            else:
                # a REPLACEMENT pod after eviction: its allowance is already
                # net of every earlier pod's spend (zero_touch sets
                # O1_SESSION_AUTHORIZED_SECONDS from the remaining
                # allocation); the eviction-to-reacquisition gap was unbilled
                available = max(0.0, authorized - elapsed)
                rule = ("replacement pod: this pod's remaining allocation "
                        "minus this process's elapsed time; the unbilled "
                        "eviction gap is not charged")
            self.state_results["available_foundation_learner_seconds"] = available
            report.update({
                "available_foundation_learner_seconds": available,
                "journalled_available_seconds": journalled,
                "journalled_pod_id": recorded_pod,
                "this_pod_id": this_pod,
                "same_pod": same_pod,
                "this_pod_authorized_seconds": authorized,
                "authorized_seconds_source": source,
                "this_pod_elapsed_seconds": round(elapsed, 6),
                "resume_wall_clock_gap_seconds": round(gap, 3),
                "rule": rule,
            })
        if restored:
            self.journal("STATE_RESULTS_REBUILT", "START_SESSION", report)
        return report

    # ---------------- prerequisites ----------------

    def require_completed(self, state: str, *, for_state: str) -> None:
        if state not in self.completed:
            raise SupervisorError(
                f"REFUSED: {for_state} cannot run before {state} has completed "
                f"(completed so far: {self.completed})")

    def require_receipt(self, name: str, *, for_state: str) -> str:
        path = self.guard.guard(os.path.join(self.out_dir, name),
                                o1_isolation.MODE_READ)
        if not os.path.isfile(path):
            raise SupervisorError(
                f"REFUSED: {for_state} requires the receipt {name!r}; it does "
                "not exist. FL never starts before the O1 artefacts are "
                "transferred and verified (contract §13).")
        return path

    # ---------------- helpers ----------------

    def _child_env(self) -> dict:
        """Environment for a configured subprocess (the O1 phase included).

        O1's pod entrypoint dispatches to THIS supervisor whenever it sees
        O1_FL_SESSION_CONFIG.  Leaving that variable in the child environment
        would make the O1 phase start a second combined session, which would
        run the O1 phase again: unbounded recursion on a paid accelerator.
        O1's entry carries its own depth marker as an independent guard;
        either alone is sufficient, and neither depends on the other being
        correct.
        """
        env = {k: v for k, v in os.environ.items()
               if k != "O1_FL_SESSION_CONFIG"}
        env["O1_B300_ENTRY_ACTIVE"] = "1"
        return env

    def _run_command(self, command: Sequence[str] | str, *, state: str,
                     timeout: float | None = None) -> dict:
        if not command:
            raise SupervisorError(f"{state}: no command configured")
        shell = isinstance(command, str)
        started = self.clock.monotonic()
        env = self._child_env()
        if state in ("TRANSFER_FL_ARTIFACTS", "TRANSFER_O1_RECORDS"):
            # The pod runs model loading HF_HUB_OFFLINE-locked; hub access
            # happens only in hf_transfer subprocesses spawned with the
            # flags stripped.  The configured TRANSFER commands are hub
            # uploads too and must get the same environment — with the
            # flags inherited, the shipped fl_transfer_command refused at
            # _assert_online_capable after the whole session was paid for.
            from .hf_transfer import OFFLINE_FLAGS
            for flag in OFFLINE_FLAGS:
                env.pop(flag, None)
        if state == "RUN_O1_CALIBRATION" and timeout:
            # O1 plans against O1_SESSION_AUTHORIZED_SECONDS: its affordability
            # gate, pre-calibration budget and watchdog.  Inheriting the
            # pod's WHOLE allowance while this supervisor kills it at
            # o1_timeout_seconds let O1 admit a calibration it could never
            # finish — killed at the timeout, everything paid, nothing
            # produced.  Hand O1 its real bound (and the wall-clock it has
            # already lost), so O1's own gate refuses CHEAPLY, before
            # calibration, when it cannot fit.
            inherited = env.get(POD_AUTHORIZED_SECONDS_ENV, "").strip()
            try:
                pod_limit = float(inherited) if inherited else float("inf")
            except ValueError:
                pod_limit = float("inf")
            # The pod's allowance also has to pay for provisioning (pod
            # creation -> START_SESSION) and for the FL reserve.  Bounding
            # O1 by the RAW pod limit let it consume both: the config gate
            # only compares o1_timeout + reserve against the CONFIG's
            # session_authorized_seconds, which on the B300 profile is far
            # larger than the pod's real allowance (21,200 configured vs
            # 18,250 actual at $7.89/h), leaving ~50 s of slack that any
            # provisioning overrun turns negative.  Reserve both here, so
            # the FL minimum survives on either profile.
            reserve = float(self.payload.get("fl_minimum_seconds")
                            or FL_MINIMUM_SECONDS_DEFAULT)
            usable = pod_limit - self._provisioning_charge() - reserve
            budget_bound = max(0.0, usable - self._elapsed())
            bound = max(0.0, min(usable, float(timeout)) - self._elapsed())
            # Refuse BEFORE spawning when the BUDGET is what ran out.  A ~0
            # bound became subprocess timeout=0, which raises TimeoutExpired
            # after ~1 ms -- an instant kill reported as
            # ABORTED_AT_RUN_O1_CALIBRATION.  Budget exhaustion must refuse
            # cheaply and leave the deployment usable.
            #
            # Only the BUDGET-derived bound is tested, and only when this pod
            # actually declared an allowance.  A deliberately short
            # o1_timeout_seconds (rehearsals, tests, a quick smoke run) is an
            # operator choice, not exhaustion, and refusing it would break
            # every configuration that legitimately bounds O1 below the floor.
            if inherited and budget_bound < MIN_O1_PHASE_SECONDS \
                    and not self.rehearsal:
                raise BudgetDependentAbort(
                    f"{state}: only {budget_bound:.0f}s of the pod allowance "
                    f"remain after provisioning and the FL reserve, below the "
                    f"{MIN_O1_PHASE_SECONDS:.0f}s the O1 phase needs to do "
                    f"anything; refusing to spend on a phase that cannot run")
            env[POD_AUTHORIZED_SECONDS_ENV] = str(int(bound))
            env["O1_PHASE_BOUND_BY_FL"] = "1"
            # kill the child at the same bound: leaving the raw o1_timeout
            # here would let O1 outlive the budget it was handed
            timeout = min(float(timeout), max(0.0, bound))
        # NOT text=True.  Strict UTF-8 decoding of a multi-hour child means a
        # single stray byte from a CUDA/NCCL/driver message raises
        # UnicodeDecodeError at the END of the phase, discarding the whole
        # O1 result after it was fully paid for.  production_entry avoids
        # text=True for exactly this reason.  Decode ourselves, replacing
        # undecodable bytes.  (Output is deliberately captured rather than
        # streamed: O1's completion markers must be namespaced before they
        # reach the container log, or the off-pod driver terminates the pod
        # the moment the O1 phase ends.)
        try:
            proc = self.runner(command, shell=shell, capture_output=True,
                               timeout=timeout, env=env,
                               cwd=self.payload.get("o1_workdir") or None)
        except subprocess.TimeoutExpired as exc:
            # Running out the clock is a function of THIS pod's allowance and
            # measured throughput, not of the deployment.  Left as a plain
            # TimeoutExpired it became ABORTED_AT_RUN_O1_CALIBRATION, which
            # the driver records as a PERMANENT deterministic refusal -- the
            # most likely instance of exactly the class that rule exists to
            # exclude.
            raise BudgetDependentAbort(
                f"{state}: the O1 phase reached its bound "
                f"({timeout:.0f}s) and was stopped") from exc
        seconds = self.clock.monotonic() - started
        # RE-EMIT the child's output on our own streams.  capture_output
        # swallowed it entirely, and O1's completion markers
        # (ZERO_TOUCH_COMPLETE / ZERO_TOUCH_ABORTED_AT_<state>) are printed
        # by the child to ITS stdout.  The off-pod driver greps the
        # CONTAINER log for exactly those, so with them captured a
        # successful O1 phase looked like an eviction and bought a
        # redundant reacquisition, and a deterministic abort lost its
        # "do not reacquire" guarantee.
        # ...BUT in a COMBINED session the O1 phase is not the end of the
        # pod.  The driver's completion witness is "ZERO_TOUCH_COMPLETE in
        # the log tail" and its monitor accepts it while the pod is still
        # RUNNING, so re-emitting O1's marker verbatim made the driver
        # terminate the pod ~20 s after the O1 phase — before any FL stage.
        # The markers are namespaced on re-emission; the SESSION's own
        # marker is printed by main() at the true end (see _session_marker).
        child_out = _namespace_o1_markers(redact(
            _as_text(getattr(proc, "stdout", ""))))
        child_err = _namespace_o1_markers(redact(
            _as_text(getattr(proc, "stderr", ""))))
        if child_out:
            print(child_out, end="" if child_out.endswith("\n") else "\n",
                  flush=True)
        if child_err:
            print(child_err, end="" if child_err.endswith("\n") else "\n",
                  file=sys.stderr, flush=True)
        record = {
            "command": command if shell else list(command),
            "returncode": int(getattr(proc, "returncode", -1)),
            "seconds": round(seconds, 6),
            "stdout_tail": child_out[-2000:],
            "stderr_tail": child_err[-2000:],
        }
        return record

    # ---------------- states ----------------

    def _discover_o1_roots(self) -> dict:
        """Widen the refusal set from O1's own manifests (path strings only).

        Called at START_SESSION (a resumed pod may already hold them) and
        again once O1 reports complete, when they are guaranteed to exist.
        Discovery only ever ADDS forbidden roots; an absent manifest leaves
        the frozen list in force and is recorded as such.
        """
        report = self.guard.discover_from_o1_manifests(
            list(self.payload.get("o1_hash_manifests") or []),
            protected=[self.payload.get("checkpoint_dir"),
                       self.payload.get("pregen_root"), self.out_dir,
                       FL_PACKAGE_ROOT,
                       (self.payload.get("fl_durable_destination")
                        if not str(self.payload.get(
                            "fl_durable_destination", "")).startswith(
                            ("hf://", "UNRESOLVED")) else None)])
        if report.get("added_roots") or report.get("exempted_roots"):
            self.journal("O1_ROOTS_DISCOVERED", "ISOLATION",
                         {"added_roots": report["added_roots"],
                          "exempted_roots": report["exempted_roots"]})
        return report

    def state_START_SESSION(self) -> dict:
        self._t0 = self.clock.monotonic()
        provisioning = self._provisioning_charge()
        return {
            "provisioning_seconds": provisioning,
            "config_path": self.config.path,
            "rehearsal": self.rehearsal,
            "unresolved_fields": self.payload.get("_unresolved_fields", []),
            "o1_roots": list(self.payload["o1_roots"]),
            "o1_root_discovery": self._discover_o1_roots(),
            "isolation": self.guard.to_dict(),
            "session_authorized_seconds": float(
                self.payload["session_authorized_seconds"]),
        }

    def state_RUN_O1_CALIBRATION(self) -> dict:
        """Run the O1 pod-side entry as an OPAQUE configured subprocess."""
        record = self._run_command(self.payload["o1_entry_command"],
                                   state="RUN_O1_CALIBRATION",
                                   timeout=self.payload.get("o1_timeout_seconds"))
        record["note"] = ("the O1 entry is opaque to FL: FL neither imports O1 "
                          "modules nor interprets O1 output")
        return record

    def state_O1_HALT_OR_COMPLETE(self) -> dict:
        """Did the O1 phase COMPLETE, or did it HALT?

        Only presence is checked; no marker content is read.  A HALT is a
        legitimate O1 outcome, but it is NOT a licence for FL to proceed:
        contract §13 makes FL conditional on the O1 records being transferred
        AND verified, and a halted O1 phase has produced no such records.  The
        session therefore aborts here, with the halt recorded, rather than
        silently running FL on an accelerator whose first workload failed.
        """
        markers = list(self.payload["o1_completion_markers"])
        present = {marker: self.custodian.exists(marker) for marker in markers}
        missing = sorted(m for m, ok in present.items() if not ok)
        entry = self.state_results.get("RUN_O1_CALIBRATION") or {}
        rc = entry.get("returncode")
        tails = ((entry.get("stdout_tail") or "")
                 + (entry.get("stderr_tail") or ""))
        # A transient refusal is checked BEFORE the missing-marker branch.
        # The transient classifications live in the PRE-ENTRY steps
        # (runner/fetch_artifacts.py, runner/check_hf_scope.py), which run
        # before production_entry writes FINAL_STATUS.json -- the only
        # declared completion marker.  So an HF outage during the 5 GB
        # checkpoint fetch always took the `missing` branch below and was
        # recorded as a PERMANENT deterministic refusal, blocking the whole
        # deployment until a human deleted a marker file.
        budget_verdict = o1_abort_is_budget_dependent(tails)
        if budget_verdict:
            # O1 refused because THIS pod ran short of time, not because the
            # deployment is broken.  Re-declare it so the driver stops the
            # session without condemning every future run.
            raise BudgetDependentAbort(
                f"the O1 phase aborted at {budget_verdict}, which depends on "
                f"this pod's allowance and measured throughput, not on the "
                f"deployment")
        if "REFUSED (transient)" in tails:
            print("REFUSED (transient): the O1 phase failed transiently; "
                  "a replacement pod may succeed", file=sys.stderr, flush=True)
            raise TransientO1Failure(
                f"O1 entry exited {rc} with a transient refusal "
                f"(completion artefacts missing: {missing or 'none'})")
        if missing:
            raise SupervisorError(
                f"O1_HALT: the O1 phase did not produce its declared "
                f"completion artefacts {missing} (entry command returncode "
                f"{rc}). FL does not start: there are no "
                "O1 records to verify or transfer (contract §13).")
        # Presence is NOT completion: the O1 production entry writes its
        # FINAL_STATUS.json on an ABORT as well, and exits non-zero for it.
        # The exit status is the one outcome-free signal FL may consult; the
        # marker content stays unread.
        if rc != 0:
            # transient already handled above, before the marker check
            raise SupervisorError(
                f"O1_HALT: the O1 entry exited with returncode {rc}; its "
                f"completion artefacts are present but a non-zero exit is "
                f"O1's own declaration of an abort. FL does not start "
                f"(contract §13).")
        return {"o1_outcome": "COMPLETE", "markers": present,
                "entry_returncode": rc,
                "o1_root_discovery": self._discover_o1_roots(),
                "note": ("presence of the markers AND a zero O1 exit status; "
                         "no marker content is read")}

    def state_VERIFY_O1_RECORDS(self) -> dict:
        reports = [self.custodian.verify_manifest(m)
                   for m in self.payload["o1_hash_manifests"]]
        bad = [r for r in reports if not r["ok"]]
        if bad:
            raise SupervisorError(
                f"O1 record verification FAILED for {[r['manifest'] for r in bad]}")
        return {
            "reports": reports,
            "paths_touched": len(self.custodian.touched),
            "method": ("recomputed SHA-256 over the files O1's own manifests "
                       "list; hashing bytes is not reading outcomes"),
        }

    def state_TRANSFER_O1_RECORDS(self) -> dict:
        record = self._run_command(self.payload["o1_transfer_command"],
                                   state="TRANSFER_O1_RECORDS",
                                   timeout=self.payload.get("o1_transfer_timeout_seconds"))
        if record["returncode"] != 0:
            raise SupervisorError(
                f"O1 transfer command failed with rc={record['returncode']}")
        receipt = {
            "schema": "flb200.o1_transfer_receipt.v1",
            "session_id": self.payload.get("session_id"),
            "label": self.label(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "transfer": record,
            "verification": self.state_results.get("VERIFY_O1_RECORDS"),
        }
        receipt_path = os.path.join(self.out_dir, O1_TRANSFER_RECEIPT)
        self.guard.write_json(receipt_path, receipt)
        self._push_durable(receipt_path)
        return receipt

    def state_CLOSE_O1_PROCESS(self) -> dict:
        self.require_completed("TRANSFER_O1_RECORDS", for_state="CLOSE_O1_PROCESS")
        close_command = self.payload.get("o1_close_command")
        record = None
        if close_command:
            record = self._run_command(
                close_command, state="CLOSE_O1_PROCESS",
                timeout=float(self.payload.get("o1_close_timeout_seconds")
                              or CLOSE_O1_TIMEOUT_SECONDS))
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "session_id": self.payload.get("session_id"),
            "label": self.label(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "close_command": record,
            "o1_roots": list(self.payload["o1_roots"]),
            "statement": ("the O1 process is closed; FL starts from a FRESH "
                          "reload of the pristine checkpoint and never from an "
                          "O1-mutated process state"),
        }
        receipt_path = os.path.join(self.out_dir, O1_CLOSE_RECEIPT)
        self.guard.write_json(receipt_path, receipt)
        self._push_durable(receipt_path)
        return receipt

    def state_RELOAD_PRISTINE_OURO(self) -> dict:
        """Verify the checkpoint tree hash before any FL work touches it."""
        self.require_completed("CLOSE_O1_PROCESS", for_state="RELOAD_PRISTINE_OURO")
        checkpoint_dir = self.payload["checkpoint_dir"]
        expected = self.payload["checkpoint_tree_sha256"]
        from ..training import model_loading

        if not self.rehearsal:
            if expected != model_loading.FROZEN_CHECKPOINT_TREE_SHA256:
                raise SupervisorError(
                    "session config binds a checkpoint tree hash that is not "
                    "the frozen §1 backbone hash; a real session refuses")
            got = model_loading.verify_checkpoint_tree(checkpoint_dir)
        else:
            got = sha256_tree(self.guard.guard(checkpoint_dir,
                                               o1_isolation.MODE_READ))
            if got != expected:
                raise SupervisorError(
                    f"rehearsal checkpoint tree hash mismatch: expected "
                    f"{expected}, got {got}")
        return {"checkpoint_dir": checkpoint_dir, "tree_sha256": got,
                "verified_against": ("frozen §1 backbone hash" if not self.rehearsal
                                     else "rehearsal stand-in tree hash"),
                "rehearsal": self.rehearsal}

    def _raw_pod_limit_seconds(self) -> float:
        """This pod's allowance from the env, unreduced.  0.0 when absent."""
        raw = os.environ.get(POD_AUTHORIZED_SECONDS_ENV, "").strip()
        try:
            return max(0.0, float(raw)) if raw else 0.0
        except ValueError:
            return 0.0

    def _wall_elapsed_since_start_session(self) -> float | None:
        """Wall seconds since START_SESSION, from this pod's own clock.

        Container uptime minus the provisioning charge.  Immune to a _t0
        reset, so a restart cannot silently re-grant time already billed.
        Returns None when the entry never stamped an epoch.
        """
        stamp = os.environ.get("O1_POD_ENTRY_EPOCH", "").strip()
        if not stamp:
            return None
        try:
            uptime = time.time() - float(stamp)
        except ValueError:
            return None
        return max(0.0, uptime - self._provisioning_charge())

    def _measure_provisioning(self) -> float:
        """Pre-supervisor overhead: POD CREATION -> START_SESSION.

        Evaluated ONCE, when the supervisor starts.  Measuring it later
        would read the whole container uptime, which by then also contains
        the O1 phase -- and since the caller separately subtracts
        ``_elapsed()`` (also the O1 phase), O1 was charged TWICE and the FL
        ladder was starved to zero for any O1 phase past ~8,500 s.
        """
        provisioning = PROVISIONING_ALLOWANCE_SECONDS
        stamp = os.environ.get("O1_POD_ENTRY_EPOCH", "").strip()
        if stamp:
            try:
                provisioning = max(provisioning, time.time() - float(stamp))
            except ValueError:
                pass
        return provisioning

    def _provisioning_charge(self) -> float:
        """The frozen provisioning charge; measured on first use only."""
        if self._provisioning_seconds is None:
            self._provisioning_seconds = self._measure_provisioning()
        return self._provisioning_seconds

    def _pod_authorized_seconds(self) -> tuple[float, str]:
        """This pod's runtime allowance and where it came from.

        ``O1_SESSION_AUTHORIZED_SECONDS`` is set per acquisition by the
        zero-touch launcher as the REMAINING allocation net of every earlier
        pod's spend, so it is the budget-true ceiling on a replacement pod.
        The config's ``session_authorized_seconds`` is the operator's whole-
        session ceiling; the smaller of the two wins.
        """
        authorized = float(self.payload["session_authorized_seconds"])
        source = "config.session_authorized_seconds"
        # The pod's allowance (and the independent watchdog armed with it)
        # counts from POD CREATION; this supervisor's clock starts at
        # START_SESSION.  Everything before that — image pull, container
        # start, credential scope, the 5 GB checkpoint fetch and hash, the
        # hardware gate, the pregen fetch — was billed against the same
        # allowance and would otherwise be paid out of the 1,200 s
        # final-transfer reserve.  Charge a fixed provisioning allowance,
        # or the measured container uptime when the entry stamped it.
        provisioning = self._provisioning_charge()
        raw = os.environ.get(POD_AUTHORIZED_SECONDS_ENV, "").strip()
        if not raw and not self.rehearsal:
            raise SupervisorError(
                f"REFUSED: {POD_AUTHORIZED_SECONDS_ENV} is unset on a real "
                f"session; without the per-pod allowance a replacement pod "
                f"would be granted the whole-session budget again")
        if raw:
            try:
                pod_limit = float(raw)
            except ValueError as exc:
                raise SupervisorError(
                    f"REFUSED: {POD_AUTHORIZED_SECONDS_ENV}={raw!r} is not "
                    f"a number") from exc
            if pod_limit < authorized:
                authorized, source = pod_limit, POD_AUTHORIZED_SECONDS_ENV
        if not self.rehearsal:
            authorized = max(0.0, authorized - provisioning)
            source += f" minus provisioning {provisioning:.0f}s"
        return authorized, source

    def _elapsed(self) -> float:
        return 0.0 if self._t0 is None else max(
            0.0, self.clock.monotonic() - self._t0)

    def state_COMPUTE_REMAINING_AUTHORIZED_TIME(self) -> dict:
        authorized, source = self._pod_authorized_seconds()
        # _elapsed() is monotonic since _t0, and run() resets _t0 on every
        # restart -- so a process that died anywhere between the O1 phase
        # completing and this state completing came back with elapsed ~= 0
        # and handed the ladder the WHOLE post-provisioning allowance,
        # ignoring the hours O1 had already burned on this same pod.  The
        # pod's own wall clock cannot be reset that way, so take whichever
        # says MORE time is gone.
        elapsed = self._elapsed()
        wall = self._wall_elapsed_since_start_session()
        if wall is not None and wall > elapsed:
            elapsed = wall
        available = max(0.0, authorized - elapsed)
        record = {
            "session_authorized_seconds": authorized,
            "authorized_seconds_source": source,
            "pod_id": os.environ.get(POD_ID_ENV) or None,
            "elapsed_seconds": round(elapsed, 6),
            "available_foundation_learner_seconds": round(available, 6),
            "rule": ("FL receives what remains after O1 closes; it never "
                     "assumes ownership of the whole rental"),
        }
        self.state_results["available_foundation_learner_seconds"] = available
        return record

    def state_RUN_FL_LADDER(self) -> dict:
        self.require_completed(FL_PREREQUISITE_STATE, for_state="RUN_FL_LADDER")
        self.require_receipt(O1_TRANSFER_RECEIPT, for_state="RUN_FL_LADDER")
        self.require_completed("CLOSE_O1_PROCESS", for_state="RUN_FL_LADDER")
        self.require_completed("RELOAD_PRISTINE_OURO", for_state="RUN_FL_LADDER")
        available = self.state_results.get("available_foundation_learner_seconds")
        if available is None:
            raise SupervisorError(
                "RUN_FL_LADDER requires COMPUTE_REMAINING_AUTHORIZED_TIME")
        if self.context_factory is None:
            raise SupervisorError(
                "no StageContext factory supplied; the supervisor does not "
                "invent one (the campaign entry point wires it)")
        ctx = self.context_factory(self)
        scheduler = Scheduler(
            available_foundation_learner_seconds=float(available),
            out_dir=os.path.join(self.out_dir, "ladder"),
            guard=self.guard, clock=self.clock, label=self.label(),
            durability=self.durability)
        runner = self.ladder_runner or (lambda sch, c: sch.run_ladder(c))
        summary = runner(scheduler, ctx)
        self.state_results["_ctx"] = ctx
        self.state_results["_scheduler"] = scheduler
        if not _ladder_admitted_a_stage(summary):
            raise NoStageAdmitted(
                "REFUSED: the FL ladder admitted no stage; the session "
                "cannot report COMPLETE and RUN_FL_LADDER is not recorded "
                "as complete so a replacement pod may retry")
        return summary

    def state_CHECKPOINT_AND_VERIFY(self) -> dict:
        if self.durability is not None:
            # anything the throttled ladder-journal mirror deferred
            self.durability.flush()
            self._drain_durability_events()
        root = os.path.join(self.out_dir, "ladder")
        manifest = result_verifier.write_result_manifest(
            root, os.path.join(self.out_dir, "FL_RESULT_MANIFEST.json"),
            label=self.label(), guard=self.guard)
        report = result_verifier.verify_result_tree(
            root, manifest, guard=self.guard)
        return {"manifest_files": manifest["n_files"],
                "total_bytes": manifest["total_bytes"], "verification": report}

    def state_TRANSFER_FL_ARTIFACTS(self) -> dict:
        archive = result_verifier.build_transfer_archive(
            os.path.join(self.out_dir, "ladder"),
            os.path.join(self.out_dir, "FL_ARTIFACTS.zip"),
            label=self.label(), guard=self.guard)
        command = self.payload.get("fl_transfer_command")
        transfer = None
        if command:
            transfer = self._run_command(
                command, state="TRANSFER_FL_ARTIFACTS",
                timeout=float(self.payload.get("fl_transfer_timeout_seconds")
                              or TRANSFER_TIMEOUT_SECONDS))
            if transfer["returncode"] != 0:
                raise SupervisorError(
                    f"FL transfer command failed with rc={transfer['returncode']}")
        return {"archive": {k: v for k, v in archive.items() if k != "manifest"},
                "transfer": transfer}

    def state_TERMINATE_ACCELERATOR(self) -> dict:
        command = self.payload.get("terminate_command")
        record = None
        if command:
            # A hung terminate command blocks the supervisor indefinitely on
            # an accelerator that is still billing, leaving only RunPod's
            # provider-side terminateAfter as a backstop.  Every other
            # configured command already carries a timeout; this one is the
            # single most expensive place to omit it.
            record = self._run_command(
                command, state="TERMINATE_ACCELERATOR",
                timeout=float(self.payload.get("terminate_timeout_seconds")
                              or TERMINATE_TIMEOUT_SECONDS))
        return {"terminate_command": record,
                "note": ("termination is a configured command; this package "
                         "never contacts a provider by itself")}

    # ---------------- the machine ----------------

    def _emergency_close(self, failed_state: str | None) -> dict:
        """Best-effort transfer, then ALWAYS attempt termination (R-C2).

        Runs from ``run``'s ``finally`` block.  Nothing here may raise: a
        failure inside the close-out would strand the pod, so every step is
        recorded (in the journal and in the returned record) and the next one
        still runs.  Termination is attempted even when the transfer failed —
        losing artefacts is bad, paying for an abandoned accelerator is worse,
        and the artefacts also live in the session directory.
        """
        record: dict[str, Any] = {"triggered_by": failed_state,
                                  "transfer": None, "terminate": None}
        if failed_state is None and "TERMINATE_ACCELERATOR" in self.completed:
            record["note"] = "the session closed normally; nothing to force"
            return record
        # FLUSH THE DURABLE MIRROR FIRST.  state_TRANSFER_FL_ARTIFACTS builds
        # an untimed deterministic zip of the whole ladder tree before it
        # uploads anything, and a preemptible eviction gives ~30 s before
        # SIGKILL.  Starting with the archive meant that on the eviction that
        # actually matters -- mid-ladder, hours in -- the container died
        # during the zip: nothing was mirrored, SESSION_FINAL_STATUS.json was
        # never written, and the transient marker was never printed.  The
        # mirror is incremental and cheap, so it is what a short window can
        # actually complete.
        if self.durability is not None:
            try:
                self.durability.flush()
                record["durable_flush"] = "COMPLETED"
            except BaseException as exc:  # noqa: BLE001 - never block close-out
                record["durable_flush"] = f"FAILED: {exc!r}"[:300]
        for state, key in (("TRANSFER_FL_ARTIFACTS", "transfer"),
                           ("TERMINATE_ACCELERATOR", "terminate")):
            if state in self.completed:
                record[key] = {"status": "ALREADY_COMPLETED"}
                continue
            if key == "transfer" and not os.path.isdir(
                    os.path.join(self.out_dir, "ladder")):
                # nothing ran, so there is nothing to transfer; an empty
                # archive would misdescribe the session
                record[key] = {"status": "SKIPPED_NO_LADDER_OUTPUT"}
                continue
            handler = getattr(self, f"state_{state}")
            # journalling sits OUTSIDE the handler's try: a full disk or a
            # guard refusal in the journal must not skip termination
            try:
                self.journal("EMERGENCY_STATE_STARTED", state,
                             {"reason": f"session aborted at {failed_state!r}"})
            except BaseException:  # noqa: BLE001
                pass
            try:
                result = handler()
            except BaseException as exc:  # noqa: BLE001 - recorded, not raised
                record[key] = {"status": "FAILED", "error": repr(exc)}
                try:
                    self.journal("EMERGENCY_STATE_FAILED", state,
                                 {"error": repr(exc)})
                except BaseException:  # noqa: BLE001 - the journal is gone
                    pass
                continue
            record[key] = {"status": "COMPLETED", "result": _jsonable(result)}
            self.state_results.setdefault(state, result)
            if state not in self.completed:
                self.completed.append(state)
            try:
                self.journal("EMERGENCY_STATE_COMPLETED", state,
                             {"result": _jsonable(result)})
            except BaseException:  # noqa: BLE001
                pass
        record["policy"] = (
            "a failed session still transfers what it has (best effort) and "
            "ALWAYS attempts termination: a rented accelerator keeps billing "
            "until it is terminated, not until the process exits")
        return record

    def run(self, *, resume: bool = True) -> dict:
        started = self.clock.monotonic()
        self._t0 = self._t0 or started
        failed_state = None
        failure = None
        failure_exc: BaseException | None = None
        try:
            if resume and self.completed:
                # INSIDE the protected region: a refusal here (e.g. the
                # per-pod allowance missing on a replacement pod) must still
                # reach _emergency_close and the session marker, or the pod
                # bills on with no verdict and the driver pays to reacquire
                self.rebuild_state_results()
            for state in STATES:
                if resume and state in self.completed:
                    self.journal("STATE_SKIPPED_RESUMED", state, {})
                    continue
                handler = getattr(self, f"state_{state}", None)
                if handler is None:  # pragma: no cover - STATES is closed
                    raise SupervisorError(f"no handler for state {state!r}")
                self.journal("STATE_STARTED", state, {})
                try:
                    result = handler()
                except BaseException as exc:  # noqa: BLE001 - recorded, never swallowed
                    failed_state = state
                    failure = repr(exc)
                    failure_exc = exc
                    self.journal("STATE_FAILED", state, {"error": failure})
                    break
                self.state_results[state] = result
                self.completed.append(state)
                self.journal("STATE_COMPLETED", state,
                             {"result": _jsonable(result)})
        except BaseException as exc:  # noqa: BLE001 - supervisor-level failure
            failed_state = failed_state or "SUPERVISOR"
            failure = failure or repr(exc)
            failure_exc = failure_exc or exc
        finally:
            self.close_out = self._emergency_close(failed_state)

        if failed_state is None:
            outcome = "COMPLETE"
        elif failed_state == "RUN_FL_LADDER" and isinstance(
                failure_exc, NoStageAdmitted):
            outcome = "NO_STAGE_ADMITTED"
        elif isinstance(failure_exc, (TransientO1Failure, EvictedBySignal)):
            # An eviction is INFRASTRUCTURE, never a deterministic defect.
            # Classifying it as ABORTED_AT_<state> made the off-pod driver
            # raise DeterministicPodFailure, refuse to reacquire, and write
            # the durable refusal marker -- so a normal spot eviction at hour
            # 4 killed the session AND blocked every future run until a human
            # deleted a file.  That is worse than the default SIGTERM
            # disposition this handler replaced.
            outcome = f"ABORTED_TRANSIENT_AT_{failed_state}"
        else:
            outcome = f"ABORTED_AT_{failed_state}"
        status = {
            "schema": FINAL_STATUS_SCHEMA,
            "session_id": self.payload.get("session_id"),
            "label": self.label(),
            "rehearsal": self.rehearsal,
            "outcome": outcome,
            "transient": isinstance(
                failure_exc, (TransientO1Failure, EvictedBySignal)),
            # the DRIVER cannot see the exception type, only the marker line,
            # so the classification has to travel in the marker itself
            "budget_dependent": isinstance(
                failure_exc, (BudgetDependentAbort, NoStageAdmitted)),
            "failed_state": failed_state,
            "failure": failure,
            "states_completed": list(self.completed),
            "states_never_started": [s for s in STATES
                                     if s not in self.completed
                                     and s != failed_state],
            "elapsed_seconds": round(self.clock.monotonic() - started, 6),
            "isolation": self.guard.to_dict(),
            "o1_paths_touched_for_hashing": len(self.custodian.touched),
            "determinism": self.determinism,
            "close_out": self.close_out,
        }
        # Termination has already been attempted by now; a full disk must
        # not turn the status write into an escaping exception that hides
        # the close-out record from main().
        try:
            self.guard.write_json(
                os.path.join(self.out_dir, "SESSION_FINAL_STATUS.json"), status)
            self.journal("SESSION_FINISHED",
                         failed_state or "TERMINATE_ACCELERATOR",
                         {"outcome": status["outcome"],
                          "close_out": self.close_out})
        except BaseException as exc:  # noqa: BLE001
            status["final_status_write_error"] = repr(exc)[:300]
            print(f"WARNING: SESSION_FINAL_STATUS.json not written: {exc!r}",
                  file=sys.stderr)
        return status


_O1_MARKER_RE = re.compile(r"ZERO_TOUCH_(COMPLETE|ABORTED_AT_[A-Z0-9_]+)")


def _redact_payload(value):
    """Redact every string inside a journal payload, at any depth.

    The journal is pushed to the results repo, and STATE_FAILED /
    EMERGENCY_STATE_FAILED carry ``repr(exc)`` -- which can quote a command
    line, an environment or a URL.  redact() was applied only to captured
    child output, so anything raised as an exception bypassed it.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {k: _redact_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_payload(v) for v in value]
    return value


def _as_text(raw) -> str:
    """Decode child output without ever raising.

    Accepts bytes (the real subprocess, which is no longer run with
    text=True) or str (injected test/rehearsal runners).
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


#: Tokens that mark an O1 abort as a function of THIS pod's allowance or
#: measured throughput rather than of the deployment.  Deliberately a LOCAL
#: copy of the driver's rule: the campaign package must not import from the
#: O1 package (contract 13 isolation), and duplicating four tokens is a much
#: smaller risk than the coupling.
_O1_BUDGET_DEPENDENT_TOKENS = ("AFFORD", "BUDGET", "ALLOWANCE", "UNAFFORDABLE")

_O1_ABORT_RE = re.compile(r"^[ \t]*O1_PHASE_ABORTED_AT_([A-Z0-9_]+)[ \t\r]*$",
                          re.MULTILINE)


def o1_abort_is_budget_dependent(tails: str) -> str | None:
    """The O1 child's abort state when that abort was budget-dependent.

    The O1 child KNOWS its affordability gate refused -- it prints
    ZERO_TOUCH_ABORTED_AT_CALIBRATION_AFFORDABILITY_CHECK -- but this
    supervisor namespaces that marker away and then reports its own
    ABORTED_AT_O1_HALT_OR_COMPLETE, which carries no budget token.  The
    driver therefore recorded a PERMANENT, deployment-wide refusal for a
    plain "not enough time left on this pod" outcome, and the operator had
    to delete a file before any future run.  The classification has to
    cross the O1 -> FL hop explicitly; nothing else carries it.
    """
    verdict = None
    for m in _O1_ABORT_RE.finditer(tails or ""):
        verdict = m.group(1)            # last marker line wins
    if verdict and any(tok in verdict for tok in _O1_BUDGET_DEPENDENT_TOKENS):
        return verdict
    return None


def _namespace_o1_markers(text: str) -> str:
    """``ZERO_TOUCH_COMPLETE`` -> ``O1_PHASE_COMPLETE`` (and
    ``ZERO_TOUCH_ABORTED_AT_X`` -> ``O1_PHASE_ABORTED_AT_X``).

    The driver's witness is a SUBSTRING match on the container log, so a
    prefix would not do: the rewritten token must not contain the literal
    at all.  Inside a combined session the literal appears exactly once, at
    the end, printed by the supervisor itself.
    """
    return _O1_MARKER_RE.sub(r"O1_PHASE_\1", text)


def _session_marker(status: Mapping[str, Any]) -> str:
    if status.get("transient"):
        return "REFUSED (transient): session aborted for a transient cause"
    """The combined session's completion witness for the off-pod driver.

    Both literals are written here verbatim (not assembled) for the same
    reason O1 writes its own that way: the marker the driver looks for and
    the marker the pod emits cannot drift apart.
    """
    if status.get("outcome") == "COMPLETE":
        return "ZERO_TOUCH_COMPLETE"
    state = str(status.get("failed_state") or status.get("outcome")
                or "SUPERVISOR")
    if status.get("budget_dependent"):
        # the driver keys on this suffix to stop the session without
        # recording a permanent, deployment-wide refusal
        state += "_BUDGET_DEPENDENT"
    return "ZERO_TOUCH_ABORTED_AT_" + state


def _ladder_admitted_a_stage(summary: Any) -> bool:
    """True when the ladder actually admitted work (not only refusals).

    A stub runner that returns no ``states`` mapping is treated as admitted
    so explicitly injected test/rehearsal runners are not reclassified.
    An empty ``states`` mapping, or one whose every value is a refusal /
    skip / block, is not an admission.
    """
    if not isinstance(summary, Mapping):
        return True
    if "states" not in summary:
        return True
    states = summary.get("states") or {}
    if not states:
        return False
    admitted = {"COMPLETE", "FAILED", "STAGE_ABORTED_OVERRUN"}
    return any(str(value) in admitted for value in states.values())


def _jsonable(value: Any) -> Any:
    """Journal-safe projection (live objects are recorded by type name)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()
                if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return f"<{type(value).__name__}>"


# --------------------------------------------------------------------------
# CLI (the pod-side entry point)
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Foundation Learner B200 combined-session supervisor")
    parser.add_argument("--config", required=True,
                        help="flb200.session_config.v1 JSON file")
    parser.add_argument("--out", default=None, help="session output directory")
    parser.add_argument("--terminate-only", action="store_true",
                        help="run only the configured terminate_command "
                             "(the entry script's preflight refused)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing journal and start over")
    return parser


_TRANSIENT_ERRNOS = frozenset({
    errno.ENOSPC, errno.EIO, errno.ENOMEM, errno.EAGAIN, errno.ECONNRESET,
    errno.ETIMEDOUT, errno.ECONNREFUSED, errno.ENETUNREACH, errno.EHOSTUNREACH})


def _transient_setup_error(exc: BaseException) -> bool:
    if isinstance(exc, (MemoryError, EvictedBySignal)):
        return True
    return isinstance(exc, OSError) and exc.errno in _TRANSIENT_ERRNOS


def _terminate_without_supervisor(config: "SessionConfig", out_dir: str,
                                  exc: BaseException) -> None:
    """Last-resort termination when the supervisor cannot even be built.

    A refusal in the constructor (corrupt journal, foreign session_id,
    guard failure) happens before ``run``'s ``finally`` exists, so without
    this the configured ``terminate_command`` would never fire and the
    accelerator would bill until the provider-side ``terminateAfter``.
    The command comes straight from the raw config payload; nothing else
    is trusted at this point.
    """
    payload = getattr(config, "payload", None)
    payload = payload if isinstance(payload, Mapping) else {}
    command = payload.get("terminate_command")
    note = {"event": "TERMINATE_WITHOUT_SUPERVISOR",
            "reason": repr(exc)[:400], "terminate_command": bool(command)}
    print(json.dumps(note, sort_keys=True), file=sys.stderr)
    if not command or (isinstance(command, str)
                       and command.startswith("UNRESOLVED")):
        return
    try:
        subprocess.run(
            command, shell=isinstance(command, str), check=False,
            timeout=float(payload.get("terminate_timeout_seconds")
                          or TERMINATE_TIMEOUT_SECONDS),
            env=SessionSupervisor._child_env(None), capture_output=True)
    except Exception as fail:  # noqa: BLE001 - recorded, never raised over exc
        print(json.dumps({"event": "TERMINATE_WITHOUT_SUPERVISOR_FAILED",
                          "error": repr(fail)[:400]}), file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """The pod-side entry point (``deploy/fl_b200_entry.sh`` execs this).

    It builds the supervisor and then wires the PRODUCTION campaign entry onto
    it (``campaign.entry.attach_to_supervisor``), which applies the §22
    determinism configuration and supplies the real ``StageContext`` factory
    and ladder runner.  Before this existed, a pod-side run reached
    ``RUN_FL_LADDER`` and refused for want of a context factory (R-C1).
    """
    args = build_parser().parse_args(argv)
    from ..eviction import install_eviction_handler

    install_eviction_handler()
    config = None
    if not args.out and not args.terminate_only:
        build_parser().error("--out is required")
    if getattr(args, "terminate_only", False):
        # the entry script's preflight refused: run the configured
        # terminate_command and nothing else (no supervisor is built)
        try:
            raw = SessionConfig.load(args.config)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"event": "TERMINATE_ONLY_CONFIG_UNREADABLE",
                              "error": repr(exc)[:300]}), file=sys.stderr)
            return 2
        _terminate_without_supervisor(raw, args.out, RuntimeError(
            "fl_b200_entry preflight refused"))
        return 0
    try:
        config = SessionConfig.load(args.config)
        supervisor = SessionSupervisor(config=config, out_dir=args.out)
        if supervisor.rehearsal:
            print(f"*** {REHEARSAL_LABEL}: this session is NOT a scientific "
                  f"run ***", file=sys.stderr)
        from . import entry as campaign_entry

        # attach_to_supervisor configures determinism (imports torch, asserts
        # deterministic algorithms took effect): a CUDA-init problem on a
        # fresh pod raises HERE, before run()'s finally exists
        campaign_entry.attach_to_supervisor(supervisor)
    except BaseException as exc:  # noqa: BLE001 - the pod must not be stranded
        if config is None:
            # the config could not even be loaded: try the RAW file for a
            # terminate_command before giving up on termination
            try:
                config = SessionConfig.load(args.config)
            except BaseException:  # noqa: BLE001 - nothing may escape here
                config = None
        if config is not None:
            _terminate_without_supervisor(config, args.out, exc)
        # Only a SMALL, explicit set is transient (same policy as
        # training/stability.classify_exception): ENOSPC/EIO/ENOMEM/EAGAIN
        # and the connection errnos.  FileNotFound/Permission/ReadOnly are
        # deterministic across pods and carry the marker — silence here is
        # indistinguishable from an eviction and buys a reacquisition loop.
        if _transient_setup_error(exc) and config is not None:
            print(f"REFUSED (transient): supervisor setup failed: {exc!r}",
                  file=sys.stderr, flush=True)
        else:
            print("ZERO_TOUCH_ABORTED_AT_SUPERVISOR_SETUP", flush=True)
        raise
    status = supervisor.run(resume=not args.no_resume)
    print(json.dumps({"outcome": status["outcome"],
                      "states_completed": status["states_completed"]},
                     indent=2, sort_keys=True))
    # the ONE place the driver's completion witness is emitted for a
    # combined session; O1's own copy was namespaced on re-emission
    print(_session_marker(status), flush=True)
    return 0 if status["outcome"] == "COMPLETE" else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
