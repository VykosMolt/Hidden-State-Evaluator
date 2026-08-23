"""RunPod Pod lifecycle controller.

States come from the pinned schema (PROVISIONING, STARTING, RUNNING, EXITED,
ERROR, TERMINATED); anything else fails closed in models.PodModel.parse.

Responsibilities: reconcile-before-create (launch nonce + canonical name),
duplicate prevention, atomic Pod-ID record, asynchronous polling to RUNNING
or timeout, container-failure detection with automatic log collection,
budget-aware work gating (95% soft stop / 100% hard stop via SpendTracker),
terminate-on-any-hard-failure, redundant termination confirmation, final
billing query, machine-readable lifecycle report.
"""
from __future__ import annotations

import json
import os
import time

from .adapter import RunpodAdapterError, RunpodV2Adapter
from .billing import BudgetViolation, as_money, remaining_compute_seconds
from .identityutil import utcnow_iso
from .redaction import redact
from .watchdog_terminate import spawn_watchdog


class LifecycleError(RuntimeError):
    def __init__(self, msg: str):
        super().__init__(redact(msg))


def _billed_usd(bill):
    """Total USD from a billing response, or None when it says nothing.

    None is deliberately not zero: an unparseable billing payload must not
    look like "this pod has cost nothing".
    """
    if isinstance(bill, dict):
        for key in ("totalAmount", "total_amount", "amount", "totalUsd"):
            if key in bill:
                try:
                    return as_money(bill[key])
                except Exception:  # noqa: BLE001
                    return None
        meta = bill.get("metadata") or {}
        totals = meta.get("totals") if isinstance(meta, dict) else None
        if isinstance(totals, dict) and "totalAmount" in totals:
            try:
                return as_money(totals["totalAmount"])
            except Exception:  # noqa: BLE001
                return None
    return None


class PodLifecycleController:
    def __init__(self, adapter: RunpodV2Adapter, out_dir: str,
                 *, startup_timeout_seconds: float = 900,
                 sleep=time.sleep, spawn_watchdog_fn=spawn_watchdog,
                 attempt: int = 1, monitor_timeout_seconds: float | None = None):
        self.adapter = adapter
        self.out_dir = out_dir
        self.startup_timeout = startup_timeout_seconds
        self.monitor_timeout = monitor_timeout_seconds
        self.sleep = sleep
        self.spawn_watchdog = spawn_watchdog_fn
        self.attempt = int(attempt)
        os.makedirs(out_dir, exist_ok=True)
        self.pod_id_path = os.path.join(out_dir, "POD_ID.json")
        self.report_path = os.path.join(out_dir, "LIFECYCLE_REPORT.json")
        # Per-acquisition evidence.  A session may create several pods; a
        # single shared report/id file keeps only the LAST one, silently
        # erasing (for example) a TERMINATION_UNCONFIRMED_LOUD_FAILURE on
        # an earlier pod — exactly the evidence an operator needs.
        self.history_dir = os.path.join(out_dir, "lifecycle")
        os.makedirs(self.history_dir, exist_ok=True)
        self.attempt_report_path = os.path.join(
            self.history_dir, f"LIFECYCLE_REPORT.attempt{self.attempt:02d}.json")
        self.pod_id_ledger = os.path.join(out_dir, "POD_IDS.jsonl")
        self.events: list[dict] = []

    def _event(self, kind: str, **fields):
        entry = {"event": kind, "utc": utcnow_iso(), **fields}
        self.events.append(json.loads(redact(json.dumps(entry))))

    def _record_pod_id(self, pod_id: str) -> None:
        record = {"pod_id": pod_id, "attempt": self.attempt,
                  "utc": utcnow_iso()}
        tmp = self.pod_id_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.pod_id_path)
        # append-only ledger of EVERY pod this session created, so an
        # earlier pod can never be lost from the record
        with open(self.pod_id_ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def provision(self, req, rendered) -> str:
        """Create exactly one Pod (reconciled, duplicate-safe)."""
        self._event("PROVISION_REQUESTED", name=req.name)
        pod = self.adapter.create_instance(req, rendered)
        self._record_pod_id(pod.id)
        self._event("POD_CREATED", pod_id=pod.id, status=pod.status)
        try:
            return self._arm_and_return(pod)
        except Exception as exc:  # noqa: BLE001
            # The pod EXISTS and is billing from here on.  Anything that
            # goes wrong while arming (an expired quote, a budget refusal)
            # must terminate it rather than leave a billable orphan whose
            # id the caller never received.
            self._event("ARMING_FAILED", pod_id=pod.id, error=str(exc)[:300])
            try:
                self.terminate_and_confirm(pod.id)
            finally:
                raise

    def _arm_and_return(self, pod) -> str:
        # arm the independent watchdog IMMEDIATELY, before waiting on startup.
        # Its deadline is the REMAINING compute allocation (spend from earlier
        # evicted pods already carried into the tracker), never the full
        # allocation: N pods each armed at the full budget would authorize
        # N x MAX_COMPUTE_USD of unattended runtime.
        self.adapter.validate_quote()
        limit = remaining_compute_seconds(
            self.adapter.accepted_quote["total_projected_hourly_usd"],
            self.adapter.session_spend_usd())
        if limit <= 0:
            # provision()'s handler terminates the pod for every failure in
            # here, so this must not terminate a second time
            self._event("BUDGET_EXHAUSTED_AT_PROVISION")
            raise BudgetViolation(
                "no compute allocation remains after provisioning")
        # Pass the adapter's own host: defaulting to api.runpod.io meant a
        # rehearsal against a mock armed a watchdog that would fire
        # POST/DELETE at PRODUCTION with the real operator key, and the
        # production host wiring was never exercised at all.
        wd_kw = {}
        base = getattr(self.adapter.readonly, "base_url", "")
        if base:
            from urllib.parse import urlparse
            parsed = urlparse(base)
            if parsed.hostname:
                wd_kw = {"host": parsed.hostname,
                         "scheme": parsed.scheme or "https"}
                if parsed.port:
                    wd_kw["port"] = parsed.port
        try:
            self.watchdog = self.spawn_watchdog(
                pod_id=pod.id, hard_limit_seconds=limit,
                out_dir=self.out_dir, **wd_kw)
        except TypeError:
            # injected test doubles may take only the core arguments
            self.watchdog = self.spawn_watchdog(
                pod_id=pod.id, hard_limit_seconds=limit, out_dir=self.out_dir)
        # PROVE it armed.  Popen returning is not evidence: a watchdog that
        # died on import was recorded as ARMED, so the independent
        # termination path could be absent while the report said otherwise.
        confirm = getattr(self.watchdog, "confirm_armed", None)
        armed = confirm() if callable(confirm) else True
        self._event("WATCHDOG_ARMED", hard_limit_seconds=limit,
                    confirmed=bool(armed),
                    session_spend_usd=str(self.adapter.session_spend_usd()))
        if not armed:
            # provision() terminates the pod for anything raised in here,
            # which is the right answer: a billing pod with no independent
            # stop is worse than a refused acquisition.
            raise LifecycleError(
                "the independent watchdog did not confirm arming; refusing "
                "to run a billing pod whose only remaining stop is the "
                "provider-side terminateAfter")
        return pod.id

    def wait_until_running(self, pod_id: str) -> str:
        """Returns "RUNNING", or "EVICTED" when interruptible capacity was
        reclaimed before the container came up (normal spot behavior: the
        remnant is terminated and the session may reacquire)."""
        # Checkpoint the durable spend THROUGH the startup wait.  Only
        # monitor() checkpointed, so a driver crash during the up-to-900 s
        # startup window forgot that pod's billing entirely and a restart
        # was handed the allowance back.  The pod bills from provisioning,
        # so this window is never free.
        def _checkpointing_sleep(seconds):
            try:
                self.adapter.checkpoint_spend()
            except Exception:  # noqa: BLE001 - never block startup on the ledger
                pass
            return self.sleep(seconds)

        try:
            pod = self.adapter.wait_for_state(
                pod_id, "RUNNING", timeout_seconds=self.startup_timeout,
                sleep=_checkpointing_sleep)
        except RunpodAdapterError as exc:
            try:
                self.adapter.checkpoint_spend()
            except Exception:  # noqa: BLE001
                pass
            if "EVICTION_SUSPECTED" in str(exc):
                self._event("EVICTED_BEFORE_RUNNING", pod_id=pod_id)
                self.collect_logs(pod_id)
                if not self.terminate_and_confirm(pod_id):
                    return "EVICTED_TERMINATION_UNCONFIRMED"
                return "EVICTED"
            self._event("STARTUP_FAILED", error=str(exc))
            self.collect_logs(pod_id)
            self.terminate_and_confirm(pod_id)
            raise LifecycleError(f"startup failed: {exc}") from None
        self.adapter.spend.mark_pod_started()
        self._event("POD_RUNNING", pod_id=pod.id,
                    started_at=pod.started_at)
        return "RUNNING"

    def monitor(self, pod_id: str, *, poll_seconds: float = 20.0,
                until=None) -> str:
        """Poll status/budget; returns 'COMPLETE' | raises on failure.

        Bounded by monitor_timeout_seconds when supplied: a pod that stays
        RUNNING and never emits its completion witness would otherwise loop
        until the budget soft-stop, paying for the whole remaining
        allocation with nothing to show.
        """
        started = time.monotonic()
        while True:
            if self.monitor_timeout is not None and \
                    time.monotonic() - started > self.monitor_timeout:
                self._event("MONITOR_TIMEOUT",
                            seconds=round(time.monotonic() - started, 1))
                self.collect_logs(pod_id)
                self.terminate_and_confirm(pod_id)
                raise LifecycleError(
                    f"pod {pod_id} produced no completion witness within "
                    f"{self.monitor_timeout}s; logs collected, terminated")
            pod = self.adapter.get_instance(pod_id)
            if self.adapter.spend is not None:
                try:
                    bill = self.adapter.get_billing_usage(pod_id)
                    self._event("BILLING_SAMPLE", data=str(bill)[:200])
                    # FEED the tracker.  effective_spend() documents itself
                    # as max(monotonic projection, live billing), but nothing
                    # ever called record_live_billing outside tests -- so the
                    # live half was unreachable and anything the quote
                    # under-priced (container disk is quoted at 0.0000/h by
                    # default) was invisible to every budget stop.
                    billed = _billed_usd(bill)
                    if billed is not None:
                        self.adapter.spend.record_live_billing(billed)
                        self._event("LIVE_BILLING_RECORDED",
                                    billed_usd=str(billed))
                except Exception:  # noqa: BLE001 - billing is supplementary
                    pass
                # A1: the durable ledger must carry THIS pod's spend, not
                # just earlier pods' carryover, or a driver crash mid-pod
                # (Ctrl-C, OOM, reboot) resets the budget on the rerun
                self.adapter.checkpoint_spend()
                if self.adapter.spend.must_terminate():
                    self._event("BUDGET_HARD_STOP")
                    self.terminate_and_confirm(pod_id)
                    raise BudgetViolation(
                        "100% of compute allocation: immediate termination "
                        "requested; no further cleanup on the billable pod")
                if self.adapter.spend.state() == "SOFT_STOP":
                    self._event("BUDGET_SOFT_STOP")
                    return "SOFT_STOP"
            if pod.status == "EXITED":
                # On interruptible capacity, EXITED without the completion
                # marker is an eviction (spot stop), not a scientific
                # failure: collect what remains, terminate the remnant, and
                # let the session decide on bounded reacquisition.
                if until is not None and until(pod):
                    return "COMPLETE"
                self._event("EVICTED", pod_id=pod_id)
                self.collect_logs(pod_id)
                # spend is frozen INSIDE terminate_and_confirm, after the
                # confirmation poll: a pod still bills while terminating.
                # An UNCONFIRMED termination must stop the session: the
                # remnant may still bill, and a second pod on top of it is
                # exactly the double-spend the ladder forbids.
                if not self.terminate_and_confirm(pod_id):
                    return "EVICTED_TERMINATION_UNCONFIRMED"
                return "EVICTED"
            if pod.status == "ERROR":
                self._event("CONTAINER_FAILURE", status=pod.status)
                self.collect_logs(pod_id)
                self.terminate_and_confirm(pod_id)
                raise LifecycleError(
                    f"container reached {pod.status}; logs collected, pod "
                    f"terminated")
            if pod.status == "TERMINATED":
                self._event("POD_TERMINATED_EXTERNALLY")
                return "TERMINATED"
            if until is not None and until(pod):
                return "COMPLETE"
            self.sleep(poll_seconds)

    def collect_logs(self, pod_id: str) -> None:
        for source, fn in (("container", self.adapter.get_container_logs),
                           ("system", self.adapter.get_system_logs)):
            try:
                text = fn(pod_id, tail=500)
                path = os.path.join(self.out_dir, f"pod_{source}_logs.txt")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(redact(text))
                self._event("LOGS_COLLECTED", source=source)
            except Exception as exc:  # noqa: BLE001
                self._event("LOG_COLLECTION_FAILED", source=source,
                            error=str(exc)[:200])

    def terminate_and_confirm(self, pod_id: str) -> bool:
        confirmed = False
        primary_error = None
        try:
            self.adapter.terminate_instance(pod_id)
            self._event("TERMINATE_REQUESTED", path="primary")
            confirmed = self.adapter.confirm_terminated(pod_id,
                                                        sleep=self.sleep)
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)
            self._event("PRIMARY_TERMINATION_FAILED", error=primary_error[:300])
        if not confirmed:
            try:
                wd = getattr(self, "watchdog", None)
                if wd is not None:
                    wd.terminate_now()
                    confirmed = self.adapter.confirm_terminated(
                        pod_id, sleep=self.sleep)
                    self._event("WATCHDOG_TERMINATION",
                                confirmed=confirmed)
            except Exception as exc:  # noqa: BLE001
                self._event("WATCHDOG_TERMINATION_FAILED",
                            error=str(exc)[:300])
        if not confirmed:
            self._event("TERMINATION_UNCONFIRMED_LOUD_FAILURE")
        # Freeze this pod's spend into the session carryover only NOW: the
        # pod kept billing throughout the terminate + confirmation poll
        # (up to confirm timeout), and that time must be charged before a
        # replacement pod's meter starts.
        if self.adapter.spend is not None:
            carried = self.adapter.spend.mark_pod_stopped()
            # Durable, not just in-memory: the nonce ledger already survived
            # a restart while the DOLLARS did not, so every rerun handed the
            # next pod a fresh full compute allocation.
            self.adapter._persist_carryover(carried)
            self._event("SPEND_FROZEN", session_spend_usd=str(carried),
                        durable=True)
        try:
            bill = self.adapter.get_billing_usage(pod_id)
            self._event("FINAL_BILLING", data=str(bill)[:300])
        except Exception:  # noqa: BLE001
            self._event("FINAL_BILLING_UNAVAILABLE")
        self.write_report(confirmed)
        return confirmed

    def write_report(self, termination_confirmed: bool) -> None:
        report = {
            "schema": "o1b300.runpod_lifecycle_report.v2",
            "attempt": self.attempt,
            "termination_confirmed": termination_confirmed,
            "events": self.events,
            "machine_readable": True,
        }
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        # the per-attempt copy is the durable evidence; the shared path is
        # kept as the "latest" convenience view
        for path in (self.attempt_report_path, self.report_path):
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
