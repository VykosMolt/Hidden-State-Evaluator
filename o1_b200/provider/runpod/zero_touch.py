#!/usr/bin/env python3
"""RunPod zero-touch session driver (B300 primary / B200 fallback, spot).

Layering: the SCIENTIFIC zero-touch state machine (hardware validation,
non-O1 equivalence, bounded benchmark, backend selection, replacement
precommit, external commit verification, affordability gate, calibration,
record verify) runs ON THE POD via the container start command.  This
driver owns the PROVIDER side with no human interaction:

  1. live-mutation interlock (or immediate refusal);
  2. read-only preflight + FRESH quote (stale quotes refused);
  3. canonical all-profile deployment render + authorization hash check;
  4. reconciled, duplicate-safe INTERRUPTIBLE Pod creation via the pinned
     GraphQL surface; watchdogs armed;
  5. monitor to RUNNING; poll logs/status; budget soft/hard stops;
  6. EVICTION of interruptible capacity is normal: terminate the remnant,
     carry the spend forward, re-quote fresh (profile re-selected), and
     reacquire — bounded by the hard dollar budget, max_pod_creations, and
     a zero-progress repeat-failure guard; the on-pod state machine
     revalidates hardware/environment from scratch on every pod and
     resumes from durable off-pod state (rows/checkpoints);
  7. wait for the pod's machine-readable FINAL_STATUS artifact;
  8. verified result download;
  9. TERMINATE (never merely stop) + independent confirmation;
 10. machine-readable local diagnostics package.

Every failure path terminates the pod and returns a complete local
diagnostic package; no path asks the user to SSH, read logs, choose a
datacenter/backend/bid, monitor spend, press Terminate, or repair anything.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .adapter import RunpodV2Adapter
from .authorization import (
    AuthorizationError, CLI_FLAG, LiveMutationAuthorization,
)
from .billing import BudgetViolation
from .identityutil import utcnow_iso
from .lifecycle import LifecycleError, PodLifecycleController
from .pod_request import build_pod_request, render_canonical_deployment
from .policy import MAX_POD_ACQUISITIONS, PROFILE_PREFERENCE, PROFILES_BY_KEY
from .preflight import run_preflight
from .redaction import redact


def load_session_config(root: str) -> dict:
    """Deployment facts resolved before launch (image digest, identities)."""
    path = os.path.join(root, "o1_b200", "provider", "runpod",
                        "RUNPOD_SESSION_CONFIG.json")
    if not os.path.exists(path):
        raise AuthorizationError(
            "RUNPOD_SESSION_CONFIG.json missing: the authorized session "
            "requires the resolved image digest and identity hashes")
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    unresolved = [k for k, v in config.items()
                  if isinstance(v, str) and v.startswith("UNRESOLVED")]
    if unresolved:
        raise AuthorizationError(
            f"session config carries unresolved template fields {unresolved}")
    return config


def _render_all_profiles(config: dict) -> dict:
    reqs = {p.key: build_pod_request(profile=p,
                                     image_digest_ref=config["image_digest_ref"],
                                     datacenter_id="AUTHORIZED-ANY")
            for p in PROFILE_PREFERENCE}
    return render_canonical_deployment(reqs, config["identities"])


def run_session(*, authorization_path: str, out_dir: str,
                cli_args: list[str], base_url: str = "https://api.runpod.io",
                api_key: str | None = None,
                root: str | None = None, sleep=time.sleep,
                config: dict | None = None,
                spawn_watchdog_fn=None,
                progress_probe=None, spend_clock=None) -> dict:
    """progress_probe: optional zero-arg callable returning a monotonic
    progress marker (e.g. count of durably synced rows); used to abort a
    repeating zero-progress 'eviction' as a container defect."""
    os.makedirs(out_dir, exist_ok=True)
    root = root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    status: dict = {"schema": "o1b300.runpod_session_status.v2",
                    "utc": utcnow_iso(), "outcome": None, "steps": [],
                    "acquisitions": []}

    def step(name, detail=None):
        status["steps"].append({"step": name, "utc": utcnow_iso(),
                                "detail": redact(str(detail)) if detail else None})

    def finish(outcome, **extra):
        status["outcome"] = outcome
        status.update(extra)
        path = os.path.join(out_dir, "RUNPOD_SESSION_STATUS.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return status

    config = config or load_session_config(root)
    expected_identity = {
        "project": config["project"],
        "package_zip_sha256": config["package_zip_sha256"],
        "provider": "runpod",
        "budget_policy_sha256": config["budget_policy_sha256"],
        "deployment_spec_sha256": None,  # bound after render below
    }
    # 2. read-only preflight first (no authorization needed)
    pre = run_preflight(base_url=base_url, api_key=api_key)
    step("READONLY_PREFLIGHT", pre["verdict"])
    if pre["verdict"] != "PASS":
        return finish("REFUSED_PREFLIGHT", preflight=pre)
    if pre.get("capacity_unavailable_now"):
        # market state, not a software failure — but nothing can be rented
        # right now, so refuse cleanly before any authorization/mutation
        return finish("REFUSED_NO_CAPACITY", preflight=pre)

    # 3. canonical ALL-PROFILE deployment render + authorization binding
    rendered = _render_all_profiles(config)
    expected_identity["deployment_spec_sha256"] = rendered["request_sha256"]
    try:
        auth = LiveMutationAuthorization.verify(
            path=authorization_path, expected_identity=expected_identity,
            cli_args=cli_args,
            nonce_ledger=os.path.join(out_dir, "consumed_nonces.txt"))
    except AuthorizationError as exc:
        step("AUTHORIZATION_REFUSED", exc)
        return finish("LIVE_MUTATION_NOT_AUTHORIZED", error=str(exc))
    step("AUTHORIZED", "interlock satisfied")

    adapter_kw = {}
    if spend_clock is not None:
        adapter_kw["spend_clock"] = spend_clock
    adapter = RunpodV2Adapter(base_url=base_url, api_key=api_key,
                              authorization=auth, sleep=sleep, **adapter_kw)
    controller_kw = {}
    if spawn_watchdog_fn is not None:
        controller_kw["spawn_watchdog_fn"] = spawn_watchdog_fn

    def pod_done(pod):
        # the on-pod state machine emits this marker after writing its
        # machine-readable FINAL_STATUS and packaging results
        try:
            return "ZERO_TOUCH_COMPLETE" in adapter.get_container_logs(
                pod.id, tail=50)
        except Exception:  # noqa: BLE001 - log endpoint may lag
            return False

    last_progress = progress_probe() if progress_probe else None
    evictions_seen = 0

    def zero_progress_abort() -> bool:
        """True when a SECOND consecutive interruption arrives with zero new
        durable progress — treated as a container defect, not an eviction."""
        nonlocal last_progress, evictions_seen
        evictions_seen += 1
        if progress_probe is None:
            return False
        progress = progress_probe()
        if progress == last_progress and evictions_seen > 1:
            return True
        last_progress = progress
        return False

    attempt = 0
    while True:
        attempt += 1
        if attempt > MAX_POD_ACQUISITIONS:
            return finish("ABORTED_ACQUISITION_LIMIT",
                          error=f"exceeded {MAX_POD_ACQUISITIONS} "
                                f"acquisitions")
        controller = PodLifecycleController(adapter, out_dir, sleep=sleep,
                                            **controller_kw)
        pod_id = None
        try:
            # fresh quote every acquisition: live spot rate, profile
            # re-selected under the frozen preference order
            quote = adapter.quote_instance(
                adapter_commit=config.get("adapter_commit", "UNKNOWN"))
            step("QUOTE", f"attempt {attempt}: {quote['profile']} "
                          f"({quote['profile_role']}) "
                          f"{quote['gpu_type_id']} @ "
                          f"{quote['bid_per_gpu_usd']}/h spot in "
                          f"{quote['datacenter_id']}")
            if quote["fallback_reason"]:
                step("FALLBACK_SELECTED", quote["fallback_reason"])
            status["acquisitions"].append(
                {"attempt": attempt, "profile": quote["profile"],
                 "fallback_reason": quote["fallback_reason"],
                 "bid_per_gpu_usd": quote["bid_per_gpu_usd"]})
            profile = PROFILES_BY_KEY[quote["profile"]]
            limit = adapter.validate_quote(quote)["hard_compute_seconds"]
            env_values = {
                "O1_B200_OUT": "/outputs",
                "O1_B200_RESULT_DESTINATION":
                    config.get("result_destination",
                               config.get("result_source", "")),
                "O1_ACQUIRED_PROFILE": quote["profile"],
                "O1_SESSION_AUTHORIZED_SECONDS": str(limit),
                "O1_HOURLY_RATE_USD": quote["total_projected_hourly_usd"],
                "O1_IMAGE_DIGEST": config["image_digest_ref"],
            }
            if os.environ.get("HF_TOKEN"):
                env_values["HF_TOKEN"] = os.environ["HF_TOKEN"]
            req = build_pod_request(
                profile=profile,
                image_digest_ref=config["image_digest_ref"],
                datacenter_id=quote["datacenter_id"],
                env_values=env_values)
            # 4. provision (reconciled, duplicate-safe) + watchdogs
            pod_id = controller.provision(req, rendered)
            # 5. run
            startup = controller.wait_until_running(pod_id)
            if startup == "EVICTED":
                step("EVICTED_DURING_STARTUP", f"attempt {attempt}")
                if zero_progress_abort():
                    return finish(
                        "ABORTED_REPEATED_FAILURE_NO_PROGRESS",
                        error="a second consecutive interruption with zero "
                              "durable progress is treated as a container "
                              "defect, not an eviction")
                continue
            outcome = controller.monitor(pod_id, until=pod_done)
            step("MONITOR_RESULT", f"attempt {attempt}: {outcome}")
            if outcome == "EVICTED":
                if zero_progress_abort():
                    return finish(
                        "ABORTED_REPEATED_FAILURE_NO_PROGRESS",
                        error="a second consecutive interruption with zero "
                              "durable progress is treated as a container "
                              "defect, not an eviction")
                continue
            # 6/7. results: collect logs + download outputs
            controller.collect_logs(pod_id)
            dest = os.path.join(out_dir, "downloaded_results")
            adapter.download_results(config.get("result_source", dest), dest)
            step("RESULTS_DOWNLOADED", dest)
        except BudgetViolation as exc:
            step("BUDGET_STOP", exc)
            if pod_id is not None:
                controller.terminate_and_confirm(pod_id)
            return finish("ABORTED_BUDGET", error=redact(str(exc)))
        except AuthorizationError as exc:
            # creation-slot exhaustion or expiry mid-session
            step("AUTHORIZATION_STOP", exc)
            if pod_id is not None:
                controller.terminate_and_confirm(pod_id)
            return finish("ABORTED_AUTHORIZATION_EXHAUSTED",
                          error=redact(str(exc)))
        except (LifecycleError, Exception) as exc:  # noqa: BLE001
            step("SESSION_FAILURE", exc)
            if pod_id is not None:
                confirmed = controller.terminate_and_confirm(pod_id)
                return finish("ABORTED_TERMINATED" if confirmed
                              else "ABORTED_TERMINATION_UNCONFIRMED",
                              error=redact(str(exc)))
            return finish("ABORTED_BEFORE_CREATE", error=redact(str(exc)))
        # 8. terminate + confirm (always; stop is never final)
        confirmed = controller.terminate_and_confirm(pod_id)
        return finish("COMPLETE" if confirmed else "TERMINATION_UNCONFIRMED",
                      termination_confirmed=confirmed,
                      acquisitions_used=attempt)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--authorization", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--execute-authorized-rental", action="store_true")
    p.add_argument("--base-url", default="https://api.runpod.io")
    a = p.parse_args()
    cli = [CLI_FLAG] if a.execute_authorized_rental else []
    status = run_session(authorization_path=a.authorization, out_dir=a.out,
                         cli_args=cli, base_url=a.base_url)
    print(json.dumps({"outcome": status["outcome"]}, indent=2))
    return 0 if status["outcome"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
