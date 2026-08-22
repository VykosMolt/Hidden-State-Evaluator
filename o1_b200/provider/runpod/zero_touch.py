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
import hashlib
import json
import os
import re
import time

from .adapter import WitnessShapeUnrecognised, RunpodV2Adapter
from .authorization import (
    AuthorizationError, CLI_FLAG, LiveMutationAuthorization,
)
from .billing import BudgetViolation, remaining_compute_seconds
from .identityutil import utcnow_iso
from .lifecycle import LifecycleError, PodLifecycleController
from .pod_request import build_pod_request, render_canonical_deployment
from .policy import MIN_REACQUISITION_SECONDS, MAX_POD_ACQUISITIONS, PROFILE_PREFERENCE, PROFILES_BY_KEY
from .preflight import run_preflight
from .redaction import redact, register_env_secrets


class DeterministicPodFailure(RuntimeError):
    """The pod reported a reproducible failure; reacquiring cannot help."""


#: Path-in-store of the result archive.  ONE definition: production_entry
#: pushes here and the off-pod driver reads from here, both relative to the
#: same destination, so a session prefix can never move one without the
#: other.  Storing an absolute result_source separately is what let the
#: prefixed store publish to <prefix>/results/... while the driver looked
#: for results/... and declared a fully paid, fully successful run ABORTED.
RESULT_ARCHIVE_REL = "results/o1_results.tar.gz"

#: Keys the session config MUST carry.  Absent is NOT the same as
#: UNRESOLVED: an absent artifact_source silently became "" and the pod
#: started with nothing to fetch, burning acquisitions before anything could
#: notice.  Missing is now a refusal, exactly like unresolved.
REQUIRED_CONFIG_KEYS = (
    "artifact_source", "result_destination", "image_digest_ref",
    "project", "identities",
    # read with config[...] for the expected identity; absent used to raise
    # a bare KeyError outside the authorization check
    "package_zip_sha256", "budget_policy_sha256",
)


#: The pod's completion witness: a WHOLE LINE, not a substring.  In a
#: combined session the O1 phase runs as a child of the FL supervisor, which
#: rewrites O1's own markers (O1_PHASE_COMPLETE) and prints the session's
#: marker once at its end; a substring match would have read a quoted,
#: prefixed or mid-line occurrence as the verdict.
# LINE-anchored.  A delimiter-only guard read a quoted or prose mention
# ('the driver wants ZERO_TOUCH_COMPLETE') as the verdict, and with
# last-wins a genuine abort followed by such a line became COMPLETE.  The
# provider's log shape is handled where it belongs (adapter.normalize_log_body
# turns a list body into real lines); the only prefix tolerated is a
# timestamp / bracketed tag at the start of the line.
_WITNESS_RE = re.compile(
    r"^[ \t]*(?:[\[(][^\])\n]*[\])][ \t]*)?(?:\d[\dT:.\-Z+]*[ \t]+)?"
    r"ZERO_TOUCH_(COMPLETE|ABORTED_AT_([A-Z0-9_]+))[ \t\r]*$",
    re.MULTILINE)


def completion_verdict(log_tail: str) -> str | None:
    """``"COMPLETE"``, the abort state name, or None (no verdict yet).
    The LAST marker LINE wins."""
    verdict = None
    for m in _WITNESS_RE.finditer(log_tail or ""):
        verdict = "COMPLETE" if m.group(1) == "COMPLETE" else (
            m.group(2) or "UNSPECIFIED_STATE")
    return verdict


def _refusal_marker_path(authorization_path: str, config: dict) -> str:
    key = hashlib.sha256(json.dumps(
        {"image": config.get("image_digest_ref"),
         "artifact_source": config.get("artifact_source"),
         "result_destination": config.get("result_destination"),
         "fl_session_config": config.get("fl_session_config", "")},
        sort_keys=True).encode()).hexdigest()[:16]
    return f"{authorization_path}.deterministic_refusal.{key}.json"


def _record_deterministic_refusal(authorization_path: str, config: dict,
                                  error: str) -> None:
    try:
        path = _refusal_marker_path(authorization_path, config)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"schema": "o1b300.deterministic_refusal.v1",
                       "utc": utcnow_iso(), "error": redact(error)[:1000],
                       "image_digest_ref": config.get("image_digest_ref")},
                      fh, indent=2, sort_keys=True)
    except OSError:
        pass


def _prior_deterministic_refusal(authorization_path: str, config: dict):
    path = _refusal_marker_path(authorization_path, config)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        doc = {}
    doc["marker"] = path
    return doc


def result_archive_uri(config: dict) -> str:
    """Where this session's result archive lives, derived — never stored."""
    destination = config.get("result_destination", "")
    if not destination:
        return ""
    return f"{destination.rstrip('/')}/{RESULT_ARCHIVE_REL}"


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
    missing = [k for k in REQUIRED_CONFIG_KEYS
               if not config.get(k)]
    if missing:
        raise AuthorizationError(
            f"session config is missing required field(s) {missing}; an "
            f"absent key is not a default — a missing artifact_source made "
            f"the pod start with nothing to fetch and cost acquisitions "
            f"before anything refused")
    stored = config.get("result_source")
    derived = result_archive_uri(config)
    if stored and stored != derived:
        raise AuthorizationError(
            f"result_source {stored!r} disagrees with the location the pod "
            f"actually publishes to, {derived!r} (result_destination + "
            f"{RESULT_ARCHIVE_REL}). Remove result_source — it is derived — "
            f"rather than maintaining two paths for one object")
    return config


def identity_env_values(config: dict) -> dict:
    """Env values that ARE deployment identity (frozen into the rendering).

    These decide what the pod fetches and where it publishes, so they are
    bound by the authorization rather than free at launch.
    """
    return {
        "O1_B200_OUT": config.get("pod_out_dir", "/outputs"),
        "O1_B200_ARTIFACT_SOURCE": config.get("artifact_source", ""),
        "O1_B200_RESULT_DESTINATION": config.get("result_destination", ""),
        # "" = O1-only.  A path here selects the combined O1 -> FL session;
        # start_b300.sh refuses at once if the file or the FL entry is
        # absent, rather than discovering it after the accelerator is paid
        # for.  Identity-bound, so enabling it needs a fresh authorization.
        "O1_FL_SESSION_CONFIG": config.get("fl_session_config", ""),
        "O1_IMAGE_DIGEST": config["image_digest_ref"],
    }


def _render_all_profiles(config: dict) -> dict:
    env_values = identity_env_values(config)
    reqs = {p.key: build_pod_request(profile=p,
                                     image_digest_ref=config["image_digest_ref"],
                                     datacenter_id="AUTHORIZED-ANY",
                                     env_values=env_values)
            for p in PROFILE_PREFERENCE}
    return render_canonical_deployment(reqs, config["identities"])


def _expected_result_digest(config: dict,
                            launch_nonce: str | None = None
                            ) -> tuple[str, str | None]:
    """The result digest the POD recorded, read from the durable store.

    Returns (state, digest) where state is:
      "VERIFIABLE" — a sidecar exists and its digest was read;
      "ABSENT"     — no durable store is configured, or no sidecar exists;
      "ERROR"      — the store is configured but could not be consulted.

    ABSENT and ERROR are deliberately NOT the same: conflating them let a
    broken store yield the same COMPLETE verdict as a fully verified run.
    """
    destination = config.get("result_destination", "")
    if not destination:
        return ("ABSENT", None)
    try:
        import json as _json
        import tempfile

        from o1_b200.runner.durability import store_for_destination
        store = store_for_destination(destination)
        listing = store.list_prefix("results")
        target = next((r for r in listing
                       if r.endswith("o1_results.tar.gz.sha256")), None)
        if target is None:
            return ("ABSENT", None)
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "digest.json")
            store.fetch_file(target, local)
            with open(local, encoding="utf-8") as fh:
                doc = _json.load(fh)
            # A sidecar from a DIFFERENT session is not this run's evidence.
            # The result key is fixed and run-agnostic, so without this an
            # aborted run adopts the previous session's archive AND the
            # sidecar that vouches for it, and the cross-check passes.
            if launch_nonce:
                got = doc.get("launch_nonce", "")
                if got != launch_nonce:
                    return ("FOREIGN", got or None)
            return ("VERIFIABLE", doc["sha256"])
    except Exception as exc:  # noqa: BLE001
        return ("ERROR", redact(str(exc))[:200])


def durable_progress_probe(destination: str):
    """Off-pod durable-progress reader for the zero-progress guard.

    The driver cannot see the pod's filesystem, but it CAN read the same
    durable store the pod syncs to: the committed-row count in the durable
    records manifest is real, externally-visible progress.  Returns None on
    any error (unknown != zero, so a store hiccup never trips the guard).
    """
    def probe():
        try:
            from o1_b200.runner.durability import (
                CheckpointDurability, store_for_destination,
            )
            store = store_for_destination(destination)
            mirror = CheckpointDurability(
                store, remote_prefix="durable_o1_records")
            return mirror.latest_row_count()
        except Exception:  # noqa: BLE001 - progress is advisory
            return None
    return probe


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
    # register every configured credential literally, BEFORE any step can
    # be recorded, so redaction never depends on a value matching a shape
    register_env_secrets()
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
    # A9: a previous invocation recorded that THIS deployment fails
    # deterministically on the pod; refuse before any money moves
    prior = _prior_deterministic_refusal(authorization_path, config)
    if prior:
        step("DETERMINISTIC_REFUSAL_ON_RECORD", prior.get("error", "")[:200])
        return finish("REFUSED_DETERMINISTIC_FAILURE_ON_RECORD",
                      error=prior.get("error"),
                      note=(f"remove {prior.get('marker')} after fixing the "
                            f"cause (a new image digest or config clears it "
                            f"automatically)"))
    # F5: the pod needs HF_TOKEN for the checkpoint fetch and the durable
    # mirror, and the driver needs it for the result download; nothing
    # checked it before the first billable pod
    if str(config.get("artifact_source", "")).startswith("hf://") \
            and not os.environ.get("HF_TOKEN"):
        return finish("REFUSED_HF_TOKEN_UNSET",
                      error="HF_TOKEN is unset but the session fetches its "
                            "artifacts from an hf:// source; the pod would "
                            "refuse at HF_SCOPE after the image pull was "
                            "paid for")
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
    owned = (pre.get("checks") or {}).get("owned_pods") or {}
    if owned.get("unexpected_active_billable_pod"):
        # An active pod we did not expect is either a leftover from an
        # earlier run or someone else's work.  Launching alongside it would
        # add a second billable resource to an account already spending, so
        # refuse and name it rather than proceed.
        return finish("REFUSED_UNEXPECTED_ACTIVE_POD", preflight=pre,
                      error=f"account already has active pod(s) "
                            f"{owned.get('active_pods')}; terminate or "
                            f"account for them before launching")
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
            # keyed to the AUTHORIZATION, never to --out: a rerun with a
            # different out-dir must not get a fresh creation budget and a
            # fresh USD allocation under the same authorization
            nonce_ledger=authorization_path + ".consumed_nonces")
    except AuthorizationError as exc:
        step("AUTHORIZATION_REFUSED", exc)
        return finish("LIVE_MUTATION_NOT_AUTHORIZED", error=str(exc))
    step("AUTHORIZED", "interlock satisfied")

    adapter_kw = {}
    if spend_clock is not None:
        adapter_kw["spend_clock"] = spend_clock
    # Optional operator ordering of the FROZEN profiles (e.g. put B200
    # first when B300 demand makes it unobtainable).  This is a preference,
    # not a new capability: the authorization already commits to every
    # frozen profile's canonical body, so no unrendered deployment becomes
    # possible.  Recorded in the session status and in every quote.
    if config.get("profile_preference"):
        adapter_kw["profile_preference"] = list(config["profile_preference"])
        status["profile_preference"] = list(config["profile_preference"])
    adapter = RunpodV2Adapter(base_url=base_url, api_key=api_key,
                              authorization=auth, sleep=sleep, **adapter_kw)
    controller_kw = {}
    if spawn_watchdog_fn is not None:
        controller_kw["spawn_watchdog_fn"] = spawn_watchdog_fn

    def _make_result_witness():
        """Durable second witness: did the pod publish its results archive?

        Independent of the pod's log endpoint, so a lost/500ing log API at
        the exact moment the container exits cannot be mistaken for
        "unfinished" and trigger a paid reacquisition.
        """
        destination = config.get("result_destination", "")
        if not destination:
            return None

        def witness():
            try:
                # THIS session's archive, not merely an archive.  The key is
                # fixed and run-agnostic, so a leftover from a previous run
                # otherwise satisfies the witness, turns an evicted pod into
                # "COMPLETE", and the driver then downloads and reports the
                # old session's numbers as this one's verified result.
                state, _ = _expected_result_digest(
                    config, auth.launch_nonce)
                return state == "VERIFIABLE"
            except Exception:  # noqa: BLE001 - advisory witness only
                return False
        return witness

    result_witness = _make_result_witness()

    def pod_done(pod, retries: int = 3):
        """Completion witness.

        A transient failure of the log endpoint must NOT be read as "not
        finished": that would classify a completed pod as evicted and pay
        for a whole redundant reacquisition.  The log probe is retried, and
        a lost log endpoint falls back to the durable result witness the pod
        publishes before exiting.
        """
        for attempt_i in range(retries):
            try:
                tail = adapter.get_container_logs(pod.id, tail=50)
                verdict = completion_verdict(tail)
                if verdict is None and "ZERO_TOUCH_" in tail and any(
                        tok in tail for tok in ("ZERO_TOUCH_COMPLETE",
                                                "ZERO_TOUCH_ABORTED_AT_")):
                    raise WitnessShapeUnrecognised(
                        "the container log carries a completion token but "
                        "no line-anchored marker could be read; the log "
                        "shape is not one the driver decodes")
                if verdict and verdict != "COMPLETE":
                    # The pod reached a DETERMINISTIC verdict and said so.
                    # Reacquiring cannot help — a fresh pod runs the same
                    # gates against the same artifacts and fails identically
                    # — so this must never be mistaken for an eviction.
                    raise DeterministicPodFailure(
                        f"the pod aborted deterministically at {verdict}; "
                        f"reacquisition would repeat it")
                # No verdict in the log is NOT completion — not even when
                # the durable O1 archive exists: in a combined session that
                # archive is published at the END OF THE O1 PHASE, hours
                # before the FL half finishes, so a spot eviction mid-FL
                # would be reported COMPLETE and the FL work abandoned.  The
                # result witness is consulted only when the log endpoint
                # itself is unavailable (below).
                return verdict == "COMPLETE"
            except (DeterministicPodFailure, WitnessShapeUnrecognised):
                raise
            except Exception:  # noqa: BLE001 - log endpoint may lag
                if attempt_i + 1 < retries:
                    sleep(2.0)
        step("COMPLETION_WITNESS_LOG_UNAVAILABLE", pod.id)
        # The durable O1 archive may stand in for an unreadable log ONLY
        # when it can actually witness the session's end: the pod must have
        # EXITED, and the session must be O1-only.  In a combined session
        # the archive appears at the end of the O1 PHASE, hours before the
        # FL half finishes; consulting it from the RUNNING-pod poll loop
        # reported a live FL session COMPLETE and terminated it.
        if config.get("fl_session_config"):
            return False
        if getattr(pod, "status", None) != "EXITED":
            return False
        return bool(result_witness and result_witness())

    if progress_probe is None:
        destination = config.get("result_destination", "")
        if destination:
            progress_probe = durable_progress_probe(destination)

    last_progress = progress_probe() if progress_probe else None
    evictions_seen = 0
    zero_progress_streak = 0

    def zero_progress_abort() -> bool:
        """True when a SECOND CONSECUTIVE interruption arrives with zero new
        durable progress — treated as a container defect, not an eviction.

        "Consecutive" is a streak of interruptions each of which added no
        durable rows.  The earlier form compared against the last value
        and counted evictions cumulatively, so one productive pod followed
        by ONE unproductive one aborted the session — a single eviction
        before the replacement's first commit is ordinary spot behaviour.
        """
        nonlocal last_progress, evictions_seen, zero_progress_streak
        evictions_seen += 1
        if progress_probe is None:
            return False
        progress = progress_probe()
        if progress is None:            # unknown is not zero
            return False
        if last_progress is not None and progress <= last_progress:
            zero_progress_streak += 1
        else:
            zero_progress_streak = 0
        last_progress = progress
        status["durable_rows_committed"] = progress
        return zero_progress_streak >= 2

    def no_progress_abort():
        """The durable mirror still holds every row earlier pods committed;
        the status says so, so a paid-for partial run is never reported as
        if nothing existed."""
        return finish(
            "ABORTED_REPEATED_FAILURE_NO_PROGRESS",
            error="two consecutive interruptions with zero new durable "
                  "progress are treated as a container defect, not an "
                  "eviction",
            durable_rows_committed=last_progress,
            durable_rows_location=(
                config.get("result_destination", "") + "/durable_o1_records"
                if config.get("result_destination") else None),
            acquisitions_used=attempt)

    attempt = 0
    while True:
        attempt += 1
        if attempt > MAX_POD_ACQUISITIONS:
            return finish("ABORTED_ACQUISITION_LIMIT",
                          error=f"exceeded {MAX_POD_ACQUISITIONS} "
                                f"acquisitions")
        controller = PodLifecycleController(adapter, out_dir, sleep=sleep,
                                            attempt=attempt, **controller_kw)
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
                 "operator_preference_applied": quote.get(
                     "operator_preference_applied", False),
                 "bid_per_gpu_usd": quote["bid_per_gpu_usd"]})
            profile = PROFILES_BY_KEY[quote["profile"]]
            adapter.validate_quote(quote)
            # the pod's own runtime allowance is the REMAINING allocation,
            # net of everything earlier pods in this session already spent
            limit = remaining_compute_seconds(
                quote["total_projected_hourly_usd"],
                adapter.session_spend_usd())
            if limit <= 0:
                return finish("ABORTED_BUDGET",
                              error="no compute allocation remains; refusing "
                                    "to acquire another pod")
            if limit < MIN_REACQUISITION_SECONDS:
                return finish("ABORTED_BUDGET",
                              error=f"only {limit}s of allocation remain, "
                                    f"below the {MIN_REACQUISITION_SECONDS}s a "
                                    f"pod needs to pull, fetch and pass its "
                                    f"gates; refusing to pay for a pod that "
                                    f"cannot do science",
                              acquisitions_used=attempt - 1)
            env_values = dict(identity_env_values(config))
            env_values.update({
                "O1_ACQUIRED_PROFILE": quote["profile"],
                "O1_SESSION_AUTHORIZED_SECONDS": str(limit),
                "O1_HOURLY_RATE_USD": quote["total_projected_hourly_usd"],
            })
            if os.environ.get("HF_TOKEN"):
                env_values["HF_TOKEN"] = os.environ["HF_TOKEN"]
                register_env_secrets()
            req = build_pod_request(
                profile=profile,
                image_digest_ref=config["image_digest_ref"],
                datacenter_id=quote["datacenter_id"],
                env_values=env_values)
            # the pod may not outlive its own paid allowance plus a
            # margin for startup and termination
            controller.monitor_timeout = limit + 1800
            # 4. provision (reconciled, duplicate-safe) + watchdogs
            pod_id = controller.provision(req, rendered)
            # 5. run
            startup = controller.wait_until_running(pod_id)
            if startup == "EVICTED_TERMINATION_UNCONFIRMED":
                return finish("ABORTED_TERMINATION_UNCONFIRMED",
                              termination_confirmed=False,
                              acquisitions_used=attempt,
                              error="the evicted pod's termination could "
                                    "not be confirmed; refusing to acquire "
                                    "another pod on top of a possible remnant")
            if startup == "EVICTED":
                step("EVICTED_DURING_STARTUP", f"attempt {attempt}")
                if zero_progress_abort():
                    return no_progress_abort()
                continue
            outcome = controller.monitor(pod_id, until=pod_done)
            step("MONITOR_RESULT", f"attempt {attempt}: {outcome}")
            if outcome == "EVICTED_TERMINATION_UNCONFIRMED":
                if status["acquisitions"]:
                    status["acquisitions"][-1]["outcome"] = outcome
                return finish("ABORTED_TERMINATION_UNCONFIRMED",
                              termination_confirmed=False,
                              acquisitions_used=attempt,
                              error="the evicted pod's termination could "
                                    "not be confirmed; refusing to acquire "
                                    "another pod on top of a possible remnant")
            if outcome == "EVICTED":
                if status["acquisitions"]:
                    status["acquisitions"][-1]["outcome"] = "EVICTED"
                if zero_progress_abort():
                    return no_progress_abort()
                continue
            if outcome != "COMPLETE":
                # SOFT_STOP (budget) and TERMINATED (external/watchdog) are
                # NOT completions: the pod never emitted its completion
                # witness, so no COMPLETE verdict may be written.  Collect
                # diagnostics, terminate, and report the real outcome.
                controller.collect_logs(pod_id)
                confirmed = controller.terminate_and_confirm(pod_id)
                return finish(
                    f"ABORTED_{outcome}",
                    termination_confirmed=confirmed,
                    acquisitions_used=attempt,
                    error=f"monitor returned {outcome} without the pod's "
                          f"completion witness; no results are claimed")
            # 6/7. results: collect logs + download outputs, then verify the
            # downloaded archive against the digest the POD recorded — a
            # fixed result path with no cross-check would happily "complete"
            # against a stale archive from an earlier attempt
            controller.collect_logs(pod_id)
            dest = os.path.join(out_dir, "downloaded_results")
            # load_session_config REFUSES a stored result_source that
            # disagrees with the derived location, so honouring an explicit
            # value here cannot reintroduce the divergence; it just keeps a
            # directly-constructed config (tests, rehearsals) working.
            got = adapter.download_results(
                config.get("result_source") or result_archive_uri(config)
                or dest, dest)
            step("RESULTS_DOWNLOADED", dest)
            witness_state, expected = _expected_result_digest(
                config, auth.launch_nonce)
            if witness_state == "VERIFIABLE" and got.get("sha256") != expected:
                confirmed = controller.terminate_and_confirm(pod_id)
                return finish(
                    "ABORTED_RESULT_DIGEST_MISMATCH",
                    termination_confirmed=confirmed,
                    acquisitions_used=attempt,
                    error=f"downloaded archive digest "
                          f"{str(got.get('sha256'))[:16]} != the digest the "
                          f"pod recorded {str(expected)[:16]}; refusing to "
                          f"claim these results")
            status["result_sha256"] = got.get("sha256")
            status["result_witness"] = witness_state
            status["result_digest_verified"] = witness_state == "VERIFIABLE"
            if witness_state == "FOREIGN":
                confirmed = controller.terminate_and_confirm(pod_id)
                return finish(
                    "ABORTED_FOREIGN_RESULT_WITNESS",
                    termination_confirmed=confirmed,
                    acquisitions_used=attempt,
                    error=f"the published result sidecar belongs to launch "
                          f"nonce {str(expected)[:12]!r}, not this session; "
                          f"refusing to claim another run's results")
            if witness_state == "ERROR":
                step("RESULT_WITNESS_UNAVAILABLE", expected)
        except DeterministicPodFailure as exc:
            step("DETERMINISTIC_POD_FAILURE", exc)
            controller.collect_logs(pod_id)
            confirmed = controller.terminate_and_confirm(pod_id)
            # A9: durable, keyed to THIS deployment + image: a rerun must
            # not pay to repeat a failure that repeats by construction
            _record_deterministic_refusal(authorization_path, config, str(exc))
            return finish("ABORTED_DETERMINISTIC_POD_FAILURE",
                          termination_confirmed=confirmed,
                          acquisitions_used=attempt, error=redact(str(exc)))
        except WitnessShapeUnrecognised as exc:
            # the pod printed a verdict the driver cannot read: stop after
            # ONE pod, with the log collected, rather than reacquire against
            # a decoder defect
            step("WITNESS_SHAPE_UNRECOGNISED", exc)
            controller.collect_logs(pod_id)
            confirmed = controller.terminate_and_confirm(pod_id)
            return finish("ABORTED_WITNESS_SHAPE_UNRECOGNISED",
                          termination_confirmed=confirmed,
                          acquisitions_used=attempt, error=redact(str(exc)))
        except BudgetViolation as exc:
            step("BUDGET_STOP", exc)
            confirmed = True
            if pod_id is not None:
                confirmed = controller.terminate_and_confirm(pod_id)
            return finish("ABORTED_BUDGET", error=redact(str(exc)),
                          termination_confirmed=confirmed,
                          acquisitions_used=attempt)
        except AuthorizationError as exc:
            # creation-slot exhaustion or expiry mid-session
            step("AUTHORIZATION_STOP", exc)
            confirmed = True
            if pod_id is not None:
                confirmed = controller.terminate_and_confirm(pod_id)
            return finish("ABORTED_AUTHORIZATION_EXHAUSTED",
                          error=redact(str(exc)),
                          termination_confirmed=confirmed,
                          acquisitions_used=attempt)
        except (LifecycleError, Exception) as exc:  # noqa: BLE001
            step("SESSION_FAILURE", exc)
            if pod_id is not None:
                confirmed = controller.terminate_and_confirm(pod_id)
                return finish("ABORTED_TERMINATED" if confirmed
                              else "ABORTED_TERMINATION_UNCONFIRMED",
                              error=redact(str(exc)))
            # pod_id is None, but that does NOT prove nothing is billing:
            # an ambiguous create, a refused duplicate, or a lost read-back
            # all land here with a pod potentially alive on the provider.
            # Saying "before create" while something bills is the single
            # most misleading thing this file can tell an operator at 3am.
            # A just-created pod may be invisible for a while on both
            # surfaces; list repeatedly before concluding, and TERMINATE
            # whatever is found (the authorization's terminate stays valid
            # past expiry).  A consumed nonce slot proves a create was
            # attempted, so "termination_confirmed" is never claimed then.
            leftovers: list = []
            listing_failed = False
            for _probe in range(6):
                try:
                    leftovers = [p.id for p in adapter.list_owned_instances()
                                 if p.status not in ("TERMINATED",)]
                    listing_failed = False
                except Exception:  # noqa: BLE001 - best effort, never masks exc
                    listing_failed = True
                if leftovers:
                    break
                sleep(20.0)
            create_attempted = bool(
                getattr(auth, "nonce_slots_consumed", lambda: 0)())
            if leftovers:
                step("UNTERMINATED_PODS_PRESENT", ",".join(map(str, leftovers)))
                confirmed_all = True
                for lid in leftovers:
                    try:
                        ok = controller.terminate_and_confirm(lid)
                    except Exception:  # noqa: BLE001
                        ok = False
                    step("LEFTOVER_TERMINATION", f"{lid}: {'confirmed' if ok else 'UNCONFIRMED'}")
                    confirmed_all = confirmed_all and ok
                return finish(
                    "ABORTED_NO_POD_RECORDED_BUT_PODS_PRESENT",
                    error=redact(str(exc)),
                    termination_confirmed=confirmed_all,
                    unterminated_pods=[] if confirmed_all else leftovers)
            if listing_failed or create_attempted:
                return finish(
                    "ABORTED_CREATE_OUTCOME_UNKNOWN",
                    error=redact(str(exc)),
                    termination_confirmed=False,
                    note=("a create may have been attempted (nonce slot "
                          "consumed) or the owned-pod listing failed; an "
                          "operator must check the provider console before "
                          "any further acquisition"))
            return finish("ABORTED_BEFORE_CREATE", error=redact(str(exc)),
                          termination_confirmed=True)
        # 8. terminate + confirm (always; stop is never final)
        confirmed = controller.terminate_and_confirm(pod_id)
        if not confirmed:
            outcome = "TERMINATION_UNCONFIRMED"
        elif status.get("result_witness") == "ERROR":
            # the results may well be correct, but nothing could confirm
            # them: that must not read as an ordinary verified COMPLETE
            outcome = "COMPLETE_RESULT_DIGEST_UNVERIFIED"
        else:
            outcome = "COMPLETE"
        return finish(outcome, termination_confirmed=confirmed,
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
