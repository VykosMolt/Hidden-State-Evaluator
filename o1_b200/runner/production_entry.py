#!/usr/bin/env python3
"""Production on-pod entry: the REAL zero-touch session composition.

This replaces the historical refusing stub: it composes the sealed state
machine with REAL components — hardware gate, real Ouro-RLTT artifact,
real non-O1 equivalence and benchmark, frozen backend selection, the
replacement precommit, external (off-pod) commit verification, and the
sealed v2.1 calibration ORCHESTRATOR (never a re-implementation) — with
continuous off-pod durability so eviction of the interruptible pod loses
at most the un-mirrored tail of work.

Every scientific decision stays where it always was: sealed package
semantics, frozen selection policy, frozen benchmark order, frozen budget
policy.  This module only wires validated parts together on the pod.

Configuration arrives via the frozen env-name contract (pod_request):

  O1_B200_OUT                 output dir (default /outputs)
  O1_B200_ARTIFACT_SOURCE     artifact staging source (verify_artifacts)
  O1_B200_RESULT_DESTINATION  durable store destination (hf://… or path)
  O1_ACQUIRED_PROFILE         B300 | B200 (the profile actually rented)
  O1_SESSION_AUTHORIZED_SECONDS   hard runtime limit from the live quote
  O1_HOURLY_RATE_USD          accepted all-in hourly rate (report only)

Fails closed on any absent/unresolved required value.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time

from .budget import BudgetWatchdog, affordability_gate
from .compare_o1_backends import compare_rows, run_backend
from .durability import CheckpointDurability, store_for_destination
from .env_report import collect_pod_report, validate_b200_report
from .persistence import atomic_write_text
from .precommit_template import finalize, load_template, resolve
from .provider_adapter import LocalProviderAdapter
from .runbuild import O1_MANIFEST_PATHS
from .benchmark_o1_b200 import _oom_types, load_benchmark_order
from .identity import domain_sha256
from .selection import benchmark_candidates, derive_gates, select_backend

from .state_machine import ZeroTouchStateMachine
from .validation_corpus import disjointness_report, load_corpus
from . import sealed_import

#: BENCHMARK_ORDER amendment 2: the pre-calibration phase (equivalence +
#: benchmark) may use at most this fraction of the runtime remaining when
#: it starts; stages run in the frozen order until the budget is spent.
PRECALIBRATION_BUDGET_FRACTION = 0.25

ROOT = sealed_import.WORKTREE_ROOT
CHECKPOINT_DIR = os.environ.get("O1_CHECKPOINT_DIR",
                                "/artifacts/ouro_rltt_local")
O1_ROWS_TOTAL = 4608
RECORDS_SYNC_EVERY_ROWS = 25
RECORDS_SYNC_POLL_SECONDS = 15.0


class ProductionEntryError(RuntimeError):
    pass


def _require_env(name: str) -> str:
    v = os.environ.get(name, "")
    if not v or v.startswith("UNRESOLVED"):
        raise ProductionEntryError(
            f"required session env {name} is absent/unresolved; the "
            f"authorized launcher must inject it")
    return v


def build_production_handlers(out_dir: str, provider: LocalProviderAdapter,
                              clock) -> dict:
    profile_key = _require_env("O1_ACQUIRED_PROFILE")
    authorized_seconds = int(float(_require_env(
        "O1_SESSION_AUTHORIZED_SECONDS")))
    result_destination = _require_env("O1_B200_RESULT_DESTINATION")
    corpus_dir = os.path.join(ROOT, "o1_b200", "corpus")
    artifact = {"kind": "ouro_rltt", "checkpoint": CHECKPOINT_DIR,
                "device": "cuda"}
    store = store_for_destination(result_destination)
    records_mirror = CheckpointDurability(
        store, remote_prefix="durable_o1_records")

    def precheck(ctx):
        ctx["instance_ref"] = provider.start_instance()
        ctx["hourly_rate"] = float(os.environ.get("O1_HOURLY_RATE_USD", "0"))
        ctx["runtime_limit"] = authorized_seconds
        ctx["profile"] = profile_key
        return {"instance": ctx["instance_ref"], "profile": profile_key,
                "runtime_limit_seconds": authorized_seconds}

    def artifact_verify(ctx):
        # O1_B200_ARTIFACT_SOURCE is deployment identity (it is part of the
        # authorization hash), so it must actually be CONSUMED — a declared
        # identity field with no reader reads as protection that is not
        # there.  When the operator stages artifacts out of band it names
        # the manifest that describes them; otherwise the image-baked
        # manifest is used.
        staged = os.environ.get("O1_B200_ARTIFACT_SOURCE", "").strip()
        # the POD manifest: container paths, and the only one whose
        # entries can resolve inside the image
        baked = os.path.join(ROOT, "o1_b200", "deploy",
                             "POD_TRANSFER_MANIFEST.json")
        default_manifest = baked
        if staged.startswith("hf://"):
            # The production shape: an hf:// INGESTION URI, already consumed
            # by runner/fetch_artifacts.py before this process started.
            # It is consumed here by asserting the fetch report names the
            # same source; the baked POD manifest then verifies the bytes.
            # (Treating it as a manifest locator refused on every real pod,
            # after the pod and the checkpoint fetch were paid for.)
            fetch_report = os.path.join(out_dir, "ARTIFACT_FETCH_REPORT.json")
            try:
                with open(fetch_report, encoding="utf-8") as fh:
                    fetched = json.load(fh)
            except (OSError, ValueError) as exc:
                raise ProductionEntryError(
                    f"O1_B200_ARTIFACT_SOURCE={staged!r} is an hf:// source "
                    f"but no artifact fetch report exists at {fetch_report} "
                    f"({exc!r}); the entrypoint did not ingest it") from None
            fetched_repo = str(fetched.get("repo") or "")
            from .fetch_artifacts import parse_hf_source as _parse_source
            expected_repo = _parse_source(staged)
            if fetched_repo and fetched_repo != expected_repo:
                raise ProductionEntryError(
                    f"the artifact fetch report names repo {fetched_repo!r} "
                    f"but the identity-bound source is {staged!r}")
            ctx["artifact_source_consumed"] = {
                "source": staged, "fetch_report": fetch_report,
                "fetched": fetched.get("fetched"),
                "already_present": fetched.get("already_present")}
        elif staged:
            candidate = (staged if staged.endswith(".json")
                         else os.path.join(staged, "TRANSFER_MANIFEST.json"))
            if not os.path.exists(candidate):
                raise ProductionEntryError(
                    f"O1_B200_ARTIFACT_SOURCE={staged!r} names no readable "
                    f"transfer manifest ({candidate}); refusing to fall "
                    f"back silently to the image-baked manifest")
            default_manifest = candidate
        manifest = os.environ.get("O1_B200_TRANSFER_MANIFEST",
                                  default_manifest)
        ctx["artifact_manifest"] = manifest
        proc = subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "o1_b200", "deploy", "verify_artifacts.py"),
             "--manifest", manifest],
            capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise ProductionEntryError(
                f"artifact verification failed:\n{proc.stdout[-1500:]}"
                f"{proc.stderr[-1500:]}")
        # Bind the checkpoint's tree hash into the model artifact NOW.
        # backends.model_artifact_sha256() requires it for the ouro_rltt
        # kind and nothing ever supplied it, so the first backend call
        # raised KeyError at NON_O1_EQUIVALENCE -- on every pod, after the
        # 5 GB fetch and the hardware gate were already paid for.  The
        # value is taken from the manifest that verify_artifacts.py has
        # just re-hashed against the on-disk tree, which is exactly the
        # "precomputed, re-checked on the target machine" contract.
        artifact["checkpoint_tree_sha256"] = _manifest_checkpoint_sha256(
            manifest, CHECKPOINT_DIR)
        ctx["checkpoint_tree_sha256"] = artifact["checkpoint_tree_sha256"]
        tasks, config = load_corpus(corpus_dir)
        rep = disjointness_report(tasks, O1_MANIFEST_PATHS)
        if rep["verdict"] != "DISJOINT":
            raise ProductionEntryError("corpus not disjoint from O1 pools")
        ctx["corpus_config"] = config
        # load_corpus verifies the corpus against its sealed hash and the
        # disjointness report is DISJOINT, so the scientific inputs are
        # provably the frozen ones (fed to the selection gates)
        ctx["corpus_config_verified"] = True
        return {"artifacts_verified": True, "disjointness": rep["verdict"],
                "artifact_source_consumed": ctx.get("artifact_source_consumed")}

    def environment_verify(ctx):
        from o1_b200.deploy.hardware_gate import (
            exercise_workloads, gather_facts, validate_facts,
        )
        gate = validate_facts(profile_key, gather_facts())
        if not gate["accepted"]:
            raise ProductionEntryError(
                "hardware gate refused this machine: "
                + "; ".join(gate["problems"]))
        gate["workload_exercises"] = exercise_workloads(CHECKPOINT_DIR,
                                                        out_dir)
        if not gate["workload_exercises"]["accepted"]:
            bad = {k: v for k, v in
                   gate["workload_exercises"]["workloads"].items()
                   if not v.get("ok")}
            raise ProductionEntryError(
                f"representative workload exercises failed: {sorted(bad)}")
        atomic_write_text(
            os.path.join(out_dir, "HARDWARE_GATE_REPORT.json"),
            json.dumps(gate, indent=2, sort_keys=True, default=str) + "\n")
        env_report = collect_pod_report(
            container_image_digest=os.environ.get("O1_IMAGE_DIGEST",
                                                  "UNKNOWN"),
            observed=gate["workload_exercises"]["workloads"].get(
                "ouro_rltt_load"))
        validated = validate_b200_report(env_report)
        atomic_write_text(
            os.path.join(out_dir, "ENVIRONMENT_REPORT.resolved.json"),
            json.dumps({"report": env_report, "validation": validated},
                       indent=2, sort_keys=True, default=str) + "\n")
        ctx["environment_report"] = validated
        ctx["environment_report_raw"] = env_report
        ctx["hardware_gate"] = gate
        return {"hardware_gate": "ACCEPTED", "profile": profile_key}

    def non_o1_equivalence(ctx):
        """Measure structural equivalence PER BENCHMARK CONFIGURATION.

        Selection ranks by throughput, so it prefers the deepest batch that
        ran — exactly the configuration whose batched numerics are least
        like the serial reference.  Borrowing one backend's verdict for
        every batch size of that backend would let a configuration be
        selected on an equivalence result never measured for it, so each
        config_id carries its own verdict and any config without one is
        ineligible (fail closed in _derive_gates).
        """
        from .benchmark_o1_b200 import load_benchmark_order
        stages = load_benchmark_order()["staged_candidates"]
        # A replacement pod after eviction must not pay for the whole
        # pre-calibration phase again: the previous pod's equivalence +
        # benchmark + selection are mirrored durably and restored when the
        # image digest and profile match (the measurements describe the
        # same software on the same accelerator class).
        restored = _restore_precalibration(ctx)
        if restored:
            return {"restored_from_durable_store": True,
                    **{cid: v.get("eligible_structurally")
                       for cid, v in sorted(ctx["equivalence"].items())}}
        phase_t0 = clock()
        remaining_at_start = ctx["runtime_limit"] - phase_t0
        ref = run_backend("REFERENCE_SERIAL", corpus_dir,
                          os.path.join(out_dir, "eq_ref"), artifact)
        ref_seconds = max(1.0, clock() - phase_t0)
        # BENCHMARK_ORDER amendment 2: the pre-calibration phase (equivalence
        # + benchmark, both over the full validation corpus per stage) is
        # bounded to a predeclared fraction of the remaining runtime, shared
        # by both phases.  Stages run in the frozen order until the budget
        # is exhausted; the reference always runs.  Without this the phase
        # was ~55% of the calibration workload, unbounded, and only checked
        # for affordability after it had been paid for.
        # BOTH reference passes (this one, and the benchmark's reference
        # stage) are charged to the budget up front: they are the slowest
        # configuration and were previously uncounted, so the phase could
        # exceed the declared bound by 2 x T_ref.
        budget = PRECALIBRATION_BUDGET_FRACTION * remaining_at_start \
            - 2.0 * ref_seconds
        if budget < 0:
            # the phase cannot fit even its two mandatory passes: say so now,
            # not after the benchmark's reference pass has also been paid
            ctx["precalibration_overrun"] = {
                "reference_pass_seconds": ref_seconds,
                "declared_budget_seconds":
                    PRECALIBRATION_BUDGET_FRACTION * remaining_at_start}
            budget = 0.0
        ctx["precalibration"] = {
            "remaining_at_start_seconds": remaining_at_start,
            "budget_seconds": budget,
            "reference_pass_seconds": ref_seconds,
            "equivalence_spent_seconds": 0.0,
        }
        comp = {"REFERENCE_SERIAL_w1_b1": {
            "eligible_structurally": True, "scientific_core_identical": True,
            "is_reference": True}}
        clean_so_far = True
        spent = 0.0
        estimate = ref_seconds
        oom_types = _oom_types()
        for entry in stages:
            cid = entry["config_id"]
            if cid in comp:
                continue
            conditional = bool(entry.get("conditional"))
            if conditional and not clean_so_far:
                # frozen stop rule: once a stage is not clean, larger
                # configurations are not attempted
                comp[cid] = {"eligible_structurally": False,
                             "skipped": "prior stage not clean"}
                continue
            # the benchmark still needs its own pass per stage, so an
            # equivalence pass may use at most half of what is left
            if estimate > 0.5 * (budget - spent):
                comp[cid] = {"eligible_structurally": False,
                             "skipped": "precalibration cost rule",
                             "estimate_seconds": estimate,
                             "budget_left_seconds": budget - spent}
                continue
            t_stage = clock()
            try:
                cand = run_backend(
                    entry["backend"], corpus_dir,
                    os.path.join(out_dir, f"eq_{cid}"), artifact,
                    worker_count=int(entry.get("workers", 1)),
                    batch_size=int(entry.get("batch", 1)))
            except oom_types as exc:
                # ANY non-reference stage that OOMs is ineligible, never an
                # abort: the reference is the terminal fallback and the
                # calibration does not use the selected backend's code path
                clean_so_far = False
                comp[cid] = {"eligible_structurally": False,
                             "skipped": f"OOM: {exc!r}"[:200]}
                spent += clock() - t_stage
                continue
            except Exception as exc:  # noqa: BLE001
                clean_so_far = False
                comp[cid] = {"eligible_structurally": False,
                             "skipped": f"integrity failure: {exc!r}"[:200]}
                spent += clock() - t_stage
                continue
            stage_seconds = clock() - t_stage
            spent += stage_seconds
            estimate = max(estimate, stage_seconds)
            comp[cid] = compare_rows(ref["rows"], cand["rows"])
            comp[cid]["stage_seconds"] = stage_seconds
            if not comp[cid].get("eligible_structurally"):
                clean_so_far = False
        ctx["precalibration"]["equivalence_spent_seconds"] = spent
        ctx["precalibration"]["stage_cost_estimate_seconds"] = estimate
        # the reference defines structural eligibility by construction; a
        # failed non-reference stage is recorded as ineligible, never raised
        ctx["equivalence"] = comp
        # the reference run defines the expected row count every benchmark
        # configuration must reproduce exactly
        ctx["corpus_row_count"] = len(ref["rows"])
        atomic_write_text(
            os.path.join(out_dir, "EQUIVALENCE_REPORT.real.json"),
            json.dumps(comp, indent=2, sort_keys=True, default=str) + "\n")
        ran = [cid for cid, v in comp.items()
               if not v.get("is_reference") and not str(
                   v.get("skipped", "")).startswith("precalibration cost rule")
               and v.get("skipped") != "prior stage not clean"]
        return {**{cid: v.get("eligible_structurally")
                   for cid, v in sorted(comp.items())},
                "precalibration_overrun": ctx.get("precalibration_overrun"),
                "no_stage_affordable": (budget > 0 and not ran)}

    def non_o1_benchmark(ctx):
        if ctx.get("precalibration_restored"):
            return {"benchmarked": len(ctx["benchmark_raw"]),
                    "restored_from_durable_store": True}
        from .benchmark_o1_b200 import run_benchmarks
        pre = ctx["precalibration"]
        rep = run_benchmarks(
            corpus_dir, os.path.join(out_dir, "benchmark"),
            mode="real-hardware", artifact=artifact,
            remaining_authorized_seconds=ctx["runtime_limit"] - clock(),
            stage_budget_seconds=max(
                0.0, pre["budget_seconds"] - pre["equivalence_spent_seconds"]),
            stage_cost_estimate_seconds=pre.get(
                "stage_cost_estimate_seconds", pre["reference_pass_seconds"]))
        ctx["benchmark_raw"] = [r for r in rep["results"]
                                if not r.get("skipped")]
        atomic_write_text(
            os.path.join(out_dir, "BENCHMARK_REPORT.real.json"),
            json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n")
        ctx["benchmark_sha256"] = __import__("hashlib").sha256(
            open(os.path.join(out_dir, "BENCHMARK_REPORT.real.json"),
                 "rb").read()).hexdigest()
        return {"benchmarked": len(ctx["benchmark_raw"])}

    def _derive_gates(entry: dict, ctx) -> dict:
        import torch
        return derive_gates(
            entry,
            equivalence=(ctx.get("equivalence") or {}).get(entry.get("config_id")),
            device_total_memory=torch.cuda.get_device_properties(0).total_memory,
            expected_rows=int(ctx.get("corpus_row_count") or 0),
            environment=ctx.get("environment_report_raw") or {},
            corpus_config_verified=bool(ctx.get("corpus_config_verified")))

    def backend_select(ctx):
        candidates = [_derive_gates(e, ctx)
                      for e in benchmark_candidates(ctx["benchmark_raw"])]
        ctx["benchmark"] = candidates
        out = select_backend(candidates)
        ctx["selected_backend"] = out["selected"]
        atomic_write_text(
            os.path.join(out_dir, "BACKEND_SELECTION.json"),
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
        if not ctx.get("precalibration_restored"):
            _persist_precalibration(ctx)
        return {"selected": out["selected"]["config_id"],
                "terminal_fallback_waiver": out.get("terminal_fallback_waiver")}

    PRECAL_REMOTE_REL = "durable_o1_precalibration/PRECALIBRATION_RESULT.json"

    def _precal_identity(ctx) -> dict:
        return {"image_digest": os.environ.get("O1_IMAGE_DIGEST", "UNKNOWN"),
                "profile": profile_key,
                "benchmark_order_sha256": domain_sha256(
                    "o1b200.benchmark_order.v1", load_benchmark_order()),
                # the measurements are OF these artifacts: a restore must
                # never carry verdicts measured against a different
                # checkpoint or corpus
                "checkpoint_tree_sha256": ctx.get("checkpoint_tree_sha256"),
                "corpus_config_sha256": domain_sha256(
                    "o1b200.corpus_config", ctx.get("corpus_config") or {}),
                # the deployment this phase was measured for.  These are
                # deployment identity (stable across launches of the same
                # config): a later authorization against the same config
                # MAY restore, and the precommit then discloses it as
                # RESTORED_FROM_<pod>@<utc>
                "result_destination": result_destination,
                "artifact_source": os.environ.get("O1_B200_ARTIFACT_SOURCE",
                                                  "")}

    def _persist_precalibration(ctx) -> None:
        payload = {
            "schema": "o1b300.precalibration_result.v1",
            "identity": _precal_identity(ctx),
            "equivalence": ctx["equivalence"],
            "corpus_row_count": ctx["corpus_row_count"],
            "benchmark_report_text": open(
                os.path.join(out_dir, "BENCHMARK_REPORT.real.json"),
                encoding="utf-8").read(),
            "precalibration": ctx.get("precalibration"),
            "measured_on": {"instance_id": os.environ.get("RUNPOD_POD_ID",
                                                          "POD"),
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime())},
        }
        local = os.path.join(out_dir, "PRECALIBRATION_RESULT.json")
        atomic_write_text(local, json.dumps(payload, sort_keys=True,
                                            default=str) + "\n")
        try:
            store.push_file(local, PRECAL_REMOTE_REL)
        except Exception as exc:  # noqa: BLE001 - advisory: a replacement pod re-measures
            atomic_write_text(os.path.join(out_dir, "PRECALIBRATION_PUSH_FAILED.txt"),
                              repr(exc)[:500] + "\n")

    def _restore_precalibration(ctx) -> bool:
        local = os.path.join(out_dir, "PRECALIBRATION_RESULT.restored.json")
        try:
            store.fetch_file(PRECAL_REMOTE_REL, local)
            with open(local, encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - absent or unreadable: measure
            atomic_write_text(
                os.path.join(out_dir, "PRECALIBRATION_RESTORE_SKIPPED.txt"),
                f"no durable pre-calibration result restored: {exc!r}\n")
            return False
        if payload.get("schema") != "o1b300.precalibration_result.v1":
            return False
        digest = os.environ.get("O1_IMAGE_DIGEST", "").strip()
        if not digest or digest.upper() == "UNKNOWN":
            # two different images lacking the digest would share an
            # identity; never restore unbound
            return False
        if payload.get("identity") != _precal_identity(ctx):
            # a different image, accelerator class, checkpoint, corpus or
            # session: measurements do not transfer; measure again
            return False
        ctx["equivalence"] = payload["equivalence"]
        ctx["corpus_row_count"] = int(payload["corpus_row_count"])
        report_path = os.path.join(out_dir, "BENCHMARK_REPORT.real.json")
        atomic_write_text(report_path, payload["benchmark_report_text"])
        rep = json.loads(payload["benchmark_report_text"])
        ctx["benchmark_raw"] = [r for r in rep["results"]
                                if not r.get("skipped")]
        ctx["benchmark_sha256"] = __import__("hashlib").sha256(
            open(report_path, "rb").read()).hexdigest()
        ctx["precalibration"] = payload.get("precalibration")
        ctx["precalibration_restored"] = True
        ctx["precalibration_source"] = {
            "restored": True, **(payload.get("measured_on") or {})}
        atomic_write_text(
            os.path.join(out_dir, "EQUIVALENCE_REPORT.real.json"),
            json.dumps(ctx["equivalence"], indent=2, sort_keys=True,
                       default=str) + "\n")
        return True


    def precommit_build(ctx):
        import torch
        template = load_template()
        gate_sha = __import__("hashlib").sha256(json.dumps(
            ctx["hardware_gate"], sort_keys=True, default=str
        ).encode()).hexdigest()
        resolved = resolve(template, {
            "provider": "runpod",
            "instance_id": os.environ.get("RUNPOD_POD_ID", "POD"),
            "gpu_uuid": torch.cuda.get_device_properties(0).uuid.__str__(),
            "driver_runtime_report_sha256": gate_sha,
            "container_image_digest_as_deployed": os.environ.get(
                "O1_IMAGE_DIGEST", "UNKNOWN"),
            "selected_eligible_backend": ctx["selected_backend"]["backend"],
            "selected_worker_batch_configuration":
                ctx["selected_backend"]["config_id"],
            "measured_benchmark_throughput_rows_per_hour":
                ctx["selected_backend"]["completed_rows_per_hour"],
            "actual_hourly_rate_usd": ctx["hourly_rate"],
            "computed_hard_runtime_seconds": ctx["runtime_limit"],
            # validate_b200_report returns {"valid", "environment_digest_sha256"}
            # -- there is no "report_sha256", so this defaulted to 64 zeros:
            # the deployed pre-registration recorded a placeholder that is
            # byte-identical to the dress rehearsal's mock, and no audit
            # could tell a real environment from a fabricated one.
            "environment_digest_sha256":
                ctx["environment_report"]["environment_digest_sha256"],
            # the BENCHMARK it names, not a second copy of the gate hash
            "final_backend_benchmark_report_sha256": ctx["benchmark_sha256"],
            # a restored pre-calibration mixes two pods in one document
            # (instance/gpu from THIS pod, benchmark from an earlier one);
            # the document must say so
            "benchmark_provenance": (
                "MEASURED_ON_THIS_POD" if not ctx.get("precalibration_restored")
                else "RESTORED_FROM_{}@{}".format(
                    (ctx.get("precalibration_source") or {}).get(
                        "instance_id", "?"),
                    (ctx.get("precalibration_source") or {}).get("utc", "?"))),
        }, mock=False)
        ctx["finalized_precommit"] = finalize(resolved)
        path = os.path.join(out_dir, "CALIBRATION_PRECOMMIT.deployed.json")
        atomic_write_text(path, json.dumps(
            ctx["finalized_precommit"], indent=2, sort_keys=True) + "\n")
        ctx["precommit_path"] = path
        ctx["calibration_binding_sha256"] = \
            ctx["finalized_precommit"]["document_sha256"]
        return {"finalized": True,
                "document_sha256": ctx["calibration_binding_sha256"]}

    def external_commit_verify(ctx):
        # Off-pod commitment: push the finalized hardware precommit, re-fetch
        # it independently, and verify byte identity.  The pod deliberately
        # has no git credentials; the durable store is the external witness.
        #
        # The key is per-acquisition: this document records THIS pod's
        # hardware facts, and a fixed key would silently replace an earlier
        # pod's commitment after partial results exist, destroying the
        # chain of pre-registrations across an evicted session.
        from .identity import sha256_file
        digest = sha256_file(ctx["precommit_path"])
        key = f"commitments/hardware/{digest[:16]}.json"
        existing = set(store.list_prefix("commitments/hardware"))
        pushed = store.push_file(ctx["precommit_path"], key)
        back = os.path.join(out_dir, "precommit_refetch.json")
        store.fetch_file(key, back)
        if sha256_file(back) != digest:
            raise ProductionEntryError(
                "external commitment verification failed: re-fetched "
                "precommit differs")
        os.remove(back)
        return {"remote_verified": True, "sha256": pushed["sha256"],
                "key": key, "prior_commitments": len(existing)}

    def affordability(ctx):
        # The sealed v2.1 orchestrator takes NO backend/worker/batch
        # parameter — it builds its own single-row serial backend. So the
        # calibration runs at the REFERENCE_SERIAL rate no matter which
        # configuration the frozen policy selected, and projecting from the
        # selected (fastest, batched) rate would under-estimate the time by
        # the whole batching speed-up and pass a gate that cannot hold.
        eq_ref = "REFERENCE_SERIAL_w1_b1"
        serial = next((c for c in ctx.get("benchmark", [])
                       if c.get("config_id") == eq_ref), None)
        if serial is None:
            raise ProductionEntryError(
                f"no measured {eq_ref} throughput; the affordability gate "
                f"cannot be projected from an unmeasured rate")
        rows_per_hour = float(serial["completed_rows_per_hour"])
        ctx["affordability_rate_basis"] = {
            "config_id": eq_ref, "rows_per_hour": rows_per_hour,
            "why": ("the sealed orchestrator generates serially; the "
                    "selected backend governs the non-O1 benchmark and the "
                    "precommit record, not calibration throughput")}
        # Progress must come from the DURABLE marker, not from local disk.
        # The restore runs inside the CALIBRATION handler, which is a LATER
        # state, and out_dir is fresh container disk on every pod -- so a
        # resumed pod always measured done == 0 here and re-projected the
        # full 4608 rows against a shrunken allowance.  The gate therefore
        # refused every resume, and refused it more certainly the deeper
        # into the run the eviction landed: the exact inverse of what the
        # durability design exists for.  None means "unknown", which is
        # deliberately not zero.
        local_done = _existing_row_count(
            os.path.join(out_dir, "o1_records.jsonl"))
        durable_done = records_mirror.latest_row_count()
        done = max(local_done, durable_done or 0)
        ctx["affordability_rows_done"] = {
            "local": local_done, "durable": durable_done, "used": done}
        projected = (O1_ROWS_TOTAL - done) / max(rows_per_hour, 1e-9) * 3600
        return affordability_gate(
            projected_calibration_seconds=projected,
            verification_transfer_reserve_seconds=1200,
            termination_reserve_seconds=300,
            remaining_authorized_runtime_seconds=(
                ctx["runtime_limit"] - clock()))

    def _build_replacement_manifest(path: str) -> str:
        """Replacement freeze manifest for the deployed runtime.

        Exactly two fields differ from the sealed
        FREEZE_MANIFEST.precalibration.json: ``code.torch`` (which the
        sealed orchestrator runtime-asserts against the live interpreter,
        so it MUST carry the deployed build) and ``code.
        runtime_version_note`` (a provenance string recording that
        substitution).  Every other field — including every artifact hash,
        design constant, seed policy and deterministic flag — is copied
        verbatim, and the diff is asserted below so the claim cannot rot.
        The file is re-serialized, so it is not byte-identical; identity is
        established field-by-field, not by bytes.
        """
        import torch
        run_root = os.path.join(ROOT, "o1_runs", "O1_V2_AXIS_BANK_REDESIGN")
        with open(os.path.join(
                run_root, "FREEZE_MANIFEST.precalibration.json"),
                encoding="utf-8") as fh:
            manifest = json.load(fh)
        sealed_torch = manifest["code"]["torch"]
        manifest["code"]["torch"] = torch.__version__
        manifest["code"]["runtime_version_note"] = (
            manifest["code"].get("runtime_version_note", "") +
            f" | INFRASTRUCTURE MIGRATION: code.torch replaced "
            f"{sealed_torch!r} -> {torch.__version__!r} for the B300/cu130 "
            f"accelerator image; all other fields verbatim from the sealed "
            f"manifest")
        # prove the claim rather than assert it: nothing outside those two
        # fields may differ from the sealed manifest
        with open(os.path.join(
                run_root, "FREEZE_MANIFEST.precalibration.json"),
                encoding="utf-8") as fh:
            original = json.load(fh)
        changed = _diff_paths(original, manifest)
        allowed = {"code.torch", "code.runtime_version_note"}
        if set(changed) - allowed:
            raise ProductionEntryError(
                f"replacement manifest changed fields outside the permitted "
                f"infrastructure substitution: {sorted(set(changed) - allowed)}")
        atomic_write_text(path, json.dumps(manifest, indent=2,
                                           sort_keys=True) + "\n")
        return path

    def _build_pod_artifact_map(path: str) -> str:
        run_root = os.path.join(ROOT, "o1_runs", "O1_V2_AXIS_BANK_REDESIGN")
        sealed = sealed_import.SEALED_DIR
        atomic_write_text(path, json.dumps({
            "artifact_hashes.parser":
                os.path.join(sealed, "o1_answer_parser_v2.py"),
            "artifact_hashes.prompt_template":
                os.path.join(sealed, "o1_prompt_template_v2.py"),
            "artifact_hashes.random_axis_tensor":
                os.path.join(run_root, "AXIS_PACKAGE_V2",
                             "random_axes_l3_24.npy"),
            "artifact_hashes.structured_axis_tensor":
                os.path.join(run_root, "AXIS_PACKAGE_V2", "axes_l3_24.npy"),
            "artifact_hashes.tokenizer":
                os.path.join(run_root, "TOKENIZER_BINDING.json"),
            "artifact_hashes.verifier_implementation":
                os.path.join(sealed, "o1_truth_table_verifier_v2.py"),
            "code.generation_module_sha256":
                os.path.join(sealed, "run_o1_v2_generation.py"),
            "code.orchestrator_module_sha256":
                os.path.join(sealed, "run_o1_v2_orchestrator.py"),
            "code.transport_module_sha256":
                os.path.join(sealed, "o1_transport_v2.py"),
            "model.checkpoint_sha256": CHECKPOINT_DIR,
        }, indent=2, sort_keys=True) + "\n")
        return path

    def calibration(ctx):
        # 1. restore any durable records from an earlier evicted pod, so the
        #    sealed orchestrator resumes at the next missing canonical row
        records_path = os.path.join(out_dir, "o1_records.jsonl")
        progress_path = os.path.join(out_dir, "o1_progress.json")
        prior = records_mirror.restore_latest(os.path.join(
            out_dir, "durable_restore"))
        if prior is not None and not os.path.exists(records_path):
            shutil.copyfile(prior["archive"], records_path)
        # restore the wall-clock progress sidecar as well: without it every
        # pod restarts the accumulator and the final metadata under-reports
        # elapsed session time (provenance, not correctness)
        if not os.path.exists(progress_path):
            try:
                if "durable_progress/PROGRESS.json" in store.list_prefix(
                        "durable_progress"):
                    store.fetch_file("durable_progress/PROGRESS.json",
                                     progress_path)
            except Exception as exc:  # noqa: BLE001 - provenance only
                _log_event(out_dir, "PROGRESS_RESTORE_FAILED",
                           error=str(exc)[:200])
        # The sealed orchestrator re-generates baseline stream 0 on resume
        # and demands BITWISE-identical tokens against the stored row.  A
        # different accelerator architecture (sm_103 vs sm_100) can change
        # those bits, which would poison the durable dataset permanently,
        # so the profile that produced the existing rows is binding.
        profile_key_path = "durable_o1_records/PROFILE.json"
        if _existing_row_count(records_path) > 0:
            try:
                import tempfile as _tf
                if profile_key_path in store.list_prefix("durable_o1_records"):
                    with _tf.TemporaryDirectory() as _t:
                        _p = os.path.join(_t, "PROFILE.json")
                        store.fetch_file(profile_key_path, _p)
                        with open(_p, encoding="utf-8") as fh:
                            bound = json.load(fh).get("profile")
                    if bound and bound != profile_key:
                        raise ProductionEntryError(
                            f"durable rows were generated on {bound} but this "
                            f"pod is {profile_key}; the sealed resume replays "
                            f"baselines bitwise, so a different accelerator "
                            f"architecture would corrupt the dataset. "
                            f"Reacquire {bound}, or start a new session "
                            f"prefix for a clean run.")
            except ProductionEntryError:
                raise
            except Exception as exc:  # noqa: BLE001 - absent marker is fine
                _log_event(out_dir, "PROFILE_BINDING_UNREADABLE",
                           error=str(exc)[:200])
        else:
            marker = os.path.join(out_dir, "PROFILE.json")
            atomic_write_text(marker, json.dumps(
                {"profile": profile_key,
                 "why": "binds the accelerator architecture that produced the "
                        "first rows; the sealed resume replays baselines "
                        "bitwise"}, indent=2, sort_keys=True) + "\n")
            # FATAL, not best-effort.  This marker is the ONLY thing that
            # stops a B300-then-B200 (or reverse) reacquisition mixing two
            # architectures into one sealed dataset: the sealed resume only
            # replays baselines for a PARTIAL task, so an eviction on a task
            # boundary performs no bitwise check at all, and the sealed row
            # schema records no GPU or arch field.  If the marker is not
            # durable, a later pod cannot be refused -- so failing to
            # publish it must stop this pod now, before rows exist, rather
            # than produce an unverifiable dataset later.
            try:
                store.push_file(marker, profile_key_path)
            except Exception as exc:  # noqa: BLE001
                _log_event(out_dir, "PROFILE_BINDING_PUBLISH_FAILED",
                           error=str(exc)[:200])
                raise ProductionEntryError(
                    f"could not publish the accelerator-profile binding "
                    f"({exc}); without it a reacquisition on a different "
                    f"architecture could not be refused, and the sealed "
                    f"dataset would be unverifiable") from None
        run_root = os.path.join(ROOT, "o1_runs", "O1_V2_AXIS_BANK_REDESIGN")
        # 2. replacement manifest (deployed torch) + pod artifact map +
        #    regenerated sealed-format precommit bound to them
        manifest_path = _build_replacement_manifest(
            os.path.join(out_dir, "FREEZE_MANIFEST.deployed.json"))
        artifact_map_path = _build_pod_artifact_map(
            os.path.join(out_dir, "RUNTIME_ARTIFACT_PATHS.pod.json"))
        sealed_precommit_path = os.path.join(
            out_dir, "CALIBRATION_PRECOMMIT.sealed_format.json")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": sealed_import.SEALED_DIR}
        # The sealed record sink binds every row to the precommit's sha256
        # and refuses to mix bindings, so the precommit must be minted ONCE
        # for the whole session and reused by every later pod — regenerating
        # it after an eviction would both break resume and overwrite the
        # session's external pre-registration after partial results exist.
        durable_key = "commitments/CALIBRATION_PRECOMMIT.sealed.json"
        reused = False
        try:
            if durable_key in store.list_prefix("commitments"):
                store.fetch_file(durable_key, sealed_precommit_path)
                reused = True
        except Exception as exc:  # noqa: BLE001 - absent witness = fresh mint
            _log_event(out_dir, "COMMITMENT_FETCH_FAILED", error=str(exc)[:200])
        if not reused:
            regen = subprocess.run(
                [sys.executable,
                 os.path.join(sealed_import.SEALED_DIR,
                              "calibration_precommit.py"),
                 "--output", sealed_precommit_path,
                 "--manifest-design", manifest_path,
                 "--artifact-paths", artifact_map_path,
                 "--calibration-task-manifest",
                 os.path.join(run_root, "COHORTS", "calibration_tasks.jsonl"),
                 # ARM the sealed "no precommit once records exist" guard
                 # rather than bypassing it by omitting the flag
                 "--records", records_path],
                capture_output=True, text=True, timeout=1800, env=env,
                cwd=sealed_import.SEALED_DIR)
            if regen.returncode != 0 or not os.path.exists(
                    sealed_precommit_path):
                raise ProductionEntryError(
                    f"sealed precommit regeneration failed:\n"
                    f"{regen.stdout[-1000:]}{regen.stderr[-1000:]}")
            # publish the one-time external witness (write-once: a later pod
            # reuses it, and the branch above never re-mints over it)
            store.push_file(sealed_precommit_path, durable_key)
        _log_event(out_dir, "SEALED_PRECOMMIT_BOUND", reused=reused)
        # 3. run the sealed v2.1 orchestrator (never re-implemented) as a
        #    subprocess; mirror the records file periodically for durability
        cmd = [
            sys.executable,
            os.path.join(sealed_import.SEALED_DIR,
                         "run_o1_v2_orchestrator.py"),
            "calibration",
            "--manifest-design", manifest_path,
            "--artifact-paths", artifact_map_path,
            "--precommit", sealed_precommit_path,
            "--calibration-task-manifest",
            os.path.join(run_root, "COHORTS", "calibration_tasks.jsonl"),
            "--axis-package", os.path.join(run_root, "AXIS_PACKAGE_V2"),
            "--checkpoint", CHECKPOINT_DIR,
            "--output", records_path,
            "--metadata-output", os.path.join(out_dir, "o1_metadata.json"),
            "--progress", progress_path,
            "--boundary-cache-dir", os.path.join(out_dir, "boundary_cache"),
        ]
        # binary stdout + explicit decoding: text=True decodes strict UTF-8,
        # so one stray byte from a CUDA/NCCL/driver message would raise
        # UnicodeDecodeError inside the drain and strand the child on a full
        # pipe — the exact hang the drain exists to prevent.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                env=env, cwd=sealed_import.SEALED_DIR)
        last_mirrored = _existing_row_count(records_path)
        log_path = os.path.join(out_dir, "o1_orchestrator.log")
        # Drain the child's output on a THREAD: readline() blocks, so
        # driving the mirror cadence from the same loop meant a quiet
        # stretch of the orchestrator stalled mirroring indefinitely and
        # silently widened the eviction-loss window far past the nominal
        # row interval.  The mirror now ticks on wall time regardless.
        drain_state = {"error": None}

        def _drain():
            """Consume the child's stdout UNCONDITIONALLY.

            If logging fails (ENOSPC on the container disk, permissions),
            the thread must keep draining anyway: a dead drain leaves the
            child blocked on a full pipe forever while the poll loop spins,
            burning the entire remaining allocation with nothing bounding
            it but the off-pod watchdog.
            """
            log = None
            try:
                log = open(log_path, "ab")
            except Exception as exc:  # noqa: BLE001
                drain_state["error"] = repr(exc)[:200]
            try:
                for line in proc.stdout:          # bytes; never decoded
                    if log is not None:
                        try:
                            log.write(line)
                            log.flush()
                        except Exception as exc:  # noqa: BLE001
                            drain_state["error"] = repr(exc)[:200]
                            try:
                                log.close()
                            except Exception:  # noqa: BLE001
                                pass
                            log = None      # keep consuming, stop logging
            except Exception as exc:  # noqa: BLE001
                drain_state["error"] = repr(exc)[:200]
            finally:
                if log is not None:
                    try:
                        log.close()
                    except Exception:  # noqa: BLE001
                        pass
        drain = threading.Thread(target=_drain, daemon=True)
        drain.start()
        while proc.poll() is None:
            time.sleep(RECORDS_SYNC_POLL_SECONDS)
            # hard wall-clock bound tied to the authorized runtime: the
            # orchestrator can never outlive the session's paid allowance
            if clock() > ctx["runtime_limit"]:
                proc.kill()
                proc.wait(timeout=120)
                rows_at_kill = _existing_row_count(records_path)
                records_mirror.sync_checkpoint(
                    records_path, {"rows": rows_at_kill,
                                   "progress": "runtime_limit_reached"})
                raise ProductionEntryError(
                    f"authorized runtime ({ctx['runtime_limit']}s) reached "
                    f"with {rows_at_kill} rows committed; orchestrator "
                    f"killed and records mirrored")
            rows_now = _existing_row_count(records_path)
            if rows_now - last_mirrored >= RECORDS_SYNC_EVERY_ROWS:
                records_mirror.sync_checkpoint(
                    records_path, {"rows": rows_now, "progress": "partial"})
                last_mirrored = rows_now
                if os.path.exists(progress_path):
                    try:
                        store.push_file(progress_path,
                                        "durable_progress/PROGRESS.json")
                    except Exception as exc:  # noqa: BLE001 - provenance only
                        _log_event(out_dir, "PROGRESS_SYNC_FAILED",
                                   error=str(exc)[:200])
        drain.join(timeout=60)
        if drain_state["error"]:
            _log_event(out_dir, "ORCHESTRATOR_LOG_CAPTURE_DEGRADED",
                       error=drain_state["error"])
        rows_final = _existing_row_count(records_path)
        records_mirror.sync_checkpoint(records_path,
                                       {"rows": rows_final,
                                        "progress": "final"
                                        if proc.returncode == 0
                                        else "interrupted"})
        if proc.returncode != 0:
            raise ProductionEntryError(
                f"sealed orchestrator exited {proc.returncode}; records "
                f"mirrored at {rows_final} rows (see o1_orchestrator.log)")
        ctx["calibration_records_path"] = records_path
        return {"rows": rows_final}

    def record_verify(ctx):
        # the sealed orchestrator already verifies each row (parser,
        # truth-table verifier, token/text binding); here we recount and
        # hash the final artifact for the transfer manifest
        from .identity import sha256_file
        path = ctx["calibration_records_path"]
        n = _existing_row_count(path)
        if n != O1_ROWS_TOTAL:
            raise ProductionEntryError(
                f"records carry {n} rows, expected {O1_ROWS_TOTAL}")
        ctx["records_sha256"] = sha256_file(path)
        return {"verified_rows": n, "records_sha256": ctx["records_sha256"]}

    def result_transfer(ctx):
        from .identity import sha256_file
        archive = os.path.join(out_dir, "o1_results.tar.gz")
        import tarfile
        with tarfile.open(archive, "w:gz") as tar:
            for name in ("o1_records.jsonl", "o1_metadata.json",
                         "HARDWARE_GATE_REPORT.json",
                         "ENVIRONMENT_REPORT.resolved.json",
                         "BENCHMARK_REPORT.real.json",
                         "BACKEND_SELECTION.json",
                         "CALIBRATION_PRECOMMIT.deployed.json"):
                p = os.path.join(out_dir, name)
                if os.path.exists(p):
                    tar.add(p, arcname=name)
        digest = sha256_file(archive)
        # ONE definition of where the archive goes, shared with the off-pod
        # driver (zero_touch.RESULT_ARCHIVE_REL).  Both are relative to the
        # same destination, so a session prefix moves both or neither.
        from o1_b200.provider.runpod.zero_touch import RESULT_ARCHIVE_REL
        store.push_file(archive, RESULT_ARCHIVE_REL)
        back = os.path.join(out_dir, "transfer_verify.tar.gz")
        store.fetch_file(RESULT_ARCHIVE_REL, back)
        if sha256_file(back) != digest:
            raise ProductionEntryError("result transfer verification failed")
        os.remove(back)
        # publish the digest sidecar LAST: the off-pod driver cross-checks
        # its download against it, so a stale archive from an earlier
        # attempt can never be reported as this session's result
        sidecar = os.path.join(out_dir, "o1_results.tar.gz.sha256")
        atomic_write_text(sidecar, json.dumps(
            {"archive": "o1_results.tar.gz", "sha256": digest,
             "rows": _existing_row_count(
                 os.path.join(out_dir, "o1_records.jsonl")),
             # WHOSE result this is.  The archive and its digest live at a
             # fixed, run-agnostic key, so without this the driver could
             # download a PREVIOUS session's archive, check it against that
             # same session's sidecar, and report a fully aborted paid run
             # as a verified COMPLETE carrying the old numbers.
             "launch_nonce": os.environ.get("O1_LAUNCH_NONCE", ""),
             "image_digest": os.environ.get("O1_IMAGE_DIGEST", "")},
            indent=2, sort_keys=True) + "\n")
        store.push_file(sidecar, "results/o1_results.tar.gz.sha256")
        # The O1 RESULT MANIFEST for the Foundation Learner handover: the
        # FL supervisor verifies O1's records by recomputing the digests this
        # file lists (o1b200.transfer_manifest.v1 shape, absolute container
        # paths) and never reads their content.  Written LAST, after the
        # archive is durable, so its existence means "O1 is done and
        # transferred".
        atomic_write_text(
            os.path.join(out_dir, "O1_RESULT_MANIFEST.json"),
            json.dumps({
                "schema": "o1b200.transfer_manifest.v1",
                "artifacts": {
                    "o1_results_archive": {
                        "path": archive, "kind": "file", "sha256": digest},
                    "o1_records": {
                        "path": os.path.join(out_dir, "o1_records.jsonl"),
                        "kind": "file",
                        "sha256": ctx.get("records_sha256") or sha256_file(
                            os.path.join(out_dir, "o1_records.jsonl"))},
                },
                "launch_nonce": os.environ.get("O1_LAUNCH_NONCE", ""),
            }, indent=2, sort_keys=True) + "\n")
        _log_event(out_dir, "RESULTS_PUBLISHED", sha256=digest)
        return {"transferred": True, "sha256": digest}

    return {
        "PRECHECK": precheck,
        "ARTIFACT_VERIFY": artifact_verify,
        "ENVIRONMENT_VERIFY": environment_verify,
        "NON_O1_EQUIVALENCE": non_o1_equivalence,
        "NON_O1_BENCHMARK": non_o1_benchmark,
        "BACKEND_SELECT": backend_select,
        "PRECOMMIT_BUILD": precommit_build,
        "EXTERNAL_COMMIT_VERIFY": external_commit_verify,
        "CALIBRATION_AFFORDABILITY_CHECK": affordability,
        "CALIBRATION": calibration,
        "RECORD_VERIFY": record_verify,
        "RESULT_TRANSFER": result_transfer,
    }


def _log_event(out_dir: str, event: str, **fields) -> None:
    """Append one machine-readable pod-side event (durability, commitments)."""
    from ..provider.runpod.redaction import redact
    line = json.dumps({"event": event,
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
                       **fields}, sort_keys=True, default=str)
    with open(os.path.join(out_dir, "production_entry_events.jsonl"), "a",
              encoding="utf-8") as fh:
        fh.write(redact(line) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _diff_paths(a, b, prefix: str = "") -> list[str]:
    """Dotted paths where two nested JSON documents differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            p = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.append(p)
            else:
                out.extend(_diff_paths(a[key], b[key], p))
        return out
    return [] if a == b else [prefix or "<root>"]


def _existing_row_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def _manifest_checkpoint_sha256(manifest_path: str, checkpoint_dir: str) -> str:
    """The checkpoint tree hash the transfer manifest pins.

    verify_artifacts.py has already recomputed sha256_tree over the mounted
    tree and compared it to this value, so reading it here binds the run to
    bytes that were verified on THIS machine rather than to an assertion.
    """
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            entries = json.load(fh)["artifacts"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ProductionEntryError(
            f"cannot read the artifact manifest {manifest_path!r} to bind "
            f"the checkpoint tree hash: {exc}") from None
    want = os.path.abspath(checkpoint_dir)
    for name, spec in sorted(entries.items()):
        if os.path.abspath(str(spec.get("path", ""))) == want:
            digest = str(spec.get("sha256", ""))
            if len(digest) != 64:
                raise ProductionEntryError(
                    f"manifest entry {name!r} carries no usable sha256 for "
                    f"the checkpoint; refusing to run unbound")
            return digest
    raise ProductionEntryError(
        f"no manifest entry describes the checkpoint at {checkpoint_dir!r}; "
        f"the run cannot be bound to a verified model artifact")


def main() -> int:
    out_dir = os.environ.get("O1_B200_OUT", "/outputs")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.monotonic()
    clock = lambda: time.monotonic() - t0  # noqa: E731
    try:
        provider = LocalProviderAdapter(out_dir)
        handlers = build_production_handlers(out_dir, provider, clock)
        watchdog = BudgetWatchdog(
            int(float(_require_env("O1_SESSION_AUTHORIZED_SECONDS"))),
            clock=clock)
        machine = ZeroTouchStateMachine(provider, out_dir, handlers,
                                        watchdog=watchdog, clock=clock)
    except Exception as exc:  # noqa: BLE001 - deterministic: env/config
        # a missing identity env, a malformed result destination: the same
        # on every pod this launch produces, so the driver must not pay to
        # reacquire.  Written literally, like the markers below.
        print("ZERO_TOUCH_ABORTED_AT_PRE_ENTRY_CONFIG")
        print(f"REFUSED: production entry setup failed: {exc!r}",
              file=sys.stderr)
        return 1
    status = machine.run()
    # The off-pod driver greps these markers.  Both are written literally
    # (not assembled) so the marker the driver looks for and the marker the
    # pod emits cannot drift apart: a deterministic abort mistaken for an
    # eviction costs a full reacquisition cycle.
    if status["outcome"] == "COMPLETE":
        print("ZERO_TOUCH_COMPLETE")
    else:
        print("ZERO_TOUCH_ABORTED_AT_" + str(
            status.get("failed_state") or status["outcome"]))
    return 0 if status["outcome"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
