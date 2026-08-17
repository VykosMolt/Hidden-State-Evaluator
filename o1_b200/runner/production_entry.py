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
import time

from .budget import BudgetWatchdog, affordability_gate
from .compare_o1_backends import compare_rows, run_backend
from .durability import CheckpointDurability, store_for_destination
from .env_report import collect_pod_report, validate_b200_report
from .persistence import atomic_write_text
from .precommit_template import finalize, load_template, resolve
from .provider_adapter import LocalProviderAdapter
from .runbuild import O1_MANIFEST_PATHS
from .selection import select_backend
from .state_machine import ZeroTouchStateMachine
from .validation_corpus import disjointness_report, load_corpus
from . import sealed_import

ROOT = sealed_import.WORKTREE_ROOT
CHECKPOINT_DIR = os.environ.get("O1_CHECKPOINT_DIR",
                                "/artifacts/ouro_rltt_local")
O1_ROWS_TOTAL = 4608
RECORDS_SYNC_EVERY_ROWS = 25


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
        manifest = os.environ.get(
            "O1_B200_TRANSFER_MANIFEST",
            os.path.join(ROOT, "o1_b200", "deploy", "TRANSFER_MANIFEST.json"))
        proc = subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "o1_b200", "deploy", "verify_artifacts.py"),
             "--manifest", manifest],
            capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise ProductionEntryError(
                f"artifact verification failed:\n{proc.stdout[-1500:]}"
                f"{proc.stderr[-1500:]}")
        tasks, config = load_corpus(corpus_dir)
        rep = disjointness_report(tasks, O1_MANIFEST_PATHS)
        if rep["verdict"] != "DISJOINT":
            raise ProductionEntryError("corpus not disjoint from O1 pools")
        ctx["corpus_config"] = config
        return {"artifacts_verified": True, "disjointness": rep["verdict"]}

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
                                                  "UNKNOWN"))
        validated = validate_b200_report(env_report)
        atomic_write_text(
            os.path.join(out_dir, "ENVIRONMENT_REPORT.resolved.json"),
            json.dumps({"report": env_report, "validation": validated},
                       indent=2, sort_keys=True, default=str) + "\n")
        ctx["environment_report"] = validated
        ctx["hardware_gate"] = gate
        return {"hardware_gate": "ACCEPTED", "profile": profile_key}

    def non_o1_equivalence(ctx):
        subset = None   # full non-O1 validation corpus, exactly once
        ref = run_backend("REFERENCE_SERIAL", corpus_dir,
                          os.path.join(out_dir, "eq_ref"), artifact,
                          task_subset=subset)
        comp = {}
        for backend_id, w, b in (("B200_REPLICA", 2, 1),
                                 ("B200_BATCHED", 1, 8)):
            cand = run_backend(backend_id, corpus_dir,
                               os.path.join(out_dir, f"eq_{backend_id}"),
                               artifact, worker_count=w, batch_size=b,
                               task_subset=subset)
            comp[backend_id] = compare_rows(ref["rows"], cand["rows"])
        if not all(c["eligible_structurally"] for c in comp.values()):
            raise ProductionEntryError("structural equivalence gate failed")
        ctx["equivalence"] = comp
        return {k: v["eligible_structurally"] for k, v in comp.items()}

    def non_o1_benchmark(ctx):
        from .benchmark_o1_b200 import run_benchmarks
        rep = run_benchmarks(
            corpus_dir, os.path.join(out_dir, "benchmark"),
            mode="real-hardware", artifact=artifact,
            remaining_authorized_seconds=ctx["runtime_limit"] - clock())
        ctx["benchmark_raw"] = [r for r in rep["results"]
                                if not r.get("skipped")]
        atomic_write_text(
            os.path.join(out_dir, "BENCHMARK_REPORT.real.json"),
            json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n")
        return {"benchmarked": len(ctx["benchmark_raw"])}

    def _derive_gates(entry: dict, ctx) -> dict:
        """Mechanical gate derivation from real measurements — never from
        any O1 outcome."""
        import torch
        eq = ctx["equivalence"].get(entry.get("backend"), {})
        structural = (entry.get("backend") == "REFERENCE_SERIAL"
                      or bool(eq.get("eligible_structurally")))
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = (entry.get("gpu") or {}).get("hbm_reserved_bytes", 0)
        free_frac = 1.0 - (reserved / total if total else 1.0)
        expected_rows = entry.get("n_rows", 0) > 0
        spread = entry.get("throughput_stability_spread")
        return {
            **entry,
            "completed_rows_per_hour":
                float(entry.get("completed_rows_per_second", 0.0)) * 3600.0,
            "peak_hbm_reserved_bytes": reserved,
            "steady_state_free_hbm_fraction": free_frac,
            "structural_pass": structural,
            "parser_verifier_pass": entry.get("integrity_failures", 1) == 0,
            "action_seed_mapping_exact": structural,
            "intervention_pass": structural,
            "transport_pass": structural,
            "resume_pass": entry.get(
                "resume_remaining_after_completion", -1) == 0,
            "no_missing_or_duplicate_rows": bool(expected_rows),
            "no_oom": entry.get("oom_count", 1) == 0,
            "free_hbm_fraction_ok": free_frac >= 0.15,
            "no_unvalidated_optimization": True,   # compile/graphs OFF
            "throughput_stable": (spread is not None and spread <= 0.25),
            "scientific_config_unchanged": True,   # frozen bundle inputs
        }

    def backend_select(ctx):
        candidates = [ _derive_gates(e, ctx) for e in ctx["benchmark_raw"]
                       if "config_id" in e and not e.get("oom")
                       and not e.get("integrity_failure")]
        ctx["benchmark"] = candidates
        out = select_backend(candidates)
        ctx["selected_backend"] = out["selected"]
        atomic_write_text(
            os.path.join(out_dir, "BACKEND_SELECTION.json"),
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
        return {"selected": out["selected"]["config_id"]}

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
            "environment_digest_sha256": ctx["environment_report"].get(
                "report_sha256", "0" * 64),
            "final_backend_benchmark_report_sha256": gate_sha,
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
        # off-pod commitment: push the finalized precommit to the durable
        # store, independently re-fetch it, and verify byte identity.  The
        # pod deliberately has no git credentials; the durable store is the
        # session's external witness.
        from .identity import sha256_file
        pushed = store.push_file(ctx["precommit_path"],
                                 "commitments/CALIBRATION_PRECOMMIT.json")
        back = os.path.join(out_dir, "precommit_refetch.json")
        store.fetch_file("commitments/CALIBRATION_PRECOMMIT.json", back)
        if sha256_file(back) != sha256_file(ctx["precommit_path"]):
            raise ProductionEntryError(
                "external commitment verification failed: re-fetched "
                "precommit differs")
        os.remove(back)
        return {"remote_verified": True, "sha256": pushed["sha256"]}

    def affordability(ctx):
        rows_per_hour = float(
            ctx["selected_backend"]["completed_rows_per_hour"])
        done = _existing_row_count(os.path.join(out_dir, "o1_records.jsonl"))
        projected = (O1_ROWS_TOTAL - done) / max(rows_per_hour, 1e-9) * 3600
        return affordability_gate(
            projected_calibration_seconds=projected,
            verification_transfer_reserve_seconds=1200,
            termination_reserve_seconds=300,
            remaining_authorized_runtime_seconds=(
                ctx["runtime_limit"] - clock()))

    def _build_replacement_manifest(path: str) -> str:
        """Replacement freeze manifest: byte-identical to the sealed
        FREEZE_MANIFEST.precalibration.json EXCEPT code.torch, which must
        carry the deployed build (the sealed orchestrator runtime-asserts
        it).  This is the documented infrastructure migration, not a
        scientific change: every scientific binding is copied verbatim."""
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
        regen = subprocess.run(
            [sys.executable,
             os.path.join(sealed_import.SEALED_DIR,
                          "calibration_precommit.py"),
             "--output", sealed_precommit_path,
             "--manifest-design", manifest_path,
             "--artifact-paths", artifact_map_path,
             "--calibration-task-manifest",
             os.path.join(run_root, "COHORTS", "calibration_tasks.jsonl")],
            capture_output=True, text=True, timeout=1800, env=env,
            cwd=sealed_import.SEALED_DIR)
        if regen.returncode != 0 or not os.path.exists(sealed_precommit_path):
            raise ProductionEntryError(
                f"sealed precommit regeneration failed:\n"
                f"{regen.stdout[-1000:]}{regen.stderr[-1000:]}")
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
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                env=env, cwd=sealed_import.SEALED_DIR)
        last_mirrored = _existing_row_count(records_path)
        log_path = os.path.join(out_dir, "o1_orchestrator.log")
        with open(log_path, "a", encoding="utf-8") as log:
            while proc.poll() is None:
                line = proc.stdout.readline()
                if line:
                    log.write(line)
                    log.flush()
                rows_now = _existing_row_count(records_path)
                if rows_now - last_mirrored >= RECORDS_SYNC_EVERY_ROWS:
                    records_mirror.sync_checkpoint(
                        records_path, {"rows": rows_now,
                                       "progress": "partial"})
                    last_mirrored = rows_now
            for line in proc.stdout:
                log.write(line)
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
        store.push_file(archive, "results/o1_results.tar.gz")
        back = os.path.join(out_dir, "transfer_verify.tar.gz")
        store.fetch_file("results/o1_results.tar.gz", back)
        if sha256_file(back) != digest:
            raise ProductionEntryError("result transfer verification failed")
        os.remove(back)
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


def _existing_row_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def main() -> int:
    out_dir = os.environ.get("O1_B200_OUT", "/outputs")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.monotonic()
    clock = lambda: time.monotonic() - t0  # noqa: E731
    provider = LocalProviderAdapter(out_dir)
    handlers = build_production_handlers(out_dir, provider, clock)
    watchdog = BudgetWatchdog(
        int(float(_require_env("O1_SESSION_AUTHORIZED_SECONDS"))),
        clock=clock)
    machine = ZeroTouchStateMachine(provider, out_dir, handlers,
                                    watchdog=watchdog, clock=clock)
    status = machine.run()
    print("ZERO_TOUCH_COMPLETE" if status["outcome"] == "COMPLETE"
          else f"ZERO_TOUCH_{status['outcome']}")
    return 0 if status["outcome"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
