"""Zero-touch state machine: dress rehearsal + failure injection everywhere."""
from __future__ import annotations

import json
import os

from _h import CORPUS_DIR, Runner, fresh_dir

from o1_b200.runner.dress_rehearsal import (
    MockClock, build_handlers, run_rehearsal,
)
from o1_b200.runner.provider_adapter import MockProviderAdapter, ProviderError
from o1_b200.runner.state_machine import (
    O1_CALIBRATION_PRECOMMIT_SHA256, STATES, ZeroTouchStateMachine,
)

SUBSET = ["b200val-000-commit_a", "b200val-004-malformed_eos",
          "b200val-008-stoch_commit"]
INJECTABLE = [s for s in STATES if s not in ("TERMINATE", "COMPLETE")]


def run() -> Runner:
    r = Runner("state_machine")

    def full_rehearsal():
        out = fresh_dir("rehearsal_ok")
        status = run_rehearsal(CORPUS_DIR, out, subset=SUBSET)
        assert status["outcome"] == "COMPLETE", status
        assert status["termination"]["termination_confirmed"] is True
        final = json.load(open(os.path.join(out, "FINAL_STATUS.json")))
        assert final["machine_readable"] is True
        assert final["human_interaction_required"] is False
        # the selection ran production's gate derivation over REAL local
        # measurements (the harness, not invented numbers); the two inputs
        # that cannot exist on a CPU rehearsal are declared and labelled
        sel = json.load(open(os.path.join(
            out, "BACKEND_SELECTION.rehearsal.json")))
        assert sel["label"] == "DRESS_REHEARSAL_LOCAL_SYNTHETIC"
        assert "NOT_MEASURED" in sel["declared_inputs"]["environment"]["source"]
        judged = {c["config_id"]: c for c in sel["all_judged"]}
        assert "REFERENCE_SERIAL_w1_b1" in judged
        compared = {"REFERENCE_SERIAL_w1_b1", "B200_REPLICA_w2_b1",
                    "B200_BATCHED_w1_b8"}
        for cid, c in judged.items():
            # only configurations the equivalence step actually compared
            # carry a verdict; the others are INELIGIBLE, exactly as on the
            # pod (a config with no equivalence verdict of its own never
            # wins on throughput)
            assert c["equivalence_measured_for_this_config"] is (
                cid in compared), cid
            if cid not in compared:
                assert "structural_pass" in c["gate_failures"], cid
            assert c["no_missing_or_duplicate_rows"] is True, cid
            assert "throughput_stability_per_worker" in c
            assert c["throughput_stability_basis"].startswith("per execution stream")
        assert sel["selected"]["config_id"] in compared
        assert sel["terminal_fallback_waiver"] is None, (
            "the rehearsal COMPLETEd only through the terminal-fallback "
            "waiver: no accelerated backend passed the gates locally")
        bench = json.load(open(os.path.join(out, "benchmark",
                                            "BENCHMARK_REPORT.json")))
        assert bench["mode"] == "LOCAL_SYNTHETIC_DRESS_REHEARSAL"
        replica = [r for r in bench["results"]
                   if r["backend"] == "B200_REPLICA"][0]
        assert "per_worker" in replica["gpu"] or \
            "hbm_reserved_bytes_workers_peak_sum" in replica["gpu"]
    r.check("full mocked end-to-end dress rehearsal reaches COMPLETE with "
            "confirmed termination, machine-readable status and REAL gate "
            "derivation over the local benchmark", full_rehearsal)

    def injection_every_state():
        for state in INJECTABLE:
            out = fresh_dir(f"inject_{state.lower()}")
            status = run_rehearsal(CORPUS_DIR, out, subset=SUBSET, benchmark_stages=1,
                                   fail_at=state)
            assert status["outcome"] == f"ABORTED_AT_{state}", (state, status)
            later = [s for s in INJECTABLE
                     if INJECTABLE.index(s) > INJECTABLE.index(state)]
            for s in later:
                assert s in status["states_never_started"], (state, s)
            assert status["termination"]["termination_requested"] is True
            assert os.path.exists(os.path.join(out, "FINAL_STATUS.json"))
    r.check("failure injected at EVERY state: later states never begin, "
            "termination is requested, status is machine-readable",
            injection_every_state)

    def records_survive_late_failure():
        out = fresh_dir("inject_record_verify")
        status = run_rehearsal(CORPUS_DIR, out, subset=SUBSET, benchmark_stages=1,
                               fail_at="RECORD_VERIFY")
        assert status["outcome"] == "ABORTED_AT_RECORD_VERIFY"
        records = os.path.join(out, "rehearsal_calibration", "records.jsonl")
        assert os.path.exists(records), "completed synthetic records lost"
        rows = [json.loads(ln) for ln in open(records)]
        assert len(rows) > 0
    r.check("valid completed synthetic records survive a downstream failure",
            records_survive_late_failure)

    def provider_failure_each_op():
        for op in ("quote_instance", "validate_single_gpu", "start_instance",
                   "upload_artifacts", "download_results"):
            out = fresh_dir(f"pfail_{op}")
            provider = MockProviderAdapter(
                fail_on={op: ProviderError(f"injected {op} failure")})
            status = run_rehearsal(CORPUS_DIR, out, subset=SUBSET, benchmark_stages=1,
                                   provider=provider)
            assert status["outcome"].startswith("ABORTED_AT_"), (op, status)
    r.check("provider failure at every provider operation aborts cleanly",
            provider_failure_each_op)

    def termination_failure_flags_human():
        out = fresh_dir("pfail_terminate")
        provider = MockProviderAdapter(
            fail_on={"terminate_instance": ProviderError("stuck instance")})
        status = run_rehearsal(CORPUS_DIR, out, subset=SUBSET, benchmark_stages=1,
                               provider=provider)
        assert status["termination"].get("termination_confirmed") is not True
        final = json.load(open(os.path.join(out, "FINAL_STATUS.json")))
        assert final["human_interaction_required"] is True
    r.check("HOSTILE provider-termination failure is surfaced as "
            "human_interaction_required", termination_failure_flags_human)

    def confirmation_launch_refused():
        out = fresh_dir("confirm_refused")
        clock = MockClock()
        provider = MockProviderAdapter()
        handlers = build_handlers(CORPUS_DIR, out, provider, clock,
                                  subset=SUBSET)
        machine = ZeroTouchStateMachine(provider, out, handlers, clock=clock)
        machine.context["requested_mode"] = "confirmatory"
        status = machine.run()
        assert status["outcome"] == "ABORTED_AT_CALIBRATION", status
        assert "confirmatory" in status["failure"]
    r.check("HOSTILE attempted confirmation launch despite "
            "authorization=false is refused", confirmation_launch_refused)

    def o1_binding_refused():
        out = fresh_dir("o1_binding_refused")
        clock = MockClock()
        provider = MockProviderAdapter()
        handlers = build_handlers(CORPUS_DIR, out, provider, clock,
                                  subset=SUBSET)
        def poison(ctx):
            ctx["calibration_binding_sha256"] = O1_CALIBRATION_PRECOMMIT_SHA256
            return {}
        original = handlers["CALIBRATION_AFFORDABILITY_CHECK"]
        handlers["CALIBRATION_AFFORDABILITY_CHECK"] = \
            lambda ctx: (poison(ctx), original(ctx))[1]
        machine = ZeroTouchStateMachine(provider, out, handlers, clock=clock)
        status = machine.run()
        assert status["outcome"] == "ABORTED_AT_CALIBRATION", status
        assert "never run real O1 calibration" in status["failure"]
    r.check("HOSTILE real O1 calibration binding is refused by the local "
            "machine unconditionally", o1_binding_refused)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
