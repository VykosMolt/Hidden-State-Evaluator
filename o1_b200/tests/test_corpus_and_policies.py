"""Corpus disjointness, O1-insertion attacks, benchmark/selection determinism."""
from __future__ import annotations

import json
import os

from _h import CORPUS_DIR, Runner, fresh_dir

from o1_b200.runner.benchmark_o1_b200 import (
    BenchmarkError, load_benchmark_order, run_benchmarks, stability_spread,
    windowed_token_rates,
)
from o1_b200.runner.runbuild import O1_MANIFEST_PATHS, build_validation_bundle
from o1_b200.runner.selection import SelectionError, select_backend
from o1_b200.runner.validation_corpus import (
    CorpusError, disjointness_report, load_corpus, write_corpus,
)

ARTIFACT = {"kind": "synthetic", "device": "cpu", "seed_tag": 0}


def _candidate(cid, backend, rows_per_hour, **over):
    c = {"config_id": cid, "backend": backend, "workers": over.pop("workers", 1),
         "batch": over.pop("batch", 1), "compaction": over.pop("compaction", False),
         "completed_rows_per_hour": rows_per_hour,
         "peak_hbm_reserved_bytes": over.pop("hbm", 10**11),
         "steady_state_free_hbm_fraction": over.pop("free", 0.4),
         "throughput_stability_spread": over.pop("spread", 0.05)}
    for g in ("structural_pass", "parser_verifier_pass",
              "action_seed_mapping_exact", "intervention_pass",
              "transport_pass", "resume_pass", "no_missing_or_duplicate_rows",
              "no_oom", "free_hbm_fraction_ok", "no_unvalidated_optimization",
              "throughput_stable", "scientific_config_unchanged"):
        c[g] = over.pop(g, True)
    c.update(over)
    return c


def run() -> Runner:
    r = Runner("corpus_policies")

    def corpus_disjoint():
        tasks, config = load_corpus(CORPUS_DIR)
        rep = disjointness_report(tasks, O1_MANIFEST_PATHS)
        assert rep["verdict"] == "DISJOINT"
        assert sum(m["n_tasks"] for m in rep["checked_manifests"]) == 96 + 2400
    r.check("mechanical zero-overlap with the 96-task calibration manifest "
            "and the 2400-task confirmatory candidate pool", corpus_disjoint)

    def o1_task_insertion_refused():
        tasks, _ = load_corpus(CORPUS_DIR)
        with open(O1_MANIFEST_PATHS[0], encoding="utf-8") as fh:
            o1_task = json.loads(fh.readline())
        poisoned = tasks + [o1_task]
        rep = disjointness_report(poisoned, O1_MANIFEST_PATHS)
        assert rep["verdict"] == "OVERLAP_FOUND"
    r.check("HOSTILE O1 task inserted into the validation corpus is detected",
            o1_task_insertion_refused)

    def out_of_population_task_refused():
        try:
            build_validation_bundle(CORPUS_DIR, ARTIFACT,
                                    task_subset=["not-a-corpus-task"])
        except ValueError:
            return
        raise AssertionError("out-of-population validation task accepted")
    r.check("HOSTILE out-of-population validation task is refused",
            out_of_population_task_refused)

    def tampered_corpus_refused():
        import shutil
        d = fresh_dir("tampered_corpus")
        for name in ("corpus_tasks.jsonl", "CORPUS_CONFIG.json",
                     "DISJOINTNESS_REPORT.json"):
            shutil.copy2(os.path.join(CORPUS_DIR, name), os.path.join(d, name))
        with open(os.path.join(d, "corpus_tasks.jsonl"), "a") as fh:
            fh.write("\n")
        try:
            load_corpus(d)
        except CorpusError:
            return
        raise AssertionError("tampered corpus accepted")
    r.check("HOSTILE tampered corpus file is refused by its sealed hash",
            tampered_corpus_refused)

    def benchmark_order_frozen_and_deterministic():
        """The order may only grow through a RECORDED amendment.

        This check used to pin the v1 list literally, which made any change
        -- principled or not -- fail identically.  That is the wrong shape:
        the policy has always carried an ``amendments`` mechanism, and the
        question is not "did the list change" but "did it change through the
        sanctioned route".  So: stages 1-10 stay byte-identical, and every
        stage BEYOND them must name an amendment that actually exists and
        that declares it left the selection rule alone.
        """
        order = load_benchmark_order()
        entries = order["staged_candidates"]
        ids = [e["config_id"] for e in entries]
        V1 = ["REFERENCE_SERIAL_w1_b1", "B200_REPLICA_w2_b1",
              "B200_REPLICA_w4_b1", "B200_REPLICA_w8_b1",
              "B200_BATCHED_w1_b4", "B200_BATCHED_w1_b8",
              "B200_BATCHED_w1_b16", "B200_BATCHED_w1_b24",
              "B200_BATCHED_w1_b32", "B200_BATCHED_w1_b48"]
        assert ids[:10] == V1, "stages 1-10 are frozen and must not be reordered"
        assert [e["stage"] for e in entries] == list(range(1, len(entries) + 1))
        assert all(e.get("conditional") is True
                   for e in entries if e["stage"] >= 8), \
            "deep-batch stages must stay behind the frozen extension rule"
        assert not any(e.get("conditional")
                       for e in entries if e["stage"] < 8)

        amendments = {a["amendment"]: a for a in order.get("amendments", [])}
        for e in entries[10:]:
            n = e.get("added_by_amendment")
            assert n in amendments, (
                f"{e['config_id']} was appended with no amendment record; the "
                f"candidate list may only grow through a recorded amendment")
            a = amendments[n]
            for field in ("date", "was", "why", "not_amended",
                          "scientific_impact", "bounded_by"):
                assert a.get(field), f"amendment {n} is missing {field!r}"
            assert "selection_rule" in a["not_amended"], (
                f"amendment {n} must declare the selection rule untouched: "
                f"adding CANDIDATES is permitted, choosing a WINNER is not")
        # whatever is appended, the terminal fallback must remain reachable
        assert ids[0] == "REFERENCE_SERIAL_w1_b1"
    r.check("benchmark policy: stages 1-10 frozen; the list may only grow "
            "through a recorded amendment that leaves selection untouched",
            benchmark_order_frozen_and_deterministic)

    def benchmark_refuses_unknown_and_unqualified_modes():
        try:
            run_benchmarks(CORPUS_DIR, fresh_dir("bench_refuse"), mode="b200")
        except BenchmarkError:
            pass
        else:
            raise AssertionError("unknown benchmark mode accepted")
        # real-hardware mode requires CUDA and the REAL artifact: with a
        # synthetic stand-in (or no CUDA) it must refuse
        try:
            run_benchmarks(CORPUS_DIR, fresh_dir("bench_refuse2"),
                           mode="real-hardware",
                           artifact={"kind": "synthetic", "device": "cpu",
                                     "seed_tag": 0})
        except BenchmarkError:
            return
        raise AssertionError("real-hardware mode ran without the real "
                             "artifact/CUDA qualification")
    r.check("benchmark harness refuses unknown modes and unqualified "
            "real-hardware runs", benchmark_refuses_unknown_and_unqualified_modes)

    def selection_deterministic_and_gated():
        cands = [
            _candidate("B200_BATCHED_w1_b16", "B200_BATCHED", 9000, batch=16,
                       structural_pass=False),      # fastest but ineligible
            _candidate("B200_BATCHED_w1_b8", "B200_BATCHED", 7000, batch=8),
            _candidate("B200_REPLICA_w4_b1", "B200_REPLICA", 7000, workers=4,
                       hbm=9 * 10**10),
            _candidate("REFERENCE_SERIAL_w1_b1", "REFERENCE_SERIAL", 1000),
        ]
        out = select_backend(cands)
        assert out["selected"]["config_id"] == "B200_REPLICA_w4_b1", \
            "tie-break failed (lower HBM should win at equal rows/hour)"
        again = select_backend(list(reversed(cands)))
        assert again["selected"]["config_id"] == out["selected"]["config_id"]
        ineligible = [c for c in out["all_judged"]
                      if c["config_id"] == "B200_BATCHED_w1_b16"][0]
        assert ineligible["eligible"] is False
    r.check("HOSTILE ineligible-but-fastest backend can never win; selection "
            "is deterministic under input reordering",
            selection_deterministic_and_gated)

    def selection_no_eligible_raises():
        try:
            select_backend([_candidate("x", "B200_BATCHED", 1.0, no_oom=False)])
        except SelectionError:
            return
        raise AssertionError("empty eligible set did not raise")
    r.check("selection with no eligible configuration refuses",
            selection_no_eligible_raises)

    def terminal_fallback_waives_only_stability():
        # nothing eligible, reference fails ONLY throughput_stable -> it is
        # the terminal fallback and the waiver is recorded
        out = select_backend([
            _candidate("B200_BATCHED_w1_b8", "B200_BATCHED", 9000, batch=8,
                       spread=0.6),
            _candidate("REFERENCE_SERIAL_w1_b1", "REFERENCE_SERIAL", 1000,
                       spread=0.4)])
        assert out["selected"]["config_id"] == "REFERENCE_SERIAL_w1_b1"
        assert out["terminal_fallback_waiver"]["waived_gates"] == [
            "throughput_stable"]
        # the waiver never extends to a scientific or safety gate
        for bad in ({"no_oom": False}, {"structural_pass": False},
                    {"free_hbm_fraction_ok": False},
                    {"scientific_config_unchanged": False}):
            try:
                select_backend([
                    _candidate("REFERENCE_SERIAL_w1_b1", "REFERENCE_SERIAL",
                               1000, spread=0.4, **bad)])
            except SelectionError:
                continue
            raise AssertionError(f"terminal fallback waived {bad}")
        # a non-reference backend is never a fallback
        try:
            select_backend([_candidate("B200_REPLICA_w2_b1", "B200_REPLICA",
                                       5000, workers=2, spread=0.4)])
        except SelectionError:
            pass
        else:
            raise AssertionError("replica became a fallback")
        # and when something IS eligible the waiver is not used
        out = select_backend([
            _candidate("B200_REPLICA_w2_b1", "B200_REPLICA", 5000, workers=2),
            _candidate("REFERENCE_SERIAL_w1_b1", "REFERENCE_SERIAL", 1000,
                       spread=0.4)])
        assert out["selected"]["config_id"] == "B200_REPLICA_w2_b1"
        assert out["terminal_fallback_waiver"] is None
    r.check("REFERENCE_SERIAL is the terminal fallback with only the "
            "stability gate waivable", terminal_fallback_waives_only_stability)

    def local_benchmark_dress_rehearsal():
        report = run_benchmarks(
            CORPUS_DIR, fresh_dir("bench_local"), mode="local-synthetic",
            task_subset=["b200val-000-commit_a", "b200val-004-malformed_eos"],
            max_stages=3)
        assert report["mode"] == "LOCAL_SYNTHETIC_DRESS_REHEARSAL"
        assert len(report["results"]) == 3
        for res in report["results"]:
            assert res["n_rows"] == 64
            assert res["integrity_failures"] == 0
            assert res["oom_count"] == 0
            assert res["resume_remaining_after_completion"] == 0
            assert len(res["throughput_windows_tokens_per_second"]) == 4
            assert res["throughput_stability_spread"] is not None
            assert "one execute_rows call" in res["throughput_stability_basis"]
    r.check("benchmark harness dress rehearsal (serial + replica w2/w4) "
            "collects metrics on the corpus", local_benchmark_dress_rehearsal)

    def stability_windows_measure_time_not_row_order():
        # a perfectly steady producer: 1 token every second for 40 s
        steady = [(float(t), 1) for t in range(1, 41)]
        rates = windowed_token_rates(steady, 0.0, 40.0)
        assert len(rates) == 4
        assert all(abs(x - 1.0) < 1e-9 for x in rates), rates
        assert stability_spread(rates) < 1e-9
        # the SAME steady producer committing in bursts of 10 (a batched
        # backend): interpolation must not read the bursts as instability
        bursty = [(10.0, 10), (20.0, 10), (30.0, 10), (40.0, 10)]
        rates = windowed_token_rates(bursty, 0.0, 40.0)
        assert all(abs(x - 1.0) < 1e-9 for x in rates), rates
        # a burst that straddles a window boundary is split pro rata
        straddle = [(15.0, 15), (40.0, 25)]
        rates = windowed_token_rates(straddle, 0.0, 40.0)
        assert all(abs(x - 1.0) < 1e-9 for x in rates), rates
        # a genuine stall in the third window IS detected
        stalled = [(float(t), 1) for t in range(1, 21)] + [(40.0, 1)] + \
            [(float(t), 1) for t in range(41, 61)]
        rates = windowed_token_rates(stalled, 0.0, 60.0)
        assert rates[2] < 0.5 * rates[0], rates
        assert stability_spread(rates) > 0.25
        # heavy rows clustered in one contiguous block of the corpus do NOT
        # move a steady token rate (the defect the old quartile split had)
        heavy_block = [(float(t), 1) for t in range(1, 21)] + \
            [(20.0 + 0.5 * i, 1) for i in range(1, 41)] + \
            [(40.0 + float(t), 1) for t in range(1, 21)]
        rows = windowed_token_rates([(t, 1) for t, _ in heavy_block], 0, 60)
        assert stability_spread(rows) > 0.25, "rows/s DOES vary here"
        # ...but at a constant 2 tokens/s the token rate is flat
        toks = [(t, 2) for t, _ in heavy_block[:20]] + \
            [(t, 1) for t, _ in heavy_block[20:60]] + \
            [(t, 2) for t, _ in heavy_block[60:]]
        assert stability_spread(windowed_token_rates(toks, 0, 60)) < 1e-9
        # degenerate inputs
        assert windowed_token_rates([], 0.0, 10.0) == [0.0] * 4
        assert windowed_token_rates(steady, 5.0, 5.0) == []
        assert stability_spread([]) is None
        assert stability_spread([0.0, 0.0]) == 1.0
    r.check("throughput stability is a time-windowed token rate, burst-safe "
            "and blind to corpus ordering",
            stability_windows_measure_time_not_row_order)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
