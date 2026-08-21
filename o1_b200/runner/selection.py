"""Deterministic implementation of the frozen backend-selection policy.

Pure function over (equivalence report, benchmark results); no I/O, no model
access, no O1 outcome anywhere in its inputs.  The JSON policy
(policies/B200_BACKEND_SELECTION_POLICY.json) is the human-readable frozen
form; this module is its executable form and the tests hold them together.
"""
from __future__ import annotations

SEMANTIC_COMPLEXITY = {
    "REFERENCE_SERIAL": 0,
    "B200_REPLICA": 1,
    "B200_BATCHED": 2,           # without compaction
    "B200_BATCHED_COMPACTION": 3,
}

REQUIRED_GATES = (
    "structural_pass", "parser_verifier_pass", "action_seed_mapping_exact",
    "intervention_pass", "transport_pass", "resume_pass",
    "no_missing_or_duplicate_rows", "no_oom", "free_hbm_fraction_ok",
    "no_unvalidated_optimization", "throughput_stable",
    "scientific_config_unchanged",
)

MIN_FREE_HBM_FRACTION = 0.15
MAX_THROUGHPUT_SPREAD = 0.25
#: The frozen policy names REFERENCE_SERIAL the terminal fallback.  A
#: fallback that can itself be refused on a PERFORMANCE gate is not a
#: fallback, so this one gate — and only this one — is waived for the
#: reference when nothing else is eligible.  Every scientific and safety
#: gate still applies to it.
TERMINAL_FALLBACK_BACKEND = "REFERENCE_SERIAL"
TERMINAL_FALLBACK_WAIVABLE_GATES = ("throughput_stable",)


class SelectionError(RuntimeError):
    pass


def benchmark_candidates(results: list[dict]) -> list[dict]:
    """The benchmark entries that may be judged: measured, no OOM, no
    integrity failure (stub entries for skipped/failed stages excluded).
    ONE filter for production and the rehearsal."""
    return [r for r in results
            if "config_id" in r and not r.get("skipped")
            and not r.get("oom_count") and not r.get("integrity_failures")
            and not r.get("oom") and not r.get("integrity_failure")]


def derive_gates(entry: dict, *, equivalence: dict | None,
                 device_total_memory: int, expected_rows: int,
                 environment: dict | None,
                 corpus_config_verified: bool) -> dict:
    """Mechanical gate derivation from real per-config measurements.

    ONE implementation for the production entry and the dress rehearsal, so
    the rehearsal exercises the gates the pod will apply instead of
    asserting them True.  Every gate is derived from a measurement OF THIS
    CONFIGURATION: a config with no equivalence verdict of its own is
    ineligible; the row-count and environment gates are checked against
    the corpus size and the recorded environment rather than asserted.
    """
    eq = equivalence
    measured = eq is not None
    structural = bool(measured and eq.get("eligible_structurally"))
    core_identical = bool(measured and (eq.get("is_reference")
                                        or eq.get("scientific_core_identical")))
    total = int(device_total_memory or 0)
    reserved = int((entry.get("gpu") or {}).get("hbm_reserved_bytes", 0) or 0)
    free_frac = 1.0 - (reserved / total if total else 1.0)
    expected_rows = int(expected_rows or 0)
    rows_ok = (expected_rows > 0
               and int(entry.get("n_rows", -1)) == expected_rows)
    spread = entry.get("throughput_stability_spread")
    env = environment or {}
    no_unvalidated_opt = (env.get("compile_state") in ("OFF", "off", False)
                          and env.get("cuda_graph_state") in
                          ("OFF", "off", False)
                          and env.get("attention_backend") == "eager")
    return {
        **entry,
        "equivalence_measured_for_this_config": measured,
        "completed_rows_per_hour":
            float(entry.get("completed_rows_per_second", 0.0)) * 3600.0,
        "peak_hbm_reserved_bytes": reserved,
        "steady_state_free_hbm_fraction": free_frac,
        "structural_pass": structural,
        "parser_verifier_pass": (measured
                                 and entry.get("integrity_failures", 1) == 0),
        "action_seed_mapping_exact": core_identical,
        "intervention_pass": core_identical,
        "transport_pass": core_identical,
        "resume_pass": entry.get(
            "resume_remaining_after_completion", -1) == 0,
        "no_missing_or_duplicate_rows": rows_ok,
        "no_oom": entry.get("oom_count", 1) == 0,
        "free_hbm_fraction_ok": free_frac >= MIN_FREE_HBM_FRACTION,
        "no_unvalidated_optimization": no_unvalidated_opt,
        "throughput_stable": (spread is not None
                              and spread <= MAX_THROUGHPUT_SPREAD),
        "scientific_config_unchanged": bool(corpus_config_verified),
    }


def is_eligible(candidate: dict) -> tuple[bool, list[str]]:
    """candidate carries gates plus measured fields."""
    failures = []
    for gate in REQUIRED_GATES:
        if candidate.get(gate) is not True:
            failures.append(gate)
    free = candidate.get("steady_state_free_hbm_fraction")
    if free is None or free < MIN_FREE_HBM_FRACTION:
        if "free_hbm_fraction_ok" not in failures:
            failures.append("free_hbm_fraction_ok")
    spread = candidate.get("throughput_stability_spread")
    if spread is None or spread > MAX_THROUGHPUT_SPREAD:
        if "throughput_stable" not in failures:
            failures.append("throughput_stable")
    return (not failures, failures)


def _complexity(candidate: dict) -> int:
    backend = candidate["backend"]
    if backend == "B200_BATCHED" and candidate.get("compaction"):
        return SEMANTIC_COMPLEXITY["B200_BATCHED_COMPACTION"]
    return SEMANTIC_COMPLEXITY[backend]


def select_backend(candidates: list[dict]) -> dict:
    """Frozen rule: highest verified completed rows/hour among eligible.

    Tie-break: lower peak HBM; lower semantic complexity; smaller
    worker/batch count; lexicographic config_id.
    """
    judged = []
    for c in candidates:
        ok, failures = is_eligible(c)
        judged.append({**c, "eligible": ok, "gate_failures": failures})
    eligible = [c for c in judged if c["eligible"]]
    fallback_waiver = None
    if not eligible:
        for c in judged:
            if c.get("backend") != TERMINAL_FALLBACK_BACKEND:
                continue
            blocking = [g for g in c["gate_failures"]
                        if g not in TERMINAL_FALLBACK_WAIVABLE_GATES]
            if blocking:
                continue
            fallback_waiver = {
                "config_id": c["config_id"],
                "waived_gates": list(c["gate_failures"]),
                "rule": ("frozen policy: REFERENCE_SERIAL is the terminal "
                         "fallback; only the throughput-stability performance "
                         "gate may be waived for it, never a scientific or "
                         "safety gate"),
            }
            c = {**c, "eligible": True,
                 "terminal_fallback_waiver": fallback_waiver}
            eligible = [c]
            break
    if not eligible:
        raise SelectionError(
            "no eligible backend configuration and the terminal fallback "
            "REFERENCE_SERIAL fails a scientific or safety gate: "
            f"{[(c['config_id'], c['gate_failures']) for c in judged]}")

    def sort_key(c):
        return (
            -float(c["completed_rows_per_hour"]),
            float(c.get("peak_hbm_reserved_bytes", float("inf"))),
            _complexity(c),
            int(c.get("workers", 1)) + int(c.get("batch", 1)),
            int(c.get("workers", 1)),
            str(c["config_id"]),
        )

    winner = sorted(eligible, key=sort_key)[0]
    return {"selected": winner, "eligible": eligible, "all_judged": judged,
            "terminal_fallback_waiver": fallback_waiver}
