"""G25 — the confirmatory cohort must BE the deterministic materialization.

WHY THIS EXISTS (finding B1, 2026-08-28)
----------------------------------------
Three independent scientific reviews of the sealed package found, and one
reproduced with a working proof of concept, that nothing verifies the
confirmatory task manifest against ``cohort_allocation.materialize()``:

  * ``o1_analysis.gate_G23_pool_membership`` checks only task id, content hash
    and generator setting;
  * ``gate_G11_stratum_consistency`` checks stratum and eligible setting, but
    neither the per-setting quota nor the candidate-pool sealed-rank prefix;
  * ``calibration_analysis.verify_calibration_binding`` recomputes the
    allocation but propagates only aggregate expected counts into the manifest;
  * ``load_task_manifest`` does not even require
    ``sealed_rank_within_generator_setting``.

So an ARBITRARY subset of eligible pool tasks -- and an arbitrary
action-to-stream Latin-square rank assignment -- passes every existing gate.
That defeats the property PREREGISTRATION.md section 14.4 claims outright:
"build_confirmatory_cohort.py materializes the cohort with zero operator
discretion". The selection is outcome-blind (no model outcome exists for any
pool task when the cohort is built), so this is not selection on measured
difficulty -- but it IS free selection on visible task content, and it is the
kind of latitude a reviewer is entitled to see closed mechanically rather than
trusted to operator discipline.

WHY IT LIVES OUTSIDE THE SEALED PACKAGE
---------------------------------------
The sealed v2.1 package is byte-pinned by ``sealed_import`` and its zip hash is
externally chronology-anchored; editing it would mint a new package identity
and invalidate the precommit that anchors the design's priority. This module
therefore RE-DERIVES the cohort using the sealed ``cohort_allocation`` code
itself (imported, never reimplemented) and compares. Same seam the batched
calibration backend uses: the sealed arithmetic stays authoritative, the
verification wraps it.

It is a PRECONDITION of the confirmatory analysis, which runs offline AFTER
calibration -- it is not on the paid-session path and gates nothing there.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from . import sealed_import

sealed_import.ensure_sealed_path()

from cohort_allocation import allocate, materialize  # noqa: E402 (sealed)


class ConfirmatoryCohortError(RuntimeError):
    """The confirmatory manifest is not the deterministic materialization."""


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _identity(rows: list[dict]) -> set[tuple]:
    """The tuple that must match exactly: which tasks, in which stratum slots.

    ``sealed_rank_within_stratum`` is included because it drives the cyclic
    Latin-square action-to-stream map (gate G13); leaving it free would let the
    same task set be re-paired against different action assignments.
    """
    out = set()
    for r in rows:
        if "sealed_rank_within_stratum" not in r:
            raise ConfirmatoryCohortError(
                f"task {r.get('task_id')!r} carries no "
                "'sealed_rank_within_stratum'; the manifest cannot be checked "
                "against the deterministic materialization")
        out.add((str(r["task_id"]), str(r["stratum"]),
                 int(r["sealed_rank_within_stratum"])))
    return out


def verify(confirmatory_path: str, candidate_pool_path: str,
           selected_settings: dict, g_confirm: int) -> dict:
    """Recompute allocate()+materialize() and require exact identity."""
    conf = _load_jsonl(confirmatory_path)
    pool = _load_jsonl(candidate_pool_path)

    eligible = {s for settings in selected_settings.values() for s in settings}
    if not eligible:
        raise ConfirmatoryCohortError(
            "calibration selected no eligible generator settings; there is "
            "nothing to materialize")
    epool = [r for r in pool if r["generator_setting_id"] in eligible]
    counts = collections.Counter(r["generator_setting_id"] for r in epool)

    try:
        expected = materialize(epool, allocate(selected_settings, dict(counts),
                                               int(g_confirm)))
    except Exception as exc:  # AllocationError and friends are sealed types
        raise ConfirmatoryCohortError(
            f"G25_COHORT_DERIVATION: the sealed allocator refused to "
            f"re-derive the cohort ({type(exc).__name__}: {exc}); the manifest "
            f"cannot be checked.") from exc
    if len(conf) != len(expected):
        raise ConfirmatoryCohortError(
            f"G25_COHORT_DERIVATION: the confirmatory manifest has "
            f"{len(conf)} rows but the materialization has {len(expected)}; "
            f"duplicate or missing rows collapse in a set comparison, so the "
            f"cardinality is checked before the identity.")
    got_ids, want_ids = _identity(conf), _identity(expected)
    if got_ids != want_ids:
        missing = sorted(t[0] for t in (want_ids - got_ids))
        extra = sorted(t[0] for t in (got_ids - want_ids))
        raise ConfirmatoryCohortError(
            f"G25_COHORT_DERIVATION: the confirmatory manifest is NOT the "
            f"deterministic materialization of the sealed candidate pool. "
            f"{len(missing)} expected task/stratum/rank triples are absent "
            f"(e.g. {missing[:3]}), {len(extra)} unexpected are present "
            f"(e.g. {extra[:3]}).")
    return {
        "schema": "o1b200.confirmatory_cohort_verification.v1",
        "gate": "G25_COHORT_DERIVATION",
        "verdict": "PASS",
        "n_confirmatory": len(conf),
        "n_candidate_pool": len(pool),
        "n_eligible_pool": len(epool),
        "g_confirm": int(g_confirm),
        "eligible_settings": sorted(eligible),
        "checked": ("task_id, stratum and sealed_rank_within_stratum match the "
                    "materialization exactly; the Latin-square action-to-stream "
                    "map is therefore pinned too"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirmatory", required=True)
    p.add_argument("--candidate-pool", required=True)
    p.add_argument("--calibration-result", required=True,
                   help="JSON carrying difficulty.selected_settings_per_stratum "
                        "and the final G_confirm")
    p.add_argument("--out")
    a = p.parse_args()
    with open(a.calibration_result, encoding="utf-8") as fh:
        cal = json.load(fh)
    selected = (cal.get("difficulty") or {}).get("selected_settings_per_stratum")
    # The sealed schema puts this under "power" (calibration_analysis writes it
    # there; build_confirmatory_cohort.py reads results["power"]["G_confirm"]).
    # It was read from a non-existent top-level "budget" block, so main() had
    # never once run against a real calibration result -- the registered test
    # calls verify() directly and never exercised this path.
    g_confirm = (cal.get("power") or {}).get("G_confirm")
    if g_confirm is None:                       # tolerate a wrapper that nests it
        g_confirm = (cal.get("budget") or {}).get("G_confirm")
    if selected is None or g_confirm is None:
        raise ConfirmatoryCohortError(
            "the calibration result carries no "
            "difficulty.selected_settings_per_stratum / budget.G_confirm; "
            "the cohort cannot be re-derived")
    report = verify(a.confirmatory, a.candidate_pool, selected, int(g_confirm))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(payload)
    print(payload.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
