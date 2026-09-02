"""G25: the confirmatory cohort must BE the deterministic materialization.

Finding B1 (2026-08-28): three independent reviews found, and one reproduced
with a working PoC, that an arbitrary subset of eligible candidate-pool tasks
with arbitrary stratum ranks passed every existing analysis gate.  These tests
are the regression: the honest cohort passes, and the reproduced substitution
is refused.
"""
from __future__ import annotations

import collections
import json
import os
import tempfile

from _h import Runner

from o1_b200.runner import sealed_import
from o1_b200.runner.verify_confirmatory_cohort import (
    ConfirmatoryCohortError, verify,
)

sealed_import.ensure_sealed_path()

from cohort_allocation import allocate, materialize  # noqa: E402 (sealed)

SELECTED = {"hard_medium": ["rules_3"], "medium": ["rules_2"]}
G_CONFIRM = 8


def _pool() -> list[dict]:
    return [{"task_id": f"t-{s}-{i:03d}", "generator_setting_id": s,
             "sealed_rank_within_generator_setting": i,
             "task_content_sha256": f"h{s}{i}",
             "options": ["True", "False", "Unknown"]}
            for s in ("rules_2", "rules_3") for i in range(12)]


def _write(directory: str, name: str, rows: list[dict]) -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _canonical(pool: list[dict]) -> list[dict]:
    counts = collections.Counter(r["generator_setting_id"] for r in pool)
    return materialize(pool, allocate(SELECTED, dict(counts), G_CONFIRM))


def run() -> Runner:
    r = Runner("confirmatory_cohort_gate")
    pool = _pool()
    canonical = _canonical(pool)
    tmp = tempfile.mkdtemp(prefix="g25_")
    pool_path = _write(tmp, "pool.jsonl", pool)

    def honest_cohort_passes():
        out = verify(_write(tmp, "good.jsonl", canonical), pool_path,
                     SELECTED, G_CONFIRM)
        assert out["verdict"] == "PASS", out
        assert out["g_confirm"] == G_CONFIRM
    r.check("the deterministic materialization passes G25", honest_cohort_passes)

    def substituted_subset_refused():
        """The reproduced exploit: a DIFFERENT eligible subset, valid hashes
        and settings, ranks reassigned.  Every pre-existing gate accepted it."""
        by_setting = collections.defaultdict(list)
        for row in pool:
            by_setting[row["generator_setting_id"]].append(row)
        adversarial = []
        for stratum, setting in (("hard_medium", "rules_3"),
                                 ("medium", "rules_2")):
            rows = sorted(by_setting[setting],
                          key=lambda x: -x["sealed_rank_within_generator_setting"])
            for rank, row in enumerate(rows[:4]):
                adversarial.append({**row, "stratum": stratum,
                                    "sealed_rank_within_stratum": rank})
        assert {t["task_id"] for t in adversarial} != {t["task_id"] for t in canonical}
        r.expect_raises(
            "substituted cohort refused", ConfirmatoryCohortError,
            lambda: verify(_write(tmp, "bad.jsonl", adversarial), pool_path,
                           SELECTED, G_CONFIRM))
    r.check("HOSTILE a substituted eligible subset is refused",
            substituted_subset_refused)

    def repaired_ranks_refused():
        """Same task set, PERMUTED stratum ranks.  The ranks drive the cyclic
        Latin-square action-to-stream map (G13), so leaving them free would let
        the same tasks be re-paired against different action assignments."""
        permuted = [dict(t) for t in canonical]
        for t in permuted:
            t["sealed_rank_within_stratum"] = (
                int(t["sealed_rank_within_stratum"]) + 1) % 4
        r.expect_raises(
            "permuted stratum ranks refused", ConfirmatoryCohortError,
            lambda: verify(_write(tmp, "perm.jsonl", permuted), pool_path,
                           SELECTED, G_CONFIRM))
    r.check("HOSTILE permuted stratum ranks are refused", repaired_ranks_refused)

    def missing_rank_field_refused():
        stripped = [{k: v for k, v in t.items()
                     if k != "sealed_rank_within_stratum"} for t in canonical]
        r.expect_raises(
            "manifest without stratum ranks refused", ConfirmatoryCohortError,
            lambda: verify(_write(tmp, "norank.jsonl", stripped), pool_path,
                           SELECTED, G_CONFIRM))
    r.check("a manifest lacking sealed_rank_within_stratum is refused",
            missing_rank_field_refused)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
