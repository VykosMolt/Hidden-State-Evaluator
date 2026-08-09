"""Plain-markdown + JSON reporting over records (contract §14)."""

from __future__ import annotations

import json

import pytest

from foundation_learner.analysis import report as RP
from foundation_learner.tests import _w4_support as S


def _arms():
    return {
        "FL3": S.constant_records("famA", 3, 0.8) + S.constant_records("famB", 3, 0.6),
        "FL1": S.constant_records("famA", 3, 0.4)
               + S.constant_records("famB", 3, 0.2),
    }


def _report(**kwargs):
    return RP.build_report(_arms(), baselines=["FL1"], n_replicates=200, seed=1,
                           **kwargs)


def test_report_contains_every_arm_and_paired_comparison():
    report = _report()
    assert report["schema"] == RP.ANALYSIS_REPORT_SCHEMA
    assert set(report["arms"]) == {"FL3", "FL1"}
    assert report["arms"]["FL3"]["metrics"]["macro_aulc"] == pytest.approx(0.7)
    comparison = report["comparisons"]["FL3_vs_FL1"]
    assert comparison["estimate"] == pytest.approx(0.4)
    assert "FL1_vs_FL1" not in report["comparisons"]


def test_unpairable_arms_are_reported_as_unavailable_not_dropped():
    arms = {"FL3": S.constant_records("famA", 3, 0.8),
            "FL1": S.constant_records("famB", 2, 0.4)}
    report = RP.build_report(arms, baselines=["FL1"], n_replicates=50, seed=1)
    entry = report["comparisons"]["FL3_vs_FL1"]
    assert "unavailable_reason" in entry
    assert entry["label_a"] == "FL3"


def test_markdown_is_plain_and_states_missing_values_as_na():
    report = _report(
        context_reset={"summary": {"reset_macro_aulc": 0.2,
                                   "history_macro_aulc": 0.7,
                                   "persistence": {"macro_retained_fraction": None,
                                                   "n_families_defined": 0,
                                                   "n_families": 2}}},
        interference={"summary": {"macro_retention_ratio": 0.5,
                                  "macro_interference_cost": 0.25,
                                  "n_chains": 4, "n_families": 2}},
        poison={"robustness": {"macro_aulc": {"clean": 0.7, "corrupted": 0.3},
                               "gap_vs_clean": {"corrupted": 0.4}}},
        remap={"robustness": {"canonical_macro_aulc": 0.7,
                              "remapped_macro_aulc": {"remap-a": 0.5},
                              "gap_vs_canonical": {"remap-a": 0.2}}},
        value_head={"spearman": 0.42,
                    "pairwise": {"accuracy": 0.6, "n_pairs": 100},
                    "calibration": {"slope": 0.9, "intercept": 0.01},
                    "regret": {"top1_regret": 0.2, "top1_regret_random": 0.5}},
        family_holdout={"unseen_instance": {"n_families": 1, "n_episodes": 3,
                                            "macro_aulc": 0.8, "r_0": 0.8,
                                            "r_K": 0.8},
                        "unseen_family": {"n_families": 1, "n_episodes": 3,
                                          "macro_aulc": 0.6, "r_0": 0.6,
                                          "r_K": 0.6}},
        notes=["a null result is a result"])
    text = RP.render_markdown(report)
    assert text.startswith("# ")
    assert "| FL3 |" in text and "| FL1 |" in text
    assert "macro retained fraction of in-context gain: n/a" in text
    assert "A -> B -> A retention and interference" in text
    assert "unseen_family" in text
    assert "not transferable learning" in text
    assert "a null result is a result" in text
    assert "top-1 regret: 0.2000 (random 0.5000, oracle 0)" in text
    assert all(ord(ch) < 128 for ch in text)  # plain ASCII markdown


def test_write_report_emits_json_and_markdown(tmp_path):
    report = _report()
    paths = RP.write_report(str(tmp_path), report, stem="fl0")
    payload = json.loads(open(paths["json"], encoding="utf-8").read())
    assert payload["schema"] == RP.ANALYSIS_REPORT_SCHEMA
    assert open(paths["json"], encoding="utf-8").read().endswith("\n")
    assert open(paths["markdown"], encoding="utf-8").read().startswith("# ")


def test_report_from_record_files_round_trips(tmp_path):
    from foundation_learner.evaluation import write_jsonl_records

    path = str(tmp_path / "fl3.jsonl")
    write_jsonl_records(path, S.constant_records("famA", 2, 0.5))
    report = RP.report_from_record_files({"FL3": path}, n_replicates=50, seed=1)
    assert report["arms"]["FL3"]["metrics"]["n_episodes"] == 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
