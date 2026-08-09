"""The 14 frozen metrics, with hand-computed values (contract §9)."""

from __future__ import annotations

import pytest

from foundation_learner.evaluation import metrics as M
from foundation_learner.tests import _w4_support as S


def _records():
    """Two families, hand-computable curves.

    family A: one episode  R = [0, 0, 1, 1, 1, 1, 1]        AULC = 5/7
    family B: two episodes R = [0, 0, 0, 0, 0, 0, 0] and
                               [0, 1, 1, 1, 1, 1, 1]        AULC = (0 + 6/7)/2
    """
    a = [S.make_record("a0", "famA", {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})]
    b = [S.make_record("b0", "famB", {k: 0 for k in range(7)}),
         S.make_record("b1", "famB", {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})]
    return a + b


# -- accessors -----------------------------------------------------------
def test_record_accessors_and_episode_aulc():
    rec = S.make_record("x", "famA", {0: 0.0, 1: 1.0})
    assert M.record_family(rec) == "famA"
    assert M.record_R(rec) == {0: 0.0, 1: 1.0}
    assert M.record_aulc(rec) == pytest.approx(0.5)
    assert M.record_aulc(rec, indices=[1]) == pytest.approx(1.0)
    assert M.record_aulc(S.make_record("y", "f", {})) is None
    with pytest.raises(KeyError):
        M.record_family({"R": {}})


# -- 1-3 -----------------------------------------------------------------
def test_macro_aulc_is_family_level_first():
    records = _records()
    fam = M.per_family_aulc(records)
    assert fam["famA"] == pytest.approx(5 / 7)
    assert fam["famB"] == pytest.approx((0.0 + 6 / 7) / 2)
    assert M.macro_aulc(records) == pytest.approx((5 / 7 + 3 / 7) / 2)
    assert M.macro_aulc([]) is None


def test_delta_aulc_against_a_baseline():
    treatment = _records()
    baseline = S.constant_records("famA", 1, 0.0) + S.constant_records("famB", 2, 0.0)
    assert M.delta_aulc(treatment, baseline) == pytest.approx(M.macro_aulc(treatment))
    assert M.delta_aulc(treatment, []) is None


# -- 4-6 -----------------------------------------------------------------
def test_curves_r0_rK_and_slope():
    records = _records()
    curve = M.macro_curve(records)
    assert curve[0] == pytest.approx(0.0)
    assert curve[1] == pytest.approx((0.0 + 0.5) / 2)
    assert curve[2] == pytest.approx((1.0 + 0.5) / 2)
    assert M.r_0(records) == pytest.approx(0.0)
    assert M.r_K(records) == pytest.approx(0.75)
    assert M.r_at(records, 4) == pytest.approx(0.75)
    slope = M.improvement_slope(records)
    assert slope is not None and slope > 0


def test_ols_slope_is_exact_on_a_known_line():
    assert M.ols_slope([0, 1, 2, 3], [1, 3, 5, 7]) == pytest.approx(2.0)
    assert M.ols_fit([0, 1, 2, 3], [1, 3, 5, 7]) == pytest.approx((2.0, 1.0))
    assert M.ols_slope([1], [1]) is None
    assert M.ols_slope([2, 2, 2], [1, 2, 3]) is None


def test_the_two_slope_conventions_coincide_on_a_shared_index_grid():
    records = _records()
    per_family = M.per_family_slopes(records)
    assert set(per_family) == {"famA", "famB"}
    assert M.slope_conventions_agree(records)
    assert M.improvement_slope(records) == pytest.approx(
        sum(per_family.values()) / 2)


# -- 7 -------------------------------------------------------------------
def test_interactions_to_threshold_reports_unidentifiable_families():
    records = _records()
    out = M.interactions_to_threshold(records, threshold=0.5)
    assert out["per_family"]["famA"] == 2      # first index with 1.0 >= 0.5
    assert out["per_family"]["famB"] == 1      # 0.5 >= 0.5
    assert out["n_identifiable"] == 2
    assert out["macro_mean"] == pytest.approx(1.5)
    hard = M.interactions_to_threshold(S.constant_records("famC", 2, 0.1))
    assert hard["per_family"]["famC"] is None
    assert hard["macro_mean"] is None and hard["n_identifiable"] == 0


# -- 8-9 -----------------------------------------------------------------
def test_related_task_transfer_and_whole_family_transfer():
    records = _records()
    assert M.related_task_transfer(records) == pytest.approx(0.75)
    assert M.whole_family_transfer(records, trained_family_ids=["famA"]) == \
        pytest.approx(3 / 7)
    assert M.whole_family_transfer(records, ["famA", "famB"]) is None


# -- 10 ------------------------------------------------------------------
def test_context_reset_persistence_is_a_retained_fraction():
    history = [S.make_record("e0", "famA", {0: 0.0, 4: 1.0, 5: 1.0, 6: 1.0})]
    reset = [S.make_record("e0", "famA", {0: 0.0, 4: 0.5, 5: 0.5, 6: 0.5})]
    out = M.context_reset_persistence(reset, history, reset_from_index=4)
    fam = out["per_family"]["famA"]
    assert fam["gain_history"] == pytest.approx(1.0)
    assert fam["gain_reset"] == pytest.approx(0.5)
    assert fam["retained_fraction"] == pytest.approx(0.5)
    assert out["macro_retained_fraction"] == pytest.approx(0.5)


def test_context_reset_persistence_is_undefined_without_an_in_context_gain():
    flat = [S.make_record("e0", "famA", {0: 0.4, 4: 0.4, 5: 0.4, 6: 0.4})]
    out = M.context_reset_persistence(flat, flat, reset_from_index=4)
    assert out["per_family"]["famA"]["retained_fraction"] is None
    assert out["macro_retained_fraction"] is None
    assert out["n_families_defined"] == 0 and out["n_families"] == 1


# -- 11 ------------------------------------------------------------------
def test_retention_interference_is_family_balanced():
    chains = [
        {"family_a": "famA", "aulc_a1": 1.0, "aulc_a2": 0.5, "aulc_b": 0.2,
         "recovery_interactions": 2},
        {"family_a": "famA", "aulc_a1": 1.0, "aulc_a2": 0.5, "aulc_b": 0.2,
         "recovery_interactions": 4},
        {"family_a": "famB", "aulc_a1": 0.8, "aulc_a2": 0.8, "aulc_b": 0.1,
         "recovery_interactions": 0},
    ]
    out = M.retention_interference(chains)
    assert out["per_family"]["famA"]["retention_ratio"] == pytest.approx(0.5)
    assert out["per_family"]["famA"]["interference_cost"] == pytest.approx(0.5)
    assert out["per_family"]["famA"]["recovery_interactions"] == pytest.approx(3.0)
    assert out["per_family"]["famB"]["retention_ratio"] == pytest.approx(1.0)
    # two chains for famA and one for famB must not weight famA twice
    assert out["macro_retention_ratio"] == pytest.approx(0.75)
    assert out["macro_interference_cost"] == pytest.approx(0.25)


# -- 12-13 ---------------------------------------------------------------
def test_poison_condition_ids_follow_the_data_layer():
    assert M.CLEAN_CONDITION == "clean"
    assert M.canonical_poison_condition("correct-informative") == "clean"
    assert M.canonical_poison_condition("partially_misleading") == "partially-misleading"
    assert M.canonical_poison_condition("corrupted") == "corrupted"
    with pytest.raises(ValueError):
        M.canonical_poison_condition("mildly_confusing")
    if S.ecology_available():
        from foundation_learner.ecology.poison import POISON_CONDITIONS
        assert M.POISON_CONDITIONS == tuple(POISON_CONDITIONS)


def test_poison_robustness_gap_against_clean():
    by_condition = {
        "clean": S.constant_records("famA", 2, 0.8) + S.constant_records("famB", 2, 0.6),
        "corrupted": S.constant_records("famA", 2, 0.4) + S.constant_records("famB", 2, 0.2),
    }
    out = M.poison_robustness(by_condition)
    assert out["macro_aulc"]["clean"] == pytest.approx(0.7)
    assert out["corrupted_gap"] == pytest.approx(0.4)
    assert out["clean_condition"] == "clean"


def test_remap_robustness_gap_against_the_canonical_surface():
    canonical = S.constant_records("famA", 2, 0.9)
    variants = {"remap-1": S.constant_records("famA", 2, 0.5, prefix="r1"),
                "remap-2": S.constant_records("famA", 2, 0.7, prefix="r2")}
    out = M.remap_robustness(canonical, variants)
    assert out["canonical_macro_aulc"] == pytest.approx(0.9)
    assert out["gap_vs_canonical"]["remap-1"] == pytest.approx(0.4)
    assert out["macro_gap"] == pytest.approx(0.3)


# -- 14 + registry -------------------------------------------------------
def test_value_ranking_metrics_delegate_to_the_analysis_layer():
    out = M.value_ranking_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert out["spearman"] == pytest.approx(1.0)
    assert out["pairwise"]["accuracy"] == pytest.approx(1.0)


def test_the_frozen_registry_lists_exactly_fourteen_metrics():
    assert len(M.METRICS_V0) == 14
    assert [e["id"] for e in M.METRICS_V0] == [str(i) for i in range(1, 15)]
    for entry in M.METRICS_V0:
        assert callable(M.metric_function(entry["name"]))
    with pytest.raises(KeyError):
        M.metric_function("not_a_metric")


def test_summarize_bundles_the_record_only_metrics():
    out = M.summarize(_records(), trained_family_ids=["famA"])
    assert out["schema"] == M.METRICS_SCHEMA
    assert out["n_families"] == 2 and out["n_episodes"] == 3
    assert out["macro_aulc"] == pytest.approx(M.macro_aulc(_records()))
    assert out["whole_family_transfer"] == pytest.approx(3 / 7)
    assert set(out["macro_curve"]) == {str(k) for k in range(7)}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
