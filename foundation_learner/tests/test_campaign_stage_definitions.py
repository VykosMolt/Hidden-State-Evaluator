"""The declarative stage table (contract §7, §10, §11).

Structure, priority order, lazy resolution, projections, and the
feature-detected mechanism rungs.  Nothing here imports ``mechanisms``: that
package is under construction by another worker and its API is deliberately not
pinned by this table.
"""
from __future__ import annotations

import os

import pytest

from foundation_learner.campaign import o1_isolation, promotion
from foundation_learner.campaign import stage_definitions as sd
from foundation_learner.campaign.affordability import BenchMeasurement, PEFT_MODE


def bench(spu=1.0, eval_per_episode=0.5):
    return BenchMeasurement(scope=PEFT_MODE, seconds_per_update=spu,
                            tokens_per_second=10.0, updates_measured=4,
                            wall_seconds=4 * spu, forward_tokens=100,
                            max_tokens_per_batch=2048,
                            eval_seconds_per_episode=eval_per_episode)


def test_the_table_covers_bench_fl0_to_fl8_and_the_sealed_opening():
    ids = [s.stage_id for s in sd.STAGE_TABLE]
    for expected in ("BENCH", "FL0", "DEV_GRID", "FL1", "FL2", "FL3", "FL4",
                     "FL5", "FL6", "FL7", "FL8", "SECOND_SEED", "SEALED_EVAL"):
        assert expected in ids
    assert len(ids) == len(set(ids))


def test_priority_order_is_the_frozen_session_order():
    order = [s.stage_id for s in sd.stages_in_priority_order()]
    assert order[0] == "BENCH"
    assert order[1] == "FL0"
    assert order.index("DEV_GRID") < order.index("FL1")
    for arm in ("FL1", "FL2", "FL3"):
        assert order.index(arm) < order.index("FL4")
    for earlier, later in (("FL4", "FL5"), ("FL5", "FL6"), ("FL6", "FL7"),
                           ("FL7", "FL8"), ("FL8", "SECOND_SEED")):
        assert order.index(earlier) < order.index(later)
    assert order[-1] == "SEALED_EVAL"


def test_every_stage_declares_work_outputs_and_fallback_work():
    for stage in sd.STAGE_TABLE:
        assert ":" in stage.work
        assert stage.outputs, f"{stage.stage_id} declares no outputs"
        assert stage.fallback_work, f"{stage.stage_id} declares no fallback"


def test_every_dotted_path_in_the_table_resolves_or_is_a_mechanism():
    for stage in sd.STAGE_TABLE:
        assert callable(sd.resolve_dotted(stage.work))
        if stage.entry:
            assert callable(sd.resolve_dotted(stage.entry))


def test_resolve_optional_returns_none_for_a_missing_module():
    assert sd.resolve_optional("foundation_learner.not_a_module:thing") is None


def test_resolve_optional_raises_for_a_present_module_missing_the_entry_point():
    with pytest.raises(sd.StageError):
        sd.resolve_optional(
            "foundation_learner.campaign.stage_definitions:not_defined_here")


def test_bad_dotted_path_is_refused():
    with pytest.raises(sd.StageError):
        sd.resolve_dotted("foundation_learner.campaign.promotion")


def test_projection_formulas():
    b = bench(spu=2.0, eval_per_episode=0.5)
    train = sd.project_stage_seconds(sd.STAGES_BY_ID["FL3"], bench=b,
                                     updates=600, eval_episodes=10)
    assert train["train_seconds"] == 1200.0 and train["eval_seconds"] == 5.0
    grid = sd.project_stage_seconds(sd.STAGES_BY_ID["DEV_GRID"], bench=b,
                                    updates=600, eval_episodes=10)
    assert grid["updates_per_config"] == 150
    assert grid["projected_seconds"] == 2 * 150 * 2.0 + 2 * 10 * 0.5
    ev = sd.project_stage_seconds(sd.STAGES_BY_ID["FL0"], bench=b, updates=None,
                                  eval_episodes=4)
    assert ev["projected_seconds"] == 2.0
    with pytest.raises(sd.StageError):
        sd.project_stage_seconds(sd.STAGES_BY_ID["FL3"], bench=b, updates=None)


def test_eval_maxima_are_frozen_per_stage():
    for stage in sd.STAGE_TABLE:
        if stage.kind in ("EVAL", "TRAIN", "GRID", "MECHANISM", "SEALED"):
            if stage.stage_id in sd.EVAL_EPISODES_PER_STAGE:
                assert stage.eval_episodes == \
                    sd.EVAL_EPISODES_PER_STAGE[stage.stage_id]


def test_mechanism_stages_are_skipped_when_the_package_is_absent(tmp_path):
    guard = o1_isolation.IsolationGuard(label="TEST")
    ctx = sd.StageContext(out_dir=str(tmp_path), pregen_root=str(tmp_path),
                          bundle_factory=lambda: None, guard=guard)
    stage = sd.StageDefinition(
        stage_id="FLX", priority=99, kind="MECHANISM",
        work="x:y", projection="MECHANISM", outputs=("report.json",),
        requires_modules=("foundation_learner.definitely_absent_module",))
    payload = sd._mechanism_stage(ctx, stage, "x:y")
    assert payload["status"] == "SKIPPED_MECHANISMS"
    assert payload["missing_modules"] == \
        ["foundation_learner.definitely_absent_module"]
    assert os.path.isfile(os.path.join(str(tmp_path), "flx", "report.json"))


def test_mechanism_stage_with_a_present_module_but_no_entry_point_is_an_error(tmp_path):
    guard = o1_isolation.IsolationGuard(label="TEST")
    ctx = sd.StageContext(out_dir=str(tmp_path), pregen_root=str(tmp_path),
                          bundle_factory=lambda: None, guard=guard)
    stage = sd.StageDefinition(
        stage_id="FLX", priority=99, kind="MECHANISM", projection="MECHANISM",
        work="x:y", outputs=("r.json",),
        requires_modules=("foundation_learner.campaign.promotion",))
    with pytest.raises(sd.StageError):
        sd._mechanism_stage(ctx, stage,
                            "foundation_learner.campaign.promotion:nope")


def test_entry_conditions_come_from_the_promotion_module(tmp_path):
    guard = o1_isolation.IsolationGuard(label="TEST")
    ctx = sd.StageContext(out_dir=str(tmp_path), pregen_root=str(tmp_path),
                          bundle_factory=lambda: None, guard=guard)
    decision = sd.fl4_entry(ctx)
    assert isinstance(decision, promotion.PromotionDecision)
    assert decision.admitted is False           # no core comparison yet
    ctx.results["_dev_metrics"] = {
        "FL3": promotion.DevMetrics(stage="FL3", macro_aulc=0.6, slope=0.1),
        "FL1": promotion.DevMetrics(stage="FL1", macro_aulc=0.5, slope=0.0),
    }
    ctx.results["FL3"] = {"ok": True}
    ctx.extra["fl4_scoreable_items_per_episode"] = [2] * 200
    assert sd.fl4_entry(ctx).admitted is True


def test_fl1_pool_rendering_matches_the_episode_surface():
    from foundation_learner.episodes.render import INSTRUCTION_HEADER_V0

    text, (start, end) = sd._render_fl1_pair("what is 2+2?", "4")
    assert text.startswith(INSTRUCTION_HEADER_V0)
    assert "TASK: what is 2+2?" in text
    assert text[start:end] == "ANSWER: 4"
    assert text.endswith("\n")


def test_generation_budget_is_frozen_outside_a_rehearsal(tmp_path):
    guard = o1_isolation.IsolationGuard(label="TEST")
    ctx = sd.StageContext(out_dir=str(tmp_path), pregen_root=str(tmp_path),
                          bundle_factory=lambda: None, guard=guard,
                          extra={"max_new_tokens": 4})
    with pytest.raises(sd.StageError):
        sd._generation_config(ctx)
    ctx.rehearsal = True
    assert sd._generation_config(ctx).max_new_tokens == 4
    ctx.extra.pop("max_new_tokens")
    assert sd._generation_config(ctx).max_new_tokens == sd.FROZEN_MAX_NEW_TOKENS
