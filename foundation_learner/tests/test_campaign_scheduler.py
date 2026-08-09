"""Scheduler admission, ordering, cadence, and journalling (contract §11, §15).

Every check uses a deterministic :class:`ManualClock` and stub work functions,
so the scheduling arithmetic is exercised without touching a model.  The stage
table itself is checked separately in ``test_campaign_stage_definitions.py``.
"""
from __future__ import annotations

import json

import pytest

from foundation_learner.campaign import o1_isolation, scheduler as sched
from foundation_learner.campaign.affordability import (AffordabilityRefusal,
                                                       BenchMeasurement,
                                                       FULL_MODEL_MODE,
                                                       PEFT_MODE)
from foundation_learner.campaign.stage_definitions import (StageContext,
                                                           StageDefinition)

CALLS: list[str] = []


def stub_work(ctx, stage):
    CALLS.append(stage.stage_id)
    ctx.results[stage.stage_id] = {"ok": True}
    return {"ok": True}


def failing_work(ctx, stage):
    raise RuntimeError("stub failure")


def refusing_entry(ctx):
    from foundation_learner.campaign.promotion import PromotionDecision

    return PromotionDecision(stage="STUB", admitted=False,
                             reasons=("FAIL: stub",), fallback=("do nothing",))


THIS = "foundation_learner.tests.test_campaign_scheduler"


def stage(stage_id="STUB", projection="TRAIN_ARM", work="stub_work",
          entry=None, eval_episodes=10, **kw):
    return StageDefinition(
        stage_id=stage_id, priority=1, kind="TRAIN", work=f"{THIS}:{work}",
        projection=projection, outputs=(), entry=entry,
        eval_episodes=eval_episodes, **kw)


def context(tmp_path, *, spu=1.0, eval_per_episode=0.1, updates=600):
    guard = o1_isolation.IsolationGuard(label="TEST")
    ctx = StageContext(out_dir=str(tmp_path / "ladder"),
                       pregen_root=str(tmp_path / "pregen"),
                       bundle_factory=lambda: None, guard=guard,
                       updates=updates, eval_episode_cap=None)
    ctx.throughput.add(BenchMeasurement(
        scope=PEFT_MODE, seconds_per_update=spu, tokens_per_second=100.0,
        updates_measured=10, wall_seconds=10 * spu, forward_tokens=1000,
        max_tokens_per_batch=2048, eval_seconds_per_episode=eval_per_episode))
    return ctx, guard


def make_scheduler(tmp_path, available, guard, clock=None):
    return sched.Scheduler(available_foundation_learner_seconds=available,
                           out_dir=str(tmp_path / "ladder"), guard=guard,
                           clock=clock or sched.ManualClock())


def test_the_clock_is_injected_and_drives_the_remaining_budget(tmp_path):
    ctx, guard = context(tmp_path)
    clock = sched.ManualClock()
    s = make_scheduler(tmp_path, 1000.0, guard, clock)
    s.start()
    assert s.remaining_authorized() == 1000.0
    clock.advance(400.0)
    assert s.elapsed() == 400.0
    assert s.remaining_authorized() == 600.0
    clock.advance(10_000.0)
    assert s.remaining_authorized() == 0.0


def test_admission_journals_both_outcomes(tmp_path):
    ctx, guard = context(tmp_path)
    s = make_scheduler(tmp_path, 5000.0, guard)
    s.start()
    s.admit("FL1", 1000.0)
    with pytest.raises(AffordabilityRefusal):
        s.admit("FL2", 100_000.0)
    events = [r["event"] for r in s.read_journal()]
    assert "STAGE_ADMITTED" in events and "STAGE_REFUSED" in events


def test_a_stage_that_violates_the_reserve_is_refused(tmp_path):
    CALLS.clear()
    ctx, guard = context(tmp_path, spu=1.0, eval_per_episode=0.0)
    # projection: 600 updates * 1 s = 600 s -> 600*1.25 + 1200 = 1950
    s = make_scheduler(tmp_path, 1949.0, guard)
    s.start()
    outcome = s.run_stage(stage(), ctx)
    assert outcome.state == sched.STATE_REFUSED
    assert CALLS == []
    s2 = make_scheduler(tmp_path / "b", 1950.0, guard)
    s2.start()
    assert s2.run_stage(stage(), ctx).state == sched.STATE_COMPLETE
    assert CALLS == ["STUB"]


def test_a_refused_entry_condition_skips_with_the_fallback(tmp_path):
    CALLS.clear()
    ctx, guard = context(tmp_path)
    s = make_scheduler(tmp_path, 1e6, guard)
    s.start()
    outcome = s.run_stage(stage(entry=f"{THIS}:refusing_entry"), ctx)
    assert outcome.state == sched.STATE_SKIPPED
    assert outcome.fallback == ("do nothing",)
    assert CALLS == []


def test_a_failing_stage_is_recorded_and_never_swallowed(tmp_path):
    ctx, guard = context(tmp_path)
    s = make_scheduler(tmp_path, 1e6, guard)
    s.start()
    outcome = s.run_stage(stage(work="failing_work",
                                fallback_work=("predeclared",)), ctx)
    assert outcome.state == sched.STATE_FAILED
    assert "stub failure" in outcome.error
    failure = [r for r in s.read_journal() if r["event"] == "STAGE_FAILED"][0]
    assert "traceback" in failure and failure["fallback"] == ["predeclared"]


def test_projection_uses_measured_throughput_only(tmp_path):
    ctx, guard = context(tmp_path, spu=2.0, eval_per_episode=0.5)
    s = make_scheduler(tmp_path, 1e6, guard)
    s.start()
    outcome = s.run_stage(stage(eval_episodes=10), ctx)
    assert outcome.projection["train_seconds"] == 1200.0
    assert outcome.projection["eval_seconds"] == 5.0
    # an unmeasured scope must block, never estimate
    ctx.scope = FULL_MODEL_MODE
    blocked = s.run_stage(stage(stage_id="STUB2"), ctx)
    assert blocked.state == sched.STATE_BLOCKED


def test_core_plan_uses_the_frozen_rule_and_journals_it(tmp_path):
    ctx, guard = context(tmp_path, spu=0.1, eval_per_episode=0.0)
    ctx.eval_episode_cap = 0
    s = make_scheduler(tmp_path, 1e6, guard)
    s.start()
    plan = s.plan_core(ctx)
    assert plan.scope == PEFT_MODE and plan.updates == 4800
    assert ctx.updates == 4800
    assert any(r["event"] == "CORE_PLAN" for r in s.read_journal())


def test_checkpoint_cadence_is_frozen_and_asserted(tmp_path):
    _, guard = context(tmp_path)
    s = make_scheduler(tmp_path, 1000.0, guard)
    assert s.checkpoint_cadence() == (600.0, 200)

    class Cfg:
        arm_id = "FL3"
        checkpoint_every_seconds = 600.0
        checkpoint_every_steps = 200

    s.assert_checkpoint_cadence(Cfg())
    Cfg.checkpoint_every_steps = 5000
    with pytest.raises(sched.SchedulerError):
        s.assert_checkpoint_cadence(Cfg())


def test_heartbeat_hook_journals_frequently(tmp_path):
    _, guard = context(tmp_path)
    s = make_scheduler(tmp_path, 1000.0, guard)
    s.start()
    hook = s.heartbeat_hook("FL3", every=2)
    for step in range(1, 7):
        hook({"step": step, "loss": 1.0 / step})
    beats = [r for r in s.read_journal() if r["event"] == "STAGE_HEARTBEAT"]
    assert [b["step"] for b in beats] == [2, 4, 6]


def test_journal_is_append_only_and_machine_readable(tmp_path):
    _, guard = context(tmp_path)
    s = make_scheduler(tmp_path, 1000.0, guard)
    s.start()
    s.journal("CUSTOM", {"x": 1})
    lines = open(s.journal_path, encoding="utf-8").read().splitlines()
    assert len(lines) == 2
    for index, line in enumerate(lines):
        record = json.loads(line)
        assert record["schema"] == sched.JOURNAL_SCHEMA
        assert record["index"] == index


def test_summary_records_the_reserve_and_states(tmp_path):
    CALLS.clear()
    ctx, guard = context(tmp_path, spu=0.01, eval_per_episode=0.0)
    s = make_scheduler(tmp_path, 1e6, guard)
    s.start()
    s.run_stage(stage(stage_id="A"), ctx)
    s.run_stage(stage(stage_id="B", work="failing_work"), ctx)
    summary = s.summary()
    assert summary["states"] == {"A": sched.STATE_COMPLETE,
                                 "B": sched.STATE_FAILED}
    assert summary["final_transfer_reserve_seconds"] == 1200.0
    assert summary["safety_factor"] == 1.25
