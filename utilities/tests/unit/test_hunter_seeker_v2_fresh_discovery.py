from __future__ import annotations

import sys
from pathlib import Path

from hunter_seeker_v2.contracts import RuntimeMode
from hunter_seeker_v2.policy import RiskArbiter

MANUAL_ROOT = Path(__file__).resolve().parents[1] / "manual"
if str(MANUAL_ROOT) not in sys.path:
    sys.path.insert(0, str(MANUAL_ROOT))

import run_hs_v2_fresh_discovery_v1 as experiment  # noqa: E402


def test_fresh_config_is_autonomous_and_goal_promotion_switch_is_explicit() -> None:
    enabled = experiment.fresh_config()
    disabled = experiment.fresh_config(goal_promotion_enabled=False)

    assert enabled.runtime_mode is RuntimeMode.AUTONOMOUS
    assert enabled.enable_online_learning is True
    assert enabled.enable_executable_models is False
    assert enabled.hypotheses.enable_goal_contrast is True
    assert enabled.hypotheses.enable_completion_motion_reach is True
    assert disabled.hypotheses.enable_goal_contrast is False
    assert disabled.hypotheses.enable_completion_motion_reach is False


def test_fresh_agent_has_no_teacher_route_checkpoint_or_durable_state() -> None:
    agent = experiment.build_fresh_agent()

    audit = experiment.freshness_audit(
        agent,
        arm="fresh_autonomous",
        checkpoint_loaded=False,
        route_used=False,
        forced_action_count=0,
    )

    assert agent.runtime_mode is RuntimeMode.AUTONOMOUS
    assert agent.teacher is None
    assert type(agent.arbiter) is RiskArbiter
    assert len(agent.graph) == 0
    assert agent.graph.edge_count == 0
    assert len(agent.evidence) == 0
    assert audit["fresh_isolation_pass"] is True


def test_freshness_audit_rejects_each_leakage_switch() -> None:
    agent = experiment.build_fresh_agent()

    assert experiment.freshness_audit(
        agent,
        arm="fresh_autonomous",
        checkpoint_loaded=True,
        route_used=False,
        forced_action_count=0,
    )["fresh_isolation_pass"] is False
    assert experiment.freshness_audit(
        agent,
        arm="fresh_autonomous",
        checkpoint_loaded=False,
        route_used=True,
        forced_action_count=1,
    )["fresh_isolation_pass"] is False
    assert experiment.freshness_audit(
        agent,
        arm="fresh_autonomous",
        checkpoint_loaded=False,
        route_used=False,
        forced_action_count=0,
    )["compatibility_mode"] is False
