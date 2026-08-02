from __future__ import annotations

import numpy as np
import pytest

from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import (
    Action,
    AgentConfig,
    BoundaryKind,
    Candidate,
    ModelConfig,
    Observation,
    Outcome,
    PolicyConfig,
    Prediction,
    ScoreTerm,
    SearchConfig,
)
from hunter_seeker_v2.diagnostics import Diagnostics, candidate_trace
from hunter_seeker_v2.policy import effective_risk


def _config(*, epsilon: float = 0.0) -> AgentConfig:
    return AgentConfig(
        seed=73,
        search=SearchConfig(beam_width=2, horizon=1),
        policy=PolicyConfig(exploration_epsilon=epsilon),
        model=ModelConfig(ensemble_size=2, latent_dim=12),
    )


def _observation(value: int) -> Observation:
    frame = np.zeros((3, 4), dtype=np.uint8)
    frame[1, 1:3] = value
    return Observation(
        frame=frame,
        available_actions=(0, 1),
        task_id="invariant-task",
    )


def test_failed_act_diagnostics_is_atomic_and_exactly_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(epsilon=1.0)
    agent = CompactHunterSeeker(
        config=config,
        safe_action_provider=lambda _observation: (0, 1),
        agent_id="same-agent",
    )
    control = CompactHunterSeeker(
        config=config,
        safe_action_provider=lambda _observation: (0, 1),
        agent_id="same-agent",
    )
    observation = _observation(2)
    original_record = Diagnostics.record_decision

    def append_then_fail(self: Diagnostics, *args, **kwargs):
        original_record(self, *args, **kwargs)
        raise RuntimeError("injected decision trace failure")

    monkeypatch.setattr(Diagnostics, "record_decision", append_then_fail)
    with pytest.raises(RuntimeError, match="decision trace failure"):
        agent.act(observation)

    assert agent.pending_decision is None
    assert agent._decision_counter == 0
    assert agent.diagnostics.decisions == ()

    monkeypatch.setattr(Diagnostics, "record_decision", original_record)
    retry = agent.act(observation)
    expected = control.act(observation)

    assert retry.action == expected.action
    assert retry.decision_id == expected.decision_id
    assert len(agent.diagnostics.decisions) == 1


def test_diagnostic_candidate_risk_matches_the_final_arbiter() -> None:
    candidate = Candidate(
        action=Action(0),
        prediction=Prediction(
            change_probability=0.0,
            progress=0.0,
            value=0.0,
            hazard=0.0,
            terminal=0.0,
            uncertainty=0.0,
            latent_delta=np.zeros(1, dtype=np.float32),
            object_delta=np.zeros(1, dtype=np.float32),
        ),
        terms=(
            ScoreTerm(
                "risk:evidence_negative",
                1.0,
                "safety",
                influences_score=False,
            ),
        ),
    )

    assert effective_risk(candidate) == pytest.approx(1.0)
    assert candidate_trace(candidate).risk == pytest.approx(
        effective_risk(candidate)
    )


def test_graph_node_diagnostics_cannot_mutate_durable_graph_state() -> None:
    agent = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
    )
    before = _observation(2)
    after = _observation(3)
    decision = agent.act(before)
    agent.observe(decision, after, Outcome())

    durable_before = agent.graph.export_state()
    view = agent.graph.node(before.state_id)
    assert view is not None
    view.visits = 999
    view.edges[decision.action.key].visits = 999
    if view.latent is not None:
        view.latent[:] = 999.0

    assert agent.graph.export_state() == durable_before


def test_graph_terminal_fact_is_monotone_for_an_ambiguous_visible_state() -> None:
    agent = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
    )
    fatal_predecessor = _observation(2)
    ambiguous = _observation(3)
    decision = agent.act(fatal_predecessor)
    agent.observe(
        decision,
        ambiguous,
        Outcome(boundary=BoundaryKind.DEATH, hazard=1.0),
    )
    assert agent.graph.is_terminal(ambiguous.state_id)

    agent.end_run()
    safe_predecessor = _observation(4)
    agent.begin_run("invariant-task", safe_predecessor)
    decision = agent.act(safe_predecessor)
    agent.observe(decision, ambiguous, Outcome())

    assert agent.graph.is_terminal(ambiguous.state_id)
