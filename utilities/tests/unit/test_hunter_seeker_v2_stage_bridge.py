from __future__ import annotations

import numpy as np

from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import (
    AgentConfig,
    BoundaryKind,
    ModelConfig,
    Observation,
    Outcome,
    PolicyConfig,
    SearchConfig,
)


def _agent() -> CompactHunterSeeker:
    return CompactHunterSeeker(
        config=AgentConfig(
            seed=73,
            search=SearchConfig(
                beam_width=2,
                horizon=1,
                max_click_candidates=4,
            ),
            policy=PolicyConfig(exploration_epsilon=0.0),
            model=ModelConfig(ensemble_size=2, latent_dim=12),
        ),
        safe_action_provider=lambda _observation: (0,),
    )


def _observation(
    frame: np.ndarray,
    *,
    stage: int,
    progress: float,
) -> Observation:
    return Observation(
        frame=frame,
        available_actions=(0,),
        task_id="task",
        stage=stage,
        progress=progress,
    )


def test_delayed_stage_board_load_is_logged_but_not_learned_as_dynamics() -> None:
    # The completion action changes only 2% of the frame, matching the
    # delayed-render pattern seen in ls20/tr87.  The next observation then
    # performs the large successor-board load.
    initial = np.zeros((20, 20), dtype=np.uint8)
    initial[2, :10] = 5
    solved = initial.copy()
    solved[2, :8] = 0
    next_board = np.full((20, 20), 7, dtype=np.uint8)
    next_board[1:4, 1:4] = 3

    agent = _agent()
    before = _observation(initial, stage=1, progress=0.0)
    completion = _observation(solved, stage=2, progress=1.0)
    agent.begin_run("task", before)
    completion_decision = agent.act(before)
    first = agent.observe(
        completion_decision,
        completion,
        Outcome(
            reward=1.0,
            progress_delta=1.0,
            boundary=BoundaryKind.LEVEL_COMPLETED,
        ),
    )

    assert first.outcome.completed
    assert agent._awaiting_visual_reset
    assert len(agent.evidence) == 1
    assert agent.graph.edge_count == 1
    assert len(agent.learner.replay) == 1
    old_stage_goals = agent.hypotheses.active("task", 1)
    assert any(
        row.kind == "count_at_most"
        and row.value_a == 5
        and row.target_count == 2
        and row.verified
        for row in old_stage_goals
    )
    assert ("task", 2) not in agent.hypotheses._initial_frames

    bridge_decision = agent.act(completion)
    assert bridge_decision.metadata["selection_method"] == "boundary_visual_reset"
    dynamics_updates = agent.dynamics.update_count
    student_updates = agent.student_policy.online_updates
    second = agent.observe(
        bridge_decision,
        _observation(next_board, stage=2, progress=1.0),
        Outcome(),
    )

    assert second.outcome.metadata["boundary_visual_reset"] is True
    assert agent.transition_count == 2
    assert agent.run_transition_count == 2
    assert len(agent.diagnostics.transitions) == 2
    assert len(agent.evidence) == 1
    assert agent.graph.edge_count == 1
    assert len(agent.learner.replay) == 1
    assert agent.dynamics.update_count == dynamics_updates
    assert agent.student_policy.online_updates == student_updates
    assert not agent._awaiting_visual_reset
    assert np.array_equal(agent.current_snapshot.observation.frame, next_board)
    assert np.array_equal(
        agent.hypotheses._initial_frames[("task", 2)],
        next_board,
    )
    active_exogenous = agent.exogenous.state_dict()["active"]
    assert active_exogenous["stage"] == 2
    assert active_exogenous["steps"] == 0
    assert active_exogenous["events"] == []


def test_immediate_stage_swap_is_not_misclassified_as_visual_bridge() -> None:
    initial = np.zeros((20, 20), dtype=np.uint8)
    initial[2:8, 2:8] = 2
    next_board = np.full((20, 20), 7, dtype=np.uint8)
    next_board[10:14, 10:14] = 3

    agent = _agent()
    before = _observation(initial, stage=1, progress=0.0)
    successor = _observation(next_board, stage=2, progress=1.0)
    agent.begin_run("task", before)
    decision = agent.act(before)
    transition = agent.observe(
        decision,
        successor,
        Outcome(
            reward=1.0,
            progress_delta=1.0,
            boundary=BoundaryKind.LEVEL_COMPLETED,
        ),
    )

    assert transition.outcome.completed
    assert not agent._awaiting_visual_reset
    assert np.array_equal(agent.current_snapshot.observation.frame, next_board)
    assert np.array_equal(
        agent.hypotheses._initial_frames[("task", 2)],
        next_board,
    )
    # Cross-stage count/region differences are not old-stage goal evidence.
    assert all(
        row.origin != "goal_contrast"
        for row in agent.hypotheses.active("task", 1)
    )
    next_decision = agent.act(successor)
    assert next_decision.metadata["selection_method"] != "boundary_visual_reset"
