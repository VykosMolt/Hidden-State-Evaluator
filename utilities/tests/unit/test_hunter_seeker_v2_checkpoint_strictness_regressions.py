from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import pytest

from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import (
    AgentConfig,
    ModelConfig,
    Observation,
    Outcome,
    PolicyConfig,
    SearchConfig,
    readonly_array,
)
from hunter_seeker_v2.hypotheses import Hypothesis, RelationalHypothesisEngine


def _config() -> AgentConfig:
    return AgentConfig(
        seed=41,
        search=SearchConfig(beam_width=2, horizon=1),
        policy=PolicyConfig(exploration_epsilon=0.0),
        model=ModelConfig(ensemble_size=2, latent_dim=12),
    )


def _agent(*, agent_id: str = "agent") -> CompactHunterSeeker:
    return CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
        agent_id=agent_id,
    )


def _observation(value: int, *, progress: float = 0.0) -> Observation:
    return Observation(
        frame=np.asarray([[0, value], [value, 0]], dtype=np.int64),
        available_actions=(0,),
        task_id="task",
        progress=progress,
    )


def _committed_state() -> dict[str, Any]:
    agent = _agent()
    before = _observation(1)
    after = _observation(2, progress=1.0)
    agent.begin_run("task", before)
    decision = agent.act(before)
    agent.observe(
        decision,
        after,
        Outcome(reward=1.0, progress_delta=1.0),
    )
    agent.end_run()
    return agent.state_dict()


def _runtime_action_float(state: dict[str, Any]) -> None:
    state["runtime"]["current_snapshot"]["observation"][
        "available_actions"
    ][0] = 0.9


def _runtime_topology_float(state: dict[str, Any]) -> None:
    state["runtime"]["current_snapshot"]["topology"]["component_count"] = 1.9


def _runtime_event_float(state: dict[str, Any]) -> None:
    state["runtime"]["current_snapshot"]["events"][0][
        "subject_track_id"
    ] = 0.9


def _runtime_nonfinite_fraction(state: dict[str, Any]) -> None:
    state["runtime"]["current_snapshot"]["topology"][
        "frontier_fraction"
    ] = float("nan")


def _runtime_fractional_counter(state: dict[str, Any]) -> None:
    state["runtime"]["decision_counter"] = 1.9


def _learning_string_counter(state: dict[str, Any]) -> None:
    state["learning"]["real_updates"] = "1"


def _numeric_agent_id(state: dict[str, Any]) -> None:
    state["agent_id"] = 7


def _malformed_student_policy(state: dict[str, Any]) -> None:
    state["models"]["student_policy"] = 7


@pytest.mark.parametrize(
    "mutate",
    (
        _runtime_action_float,
        _runtime_topology_float,
        _runtime_event_float,
        _runtime_nonfinite_fraction,
        _runtime_fractional_counter,
        _learning_string_counter,
        _numeric_agent_id,
        _malformed_student_policy,
    ),
)
def test_full_checkpoint_rejects_coercions_atomically(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    state = copy.deepcopy(_committed_state())
    mutate(state)
    destination = _agent(agent_id="destination")
    identities = {
        name: id(getattr(destination, name))
        for name in (
            "dynamics",
            "prior",
            "affordances",
            "graph",
            "hypotheses",
            "learner",
            "diagnostics",
            "search_engine",
        )
    }

    with pytest.raises(ValueError):
        destination.load_state_dict(state)

    assert destination.agent_id == "destination"
    assert destination._task_id is None
    assert {
        name: id(getattr(destination, name))
        for name in identities
    } == identities


def _replay_action_float(state: dict[str, Any]) -> None:
    state["learning"]["items"][0]["transition"]["action"][0] = 0.9


def _replay_string_terminated(state: dict[str, Any]) -> None:
    state["learning"]["items"][0]["transition"]["outcome"][
        "terminated"
    ] = "false"


def _replay_string_frame_changed(state: dict[str, Any]) -> None:
    state["learning"]["items"][0]["transition"]["frame_changed"] = "false"


def _replay_string_teacher(state: dict[str, Any]) -> None:
    state["learning"]["items"][0]["teacher"] = "false"


@pytest.mark.parametrize(
    "mutate",
    (
        _replay_action_float,
        _replay_string_terminated,
        _replay_string_frame_changed,
        _replay_string_teacher,
    ),
)
def test_replay_checkpoint_rejects_numeric_and_boolean_coercions(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    state = copy.deepcopy(_committed_state())
    mutate(state)

    with pytest.raises(ValueError):
        _agent().load_state_dict(state)


def _hypothesis_state() -> dict[str, Any]:
    engine = RelationalHypothesisEngine()
    hypothesis = Hypothesis(
        hypothesis_id="reach|1|2",
        kind="reach",
        region_a=(0, 0, 0, 0),
        verified=True,
        initial_potential=0.5,
        value_a=1,
        value_b=2,
        origin="completion_motion",
    )
    hypothesis.deltas[(0, "target")] = [-0.25, 2.0]
    engine._by_scope[("task", 1)] = [hypothesis]
    frame = readonly_array(np.asarray([[0, 1], [2, 0]], dtype=np.int64))
    engine._initial_frames[("task", 1)] = frame
    engine._state_frames["state"] = frame
    engine._count_traces[("task", 1)] = [{0: 2, 1: 1, 2: 1}]
    engine._potential_cache[("reach|1|2", "state")] = 0.5
    engine.observed_transitions = 2
    engine.promotions = 1
    engine.goal_proposals = 1
    return engine.state_dict()


def _hypothesis_string_bool(state: dict[str, Any]) -> None:
    state["scopes"]["task␟1"][0]["verified"] = "false"


def _hypothesis_clipped_potential(state: dict[str, Any]) -> None:
    state["potential_cache"][0]["potential"] = 2.5


def _hypothesis_fractional_counter(state: dict[str, Any]) -> None:
    state["observed_transitions"] = 1.9


def _hypothesis_malformed_frame_row(state: dict[str, Any]) -> None:
    state["initial_frames"].append(7)


def _hypothesis_fractional_delta_count(state: dict[str, Any]) -> None:
    state["scopes"]["task␟1"][0]["deltas"]["0␟target"][1] = 1.5


@pytest.mark.parametrize(
    "mutate",
    (
        _hypothesis_string_bool,
        _hypothesis_clipped_potential,
        _hypothesis_fractional_counter,
        _hypothesis_malformed_frame_row,
        _hypothesis_fractional_delta_count,
    ),
)
def test_hypothesis_checkpoint_rejects_silent_repairs(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    state = copy.deepcopy(_hypothesis_state())
    mutate(state)

    with pytest.raises(ValueError):
        RelationalHypothesisEngine.from_state(state)


def test_strict_hypothesis_state_roundtrips_completion_motion() -> None:
    state = _hypothesis_state()

    restored = RelationalHypothesisEngine.from_state(state)

    assert restored.state_dict() == state
    hypothesis = restored.active("task", 1)[0]
    assert hypothesis.origin == "completion_motion"
    assert hypothesis.verified is True
