from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hunter_seeker_v2.adapters import (
    AdapterError,
    ArcActionAdapter,
    ArcObservationAdapter,
    ArcOutcomeAdapter,
    CategoricalActionAdapter,
    CategoricalObservationAdapter,
    MockActionAdapter,
    MockObservationAdapter,
    MockOutcomeAdapter,
    ObservationUnavailable,
)
from hunter_seeker_v2.contracts import (
    Action,
    BoundaryKind,
    Decision,
    Observation,
    Outcome,
    Representation,
    RuntimeMode,
    Topology,
    Transition,
    WorldSnapshot,
)
from hunter_seeker_v2.diagnostics import Diagnostics
from hunter_seeker_v2.run_arc import (
    _normalize_environment_result,
    run_episode,
)


class GameAction(Enum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION5 = 5
    ACTION6 = 6
    ACTION7 = 7


def _arc_raw(
    value: int,
    *,
    shape: tuple[int, int] = (3, 5),
    levels: int = 0,
    state: str = "ACTIVE",
    frames: bool = True,
) -> SimpleNamespace:
    frame = [np.full(shape, value, dtype=np.uint8)] if frames else []
    return SimpleNamespace(
        frame=frame,
        available_actions=[1, 2],
        levels_completed=levels,
        state=state,
    )


def test_arc_observation_accepts_odd_shapes_and_reuses_animation_gap_frame() -> None:
    adapter = ArcObservationAdapter()
    raw = _arc_raw(3, shape=(3, 5), levels=2)
    first = adapter.observation(raw, task_id="odd")

    raw.frame[0][0, 0] = 9
    gap = adapter.observation(
        _arc_raw(0, shape=(3, 5), levels=2, frames=False),
        task_id="odd",
    )

    assert first.frame.shape == (3, 5)
    assert first.frame[0, 0] == 3
    assert not first.frame.flags.writeable
    assert first.stage == 3
    assert first.progress == 2.0
    assert gap.metadata["frame_available"] is False
    np.testing.assert_array_equal(gap.frame, first.frame)

    adapter.clear("odd")
    with pytest.raises(ObservationUnavailable):
        adapter.observation(
            _arc_raw(0, shape=(3, 5), frames=False),
            task_id="odd",
        )


def test_generic_observation_validates_labels_actions_and_dimensions() -> None:
    adapter = CategoricalObservationAdapter(
        n_values=5,
        pad_value=5,
        n_actions=3,
        default_actions=(0, 2),
    )
    observation = adapter.observation(
        {"grid": np.asarray([[0, 1, 5], [2, 3, 4]]), "progress": 1},
        task_id="generic",
    )

    assert observation.frame.shape == (2, 3)
    assert observation.available_actions == (0, 2)
    assert observation.stage == 2

    with pytest.raises(AdapterError, match="shape"):
        adapter.observation(
            {"grid": np.zeros((2, 2, 2), dtype=np.uint8)},
            task_id="bad",
        )
    with pytest.raises(AdapterError, match="exceeds"):
        adapter.observation(
            {"grid": np.asarray([[6]], dtype=np.uint8)},
            task_id="bad",
        )
    separated_pad = CategoricalObservationAdapter(
        n_values=5,
        pad_value=9,
        n_actions=1,
    )
    with pytest.raises(AdapterError, match="not pad value"):
        separated_pad.observation(
            {"grid": np.asarray([[7]], dtype=np.uint8)},
            task_id="bad",
        )
    with pytest.raises(AdapterError, match="outside"):
        adapter.observation(
            {
                "grid": np.asarray([[1]], dtype=np.uint8),
                "available_actions": [3],
            },
            task_id="bad",
        )
    with pytest.raises(AdapterError, match="finite"):
        adapter.observation(
            {"grid": np.asarray([[1]], dtype=np.uint8), "progress": np.nan},
            task_id="bad",
        )


def test_arc_action_decode_preserves_enum_and_click_payload() -> None:
    adapter = ArcActionAdapter()

    click, kwargs = adapter.decode(Action(6, x=4, y=2), GameAction)
    move, move_kwargs = adapter.decode(Action(1), GameAction)
    bootstrap, bootstrap_kwargs = adapter.bootstrap(GameAction)

    assert click is GameAction.ACTION6
    assert kwargs == {"data": {"x": 4, "y": 2}}
    assert move is GameAction.ACTION1
    assert move_kwargs == {}
    assert bootstrap is GameAction.RESET
    assert bootstrap_kwargs == {}
    assert adapter.click_action_index() == 6


def test_generic_and_mock_action_spaces_do_not_assume_arc_clicks() -> None:
    mock = MockActionAdapter()
    env_action, kwargs = mock.decode(Action(2, x=9, y=8))
    assert env_action == 2
    assert kwargs == {}
    assert mock.click_action_index() is None

    positional = CategoricalActionAdapter(
        action_values=("wait", "point"),
        action_names=("WAIT", "POINT"),
        positional_action_index=1,
        position_kwargs=lambda action: {"position": (action.x, action.y)},
    )
    env_action, kwargs = positional.decode(Action(1, x=7, y=6))
    assert env_action == "point"
    assert kwargs == {"position": (7, 6)}
    assert positional.click_action_index() == 1
    observation = Observation(
        frame=np.zeros((2, 3), dtype=np.uint8),
        available_actions=(1,),
        task_id="generic",
    )
    assert tuple(positional.safe_action_indices(observation)) == (1,)


def test_outcomes_are_incremental_and_terminal_boundaries_are_authoritative() -> None:
    adapter = ArcOutcomeAdapter()
    observations = ArcObservationAdapter()
    before_raw = _arc_raw(0, levels=0)
    level_raw = _arc_raw(1, levels=1)
    death_raw = _arc_raw(2, levels=1, state="GAME_OVER")
    before = observations.observation(before_raw, task_id="arc")
    level = observations.observation(level_raw, task_id="arc")
    death = observations.observation(death_raw, task_id="arc")

    level_outcome = adapter.outcome(
        before,
        level,
        raw_after=level_raw,
    )
    death_outcome = adapter.outcome(
        level,
        death,
        raw_after=death_raw,
    )
    explicit_death = adapter.outcome(
        level,
        death,
        raw_after=_arc_raw(2, levels=1, state="ACTIVE"),
        info={"boundary": BoundaryKind.DEATH},
    )

    assert level_outcome.boundary is BoundaryKind.LEVEL_COMPLETED
    assert level_outcome.progress_delta == 1.0
    assert level_outcome.reward == 115.0
    assert not level_outcome.terminated
    assert death_outcome.boundary is BoundaryKind.DEATH
    assert death_outcome.terminated
    assert death_outcome.hazard == 1.0
    assert explicit_death.terminated


def _snapshot(observation: Observation, step: int) -> WorldSnapshot:
    return WorldSnapshot(
        observation=observation,
        objects=(),
        events=(),
        topology=Topology(),
        representation=Representation(
            global_vector=np.zeros(2, dtype=np.float32),
            spatial=np.zeros((1, *observation.frame.shape), dtype=np.float32),
        ),
        step=step,
    )


class _SpyAgent:
    agent_id = "spy"

    def __init__(self, events: list[str], actions: list[int]) -> None:
        self.events = events
        self.actions = list(actions)
        self.snapshot: WorldSnapshot | None = None
        self.step = 0

    def begin_run(
        self,
        task_id: str,
        observation: Observation | None = None,
    ) -> None:
        assert observation is not None
        assert task_id == observation.task_id
        self.events.append("begin")
        self.snapshot = _snapshot(observation, 0)

    def act(self, observation: Observation) -> Decision:
        assert self.snapshot is not None
        index = self.actions.pop(0)
        self.events.append(f"act:{index}")
        return Decision(
            decision_id=f"d{self.step}",
            agent_id=self.agent_id,
            snapshot_id=self.snapshot.state_id,
            action=Action(index),
            score=0.0,
            candidates=(),
            mode=RuntimeMode.AUTONOMOUS,
            step=self.step,
        )

    def observe(
        self,
        decision: Decision,
        next_observation: Observation,
        outcome: Outcome,
    ) -> Transition:
        assert self.snapshot is not None
        self.events.append(f"observe:{outcome.boundary.value}")
        transition = Transition(
            transition_id=f"t{self.step}",
            decision_id=decision.decision_id,
            task_id=next_observation.task_id,
            stage=self.snapshot.observation.stage,
            step=self.step,
            before=self.snapshot,
            action=decision.action,
            after_observation=next_observation,
            outcome=outcome,
            frame_changed=not np.array_equal(
                self.snapshot.observation.frame,
                next_observation.frame,
            ),
            after_state_id=next_observation.state_id,
        )
        self.step += 1
        self.snapshot = _snapshot(next_observation, self.step)
        return transition

    def on_level_complete(self, level: int) -> None:
        self.events.append(f"level:{level}")

    def on_game_over(self) -> None:
        self.events.append("game_over")

    def on_boundary(self, transition: Transition) -> None:
        self.events.append(f"boundary:{transition.outcome.boundary.value}")

    def end_run(self, outcome: Outcome | None = None) -> None:
        assert outcome is not None
        self.events.append(f"end:{outcome.boundary.value}")


class _ArcEnvironment:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.actions: list[tuple[GameAction, dict[str, Any]]] = []

    def step(self, action: GameAction, **kwargs: Any) -> SimpleNamespace:
        self.actions.append((action, kwargs))
        self.events.append(f"env:{action.value}")
        if action is GameAction.RESET:
            return _arc_raw(0, levels=0)
        if action is GameAction.ACTION1:
            return _arc_raw(1, levels=1)
        if action is GameAction.ACTION2:
            return _arc_raw(2, levels=1, state="GAME_OVER")
        raise AssertionError(f"unexpected action: {action}")


def test_runner_commits_each_action_before_boundary_callbacks_and_breaks() -> None:
    events: list[str] = []
    environment = _ArcEnvironment(events)
    agent = _SpyAgent(events, [1, 2])

    result = run_episode(
        environment,
        agent,
        task_id="arc",
        observation_adapter=ArcObservationAdapter(),
        action_adapter=ArcActionAdapter(),
        outcome_adapter=ArcOutcomeAdapter(),
        action_space=GameAction,
        # Death on the last allowed step must remain authoritative over the
        # runner's coincident local budget boundary.
        max_steps=2,
    )

    assert result.steps == 2
    assert len(result.transitions) == 2
    assert result.transitions[0].action.index == 1
    assert result.transitions[0].outcome.boundary is BoundaryKind.LEVEL_COMPLETED
    assert result.transitions[1].action.index == 2
    assert result.transitions[1].outcome.boundary is BoundaryKind.DEATH
    assert result.final_outcome.boundary is BoundaryKind.DEATH
    assert events.index("observe:level_completed") < events.index("level:1")
    assert events.index("observe:level_completed") < events.index(
        "boundary:level_completed"
    )
    assert events.index("observe:death") < events.index("game_over")
    assert events.index("observe:death") < events.index("boundary:death")
    assert events[-1] == "end:death"


class _NeverDoneEnvironment:
    def __init__(self) -> None:
        self.value = 0

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {
                "grid": np.zeros((2, 3), dtype=np.uint8),
                "available_actions": [0, 1],
                "progress": 0,
                "stage": 1,
                "state": "ACTIVE",
            },
            {},
        )

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        del action
        self.value += 1
        return (
            {
                "grid": np.full((2, 3), self.value, dtype=np.uint8),
                "available_actions": [0, 1],
                "progress": 0,
                "stage": 1,
                "state": "ACTIVE",
            },
            2.5,
            False,
            False,
            {"environment_step": self.value},
        )


def test_runner_commits_local_time_limit_on_final_causal_transition() -> None:
    events: list[str] = []
    result = run_episode(
        _NeverDoneEnvironment(),
        _SpyAgent(events, [0, 1]),
        task_id="mock",
        observation_adapter=MockObservationAdapter(n_values=8, n_actions=2),
        action_adapter=MockActionAdapter(n_actions=2),
        outcome_adapter=MockOutcomeAdapter(),
        max_steps=2,
        bootstrap=False,
    )

    assert result.steps == 2
    assert len(result.transitions) == 2
    assert result.transitions[0].outcome.boundary is BoundaryKind.NONE
    final_transition = result.transitions[-1]
    assert final_transition.outcome is result.final_outcome
    assert final_transition.outcome.boundary is BoundaryKind.TIME_LIMIT
    assert final_transition.outcome.truncated is True
    assert final_transition.outcome.terminated is False
    assert final_transition.outcome.reward == 2.5
    assert final_transition.outcome.metadata == {
        "state": "ACTIVE",
        "frame_changed": True,
        "level_completed": False,
        "progress_before": 0.0,
        "progress_after": 0.0,
        "max_steps": 2,
    }
    assert events.count("observe:time_limit") == 1
    assert events.index("observe:time_limit") < events.index("boundary:time_limit")
    assert events[-1] == "end:time_limit"


@dataclass
class _MockEnvironment:
    value: int = 0

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {
                "grid": np.zeros((2, 3), dtype=np.uint8),
                "available_actions": [0, 1],
                "progress": 0,
                "stage": 1,
                "state": "ACTIVE",
            },
            {},
        )

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.value += 1
        return (
            {
                "grid": np.full((2, 3), self.value, dtype=np.uint8),
                "available_actions": [0, 1],
                "progress": 0,
                "stage": 1,
                "state": "ACTIVE",
            },
            2.5,
            False,
            self.value >= 1,
            {},
        )


def test_runner_supports_clickless_gymnasium_style_environment() -> None:
    events: list[str] = []
    result = run_episode(
        _MockEnvironment(),
        _SpyAgent(events, [1]),
        task_id="mock",
        observation_adapter=MockObservationAdapter(n_values=8, n_actions=2),
        action_adapter=MockActionAdapter(n_actions=2),
        outcome_adapter=MockOutcomeAdapter(),
        max_steps=4,
        bootstrap=False,
    )

    assert result.steps == 1
    assert result.final_observation.frame.shape == (2, 3)
    assert result.final_outcome.reward == 2.5
    assert result.final_outcome.truncated
    assert result.final_outcome.boundary is BoundaryKind.TIME_LIMIT


class _BrokenEnvironment(_MockEnvironment):
    def step(self, action: int) -> dict[str, Any]:
        raise OSError(f"failed action {action}")


def test_runner_commits_environment_error_before_ending_run() -> None:
    events: list[str] = []
    result = run_episode(
        _BrokenEnvironment(),
        _SpyAgent(events, [1]),
        task_id="broken",
        observation_adapter=MockObservationAdapter(n_values=8, n_actions=2),
        action_adapter=MockActionAdapter(n_actions=2),
        outcome_adapter=MockOutcomeAdapter(),
        max_steps=4,
        bootstrap=False,
    )

    assert result.steps == 1
    assert result.final_outcome.boundary is BoundaryKind.ENVIRONMENT_ERROR
    assert result.final_outcome.terminated
    assert result.final_outcome.metadata["exception_type"] == "OSError"
    assert events.index("observe:environment_error") < events.index(
        "boundary:environment_error"
    )
    assert events[-1] == "end:environment_error"


def test_rejected_observation_does_not_mutate_fallback_frame_cache() -> None:
    adapter = MockObservationAdapter(n_values=8, n_actions=2)
    first = adapter.observation(
        {"grid": [[1]], "available_actions": [0]},
        task_id="task",
    )

    with pytest.raises(AdapterError, match="outside"):
        adapter.observation(
            {"grid": [[2]], "available_actions": [99]},
            task_id="task",
        )

    fallback = adapter.observation(
        {"grid": None, "available_actions": [0]},
        task_id="task",
    )
    np.testing.assert_array_equal(fallback.frame, first.frame)


@pytest.mark.parametrize("invalid_index", [1.0, 1.9, True, "1"])
def test_action_indices_require_integer_values(invalid_index: Any) -> None:
    observation_adapter = MockObservationAdapter(n_values=8, n_actions=2)
    with pytest.raises(AdapterError, match="integer index"):
        observation_adapter.observation(
            {"grid": [[1]], "available_actions": [invalid_index]},
            task_id="task",
        )

    # Invalid values cannot enter the canonical Action contract.  The raw
    # observation path above independently retains its AdapterError boundary.
    with pytest.raises(ValueError, match="action index must be an integer"):
        Action(invalid_index)


def test_stage_and_outcome_flags_use_strict_types() -> None:
    observations = MockObservationAdapter(n_values=8, n_actions=2)
    with pytest.raises(AdapterError, match="integer"):
        observations.observation(
            {"grid": [[1]], "available_actions": [0], "stage": 1.5},
            task_id="task",
        )

    before = observations.observation(
        {"grid": [[1]], "available_actions": [0]},
        task_id="task",
    )
    after_raw = {
        "grid": [[1]],
        "available_actions": [0],
        "terminated": "false",
    }
    after = observations.observation(after_raw, task_id="task")
    outcomes = MockOutcomeAdapter()
    with pytest.raises(AdapterError, match="terminated must be a bool"):
        outcomes.outcome(before, after, raw_after=after_raw)
    with pytest.raises(AdapterError, match="truncated must be a bool"):
        outcomes.outcome(
            before,
            after,
            raw_after={"grid": [[1]]},
            info={"truncated": "false"},
        )
    with pytest.raises(AdapterError, match="valid BoundaryKind"):
        outcomes.outcome(
            before,
            after,
            raw_after={"grid": [[1]]},
            info={"boundary": "game_over_typo"},
        )


def test_gym_tuple_fields_are_authoritative_and_boolean_typed() -> None:
    _raw, info = _normalize_environment_result(
        (
            {"grid": [[1]]},
            2.5,
            True,
            False,
            {
                "reward": -999.0,
                "terminated": False,
                "truncated": True,
            },
        )
    )
    assert info["reward"] == 2.5
    assert info["terminated"] is True
    assert info["truncated"] is False

    with pytest.raises(AdapterError, match="Gymnasium terminated must be a bool"):
        _normalize_environment_result(
            ({"grid": [[1]]}, 0.0, "false", False, {})
        )
    with pytest.raises(AdapterError, match="Gym done must be a bool"):
        _normalize_environment_result(
            ({"grid": [[1]]}, 0.0, "false", {})
        )


class _InvalidOutcomeAdapter:
    def outcome(self, *args: Any, **kwargs: Any) -> dict[str, bool]:
        del args, kwargs
        return {"not_an_outcome": True}


def test_runner_commits_invalid_outcome_adapter_result_as_environment_error() -> None:
    events: list[str] = []
    result = run_episode(
        _MockEnvironment(),
        _SpyAgent(events, [1]),
        task_id="invalid-outcome",
        observation_adapter=MockObservationAdapter(n_values=8, n_actions=2),
        action_adapter=MockActionAdapter(n_actions=2),
        outcome_adapter=_InvalidOutcomeAdapter(),
        max_steps=1,
        bootstrap=False,
    )

    assert result.steps == 1
    assert result.final_outcome.boundary is BoundaryKind.ENVIRONMENT_ERROR
    assert result.final_outcome.metadata["exception_type"] == "TypeError"
    assert events.index("observe:environment_error") < events.index(
        "boundary:environment_error"
    )


def _diagnostic_transition_state(
    transition_id: str,
    *,
    terminal: Any = False,
    model_loss: Any = 0.0,
) -> dict[str, Any]:
    return {
        "transition_id": transition_id,
        "decision_id": f"decision-{transition_id}",
        "task_id": "task",
        "stage": 1,
        "step": 0,
        "state_id": "before",
        "successor_id": "after",
        "action": [0, -1, -1],
        "frame_changed": False,
        "reward": 0.0,
        "progress": 0.0,
        "hazard": 0.0,
        "terminal": terminal,
        "boundary": BoundaryKind.NONE.value,
        "events": [],
        "model_loss": model_loss,
        "prediction_error": 0.0,
    }


def test_diagnostics_checkpoint_parser_rejects_coercion_and_truncation() -> None:
    with pytest.raises(ValueError, match="exceed the declared capacity"):
        Diagnostics.from_state(
            {
                "max_decisions": 1,
                "max_transitions": 1,
                "decisions": [],
                "transitions": [
                    _diagnostic_transition_state("one"),
                    _diagnostic_transition_state("two"),
                ],
            }
        )

    with pytest.raises(ValueError, match="terminal must be a bool"):
        Diagnostics.from_state(
            {
                "max_decisions": 1,
                "max_transitions": 1,
                "decisions": [],
                "transitions": [
                    _diagnostic_transition_state("one", terminal="false")
                ],
            }
        )

    with pytest.raises(ValueError, match="model_loss must be finite"):
        Diagnostics.from_state(
            {
                "max_decisions": 1,
                "max_transitions": 1,
                "decisions": [],
                "transitions": [
                    _diagnostic_transition_state("one", model_loss=float("nan"))
                ],
            }
        )


def test_runner_rejects_noninteger_step_budget_before_starting() -> None:
    environment = _MockEnvironment()
    events: list[str] = []

    with pytest.raises(ValueError, match="non-negative integer"):
        run_episode(
            environment,
            _SpyAgent(events, [1]),
            task_id="bad-budget",
            observation_adapter=MockObservationAdapter(n_values=8, n_actions=2),
            action_adapter=MockActionAdapter(n_actions=2),
            outcome_adapter=MockOutcomeAdapter(),
            max_steps=1.5,
            bootstrap=False,
        )

    assert environment.value == 0
    assert events == []
