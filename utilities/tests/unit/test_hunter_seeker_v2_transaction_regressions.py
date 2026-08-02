from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from hunter_seeker_v2.adapters import (
    AdapterError,
    MockActionAdapter,
    MockObservationAdapter,
    MockOutcomeAdapter,
)
from hunter_seeker_v2.agent import (
    AgentStateError,
    CompactHunterSeeker,
)
from hunter_seeker_v2.contracts import (
    Action,
    AgentConfig,
    BoundaryKind,
    CompetenceState,
    EventKind,
    ModelConfig,
    ObjectState,
    Observation,
    Outcome,
    PolicyConfig,
    Representation,
    SearchConfig,
    WorldEvent,
)
from hunter_seeker_v2.diagnostics import Diagnostics
from hunter_seeker_v2.executable import ReplayCase
from hunter_seeker_v2.run_arc import run_episode


def _config(*, strict_finite: bool = True) -> AgentConfig:
    return AgentConfig(
        seed=41,
        search=SearchConfig(
            beam_width=2,
            horizon=1,
            max_click_candidates=4,
        ),
        policy=PolicyConfig(exploration_epsilon=0.0),
        model=ModelConfig(ensemble_size=2, latent_dim=12),
        strict_finite=strict_finite,
    )


def _observation(
    value: int | float,
    *,
    task_id: str = "task",
    actions: tuple[int, ...] = (0,),
    progress: float = 0.0,
    dtype: Any = np.uint8,
) -> Observation:
    frame = np.zeros((4, 5), dtype=dtype)
    frame[1:3, 1:3] = value
    return Observation(
        frame=frame,
        available_actions=actions,
        task_id=task_id,
        progress=progress,
    )


def _agent(*, strict_finite: bool = True) -> CompactHunterSeeker:
    return CompactHunterSeeker(
        config=_config(strict_finite=strict_finite),
        safe_action_provider=lambda _observation: (0,),
    )


def _normalized(value: Any) -> Any:
    """Convert component state into a comparison-safe immutable value."""

    if isinstance(value, np.ndarray):
        return (
            "array",
            str(value.dtype),
            tuple(int(v) for v in value.shape),
            np.ascontiguousarray(value).tobytes(),
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((str(key), _normalized(item)) for key, item in value.items()),
                key=lambda row: row[0],
            )
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_normalized(item) for item in value), key=repr))
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(_normalized(item) for item in value)
    return value


_COMPONENT_NAMES = (
    "perception",
    "graph",
    "evidence",
    "prior",
    "affordances",
    "ego",
    "exogenous",
    "hypotheses",
    "dynamics",
    "competence",
    "learner",
    "buffer",
    "student_policy",
    "diagnostics",
    "search_engine",
)


def _component_identities(agent: CompactHunterSeeker) -> tuple[tuple[str, int], ...]:
    return tuple((name, id(getattr(agent, name))) for name in _COMPONENT_NAMES)


def _mutable_state(agent: CompactHunterSeeker) -> Any:
    """Snapshot every observe-mutated component and transaction counter."""

    learner_state = {
        "real_updates": agent.learner.real_updates,
        "replay_updates": agent.learner.replay_updates,
        "last_replay_loss": agent.learner.last_replay_loss,
        "rng_state": agent.learner._rng.bit_generator.state,
        "replay_ids": tuple(
            item.transition.transition_id for item in agent.learner.replay.items
        ),
        "replay_sources": tuple(
            (item.source, item.teacher) for item in agent.learner.replay.items
        ),
    }
    state = {
        "perception": agent.perception.export_state(),
        "graph": agent.graph.export_state(),
        "evidence": agent.evidence.export_state(),
        "prior": agent.prior.state_dict(),
        "affordances": agent.affordances.state_dict(),
        "ego": agent.ego.state_dict(),
        "exogenous": agent.exogenous.state_dict(),
        "hypotheses": agent.hypotheses.state_dict(),
        "dynamics": agent.dynamics.state_dict(),
        "competence": agent.competence.state_dict(),
        "learner": learner_state,
        "buffer_ids": tuple(
            item.transition.transition_id for item in agent.buffer.items
        ),
        "student_policy": agent.student_policy.state_dict(),
        "diagnostics": agent.diagnostics.export_state(),
        "runtime_rng": agent._rng.bit_generator.state,
        "runtime": {
            "task_id": agent._task_id,
            "step": agent._step,
            "decision_counter": agent._decision_counter,
            "current_state_id": (
                None
                if agent.current_snapshot is None
                else agent.current_snapshot.state_id
            ),
            "pending_id": (
                None
                if agent.pending_decision is None
                else agent.pending_decision.decision_id
            ),
            "committed": frozenset(agent._committed_decision_ids),
            "last_transition_id": (
                None
                if agent._last_transition is None
                else agent._last_transition.transition_id
            ),
            "run_active": agent._run_active,
            "resume_ready": agent._resume_ready,
            "run_transition_count": agent._run_transition_count,
            "total_transition_count": agent._total_transition_count,
        },
    }
    return _normalized(state)


def test_replaced_decision_capabilities_are_rejected_without_any_mutation() -> None:
    agent = _agent()
    before = _observation(2)
    after = _observation(3)
    decision = agent.act(before)
    state_before = _mutable_state(agent)
    identities_before = _component_identities(agent)

    impostors = (
        replace(decision),
        replace(decision, score=decision.score + 1000.0),
    )
    for impostor in impostors:
        assert impostor is not decision
        with pytest.raises(AgentStateError, match="exact immutable Decision"):
            agent.observe(impostor, after, Outcome())
        assert agent.pending_decision is decision
        assert _component_identities(agent) == identities_before
        assert _mutable_state(agent) == state_before


def test_late_observe_exception_rolls_back_all_state_and_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent()
    before = _observation(2)
    after = _observation(4)
    decision = agent.act(before)
    state_before = _mutable_state(agent)
    identities_before = _component_identities(agent)
    original_record_transition = Diagnostics.record_transition

    def fail_after_learning(self: Diagnostics, *args: Any, **kwargs: Any) -> Any:
        del self, args, kwargs
        raise RuntimeError("injected late diagnostics failure")

    monkeypatch.setattr(Diagnostics, "record_transition", fail_after_learning)
    with pytest.raises(RuntimeError, match="injected late diagnostics failure"):
        agent.observe(decision, after, Outcome(reward=0.25))

    assert agent.pending_decision is decision
    assert _component_identities(agent) == identities_before
    assert _mutable_state(agent) == state_before

    monkeypatch.setattr(Diagnostics, "record_transition", original_record_transition)
    transition = agent.observe(decision, after, Outcome(reward=0.25))

    assert transition.decision_id == decision.decision_id
    assert agent.pending_decision is None
    assert agent.transition_count == 1
    assert agent.run_transition_count == 1
    assert len(agent.evidence) == 1
    assert agent.graph.edge_count == 1
    assert len(agent.buffer) == 1
    assert len(agent.diagnostics.transitions) == 1


class _CountingRepresentationBackend:
    transactionally_stateful = True

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, observation: Observation, _objects) -> Representation:
        self.encode_calls += 1
        return Representation(
            global_vector=np.full(12, self.encode_calls / 100.0, dtype=np.float32),
            spatial=np.zeros(observation.frame.shape, dtype=np.float32),
        )

    def state_dict(self) -> dict[str, int]:
        return {"encode_calls": self.encode_calls}


class _HookedUncopyableRepresentationBackend(_CountingRepresentationBackend):
    def __deepcopy__(self, _memo):
        raise TypeError("external resource is not copyable")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.encode_calls = int(state["encode_calls"])


class _UncopyableStatelessRepresentationBackend:
    def __deepcopy__(self, _memo):
        raise TypeError("opaque frozen external handle")

    def encode(self, observation: Observation, _objects) -> Representation:
        return Representation(
            global_vector=np.zeros(12, dtype=np.float32),
            spatial=np.zeros(observation.frame.shape, dtype=np.float32),
        )


@pytest.mark.parametrize(
    "backend",
    [_CountingRepresentationBackend(), _HookedUncopyableRepresentationBackend()],
)
def test_failed_observe_restores_stateful_representation_backend_exactly(
    monkeypatch: pytest.MonkeyPatch,
    backend: Any,
) -> None:
    agent = CompactHunterSeeker(
        config=_config(),
        representation_backend=backend,
        safe_action_provider=lambda _observation: (0,),
    )
    decision = agent.act(_observation(2))
    backend_before = backend.state_dict()
    backend_identity = id(backend)
    original_record_transition = Diagnostics.record_transition

    def fail_after_representation_refresh(
        self: Diagnostics,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del self, args, kwargs
        raise RuntimeError("late failure after representation refresh")

    monkeypatch.setattr(Diagnostics, "record_transition", fail_after_representation_refresh)
    with pytest.raises(RuntimeError, match="after representation refresh"):
        agent.observe(decision, _observation(3), Outcome(reward=0.25))

    assert agent.pending_decision is decision
    assert agent.representation_backend is backend
    assert id(agent.representation_backend) == backend_identity
    assert backend.state_dict() == backend_before

    monkeypatch.setattr(Diagnostics, "record_transition", original_record_transition)
    agent.observe(decision, _observation(3), Outcome(reward=0.25))

    assert agent.pending_decision is None
    assert agent.representation_backend.state_dict()["encode_calls"] > (
        backend_before["encode_calls"]
    )


def test_uncopyable_stateless_representation_backend_remains_compatible() -> None:
    backend = _UncopyableStatelessRepresentationBackend()
    agent = CompactHunterSeeker(
        config=_config(),
        representation_backend=backend,
        safe_action_provider=lambda _observation: (0,),
    )
    decision = agent.act(_observation(2))

    transition = agent.observe(decision, _observation(3), Outcome())

    assert transition.decision_id == decision.decision_id
    assert agent.representation_backend is backend


def test_disabling_online_learning_also_disables_student_updates() -> None:
    config = replace(_config(), enable_online_learning=False)
    agent = CompactHunterSeeker(
        config=config,
        safe_action_provider=lambda _observation: (0,),
    )
    before = _observation(2)
    decision = agent.act(before)

    agent.observe(decision, _observation(3), Outcome(reward=1.0))

    assert agent.student_policy.online_updates == 0
    assert agent.student_policy.state_dict()["task_rows"] == {}
    assert agent.learner.real_updates == 0
    # The immutable experience ledger and replay queue still record reality;
    # the flag controls parameter updates rather than erasing observations.
    assert len(agent.evidence) == 1
    assert len(agent.buffer) == 1


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_strict_finite_rejects_nonfinite_observation_frames(
    bad_value: float,
) -> None:
    agent = _agent(strict_finite=True)
    observation = _observation(bad_value, dtype=np.float64)

    with pytest.raises(ValueError, match="nonfinite"):
        agent.act(observation)

    assert agent.pending_decision is None
    assert agent.current_snapshot is None
    assert agent.transition_count == 0
    assert len(agent.evidence) == 0
    assert agent.graph.edge_count == 0


def _seed_episode_local_state(agent: CompactHunterSeeker) -> None:
    agent.competence.state = CompetenceState(
        dynamics_error_ema=0.4,
        hazard_calibration_error=0.3,
        predicted_success=0.2,
        recent_realized_progress=0.7,
        model_disagreement=0.1,
        stagnation_count=9,
        expected_learning_gain=0.6,
        remaining_risk_budget=0.2,
        time_pressure=0.8,
    )
    controlled = ObjectState(
        object_id=1,
        track_id=7,
        value=2,
        area=1,
        centroid_x=2.0,
        centroid_y=2.0,
        bbox=(2, 2, 2, 2),
        signature="value:2|shape:1x1",
    )
    agent.ego.observe(
        action=Action(0),
        events=(
            WorldEvent(
                EventKind.MOVED,
                subject_track_id=7,
                magnitude=1.0,
                metadata={"dx": 1.0, "dy": 0.0},
            ),
        ),
        visible_objects=(controlled,),
    )


def test_fresh_begin_resets_episode_state_but_checkpoint_resume_preserves_it() -> None:
    observation = _observation(2)

    fresh = _agent()
    fresh.begin_run("task", observation)
    _seed_episode_local_state(fresh)
    assert fresh.ego.state_dict()["track_buckets"]
    fresh.end_run()
    fresh.begin_run("task", observation)

    assert fresh.ego.state_dict()["track_buckets"] == {}
    assert fresh.ego.state_dict()["track_motion"] == {}
    assert fresh.competence.state.stagnation_count == 0
    assert fresh.competence.state.recent_realized_progress == 0.0
    assert fresh.competence.state.remaining_risk_budget == 1.0
    assert fresh.competence.state.time_pressure == 0.0
    # Durable calibration survives a genuinely fresh episode.
    assert fresh.competence.state.dynamics_error_ema == pytest.approx(0.4)
    assert fresh.competence.state.hazard_calibration_error == pytest.approx(0.3)

    source = _agent()
    source.begin_run("task", observation)
    _seed_episode_local_state(source)
    checkpoint = source.state_dict()
    restored = _agent()
    restored.load_state_dict(checkpoint)
    competence_before = restored.competence.state_dict()
    ego_before = restored.ego.state_dict()
    assert restored._resume_ready is True
    assert restored.current_snapshot is not None

    restored.begin_run("task", restored.current_snapshot.observation)

    assert restored.competence.state_dict() == competence_before
    assert restored.ego.state_dict() == ego_before
    assert restored._resume_ready is False
    assert restored.run_transition_count == 0


class _DecodeFailureAdapter(MockActionAdapter):
    def decode(self, action: Action, action_space: Any = None):
        del action, action_space
        raise AdapterError("deliberate decode rejection")


class _ResetOnlyEnvironment:
    def __init__(self) -> None:
        self.step_calls = 0

    def reset(self):
        return (
            {
                "grid": np.zeros((2, 3), dtype=np.uint8),
                "available_actions": [0],
                "progress": 0,
                "stage": 1,
                "state": "ACTIVE",
            },
            {},
        )

    def step(self, action: int):
        del action
        self.step_calls += 1
        raise AssertionError("decode failure must happen before environment.step")


class _BudgetEnvironment:
    def __init__(self) -> None:
        self.step_calls = 0

    def reset(self):
        return (
            {
                "grid": np.zeros((2, 3), dtype=np.uint8),
                "available_actions": [0],
                "progress": 0,
                "stage": 1,
                "state": "ACTIVE",
            },
            {},
        )

    def step(self, action: int):
        assert action == 0
        self.step_calls += 1
        return (
            {
                "grid": np.full(
                    (2, 3),
                    self.step_calls,
                    dtype=np.uint8,
                ),
                "available_actions": [0],
                "progress": 0,
                "stage": 1,
                "state": "ACTIVE",
            },
            0.25,
            False,
            False,
            {},
        )


def test_decode_failure_cancels_pending_without_transition_or_learning() -> None:
    environment = _ResetOnlyEnvironment()
    agent = _agent()

    with pytest.raises(AdapterError, match="deliberate decode rejection"):
        run_episode(
            environment,
            agent,
            task_id="task",
            observation_adapter=MockObservationAdapter(n_values=8, n_actions=1),
            action_adapter=_DecodeFailureAdapter(n_actions=1),
            outcome_adapter=MockOutcomeAdapter(),
            max_steps=2,
            bootstrap=False,
        )

    assert environment.step_calls == 0
    assert agent.pending_decision is None
    assert agent.transition_count == 0
    assert agent.run_transition_count == 0
    assert len(agent.evidence) == 0
    assert agent.graph.edge_count == 0
    assert len(agent.buffer) == 0
    assert agent.learner.real_updates == 0
    assert len(agent.diagnostics.transitions) == 0


def test_local_step_limit_is_recorded_in_full_agent_ledger() -> None:
    environment = _BudgetEnvironment()
    agent = _agent()

    result = run_episode(
        environment,
        agent,
        task_id="task",
        observation_adapter=MockObservationAdapter(n_values=8, n_actions=1),
        action_adapter=MockActionAdapter(n_actions=1),
        outcome_adapter=MockOutcomeAdapter(),
        max_steps=2,
        bootstrap=False,
    )

    assert environment.step_calls == 2
    assert result.steps == 2
    assert agent.run_transition_count == 2
    assert agent.transition_count == 2
    assert len(agent.evidence) == 2
    assert agent.evidence.records[0].boundary == BoundaryKind.NONE.value
    assert agent.evidence.records[-1].boundary == BoundaryKind.TIME_LIMIT.value
    assert len(agent.diagnostics.transitions) == 2
    assert agent.diagnostics.transitions[0].boundary == BoundaryKind.NONE.value
    assert agent.diagnostics.transitions[-1].boundary == BoundaryKind.TIME_LIMIT.value
    assert result.transitions[-1].outcome.boundary is BoundaryKind.TIME_LIMIT
    assert result.transitions[-1].outcome.truncated is True


def _train_student(agent: CompactHunterSeeker, *, reward: float) -> None:
    observation = _observation(2, actions=(0, 1))
    agent.begin_run("task", observation)
    assert agent.current_snapshot is not None
    for _ in range(4):
        agent.student_policy.observe_outcome(
            agent.current_snapshot,
            Action(0),
            Outcome(reward=reward),
            weight=1.5,
        )


def test_student_policy_roundtrips_in_full_and_weights_only_checkpoints(
    tmp_path: Any,
) -> None:
    source = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0, 1),
    )
    _train_student(source, reward=2.0)
    expected_student = source.student_policy.state_dict()
    checkpoint_path = tmp_path / "student-checkpoint.json"
    source.save_checkpoint(checkpoint_path)

    full = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0, 1),
    )
    full.load_checkpoint(checkpoint_path)
    assert full.student_policy.state_dict() == expected_student

    destination = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0, 1),
        agent_id="preserved-destination",
    )
    before = _observation(4, actions=(0,))
    after = _observation(5, actions=(0,))
    decision = destination.act(before)
    destination.observe(decision, after, Outcome(reward=-1.0, hazard=0.5))
    destination.end_run()
    destination.student_policy.observe_outcome(
        destination.current_snapshot,
        Action(1),
        Outcome(reward=-2.0, hazard=1.0),
    )
    graph_before = _normalized(destination.graph.export_state())
    evidence_before = _normalized(destination.evidence.export_state())
    replay_before = tuple(
        item.transition.transition_id for item in destination.buffer.items
    )
    runtime_before = (
        destination.agent_id,
        destination._task_id,
        destination._step,
        destination._decision_counter,
        destination.current_snapshot,
        destination._run_active,
        destination._run_transition_count,
        destination._total_transition_count,
    )

    destination.load_checkpoint(checkpoint_path, weights_only=True)

    assert destination.student_policy.state_dict() == expected_student
    assert _normalized(destination.graph.export_state()) == graph_before
    assert _normalized(destination.evidence.export_state()) == evidence_before
    assert tuple(
        item.transition.transition_id for item in destination.buffer.items
    ) == replay_before
    assert (
        destination.agent_id,
        destination._task_id,
        destination._step,
        destination._decision_counter,
        destination.current_snapshot,
        destination._run_active,
        destination._run_transition_count,
        destination._total_transition_count,
    ) == runtime_before


def test_game_completion_is_not_an_adverse_prediction_or_executable_target() -> None:
    agent = _agent()
    before = _observation(2)
    after = _observation(2)
    decision = agent.act(before)
    transition = agent.observe(
        decision,
        after,
        Outcome(reward=1.0, boundary=BoundaryKind.GAME_COMPLETED),
    )
    assert transition.outcome.terminated is True
    assert transition.outcome.completed is True

    chosen = next(
        row for row in decision.candidates if row.action.key == decision.action.key
    )
    expected_prediction_error = np.mean(
        (
            abs(chosen.prediction.change_probability - 0.0),
            abs(chosen.prediction.progress - 0.0),
            abs(chosen.prediction.hazard - 0.0),
            abs(chosen.prediction.terminal - 0.0),
        )
    )
    assert agent.diagnostics.transitions[-1].prediction_error == pytest.approx(
        expected_prediction_error
    )

    replay_case = ReplayCase.from_transition(transition)
    assert replay_case.terminal is False
    assert agent.current_snapshot is not None
    target = agent.dynamics._target(
        transition.before,
        agent.current_snapshot,
        transition,
    )
    assert target[4] == 0.0
    record = agent.evidence.records[-1]
    assert record.completed is True
    assert record.adverse_terminal is False
    assert record.negative is False
