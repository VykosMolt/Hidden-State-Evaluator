from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hunter_seeker_v2.contracts import (
    Action,
    Candidate,
    Observation,
    Outcome,
    Prediction,
    Representation,
    RuntimeMode,
    ScoreTerm,
    Topology,
    Transition,
    WorldSnapshot,
    stable_frame_hash,
)
from hunter_seeker_v2.learning import ReplayBuffer, ReplayItem
from hunter_seeker_v2.teacher import (
    TeacherAccessGuard,
    TeacherExample,
    TeacherRequest,
    TeacherResponse,
    TrajectoryTeacher,
)


def _observation(value: int, *, progress: float = 0.0) -> Observation:
    return Observation(
        frame=np.asarray([[value]], dtype=np.int64),
        available_actions=(0,),
        task_id="task",
        progress=progress,
    )


def _snapshot(observation: Observation, step: int) -> WorldSnapshot:
    return WorldSnapshot(
        observation=observation,
        objects=(),
        events=(),
        topology=Topology(),
        representation=Representation(
            global_vector=np.zeros(2, dtype=np.float32),
            spatial=np.zeros((1, 1), dtype=np.float32),
        ),
        step=step,
    )


def _prediction() -> Prediction:
    return Prediction(
        change_probability=0.0,
        progress=0.0,
        value=0.0,
        hazard=0.0,
        terminal=0.0,
        uncertainty=0.0,
        latent_delta=np.zeros(1, dtype=np.float32),
        object_delta=np.zeros(1, dtype=np.float32),
    )


def _replay_item() -> ReplayItem:
    before_observation = _observation(0)
    after_observation = _observation(1)
    before = _snapshot(before_observation, 0)
    after = _snapshot(after_observation, 1)
    transition = Transition(
        transition_id="transition",
        decision_id="decision",
        task_id="task",
        stage=1,
        step=0,
        before=before,
        action=Action(0),
        after_observation=after_observation,
        outcome=Outcome(),
        frame_changed=True,
        after_state_id=after_observation.state_id,
    )
    return ReplayItem(before=before, after=after, transition=transition)


@pytest.mark.parametrize("value", [True, 1.0, 1.9, "1"])
def test_action_and_observation_indices_reject_integer_coercion(value) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Action(value)
    with pytest.raises(ValueError, match="must be an integer"):
        Observation(np.zeros((1, 1)), (value,), "task")


def test_action_rejects_partial_position_and_observation_rejects_clamping() -> None:
    with pytest.raises(ValueError, match="both be -1"):
        Action(0, x=2, y=-1)
    with pytest.raises(ValueError, match="stage"):
        Observation(np.zeros((1, 1)), (0,), "task", stage=0)
    with pytest.raises(ValueError, match="progress"):
        Observation(np.zeros((1, 1)), (0,), "task", progress=True)


def test_object_arrays_cannot_enter_immutable_contract_or_state_identity() -> None:
    mutable: list[int] = []
    frame = np.empty((1, 1), dtype=object)
    frame[0, 0] = mutable

    with pytest.raises(ValueError, match="object arrays"):
        Observation(frame, (0,), "task")
    with pytest.raises(ValueError, match="object arrays"):
        stable_frame_hash(frame)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"terminated": "false"},
        {"truncated": 0},
        {"reward": float("nan")},
        {"progress_delta": float("inf")},
        {"hazard": -0.1},
        {"hazard": 1.1},
    ],
)
def test_outcome_truth_rejects_malformed_values_instead_of_repairing(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        Outcome(**kwargs)


def test_nested_candidate_and_snapshot_sequences_are_detached() -> None:
    terms = [ScoreTerm("first", 1.0, "value")]
    candidate = Candidate(Action(0), _prediction(), terms=terms)
    terms.append(ScoreTerm("late", 10.0, "value"))
    assert candidate.score == pytest.approx(1.0)
    assert isinstance(candidate.terms, tuple)

    objects: list[object] = []
    events: list[object] = []
    snapshot = WorldSnapshot(
        observation=_observation(0),
        objects=objects,
        events=events,
        topology=Topology(),
        representation=Representation(
            global_vector=np.zeros(1),
            spatial=np.zeros((1, 1)),
        ),
        step=0,
    )
    objects.append(object())
    events.append(object())
    assert snapshot.objects == ()
    assert snapshot.events == ()


def test_topology_detaches_nested_lists_and_rejects_duplicate_edges() -> None:
    edge = [2, 1]
    edges = [edge]
    reachable = [2]
    topology = Topology(
        object_adjacencies=edges,
        reachable_object_ids=reachable,
    )
    edge[0] = 99
    edges.clear()
    reachable.append(3)
    assert topology.object_adjacencies == ((1, 2),)
    assert topology.reachable_object_ids == (2,)

    with pytest.raises(ValueError, match="unique"):
        Topology(object_adjacencies=((1, 2), (2, 1)))


def test_transition_and_replay_reject_incoherent_or_noncanonical_rows() -> None:
    item = _replay_item()
    with pytest.raises(ValueError, match="after_state_id"):
        Transition(
            transition_id="bad",
            decision_id="decision",
            task_id="task",
            stage=1,
            step=0,
            before=item.before,
            action=Action(0),
            after_observation=item.after.observation,
            outcome=Outcome(),
            frame_changed=True,
            after_state_id="not-the-observation",
        )
    with pytest.raises(TypeError, match="teacher must be a bool"):
        ReplayItem(
            before=item.before,
            after=item.after,
            transition=item.transition,
            teacher="false",
        )
    with pytest.raises(ValueError, match="capacity must be an integer"):
        ReplayBuffer(1.9)
    buffer = ReplayBuffer(2)
    with pytest.raises(TypeError, match="only ReplayItem"):
        buffer.push("not-a-replay-item")
    buffer.push(item)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        buffer.sample(
            1,
            teacher_fraction=float("nan"),
            rng=np.random.default_rng(0),
        )


class _NoncanonicalResponseTeacher:
    teacher_id = "malformed"

    def suggest(self, request: TeacherRequest):
        return SimpleNamespace(
            action=request.candidates[0],
            confidence=float("nan"),
            source=self.teacher_id,
            metadata={},
        )

    def examples(self, query):
        del query
        return ()


def test_teacher_guard_rejects_noncanonical_response_after_auditing_read() -> None:
    guard = TeacherAccessGuard(
        _NoncanonicalResponseTeacher(),
        mode=RuntimeMode.COMPAT_ASSISTED,
    )
    request = TeacherRequest("task", 1, "state", (Action(0),))

    with pytest.raises(TypeError, match="TeacherResponse"):
        guard.action_selection().suggest(request)

    assert guard.audit.attempts == 1
    assert guard.audit.allowed == 1
    assert guard.audit.action_selection_reads == 1


def test_teacher_request_rejects_duplicate_action_keys() -> None:
    with pytest.raises(ValueError, match="duplicate action keys"):
        TeacherRequest(
            "task",
            1,
            "state",
            (Action(0, name="first"), Action(0, name="second")),
        )


@pytest.mark.parametrize("confidence", [float("nan"), -0.1, 1.1, "0.5", True])
def test_teacher_response_rejects_malformed_confidence(confidence) -> None:
    with pytest.raises((TypeError, ValueError)):
        TeacherResponse(Action(0), confidence, "teacher")


@pytest.mark.parametrize("weight", [float("nan"), -0.1, "1.0", True])
def test_teacher_example_rejects_malformed_weight(weight) -> None:
    with pytest.raises(ValueError):
        TeacherExample("task", 1, "state", Action(0), weight=weight)


def test_npz_teacher_canonicalizes_integral_float_frames_and_stage_jumps(
    tmp_path,
) -> None:
    frames = np.asarray([[[0.0]], [[2.0]]], dtype=np.float32)
    after_frames = np.asarray([[[2.0]], [[3.0]]], dtype=np.float32)
    actions = np.asarray([[0, -1, -1], [0, -1, -1]], dtype=np.int64)
    levels = np.asarray([0, 2], dtype=np.int64)
    path = tmp_path / "trajectory.npz"
    np.savez(
        path,
        frames=frames,
        frames_after=after_frames,
        actions=actions,
        levels=levels,
    )

    teacher = TrajectoryTeacher.from_npz(str(path), task_id="task")
    rows = teacher._examples
    live = Observation(
        frame=frames[0].astype(np.int64),
        available_actions=(0,),
        task_id="task",
        stage=1,
        progress=0.0,
    )
    assert rows[0].state_id == live.state_id
    assert rows[0].successor_state_id == stable_frame_hash(
        after_frames[0].astype(np.int64),
        task_id="task",
        stage=3,
        available_actions=(0,),
        progress=2.0,
    )


def test_npz_teacher_rejects_fractional_action_rows(tmp_path) -> None:
    path = tmp_path / "fractional-actions.npz"
    np.savez(
        path,
        frames=np.zeros((1, 1, 1), dtype=np.int64),
        actions=np.asarray([[0.5, -1.0, -1.0]], dtype=np.float64),
        levels=np.zeros(1, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="actions must contain integer"):
        TrajectoryTeacher.from_npz(str(path), task_id="task")
