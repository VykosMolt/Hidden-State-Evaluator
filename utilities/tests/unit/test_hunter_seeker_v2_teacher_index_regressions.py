"""Regression tests for teacher state-identity reachability and student
feature locality.

F1: a live ``Observation.state_id`` hashes available_actions and progress
into the identity, while ``TrajectoryTeacher.from_npz`` historically hashed
frames without them, so the exact-state index could never fire for NPZ
teachers and every match rode the content fallback with ``exact_state=False``.
``from_npz`` now always declares an action inventory (explicit
``available_actions=...``, else inferred from the demonstrator's used
actions), which both restores exact-index reachability and hands
distillation the true legal alternative set — an action absent from that set
never receives negative updates and the head collapses onto it (the wa30
defect).

F3: the student's random-Fourier state map is an RBF kernel whose bandwidth
must sit near the reciprocal of typical inter-state latent distances.  At the
original ``state_feature_scale=80`` even adjacent recorded states produced
orthogonal features (cosine -0.11), reducing the head to a hash table that
steered zero autonomous route steps.  These tests pin the locality/separation
regime so the scale cannot silently regress to either extreme.
"""

from __future__ import annotations

import numpy as np
import pytest

from hunter_seeker_v2.contracts import (
    Action,
    Observation,
    Outcome,
    Representation,
    Topology,
    WorldSnapshot,
    stable_frame_hash,
)
from hunter_seeker_v2.student import (
    StateConditionedStudentPolicy,
    StudentPolicyConfig,
    StudentTeacherSample,
)
from hunter_seeker_v2.teacher import TeacherExample, TeacherRequest, TrajectoryTeacher

CANDIDATES = tuple(Action(index) for index in (1, 2, 3, 4))


def _write_trajectory(tmp_path):
    frames = np.zeros((2, 5, 7), dtype=np.int8)
    frames[0, 2, 3] = 3
    frames[1, 2, 4] = 3
    frames_after = np.roll(frames, -1, axis=0)
    actions = np.array([[3, -1, -1], [1, -1, -1]], dtype=np.int64)
    levels = np.zeros(2, dtype=np.int64)
    path = tmp_path / "traj.npz"
    np.savez(
        path,
        frames=frames,
        frames_after=frames_after,
        actions=actions,
        levels=levels,
    )
    return str(path), frames


def _live_state_id(frame):
    return Observation(
        frame=frame,
        available_actions=(1, 2, 3, 4),
        task_id="game",
        stage=1,
        progress=0.0,
    ).state_id


def test_live_state_id_reaches_npz_exact_index(tmp_path):
    path, frames = _write_trajectory(tmp_path)
    teacher = TrajectoryTeacher.from_npz(
        path,
        task_id="game",
        available_actions=(1, 2, 3, 4),
    )
    response = teacher.suggest(
        TeacherRequest(
            task_id="game",
            stage=1,
            state_id=_live_state_id(frames[0]),
            candidates=CANDIDATES,
        )
    )
    assert response is not None
    assert response.action == Action(3, -1, -1)
    assert response.metadata["exact_state"] is True


def test_declared_actions_flow_into_example_metadata(tmp_path):
    path, _frames = _write_trajectory(tmp_path)
    teacher = TrajectoryTeacher.from_npz(
        path,
        task_id="game",
        available_actions=(1, 2, 3, 4),
    )
    for example in teacher._examples:
        assert example.metadata["available_actions"] == (1, 2, 3, 4)


def test_declared_inventory_cannot_omit_a_demonstrated_action(tmp_path):
    path, _frames = _write_trajectory(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"omit demonstrated action indices: \[3\]",
    ):
        TrajectoryTeacher.from_npz(
            path,
            task_id="game",
            available_actions=(1, 2),
        )


def test_inferred_inventory_matches_used_actions(tmp_path):
    path, frames = _write_trajectory(tmp_path)
    teacher = TrajectoryTeacher.from_npz(path, task_id="game")
    for example in teacher._examples:
        assert example.metadata["available_actions"] == (1, 3)
    # Exact index fires when the live action set equals the inferred one.
    response = teacher.suggest(
        TeacherRequest(
            task_id="game",
            stage=1,
            state_id=Observation(
                frame=frames[0],
                available_actions=(1, 3),
                task_id="game",
                stage=1,
                progress=0.0,
            ).state_id,
            candidates=CANDIDATES,
        )
    )
    assert response is not None
    assert response.action == Action(3, -1, -1)
    assert response.metadata["exact_state"] is True


def test_mismatched_live_action_set_needs_content_fallback(tmp_path):
    path, frames = _write_trajectory(tmp_path)
    teacher = TrajectoryTeacher.from_npz(path, task_id="game")
    # Live set (1,2,3,4) differs from the inferred inventory (1,3).
    request = TeacherRequest(
        task_id="game",
        stage=1,
        state_id=_live_state_id(frames[0]),
        candidates=CANDIDATES,
    )
    blind = teacher.suggest(request)
    assert blind is not None and blind.action is None

    content_id = stable_frame_hash(frames[0], task_id="game", stage=0)
    response = teacher.suggest(
        TeacherRequest(
            task_id="game",
            stage=1,
            state_id=_live_state_id(frames[0]),
            candidates=CANDIDATES,
            metadata={"content_state_id": content_id},
        )
    )
    assert response is not None
    assert response.action == Action(3, -1, -1)
    # Fallback matches must never masquerade as exact-index hits.
    assert response.metadata["exact_state"] is False


def test_content_fallback_respects_stage_window(tmp_path):
    path, frames = _write_trajectory(tmp_path)
    teacher = TrajectoryTeacher.from_npz(path, task_id="game")
    content_id = stable_frame_hash(frames[0], task_id="game", stage=0)
    response = teacher.suggest(
        TeacherRequest(
            task_id="game",
            stage=3,  # recorded at stage 1; |3-1| > 1 must not alias
            state_id="no-such-state",
            candidates=CANDIDATES,
            metadata={"content_state_id": content_id},
        )
    )
    assert response is not None
    assert response.action is None


def _representation(values: np.ndarray) -> Representation:
    return Representation(
        global_vector=values[:16],
        spatial=values[16:].reshape(4, 20),
    )


def test_student_feature_map_keeps_locality_and_separation():
    policy = StateConditionedStudentPolicy()
    generator = np.random.default_rng(7)
    base = generator.normal(0.0, 0.33, size=96)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    features_base = policy._state_features(_representation(base))

    # Near-duplicate states (latent distance ~0.03, e.g. a live revisit of a
    # recorded state) must stay recognizable to trained rows.
    nudge = generator.normal(0.0, 1.0, size=96)
    near = base + 0.03 * nudge / np.linalg.norm(nudge)
    assert cosine(features_base, policy._state_features(_representation(near))) > 0.8

    # States at the observed median pairwise distance (~0.39) must not look
    # identical, or the head collapses to a dominant-action classifier.
    push = generator.normal(0.0, 1.0, size=96)
    far = base + 0.39 * push / np.linalg.norm(push)
    assert cosine(features_base, policy._state_features(_representation(far))) < 0.6


def _snapshot(
    representation: Representation,
    *,
    task_id: str = "game",
    actions: tuple[int, ...] = (1, 2),
) -> WorldSnapshot:
    observation = Observation(
        frame=np.zeros((3, 7), dtype=np.uint8),
        available_actions=actions,
        task_id=task_id,
    )
    return WorldSnapshot(
        observation=observation,
        objects=(),
        events=(),
        topology=Topology(),
        representation=representation,
        step=0,
    )


def _teacher_sample(
    representation: Representation,
    action_index: int,
    *,
    task_id: str = "game",
    candidates: tuple[int, ...] = (1, 2),
) -> StudentTeacherSample:
    return StudentTeacherSample(
        snapshot=_snapshot(representation, task_id=task_id, actions=candidates),
        example=TeacherExample(
            task_id=task_id,
            stage=1,
            state_id=f"state-{action_index}",
            action=Action(action_index),
        ),
        candidates=tuple(Action(index) for index in candidates),
    )


def test_neutral_online_outcomes_do_not_erode_teacher_margins():
    policy = StateConditionedStudentPolicy(StudentPolicyConfig(state_dim=8))
    generator = np.random.default_rng(11)
    representation = _representation(generator.normal(0.0, 0.33, size=96))
    policy.observe_teacher_batch([_teacher_sample(representation, 1)])
    snapshot = _snapshot(representation)
    margin_before = (
        policy.score(snapshot, Action(1)).value
        - policy.score(snapshot, Action(2)).value
    )
    assert margin_before > 0
    neutral = Outcome()
    for _ in range(200):
        policy.observe_outcome(snapshot, Action(1), neutral)
    margin_after = (
        policy.score(snapshot, Action(1)).value
        - policy.score(snapshot, Action(2)).value
    )
    # Neutral steps carry no preference information; they must not wash out
    # the distilled margin (measured live: a completing head degenerated into
    # a single-action loop within one episode of neutral updates).
    assert margin_after == pytest.approx(margin_before)
    # Informative outcomes still update parameters.
    death = Outcome(terminated=True, hazard=1.0, boundary="death")
    assert policy.observe_outcome(snapshot, Action(1), death) > 0


def test_distillation_fits_per_task_bandwidth_from_geometry():
    policy = StateConditionedStudentPolicy(StudentPolicyConfig(state_dim=8))
    generator = np.random.default_rng(23)
    # Coarse task: differently-labeled states far apart -> small scale.
    coarse = [
        _teacher_sample(
            _representation(generator.normal(0.0, 0.33, size=96) + offset),
            action,
            task_id="coarse",
        )
        for action, offset in ((1, 0.0), (2, 0.5), (1, 1.0), (2, 1.5))
    ]
    # Fine task: differently-labeled states nearly identical -> large scale.
    base = generator.normal(0.0, 0.33, size=96)
    fine = [
        _teacher_sample(
            _representation(base + offset),
            action,
            task_id="fine",
        )
        for action, offset in ((1, 0.0), (2, 0.01), (1, 0.02), (2, 0.03))
    ]
    policy.observe_teacher_batch(coarse + fine)
    scales = policy.summary()["task_scales"]
    assert set(scales) == {"coarse", "fine"}
    assert scales["fine"] > scales["coarse"]
    # Round-trips through persistence.
    restored = StateConditionedStudentPolicy.from_state(policy.state_dict())
    assert restored.summary()["task_scales"] == scales


def test_later_distillation_batch_cannot_reinterpret_existing_task_rows():
    policy = StateConditionedStudentPolicy(
        StudentPolicyConfig(
            state_dim=8,
            teacher_epochs=1,
            balance_teacher_actions=False,
            positive_transfer_scale=0.0,
            transfer_teacher_positives=False,
        )
    )
    generator = np.random.default_rng(41)
    base = generator.normal(0.0, 0.33, size=96)
    initial = [
        _teacher_sample(
            _representation(base + offset),
            action,
            task_id="incremental",
            candidates=(1, 2),
        )
        for action, offset in ((1, 0.0), (2, 0.8))
    ]
    policy.observe_teacher_batch(initial)

    probe = _snapshot(initial[0].snapshot.representation, task_id="incremental")
    scale_before = policy.summary()["task_scales"]["incremental"]
    row_before = np.asarray(
        policy.state_dict()["task_rows"]["incremental"]["1"]["weights"],
        dtype=np.float64,
    )
    score_before = policy.score(probe, Action(1)).value

    # The second import has a radically different geometry and does not
    # contain action 1.  Refitting the feature map here used to change action
    # 1's score even though its learned row was byte-for-byte untouched.
    close = generator.normal(0.0, 0.33, size=96)
    later = [
        _teacher_sample(
            _representation(close + offset),
            action,
            task_id="incremental",
            candidates=(2, 3),
        )
        for action, offset in ((2, 0.0), (3, 0.001))
    ]
    policy.observe_teacher_batch(later)

    assert policy.summary()["task_scales"]["incremental"] == pytest.approx(scale_before)
    np.testing.assert_array_equal(
        policy.state_dict()["task_rows"]["incremental"]["1"]["weights"],
        row_before,
    )
    assert policy.score(probe, Action(1)).value == pytest.approx(score_before)
