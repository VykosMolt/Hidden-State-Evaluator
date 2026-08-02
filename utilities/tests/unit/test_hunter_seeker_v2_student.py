from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import (
    Action,
    AgentConfig,
    BoundaryKind,
    Observation,
    Outcome,
    PolicyConfig,
    Representation,
    RuntimeMode,
    SearchConfig,
    Topology,
    Transition,
    WorldSnapshot,
)
from hunter_seeker_v2.student import (
    StateConditionedStudentPolicy,
    StudentPolicyConfig,
    StudentRepresentationTrajectory,
    StudentTeacherSample,
)
from hunter_seeker_v2.teacher import (
    TeacherAccessGuard,
    TeacherExample,
    TeacherQuery,
)


def _representation(
    *values: float,
    spatial: np.ndarray | None = None,
) -> Representation:
    return Representation(
        global_vector=np.asarray(values, dtype=np.float32),
        spatial=(
            np.zeros((3, 4), dtype=np.float32)
            if spatial is None
            else np.asarray(spatial, dtype=np.float32)
        ),
    )


def _snapshot(
    task_id: str,
    representation: Representation,
    *,
    actions: tuple[int, ...] = (0, 1),
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


def _config(**overrides: object) -> StudentPolicyConfig:
    values: dict[str, object] = {
        "state_dim": 8,
        "learning_rate": 0.45,
        "l2": 0.0,
        "score_bound": 0.5,
        "confidence_prior": 0.5,
        "max_weight_norm": 12.0,
        "positive_transfer_scale": 0.25,
        "transfer_teacher_positives": True,
        "transfer_online_positives": True,
    }
    values.update(overrides)
    return StudentPolicyConfig(**values)


def _assert_row_state_equal(
    actual: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert set(actual) == set(expected)
    np.testing.assert_allclose(actual["weights"], expected["weights"], atol=1e-12)
    for key in ("support", "positive_support", "negative_support"):
        assert actual[key] == pytest.approx(expected[key])


def _assert_task_rows_equal(
    actual: dict[str, dict[str, dict[str, object]]],
    expected: dict[str, dict[str, dict[str, object]]],
) -> None:
    assert set(actual) == set(expected)
    for task_id in actual:
        assert set(actual[task_id]) == set(expected[task_id])
        for action_index in actual[task_id]:
            _assert_row_state_equal(
                actual[task_id][action_index],
                expected[task_id][action_index],
            )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("learning_rate", "0.25"),
        ("l2", False),
        ("runtime_weight", True),
        ("positive_transfer_scale", "0.5"),
    ),
)
def test_student_config_rejects_coerced_numeric_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        StudentPolicyConfig(**{field: value})


class _DatasetTeacher:
    teacher_id = "mapping-dataset"

    def __init__(self, examples: tuple[TeacherExample, ...]) -> None:
        self._examples = examples
        self.suggest_calls = 0
        self.example_calls = 0

    def suggest(self, _request):
        self.suggest_calls += 1
        raise AssertionError("runtime student scoring must not call the teacher")

    def examples(self, query: TeacherQuery):
        self.example_calls += 1
        return tuple(row for row in self._examples if row.task_id == query.task_id)


def _mapping_examples(
    task_id: str,
    mapping: tuple[int, int],
) -> tuple[TeacherExample, ...]:
    return (
        TeacherExample(
            task_id=task_id,
            stage=1,
            state_id=f"private-state-id-{mapping}-a",
            action=Action(mapping[0]),
        ),
        TeacherExample(
            task_id=task_id,
            stage=1,
            state_id=f"private-state-id-{mapping}-b",
            action=Action(mapping[1]),
        ),
    )


def _train_mapping(
    mapping: tuple[int, int],
) -> tuple[
    StateConditionedStudentPolicy,
    tuple[WorldSnapshot, WorldSnapshot],
    _DatasetTeacher,
    TeacherAccessGuard,
]:
    task_id = "mapping-task"
    snapshots = (
        _snapshot(task_id, _representation(1.0, 0.0)),
        _snapshot(task_id, _representation(0.0, 1.0)),
    )
    provider = _DatasetTeacher(_mapping_examples(task_id, mapping))
    guard = TeacherAccessGuard(provider, mode=RuntimeMode.STUDENT)
    examples = guard.offline_training().examples(TeacherQuery(task_id=task_id))
    head = StateConditionedStudentPolicy(
        _config(positive_transfer_scale=0.0, transfer_teacher_positives=False)
    )
    legal = (Action(0), Action(1))
    for _ in range(40):
        for snapshot, example in zip(snapshots, examples, strict=True):
            head.observe_teacher(snapshot, example, candidates=legal)
    return head, snapshots, provider, guard


def test_identical_frequency_opposite_state_mappings_flip_candidate_ranking_without_teacher_reads(
) -> None:
    left, left_states, left_teacher, left_guard = _train_mapping((0, 1))
    right, right_states, right_teacher, right_guard = _train_mapping((1, 0))

    left_examples = _mapping_examples("mapping-task", (0, 1))
    right_examples = _mapping_examples("mapping-task", (1, 0))
    assert Counter(row.action.index for row in left_examples) == Counter(
        row.action.index for row in right_examples
    ) == Counter({0: 1, 1: 1})

    assert left.score(left_states[0], Action(0)).value > left.score(
        left_states[0], Action(1)
    ).value
    assert left.score(left_states[1], Action(1)).value > left.score(
        left_states[1], Action(0)
    ).value
    assert right.score(right_states[0], Action(1)).value > right.score(
        right_states[0], Action(0)
    ).value
    assert right.score(right_states[1], Action(0)).value > right.score(
        right_states[1], Action(1)
    ).value
    left_distribution = left.distribution(
        left_states[0],
        (Action(0), Action(1)),
    )
    assert sum(left_distribution.probabilities) == pytest.approx(1.0)
    assert left_distribution.probabilities[0] > left_distribution.probabilities[1]

    # Runtime scores are learned functions of representation and candidate;
    # the teacher and its exact state ids are not retained or queried.
    for provider, guard in ((left_teacher, left_guard), (right_teacher, right_guard)):
        assert provider.example_calls == 1
        assert provider.suggest_calls == 0
        assert guard.audit.action_selection_reads == 0
    serialized = repr(left.state_dict()) + repr(right.state_dict())
    assert "private-state-id" not in serialized


def test_balanced_batch_recovers_minority_route_without_inflating_support() -> None:
    task_id = "imbalanced-route"
    majority = _snapshot(task_id, _representation(1.0, 0.0))
    minority = _snapshot(task_id, _representation(0.0, 1.0))
    legal = (Action(0), Action(1))
    samples = tuple(
        StudentTeacherSample(
            majority,
            TeacherExample(task_id, 1, f"majority-{index}", Action(0)),
            legal,
        )
        for index in range(8)
    ) + (
        StudentTeacherSample(
            minority,
            TeacherExample(task_id, 1, "minority", Action(1)),
            legal,
        ),
    )
    head = StateConditionedStudentPolicy(
        _config(
            positive_transfer_scale=0.0,
            transfer_teacher_positives=False,
            teacher_epochs=8,
            balance_teacher_actions=True,
        )
    )

    head.observe_teacher_batch(samples)

    assert head.score(majority, Action(0)).value > head.score(majority, Action(1)).value
    assert head.score(minority, Action(1)).value > head.score(minority, Action(0)).value
    assert head.teacher_updates == len(samples)
    state = head.state_dict()
    total_support = sum(
        row["support"]
        for action_rows in state["task_rows"].values()
        for row in action_rows.values()
    )
    # One positive and one contrastive row per empirical example, regardless
    # of the eight optimization epochs.
    assert total_support == pytest.approx(2.0 * len(samples))


def test_positional_candidates_with_one_action_index_learn_state_dependent_clicks(
) -> None:
    task_id = "click-task"
    state_a = _snapshot(task_id, _representation(1.0, 0.0), actions=(6,))
    state_b = _snapshot(task_id, _representation(0.0, 1.0), actions=(6,))
    left = Action(6, x=0, y=1, name="click")
    right = Action(6, x=6, y=1, name="click")
    head = StateConditionedStudentPolicy(
        _config(positive_transfer_scale=0.0, transfer_teacher_positives=False)
    )
    for _ in range(60):
        head.observe_teacher(
            state_a,
            TeacherExample(task_id, 1, "unused-a", left),
            candidates=(left, right),
        )
        head.observe_teacher(
            state_b,
            TeacherExample(task_id, 1, "unused-b", right),
            candidates=(left, right),
        )

    assert head.score(state_a, left).value > head.score(state_a, right).value
    assert head.score(state_b, right).value > head.score(state_b, left).value
    click_distribution = head.distribution(state_b, (left, right))
    assert click_distribution.probabilities[1] > click_distribution.probabilities[0]


def test_trajectory_credit_updates_every_weighted_representation_and_scores_terminal_only(
) -> None:
    config = _config(positive_transfer_scale=0.0, transfer_teacher_positives=False)
    snapshot = _snapshot("trajectory-task", _representation(-1.0, -1.0))
    representations = (
        _representation(1.0, 0.0),
        _representation(0.0, 1.0),
        _representation(-1.0, 0.5),
    )
    trajectory = StudentRepresentationTrajectory(
        representations,
        loop_weights=(2.0, 3.0, 5.0),
        terminal_index=2,
    )
    assert trajectory.loop_weights == pytest.approx((0.2, 0.3, 0.5))
    example = TeacherExample("trajectory-task", 1, "not-retained", Action(0))
    legal = (Action(0), Action(1))

    trajectory_head = StateConditionedStudentPolicy(config)
    trajectory_head.observe_teacher(
        snapshot,
        example,
        candidates=legal,
        trajectory=trajectory,
    )
    manual_head = StateConditionedStudentPolicy(config)
    for representation, loop_weight in zip(
        trajectory.representations,
        trajectory.loop_weights,
        strict=True,
    ):
        manual_head.observe_teacher(
            snapshot,
            replace(example, weight=loop_weight),
            candidates=legal,
            trajectory=StudentRepresentationTrajectory((representation,), (1.0,)),
        )
    _assert_task_rows_equal(
        trajectory_head.state_dict()["task_rows"],
        manual_head.state_dict()["task_rows"],
    )

    # Nonterminal elements affect credit, never the runtime representation.
    alternate_early = StudentRepresentationTrajectory(
        (_representation(99.0, -99.0), _representation(50.0), representations[-1]),
        loop_weights=(0.8, 0.1, 0.1),
        terminal_index=2,
    )
    assert trajectory_head.score(snapshot, Action(0), trajectory=trajectory) == (
        trajectory_head.score(snapshot, Action(0), trajectory=alternate_early)
    )


def test_online_trajectory_credit_matches_weighted_updates_and_transfer_is_positive_only(
) -> None:
    config = _config(positive_transfer_scale=0.5, transfer_teacher_positives=False)
    source = _snapshot("source-task", _representation(1.0, 0.5))
    target = _snapshot("target-task", _representation(1.0, 0.5))
    trajectory = StudentRepresentationTrajectory(
        (_representation(1.0, 0.0), _representation(0.5, 1.0)),
        loop_weights=(1.0, 3.0),
        terminal_index=1,
    )

    head = StateConditionedStudentPolicy(config)
    failure = Outcome(terminated=True, boundary=BoundaryKind.DEATH, hazard=1.0)
    failed_transition = Transition(
        transition_id="failure",
        decision_id="decision-failure",
        task_id="source-task",
        stage=1,
        step=0,
        before=source,
        action=Action(0),
        after_observation=source.observation,
        outcome=failure,
        frame_changed=False,
        after_state_id=source.state_id,
    )
    for _ in range(12):
        head.observe_transition(failed_transition, trajectory=trajectory)
    assert head.score(source, Action(0), trajectory=trajectory).value < 0.0
    assert head.score(target, Action(0), trajectory=trajectory).value == 0.0
    assert "0" not in head.state_dict()["transfer_rows"]

    success = Outcome(reward=1.0, progress_delta=1.0)
    successful_transition = replace(
        failed_transition,
        transition_id="success",
        decision_id="decision-success",
        action=Action(1),
        outcome=success,
    )
    reference = StateConditionedStudentPolicy(config)
    head.observe_transition(successful_transition, trajectory=trajectory)
    for representation, loop_weight in zip(
        trajectory.representations,
        trajectory.loop_weights,
        strict=True,
    ):
        reference.observe_transition(
            successful_transition,
            weight=loop_weight,
            trajectory=StudentRepresentationTrajectory((representation,), (1.0,)),
        )
    assert head.score(target, Action(1), trajectory=trajectory).value > 0.0
    _assert_row_state_equal(
        head.state_dict()["transfer_rows"]["1"],
        reference.state_dict()["transfer_rows"]["1"],
    )


def test_zero_weight_is_noop_and_state_roundtrip_preserves_bounded_scores() -> None:
    config = _config()
    snapshot = _snapshot("roundtrip-task", _representation(0.25, 1.0))
    head = StateConditionedStudentPolicy(config)
    zero = TeacherExample(
        task_id="roundtrip-task",
        stage=1,
        state_id="must-not-be-stored",
        action=Action(0),
        weight=0.0,
    )
    before = head.state_dict()
    assert head.observe_teacher(snapshot, zero, candidates=(Action(0), Action(1))) == 0
    assert head.state_dict() == before

    weighted = replace(zero, weight=0.5)
    head.observe_teacher(snapshot, weighted, candidates=(Action(0), Action(1)))
    score = head.score(snapshot, Action(0))
    assert score.task_support == pytest.approx(0.5)
    assert 0.0 < score.confidence < 1.0
    assert abs(score.value) <= config.score_bound

    restored = StateConditionedStudentPolicy.from_state(head.state_dict())
    assert restored.state_dict() == head.state_dict()
    assert restored.score(snapshot, Action(0)) == head.score(snapshot, Action(0))
    assert restored.score(snapshot, Action(1)) == head.score(snapshot, Action(1))


def test_state_loader_rejects_component_support_above_total_support() -> None:
    snapshot = _snapshot("invalid-support", _representation(0.25, 1.0))
    head = StateConditionedStudentPolicy(_config())
    example = TeacherExample(
        task_id="invalid-support",
        stage=1,
        state_id="invalid-support-state",
        action=Action(0),
    )
    head.observe_teacher(snapshot, example, candidates=(Action(0), Action(1)))
    state = head.state_dict()
    row = state["task_rows"]["invalid-support"]["0"]
    row["support"] = 1.0
    row["positive_support"] = 1.0
    row["negative_support"] = 1.0

    with pytest.raises(ValueError, match="cannot exceed total support"):
        StateConditionedStudentPolicy.from_state(state)


def _trained_student_state() -> dict[str, object]:
    snapshot = _snapshot("strict-load", _representation(0.25, 1.0))
    head = StateConditionedStudentPolicy(_config())
    head.observe_teacher(
        snapshot,
        TeacherExample(
            task_id="strict-load",
            stage=1,
            state_id="strict-load-state",
            action=Action(0),
        ),
        candidates=(Action(0), Action(1)),
    )
    return head.state_dict()


def test_student_loader_rejects_coerced_scalars_and_action_keys() -> None:
    base = _trained_student_state()

    malformed: list[tuple[dict[str, object], str]] = []
    version = deepcopy(base)
    version["version"] = True
    malformed.append((version, "version must be an integer"))

    feature_dim = deepcopy(base)
    feature_dim["feature_dim"] = float(feature_dim["feature_dim"])
    malformed.append((feature_dim, "feature_dim must be an integer"))

    counter = deepcopy(base)
    counter["teacher_updates"] = 1.5
    malformed.append((counter, "teacher_updates must be an integer"))

    negative_counter = deepcopy(base)
    negative_counter["online_updates"] = -1
    malformed.append((negative_counter, "online_updates must be >= 0"))

    action_key = deepcopy(base)
    row = action_key["task_rows"]["strict-load"].pop("0")
    action_key["task_rows"]["strict-load"]["00"] = row
    malformed.append((action_key, "canonical non-negative integer string"))

    scale = deepcopy(base)
    scale["task_scales"]["strict-load"] = "8.0"
    malformed.append((scale, "task scale must be a finite number"))

    boolean_weight = deepcopy(base)
    boolean_weight["task_rows"]["strict-load"]["0"]["weights"][0] = True
    malformed.append((boolean_weight, "weights must contain only numbers"))

    for state, message in malformed:
        with pytest.raises(ValueError, match=message):
            StateConditionedStudentPolicy.from_state(state)


def test_student_loader_rejects_inconsistent_transfer_rows_and_weight_norm() -> None:
    transfer_state = _trained_student_state()
    transfer = transfer_state["transfer_rows"]["0"]
    transfer["support"] = 2.0
    transfer["positive_support"] = 1.0
    transfer["negative_support"] = 1.0
    with pytest.raises(ValueError, match="positive support only"):
        StateConditionedStudentPolicy.from_state(transfer_state)

    oversized = _trained_student_state()
    weights = oversized["task_rows"]["strict-load"]["0"]["weights"]
    weights[:] = [0.0] * len(weights)
    weights[0] = float(_config().max_weight_norm) + 1.0
    with pytest.raises(ValueError, match="weight norm exceeds"):
        StateConditionedStudentPolicy.from_state(oversized)


def test_integrated_student_term_alone_can_flip_the_runtime_decision() -> None:
    observation = Observation(
        frame=np.asarray(
            [[0, 0, 0], [0, 3, 0], [0, 0, 0]],
            dtype=np.uint8,
        ),
        available_actions=(0, 1),
        task_id="decision-causality",
    )
    config = AgentConfig(
        runtime_mode=RuntimeMode.STUDENT,
        seed=73,
        search=SearchConfig(horizon=1, beam_width=2),
        policy=PolicyConfig(exploration_epsilon=0.0, risk_limit=2.0),
    )

    baseline = CompactHunterSeeker(
        config=config,
        safe_action_provider=lambda _observation: (0, 1),
    )
    baseline_decision = baseline.act(observation)
    target_index = 1 - baseline_decision.action.index

    trained_head = StateConditionedStudentPolicy(
        StudentPolicyConfig(
            positive_transfer_scale=0.0,
            transfer_teacher_positives=False,
        )
    )
    trained = CompactHunterSeeker(
        config=config,
        safe_action_provider=lambda _observation: (0, 1),
        student_policy=trained_head,
    )
    trained.begin_run(observation.task_id, observation)
    snapshot = trained.current_snapshot
    assert snapshot is not None
    example = TeacherExample(
        task_id=observation.task_id,
        stage=observation.stage,
        state_id="not-a-runtime-lookup",
        action=Action(target_index),
    )
    for _ in range(40):
        trained_head.observe_teacher(
            snapshot,
            example,
            candidates=(Action(0), Action(1)),
        )

    trained_decision = trained.act(observation)

    assert trained_decision.action.index == target_index
    assert trained_decision.action.index != baseline_decision.action.index
    without_student = min(
        trained_decision.candidates,
        key=lambda candidate: (
            -(
                candidate.score
                - next(
                    term.value
                    for term in candidate.terms
                    if term.name == "student_policy"
                )
            ),
            candidate.action.key,
        ),
    )
    assert without_student.action.key == baseline_decision.action.key
