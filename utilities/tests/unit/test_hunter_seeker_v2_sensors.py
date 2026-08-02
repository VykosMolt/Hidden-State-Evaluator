from __future__ import annotations

from dataclasses import fields
from typing import Sequence

import numpy as np
import pytest

from hunter_seeker_v2.contracts import (
    Action,
    Candidate,
    Observation,
    Prediction,
    Representation,
    RuntimeMode,
    TapRole,
    Topology,
    WorldSnapshot,
)
from hunter_seeker_v2.executable import (
    ExecutableModelNotVerified,
    ExecutableModelRegistry,
    ExecutablePlan,
    ExecutablePrediction,
    ExecutableState,
    ReplayCase,
    ReplayVerificationPolicy,
)
from hunter_seeker_v2.taps import (
    AntisymmetricPairwiseTap,
    CalibratedPointwiseTap,
    PointwiseReading,
    SurvivalRetainer,
    TapBundle,
    TapCalibration,
)
from hunter_seeker_v2.teacher import (
    TeacherAccessAudit,
    TeacherAccessGuard,
    TeacherAccessPhase,
    TeacherAccessViolation,
    TeacherExample,
    TeacherQuery,
    TeacherRequest,
    TeacherResponse,
    TrajectoryTeacher,
)


def _snapshot(*, value: int = 0, step: int = 0) -> WorldSnapshot:
    observation = Observation(
        frame=np.full((3, 3), value, dtype=np.uint8),
        available_actions=(0, 1, 2),
        task_id="task",
        stage=1,
    )
    representation = Representation(
        global_vector=np.asarray([value], dtype=np.float32),
        spatial=np.zeros((1, 3, 3), dtype=np.float32),
    )
    return WorldSnapshot(
        observation=observation,
        objects=(),
        events=(),
        topology=Topology(),
        representation=representation,
        step=step,
    )


def _candidate(action_index: int, *, value: float) -> Candidate:
    return Candidate(
        action=Action(action_index),
        prediction=Prediction(
            change_probability=0.5,
            progress=0.0,
            value=value,
            hazard=0.0,
            terminal=0.0,
            uncertainty=0.1,
            latent_delta=np.zeros(1, dtype=np.float32),
            object_delta=np.zeros(1, dtype=np.float32),
        ),
    )


def test_pointwise_taps_are_calibrated_bounded_finite_and_role_scoped() -> None:
    calibration = TapCalibration(
        calibration_id="hazard-held-out-v1",
        sample_count=128,
        lower_bound=0.0,
        upper_bound=1.0,
        scale=0.5,
        offset=0.1,
        confidence=0.8,
    )
    clipped = CalibratedPointwiseTap(
        tap_id="hazard",
        role=TapRole.HAZARD,
        calibration=calibration,
        scorer=lambda snapshot, candidate: 10.0,
    ).read(_snapshot(), _candidate(0, value=0.0))

    assert clipped.role is TapRole.HAZARD
    assert clipped.raw_value == 10.0
    assert clipped.value == 1.0
    assert clipped.lower_bound == 0.0
    assert clipped.upper_bound == 1.0
    assert clipped.confidence == pytest.approx(0.8)
    assert clipped.calibration_id == "hazard-held-out-v1"
    assert clipped.calibration_samples == 128

    with pytest.raises(ValueError, match="raw value must be finite"):
        CalibratedPointwiseTap(
            tap_id="hazard-nan-invalid",
            role=TapRole.HAZARD,
            calibration=calibration,
            scorer=lambda snapshot, candidate: float("nan"),
        ).read(_snapshot(), _candidate(0, value=0.0))

    with pytest.raises(ValueError, match="sample_count"):
        TapCalibration(
            calibration_id="not-calibrated",
            sample_count=0,
            lower_bound=0.0,
            upper_bound=1.0,
        )
    with pytest.raises(ValueError, match="cannot directly influence score"):
        CalibratedPointwiseTap(
            tap_id="survival",
            role=TapRole.SURVIVAL,
            calibration=calibration,
            scorer=lambda snapshot, candidate: 0.5,
            influences_score=True,
        )


@pytest.mark.parametrize(
    "factory",
    (
        lambda: TapCalibration(
            calibration_id="bool-count",
            sample_count=True,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        lambda: TapCalibration(
            calibration_id="nan-bound",
            sample_count=1,
            lower_bound=float("nan"),
            upper_bound=1.0,
        ),
        lambda: TapCalibration(
            calibration_id="bad-confidence",
            sample_count=1,
            lower_bound=0.0,
            upper_bound=1.0,
            confidence=1.5,
        ),
        lambda: TapBundle(pairwise_weight=False),
    ),
)
def test_tap_configuration_rejects_silent_numeric_coercion(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_tap_boolean_and_provenance_counters_are_strict() -> None:
    calibration = TapCalibration(
        calibration_id="strict-values",
        sample_count=10,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    with pytest.raises(TypeError, match="influences_score must be a bool"):
        CalibratedPointwiseTap(
            tap_id="truthy-string",
            role=TapRole.VALUE,
            calibration=calibration,
            scorer=lambda snapshot, candidate: 0.0,
            influences_score="false",
        )
    with pytest.raises(ValueError, match="calibration_samples must be >= 0"):
        PointwiseReading(
            tap_id="negative-counter",
            role=TapRole.DIAGNOSTIC,
            raw_value=0.0,
            value=0.0,
            lower_bound=-1.0,
            upper_bound=1.0,
            confidence=1.0,
            calibration_id="",
            calibration_samples=-1,
            influences_score=False,
        )


def test_pairwise_wrapper_scores_both_orders_and_is_exactly_antisymmetric() -> None:
    calls: list[tuple[int, int]] = []

    def directional_score(
        snapshot: WorldSnapshot,
        left: Candidate,
        right: Candidate,
    ) -> float:
        calls.append((left.action.index, right.action.index))
        return 2.0 * left.prediction.value - 0.25 * right.prediction.value

    tap = AntisymmetricPairwiseTap(
        tap_id="relational-value",
        calibration=TapCalibration(
            calibration_id="pairwise-held-out-v1",
            sample_count=256,
            lower_bound=-2.0,
            upper_bound=2.0,
        ),
        scorer=directional_score,
    )
    snapshot = _snapshot()
    left = _candidate(0, value=3.0)
    right = _candidate(1, value=1.0)

    left_right = tap.compare(snapshot, left, right)
    right_left = tap.compare(snapshot, right, left)

    assert calls[:2] == [(0, 1), (1, 0)]
    assert calls[2:] == [(1, 0), (0, 1)]
    assert left_right.value == -right_left.value
    assert left_right.flipped().value == -left_right.value
    assert left_right.flipped().left_action == right.action
    assert abs(left_right.value) <= left_right.bound


def test_pairwise_calibration_cannot_introduce_bias_or_asymmetric_clipping() -> None:
    with pytest.raises(ValueError, match="offset"):
        AntisymmetricPairwiseTap(
            tap_id="biased",
            calibration=TapCalibration(
                calibration_id="bad",
                sample_count=2,
                lower_bound=-1.0,
                upper_bound=1.0,
                offset=0.1,
            ),
            scorer=lambda snapshot, left, right: 0.0,
        )
    with pytest.raises(ValueError, match="symmetric"):
        AntisymmetricPairwiseTap(
            tap_id="asymmetric",
            calibration=TapCalibration(
                calibration_id="bad",
                sample_count=2,
                lower_bound=-1.0,
                upper_bound=2.0,
            ),
            scorer=lambda snapshot, left, right: 0.0,
        )


def test_survival_tap_can_rank_and_retain_but_cannot_select_final_action() -> None:
    tap = CalibratedPointwiseTap(
        tap_id="branch-survival",
        role=TapRole.SURVIVAL,
        calibration=TapCalibration(
            calibration_id="survival-held-out-v1",
            sample_count=200,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        scorer=lambda snapshot, candidate: candidate.prediction.value,
        influences_score=False,
    )
    retainer = SurvivalRetainer(tap)
    candidates = (
        _candidate(0, value=0.2),
        _candidate(1, value=0.9),
        _candidate(2, value=0.5),
    )

    retention = retainer.retain(_snapshot(), candidates, top_k=2)

    assert [row.candidate.action.index for row in retention.ranking] == [1, 2, 0]
    assert [candidate.action.index for candidate in retention.retained] == [1, 2]
    assert [candidate.action.index for candidate in retention.dropped] == [0]
    assert all(not row.reading.influences_score for row in retention.ranking)
    assert not hasattr(retainer, "select")
    assert not hasattr(retention, "selected_action")
    with pytest.raises(ValueError, match="cannot retain only one"):
        retainer.retain(_snapshot(), candidates, top_k=1)


def test_tap_bundle_rejects_duplicate_ids_before_one_reading_can_overwrite_another() -> None:
    calibration = TapCalibration(
        calibration_id="duplicate-id-test",
        sample_count=10,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    high_hazard = CalibratedPointwiseTap(
        tap_id="same-hazard",
        role=TapRole.HAZARD,
        calibration=calibration,
        scorer=lambda snapshot, candidate: 1.0,
    )
    low_hazard = CalibratedPointwiseTap(
        tap_id="same-hazard",
        role=TapRole.HAZARD,
        calibration=calibration,
        scorer=lambda snapshot, candidate: 0.0,
    )

    with pytest.raises(ValueError, match="tap ids must be unique"):
        TapBundle(pointwise=(high_hazard, low_hazard))


def test_survival_bundle_preserves_global_least_risk_candidate() -> None:
    tap = CalibratedPointwiseTap(
        tap_id="risk-blind-survival",
        role=TapRole.SURVIVAL,
        calibration=TapCalibration(
            calibration_id="risk-blind-survival-test",
            sample_count=20,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        scorer=lambda snapshot, candidate: float(candidate.action.index) / 2.0,
        influences_score=False,
    )

    def candidate(action_index: int, hazard: float) -> Candidate:
        return Candidate(
            action=Action(action_index),
            prediction=Prediction(
                change_probability=0.5,
                progress=0.0,
                value=0.0,
                hazard=hazard,
                terminal=0.0,
                uncertainty=0.1,
                latent_delta=np.zeros(1, dtype=np.float32),
                object_delta=np.zeros(1, dtype=np.float32),
            ),
        )

    candidates = (
        candidate(0, 0.0),
        candidate(1, 0.9),
        candidate(2, 0.9),
    )
    retained = TapBundle(
        survival=SurvivalRetainer(tap),
    ).retain(_snapshot(), candidates, top_k=2)

    assert len(retained) == 2
    assert 0 in {candidate.action.index for candidate in retained}


class _SpyTeacher:
    teacher_id = "spy"

    def __init__(self) -> None:
        self.suggest_calls = 0
        self.example_calls = 0

    def suggest(self, request: TeacherRequest) -> TeacherResponse:
        self.suggest_calls += 1
        return TeacherResponse(
            action=request.candidates[0],
            confidence=0.9,
            source=self.teacher_id,
        )

    def examples(self, query: TeacherQuery) -> Sequence[TeacherExample]:
        self.example_calls += 1
        return (
            TeacherExample(
                task_id=query.task_id,
                stage=query.stage or 1,
                state_id="demonstration-state",
                action=Action(0),
            ),
        )


def _teacher_request() -> TeacherRequest:
    return TeacherRequest(
        task_id="task",
        stage=1,
        state_id="current-state",
        candidates=(Action(0), Action(1)),
    )


@pytest.mark.parametrize("mode", [RuntimeMode.AUTONOMOUS, RuntimeMode.STUDENT])
def test_teacher_runtime_modes_block_action_advice_before_teacher_call(
    mode: RuntimeMode,
) -> None:
    teacher = _SpyTeacher()
    guard = TeacherAccessGuard(teacher, mode=mode)

    with pytest.raises(TeacherAccessViolation, match="forbids teacher reads"):
        guard.action_selection().suggest(_teacher_request())

    assert teacher.suggest_calls == 0
    assert guard.audit.attempts == 1
    assert guard.audit.allowed == 0
    assert guard.audit.denied == 1
    assert guard.audit.action_selection_reads == 0
    guard.assert_no_action_selection_reads()


def test_student_can_train_offline_and_compat_mode_is_audited_at_runtime() -> None:
    student_teacher = _SpyTeacher()
    student = TeacherAccessGuard(student_teacher, mode=RuntimeMode.STUDENT)
    examples = student.offline_training().examples(
        TeacherQuery(task_id="task", stage=1)
    )

    assert len(examples) == 1
    assert student_teacher.example_calls == 1
    assert student.audit.allowed == 1
    student.assert_no_action_selection_reads()

    compat_teacher = _SpyTeacher()
    audit = TeacherAccessAudit(max_records=1)
    compat = TeacherAccessGuard(
        compat_teacher,
        mode=RuntimeMode.COMPAT_ASSISTED,
        audit=audit,
    )
    assert compat.action_selection().suggest(_teacher_request()) is not None
    assert compat.action_selection().suggest(_teacher_request()) is not None
    compat.offline_training().examples(TeacherQuery(task_id="task"))

    assert compat_teacher.suggest_calls == 2
    assert audit.attempts == 3
    assert audit.action_selection_reads == 2
    assert len(audit.records) == 1
    assert audit.records[0].phase is TeacherAccessPhase.OFFLINE_TRAINING
    with pytest.raises(AssertionError, match="action selection"):
        compat.assert_no_action_selection_reads()


class _MalformedDatasetTeacher:
    teacher_id = "malformed-dataset"

    def __init__(self, rows) -> None:
        self.rows = rows

    def suggest(self, request: TeacherRequest) -> TeacherResponse | None:
        del request
        return None

    def examples(self, query: TeacherQuery):
        del query
        return self.rows


def test_teacher_guard_rejects_non_example_dataset_rows() -> None:
    guard = TeacherAccessGuard(
        _MalformedDatasetTeacher(("not-an-example",)),
        mode=RuntimeMode.STUDENT,
    )

    with pytest.raises(TypeError, match="row 0 is not a TeacherExample"):
        guard.offline_training().examples(TeacherQuery(task_id="task"))


@pytest.mark.parametrize(
    ("example", "message"),
    [
        (
            TeacherExample(
                task_id="other-task",
                stage=1,
                state_id="state",
                action=Action(0),
            ),
            "does not match query task",
        ),
        (
            TeacherExample(
                task_id="task",
                stage=2,
                state_id="state",
                action=Action(0),
            ),
            "does not match query stage",
        ),
    ],
)
def test_teacher_guard_rejects_examples_outside_query(
    example: TeacherExample,
    message: str,
) -> None:
    guard = TeacherAccessGuard(
        _MalformedDatasetTeacher((example,)),
        mode=RuntimeMode.STUDENT,
    )

    with pytest.raises(ValueError, match=message):
        guard.offline_training().examples(
            TeacherQuery(task_id="task", stage=1)
        )


def test_trajectory_teacher_loads_npz_and_abstains_outside_exact_state(tmp_path) -> None:
    frames = np.zeros((2, 3, 4), dtype=np.uint8)
    frames[1, 1, 1] = 2
    frames_after = frames.copy()
    frames_after[0, 0, 0] = 1
    actions = np.asarray([[1, -1, -1], [6, 1, 1]], dtype=np.int64)
    levels = np.asarray([0, 1], dtype=np.int64)
    path = tmp_path / "trajectory.npz"
    np.savez(
        path,
        frames=frames,
        frames_after=frames_after,
        actions=actions,
        levels=levels,
    )
    teacher = TrajectoryTeacher.from_npz(str(path), task_id="task")
    examples = tuple(teacher.examples(TeacherQuery(task_id="task")))
    assert len(examples) == 2

    exact = TeacherRequest(
        task_id="task",
        stage=examples[0].stage,
        state_id=examples[0].state_id,
        candidates=(Action(1), Action(2)),
    )
    response = teacher.suggest(exact)
    assert response is not None
    assert response.action == Action(1)
    assert response.metadata["exact_state"] is True

    miss = teacher.suggest(
        TeacherRequest(
            task_id="task",
            stage=1,
            state_id="unseen",
            candidates=(Action(1),),
        )
    )
    assert miss is not None
    assert miss.action is None
    assert miss.metadata["reason"] == "exact_state_abstention"


class _RuleModel:
    verification_transient_fields = ("plan_calls", "predict_inputs")

    def __init__(
        self,
        model_id: str,
        *,
        complexity: float,
        successors: dict[tuple[str, tuple[int, int, int]], str | None],
    ) -> None:
        self.model_id = model_id
        self.model_version = "1"
        self.complexity = complexity
        self.successors = successors
        self.predict_inputs: list[tuple[ExecutableState, Action]] = []
        self.plan_calls = 0

    def predict(
        self,
        state: ExecutableState,
        action: Action,
    ) -> ExecutablePrediction:
        self.predict_inputs.append((state, action))
        successor = self.successors.get((state.state_id, action.key))
        if successor is None:
            return ExecutablePrediction(applicable=False)
        return ExecutablePrediction(
            applicable=True,
            successor_state_id=successor,
            change_probability=1.0,
        )

    def plan(
        self,
        state: ExecutableState,
        actions: Sequence[Action],
        *,
        horizon: int,
    ) -> ExecutablePlan | None:
        self.plan_calls += 1
        return ExecutablePlan(
            model_id=self.model_id,
            model_version=self.model_version,
            actions=(actions[0],),
            score=1.0,
        )


def _replay_fixture() -> tuple[
    ExecutableState,
    tuple[Action, Action],
    tuple[ReplayCase, ReplayCase],
]:
    state = ExecutableState.from_snapshot(_snapshot())
    actions = (Action(0), Action(1))
    cases = (
        ReplayCase(
            case_id="case-0",
            before=state,
            action=actions[0],
            after_state_id="successor-0",
            frame_changed=True,
        ),
        ReplayCase(
            case_id="case-1",
            before=state,
            action=actions[1],
            after_state_id="successor-1",
            frame_changed=True,
        ),
    )
    return state, actions, cases


def _successor_map(
    state: ExecutableState,
    actions: Sequence[Action],
    successors: Sequence[str | None],
) -> dict[tuple[str, tuple[int, int, int]], str | None]:
    return {
        (state.state_id, action.key): successor
        for action, successor in zip(actions, successors, strict=True)
    }


def test_executable_registry_requires_replay_verification_and_ranks_models() -> None:
    state, actions, cases = _replay_fixture()
    registry = ExecutableModelRegistry(
        policy=ReplayVerificationPolicy(
            minimum_cases=2,
            minimum_coverage=0.5,
            maximum_error_rate=0.0,
        )
    )
    complex_full = _RuleModel(
        "complex-full",
        complexity=10.0,
        successors=_successor_map(
            state,
            actions,
            ("successor-0", "successor-1"),
        ),
    )
    compact_full = _RuleModel(
        "compact-full",
        complexity=2.0,
        successors=_successor_map(
            state,
            actions,
            ("successor-0", "successor-1"),
        ),
    )
    partial = _RuleModel(
        "partial",
        complexity=1.0,
        successors=_successor_map(state, actions, ("successor-0", None)),
    )
    wrong = _RuleModel(
        "wrong",
        complexity=0.0,
        successors=_successor_map(state, actions, ("wrong-0", "wrong-1")),
    )
    for model in (complex_full, compact_full, partial, wrong):
        registry.register(model)

    assert registry.plan(state, actions, horizon=2) is None
    with pytest.raises(ExecutableModelNotVerified):
        registry.plan(state, actions, horizon=2, model_id="compact-full")
    assert compact_full.plan_calls == 0

    complex_report = registry.verify("complex-full", cases)
    compact_report = registry.verify("compact-full", cases)
    partial_report = registry.verify("partial", cases)
    wrong_report = registry.verify("wrong", cases)

    assert complex_report.verified
    assert compact_report.verified
    assert partial_report.verified
    assert partial_report.coverage == pytest.approx(0.5)
    assert not wrong_report.verified
    assert [row.model_id for row in registry.ranked_verifications()] == [
        "compact-full",
        "complex-full",
        "partial",
    ]

    plan = registry.plan(state, actions, horizon=2)
    assert plan is not None
    assert plan.model_id == "compact-full"
    assert compact_full.plan_calls == 1
    assert complex_full.plan_calls == 0
    assert partial.plan_calls == 0
    assert wrong.plan_calls == 0


def test_executable_models_never_receive_replay_future_and_versions_reverify() -> None:
    state, actions, cases = _replay_fixture()
    model = _RuleModel(
        "rules",
        complexity=1.0,
        successors=_successor_map(
            state,
            actions,
            ("successor-0", "successor-1"),
        ),
    )
    registry = ExecutableModelRegistry(
        policy=ReplayVerificationPolicy(
            minimum_cases=2,
            minimum_coverage=1.0,
            maximum_error_rate=0.0,
        )
    )
    registry.register(model)
    assert registry.verify("rules", cases).verified

    exposed_fields = {field.name for field in fields(ExecutableState)}
    assert exposed_fields.isdisjoint(
        {
            "after",
            "after_observation",
            "after_state_id",
            "outcome",
            "successor",
            "transition",
        }
    )
    assert len(model.predict_inputs) == len(cases)
    assert all(isinstance(input_state, ExecutableState) for input_state, _ in model.predict_inputs)
    assert all(
        not hasattr(input_state, "after_state_id")
        and not hasattr(input_state, "after_observation")
        and not hasattr(input_state, "outcome")
        for input_state, _ in model.predict_inputs
    )

    assert registry.plan(state, actions, horizon=1, model_id="rules") is not None
    model.model_version = "2"
    with pytest.raises(ExecutableModelNotVerified):
        registry.plan(state, actions, horizon=1, model_id="rules")
    assert registry.verification("rules") is None
