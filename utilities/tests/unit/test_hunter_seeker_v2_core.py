from __future__ import annotations

import copy

import numpy as np
import pytest

from hunter_seeker_v2.agent import (
    CompactHunterSeeker,
    DuplicateDecisionError,
)
from hunter_seeker_v2.contracts import (
    Action,
    AgentConfig,
    BoundaryKind,
    Candidate,
    CompetenceState,
    EvidenceScope,
    LearningConfig,
    ModelConfig,
    Observation,
    Outcome,
    PolicyConfig,
    Prediction,
    RuntimeMode,
    ScoreTerm,
    SearchConfig,
)
from hunter_seeker_v2.memory import (
    EvidenceRecord,
    EvidenceScore,
    EvidenceStore,
    StateGraph,
)
from hunter_seeker_v2.policy import PolicyScorer, RiskArbiter
from hunter_seeker_v2.models import GridFeatureBackend
from hunter_seeker_v2.executable import (
    ExecutableModelRegistry,
    ExecutablePlan,
    ExecutablePrediction,
    ExecutableState,
    ReplayCase,
    ReplayVerificationPolicy,
)
from hunter_seeker_v2.teacher import (
    TeacherAccessGuard,
    TeacherExample,
    TeacherQuery,
    TeacherRequest,
    TrajectoryTeacher,
)
from hunter_seeker_v2.taps import (
    AntisymmetricPairwiseTap,
    CalibratedPointwiseTap,
    SurvivalRetainer,
    TapBundle,
    TapCalibration,
)
from hunter_seeker_v2.contracts import TapRole


def _config(
    *,
    seed: int = 3,
    epsilon: float = 0.0,
    mode: RuntimeMode = RuntimeMode.AUTONOMOUS,
) -> AgentConfig:
    return AgentConfig(
        runtime_mode=mode,
        seed=seed,
        search=SearchConfig(
            beam_width=3,
            horizon=1,
            max_click_candidates=8,
        ),
        policy=PolicyConfig(
            exploration_epsilon=epsilon,
            risk_limit=0.65,
        ),
        model=ModelConfig(
            ensemble_size=3,
            latent_dim=16,
            learning_rate=0.08,
        ),
    )


def _obs(
    value: int,
    *,
    task: str = "task",
    actions: tuple[int, ...] = (0, 1),
    progress: float = 0.0,
    shape: tuple[int, int] = (5, 7),
) -> Observation:
    frame = np.zeros(shape, dtype=np.uint8)
    frame[1:3, 1:3] = value
    return Observation(
        frame=frame,
        available_actions=actions,
        task_id=task,
        progress=progress,
    )


def _prediction(
    *,
    progress: float = 0.0,
    value: float = 0.0,
    hazard: float = 0.0,
    terminal: float = 0.0,
    uncertainty: float = 0.0,
) -> Prediction:
    return Prediction(
        change_probability=0.5,
        progress=progress,
        value=value,
        hazard=hazard,
        terminal=terminal,
        uncertainty=uncertainty,
        latent_delta=np.zeros(2, dtype=np.float32),
        object_delta=np.zeros(2, dtype=np.float32),
    )


def test_terminal_completing_action_is_committed_once_and_owns_outcome() -> None:
    agent = CompactHunterSeeker(
        config=_config(),
        click_action_index=None,
        safe_action_provider=lambda _obs: (0,),
    )
    before = _obs(2, actions=(0,), progress=0.0)
    after = _obs(3, actions=(0,), progress=1.0)
    agent.begin_run("task", before)

    decision = agent.act(before)
    transition = agent.observe(
        decision,
        after,
        Outcome(
            reward=1.0,
            progress_delta=1.0,
            terminated=True,
            boundary=BoundaryKind.LEVEL_COMPLETED,
        ),
    )

    assert agent.transition_count == 1
    assert agent.run_transition_count == 1
    assert len(agent.evidence) == 1
    assert agent.graph.edge_count == 1
    assert transition.action == decision.action
    assert transition.outcome.completed is True
    assert transition.after_state_id == after.state_id
    assert agent.diagnostics.transitions[-1].transition_id == transition.transition_id
    agent.on_level_complete(1)


def test_duplicate_observe_is_rejected_without_duplicate_learning() -> None:
    agent = CompactHunterSeeker(config=_config(), safe_action_provider=lambda _obs: (0,))
    before = _obs(2, actions=(0,))
    after = _obs(3, actions=(0,))
    decision = agent.act(before)
    transition = agent.observe(decision, after, Outcome())
    counts = (
        agent.transition_count,
        len(agent.evidence),
        agent.graph.edge_count,
        agent.dynamics.update_count,
    )

    with pytest.raises(DuplicateDecisionError):
        agent.observe(decision, after, Outcome())

    assert (
        agent.transition_count,
        len(agent.evidence),
        agent.graph.edge_count,
        agent.dynamics.update_count,
    ) == counts
    assert transition.action == decision.action


def test_death_is_attributed_to_fatal_action_not_predecessor() -> None:
    agent = CompactHunterSeeker(config=_config(), safe_action_provider=lambda _obs: (1,))
    before = _obs(4, actions=(1,))
    after = _obs(4, actions=(1,))
    decision = agent.act(before)
    transition = agent.observe(
        decision,
        after,
        Outcome(
            terminated=True,
            boundary=BoundaryKind.DEATH,
            hazard=1.0,
        ),
    )

    record = agent.evidence.records[-1]
    assert record.action == decision.action.key
    assert record.terminal is True
    assert record.hazard == 1.0
    assert transition.outcome.failed is True
    agent.on_game_over()


def test_act_search_is_speculative_and_does_not_mutate_durable_knowledge() -> None:
    agent = CompactHunterSeeker(config=_config(), safe_action_provider=lambda _obs: (0, 1))
    observation = _obs(5)
    agent.begin_run("task", observation)
    perception_before = copy.deepcopy(agent.perception.export_state())
    graph_before = copy.deepcopy(agent.graph.export_state())
    evidence_before = copy.deepcopy(agent.evidence.export_state())
    model_updates_before = agent.dynamics.update_count

    decision = agent.act(observation)

    assert decision.candidates
    assert agent.perception.export_state() == perception_before
    assert agent.graph.export_state() == graph_before
    assert agent.evidence.export_state() == evidence_before
    assert agent.dynamics.update_count == model_updates_before


def test_empty_legal_actions_use_adapter_safe_fallback_on_odd_grid() -> None:
    agent = CompactHunterSeeker(
        config=_config(),
        click_action_index=None,
        safe_action_provider=lambda _obs: (2,),
    )
    observation = _obs(6, actions=(), shape=(5, 9))
    decision = agent.act(observation)

    assert decision.action == Action(2)
    assert decision.action.x == -1
    assert decision.action.y == -1


def test_negative_evidence_is_exactly_neutral_across_tasks() -> None:
    store = EvidenceStore()
    store.append(
        EvidenceRecord(
            transition_id="death-a",
            task_id="task-a",
            stage=1,
            step=0,
            state_id="same-looking-state",
            successor_id="terminal",
            action=(1, -1, -1),
            object_signature="target:shape",
            effect_signature="hazard:1",
            event_kinds=("hazard", "terminal"),
            frame_changed=False,
            progress=0.0,
            reward=0.0,
            hazard=1.0,
            terminal=True,
            boundary=BoundaryKind.DEATH.value,
            uncertainty=0.0,
            confidence=1.0,
            scope=EvidenceScope.TASK,
        )
    )

    same_task = store.score(
        task_id="task-a",
        stage=1,
        state_id="same-looking-state",
        action=Action(1),
        object_signature="target:shape",
    )
    other_task = store.score(
        task_id="task-b",
        stage=1,
        state_id="same-looking-state",
        action=Action(1),
        object_signature="target:shape",
    )

    assert same_task.negative > 0.0
    assert same_task.total < 0.0
    assert other_task.negative == 0.0
    assert other_task.total == 0.0


def test_effect_ledger_exposes_task_scoped_inverse_action_diagnostic() -> None:
    store = EvidenceStore()
    for index, action in enumerate(((2, -1, -1), (2, -1, -1), (1, -1, -1))):
        store.append(
            EvidenceRecord(
                transition_id=f"effect-{index}",
                task_id="task",
                stage=1,
                step=index,
                state_id=f"s{index}",
                successor_id=f"n{index}",
                action=action,
                object_signature="",
                effect_signature="events:moved|objects:0",
                event_kinds=("moved",),
                frame_changed=True,
                progress=0.0,
                reward=0.0,
                hazard=0.0,
                terminal=False,
                boundary=BoundaryKind.NONE.value,
                uncertainty=0.0,
                confidence=1.0,
                scope=EvidenceScope.TRANSFERABLE,
            )
        )

    prediction = store.infer_action(
        "events:moved|objects:0",
        task_id="task",
    )

    assert prediction.action == (2, -1, -1)
    assert prediction.support == 3
    assert prediction.confidence == pytest.approx(2.0 / 3.0)


def test_uncertainty_is_bonus_for_safe_real_action_and_penalty_in_imagination() -> None:
    agent = CompactHunterSeeker(config=_config())
    snapshot = agent._commit_initial_snapshot(_obs(2, actions=(0,)))
    scorer = PolicyScorer(agent.config.policy)
    evidence = EvidenceStore().score(
        task_id="task",
        stage=1,
        state_id=snapshot.state_id,
        action=Action(0),
    )
    prediction = _prediction(uncertainty=0.8, hazard=0.1)
    competence = CompetenceState(
        expected_learning_gain=1.0,
        remaining_risk_budget=1.0,
    )

    real = scorer.score(
        snapshot=snapshot,
        action=Action(0),
        prediction=prediction,
        evidence=evidence,
        competence=competence,
        imagined=False,
    )
    imagined = scorer.score(
        snapshot=snapshot,
        action=Action(0),
        prediction=prediction,
        evidence=evidence,
        competence=competence,
        imagined=True,
    )

    real_terms = {term.name: term.value for term in real.terms}
    imagined_terms = {term.name: term.value for term in imagined.terms}
    assert real_terms["epistemic_exploration"] > 0.0
    assert imagined_terms["model_uncertainty_penalty"] < 0.0


def test_aggregate_risk_blocks_exploration_bonus_even_when_model_hazard_is_low() -> None:
    agent = CompactHunterSeeker(config=_config())
    snapshot = agent._commit_initial_snapshot(_obs(2, actions=(0,)))
    scorer = PolicyScorer(agent.config.policy)
    evidence = EvidenceScore(
        total=-1.0,
        positive=0.0,
        negative=1.0,
        no_change=0.0,
        support=1,
        negative_support=1,
        positive_support=0,
    )
    candidate = scorer.score(
        snapshot=snapshot,
        action=Action(0),
        prediction=_prediction(uncertainty=1.0, hazard=0.1),
        evidence=evidence,
        competence=CompetenceState(
            expected_learning_gain=1.0,
            remaining_risk_budget=1.0,
        ),
        imagined=False,
    )

    terms = {term.name: term.value for term in candidate.terms}
    assert RiskArbiter.effective_risk(candidate) > agent.config.policy.risk_limit
    assert terms["epistemic_exploration"] == 0.0
    assert terms["expected_learning_progress"] == 0.0


def test_score_conservation_and_random_exploration_share_risk_gate() -> None:
    safe = Candidate(
        action=Action(0),
        prediction=_prediction(progress=0.1, hazard=0.1),
        terms=(ScoreTerm("safe", 0.1, "value"),),
    )
    risky = Candidate(
        action=Action(1),
        prediction=_prediction(progress=1.0, value=1.0, hazard=1.0),
        terms=(ScoreTerm("risky", 10.0, "value"),),
    )
    assert safe.score == sum(term.value for term in safe.terms)
    assert risky.score == sum(term.value for term in risky.terms)

    arbiter = RiskArbiter(PolicyConfig(risk_limit=0.5, exploration_epsilon=1.0))
    for seed in range(10):
        result = arbiter.choose(
            (risky, safe),
            rng=np.random.default_rng(seed),
            epsilon=1.0,
        )
        assert result.candidate.action == safe.action
        assert result.method == "safe_random"


class _Teacher:
    teacher_id = "teacher"

    def __init__(self) -> None:
        self.suggest_calls = 0

    def suggest(self, request: TeacherRequest):
        self.suggest_calls += 1
        return None

    def examples(self, query: TeacherQuery):
        return (
            TeacherExample(
                task_id=query.task_id,
                stage=query.stage or 1,
                state_id="s",
                action=Action(1),
                weight=4.0,
            ),
        )


def test_student_distillation_moves_runtime_prior_without_teacher_read_in_act() -> None:
    provider = _Teacher()
    guard = TeacherAccessGuard(provider, mode=RuntimeMode.STUDENT)
    agent = CompactHunterSeeker(
        config=_config(mode=RuntimeMode.STUDENT),
        teacher=guard,
        safe_action_provider=lambda _obs: (0, 1),
    )
    before_score = agent.prior.score("task", 1)
    assert agent.distill_teacher(TeacherQuery(task_id="task", stage=1)) == 1
    after_score = agent.prior.score("task", 1)
    assert after_score > before_score

    agent.act(_obs(2))
    assert provider.suggest_calls == 0
    assert guard.audit.action_selection_reads == 0


def test_save_load_reproduces_first_resumed_decision(tmp_path) -> None:
    config = _config(seed=17)
    agent = CompactHunterSeeker(
        config=config,
        safe_action_provider=lambda _obs: (0, 1),
    )
    before = _obs(2)
    after = _obs(3)
    decision = agent.act(before)
    agent.observe(decision, after, Outcome())
    checkpoint = tmp_path / "compact.json"
    agent.save_checkpoint(str(checkpoint))

    expected = agent.act(after)

    restored = CompactHunterSeeker(
        config=config,
        safe_action_provider=lambda _obs: (0, 1),
    )
    restored.load_checkpoint(str(checkpoint))
    restored.begin_run("task", after)
    actual = restored.act(after)

    assert actual.action == expected.action
    assert actual.score == pytest.approx(expected.score, abs=1e-8)
    assert [candidate.action for candidate in actual.candidates] == [
        candidate.action for candidate in expected.candidates
    ]


class _OtherGridBackend(GridFeatureBackend):
    pass


def test_checkpoint_rejects_representation_backend_mismatch(tmp_path) -> None:
    config = _config()
    agent = CompactHunterSeeker(
        config=config,
        safe_action_provider=lambda _obs: (0,),
    )
    checkpoint = tmp_path / "backend.json"
    agent.save_checkpoint(str(checkpoint))
    mismatched = CompactHunterSeeker(
        config=config,
        representation_backend=_OtherGridBackend(
            latent_dim=config.model.latent_dim
        ),
        safe_action_provider=lambda _obs: (0,),
    )

    with pytest.raises(ValueError, match="representation backend"):
        mismatched.load_checkpoint(str(checkpoint))


def test_repeated_training_makes_observed_good_action_beat_observed_death() -> None:
    agent = CompactHunterSeeker(
        config=_config(seed=9),
        safe_action_provider=lambda _obs: (0, 1),
    )
    start = _obs(2, actions=(0,))
    good_after = _obs(3, actions=(0,), progress=1.0)
    for _ in range(3):
        agent.begin_run("task", start)
        decision = agent.act(start)
        assert decision.action.index == 0
        agent.observe(
            decision,
            good_after,
            Outcome(reward=1.0, progress_delta=1.0),
        )
        agent.end_run()

    bad_start = _obs(2, actions=(1,))
    bad_after = _obs(2, actions=(1,))
    for _ in range(3):
        agent.begin_run("task", bad_start)
        decision = agent.act(bad_start)
        assert decision.action.index == 1
        agent.observe(
            decision,
            bad_after,
            Outcome(
                terminated=True,
                boundary=BoundaryKind.DEATH,
                hazard=1.0,
            ),
        )
        agent.end_run()

    evaluation = _obs(2, actions=(0, 1))
    agent.begin_run("task", evaluation)
    chosen = agent.act(evaluation)

    assert chosen.action.index == 0
    by_action = {candidate.action.index: candidate for candidate in chosen.candidates}
    assert by_action[0].score > by_action[1].score


def test_replay_updates_actual_model_and_is_not_available_to_action_lookup() -> None:
    config = AgentConfig(
        seed=5,
        search=SearchConfig(beam_width=2, horizon=1),
        policy=PolicyConfig(exploration_epsilon=0.0),
        model=ModelConfig(ensemble_size=2, latent_dim=12),
        learning=LearningConfig(
            replay_capacity=16,
            replay_every=1,
            replay_batch_size=1,
            replay_updates=1,
            teacher_fraction=0.0,
        ),
    )
    agent = CompactHunterSeeker(
        config=config,
        safe_action_provider=lambda _obs: (0,),
    )
    before = _obs(2, actions=(0,))
    after = _obs(3, actions=(0,))
    decision = agent.act(before)
    updates_before = agent.dynamics.update_count
    weights_before = [row.copy() for row in agent.dynamics._weights]
    agent.observe(decision, after, Outcome(reward=1.0))

    assert len(agent.buffer) == 1
    assert agent.dynamics.update_count >= updates_before + 2
    assert any(
        not np.array_equal(before_row, after_row)
        for before_row, after_row in zip(weights_before, agent.dynamics._weights)
    )
    assert agent.learner.replay_updates == 1
    assert not hasattr(agent.buffer, "score")
    assert not hasattr(agent.buffer, "lookup_action")


def test_role_taps_are_wired_as_bounded_terms_and_survival_only_retains_top_k() -> None:
    calibration = TapCalibration(
        calibration_id="unit-heldout",
        sample_count=32,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    value_tap = CalibratedPointwiseTap(
        tap_id="value",
        role=TapRole.VALUE,
        calibration=calibration,
        scorer=lambda _snapshot, candidate: float(candidate.action.index) / 2.0,
    )
    pair_tap = AntisymmetricPairwiseTap(
        tap_id="pair",
        calibration=calibration,
        scorer=lambda _snapshot, left, right: float(
            left.action.index - right.action.index
        ),
    )
    survival_calibration = TapCalibration(
        calibration_id="survival-heldout",
        sample_count=32,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    survival = SurvivalRetainer(
        CalibratedPointwiseTap(
            tap_id="survival",
            role=TapRole.SURVIVAL,
            calibration=survival_calibration,
            scorer=lambda _snapshot, candidate: float(candidate.action.index) / 2.0,
            influences_score=False,
        )
    )
    agent = CompactHunterSeeker(
        config=AgentConfig(
            seed=1,
            search=SearchConfig(beam_width=2, horizon=1),
            policy=PolicyConfig(exploration_epsilon=0.0),
            model=ModelConfig(ensemble_size=2, latent_dim=12),
        ),
        safe_action_provider=lambda _obs: (0, 1, 2),
        tap_bundle=TapBundle(
            pointwise=(value_tap,),
            pairwise=(pair_tap,),
            survival=survival,
        ),
    )
    decision = agent.act(_obs(2, actions=(0, 1, 2)))

    assert len(decision.candidates) == 2
    assert {candidate.action.index for candidate in decision.candidates} == {1, 2}
    for candidate in decision.candidates:
        names = {term.name for term in candidate.terms}
        assert "tap:value:value" in names
        assert "tap:pairwise" in names


class _VerifiedRuleModel:
    model_id = "rule"
    model_version = "1"
    complexity = 1.0

    def __init__(self, state_id: str) -> None:
        self.state_id = state_id

    def predict(self, state: ExecutableState, action: Action) -> ExecutablePrediction:
        if state.state_id == self.state_id and action.index == 1:
            return ExecutablePrediction(
                applicable=True,
                successor_state_id="rule-successor",
                change_probability=1.0,
                progress_delta=1.0,
                hazard=0.0,
            )
        return ExecutablePrediction(applicable=False)

    def plan(self, state, actions, *, horizon):
        action = next((row for row in actions if row.index == 1), None)
        if action is None:
            return None
        return ExecutablePlan(
            model_id=self.model_id,
            model_version=self.model_version,
            actions=(action,),
            score=1.0,
        )


def test_replay_verified_executable_model_can_predict_and_plan_in_live_search() -> None:
    observation = _obs(2, actions=(0, 1))
    bootstrap_agent = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _obs: (0, 1),
    )
    bootstrap_agent.begin_run("task", observation)
    snapshot = bootstrap_agent.current_snapshot
    assert snapshot is not None

    registry = ExecutableModelRegistry(
        policy=ReplayVerificationPolicy(
            minimum_cases=1,
            minimum_coverage=1.0,
            maximum_error_rate=0.0,
        )
    )
    model = _VerifiedRuleModel(snapshot.state_id)
    registry.register(model)
    report = registry.verify(
        model.model_id,
        (
            ReplayCase(
                case_id="case",
                before=ExecutableState.from_snapshot(snapshot),
                action=Action(1),
                after_state_id="rule-successor",
                frame_changed=True,
                progress_delta=1.0,
            ),
        ),
    )
    assert report.verified

    agent = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _obs: (0, 1),
        executable_registry=registry,
    )
    decision = agent.act(observation)

    assert decision.action.index == 1
    chosen = next(
        row for row in decision.candidates if row.action == decision.action
    )
    assert chosen.prediction.source == "executable:rule"
    assert any(term.name == "executable_plan" for term in chosen.terms)


def test_nonfinite_model_outputs_fail_safe_and_scores_remain_finite() -> None:
    prediction = Prediction(
        change_probability=float("nan"),
        progress=float("inf"),
        value=float("-inf"),
        hazard=float("nan"),
        terminal=float("nan"),
        uncertainty=float("nan"),
        latent_delta=np.asarray([np.nan, np.inf], dtype=np.float32),
        object_delta=np.asarray([np.nan], dtype=np.float32),
    )
    candidate = Candidate(
        action=Action(0),
        prediction=prediction,
        terms=(
            ScoreTerm("nan", float("nan"), "diagnostic"),
            ScoreTerm("finite", 0.25, "value"),
        ),
    )

    assert prediction.change_probability == 0.0
    assert prediction.progress == 0.0
    assert prediction.value == 0.0
    assert prediction.hazard == 1.0
    assert prediction.terminal == 1.0
    assert prediction.uncertainty == 1.0
    assert np.isfinite(prediction.latent_delta).all()
    assert np.isfinite(prediction.object_delta).all()
    assert candidate.score == pytest.approx(0.25)


def test_npz_teacher_trains_student_models_without_becoming_runtime_graph_lookup(
    tmp_path,
) -> None:
    frame = np.zeros((1, 5, 7), dtype=np.uint8)
    frame[0, 1:3, 1:3] = 2
    frame_after = frame.copy()
    frame_after[0, 1:3, 2:4] = 2
    path = tmp_path / "teacher.npz"
    np.savez(
        path,
        frames=frame,
        frames_after=frame_after,
        actions=np.asarray([[0, -1, -1]], dtype=np.int64),
        levels=np.asarray([0], dtype=np.int64),
    )
    provider = TrajectoryTeacher.from_npz(str(path), task_id="task")
    guard = TeacherAccessGuard(provider, mode=RuntimeMode.STUDENT)
    agent = CompactHunterSeeker(
        config=_config(mode=RuntimeMode.STUDENT),
        teacher=guard,
        safe_action_provider=lambda _obs: (0,),
    )
    weights_before = [row.copy() for row in agent.dynamics._weights]

    count = agent.distill_teacher(TeacherQuery(task_id="task"))

    assert count == 1
    assert len(agent.buffer) == 1
    assert agent.buffer.expert_fraction() == 1.0
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(weights_before, agent.dynamics._weights)
    )
    assert len(agent.graph) == 0
    assert len(agent.evidence) == 0
    assert guard.audit.action_selection_reads == 0


def test_graph_score_terms_follow_search_config() -> None:
    agent = CompactHunterSeeker(config=_config())
    snapshot = agent._commit_initial_snapshot(_obs(2, actions=(0,)))
    scorer = PolicyScorer(
        agent.config.policy,
        search=SearchConfig(
            exact_graph_bonus=0.21,
            no_change_penalty=0.42,
            loop_penalty=0.33,
        ),
    )
    evidence = EvidenceStore().score(
        task_id="task",
        stage=1,
        state_id=snapshot.state_id,
        action=Action(0),
    )

    candidate = scorer.score(
        snapshot=snapshot,
        action=Action(0),
        prediction=_prediction(),
        evidence=evidence,
        competence=CompetenceState(),
        graph_stats={"known": True, "no_change_rate": 1.0, "self_loop_rate": 1.0},
    )

    terms = {term.name: term.value for term in candidate.terms}
    assert terms["exact_graph"] == pytest.approx(0.21)
    assert terms["known_no_change"] == pytest.approx(-0.42)
    assert terms["known_loop"] == pytest.approx(-0.33)
    assert agent.scorer.search is agent.config.search


def test_hazard_miscalibration_brakes_epistemic_exploration() -> None:
    agent = CompactHunterSeeker(config=_config())
    snapshot = agent._commit_initial_snapshot(_obs(2, actions=(0,)))
    scorer = PolicyScorer(agent.config.policy)
    evidence = EvidenceStore().score(
        task_id="task",
        stage=1,
        state_id=snapshot.state_id,
        action=Action(0),
    )
    prediction = _prediction(uncertainty=0.8, hazard=0.1)

    def exploration_term(competence: CompetenceState) -> float:
        candidate = scorer.score(
            snapshot=snapshot,
            action=Action(0),
            prediction=prediction,
            evidence=evidence,
            competence=competence,
        )
        return {term.name: term.value for term in candidate.terms}[
            "epistemic_exploration"
        ]

    calibrated = exploration_term(
        CompetenceState(expected_learning_gain=1.0, remaining_risk_budget=1.0)
    )
    miscalibrated = exploration_term(
        CompetenceState(
            expected_learning_gain=1.0,
            remaining_risk_budget=1.0,
            hazard_calibration_error=0.8,
        )
    )

    assert calibrated > 0.0
    assert 0.0 <= miscalibrated < calibrated
    assert miscalibrated == pytest.approx(calibrated * 0.2)


def test_incremental_graph_distances_match_full_recompute() -> None:
    agent = CompactHunterSeeker(config=_config(), safe_action_provider=lambda _obs: (0,))
    state_a = _obs(1, actions=(0, 1))
    state_b = _obs(2, actions=(0, 1))
    state_c = _obs(3, actions=(0, 1), progress=1.0)
    state_d = _obs(4, actions=(0, 1), progress=1.0)
    agent.begin_run("task", state_a)

    walk = (
        (state_b, Outcome()),
        (state_c, Outcome(progress_delta=1.0)),
        (state_d, Outcome()),
        (state_a, Outcome()),
    )
    observation = state_a
    for next_observation, outcome in walk:
        decision = agent.act(observation)
        agent.observe(decision, next_observation, outcome)
        observation = next_observation

    assert agent.graph.node(state_c.state_id).distance_to_progress == 0
    assert agent.graph.node(state_b.state_id).distance_to_progress == 1
    assert agent.graph.node(state_a.state_id).distance_to_progress == 2
    assert agent.graph.node(state_d.state_id).distance_to_progress == 3

    exported = agent.graph.export_state()
    reloaded = StateGraph.from_state(exported)
    for state_id, row in exported["nodes"].items():
        assert (
            reloaded.node(state_id).distance_to_progress
            == row["distance_to_progress"]
        )


def test_cross_task_positive_transfer_requires_matched_task_support() -> None:
    def _positive_record(
        *,
        transition_id: str,
        task_id: str,
        effect_signature: str,
        scope: EvidenceScope,
    ) -> EvidenceRecord:
        return EvidenceRecord(
            transition_id=transition_id,
            task_id=task_id,
            stage=1,
            step=0,
            state_id=f"state-{transition_id}",
            successor_id=f"next-{transition_id}",
            action=(1, -1, -1),
            object_signature="",
            effect_signature=effect_signature,
            event_kinds=("progress",),
            frame_changed=True,
            progress=1.0,
            reward=0.0,
            hazard=0.0,
            terminal=False,
            boundary=BoundaryKind.NONE.value,
            uncertainty=0.0,
            confidence=1.0,
            scope=scope,
        )

    store = EvidenceStore()
    store.append(
        _positive_record(
            transition_id="cross-task",
            task_id="task-a",
            effect_signature="events:moved|progress:1",
            scope=EvidenceScope.TRANSFERABLE,
        )
    )
    # A same-task positive record whose signatures do not match the query must
    # not exempt cross-task transfer from the repeated-support requirement.
    store.append(
        _positive_record(
            transition_id="unrelated-same-task",
            task_id="task-b",
            effect_signature="events:transformed|objects:3",
            scope=EvidenceScope.TASK,
        )
    )

    single_support = store.score(
        task_id="task-b",
        stage=1,
        state_id="query-state",
        action=Action(1),
        effect_signature="events:moved|progress:1",
    )
    assert single_support.positive == 0.0
    assert single_support.positive_support == 1

    store.append(
        _positive_record(
            transition_id="cross-task-2",
            task_id="task-c",
            effect_signature="events:moved|progress:1",
            scope=EvidenceScope.TRANSFERABLE,
        )
    )
    repeated_support = store.score(
        task_id="task-b",
        stage=1,
        state_id="query-state",
        action=Action(1),
        effect_signature="events:moved|progress:1",
    )
    assert repeated_support.positive > 0.0
    assert repeated_support.positive_support == 2
