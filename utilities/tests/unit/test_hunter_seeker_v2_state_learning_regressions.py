from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from hunter_seeker_v2.contracts import (
    Action,
    BoundaryKind,
    CompetenceState,
    EgoConfig,
    EventKind,
    ExogenousConfig,
    HypothesisConfig,
    LearningConfig,
    MemoryConfig,
    ModelConfig,
    ObjectState,
    Observation,
    Outcome,
    Representation,
    Topology,
    Transition,
    WorldEvent,
    WorldSnapshot,
)
from hunter_seeker_v2.ego import ControlAttribution
from hunter_seeker_v2.exogenous import ExogenousChangeFilter
from hunter_seeker_v2.hypotheses import Hypothesis, RelationalHypothesisEngine
from hunter_seeker_v2.learning import CompactLearner, ReplayBuffer, ReplayItem
from hunter_seeker_v2.memory import (
    EvidenceRecord,
    EvidenceScope,
    EvidenceStore,
    StateGraph,
    evidence_from_transition,
)
from hunter_seeker_v2.models import (
    ActionPrior,
    AffordanceModel,
    CompetenceMonitor,
    DynamicsEnsemble,
)
from hunter_seeker_v2.perception import PerceptionSystem


def _snapshot(
    frame: np.ndarray,
    *,
    task_id: str = "task",
    stage: int = 1,
    step: int = 0,
    progress: float = 0.0,
    latent_dim: int = 8,
) -> WorldSnapshot:
    observation = Observation(
        frame=np.asarray(frame),
        available_actions=(0, 1, 2),
        task_id=task_id,
        stage=stage,
        progress=progress,
    )
    return WorldSnapshot(
        observation=observation,
        objects=(),
        events=(),
        topology=Topology(),
        representation=Representation(
            global_vector=np.zeros(latent_dim, dtype=np.float32),
            spatial=np.zeros((1, 1), dtype=np.float32),
        ),
        step=step,
    )


def _transition(
    *,
    task_id: str = "task",
    action: Action = Action(1),
    outcome: Outcome = Outcome(),
    before_frame: np.ndarray | None = None,
    after_frame: np.ndarray | None = None,
    transition_id: str = "transition",
) -> tuple[WorldSnapshot, WorldSnapshot, Transition]:
    before_arr = (
        np.zeros((2, 2), dtype=np.uint8)
        if before_frame is None
        else np.asarray(before_frame)
    )
    after_arr = (
        before_arr.copy() if after_frame is None else np.asarray(after_frame)
    )
    before = _snapshot(before_arr, task_id=task_id)
    after = _snapshot(after_arr, task_id=task_id, step=1)
    transition = Transition(
        transition_id=transition_id,
        decision_id=f"decision-{transition_id}",
        task_id=task_id,
        stage=1,
        step=0,
        before=before,
        action=action,
        after_observation=after.observation,
        outcome=outcome,
        frame_changed=not np.array_equal(before_arr, after_arr),
        after_state_id=after.state_id,
    )
    return before, after, transition


def test_success_completion_is_not_negative_or_terminal_risk() -> None:
    before, after, transition = _transition(
        outcome=Outcome(
            reward=1.0,
            progress_delta=1.0,
            terminated=True,
            boundary=BoundaryKind.GAME_COMPLETED,
        ),
        after_frame=np.ones((2, 2), dtype=np.uint8),
    )
    record = evidence_from_transition(transition, uncertainty=0.0)
    assert record.positive is True
    assert record.completed is True
    assert record.adverse_terminal is False
    assert record.negative is False

    graph = StateGraph()
    graph.observe(
        transition,
        before_latent=before.representation.global_vector,
        after_latent=after.representation.global_vector,
        before_object_summary=np.zeros(8, dtype=np.float32),
        after_object_summary=np.zeros(8, dtype=np.float32),
        uncertainty=0.0,
    )
    prediction = graph.exact_prediction(
        before.state_id,
        transition.action,
        latent_dim=8,
        object_dim=8,
    )
    assert prediction is not None
    assert prediction.terminal == 0.0

    dynamics = DynamicsEnsemble(ModelConfig(ensemble_size=2, latent_dim=8), seed=4)
    assert dynamics._target(before, after, transition)[4] == 0.0

    affordances = AffordanceModel()
    affordances.observe(object_signature="goal", transition=transition)
    assert affordances.belief("goal", task_id="task")[1:] == pytest.approx(
        (0.0, np.tanh(2.0) / 3.0, 0.0)
    )


def test_negative_priors_and_affordances_are_task_local_but_positive_transfers() -> None:
    prior = ActionPrior()
    _before, _after, death = _transition(
        task_id="task-a",
        outcome=Outcome(
            terminated=True,
            boundary=BoundaryKind.DEATH,
            hazard=1.0,
        ),
        transition_id="death",
    )
    prior.observe(death)
    assert prior.score("task-a", 1) < 0.0
    assert prior.score("unseen-task", 1) == 0.0

    _before, _after, progress = _transition(
        task_id="task-a",
        action=Action(2),
        outcome=Outcome(reward=1.0),
        transition_id="progress",
    )
    prior.observe(progress)
    assert prior.score("unseen-task", 2) == 0.0
    prior.observe(progress, transferable=True)
    assert prior.score("unseen-task", 2) > 0.0

    affordances = AffordanceModel()
    affordances.observe_signature(
        "shared-shape",
        task_id="task-a",
        hazard=1.0,
        terminal=1.0,
    )
    assert affordances.belief("shared-shape", task_id="task-a")[1] > 0.0
    assert affordances.belief("shared-shape", task_id="task-b")[1] == 0.0
    affordances.observe_signature(
        "shared-shape",
        task_id="task-a",
        changed=1.0,
        reward=1.0,
        transferable_positive=True,
    )
    transferred = affordances.belief("shared-shape", task_id="task-b")
    assert transferred[0] > 0.0
    assert transferred[2] > 0.0
    assert transferred[1] == 0.0
    assert transferred[3] == 0.0


def test_opt_in_negative_evidence_transfer_requires_object_or_effect_match() -> None:
    record = EvidenceRecord(
        transition_id="hazard-a",
        task_id="task-a",
        stage=1,
        step=0,
        state_id="state-a",
        successor_id="terminal-a",
        action=Action(1).key,
        object_signature="shape|red",
        effect_signature="collision|death",
        event_kinds=("hazard",),
        frame_changed=True,
        progress=0.0,
        reward=0.0,
        hazard=1.0,
        terminal=True,
        boundary=BoundaryKind.DEATH.value,
        uncertainty=0.0,
        confidence=1.0,
        scope=EvidenceScope.TASK,
    )
    disabled = EvidenceStore(MemoryConfig(negative_transfer=False))
    enabled = EvidenceStore(MemoryConfig(negative_transfer=True))
    disabled.append(record)
    enabled.append(record)

    query = dict(
        task_id="task-b",
        stage=1,
        state_id="state-b",
        action=Action(1),
        object_signature="shape|red",
        effect_signature="collision|death",
    )
    assert disabled.score(**query).negative == 0.0
    assert enabled.score(**query).negative > 0.0
    assert enabled.score(**{**query, "object_signature": "", "effect_signature": ""}).negative == 0.0


def test_replay_updates_weights_without_inflating_empirical_support() -> None:
    before, after, transition = _transition(
        outcome=Outcome(reward=1.0),
        after_frame=np.ones((2, 2), dtype=np.uint8),
    )
    dynamics = DynamicsEnsemble(
        ModelConfig(ensemble_size=2, latent_dim=8, bootstrap_probability=1.0),
        seed=5,
    )
    prior = ActionPrior()
    affordances = AffordanceModel()
    learner = CompactLearner(
        dynamics=dynamics,
        prior=prior,
        affordances=affordances,
        config=LearningConfig(
            replay_every=1,
            replay_batch_size=1,
            replay_updates=4,
            teacher_fraction=0.0,
        ),
        seed=6,
    )
    item = ReplayItem(
        before=before,
        after=after,
        transition=transition,
        target_object_signature="target",
    )
    learner.observe(item)
    prior_state = prior.state_dict()
    affordance_state = affordances.state_dict()
    support = dict(dynamics._support)
    learner.replay_step()

    assert dynamics.update_count == 5
    assert dynamics._support == support
    assert prior.state_dict() == prior_state
    assert affordances.state_dict() == affordance_state


def test_replay_buffer_deepcopy_shares_frozen_items_but_not_queue() -> None:
    before, after, transition = _transition()
    item = ReplayItem(before=before, after=after, transition=transition)
    buffer = ReplayBuffer(4)
    buffer.push(item)
    duplicate = deepcopy(buffer)

    assert duplicate is not buffer
    assert duplicate.items[0] is item
    duplicate.push(item)
    assert len(duplicate) == 2
    assert len(buffer) == 1


def test_dynamics_rng_and_episode_local_competence_roundtrip() -> None:
    dynamics = DynamicsEnsemble(ModelConfig(ensemble_size=2, latent_dim=8), seed=11)
    restored = DynamicsEnsemble.from_state(dynamics.state_dict(), seed=999)
    assert restored._rng.random(8) == pytest.approx(dynamics._rng.random(8))

    monitor = CompetenceMonitor()
    monitor.state = CompetenceState(
        dynamics_error_ema=0.4,
        hazard_calibration_error=0.3,
        predicted_success=0.2,
        recent_realized_progress=0.5,
        model_disagreement=0.1,
        stagnation_count=17,
        expected_learning_gain=0.6,
        remaining_risk_budget=0.2,
        time_pressure=0.9,
    )
    reset = monitor.begin_episode(initial_risk_budget=0.8)
    assert reset.stagnation_count == 0
    assert reset.recent_realized_progress == 0.0
    assert reset.remaining_risk_budget == pytest.approx(0.8)
    assert reset.time_pressure == 0.0
    assert reset.dynamics_error_ema == pytest.approx(0.4)
    assert reset.hazard_calibration_error == pytest.approx(0.3)


def _ego_object(track_id: int, signature: str) -> ObjectState:
    return ObjectState(
        object_id=0,
        track_id=track_id,
        value=3,
        area=1,
        centroid_x=1.0,
        centroid_y=1.0,
        bbox=(1, 1, 1, 1),
        signature=signature,
    )


def test_ego_track_ids_reset_at_episode_boundary_with_explicit_signature_bootstrap() -> None:
    model = ControlAttribution(EgoConfig(min_support=1))
    subject = _ego_object(0, "avatar-shape")
    for _ in range(2):
        for action_index, dx in ((0, -1.0), (1, 1.0)):
            model.observe(
                action=Action(action_index),
                events=(
                    WorldEvent(
                        EventKind.MOVED,
                        subject_track_id=0,
                        object_signature=subject.signature,
                        metadata={"dx": dx, "dy": 0.0},
                    ),
                ),
                visible_objects=(subject,),
            )
    assert model.influence(0) > 0.0
    assert model.influence(0, "unrelated-shape") == 0.0
    model.begin_episode("unrelated-task")
    assert model.influence(0) == 0.0
    assert model.influence(0, "avatar-shape") > 0.0


def test_stage_shape_boundary_does_not_train_old_hypothesis_on_new_frame() -> None:
    engine = RelationalHypothesisEngine(HypothesisConfig())
    hypothesis = Hypothesis(
        hypothesis_id="old-stage",
        kind="uniform",
        region_a=(0, 0, 6, 2),
    )
    engine._by_scope[("task", 1)] = [hypothesis]
    before = Observation(
        frame=np.arange(21, dtype=np.uint8).reshape(3, 7),
        available_actions=(0,),
        task_id="task",
        stage=1,
    )
    after = Observation(
        frame=np.zeros((2, 2), dtype=np.uint8),
        available_actions=(0,),
        task_id="task",
        stage=2,
    )
    assert (
        engine.observe_transition(
            before_observation=before,
            after_observation=after,
            objects=(),
            action=Action(0),
            progressed=False,
            before_memory_id="old",
            after_memory_id="new",
        )
        == 0
    )
    assert hypothesis.deltas == {}
    restored = RelationalHypothesisEngine.from_state(engine.state_dict())
    assert restored.state_dict() == engine.state_dict()


def test_canonical_hypothesis_is_tie_safe_and_honors_exogenous_masks() -> None:
    frame = np.asarray(
        [
            [1, 2, 9, 7],
            [2, 1, 7, 9],
        ],
        dtype=np.uint8,
    )
    hypothesis = Hypothesis(
        hypothesis_id="palette-tie",
        kind="equal_canonical",
        region_a=(0, 0, 1, 1),
        region_b=(2, 0, 3, 1),
        lattice=(2, 2),
    )
    assert hypothesis.potential(frame) == 0.0

    noisy = frame.copy()
    noisy[0, 3] = 8
    assert hypothesis.potential(noisy) > 0.0
    mask = frozenset({(0, 3)})
    assert hypothesis.potential(noisy, mask) == 0.0
    assert (3, 0) not in hypothesis.mismatch_cells(noisy, mask)


def _record_clock_episode(
    filter_: ExogenousChangeFilter,
    actions: tuple[int, int],
) -> None:
    filter_.begin_episode("task")
    frames = (
        np.asarray([[0, 0]], dtype=np.uint8),
        np.asarray([[1, 0]], dtype=np.uint8),
        np.asarray([[1, 0]], dtype=np.uint8),
    )
    for index, action in enumerate(actions):
        filter_.observe_transition(
            task_id="task",
            before_frame=frames[index],
            after_frame=frames[index + 1],
            action=Action(action),
        )
    assert filter_.end_episode() is True


def test_exogenous_requires_intervention_on_event_and_persists_active_episode() -> None:
    config = ExogenousConfig(min_common_horizon=2, min_confirmations=1)
    late_only = ExogenousChangeFilter(config)
    _record_clock_episode(late_only, (0, 0))
    _record_clock_episode(late_only, (0, 1))
    assert late_only.mask_cells("task", (1, 2)) == frozenset()

    direct = ExogenousChangeFilter(config)
    _record_clock_episode(direct, (0, 0))
    _record_clock_episode(direct, (1, 0))
    assert direct.mask_cells("task", (1, 2)) == frozenset({(0, 0)})
    assert direct.finalized_episodes == 2
    frame = np.asarray([[7, 0]], dtype=np.uint8)
    base = Observation(frame, (0,), task_id="task", progress=0.0)
    other_actions = Observation(frame, (1,), task_id="task", progress=0.0)
    other_progress = Observation(frame, (0,), task_id="task", progress=0.5)
    assert direct.masked_state_id(base) != direct.masked_state_id(other_actions)
    assert direct.masked_state_id(base) != direct.masked_state_id(other_progress)

    active = ExogenousChangeFilter(config)
    active.begin_episode("task")
    active.observe_transition(
        task_id="task",
        before_frame=np.asarray([[0, 0]], dtype=np.uint8),
        after_frame=np.asarray([[1, 0]], dtype=np.uint8),
        action=Action(0),
    )
    restored = ExogenousChangeFilter.from_state(active.state_dict(), config=config)
    assert restored.state_dict() == active.state_dict()
    assert restored.end_episode() is True
    assert restored.finalized_episodes == 1


def test_perception_resets_track_namespace_on_stage_change() -> None:
    perception = PerceptionSystem()
    first = np.zeros((3, 4), dtype=np.uint8)
    first[1, 1] = 3
    second = np.zeros((3, 4), dtype=np.uint8)
    second[1, 2] = 3
    stage_one = perception.observe(
        Observation(first, (0,), task_id="task", stage=1)
    )
    stage_two = perception.observe(
        Observation(second, (0,), task_id="task", stage=2)
    )

    assert stage_one.objects[0].track_id == 0
    assert stage_two.objects[0].track_id == 0
    assert {event.kind for event in stage_two.events} == {EventKind.APPEARED}


@pytest.mark.parametrize(
    "factory",
    (
        lambda: MemoryConfig(negative_transfer="false"),
        lambda: ExogenousConfig(enabled="false"),
        lambda: HypothesisConfig(enabled="false"),
        lambda: EgoConfig(enabled="false"),
    ),
)
def test_boolean_configuration_rejects_truthy_strings(factory) -> None:
    with pytest.raises(TypeError, match="must be a bool"):
        factory()


def test_model_state_loaders_reject_nonfinite_and_negative_counters() -> None:
    dynamics = DynamicsEnsemble(ModelConfig(ensemble_size=1, latent_dim=8))
    nonfinite = deepcopy(dynamics.state_dict())
    nonfinite["weights"][0][0][0] = float("nan")
    with pytest.raises(ValueError, match="weights must be finite"):
        DynamicsEnsemble.from_state(nonfinite)

    negative_support = deepcopy(dynamics.state_dict())
    negative_support["support"]["state\u241f0,-1,-1"] = -1
    with pytest.raises(ValueError, match="must be >= 0"):
        DynamicsEnsemble.from_state(negative_support)

    prior = ActionPrior()
    prior_state = prior.state_dict()
    prior_state["task"]["task\u241f0"] = [1.0, float("inf")]
    with pytest.raises(ValueError, match="must be finite"):
        ActionPrior.from_state(prior_state)

    affordance = AffordanceModel()
    affordance_state = affordance.state_dict()
    affordance_state["task"]["task\u241fsig"] = {
        "count": -1.0,
        "changed": 0.0,
        "hazard": 0.0,
        "reward": 0.0,
        "terminal": 0.0,
    }
    with pytest.raises(ValueError, match="must be >= 0"):
        AffordanceModel.from_state(affordance_state)

    competence = CompetenceMonitor()
    competence_state = competence.state_dict()
    competence_state["state"]["remaining_risk_budget"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        CompetenceMonitor.from_state(competence_state)


def test_model_state_loaders_reject_coerced_keys_and_impossible_rows() -> None:
    dynamics = DynamicsEnsemble(ModelConfig(ensemble_size=1, latent_dim=8))
    string_weight = deepcopy(dynamics.state_dict())
    string_weight["weights"][0][0][0] = "0.0"
    with pytest.raises(ValueError, match="weights must contain only numbers"):
        DynamicsEnsemble.from_state(string_weight)

    aliased_support = deepcopy(dynamics.state_dict())
    aliased_support["support"]["state\u241f00,-1,-1"] = 1
    with pytest.raises(ValueError, match="canonical action key"):
        DynamicsEnsemble.from_state(aliased_support)

    prior = ActionPrior()
    nonpositive_transfer = prior.state_dict()
    nonpositive_transfer["global"]["0"] = [1.0, -1.0]
    with pytest.raises(ValueError, match="require positive total support"):
        ActionPrior.from_state(nonpositive_transfer)

    aliased_prior = prior.state_dict()
    aliased_prior["global"]["00"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="canonical non-negative integer string"):
        ActionPrior.from_state(aliased_prior)

    affordance = AffordanceModel()
    impossible_affordance = affordance.state_dict()
    impossible_affordance["task"]["task\u241fsig"] = {
        "count": 1.0,
        "changed": 2.0,
        "hazard": 0.0,
        "reward": 0.0,
        "terminal": 0.0,
    }
    with pytest.raises(ValueError, match="changed cannot exceed count"):
        AffordanceModel.from_state(impossible_affordance)


def test_competence_loader_roundtrips_negative_realized_progress() -> None:
    monitor = CompetenceMonitor()
    monitor.state = CompetenceState(recent_realized_progress=-0.5)

    restored = CompetenceMonitor.from_state(monitor.state_dict())

    assert restored.state == monitor.state


def test_memory_state_loaders_reject_coerced_and_inconsistent_rows() -> None:
    malformed_graph = {
        "nodes": {
            "state": {
                "visits": 0,
                "available_actions": [0],
                "latent": [0.0],
                "object_summary": [0.0],
                "progress": 0.0,
                "terminal": "false",
                "distance_to_progress": None,
                "edges": {},
            }
        },
        "progress_states": [],
        "transition_ids": [],
    }
    with pytest.raises(ValueError, match="must be a bool"):
        StateGraph.from_state(malformed_graph)

    record = EvidenceRecord(
        transition_id="evidence",
        task_id="task",
        stage=1,
        step=0,
        state_id="before",
        successor_id="after",
        action=(0, -1, -1),
        object_signature="",
        effect_signature="",
        event_kinds=(),
        frame_changed=False,
        progress=0.0,
        reward=0.0,
        hazard=0.0,
        terminal=False,
        boundary=BoundaryKind.NONE.value,
        uncertainty=0.0,
        confidence=1.0,
        scope=EvidenceScope.TASK,
    )
    store = EvidenceStore()
    store.append(record)
    missing_tombstone = store.export_state()
    missing_tombstone["seen_transition_ids"] = []
    with pytest.raises(ValueError, match="must include retained"):
        EvidenceStore.from_state(missing_tombstone)

    bad_evidence = store.export_state()
    bad_evidence["records"][0]["hazard"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        EvidenceStore.from_state(bad_evidence)
