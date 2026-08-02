from __future__ import annotations

import copy

import numpy as np
import pytest

from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import (
    Action,
    AgentConfig,
    ModelConfig,
    Observation,
    Outcome,
    PolicyConfig,
    SearchConfig,
    TapRole,
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
from hunter_seeker_v2.models import GridFeatureBackend
from hunter_seeker_v2.representation import (
    FrozenTapRepresentationBackend,
    TapConnector,
)
from hunter_seeker_v2.taps import (
    CalibratedPointwiseTap,
    TapBundle,
    TapCalibration,
)


def _config(*, risk_limit: float = 0.7) -> AgentConfig:
    return AgentConfig(
        seed=19,
        search=SearchConfig(beam_width=2, horizon=1),
        policy=PolicyConfig(
            exploration_epsilon=0.0,
            risk_limit=risk_limit,
        ),
        model=ModelConfig(ensemble_size=2, latent_dim=12),
    )


def _observation(
    value: int,
    *,
    task_id: str = "task",
    actions: tuple[int, ...] = (0,),
    progress: float = 0.0,
) -> Observation:
    return Observation(
        frame=np.asarray([[0, value], [value, 0]], dtype=np.int64),
        available_actions=actions,
        task_id=task_id,
        progress=progress,
    )


def _tap_bundle(value: float) -> TapBundle:
    def scorer(_snapshot, _candidate) -> float:
        return float(value)

    return TapBundle(
        pointwise=(
            CalibratedPointwiseTap(
                tap_id="checkpoint-value",
                role=TapRole.VALUE,
                calibration=TapCalibration(
                    calibration_id="checkpoint-calibration",
                    sample_count=8,
                    lower_bound=-1.0,
                    upper_bound=1.0,
                ),
                scorer=scorer,
            ),
        ),
    )


def _commit_one(agent: CompactHunterSeeker, *, before_value: int, after_value: int) -> None:
    before = _observation(before_value)
    after = _observation(after_value, progress=1.0)
    agent.begin_run("task", before)
    decision = agent.act(before)
    agent.observe(decision, after, Outcome(reward=1.0, progress_delta=1.0))
    agent.end_run()


def test_load_rejects_pending_destination_before_touching_it() -> None:
    source = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
    )
    state = source.state_dict()
    destination = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
    )
    before = _observation(1)
    decision = destination.act(before)
    model_identity = id(destination.dynamics)
    graph_identity = id(destination.graph)

    with pytest.raises(RuntimeError, match="decision is pending"):
        destination.load_state_dict(state)

    assert destination.pending_decision is decision
    assert id(destination.dynamics) == model_identity
    assert id(destination.graph) == graph_identity


def test_load_validates_full_config_and_full_same_class_backend_signature() -> None:
    source = CompactHunterSeeker(
        config=_config(),
        representation_backend=GridFeatureBackend(
            latent_dim=12,
            spatial_shape=(8, 8),
            histogram_bins=16,
        ),
        safe_action_provider=lambda _observation: (0,),
    )
    state = source.state_dict()

    config_mismatch = CompactHunterSeeker(
        config=_config(risk_limit=0.2),
        # Pin the backend to the checkpoint's so the configuration check is
        # the first divergence regardless of the default spatial shape.
        representation_backend=GridFeatureBackend(
            latent_dim=12,
            spatial_shape=(8, 8),
            histogram_bins=16,
        ),
        safe_action_provider=lambda _observation: (0,),
    )
    with pytest.raises(ValueError, match="agent configuration"):
        config_mismatch.load_state_dict(state)

    backend_mismatch = CompactHunterSeeker(
        config=_config(),
        representation_backend=GridFeatureBackend(
            latent_dim=12,
            spatial_shape=(4, 4),
            histogram_bins=16,
        ),
        safe_action_provider=lambda _observation: (0,),
    )
    with pytest.raises(ValueError, match="representation backend"):
        backend_mismatch.load_state_dict(state)


class _WeightedExtractor:
    def __init__(self, weight: float) -> None:
        self.weight = np.asarray([weight], dtype=np.float32)

    def state_dict(self):
        return {"weight": self.weight}

    def __call__(self, observation, objects):
        del observation, objects
        return {"tap": self.weight}


def test_backend_signature_fingerprints_frozen_component_weights() -> None:
    source = CompactHunterSeeker(
        config=_config(),
        representation_backend=FrozenTapRepresentationBackend(
            _WeightedExtractor(1.0),
            connector=TapConnector(latent_dim=12),
        ),
        safe_action_provider=lambda _observation: (0,),
    )
    state = source.state_dict()
    different_weights = CompactHunterSeeker(
        config=_config(),
        representation_backend=FrozenTapRepresentationBackend(
            _WeightedExtractor(2.0),
            connector=TapConnector(latent_dim=12),
        ),
        safe_action_provider=lambda _observation: (0,),
    )

    with pytest.raises(ValueError, match="representation backend"):
        different_weights.load_state_dict(state)


def test_checkpoint_signature_rejects_behaviorally_different_tap_bundle() -> None:
    source = CompactHunterSeeker(
        config=_config(),
        tap_bundle=_tap_bundle(0.25),
        safe_action_provider=lambda _observation: (0,),
    )
    state = source.state_dict()
    matching = CompactHunterSeeker(
        config=_config(),
        tap_bundle=_tap_bundle(0.25),
        safe_action_provider=lambda _observation: (0,),
        agent_id="matching",
    )
    matching.load_state_dict(state)
    assert matching.agent_id == source.agent_id

    different = CompactHunterSeeker(
        config=_config(),
        tap_bundle=_tap_bundle(0.75),
        safe_action_provider=lambda _observation: (0,),
        agent_id="unchanged",
    )
    dynamics_identity = id(different.dynamics)
    tap_identity = id(different._tap_bundle)

    with pytest.raises(ValueError, match="tap bundle"):
        different.load_state_dict(state)

    assert different.agent_id == "unchanged"
    assert id(different.dynamics) == dynamics_identity

    # A weights-only import intentionally preserves the destination's runtime
    # policy dependencies; only a resumable full load requires their identity.
    different.load_state_dict(state, weights_only=True)
    assert different.agent_id == "unchanged"
    assert id(different._tap_bundle) == tap_identity


def test_malformed_late_checkpoint_field_cannot_partially_mutate_agent() -> None:
    source = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
    )
    _commit_one(source, before_value=1, after_value=2)
    state = copy.deepcopy(source.state_dict())
    state["runtime"]["rng_state"] = {"not": "a numpy bit generator state"}

    destination = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
        agent_id="destination",
    )
    _commit_one(destination, before_value=3, after_value=4)
    identities = {
        name: id(getattr(destination, name))
        for name in (
            "dynamics",
            "prior",
            "affordances",
            "competence",
            "ego",
            "graph",
            "evidence",
            "perception",
            "exogenous",
            "hypotheses",
            "learner",
            "buffer",
            "diagnostics",
            "search_engine",
        )
    }
    runtime_before = (
        destination.agent_id,
        destination._task_id,
        destination._step,
        destination._decision_counter,
        destination._run_active,
        destination._run_transition_count,
        destination._total_transition_count,
    )

    with pytest.raises(ValueError):
        destination.load_state_dict(state)

    assert {
        name: id(getattr(destination, name)) for name in identities
    } == identities
    assert (
        destination.agent_id,
        destination._task_id,
        destination._step,
        destination._decision_counter,
        destination._run_active,
        destination._run_transition_count,
        destination._total_transition_count,
    ) == runtime_before


def test_weights_only_replaces_models_and_preserves_destination_state() -> None:
    source = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
    )
    source.dynamics._weights[0][0, 0] += 2.0
    source.prior.observe_label(task_id="source", action_index=0, weight=3.0)
    source.affordances.observe_signature("source-object", reward=1.0)
    source_state = source.state_dict()

    destination = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
        agent_id="keep-me",
    )
    _commit_one(destination, before_value=5, after_value=6)
    preserved_objects = {
        name: getattr(destination, name)
        for name in (
            "graph",
            "evidence",
            "perception",
            "exogenous",
            "hypotheses",
            "learner",
            "buffer",
            "diagnostics",
        )
    }
    preserved_payloads = {
        "graph": copy.deepcopy(destination.graph.export_state()),
        "evidence": copy.deepcopy(destination.evidence.export_state()),
        "perception": copy.deepcopy(destination.perception.export_state()),
        "exogenous": copy.deepcopy(destination.exogenous.state_dict()),
        "hypotheses": copy.deepcopy(destination.hypotheses.state_dict()),
    }
    replay_items = destination.buffer.items
    runtime_before = (
        destination.agent_id,
        destination._task_id,
        destination._step,
        destination._decision_counter,
        destination.current_snapshot,
        destination._run_active,
        destination._run_transition_count,
        destination._total_transition_count,
        frozenset(destination._committed_decision_ids),
    )

    destination.load_state_dict(source_state, weights_only=True)

    for name, value in preserved_objects.items():
        assert getattr(destination, name) is value
    assert destination.graph.export_state() == preserved_payloads["graph"]
    assert destination.evidence.export_state() == preserved_payloads["evidence"]
    assert destination.perception.export_state() == preserved_payloads["perception"]
    assert destination.exogenous.state_dict() == preserved_payloads["exogenous"]
    assert destination.hypotheses.state_dict() == preserved_payloads["hypotheses"]
    assert destination.buffer.items == replay_items
    assert (
        destination.agent_id,
        destination._task_id,
        destination._step,
        destination._decision_counter,
        destination.current_snapshot,
        destination._run_active,
        destination._run_transition_count,
        destination._total_transition_count,
        frozenset(destination._committed_decision_ids),
    ) == runtime_before
    assert destination.dynamics.state_dict() == source_state["models"]["dynamics"]
    assert destination.prior.state_dict() == source_state["models"]["prior"]
    assert destination.affordances.state_dict() == source_state["models"]["affordances"]
    assert destination.learner.dynamics is destination.dynamics
    assert destination.learner.prior is destination.prior
    assert destination.learner.affordances is destination.affordances


def test_active_run_before_first_observation_roundtrips_checkpoint(
    tmp_path: Any,
) -> None:
    source = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
    )
    source.begin_run("pre-observation")
    assert source._run_active
    assert source.current_snapshot is None

    path = tmp_path / "pre-observation.json"
    source.save_checkpoint(path)
    destination = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
    )
    destination.load_checkpoint(path)

    assert destination._task_id == "pre-observation"
    assert destination._run_active
    assert destination.current_snapshot is None
    assert destination._step == 0
    assert destination._run_transition_count == 0
    decision = destination.act(
        _observation(7, task_id="pre-observation", actions=(0,))
    )
    assert decision.snapshot_id == destination.current_snapshot.state_id


def test_awaiting_visual_reset_roundtrips_and_legacy_default_is_false() -> None:
    source = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
    )
    source.begin_run("task", _observation(3, actions=(0,)))
    source._awaiting_visual_reset = True
    state = source.state_dict()
    assert state["runtime"]["awaiting_visual_reset"] is True

    restored = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
    )
    restored.load_state_dict(state)
    assert restored._awaiting_visual_reset is True

    legacy = copy.deepcopy(state)
    del legacy["runtime"]["awaiting_visual_reset"]
    restored.load_state_dict(legacy)
    assert restored._awaiting_visual_reset is False


@pytest.mark.parametrize(
    ("runtime_patch", "message"),
    [
        (
            {
                "task_id": None,
                "run_active": True,
                "current_snapshot": None,
            },
            "active checkpoint run requires a runtime task_id",
        ),
        (
            {
                "task_id": "task",
                "run_active": True,
                "current_snapshot": None,
                "step": 1,
            },
            "without a current snapshot cannot contain in-run progress",
        ),
        (
            {"run_active": "false"},
            "run_active must be a bool",
        ),
        (
            {"awaiting_visual_reset": 1},
            "awaiting_visual_reset must be a bool",
        ),
        (
            {
                "task_id": "task",
                "run_active": True,
                "current_snapshot": None,
                "awaiting_visual_reset": True,
            },
            "awaiting a visual reset requires a current snapshot",
        ),
    ],
)
def test_malformed_checkpoint_task_runtime_combinations_fail_closed(
    runtime_patch: dict[str, Any],
    message: str,
) -> None:
    source = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
    )
    source.begin_run("task")
    state = copy.deepcopy(source.state_dict())
    state["runtime"].update(runtime_patch)
    destination = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0,),
        agent_id="unchanged",
    )

    with pytest.raises(ValueError, match=message):
        destination.load_state_dict(state)

    assert destination.agent_id == "unchanged"
    assert destination._task_id is None
    assert destination.current_snapshot is None
    assert destination._run_active is False


class _RuleModel:
    model_version = "1"
    verification_transient_fields = ("plan_calls",)

    def __init__(
        self,
        model_id: str,
        *,
        complexity: float = 1.0,
        hazard: float = 0.0,
        change_probability: float = 1.0,
        progress_delta: float = 0.0,
        planned_action: Action | None = None,
    ) -> None:
        self.model_id = model_id
        self.complexity = complexity
        self.hazard = hazard
        self.change_probability = change_probability
        self.progress_delta = progress_delta
        self.planned_action = planned_action
        self.raise_prediction = False
        self.plan_calls = 0

    def predict(self, state: ExecutableState, action: Action) -> ExecutablePrediction:
        if self.raise_prediction:
            raise RuntimeError("reverification failed")
        return ExecutablePrediction(
            applicable=True,
            successor_state_id="successor",
            change_probability=self.change_probability,
            progress_delta=self.progress_delta,
            hazard=self.hazard,
        )

    def plan(self, state, actions, *, horizon):
        self.plan_calls += 1
        if self.planned_action is None:
            return None
        return ExecutablePlan(
            model_id=self.model_id,
            model_version=self.model_version,
            actions=(self.planned_action,),
        )


def _replay_case(*, hazard: float = 0.0) -> tuple[ExecutableState, ReplayCase]:
    bootstrap = CompactHunterSeeker(
        config=_config(),
        safe_action_provider=lambda _observation: (0, 1),
    )
    observation = _observation(1, actions=(0, 1))
    bootstrap.begin_run("task", observation)
    assert bootstrap.current_snapshot is not None
    state = ExecutableState.from_snapshot(bootstrap.current_snapshot)
    case = ReplayCase(
        case_id="case-1",
        before=state,
        action=Action(0),
        after_state_id="successor",
        frame_changed=True,
        hazard=hazard,
    )
    return state, case


def _registry() -> ExecutableModelRegistry:
    return ExecutableModelRegistry(
        policy=ReplayVerificationPolicy(
            minimum_cases=1,
            minimum_coverage=1.0,
            maximum_error_rate=0.0,
            maximum_safety_error=0.0,
        )
    )


def _verified_registry(*, planned_action: Action) -> ExecutableModelRegistry:
    _state, case = _replay_case()
    registry = _registry()
    registry.register(_RuleModel("checkpoint-rules", planned_action=planned_action))
    assert registry.verify("checkpoint-rules", (case,)).verified
    return registry


def test_checkpoint_signature_rejects_different_executable_registry_behavior() -> None:
    source = CompactHunterSeeker(
        config=_config(),
        executable_registry=_verified_registry(planned_action=Action(0)),
        safe_action_provider=lambda _observation: (0, 1),
    )
    state = source.state_dict()
    matching = CompactHunterSeeker(
        config=_config(),
        executable_registry=_verified_registry(planned_action=Action(0)),
        safe_action_provider=lambda _observation: (0, 1),
        agent_id="matching",
    )
    matching.load_state_dict(state)
    assert matching.agent_id == source.agent_id

    different = CompactHunterSeeker(
        config=_config(),
        executable_registry=_verified_registry(planned_action=Action(1)),
        safe_action_provider=lambda _observation: (0, 1),
        agent_id="unchanged",
    )
    graph_identity = id(different.graph)

    with pytest.raises(ValueError, match="executable registry"):
        different.load_state_dict(state)

    assert different.agent_id == "unchanged"
    assert id(different.graph) == graph_identity


def test_duplicate_case_ids_do_not_count_and_invalidate_old_verification() -> None:
    state, case = _replay_case()
    model = _RuleModel("rules", planned_action=Action(0))
    registry = _registry()
    registry.register(model)
    assert registry.verify("rules", (case,)).verified

    with pytest.raises(ValueError, match="duplicate replay case_id"):
        registry.verify("rules", (case, case))

    assert registry.verification("rules") is None
    with pytest.raises(ExecutableModelNotVerified):
        registry.plan(state, (Action(0),), horizon=1, model_id="rules")


def test_safety_auxiliary_error_blocks_verification() -> None:
    _state, hazardous_case = _replay_case(hazard=1.0)
    model = _RuleModel("unsafe", hazard=0.0)
    registry = _registry()
    registry.register(model)

    report = registry.verify("unsafe", (hazardous_case,))

    assert not report.verified
    assert report.matches == 1
    assert report.maximum_safety_error == pytest.approx(1.0)
    assert "safety error" in report.reason


def test_failed_reverification_invalidates_stale_report() -> None:
    _state, case = _replay_case()
    model = _RuleModel("rules")
    registry = _registry()
    registry.register(model)
    assert registry.verify("rules", (case,)).verified
    model.raise_prediction = True

    with pytest.raises(RuntimeError, match="reverification failed"):
        registry.verify("rules", (case,))

    assert registry.verification("rules") is None


def test_ranked_planning_falls_through_inapplicable_model() -> None:
    state, case = _replay_case()
    first = _RuleModel("first", complexity=1.0, planned_action=None)
    second = _RuleModel("second", complexity=2.0, planned_action=Action(1))
    registry = _registry()
    registry.register(first)
    registry.register(second)
    assert registry.verify("first", (case,)).verified
    assert registry.verify("second", (case,)).verified

    plan = registry.plan(state, (Action(0), Action(1)), horizon=1)

    assert plan is not None
    assert plan.model_id == "second"
    assert first.plan_calls == 1
    assert second.plan_calls == 1


def test_nonfinite_executable_safety_outputs_fail_closed() -> None:
    prediction = ExecutablePrediction(
        applicable=True,
        successor_state_id="successor",
        hazard=float("nan"),
        terminal=float("inf"),
    )

    assert prediction.hazard == 1.0
    assert prediction.terminal == 1.0

    _state, safe_case = _replay_case(hazard=0.0)
    registry = _registry()
    registry.register(_RuleModel("nonfinite", hazard=float("nan")))

    report = registry.verify("nonfinite", (safe_case,))

    assert not report.verified
    assert report.maximum_safety_error == pytest.approx(1.0)
    assert "safety error" in report.reason


def test_auxiliary_prediction_error_is_a_verification_gate() -> None:
    _state, case = _replay_case()
    registry = _registry()
    registry.register(
        _RuleModel(
            "wrong-auxiliary",
            change_probability=0.0,
            progress_delta=1_000.0,
        )
    )

    report = registry.verify("wrong-auxiliary", (case,))

    assert not report.verified
    assert report.matches == 1
    assert report.mean_auxiliary_error == pytest.approx(0.5)
    assert "auxiliary error" in report.reason


def test_verified_behavior_mutation_invalidates_report_before_prediction() -> None:
    state, case = _replay_case()
    model = _RuleModel("mutable")
    registry = _registry()
    registry.register(model)
    assert registry.verify("mutable", (case,)).verified

    model.progress_delta = 1.0

    assert registry.verification("mutable") is None
    with pytest.raises(ExecutableModelNotVerified, match="has not passed"):
        registry.predict("mutable", state, Action(0))


def test_model_id_mutation_is_rejected_under_the_registered_key() -> None:
    state, case = _replay_case()
    model = _RuleModel("stable-id", planned_action=Action(0))
    registry = _registry()
    registry.register(model)
    model.model_id = "renamed"

    with pytest.raises(ValueError, match="model_id.*changed"):
        registry.verify("stable-id", (case,))

    assert registry.registered_ids() == ("stable-id",)
    assert registry.verification("stable-id") is None
    with pytest.raises(ExecutableModelNotVerified, match="has not passed"):
        registry.plan(
            state,
            (Action(0),),
            horizon=1,
            model_id="stable-id",
        )

    model.model_id = "stable-id"
    assert registry.verify(" stable-id ", (case,)).verified
    model.model_id = "renamed-after-verification"

    assert registry.verification("stable-id") is None
    with pytest.raises(ExecutableModelNotVerified, match="has not passed"):
        registry.predict("stable-id", state, Action(0))
