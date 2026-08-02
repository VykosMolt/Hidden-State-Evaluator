"""Transactional compact Hunter-Seeker agent."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from hashlib import blake2b
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from .contracts import (
    Action,
    AgentConfig,
    BoundaryKind,
    Candidate,
    Decision,
    EventKind,
    Observation,
    Outcome,
    Prediction,
    RepresentationBackend,
    RuntimeMode,
    ScoreTerm,
    Transition,
    WorldEvent,
    WorldSnapshot,
    frozen_mapping,
    stable_frame_hash,
)
from .diagnostics import Diagnostics
from .ego import ControlAttribution
from .exogenous import ExogenousChangeFilter
from .hypotheses import RelationalHypothesisEngine
from .learning import CompactLearner, ReplayItem
from .memory import (
    EvidenceStore,
    StateGraph,
    combined_object_signature,
    object_summary,
)
from .models import (
    ActionPrior,
    AffordanceModel,
    CompetenceMonitor,
    DynamicsEnsemble,
    GridFeatureBackend,
)
from .perception import PerceptionResult, PerceptionSystem
from .policy import ArbitrationResult, PolicyScorer, RiskArbiter
from .search import SearchEngine, SearchResult, TapReader
from .student import StateConditionedStudentPolicy, StudentTeacherSample
from .taps import TapBundle
from .teacher import (
    Teacher,
    TeacherAccessGuard,
    TeacherQuery,
    TeacherRequest,
)


class AgentStateError(RuntimeError):
    """The caller violated the act/observe transaction state machine."""


class DuplicateDecisionError(AgentStateError):
    """A committed decision was submitted to ``observe`` again."""


@dataclass(slots=True)
class _PendingDecision:
    decision: Decision
    snapshot: WorldSnapshot
    chosen_prediction_source: str


@dataclass(slots=True)
class _RepresentationBackendTransaction:
    """Isolate or snapshot one representation backend for ``observe``.

    The compact built-ins explicitly declare their encoders read-only.  A
    stateful backend can instead provide a conventional
    ``state_dict``/``load_state_dict`` pair; otherwise a private deep-copied
    fork is promoted only when the transaction succeeds.  An uncopyable
    external backend remains usable when it is stateless, preserving the
    minimal RepresentationBackend protocol.
    """

    original: RepresentationBackend
    staged: RepresentationBackend
    restore: Callable[[Any], Any] | None = None
    saved_state: Any = None

    @classmethod
    def prepare(
        cls,
        backend: RepresentationBackend,
    ) -> "_RepresentationBackendTransaction":
        if bool(getattr(backend, "transactionally_stateless", False)):
            return cls(original=backend, staged=backend)

        state_fn = getattr(backend, "state_dict", None)
        load_fn = getattr(backend, "load_state_dict", None)
        if callable(state_fn) and callable(load_fn):
            try:
                saved_state = copy.deepcopy(state_fn())
            except Exception:
                # A deepcopyable backend can still be isolated as a whole.
                pass
            else:
                return cls(
                    original=backend,
                    staged=backend,
                    restore=load_fn,
                    saved_state=saved_state,
                )

        try:
            staged = copy.deepcopy(backend)
        except Exception as exc:
            if bool(getattr(backend, "transactionally_stateful", False)):
                raise AgentStateError(
                    "stateful representation backend must be deepcopyable or "
                    "provide state_dict/load_state_dict for atomic observe"
                ) from exc
            # The protocol intentionally continues to accept opaque frozen or
            # stateless callables that cannot be copied (locks, device handles,
            # extension objects, and similar external resources).
            return cls(original=backend, staged=backend)
        if staged is backend:
            if bool(getattr(backend, "transactionally_stateful", False)):
                raise AgentStateError(
                    "stateful representation backend deepcopy returned itself"
                )
            return cls(original=backend, staged=backend)
        return cls(original=backend, staged=staged)

    def rollback(self) -> None:
        if self.restore is not None:
            self.restore(copy.deepcopy(self.saved_state))

    def commit(self) -> RepresentationBackend:
        """Promote staged state while preserving external identity when possible."""

        if self.staged is self.original:
            return self.original
        original_dict = getattr(self.original, "__dict__", None)
        staged_dict = getattr(self.staged, "__dict__", None)
        if isinstance(original_dict, dict) and isinstance(staged_dict, dict):
            saved = copy.deepcopy(original_dict)
            try:
                promoted = copy.deepcopy(staged_dict)
                original_dict.clear()
                original_dict.update(promoted)
            except BaseException:
                original_dict.clear()
                original_dict.update(saved)
                raise
            return self.original
        return self.staged


SafeActionProvider = Callable[[Observation | None], Sequence[int]]


def _prediction_error(decision: Decision, outcome: Outcome, frame_changed: bool) -> float:
    chosen = next(
        (
            candidate
            for candidate in decision.candidates
            if candidate.action.key == decision.action.key
        ),
        None,
    )
    if chosen is None:
        return 0.0
    prediction = chosen.prediction
    target_hazard = max(outcome.hazard, 1.0 if outcome.failed else 0.0)
    values = [
        abs(prediction.change_probability - float(frame_changed)),
        abs(prediction.progress - outcome.progress_delta),
        abs(prediction.hazard - target_hazard),
        abs(
            prediction.terminal
            - float(outcome.terminated and not outcome.completed)
        ),
    ]
    return float(np.mean(values))


def _target_signature(snapshot: WorldSnapshot, action: Action) -> str:
    if not action.has_position or not snapshot.objects:
        return ""
    target = min(
        snapshot.objects,
        key=lambda obj: (obj.centroid_x - action.x) ** 2
        + (obj.centroid_y - action.y) ** 2,
    )
    return str(target.signature)


def _effect_signature(
    *,
    events: Sequence[WorldEvent],
    before: WorldSnapshot,
    after: WorldSnapshot,
    outcome: Outcome,
) -> str:
    kinds = ",".join(sorted(event.kind.value for event in events))
    object_delta = len(after.objects) - len(before.objects)
    moving = sum(int(event.kind == EventKind.MOVED) for event in events)
    transformed = sum(int(event.kind == EventKind.TRANSFORMED) for event in events)
    return (
        f"events:{kinds}|objects:{object_delta}|moved:{moving}|"
        f"transformed:{transformed}|progress:{int(outcome.progress_delta > 0)}|"
        f"hazard:{int(outcome.hazard > 0 or outcome.failed)}"
    )


class CompactHunterSeeker:
    """Small, composition-based Hunter-Seeker runtime.

    ``act`` is read-only with respect to durable world knowledge except for the
    decision trace and pending transaction.  ``observe`` is the only method
    that commits a real transition and updates learning/memory.
    """

    format_name = "compact_hunter_seeker"
    supports_extended_checkpoint_load = True

    def __init__(
        self,
        *,
        config: AgentConfig | None = None,
        representation_backend: RepresentationBackend | None = None,
        click_action_index: int | None = None,
        safe_action_provider: SafeActionProvider | None = None,
        tap_reader: TapReader | None = None,
        tap_bundle: TapBundle | None = None,
        teacher: Teacher | TeacherAccessGuard | None = None,
        executable_registry: Any | None = None,
        student_policy: StateConditionedStudentPolicy | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.agent_id = str(agent_id or uuid4().hex)
        self.representation_backend = representation_backend or GridFeatureBackend(
            latent_dim=self.config.model.latent_dim
        )
        self.perception = PerceptionSystem(self.config.perception)
        self.graph = StateGraph()
        self.evidence = EvidenceStore(self.config.memory)
        self.prior = ActionPrior()
        self.affordances = AffordanceModel()
        self.ego = ControlAttribution(self.config.ego)
        self.exogenous = ExogenousChangeFilter(self.config.exogenous)
        self.hypotheses = RelationalHypothesisEngine(self.config.hypotheses)
        self.dynamics = DynamicsEnsemble(
            self.config.model,
            seed=self.config.seed,
        )
        self.competence = CompetenceMonitor()
        self.student_policy = student_policy or StateConditionedStudentPolicy()
        self.learner = CompactLearner(
            dynamics=self.dynamics,
            prior=self.prior,
            affordances=self.affordances,
            config=self.config.learning,
            seed=self.config.seed + 1,
        )
        self.buffer = self.learner.replay
        self.scorer = PolicyScorer(self.config.policy, search=self.config.search)
        self.arbiter = RiskArbiter(self.config.policy)
        self._click_action_index = (
            None if click_action_index is None else int(click_action_index)
        )
        self._safe_action_provider = safe_action_provider or (lambda _obs: ())
        self._tap_reader = tap_reader
        self._tap_bundle = tap_bundle
        if teacher is None:
            self.teacher: TeacherAccessGuard | None = None
        elif isinstance(teacher, TeacherAccessGuard):
            if teacher.mode is not self.runtime_mode:
                raise ValueError(
                    "teacher guard mode does not match agent runtime mode"
                )
            self.teacher = teacher
        else:
            self.teacher = TeacherAccessGuard(
                teacher,
                mode=self.runtime_mode,
            )
        self.executable_registry = executable_registry
        self.search_engine = self._new_search_engine()
        self.diagnostics = Diagnostics()
        self._rng = np.random.default_rng(int(self.config.seed))
        self._task_id: str | None = None
        self._step = 0
        self._decision_counter = 0
        self._current_snapshot: WorldSnapshot | None = None
        self._pending: _PendingDecision | None = None
        self._committed_decision_ids: set[str] = set()
        self._last_transition: Transition | None = None
        self._run_active = False
        self._resume_ready = False
        self._awaiting_visual_reset = False
        self._run_transition_count = 0
        self._total_transition_count = 0

    def _new_search_engine(self) -> SearchEngine:
        return SearchEngine(
            model=self.dynamics,
            graph=self.graph,
            evidence=self.evidence,
            prior=self.prior,
            scorer=self.scorer,
            click_action_index=self._click_action_index,
            config=self.config.search,
            tap_reader=self._tap_reader,
            tap_bundle=self._tap_bundle,
            executable_registry=(
                self.executable_registry
                if self.config.enable_executable_models
                else None
            ),
            ego_model=self.ego if self.config.ego.enabled else None,
            hypothesis_engine=(
                self.hypotheses if self.config.hypotheses.enabled else None
            ),
            exogenous_filter=(
                self.exogenous if self.config.exogenous.enabled else None
            ),
        )

    @property
    def runtime_mode(self) -> RuntimeMode:
        return self.config.runtime_mode

    @property
    def transition_count(self) -> int:
        return int(self._total_transition_count)

    @property
    def run_transition_count(self) -> int:
        return int(self._run_transition_count)

    @property
    def current_snapshot(self) -> WorldSnapshot | None:
        return self._current_snapshot

    @property
    def pending_decision(self) -> Decision | None:
        return self._pending.decision if self._pending is not None else None

    def begin_run(
        self,
        task_id: str,
        observation: Observation | None = None,
    ) -> None:
        if self._pending is not None:
            raise AgentStateError("cannot begin a run with an unobserved decision")
        if (
            self._resume_ready
            and self._current_snapshot is not None
            and self._current_snapshot.observation.task_id == str(task_id)
            and (
                observation is None
                or observation.state_id == self._current_snapshot.state_id
            )
        ):
            self._task_id = str(task_id)
            self._run_active = True
            self._run_transition_count = 0
            self._resume_ready = False
            return
        self._task_id = str(task_id)
        self._step = 0
        self._run_transition_count = 0
        self._last_transition = None
        self._run_active = True
        self._resume_ready = False
        self._awaiting_visual_reset = False
        self.perception.reset(task_id=self._task_id)
        initial_stage = 1 if observation is None else int(observation.stage)
        self.exogenous.begin_episode(self._task_id, stage=initial_stage)
        self.hypotheses.begin_episode(self._task_id)
        self.ego.begin_episode(self._task_id)
        self.competence.begin_episode()
        self._current_snapshot = None
        if observation is not None:
            self._validate_observation(observation)
            self._current_snapshot = self._commit_initial_snapshot(observation)

    def _validate_observation(self, observation: Observation) -> None:
        if not isinstance(observation, Observation):
            raise TypeError("CompactHunterSeeker expects canonical Observation values")
        if self._task_id is None:
            self._task_id = str(observation.task_id)
        if str(observation.task_id) != self._task_id:
            raise AgentStateError(
                f"observation task {observation.task_id!r} does not match "
                f"active task {self._task_id!r}"
            )
        if self.config.strict_finite:
            try:
                finite = bool(np.isfinite(np.asarray(observation.frame)).all())
            except TypeError as exc:
                raise ValueError(
                    "strict_finite requires a numeric observation frame"
                ) from exc
            if not finite:
                raise ValueError("observation frame contains nonfinite values")

    def _snapshot_from_perception(
        self,
        observation: Observation,
        result: PerceptionResult,
        *,
        events: Sequence[WorldEvent] | None = None,
    ) -> WorldSnapshot:
        objects = self.ego.enrich(
            self.affordances.enrich(
                result.objects,
                task_id=observation.task_id,
            )
        )
        representation = self.representation_backend.encode(observation, objects)
        if self.config.strict_finite:
            arrays = (
                representation.global_vector,
                representation.spatial,
                *representation.taps.values(),
            )
            if any(not np.isfinite(np.asarray(value)).all() for value in arrays):
                raise ValueError("representation backend returned nonfinite features")
        return WorldSnapshot(
            observation=observation,
            objects=objects,
            events=tuple(result.events if events is None else events),
            topology=result.topology,
            representation=representation,
            step=int(self._step),
            memory_state_id=self.exogenous.masked_state_id(observation),
        )

    def _commit_initial_snapshot(self, observation: Observation) -> WorldSnapshot:
        original_perception = self.perception
        original_hypotheses = self.hypotheses
        original_backend = self.representation_backend
        original_search = self.search_engine
        staged_perception = self.perception.fork()
        staged_hypotheses = copy.deepcopy(self.hypotheses)
        backend_transaction = _RepresentationBackendTransaction.prepare(
            self.representation_backend
        )
        try:
            self.perception = staged_perception
            self.hypotheses = staged_hypotheses
            self.representation_backend = backend_transaction.staged
            result = self.perception.observe(observation, step=self._step)
            snapshot = self._snapshot_from_perception(observation, result)
            self.hypotheses.ensure_proposals(
                observation,
                snapshot.objects,
                mask_cells=self.exogenous.mask_cells(
                    observation.task_id,
                    tuple(int(v) for v in observation.frame.shape),
                    stage=observation.stage,
                ),
                memory_state_id=snapshot.memory_id,
            )
            staged_search = self._new_search_engine()
            self.representation_backend = backend_transaction.commit()
            self.search_engine = staged_search
            return snapshot
        except BaseException:
            rollback_error: BaseException | None = None
            try:
                backend_transaction.rollback()
            except BaseException as exc:
                rollback_error = exc
            self.perception = original_perception
            self.hypotheses = original_hypotheses
            self.representation_backend = original_backend
            self.search_engine = original_search
            if rollback_error is not None:
                raise AgentStateError(
                    "representation backend rollback failed after initial "
                    "snapshot error"
                ) from rollback_error
            raise

    def _snapshot_for_act(self, observation: Observation) -> WorldSnapshot:
        self._validate_observation(observation)
        if self._current_snapshot is None:
            self._current_snapshot = self._commit_initial_snapshot(observation)
        elif self._current_snapshot.state_id != observation.state_id:
            raise AgentStateError(
                "act received a new state before the previous environment "
                "transition was committed through observe"
            )
        return self._current_snapshot

    def _decision_id(self, snapshot: WorldSnapshot, action: Action) -> str:
        self._decision_counter += 1
        digest = blake2b(digest_size=16)
        digest.update(self.agent_id.encode("utf-8"))
        digest.update(snapshot.state_id.encode("ascii"))
        digest.update(int(self._decision_counter).to_bytes(8, "little", signed=False))
        digest.update(np.asarray(action.key, dtype=np.int64).tobytes())
        return digest.hexdigest()

    def _assert_teacher_boundary(self) -> None:
        if (
            self.teacher is not None
            and self.runtime_mode in {RuntimeMode.AUTONOMOUS, RuntimeMode.STUDENT}
        ):
            self.teacher.assert_no_action_selection_reads()

    def _teacher_assisted_candidates(
        self,
        snapshot: WorldSnapshot,
        candidates: Sequence[Any],
    ) -> tuple[Any, ...]:
        rows = tuple(candidates)
        if (
            self.teacher is None
            or self.runtime_mode is not RuntimeMode.COMPAT_ASSISTED
            or not rows
        ):
            return rows
        response = self.teacher.action_selection().suggest(
            TeacherRequest(
                task_id=snapshot.observation.task_id,
                stage=snapshot.observation.stage,
                state_id=snapshot.state_id,
                candidates=tuple(candidate.action for candidate in rows),
                metadata={
                    "step": snapshot.step,
                    "content_state_id": stable_frame_hash(
                        snapshot.observation.frame,
                        task_id=snapshot.observation.task_id,
                        stage=0,
                    ),
                },
            )
        )
        if response is None or response.action is None:
            return rows
        result = []
        for candidate in rows:
            terms = candidate.terms
            if candidate.action.key == response.action.key:
                terms = terms + (
                    ScoreTerm(
                        "teacher_assistance",
                        0.75 * float(response.confidence),
                        "teacher",
                        source=response.source,
                    ),
                )
            result.append(replace(candidate, terms=terms))
        return tuple(result)

    @staticmethod
    def _frame_change_fraction(
        before_frame: np.ndarray,
        after_frame: np.ndarray,
    ) -> float:
        before = np.asarray(before_frame)
        after = np.asarray(after_frame)
        if before.shape != after.shape:
            return 1.0
        if before.size <= 0:
            return 0.0
        return float(np.count_nonzero(before != after) / before.size)

    def _boundary_bridge_decision(
        self,
        snapshot: WorldSnapshot,
        *,
        fallback_action_indices: Sequence[int],
    ) -> Decision:
        """Select the deterministic action ARC uses to reveal the next board.

        ARC reports the new numeric stage on the action that solves the old
        board, but renders the new board only after one further environment
        action.  That action is still represented by an immutable Decision and
        Transition, while its screen replacement is kept out of causal model
        learning.
        """

        actions = self.search_engine.generator.generate(
            snapshot,
            fallback_action_indices=fallback_action_indices,
        )
        latent_delta = np.zeros_like(
            snapshot.representation.global_vector,
            dtype=np.float32,
        )
        candidates = tuple(
            Candidate(
                action=action,
                prediction=Prediction(
                    change_probability=0.0,
                    progress=0.0,
                    value=0.0,
                    hazard=0.0,
                    terminal=0.0,
                    uncertainty=1.0,
                    latent_delta=latent_delta,
                    object_delta=np.zeros(0, dtype=np.float32),
                    source="boundary_visual_reset",
                ),
                terms=(
                    ScoreTerm(
                        "boundary_visual_reset",
                        0.0,
                        "boundary",
                        source="runtime_protocol",
                    ),
                ),
                path=(action,),
                source="boundary_visual_reset",
            )
            for action in actions
        )
        chosen = min(candidates, key=lambda candidate: candidate.action.key)
        decision = Decision(
            decision_id=self._decision_id(snapshot, chosen.action),
            agent_id=self.agent_id,
            snapshot_id=snapshot.state_id,
            action=chosen.action,
            score=chosen.score,
            candidates=candidates,
            mode=self.runtime_mode,
            step=self._step,
            metadata=frozen_mapping(
                {
                    "task_id": snapshot.observation.task_id,
                    "stage": snapshot.observation.stage,
                    "selection_method": "boundary_visual_reset",
                    "safe_candidate_count": len(candidates),
                    "effective_risk": 0.0,
                    "expanded_nodes": 0,
                    "transposition_hits": 0,
                    "effective_horizon": 0,
                    "graph_goal_id": "",
                    "graph_goal_kind": "",
                    "graph_goal_target_id": "",
                    "graph_goal_path_length": 0,
                    "graph_goal_phi_delta": 0.0,
                    "graph_goal_expanded_nodes": 0,
                }
            ),
        )
        self._pending = _PendingDecision(
            decision=decision,
            snapshot=snapshot,
            chosen_prediction_source="boundary_visual_reset",
        )
        self.diagnostics.record_decision(
            decision,
            method="boundary_visual_reset",
            safe_candidate_count=len(candidates),
            expanded_nodes=0,
            transposition_hits=0,
            effective_horizon=0,
        )
        return decision

    def act(self, observation: Observation) -> Decision:
        if self._pending is not None:
            raise AgentStateError(
                f"decision {self._pending.decision.decision_id} must be observed "
                "before another action is selected"
            )
        if not self._run_active:
            self.begin_run(observation.task_id)
        snapshot = self._snapshot_for_act(observation)
        self._assert_teacher_boundary()
        # Decision construction consumes the policy RNG, increments the
        # capability counter, and records a trace.  Stage those small runtime
        # effects so an exceptional diagnostics/search hook cannot strand an
        # unobservable pending action or perturb the exact retry.
        counter_before = int(self._decision_counter)
        rng_state_before = copy.deepcopy(self._rng.bit_generator.state)
        diagnostics_before = self.diagnostics
        self.diagnostics = copy.deepcopy(diagnostics_before)
        try:
            fallback = tuple(
                int(v) for v in self._safe_action_provider(observation)
            )
            if self._awaiting_visual_reset:
                return self._boundary_bridge_decision(
                    snapshot,
                    fallback_action_indices=fallback,
                )
            search = self.search_engine.search(
                snapshot,
                competence=self.competence.state,
                fallback_action_indices=fallback,
                horizon_adjustment=self.competence.planning_horizon_adjustment(),
            )
            student_candidates = tuple(
                replace(
                    candidate,
                    terms=candidate.terms
                    + (
                        ScoreTerm(
                            "student_policy",
                            self.student_policy.score(
                                snapshot,
                                candidate.action,
                            ).value
                            * float(self.student_policy.config.runtime_weight),
                            "learned_value",
                            source="state_conditioned_student",
                        ),
                    ),
                )
                for candidate in search.candidates
            )
            candidates = self._teacher_assisted_candidates(
                snapshot,
                student_candidates,
            )
            arbitration = self.arbiter.choose(
                candidates,
                rng=self._rng,
            )
            chosen = arbitration.candidate
            decision = Decision(
                decision_id=self._decision_id(snapshot, chosen.action),
                agent_id=self.agent_id,
                snapshot_id=snapshot.state_id,
                action=chosen.action,
                score=chosen.score,
                candidates=tuple(candidates),
                mode=self.runtime_mode,
                step=self._step,
                metadata=frozen_mapping(
                    {
                        "task_id": snapshot.observation.task_id,
                        "stage": snapshot.observation.stage,
                        "selection_method": arbitration.method,
                        "safe_candidate_count": arbitration.safe_candidate_count,
                        "effective_risk": arbitration.effective_risk,
                        "expanded_nodes": search.expanded_nodes,
                        "transposition_hits": search.transposition_hits,
                        "effective_horizon": search.effective_horizon,
                        "graph_goal_id": search.graph_goal_id,
                        "graph_goal_kind": search.graph_goal_kind,
                        "graph_goal_target_id": search.graph_goal_target_id,
                        "graph_goal_path_length": search.graph_goal_path_length,
                        "graph_goal_phi_delta": search.graph_goal_phi_delta,
                        "graph_goal_expanded_nodes": (
                            search.graph_goal_expanded_nodes
                        ),
                    }
                ),
            )
            self._pending = _PendingDecision(
                decision=decision,
                snapshot=snapshot,
                chosen_prediction_source=chosen.prediction.source,
            )
            self.diagnostics.record_decision(
                decision,
                method=arbitration.method,
                safe_candidate_count=arbitration.safe_candidate_count,
                expanded_nodes=search.expanded_nodes,
                transposition_hits=search.transposition_hits,
                effective_horizon=search.effective_horizon,
            )
            return decision
        except BaseException:
            self._pending = None
            self._decision_counter = counter_before
            self._rng.bit_generator.state = rng_state_before
            self.diagnostics = diagnostics_before
            raise

    def distill_teacher(self, query: TeacherQuery) -> int:
        """Train student models from explicitly offline teacher examples.

        The teacher guard is used only in this explicit offline phase.  The
        state-conditioned head receives the represented pre-action state and
        legal alternatives; action selection later reads only learned student
        parameters.
        """

        if self.teacher is None:
            return 0
        examples = self.teacher.offline_training().examples(query)
        trained = 0
        student_samples: list[StudentTeacherSample] = []
        for example in examples:
            if float(example.weight) <= 0.0:
                continue
            frame = example.metadata.get("frame")
            frame_after = example.metadata.get("frame_after")
            if frame is None or frame_after is None:
                self.prior.observe_label(
                    task_id=example.task_id,
                    action_index=example.action.index,
                    weight=example.weight,
                    transferable=True,
                )
                trained += 1
                continue
            progress = float(example.metadata.get("progress", example.stage - 1))
            progress_delta = float(example.metadata.get("progress_delta", 0.0))
            declared_actions = tuple(
                int(value)
                for value in example.metadata.get("available_actions", ())
            )
            before_observation = Observation(
                frame=np.asarray(frame),
                available_actions=declared_actions or (example.action.index,),
                task_id=example.task_id,
                stage=example.stage,
                progress=progress,
                metadata={"source": example.source},
            )
            provider_actions = tuple(
                int(value) for value in self._safe_action_provider(None)
            )
            legal_indices = tuple(
                dict.fromkeys(
                    (
                        *before_observation.available_actions,
                        *provider_actions,
                        int(example.action.index),
                    )
                )
            )
            before_observation = replace(
                before_observation,
                available_actions=legal_indices,
            )
            after_observation = Observation(
                frame=np.asarray(frame_after),
                available_actions=legal_indices,
                task_id=example.task_id,
                stage=example.stage + int(progress_delta > 0.0),
                progress=progress + progress_delta,
                metadata={"source": example.source},
            )
            local_perception = PerceptionSystem(self.config.perception)
            before_result = local_perception.observe(before_observation, step=0)
            after_result = local_perception.observe(after_observation, step=1)
            before_objects = self.affordances.enrich(
                before_result.objects,
                task_id=example.task_id,
            )
            after_objects = self.affordances.enrich(
                after_result.objects,
                task_id=example.task_id,
            )
            before_snapshot = WorldSnapshot(
                observation=before_observation,
                objects=before_objects,
                events=before_result.events,
                topology=before_result.topology,
                representation=self.representation_backend.encode(
                    before_observation,
                    before_objects,
                ),
                step=0,
            )
            after_snapshot = WorldSnapshot(
                observation=after_observation,
                objects=after_objects,
                events=after_result.events,
                topology=after_result.topology,
                representation=self.representation_backend.encode(
                    after_observation,
                    after_objects,
                ),
                step=1,
            )
            legal_candidates = [example.action]
            legal_candidates.extend(
                Action(index)
                for index in legal_indices
                if int(index) != int(example.action.index)
            )
            student_samples.append(
                StudentTeacherSample(
                    snapshot=before_snapshot,
                    example=example,
                    candidates=tuple(legal_candidates),
                )
            )
            frame_changed = not np.array_equal(
                before_observation.frame,
                after_observation.frame,
            )
            outcome = Outcome(
                reward=float(example.weight)
                * (progress_delta + 0.1 * float(frame_changed)),
                progress_delta=progress_delta,
                boundary=(
                    BoundaryKind.LEVEL_COMPLETED
                    if progress_delta > 0.0
                    else BoundaryKind.NONE
                ),
                metadata={"teacher": example.source},
            )
            transition = Transition(
                transition_id=blake2b(
                    (
                        f"teacher|{example.source}|{example.task_id}|"
                        f"{example.stage}|{example.state_id}|{example.action.key}"
                    ).encode("utf-8"),
                    digest_size=16,
                ).hexdigest(),
                decision_id=f"teacher:{example.source}:{example.state_id}",
                task_id=example.task_id,
                stage=example.stage,
                step=0,
                before=before_snapshot,
                action=example.action,
                after_observation=after_observation,
                outcome=outcome,
                frame_changed=frame_changed,
                after_state_id=after_observation.state_id,
            )
            self.learner.observe(
                ReplayItem(
                    before=before_snapshot,
                    after=after_snapshot,
                    transition=transition,
                    target_object_signature=_target_signature(
                        before_snapshot,
                        example.action,
                    ),
                    source=example.source,
                    teacher=True,
                ),
                learn_immediately=True,
            )
            trained += 1
        self.student_policy.observe_teacher_batch(student_samples)
        return trained

    def _attribute_contact_outcome(
        self,
        before: WorldSnapshot,
        after: WorldSnapshot,
        outcome: Outcome,
    ) -> None:
        """Blame objects adjacent to the controlled set for a bad outcome.

        The controlled set itself is protected: a signature belonging to any
        controlled body is never given hazard evidence by this route.
        """

        if not self.config.ego.enabled:
            return
        severity = float(max(outcome.hazard, 1.0 if outcome.failed else 0.0))
        if severity <= 0.0:
            return
        controlled = self.ego.controlled_objects(before.objects)
        if not controlled:
            return
        controlled_track_ids = {obj.track_id for obj in controlled}
        protected_signatures = {obj.signature for obj in controlled}

        def adjacent_signatures(
            snapshot: WorldSnapshot,
            *,
            controlled_ids: set[int],
            controlled_names: set[str],
        ) -> set[str]:
            by_object_id = {obj.object_id: obj for obj in snapshot.objects}
            result: set[str] = set()
            for left_id, right_id in snapshot.topology.object_adjacencies:
                left = by_object_id.get(int(left_id))
                right = by_object_id.get(int(right_id))
                if left is None or right is None:
                    continue
                for ego_obj, other in ((left, right), (right, left)):
                    is_controlled = (
                        ego_obj.track_id in controlled_ids
                        or (
                            bool(ego_obj.signature)
                            and ego_obj.signature in controlled_names
                        )
                    )
                    other_controlled = (
                        other.track_id in controlled_ids
                        or other.signature in controlled_names
                    )
                    if (
                        is_controlled
                        and not other_controlled
                        and other.signature
                    ):
                        result.add(other.signature)
            return result

        blamed = adjacent_signatures(
            after,
            controlled_ids=controlled_track_ids,
            controlled_names=protected_signatures,
        )
        # A fatal contact can remove the controlled body from the post-action
        # frame.  In that narrow case, allow the former adjacency only for an
        # object signature that demonstrably survives into the new frame.
        surviving_signatures = {obj.signature for obj in after.objects if obj.signature}
        controlled_survived = any(
            obj.track_id in controlled_track_ids
            or obj.signature in protected_signatures
            for obj in after.objects
        )
        if not blamed and not controlled_survived:
            blamed.update(
                signature
                for signature in adjacent_signatures(
                    before,
                    controlled_ids=controlled_track_ids,
                    controlled_names=protected_signatures,
                )
                if signature in surviving_signatures
            )
        for signature in sorted(blamed):
            self.affordances.observe_signature(
                signature,
                task_id=after.observation.task_id,
                hazard=severity,
                terminal=float(outcome.terminated and not outcome.completed),
            )

    @staticmethod
    def _chosen_candidate(decision: Decision):
        for candidate in decision.candidates:
            if candidate.action.key == decision.action.key:
                return candidate
        raise AgentStateError("chosen action is absent from the decision candidate set")

    def observe(
        self,
        decision: Decision,
        next_observation: Observation,
        outcome: Outcome,
    ) -> Transition:
        """Atomically commit exactly the immutable decision returned by ``act``.

        Every mutable learner/memory component is updated on a staged copy.
        If perception, representation, learning, diagnostics, or any other
        commit hook raises, both the pending capability and all durable state
        remain exactly as they were before this call, so the caller may retry
        or cancel deliberately.
        """

        if not isinstance(decision, Decision):
            raise TypeError("observe requires a canonical Decision")
        if decision.decision_id in self._committed_decision_ids:
            raise DuplicateDecisionError(
                f"decision {decision.decision_id} was already committed"
            )
        pending = self._pending
        if pending is None:
            raise AgentStateError("observe called without a pending decision")
        if decision is not pending.decision:
            raise AgentStateError(
                "observe requires the exact immutable Decision returned by act; "
                "copied or altered decision payloads are rejected"
            )
        if not isinstance(next_observation, Observation):
            raise TypeError("observe requires a canonical Observation")
        if not isinstance(outcome, Outcome):
            raise TypeError("observe requires a canonical Outcome")

        component_names = (
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
        )
        originals = {name: getattr(self, name) for name in component_names}
        staged = copy.deepcopy(originals)
        scalar_names = (
            "_task_id",
            "_step",
            "_current_snapshot",
            "_pending",
            "_committed_decision_ids",
            "_last_transition",
            "_run_active",
            "_resume_ready",
            "_awaiting_visual_reset",
            "_run_transition_count",
            "_total_transition_count",
        )
        original_scalars = {name: getattr(self, name) for name in scalar_names}
        original_search = self.search_engine
        backend_transaction = _RepresentationBackendTransaction.prepare(
            self.representation_backend
        )
        try:
            for name, value in staged.items():
                setattr(self, name, value)
            self.representation_backend = backend_transaction.staged
            # The set is mutated in-place on success, so give the staged
            # transaction its own copy as well.
            self._committed_decision_ids = set(self._committed_decision_ids)
            self.search_engine = self._new_search_engine()
            transition = self._observe_impl(
                pending.decision,
                next_observation,
                outcome,
            )
            self.representation_backend = backend_transaction.commit()
            return transition
        except BaseException:
            rollback_error: BaseException | None = None
            try:
                backend_transaction.rollback()
            except BaseException as exc:
                rollback_error = exc
            for name, value in originals.items():
                setattr(self, name, value)
            for name, value in original_scalars.items():
                setattr(self, name, value)
            self.representation_backend = backend_transaction.original
            self.search_engine = original_search
            if rollback_error is not None:
                raise AgentStateError(
                    "representation backend rollback failed after observe error"
                ) from rollback_error
            raise

    def _observe_visual_bridge(
        self,
        decision: Decision,
        next_observation: Observation,
        outcome: Outcome,
        *,
        frame_changed: bool,
    ) -> Transition:
        """Commit ARC's delayed board reveal without learning it as dynamics."""

        if self._pending is None:
            raise AgentStateError("visual reset requires a pending decision")
        pending = self._pending
        self._step += 1
        self.perception.reset(task_id=next_observation.task_id)
        self.ego.reset_tracks()
        self.exogenous.begin_episode(
            next_observation.task_id,
            stage=next_observation.stage,
        )
        perception_result = self.perception.observe(
            next_observation,
            step=self._step,
        )
        events = list(perception_result.events)
        events.append(
            WorldEvent(
                EventKind.FRAME_CHANGED if frame_changed else EventKind.NO_CHANGE,
                magnitude=1.0,
                metadata={"boundary_visual_reset": True},
            )
        )
        if outcome.hazard > 0.0 or outcome.failed:
            events.append(
                WorldEvent(
                    EventKind.HAZARD,
                    magnitude=float(
                        max(outcome.hazard, 1.0 if outcome.failed else 0.0)
                    ),
                )
            )
        if outcome.terminated:
            events.append(
                WorldEvent(
                    EventKind.TERMINAL,
                    magnitude=1.0,
                    metadata={"boundary": outcome.boundary.value},
                )
            )
        after_snapshot = self._snapshot_from_perception(
            next_observation,
            perception_result,
            events=events,
        )
        self.hypotheses.ensure_proposals(
            next_observation,
            after_snapshot.objects,
            mask_cells=self.exogenous.mask_cells(
                next_observation.task_id,
                tuple(int(v) for v in next_observation.frame.shape),
                stage=next_observation.stage,
            ),
            memory_state_id=after_snapshot.memory_id,
        )
        outcome = replace(
            outcome,
            metadata={
                **dict(outcome.metadata),
                "boundary_visual_reset": True,
            },
        )
        transition_id = blake2b(
            (
                f"{self.agent_id}|{decision.decision_id}|{pending.snapshot.state_id}|"
                f"{decision.action.key}|{after_snapshot.state_id}|{self._step}"
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        transition = Transition(
            transition_id=transition_id,
            decision_id=decision.decision_id,
            task_id=next_observation.task_id,
            stage=pending.snapshot.observation.stage,
            step=pending.snapshot.step,
            before=pending.snapshot,
            action=decision.action,
            after_observation=next_observation,
            outcome=outcome,
            frame_changed=frame_changed,
            after_state_id=after_snapshot.state_id,
        )
        self.diagnostics.record_transition(
            transition,
            events=events,
            model_loss=0.0,
            prediction_error=0.0,
        )
        self._current_snapshot = after_snapshot
        self._pending = None
        self._committed_decision_ids.add(decision.decision_id)
        self._last_transition = transition
        self._awaiting_visual_reset = False
        self._run_transition_count += 1
        self._total_transition_count += 1
        return transition

    def _observe_impl(
        self,
        decision: Decision,
        next_observation: Observation,
        outcome: Outcome,
    ) -> Transition:
        if decision.decision_id in self._committed_decision_ids:
            raise DuplicateDecisionError(
                f"decision {decision.decision_id} was already committed"
            )
        if self._pending is None:
            raise AgentStateError("observe called without a pending decision")
        pending = self._pending
        if decision.decision_id != pending.decision.decision_id:
            raise AgentStateError(
                f"observe received decision {decision.decision_id}, expected "
                f"{pending.decision.decision_id}"
            )
        if decision.agent_id != self.agent_id:
            raise AgentStateError("decision belongs to a different agent")
        self._validate_observation(next_observation)
        stage_changed = (
            int(next_observation.stage)
            != int(pending.snapshot.observation.stage)
        )
        observed_delta = max(
            0.0,
            float(next_observation.progress - pending.snapshot.observation.progress),
        )
        if abs(observed_delta - outcome.progress_delta) > 1e-9:
            outcome = replace(
                outcome,
                progress_delta=max(observed_delta, outcome.progress_delta),
            )
        frame_changed = not np.array_equal(
            pending.snapshot.observation.frame,
            next_observation.frame,
        )
        if self._awaiting_visual_reset:
            return self._observe_visual_bridge(
                decision,
                next_observation,
                outcome,
                frame_changed=frame_changed,
            )
        change_fraction = self._frame_change_fraction(
            pending.snapshot.observation.frame,
            next_observation.frame,
        )
        delayed_visual_bridge = bool(
            stage_changed
            and outcome.completed
            and not outcome.terminated
            and not outcome.truncated
            and change_fraction
            <= float(self.config.boundary_bridge_change_fraction)
        )
        immediate_stage_swap = bool(
            stage_changed
            and outcome.completed
            and not delayed_visual_bridge
        )
        # Capture old-stage causal motion evidence before the stage reset
        # clears track-local ego statistics.  This is used only by the narrow
        # completion-motion reach fallback; the new board itself is never
        # interpreted as old-stage goal evidence.
        completion_motion_predictions: dict[
            int, tuple[float, float, float]
        ] = {}
        if immediate_stage_swap:
            for obj in pending.snapshot.objects:
                prediction = self.ego.controlled_motion_prediction(
                    obj,
                    decision.action,
                )
                if prediction is not None:
                    completion_motion_predictions[int(obj.track_id)] = prediction
        self._step += 1
        if stage_changed and not delayed_visual_bridge:
            self.perception.reset(task_id=next_observation.task_id)
        perception_result = self.perception.observe(
            next_observation,
            step=self._step,
        )
        events = list(perception_result.events)
        events.append(
            WorldEvent(
                EventKind.FRAME_CHANGED if frame_changed else EventKind.NO_CHANGE,
                magnitude=1.0,
            )
        )
        if outcome.progress_delta > 0.0:
            events.append(
                WorldEvent(
                    EventKind.PROGRESS,
                    magnitude=float(outcome.progress_delta),
                )
            )
        if outcome.hazard > 0.0 or outcome.failed:
            events.append(
                WorldEvent(
                    EventKind.HAZARD,
                    magnitude=float(
                        max(outcome.hazard, 1.0 if outcome.failed else 0.0)
                    ),
                )
            )
        if outcome.terminated:
            events.append(
                WorldEvent(
                    EventKind.TERMINAL,
                    magnitude=1.0,
                    metadata={"boundary": outcome.boundary.value},
                )
            )
        if stage_changed:
            # Perception intentionally starts a new track namespace at a stage
            # boundary.  Motion events across that reset cannot ground causal
            # control attribution for either stage.
            self.ego.reset_tracks()
        else:
            self.ego.observe(
                action=decision.action,
                events=perception_result.events,
                visible_objects=pending.snapshot.objects,
            )
        before_stage = int(pending.snapshot.observation.stage)
        self.exogenous.observe_transition(
            task_id=next_observation.task_id,
            before_frame=pending.snapshot.observation.frame,
            after_frame=next_observation.frame,
            action=decision.action,
            stage=before_stage,
            after_stage=(
                before_stage
                if delayed_visual_bridge
                else int(next_observation.stage)
            ),
        )
        if delayed_visual_bridge:
            # The solved frame still belongs to the old stage.  Finalize that
            # comparable segment now and open a clean successor-stage segment;
            # the next giant board-load transition is handled by the bridge
            # path above and never admitted as exogenous evidence.
            self.exogenous.begin_episode(
                next_observation.task_id,
                stage=next_observation.stage,
            )
        after_snapshot = self._snapshot_from_perception(
            next_observation,
            perception_result,
            events=events,
        )
        if not stage_changed:
            # Contact attribution must use the post-action topology.  Using
            # the pre-action graph blames whatever used to be adjacent rather
            # than what the controlled body actually contacted.
            self._attribute_contact_outcome(
                pending.snapshot,
                after_snapshot,
                outcome,
            )
        transition_id = blake2b(
            (
                f"{self.agent_id}|{decision.decision_id}|{pending.snapshot.state_id}|"
                f"{decision.action.key}|{after_snapshot.state_id}|{self._step}"
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        transition = Transition(
            transition_id=transition_id,
            decision_id=decision.decision_id,
            task_id=next_observation.task_id,
            stage=pending.snapshot.observation.stage,
            step=pending.snapshot.step,
            before=pending.snapshot,
            action=decision.action,
            after_observation=next_observation,
            outcome=outcome,
            frame_changed=frame_changed,
            after_state_id=after_snapshot.state_id,
        )
        if self.config.enable_online_learning:
            self.student_policy.observe_transition(transition)
        chosen = self._chosen_candidate(decision)
        target_signature = _target_signature(pending.snapshot, decision.action)
        scene_signature = combined_object_signature(
            pending.snapshot.objects,
            decision.action,
        )
        effect_signature = _effect_signature(
            events=events,
            before=pending.snapshot,
            after=after_snapshot,
            outcome=outcome,
        )

        replay_item = ReplayItem(
            before=pending.snapshot,
            after=after_snapshot,
            transition=transition,
            target_object_signature=target_signature,
            source="online",
            teacher=False,
        )
        model_loss = self.learner.observe(
            replay_item,
            learn_immediately=self.config.enable_online_learning,
        )
        if self.config.enable_online_learning:
            self.learner.maybe_replay(self._total_transition_count + 1)
        record = self.graph.observe(
            transition,
            before_latent=pending.snapshot.representation.global_vector,
            after_latent=after_snapshot.representation.global_vector,
            before_object_summary=object_summary(pending.snapshot.objects),
            after_object_summary=object_summary(after_snapshot.objects),
            uncertainty=chosen.prediction.uncertainty,
            events=events,
            object_signature=scene_signature,
            effect_signature=effect_signature,
            source="online",
            model_version=f"dynamics:{self.dynamics.update_count}",
            before_state_id=pending.snapshot.memory_id,
            after_state_id=after_snapshot.memory_id,
        )
        self.evidence.append(record)
        self.hypotheses.observe_transition(
            before_observation=pending.snapshot.observation,
            after_observation=next_observation,
            objects=pending.snapshot.objects,
            action=decision.action,
            progressed=(
                outcome.progress_delta > 0.0 or outcome.completed
            ),
            before_memory_id=pending.snapshot.memory_id,
            after_memory_id=after_snapshot.memory_id,
            mask_cells=self.exogenous.mask_cells(
                pending.snapshot.observation.task_id,
                tuple(
                    int(v)
                    for v in pending.snapshot.observation.frame.shape
                ),
                stage=pending.snapshot.observation.stage,
            ),
            completion_observation=(
                next_observation
                if delayed_visual_bridge or not stage_changed
                else None
            ),
            completion_objects=(
                after_snapshot.objects
                if delayed_visual_bridge or not stage_changed
                else None
            ),
            completion_frame_available=not immediate_stage_swap,
            controlled_motion_predictions=completion_motion_predictions,
        )
        self.competence.update(
            prediction=chosen.prediction,
            transition=transition,
            model_loss=model_loss,
            learning_progress=self.dynamics.learning_progress_ema,
            time_pressure=self.exogenous.time_pressure(
                pending.snapshot.observation.task_id,
                tuple(
                    int(v)
                    for v in pending.snapshot.observation.frame.shape
                ),
                self._step,
                stage=pending.snapshot.observation.stage,
            ),
        )

        # Refresh durable current features after the affordance update so the
        # next decision sees newly grounded object beliefs.
        refreshed_objects = self.ego.enrich(
            self.affordances.enrich(
                perception_result.objects,
                task_id=next_observation.task_id,
            )
        )
        after_snapshot = replace(
            after_snapshot,
            objects=refreshed_objects,
            representation=self.representation_backend.encode(
                next_observation,
                refreshed_objects,
            ),
        )
        if stage_changed and not delayed_visual_bridge:
            self.hypotheses.ensure_proposals(
                next_observation,
                after_snapshot.objects,
                mask_cells=self.exogenous.mask_cells(
                    next_observation.task_id,
                    tuple(int(v) for v in next_observation.frame.shape),
                    stage=next_observation.stage,
                ),
                memory_state_id=after_snapshot.memory_id,
            )
        prediction_error = _prediction_error(decision, outcome, frame_changed)
        self.diagnostics.record_transition(
            transition,
            events=events,
            model_loss=model_loss,
            prediction_error=prediction_error,
        )
        self._current_snapshot = after_snapshot
        self._pending = None
        self._committed_decision_ids.add(decision.decision_id)
        self._last_transition = transition
        self._awaiting_visual_reset = delayed_visual_bridge
        self._run_transition_count += 1
        self._total_transition_count += 1
        return transition

    def cancel_decision(self, decision: Decision, *, reason: str = "") -> None:
        """Cancel an unexecuted pending decision without creating evidence.

        This is reserved for adapter failures that occur before an environment
        action is issued.  Once ``environment.step`` has been attempted, the
        caller must commit an outcome instead of erasing the action.
        """

        del reason  # Kept in the API for caller-side audit/log context.
        pending = self._pending
        if pending is None:
            raise AgentStateError("cannot cancel without a pending decision")
        if decision is not pending.decision:
            raise AgentStateError(
                "only the exact pending Decision capability can be cancelled"
            )
        self._pending = None

    def on_level_complete(self, _level_number: int | None = None) -> None:
        """Compatibility callback; the transition was already committed."""

        if self._last_transition is None or not self._last_transition.outcome.completed:
            raise AgentStateError(
                "level-complete callback must follow an observed completion transition"
            )

    def on_game_over(self) -> None:
        """Compatibility callback; no delayed terminal repair is performed."""

        if (
            self._last_transition is None
            or not self._last_transition.outcome.terminated
        ):
            raise AgentStateError(
                "game-over callback must follow an observed terminal transition"
            )

    def end_run(self, outcome: Outcome | None = None) -> None:
        if self._pending is not None:
            raise AgentStateError(
                "cannot end run while an executed decision remains unobserved"
            )
        if outcome is not None and outcome.terminated and self._last_transition is None:
            raise AgentStateError("terminal run outcome has no committed transition")
        self.exogenous.end_episode()
        self._run_active = False

    def measurement_summary(self) -> dict[str, Any]:
        summary = self.diagnostics.summary()
        summary.update(
            {
                "format": self.format_name,
                "runtime_mode": self.runtime_mode.value,
                "task_id": self._task_id,
                "run_transition_count": self._run_transition_count,
                "total_transition_count": self._total_transition_count,
                "state_nodes": len(self.graph),
                "state_edges": self.graph.edge_count,
                "evidence_records": len(self.evidence),
                "dynamics_updates": self.dynamics.update_count,
                "dynamics_loss": self.dynamics.last_loss,
                "learning_progress": self.dynamics.learning_progress_ema,
                "replay_size": len(self.learner.replay),
                "replay_updates": self.learner.replay_updates,
                "replay_change_rate": self.learner.replay.change_rate(),
                "replay_teacher_fraction": self.learner.replay.expert_fraction(),
                "graph_revisits": self.graph.revisit_count,
                "ego": self.ego.summary(),
                "exogenous": self.exogenous.summary(),
                "hypotheses": self.hypotheses.summary(),
                "student_policy": self.student_policy.summary(),
                "competence": {
                    key: value for key, value in asdict(self.competence.state).items()
                },
            }
        )
        return summary

    def state_dict(self, *, include_diagnostics: bool = True) -> dict[str, Any]:
        from .persistence import agent_state_dict

        return agent_state_dict(
            self,
            include_diagnostics=include_diagnostics,
        )

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        reset_optimizer: bool = False,
        weights_only: bool = False,
    ) -> None:
        from .persistence import load_agent_state

        load_agent_state(
            self,
            state,
            reset_optimizer=reset_optimizer,
            weights_only=weights_only,
        )

    def save_checkpoint(
        self,
        path: str,
        *,
        include_diagnostics: bool = True,
    ) -> None:
        from .persistence import save_checkpoint

        save_checkpoint(
            self,
            path,
            include_diagnostics=include_diagnostics,
        )

    def load_checkpoint(
        self,
        path: str,
        *,
        reset_optimizer: bool = False,
        weights_only: bool = False,
    ) -> None:
        from .persistence import load_checkpoint

        load_checkpoint(
            self,
            path,
            reset_optimizer=reset_optimizer,
            weights_only=weights_only,
        )

    def dump_events_for_sleep(self, path: str) -> int:
        """Compatibility export of the canonical transition trace."""

        import json
        from pathlib import Path

        rows = [
            {
                "transition_id": row.transition_id,
                "task_id": row.task_id,
                "stage": row.stage,
                "step": row.step,
                "state_id": row.state_id,
                "successor_id": row.successor_id,
                "action": list(row.action),
                "events": list(row.events),
                "progress": row.progress,
                "hazard": row.hazard,
                "terminal": row.terminal,
                "boundary": row.boundary,
            }
            for row in self.diagnostics.transitions
        ]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
        return len(rows)


__all__ = [
    "AgentStateError",
    "CompactHunterSeeker",
    "DuplicateDecisionError",
    "SafeActionProvider",
]
