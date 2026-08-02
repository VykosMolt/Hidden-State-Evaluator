from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from hunter_seeker_v2.adapters import AdapterError, CategoricalActionAdapter
from hunter_seeker_v2.contracts import (
    Action,
    CompetenceState,
    Observation,
    PolicyConfig,
    Prediction,
    Representation,
    SearchConfig,
    Topology,
    WorldSnapshot,
    stable_frame_hash,
)
from hunter_seeker_v2.executable import ExecutablePrediction
from hunter_seeker_v2.memory import EvidenceStore, GraphEdgeView
from hunter_seeker_v2.models import ActionPrior
from hunter_seeker_v2.policy import PolicyScorer
from hunter_seeker_v2.search import SearchEngine
from hunter_seeker_v2.teacher import (
    TeacherExample,
    TeacherRequest,
    TrajectoryTeacher,
)


def _snapshot(*, actions: tuple[int, ...] = (0, 1)) -> WorldSnapshot:
    observation = Observation(
        frame=np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
        available_actions=actions,
        task_id="security-task",
        stage=1,
    )
    return WorldSnapshot(
        observation=observation,
        objects=(),
        events=(),
        topology=Topology(),
        representation=Representation(
            global_vector=np.zeros(1, dtype=np.float32),
            spatial=np.zeros((1, 2, 2), dtype=np.float32),
        ),
        step=0,
    )


def _prediction(successor: str | None = None) -> Prediction:
    return Prediction(
        change_probability=0.5,
        progress=0.0,
        value=0.0,
        hazard=0.0,
        terminal=0.0,
        uncertainty=0.1,
        latent_delta=np.zeros(1, dtype=np.float32),
        object_delta=np.zeros(8, dtype=np.float32),
        exact_successor_id=successor,
        source="exact-test",
    )


class _NeverFallbackModel:
    latent_dim = 1
    object_dim = 8

    def predict(self, snapshot: WorldSnapshot, action: Action) -> Prediction:
        raise AssertionError("the learned model must not replace an applicable rule")

    def predict_from_features(self, **kwargs: Any) -> Prediction:
        raise AssertionError("the learned model must not replace an exact graph edge")


class _NeutralFallbackModel:
    """Permit an explicitly untried root action in graph-frontier tests."""

    latent_dim = 1
    object_dim = 8

    def predict(self, snapshot: WorldSnapshot, action: Action) -> Prediction:
        del snapshot, action
        return _prediction()

    def predict_from_features(self, **kwargs: Any) -> Prediction:
        del kwargs
        return _prediction()


class _NullGraph:
    def exact_prediction(self, *args: Any, **kwargs: Any) -> None:
        return None

    def action_stats(self, *args: Any, **kwargs: Any) -> Mapping[str, object]:
        return {}


def _engine(
    *,
    model: Any | None = None,
    graph: Any | None = None,
    registry: Any | None = None,
    config: SearchConfig | None = None,
    tap_reader: Any | None = None,
    hypothesis_engine: Any | None = None,
) -> SearchEngine:
    search_config = config or SearchConfig(horizon=1, beam_width=2)
    return SearchEngine(
        model=_NeverFallbackModel() if model is None else model,
        graph=graph or _NullGraph(),
        evidence=EvidenceStore(),
        prior=ActionPrior(),
        scorer=PolicyScorer(PolicyConfig(), search=search_config),
        click_action_index=None,
        config=search_config,
        tap_reader=tap_reader,
        executable_registry=registry,
        hypothesis_engine=hypothesis_engine,
    )


def test_search_rejects_uncalibrated_raw_tap_reader() -> None:
    with pytest.raises(ValueError, match="calibrated TapBundle"):
        _engine(tap_reader=lambda snapshot, action, prediction: {"value": 1.0})


@dataclass(frozen=True)
class _Report:
    model_id: str
    coverage: float = 1.0
    error_rate: float = 0.0
    mean_auxiliary_error: float = 0.0


class _FallthroughRegistry:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ranked_verifications(self) -> tuple[_Report, ...]:
        return (_Report("top-specialist"), _Report("applicable-specialist"))

    def predict(
        self,
        model_id: str,
        state: Any,
        action: Action,
    ) -> ExecutablePrediction:
        del state, action
        self.calls.append(model_id)
        if model_id == "top-specialist":
            return ExecutablePrediction(applicable=False)
        return ExecutablePrediction(
            applicable=True,
            successor_state_id="rule-successor",
            change_probability=1.0,
            progress_delta=1.0,
        )


def test_prediction_falls_through_a_nonapplicable_top_verified_model() -> None:
    registry = _FallthroughRegistry()
    prediction = _engine(registry=registry)._prediction(_snapshot(), Action(0))

    assert registry.calls == ["top-specialist", "applicable-specialist"]
    assert prediction.source == "executable:applicable-specialist"
    assert prediction.exact_successor_id == "rule-successor"


class _ExactBranchGraph:
    """Return identical features while keeping exact successor identities distinct."""

    def __init__(self, root_id: str) -> None:
        self.root_id = root_id

    def exact_prediction(
        self,
        state_id: str,
        action: Action,
        *,
        latent_dim: int,
        object_dim: int,
    ) -> Prediction | None:
        assert latent_dim == 1
        assert object_dim == 8
        if state_id == self.root_id:
            return _prediction("branch-root")
        if state_id == "branch-root":
            return _prediction("branch-a" if action.index == 0 else "branch-b")
        if state_id in {"branch-a", "branch-b"}:
            return _prediction(f"{state_id}:{action.index}")
        return None

    def action_stats(self, *args: Any, **kwargs: Any) -> Mapping[str, object]:
        return {}


def test_exact_state_beam_nodes_do_not_alias_when_features_are_identical() -> None:
    snapshot = _snapshot(actions=(0, 1))
    config = SearchConfig(horizon=3, beam_width=2, transposition_limit=100)
    result = _engine(
        graph=_ExactBranchGraph(snapshot.memory_id),
        config=config,
    ).search(snapshot, competence=CompetenceState())

    # Each of two roots expands two exact branches and then both actions from
    # both distinct branch state IDs: 2 + 2 * (2 + 4) = 14 nodes.  A key made
    # only from their identical latent/object features would collapse four.
    assert result.expanded_nodes == 14
    assert result.transposition_hits == 0


@dataclass(frozen=True)
class _ObservedEdge:
    action: Action
    successor_id: str
    value: float = 0.0
    hazard: float = 0.0
    adverse_terminal_rate: float = 0.0
    distance_to_progress: int | None = None


class _ObservedGraph:
    """Small exact graph with independently declared successor inventories."""

    def __init__(
        self,
        edges: Mapping[str, Sequence[_ObservedEdge]],
        available: Mapping[str, Sequence[int]],
        *,
        reverse_edges: bool = False,
        terminal: Sequence[str] = (),
    ) -> None:
        self._edges = {
            str(state_id): tuple(rows)
            for state_id, rows in edges.items()
        }
        self._available = {
            str(state_id): tuple(int(index) for index in indices)
            for state_id, indices in available.items()
        }
        self._reverse_edges = bool(reverse_edges)
        self._terminal = {str(state_id) for state_id in terminal}

    def _edge(self, state_id: str, action: Action) -> _ObservedEdge | None:
        return next(
            (
                edge
                for edge in self._edges.get(str(state_id), ())
                if edge.action.key == action.key
            ),
            None,
        )

    def exact_prediction(
        self,
        state_id: str,
        action: Action,
        *,
        latent_dim: int,
        object_dim: int,
    ) -> Prediction | None:
        edge = self._edge(state_id, action)
        if edge is None:
            return None
        return Prediction(
            change_probability=1.0,
            progress=0.0,
            value=edge.value,
            hazard=edge.hazard,
            terminal=0.0,
            uncertainty=0.1,
            latent_delta=np.zeros(latent_dim, dtype=np.float32),
            object_delta=np.zeros(object_dim, dtype=np.float32),
            exact_successor_id=edge.successor_id,
            source="exact_graph",
        )

    def action_stats(
        self,
        state_id: str,
        action: Action,
    ) -> Mapping[str, object]:
        edge = self._edge(state_id, action)
        if edge is None:
            return {"known": False}
        return {
            "known": True,
            "successor_id": edge.successor_id,
            "no_change_rate": 0.0,
            "self_loop_rate": 0.0,
            "distance_to_progress": edge.distance_to_progress,
        }

    def outgoing(self, state_id: str) -> tuple[GraphEdgeView, ...]:
        rows = tuple(
            GraphEdgeView(
                action=edge.action,
                successor_id=edge.successor_id,
                visits=4,
                dominant_fraction=1.0,
                change_rate=1.0,
                hazard=edge.hazard,
                adverse_terminal_rate=edge.adverse_terminal_rate,
                uncertainty=0.1,
            )
            for edge in self._edges.get(str(state_id), ())
        )
        return tuple(reversed(rows)) if self._reverse_edges else rows

    def available_action_indices(self, state_id: str) -> tuple[int, ...]:
        return self._available.get(str(state_id), ())

    def is_terminal(self, state_id: str) -> bool:
        return str(state_id) in self._terminal


@dataclass(frozen=True)
class _PlanningGoal:
    hypothesis_id: str
    kind: str
    current_potential: float


class _GoalPlanningEngine:
    def __init__(
        self,
        goals: Sequence[_PlanningGoal],
        potentials: Mapping[str, float | None],
        signals: Mapping[int, float] | None = None,
    ) -> None:
        self._goals = tuple(goals)
        self._potentials = dict(potentials)
        self._signals = dict(signals or {})

    def mismatch_points(self, *args: Any, **kwargs: Any) -> tuple[()]:
        del args, kwargs
        return ()

    def candidate_signal(self, *args: Any, **kwargs: Any) -> float:
        del kwargs
        action = args[1]
        return float(self._signals.get(int(action.index), 0.0))

    def rollout_signal(self, *args: Any, **kwargs: Any) -> float:
        del args, kwargs
        return 0.0

    def planning_goals(self, *args: Any, **kwargs: Any) -> tuple[_PlanningGoal, ...]:
        del args, kwargs
        return self._goals

    def cached_goal_potential(
        self,
        *,
        state_id: str,
        **kwargs: Any,
    ) -> float | None:
        del kwargs
        return self._potentials.get(str(state_id))


def _graph_goal_search(
    *,
    reverse_edges: bool = False,
    risky_first_edge: bool = False,
    goals: Sequence[_PlanningGoal] | None = None,
) -> tuple[WorldSnapshot, Any]:
    snapshot = _snapshot(actions=(0, 1))
    start = snapshot.memory_id
    graph = _ObservedGraph(
        {
            start: (
                _ObservedEdge(
                    Action(0),
                    "route-a",
                    hazard=0.9 if risky_first_edge else 0.0,
                ),
                _ObservedEdge(Action(1), "route-b"),
            ),
            "route-a": (_ObservedEdge(Action(2), "low-potential"),),
        },
        {
            start: (0, 1),
            "route-a": (2,),
            "route-b": (),
            "low-potential": (),
        },
        reverse_edges=reverse_edges,
    )
    engine = _GoalPlanningEngine(
        (
            (_PlanningGoal("verified-goal", "reach", 0.9),)
            if goals is None
            else tuple(goals)
        ),
        {
            "route-a": 0.8,
            "route-b": 0.4,
            "low-potential": 0.1,
        },
    )
    config = SearchConfig(
        horizon=1,
        beam_width=2,
        graph_goal_max_edge_risk=0.7,
    )
    result = _engine(
        graph=graph,
        config=config,
        hypothesis_engine=engine,
    ).search(snapshot, competence=CompetenceState())
    return snapshot, result


def test_graph_goal_search_deterministically_routes_multiple_edges_to_lower_phi() -> None:
    _snapshot_row, forward = _graph_goal_search()
    _snapshot_row, reversed_rows = _graph_goal_search(reverse_edges=True)

    for result in (forward, reversed_rows):
        guided = next(
            candidate
            for candidate in result.candidates
            if candidate.action == Action(0)
        )
        assert tuple(action.index for action in guided.path) == (0, 2)
        assert result.graph_goal_id == "verified-goal"
        assert result.graph_goal_target_id == "low-potential"
        assert result.graph_goal_path_length == 2
        assert result.graph_goal_phi_delta == pytest.approx(0.8)
        assert any(term.name == "graph_goal_plan" for term in guided.terms)

    assert tuple(
        tuple(action.key for action in candidate.path)
        for candidate in forward.candidates
    ) == tuple(
        tuple(action.key for action in candidate.path)
        for candidate in reversed_rows.candidates
    )


def test_graph_goal_search_prioritizes_terminal_progress_over_untried_frontier() -> None:
    snapshot = _snapshot(actions=(0, 1))
    completion_id = "next-stage"
    graph = _ObservedGraph(
        {
            snapshot.memory_id: (
                _ObservedEdge(
                    Action(0),
                    completion_id,
                    distance_to_progress=0,
                ),
            ),
        },
        {
            snapshot.memory_id: (0, 1),
            completion_id: (),
        },
        terminal=(completion_id,),
    )
    goals = _GoalPlanningEngine(
        (_PlanningGoal("verified-goal", "reach", 0.9),),
        {completion_id: None},
    )

    result = _engine(
        model=_NeutralFallbackModel(),
        graph=graph,
        hypothesis_engine=goals,
    ).search(snapshot, competence=CompetenceState())

    guided = next(
        candidate
        for candidate in result.candidates
        if any(term.name == "graph_goal_plan" for term in candidate.terms)
    )
    assert guided.action == Action(0)
    assert tuple(action.index for action in guided.path) == (0,)
    assert result.graph_goal_target_id == completion_id
    assert result.graph_goal_path_length == 1
    assert result.graph_goal_phi_delta == 0.0
    term = next(term for term in guided.terms if term.name == "graph_goal_plan")
    assert ":progress:" in term.source


def test_graph_goal_search_finds_deeper_progress_and_rejects_risky_progress() -> None:
    snapshot = _snapshot(actions=(0, 1))
    completion_id = "next-stage"
    goals = _GoalPlanningEngine(
        (_PlanningGoal("verified-goal", "reach", 0.9),),
        {
            "route": 0.9,
            completion_id: None,
        },
        # The exact successful route initially moves away from the goal.
        # Graph planning must reconcile, rather than be vetoed by, that
        # deliberately myopic one-step potential.
        signals={0: -1.0, 1: 0.75},
    )

    safe_graph = _ObservedGraph(
        {
            snapshot.memory_id: (_ObservedEdge(Action(0), "route"),),
            "route": (
                _ObservedEdge(
                    Action(2),
                    completion_id,
                    distance_to_progress=0,
                ),
            ),
        },
        {
            snapshot.memory_id: (0, 1),
            "route": (2,),
            completion_id: (),
        },
        terminal=(completion_id,),
    )
    safe = _engine(
        model=_NeutralFallbackModel(),
        graph=safe_graph,
        hypothesis_engine=goals,
    ).search(snapshot, competence=CompetenceState())
    guided = next(
        candidate
        for candidate in safe.candidates
        if any(term.name == "graph_goal_plan" for term in candidate.terms)
    )
    assert guided.action == Action(0)
    assert tuple(action.index for action in guided.path) == (0, 2)
    assert safe.graph_goal_target_id == completion_id
    progress_term = next(
        term for term in guided.terms if term.name == "graph_goal_plan"
    )
    assert progress_term.value == pytest.approx(
        SearchConfig().graph_goal_bonus_bound
    )
    reconciliation = next(
        term
        for term in guided.terms
        if term.name == "graph_goal_detour_reconciliation"
    )
    assert reconciliation.value == pytest.approx(
        PolicyConfig().hypothesis_weight
    )

    risky_edges = (
        _ObservedEdge(
            Action(0),
            completion_id,
            hazard=0.9,
            distance_to_progress=0,
        ),
        _ObservedEdge(
            Action(0),
            completion_id,
            adverse_terminal_rate=1.0,
            distance_to_progress=0,
        ),
    )
    for risky_edge in risky_edges:
        risky_graph = _ObservedGraph(
            {snapshot.memory_id: (risky_edge,)},
            {
                snapshot.memory_id: (0, 1),
                completion_id: (),
            },
            terminal=(completion_id,),
        )
        risky = _engine(
            model=_NeutralFallbackModel(),
            graph=risky_graph,
            config=SearchConfig(
                horizon=1,
                beam_width=2,
                graph_goal_max_edge_risk=0.7,
            ),
            hypothesis_engine=goals,
        ).search(snapshot, competence=CompetenceState())
        risky_guided = next(
            candidate
            for candidate in risky.candidates
            if any(term.name == "graph_goal_plan" for term in candidate.terms)
        )
        assert risky_guided.action == Action(1)
        assert risky.graph_goal_target_id != completion_id


def test_graph_goal_search_excludes_risky_edges_and_is_noop_without_a_goal() -> None:
    _snapshot_row, safe_alternative = _graph_goal_search(risky_first_edge=True)
    guided = next(
        candidate
        for candidate in safe_alternative.candidates
        if any(term.name == "graph_goal_plan" for term in candidate.terms)
    )
    assert guided.action == Action(1)
    assert safe_alternative.graph_goal_target_id == "route-b"

    _snapshot_row, no_goal = _graph_goal_search(goals=())
    assert no_goal.graph_goal_id == ""
    assert no_goal.graph_goal_target_id == ""
    assert all(
        term.name != "graph_goal_plan"
        for candidate in no_goal.candidates
        for term in candidate.terms
    )


def test_graph_goal_search_does_not_traverse_an_illegal_successor_edge() -> None:
    snapshot = _snapshot(actions=(0, 1))
    graph = _ObservedGraph(
        {
            snapshot.memory_id: (
                _ObservedEdge(Action(0), "stale-route"),
                _ObservedEdge(Action(1), "legal-route"),
            ),
            # This historical edge is present, but action 2 is not in the
            # successor's declared current inventory.
            "stale-route": (_ObservedEdge(Action(2), "illegal-low"),),
        },
        {
            snapshot.memory_id: (0, 1),
            "stale-route": (),
            "legal-route": (),
            "illegal-low": (),
        },
    )
    goals = _GoalPlanningEngine(
        (_PlanningGoal("verified-goal", "reach", 0.9),),
        {
            "stale-route": 0.9,
            "legal-route": 0.4,
            "illegal-low": 0.0,
        },
    )
    config = SearchConfig(horizon=1, beam_width=2)

    result = _engine(
        graph=graph,
        config=config,
        hypothesis_engine=goals,
    ).search(snapshot, competence=CompetenceState())

    assert result.graph_goal_target_id == "legal-route"
    guided = next(
        candidate
        for candidate in result.candidates
        if any(term.name == "graph_goal_plan" for term in candidate.terms)
    )
    assert guided.action == Action(1)
    assert tuple(action.index for action in guided.path) == (1,)


def test_exact_rollout_obeys_each_successors_available_action_inventory() -> None:
    snapshot = _snapshot(actions=(0,))
    graph = _ObservedGraph(
        {
            snapshot.memory_id: (_ObservedEdge(Action(0), "successor"),),
            "successor": (
                _ObservedEdge(Action(0), "illegal", value=1.0),
                _ObservedEdge(Action(1), "legal", value=0.0),
            ),
        },
        {
            snapshot.memory_id: (0,),
            "successor": (1,),
            "illegal": (),
            "legal": (),
        },
    )
    config = SearchConfig(
        horizon=2,
        beam_width=2,
        graph_goal_search_enabled=False,
    )

    result = _engine(graph=graph, config=config).search(
        snapshot,
        competence=CompetenceState(),
    )

    assert tuple(action.index for action in result.candidates[0].path) == (0, 1)


def test_exact_rollout_does_not_expand_a_terminal_successor() -> None:
    snapshot = _snapshot(actions=(0,))
    graph = _ObservedGraph(
        {
            snapshot.memory_id: (_ObservedEdge(Action(0), "terminal"),),
            # A stale action inventory/edge on a terminal frame must never
            # become an imagined continuation.
            "terminal": (_ObservedEdge(Action(1), "phantom", value=10.0),),
        },
        {
            snapshot.memory_id: (0,),
            "terminal": (1,),
            "phantom": (),
        },
        terminal=("terminal",),
    )
    config = SearchConfig(
        horizon=2,
        beam_width=2,
        graph_goal_search_enabled=False,
    )

    result = _engine(graph=graph, config=config).search(
        snapshot,
        competence=CompetenceState(),
    )

    candidate = result.candidates[0]
    assert tuple(action.index for action in candidate.path) == (0,)
    assert all(term.name != "planned_future" for term in candidate.terms)


def test_successor_transposition_retains_the_higher_scoring_exact_path() -> None:
    snapshot = _snapshot(actions=(0,))
    graph = _ObservedGraph(
        {
            snapshot.memory_id: (_ObservedEdge(Action(0), "successor"),),
            "successor": (
                _ObservedEdge(Action(1), "shared", value=-1.0),
                _ObservedEdge(Action(2), "shared", value=1.0),
            ),
        },
        {
            snapshot.memory_id: (0,),
            "successor": (1, 2),
            "shared": (),
        },
    )
    config = SearchConfig(
        horizon=2,
        beam_width=2,
        transposition_limit=100,
        graph_goal_search_enabled=False,
    )

    result = _engine(graph=graph, config=config).search(
        snapshot,
        competence=CompetenceState(),
    )

    candidate = result.candidates[0]
    assert tuple(action.index for action in candidate.path) == (0, 2)
    assert result.transposition_hits == 1
    planned = next(term for term in candidate.terms if term.name == "planned_future")
    assert planned.source == "exact_graph_rollout"


def _request(
    *,
    stage: int = 1,
    state_id: str = "same-state",
    actions: Sequence[Action] = (Action(0), Action(1)),
    metadata: Mapping[str, object] | None = None,
) -> TeacherRequest:
    return TeacherRequest(
        task_id="teacher-task",
        stage=stage,
        state_id=state_id,
        candidates=tuple(actions),
        metadata={} if metadata is None else metadata,
    )


def test_teacher_aggregates_all_weight_for_each_action_before_ranking() -> None:
    teacher = TrajectoryTeacher(
        (
            TeacherExample("teacher-task", 1, "same-state", Action(0), weight=0.4),
            TeacherExample("teacher-task", 1, "same-state", Action(0), weight=0.4),
            TeacherExample("teacher-task", 1, "same-state", Action(1), weight=0.7),
        )
    )

    response = teacher.suggest(_request())

    assert response is not None
    assert response.action == Action(0)
    assert response.confidence == pytest.approx(0.8 / 1.5)
    assert response.metadata["action_support"] == 2
    assert response.metadata["support"] == 3


def test_duplicate_teacher_rows_for_one_action_have_full_confidence() -> None:
    teacher = TrajectoryTeacher(
        (
            TeacherExample("teacher-task", 1, "same-state", Action(1), weight=0.25),
            TeacherExample("teacher-task", 1, "same-state", Action(1), weight=0.75),
        )
    )

    response = teacher.suggest(_request())

    assert response is not None
    assert response.action == Action(1)
    assert response.confidence == pytest.approx(1.0)
    assert response.metadata["action_support"] == 2


def test_teacher_content_fallback_is_restricted_to_adjacent_stages() -> None:
    frame = np.asarray([[3, 3], [0, 3]], dtype=np.uint8)
    content_id = stable_frame_hash(frame, task_id="teacher-task", stage=0)
    teacher = TrajectoryTeacher(
        (
            TeacherExample(
                "teacher-task",
                1,
                "recorded-state",
                Action(1),
                metadata={"frame": frame},
            ),
        )
    )

    adjacent = teacher.suggest(
        _request(stage=2, state_id="live-state", metadata={"content_state_id": content_id})
    )
    far = teacher.suggest(
        _request(stage=3, state_id="live-state", metadata={"content_state_id": content_id})
    )

    assert adjacent is not None and adjacent.action == Action(1)
    assert adjacent.metadata["exact_state"] is False
    assert far is not None and far.action is None
    assert far.confidence == 0.0


def test_categorical_action_decode_never_falls_back_from_invalid_action() -> None:
    adapter = CategoricalActionAdapter(
        action_values=("wait", "move"),
        bootstrap_index=0,
        fallback_indices=(0,),
    )

    with pytest.raises(AdapterError, match=r"outside \[0, 2\)"):
        adapter.decode(Action(99))


def test_positional_action_requires_coordinates_instead_of_silent_payload() -> None:
    adapter = CategoricalActionAdapter(
        action_values=("wait", "point"),
        positional_action_index=1,
    )

    with pytest.raises(AdapterError, match="requires explicit x/y"):
        adapter.decode(Action(1))
