"""Candidate generation and bounded receding-horizon search."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import (
    Action,
    Candidate,
    CompetenceState,
    Prediction,
    ScoreTerm,
    SearchConfig,
    WorldSnapshot,
)
from .executable import ExecutableModelRegistry, ExecutableState
from .memory import (
    EvidenceStore,
    StateGraph,
    combined_object_signature,
    object_summary,
)
from .models import ActionPrior, DynamicsEnsemble
from .policy import PolicyScorer
from .taps import TapBundle


TapReader = Callable[[WorldSnapshot, Action, Prediction], Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class SearchResult:
    candidates: tuple[Candidate, ...]
    expanded_nodes: int
    transposition_hits: int
    effective_horizon: int
    graph_goal_id: str = ""
    graph_goal_kind: str = ""
    graph_goal_target_id: str = ""
    graph_goal_path_length: int = 0
    graph_goal_phi_delta: float = 0.0
    graph_goal_expanded_nodes: int = 0


@dataclass(frozen=True, slots=True)
class _GraphGoalPlan:
    goal_id: str
    goal_kind: str
    target_id: str
    path: tuple[Action, ...]
    phi_delta: float
    bonus: float
    expanded_nodes: int
    target_kind: str


def _latent_key(latent: np.ndarray, obj_summary: np.ndarray, depth: int) -> str:
    digest = blake2b(digest_size=12)
    digest.update(
        np.round(np.asarray(latent, dtype=np.float32), decimals=3).tobytes()
    )
    digest.update(
        np.round(np.asarray(obj_summary, dtype=np.float32), decimals=3).tobytes()
    )
    digest.update(int(depth).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _predicted_effect_signature(prediction: Prediction) -> str:
    """Coarse effect signature available before an action is executed.

    Cross-task positive evidence is admissible only through an effect match.
    The previous caller never supplied one, making that documented route
    unreachable.  These fields are deliberately coarse and model-grounded;
    adverse evidence remains task-local regardless of this estimate.
    """

    object_motion = float(np.linalg.norm(prediction.object_delta))
    latent_motion = float(np.linalg.norm(prediction.latent_delta))
    return (
        "events:|objects:0|"
        f"moved:{int(max(object_motion, latent_motion) > 0.05)}|"
        "transformed:0|"
        f"progress:{int(prediction.progress > 0.05)}|"
        f"hazard:{int(prediction.hazard > 0.5)}"
    )


class CandidateGenerator:
    """Legal actions plus compact object-aware click proposals."""

    def __init__(
        self,
        *,
        click_action_index: int | None,
        config: SearchConfig | None = None,
    ) -> None:
        self.click_action_index = (
            None if click_action_index is None else int(click_action_index)
        )
        self.config = config or SearchConfig()

    def generate(
        self,
        snapshot: WorldSnapshot,
        *,
        fallback_action_indices: Sequence[int] = (),
        extra_click_points: Sequence[tuple[int, int]] = (),
    ) -> tuple[Action, ...]:
        available = tuple(
            dict.fromkeys(
                int(value)
                for value in (
                    snapshot.observation.available_actions
                    or tuple(fallback_action_indices)
                )
            )
        )
        if not available:
            raise ValueError("no legal or fallback actions available")
        actions: list[Action] = []
        click_index = self.click_action_index
        for index in available:
            if click_index is None or index != click_index:
                actions.append(Action(index=index))
                continue
            height, width = snapshot.observation.frame.shape
            points: list[tuple[int, int]] = []
            deduped_points: set[tuple[int, int]] = set()
            # Goal-mismatch cells outrank generic centroid proposals.
            for x, y in tuple(extra_click_points) + self._click_points(snapshot):
                point = (
                    int(np.clip(x, 0, width - 1)),
                    int(np.clip(y, 0, height - 1)),
                )
                if point not in deduped_points:
                    deduped_points.add(point)
                    points.append(point)
            actions.extend(
                Action(index=index, x=x, y=y, name="click")
                for x, y in points[: int(self.config.max_click_candidates)]
            )
        # Stable deduplication protects domains whose adapters repeat action ids.
        deduped: dict[tuple[int, int, int], Action] = {}
        for action in actions:
            deduped.setdefault(action.key, action)
        return tuple(deduped.values())

    def _click_points(self, snapshot: WorldSnapshot) -> tuple[tuple[int, int], ...]:
        height, width = snapshot.observation.frame.shape
        ranked = sorted(
            snapshot.objects,
            key=lambda obj: (
                -float(obj.rewarding),
                -float(obj.controllable),
                float(obj.hazard),
                int(obj.area),
                int(obj.object_id),
            ),
        )
        points: list[tuple[int, int]] = [
            (
                int(np.clip(round(obj.centroid_x), 0, width - 1)),
                int(np.clip(round(obj.centroid_y), 0, height - 1)),
            )
            for obj in ranked
        ]
        # Generic fallbacks ensure click-only empty/background states remain
        # explorable without enumerating the whole grid.
        points.extend(
            [
                (width // 2, height // 2),
                (width // 4, height // 4),
                (3 * width // 4, height // 4),
                (width // 4, 3 * height // 4),
                (3 * width // 4, 3 * height // 4),
            ]
        )
        result: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for x, y in points:
            point = (
                int(np.clip(x, 0, width - 1)),
                int(np.clip(y, 0, height - 1)),
            )
            if point not in seen:
                seen.add(point)
                result.append(point)
        return tuple(result)


class SearchEngine:
    """Exact graph first, learned ensemble second, conservative imagination."""

    def __init__(
        self,
        *,
        model: DynamicsEnsemble,
        graph: StateGraph,
        evidence: EvidenceStore,
        prior: ActionPrior,
        scorer: PolicyScorer,
        click_action_index: int | None,
        config: SearchConfig | None = None,
        tap_reader: TapReader | None = None,
        tap_bundle: TapBundle | None = None,
        executable_registry: ExecutableModelRegistry | None = None,
        ego_model: Any | None = None,
        hypothesis_engine: Any | None = None,
        exogenous_filter: Any | None = None,
    ) -> None:
        self.model = model
        self.graph = graph
        self.evidence = evidence
        self.prior = prior
        self.scorer = scorer
        self.config = config or SearchConfig()
        self.generator = CandidateGenerator(
            click_action_index=click_action_index,
            config=self.config,
        )
        if tap_reader is not None:
            raise ValueError(
                "uncalibrated tap_reader callbacks are not supported; "
                "install a calibrated TapBundle instead"
            )
        self.tap_reader = None
        self.tap_bundle = tap_bundle
        self.executable_registry = executable_registry
        self.ego_model = ego_model
        self.hypothesis_engine = hypothesis_engine
        self.exogenous_filter = exogenous_filter

    def _mask_cells(self, snapshot: WorldSnapshot) -> frozenset[tuple[int, int]]:
        if self.exogenous_filter is None:
            return frozenset()
        return self.exogenous_filter.mask_cells(
            snapshot.observation.task_id,
            tuple(int(v) for v in snapshot.observation.frame.shape),
            stage=snapshot.observation.stage,
        )

    def _graph_goal_plan(
        self,
        snapshot: WorldSnapshot,
        candidates: Sequence[Candidate],
        *,
        mask_cells: frozenset[tuple[int, int]],
    ) -> _GraphGoalPlan | None:
        """Bounded deterministic weighted-A* guidance over observed edges.

        Reach/count potentials are useful heuristics, not admissible distance
        functions through walls, so this deliberately makes no optimality
        claim.  Only verified completion-grounded goals participate, every
        traversed edge is a committed real transition, and the resulting
        first-action bonus still passes through the ordinary risk arbiter.
        """

        cfg = self.config
        engine = self.hypothesis_engine
        if (
            not cfg.graph_goal_search_enabled
            or cfg.graph_goal_bonus_bound <= 0.0
            or engine is None
            or not candidates
        ):
            return None
        outgoing = getattr(self.graph, "outgoing", None)
        available = getattr(self.graph, "available_action_indices", None)
        is_terminal = getattr(self.graph, "is_terminal", None)
        action_stats = getattr(self.graph, "action_stats", None)
        planning_goals = getattr(engine, "planning_goals", None)
        cached_potential = getattr(engine, "cached_goal_potential", None)
        if not all(
            callable(value)
            for value in (
                outgoing,
                available,
                is_terminal,
                action_stats,
                planning_goals,
                cached_potential,
            )
        ):
            return None
        goals = tuple(planning_goals(snapshot, mask_cells=mask_cells))
        if not goals:
            return None
        actionable = {candidate.action.key: candidate for candidate in candidates}
        start_id = str(snapshot.memory_id)
        if not start_id:
            return None

        best_plan: _GraphGoalPlan | None = None
        best_rank: tuple[Any, ...] | None = None
        total_expanded = 0
        expansion_budget = int(cfg.graph_goal_expansion_limit)
        depth_limit = int(cfg.graph_goal_depth_limit)
        improvement_epsilon = float(cfg.graph_goal_improvement_epsilon)
        plateau_tolerance = float(cfg.graph_goal_plateau_tolerance)

        for goal in goals:
            if total_expanded >= expansion_budget:
                break
            start_phi = float(goal.current_potential)
            # (f, g, depth, path keys, state id, path)
            queue: list[
                tuple[
                    float,
                    float,
                    int,
                    tuple[tuple[int, int, int], ...],
                    str,
                    tuple[Action, ...],
                ]
            ] = [(float(cfg.graph_goal_heuristic_weight) * start_phi, 0.0, 0, (), start_id, ())]
            best_g: dict[str, float] = {start_id: 0.0}
            goal_best: tuple[
                tuple[Any, ...],
                str,
                tuple[Action, ...],
                float | None,
                str,
            ] | None = None

            while queue and total_expanded < expansion_budget:
                _f, g_cost, depth, path_keys, state_id, path = heapq.heappop(
                    queue
                )
                if g_cost > best_g.get(state_id, float("inf")) + 1e-12:
                    continue
                total_expanded += 1
                state_edges = tuple(outgoing(state_id))
                legal_indices = set(int(value) for value in available(state_id))
                tried_indices = {int(edge.action.index) for edge in state_edges}
                if state_id == start_id:
                    untried_actions = tuple(
                        candidate.action
                        for candidate in candidates
                        if not bool(
                            self.graph.action_stats(
                                start_id,
                                candidate.action,
                            ).get("known", False)
                        )
                    )
                    frontier = bool(untried_actions)
                else:
                    untried_actions = ()
                    frontier = bool(legal_indices - tried_indices)

                phi = (
                    start_phi
                    if state_id == start_id
                    else cached_potential(
                        task_id=snapshot.observation.task_id,
                        stage=snapshot.observation.stage,
                        hypothesis_id=goal.hypothesis_id,
                        state_id=state_id,
                        mask_cells=mask_cells,
                    )
                )
                phi_value = (
                    None
                    if phi is None or not np.isfinite(float(phi))
                    else float(np.clip(float(phi), 0.0, 1.0))
                )
                target_kind = ""
                category = 99
                target_path = path
                if (
                    depth > 0
                    and phi_value is not None
                    and phi_value <= start_phi - improvement_epsilon
                ):
                    category = 0
                    target_kind = "improving"
                elif (
                    frontier
                    and phi_value is not None
                    and phi_value <= start_phi + plateau_tolerance
                ):
                    category = 1
                    target_kind = "frontier"
                    if depth == 0:
                        # Pick one concrete untried root action; the final
                        # arbiter still decides whether its risk is acceptable.
                        chosen = min(
                            untried_actions,
                            key=lambda action: (
                                -float(actionable[action.key].score),
                                action.key,
                            ),
                        )
                        target_path = (chosen,)
                elif depth > 0 and phi_value is None:
                    # Historical nodes observed before the goal was learned
                    # are explicit unknowns, never falsely treated as Phi=0.
                    category = 2
                    target_kind = "uncached"

                if target_kind and target_path and target_path[0].key in actionable:
                    heuristic = (
                        phi_value if phi_value is not None else 1.0
                    )
                    rank = (
                        category,
                        g_cost
                        + float(cfg.graph_goal_heuristic_weight) * heuristic,
                        len(target_path),
                        path_keys or tuple(action.key for action in target_path),
                        state_id,
                    )
                    if goal_best is None or rank < goal_best[0]:
                        goal_best = (
                            rank,
                            state_id,
                            target_path,
                            phi_value,
                            target_kind,
                        )

                if depth >= depth_limit or bool(is_terminal(state_id)):
                    continue
                for edge in state_edges:
                    if (
                        (
                            state_id != start_id
                            and int(edge.action.index) not in legal_indices
                        )
                        or
                        int(edge.visits) < int(cfg.graph_goal_min_edge_visits)
                        or float(edge.dominant_fraction)
                        < float(cfg.graph_goal_min_dominant_fraction)
                        or float(edge.effective_risk)
                        > float(cfg.graph_goal_max_edge_risk)
                        or edge.successor_id == state_id
                    ):
                        continue
                    next_path = path + (edge.action,)
                    if next_path[0].key not in actionable:
                        continue
                    next_g = (
                        g_cost
                        + float(cfg.graph_goal_step_cost)
                        + float(edge.effective_risk)
                    )
                    edge_stats = action_stats(state_id, edge.action)
                    distance_to_progress = edge_stats.get(
                        "distance_to_progress"
                    )
                    direct_progress = bool(
                        not isinstance(distance_to_progress, (bool, np.bool_))
                        and isinstance(distance_to_progress, (int, np.integer))
                        and int(distance_to_progress) == 0
                    )
                    next_keys = path_keys + (edge.action.key,)
                    if direct_progress:
                        # A committed edge into a progress-labelled state is
                        # stronger evidence than either an untried frontier
                        # or a heuristic Phi decrease.  Boundary successors
                        # routinely have no old-stage frame/potential, so
                        # replay the causal action without pretending that
                        # unknown geometry has Phi=0.  The ordinary legality,
                        # support, dominance, and risk gates above still
                        # apply, including to terminal completion successors.
                        rank = (
                            -1,
                            next_g,
                            len(next_path),
                            next_keys,
                            str(edge.successor_id),
                        )
                        if goal_best is None or rank < goal_best[0]:
                            goal_best = (
                                rank,
                                str(edge.successor_id),
                                next_path,
                                None,
                                "progress",
                            )
                        continue
                    if bool(is_terminal(edge.successor_id)):
                        continue
                    if next_g >= best_g.get(edge.successor_id, float("inf")) - 1e-12:
                        continue
                    best_g[edge.successor_id] = next_g
                    next_phi = cached_potential(
                        task_id=snapshot.observation.task_id,
                        stage=snapshot.observation.stage,
                        hypothesis_id=goal.hypothesis_id,
                        state_id=edge.successor_id,
                        mask_cells=mask_cells,
                    )
                    heuristic = (
                        1.0
                        if next_phi is None or not np.isfinite(float(next_phi))
                        else float(np.clip(float(next_phi), 0.0, 1.0))
                    )
                    heapq.heappush(
                        queue,
                        (
                            next_g
                            + float(cfg.graph_goal_heuristic_weight) * heuristic,
                            next_g,
                            depth + 1,
                            next_keys,
                            str(edge.successor_id),
                            next_path,
                        ),
                    )

            if goal_best is None:
                continue
            rank, target_id, path, target_phi, target_kind = goal_best
            phi_delta = (
                max(0.0, start_phi - target_phi)
                if target_phi is not None
                else 0.0
            )
            epistemic_scale = {
                "progress": 1.0,
                "improving": 1.0,
                "frontier": 0.60,
                "uncached": 0.35,
            }[target_kind]
            # A path ending in committed positive progress is proof, not an
            # exploratory heuristic whose confidence should decay with path
            # length.  Path length already contributes to A*'s g-cost and
            # deterministic ranking; discounting the policy term again let a
            # speculative one-step model prediction beat an exact successful
            # route on tu93.
            depth_scale = (
                1.0
                if target_kind == "progress"
                else 1.0
                / (
                    1.0
                    + float(cfg.graph_goal_step_cost)
                    * max(0, len(path) - 1)
                )
            )
            bonus = (
                float(cfg.graph_goal_bonus_bound)
                * epistemic_scale
                * depth_scale
            )
            plan = _GraphGoalPlan(
                goal_id=goal.hypothesis_id,
                goal_kind=goal.kind,
                target_id=target_id,
                path=path,
                phi_delta=phi_delta,
                bonus=float(
                    np.clip(bonus, 0.0, float(cfg.graph_goal_bonus_bound))
                ),
                expanded_nodes=total_expanded,
                target_kind=target_kind,
            )
            global_rank = (
                *rank,
                goal.kind,
                goal.hypothesis_id,
            )
            if best_rank is None or global_rank < best_rank:
                best_rank = global_rank
                best_plan = plan

        if best_plan is None:
            return None
        return _GraphGoalPlan(
            goal_id=best_plan.goal_id,
            goal_kind=best_plan.goal_kind,
            target_id=best_plan.target_id,
            path=best_plan.path,
            phi_delta=best_plan.phi_delta,
            bonus=best_plan.bonus,
            expanded_nodes=total_expanded,
            target_kind=best_plan.target_kind,
        )

    @staticmethod
    def _apply_graph_goal_plan(
        candidates: Sequence[Candidate],
        plan: _GraphGoalPlan | None,
    ) -> tuple[Candidate, ...]:
        if plan is None or not plan.path:
            return tuple(candidates)
        first_key = plan.path[0].key
        result: list[Candidate] = []
        for candidate in candidates:
            if candidate.action.key != first_key:
                result.append(candidate)
                continue
            # The graph plan exists specifically to permit multi-step detours
            # around obstacles.  A negative one-step potential on that first
            # edge is therefore already accounted for by the exact path and
            # must not veto it a second time.  Keep the original term visible
            # in the audit trace and add an equally bounded reconciliation
            # term rather than deleting evidence.
            local_detour_penalty = -sum(
                float(term.value)
                for term in candidate.terms
                if term.influences_score
                and term.name == "hypothesis_potential"
                and float(term.value) < 0.0
            )
            additions: tuple[ScoreTerm, ...] = (
                ScoreTerm(
                    "graph_goal_plan",
                    plan.bonus,
                    "planning",
                    source=(
                        f"goal_graph:{plan.goal_kind}:"
                        f"{plan.target_kind}:{plan.goal_id}"
                    ),
                ),
            )
            if local_detour_penalty > 0.0:
                additions += (
                    ScoreTerm(
                        "graph_goal_detour_reconciliation",
                        local_detour_penalty,
                        "planning",
                        source=f"goal_graph:{plan.goal_id}",
                    ),
                )
            result.append(
                Candidate(
                    action=candidate.action,
                    prediction=candidate.prediction,
                    terms=candidate.terms + additions,
                    depth=candidate.depth,
                    path=plan.path,
                    source=candidate.source,
                )
            )
        return tuple(result)

    def _prediction(self, snapshot: WorldSnapshot, action: Action) -> Prediction:
        exact = self.graph.exact_prediction(
            snapshot.memory_id,
            action,
            latent_dim=self.model.latent_dim,
            object_dim=self.model.object_dim,
        )
        if exact is not None:
            return exact
        registry = self.executable_registry
        if registry is not None:
            ranked = registry.ranked_verifications()
            for report in ranked:
                executable = registry.predict(
                    report.model_id,
                    ExecutableState.from_snapshot(snapshot),
                    action,
                )
                if not executable.applicable:
                    continue
                uncertainty = float(
                    np.clip(
                        report.error_rate
                        + (1.0 - report.coverage)
                        + 0.5 * report.mean_auxiliary_error,
                        0.0,
                        1.0,
                    )
                )
                return Prediction(
                    change_probability=executable.change_probability,
                    progress=executable.progress_delta,
                    value=executable.progress_delta - executable.hazard,
                    hazard=executable.hazard,
                    terminal=executable.terminal,
                    uncertainty=uncertainty,
                    latent_delta=np.zeros(
                        self.model.latent_dim,
                        dtype=np.float32,
                    ),
                    object_delta=np.zeros(
                        self.model.object_dim,
                        dtype=np.float32,
                    ),
                    exact_successor_id=executable.successor_state_id,
                    source=f"executable:{report.model_id}",
                )
        return self.model.predict(snapshot, action)

    def _root_candidate(
        self,
        snapshot: WorldSnapshot,
        action: Action,
        competence: CompetenceState,
    ) -> Candidate:
        prediction = self._prediction(snapshot, action)
        object_signature = combined_object_signature(snapshot.objects, action)
        evidence = self.evidence.score(
            task_id=snapshot.observation.task_id,
            stage=snapshot.observation.stage,
            state_id=snapshot.memory_id,
            action=action,
            object_signature=object_signature,
            effect_signature=_predicted_effect_signature(prediction),
        )
        provisional = Candidate(action=action, prediction=prediction)
        tap_values: dict[str, float] = {}
        if self.tap_bundle is not None:
            tap_values.update(
                self.tap_bundle.pointwise_values(snapshot, provisional)
            )
        ego_hazard = (
            float(self.ego_model.motion_hazard(snapshot, action))
            if self.ego_model is not None
            else 0.0
        )
        graph_stats = self.graph.action_stats(snapshot.memory_id, action)
        hypothesis_signal = 0.0
        if self.hypothesis_engine is not None:
            successor = graph_stats.get("successor_id")
            hypothesis_signal = float(
                self.hypothesis_engine.candidate_signal(
                    snapshot,
                    action,
                    successor_memory_id=(
                        str(successor) if successor else None
                    ),
                    mask_cells=self._mask_cells(snapshot),
                )
            )
        return self.scorer.score(
            snapshot=snapshot,
            action=action,
            prediction=prediction,
            evidence=evidence,
            competence=competence,
            prior_score=self.prior.score(
                snapshot.observation.task_id,
                action.index,
            ),
            graph_stats=graph_stats,
            tap_values=tap_values,
            ego_hazard=ego_hazard,
            hypothesis_signal=hypothesis_signal,
            imagined=False,
            depth=0,
            path=(action,),
            source="root",
        )

    def search(
        self,
        snapshot: WorldSnapshot,
        *,
        competence: CompetenceState,
        fallback_action_indices: Sequence[int] = (),
        horizon_adjustment: int = 0,
    ) -> SearchResult:
        mask_cells = self._mask_cells(snapshot)
        extra_click_points: tuple[tuple[int, int], ...] = ()
        if self.hypothesis_engine is not None:
            extra_click_points = tuple(
                self.hypothesis_engine.mismatch_points(
                    snapshot,
                    mask_cells=mask_cells,
                )
            )
        actions = self.generator.generate(
            snapshot,
            fallback_action_indices=fallback_action_indices,
            extra_click_points=extra_click_points,
        )
        roots = [
            self._root_candidate(snapshot, action, competence)
            for action in actions
        ]
        if self.tap_bundle is not None:
            roots = list(self.tap_bundle.apply_pairwise(snapshot, roots))
            roots = list(
                self.tap_bundle.retain(
                    snapshot,
                    roots,
                    top_k=max(2, int(self.config.beam_width)),
                )
            )
        graph_plan = self._graph_goal_plan(
            snapshot,
            roots,
            mask_cells=mask_cells,
        )
        horizon = max(
            1,
            int(self.config.horizon) + int(horizon_adjustment),
        )
        registry = self.executable_registry
        if registry is not None and roots:
            plan = registry.plan(
                ExecutableState.from_snapshot(snapshot),
                tuple(candidate.action for candidate in roots),
                horizon=horizon,
            )
            if plan is not None:
                first_action = plan.actions[0].key
                plan_value = float(
                    np.clip(0.10 + 0.05 * plan.score, -0.20, 0.25)
                )
                roots = [
                    Candidate(
                        action=candidate.action,
                        prediction=candidate.prediction,
                        terms=(
                            candidate.terms
                            + (
                                ScoreTerm(
                                    "executable_plan",
                                    plan_value,
                                    "planning",
                                    source=f"executable:{plan.model_id}",
                                ),
                            )
                            if candidate.action.key == first_action
                            else candidate.terms
                        ),
                        depth=candidate.depth,
                        path=candidate.path,
                        source=candidate.source,
                    )
                    for candidate in roots
                ]
        if horizon <= 1 or not roots:
            planned_roots = self._apply_graph_goal_plan(roots, graph_plan)
            return SearchResult(
                candidates=planned_roots,
                expanded_nodes=len(roots)
                + (graph_plan.expanded_nodes if graph_plan is not None else 0),
                transposition_hits=0,
                effective_horizon=horizon,
                graph_goal_id=(
                    graph_plan.goal_id if graph_plan is not None else ""
                ),
                graph_goal_kind=(
                    graph_plan.goal_kind if graph_plan is not None else ""
                ),
                graph_goal_target_id=(
                    graph_plan.target_id if graph_plan is not None else ""
                ),
                graph_goal_path_length=(
                    len(graph_plan.path) if graph_plan is not None else 0
                ),
                graph_goal_phi_delta=(
                    graph_plan.phi_delta if graph_plan is not None else 0.0
                ),
                graph_goal_expanded_nodes=(
                    graph_plan.expanded_nodes if graph_plan is not None else 0
                ),
            )

        expanded = len(roots) + (
            graph_plan.expanded_nodes if graph_plan is not None else 0
        )
        transposition_hits = 0
        enriched: list[Candidate] = []
        width = max(1, int(self.config.beam_width))
        future_actions = actions
        frame_shape = tuple(int(v) for v in snapshot.observation.frame.shape)
        root_object_summary = object_summary(snapshot.objects)
        _Node = tuple[
            float,
            np.ndarray,
            np.ndarray,
            tuple[Action, ...],
            str | None,
            bool,
        ]
        for root in roots:
            latent = (
                snapshot.representation.global_vector
                + root.prediction.latent_delta
            )
            obj = root_object_summary + root.prediction.object_delta
            beam: list[_Node] = [
                (
                    0.0,
                    latent,
                    obj,
                    (root.action,),
                    root.prediction.exact_successor_id,
                    root.prediction.source == "exact_graph",
                )
            ]
            seen: set[str] = set()
            best_future = float("-inf")
            best_path = (root.action,)
            best_all_exact = False
            for depth in range(1, horizon):
                next_by_key: dict[str, _Node] = {}
                for (
                    cumulative,
                    current_latent,
                    current_obj,
                    path,
                    state_id,
                    all_exact,
                ) in beam:
                    node_actions = future_actions
                    if state_id:
                        outgoing = getattr(self.graph, "outgoing", None)
                        available = getattr(
                            self.graph,
                            "available_action_indices",
                            None,
                        )
                        is_terminal = getattr(self.graph, "is_terminal", None)
                        if callable(is_terminal) and bool(is_terminal(state_id)):
                            # An exact successor marked terminal has no legal
                            # continuation in this rollout, even if an adapter
                            # happened to expose stale actions on its final
                            # frame or the same pixels were later revisited.
                            continue
                        if callable(outgoing) and callable(available):
                            legal_indices = {
                                int(index) for index in available(state_id)
                            }
                            exact_actions = [
                                edge.action
                                for edge in outgoing(state_id)
                                if int(edge.action.index) in legal_indices
                            ]
                            seen_indexes = {
                                int(action.index) for action in exact_actions
                            }
                            legal_untried = [
                                Action(index=int(index))
                                for index in legal_indices
                                if int(index) not in seen_indexes
                                and int(index)
                                != self.generator.click_action_index
                            ]
                            deduped_actions = {
                                action.key: action
                                for action in (*exact_actions, *legal_untried)
                            }
                            node_actions = tuple(
                                deduped_actions[key]
                                for key in sorted(deduped_actions)
                            )
                    for action in node_actions:
                        # Exact graph successors chain first; the learned
                        # ensemble covers unvisited continuations.
                        prediction = None
                        graph_stats: Mapping[str, object] = {}
                        used_exact = False
                        if state_id:
                            prediction = self.graph.exact_prediction(
                                state_id,
                                action,
                                latent_dim=self.model.latent_dim,
                                object_dim=self.model.object_dim,
                            )
                            used_exact = prediction is not None
                            if used_exact:
                                graph_stats = self.graph.action_stats(
                                    state_id,
                                    action,
                                )
                        if prediction is None:
                            prediction = self.model.predict_from_features(
                                latent=current_latent,
                                obj_summary=current_obj,
                                action=action,
                                frame_shape=frame_shape,
                                topology=snapshot.topology,
                                state_id="",
                            )
                        next_state_id = prediction.exact_successor_id
                        # Exact graph chains retain their observed identity and
                        # graph statistics. Learned continuations deliberately
                        # remain state-blank.
                        evidence = self.evidence.score(
                            task_id=snapshot.observation.task_id,
                            stage=snapshot.observation.stage,
                            state_id=str(state_id) if used_exact else "",
                            action=action,
                            object_signature=combined_object_signature(
                                snapshot.objects,
                                action,
                            ),
                            effect_signature=_predicted_effect_signature(prediction),
                        )
                        hypothesis_signal = 0.0
                        if self.hypothesis_engine is not None:
                            hypothesis_signal = float(
                                self.hypothesis_engine.rollout_signal(
                                    snapshot,
                                    action,
                                    current_state_id=state_id,
                                    successor_state_id=next_state_id,
                                    mask_cells=mask_cells,
                                )
                            )
                        imagined = self.scorer.score(
                            snapshot=snapshot,
                            action=action,
                            prediction=prediction,
                            evidence=evidence,
                            competence=competence,
                            prior_score=self.prior.score(
                                snapshot.observation.task_id,
                                action.index,
                            ),
                            graph_stats=graph_stats,
                            hypothesis_signal=hypothesis_signal,
                            imagined=True,
                            depth=depth,
                            path=path + (action,),
                            source="imagined",
                        )
                        discount = 0.85**depth
                        total = cumulative + discount * imagined.score
                        next_latent = current_latent + prediction.latent_delta
                        next_obj = current_obj + prediction.object_delta
                        state_key = (
                            f"exact:{next_state_id}"
                            if next_state_id
                            else (
                                "latent:"
                                f"{_latent_key(next_latent, next_obj, depth)}"
                            )
                        )
                        key = f"{state_key}|depth:{depth}"
                        node: _Node = (
                            total,
                            next_latent,
                            next_obj,
                            path + (action,),
                            next_state_id,
                            all_exact and used_exact,
                        )
                        previous = next_by_key.get(key)
                        if previous is not None:
                            transposition_hits += 1
                            if (
                                total < previous[0] - 1e-12
                                or (
                                    abs(total - previous[0]) <= 1e-12
                                    and tuple(a.key for a in node[3])
                                    >= tuple(a.key for a in previous[3])
                                )
                            ):
                                continue
                        elif len(seen) >= int(self.config.transposition_limit):
                            continue
                        else:
                            seen.add(key)
                        next_by_key[key] = node
                        expanded += 1
                next_beam = list(next_by_key.values())
                if not next_beam:
                    break
                next_beam.sort(
                    key=lambda row: (
                        -row[0],
                        tuple(action.key for action in row[3]),
                    )
                )
                beam = next_beam[:width]
                if beam[0][0] > best_future:
                    best_future = float(beam[0][0])
                    best_path = beam[0][3]
                    best_all_exact = bool(beam[0][5])
            if np.isfinite(best_future):
                terms = root.terms + (
                    ScoreTerm(
                        "planned_future",
                        0.50 * best_future,
                        "planning",
                        source=(
                            "exact_graph_rollout"
                            if best_all_exact
                            else "ensemble_rollout"
                        ),
                    ),
                )
                enriched.append(
                    Candidate(
                        action=root.action,
                        prediction=root.prediction,
                        terms=terms,
                        depth=0,
                        path=best_path,
                        source=root.source,
                    )
                )
            else:
                enriched.append(root)
        planned_candidates = self._apply_graph_goal_plan(
            enriched,
            graph_plan,
        )
        return SearchResult(
            candidates=planned_candidates,
            expanded_nodes=expanded,
            transposition_hits=transposition_hits,
            effective_horizon=horizon,
            graph_goal_id=(
                graph_plan.goal_id if graph_plan is not None else ""
            ),
            graph_goal_kind=(
                graph_plan.goal_kind if graph_plan is not None else ""
            ),
            graph_goal_target_id=(
                graph_plan.target_id if graph_plan is not None else ""
            ),
            graph_goal_path_length=(
                len(graph_plan.path) if graph_plan is not None else 0
            ),
            graph_goal_phi_delta=(
                graph_plan.phi_delta if graph_plan is not None else 0.0
            ),
            graph_goal_expanded_nodes=(
                graph_plan.expanded_nodes if graph_plan is not None else 0
            ),
        )


__all__ = [
    "CandidateGenerator",
    "SearchEngine",
    "SearchResult",
    "TapReader",
]
