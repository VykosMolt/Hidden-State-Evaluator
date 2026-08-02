"""Explicit score assembly and the single risk-arbitration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .contracts import (
    Action,
    Candidate,
    CompetenceState,
    PolicyConfig,
    Prediction,
    ScoreTerm,
    SearchConfig,
    WorldSnapshot,
)
from .memory import EvidenceScore


def exploration_scale(competence: CompetenceState) -> float:
    """Single exploration multiplier over the operational self-state.

    Miscalibrated hazard predictions brake exploration before a spent risk
    budget does, so an agent that cannot yet trust its own danger estimates
    does not chase uncertainty bonuses.
    """

    stagnation_boost = min(float(competence.stagnation_count) / 12.0, 1.0)
    calibration_brake = 1.0 - min(float(competence.hazard_calibration_error), 0.8)
    # A consumed exogenous budget shifts weight from probing to exploiting.
    urgency_brake = 1.0 - 0.6 * float(np.clip(competence.time_pressure, 0.0, 1.0))
    return float(
        np.clip(
            (
                0.35
                + float(np.clip(competence.expected_learning_gain, 0.0, 1.0))
                + 0.5 * stagnation_boost
            )
            * calibration_brake
            * urgency_brake
            * (0.35 + 0.65 * float(np.clip(competence.remaining_risk_budget, 0.0, 1.0))),
            0.0,
            2.0,
        )
    )


@dataclass(frozen=True, slots=True)
class ArbitrationResult:
    candidate: Candidate
    method: str
    safe_candidate_count: int
    effective_risk: float


def effective_risk(candidate: Candidate) -> float:
    """Return the single aggregate risk used by every policy boundary."""

    explicit_risk = sum(
        max(0.0, term.value)
        for term in candidate.terms
        if term.name.startswith("risk:")
    )
    if explicit_risk > 0.0:
        policy_risk = explicit_risk
    else:
        # Compatibility for externally constructed candidates which have not
        # yet adopted explicit non-scoring risk terms.
        policy_risk = sum(
            -term.value
            for term in candidate.terms
            if term.name in {"evidence_negative", "target_hazard_affordance"}
            and term.value < 0.0
        )
    return float(
        np.clip(
            candidate.prediction.hazard
            + 0.5 * candidate.prediction.terminal
            + policy_risk,
            0.0,
            2.0,
        )
    )


def _bounded(value: float, bound: float = 1.0) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -abs(bound), abs(bound)))


def _target_object(snapshot: WorldSnapshot, action: Action):
    if not action.has_position or not snapshot.objects:
        return None
    inside = [
        obj
        for obj in snapshot.objects
        if obj.bbox[0] <= action.x <= obj.bbox[2]
        and obj.bbox[1] <= action.y <= obj.bbox[3]
    ]
    pool = inside or list(snapshot.objects)
    return min(
        pool,
        key=lambda obj: (obj.centroid_x - action.x) ** 2
        + (obj.centroid_y - action.y) ** 2,
    )


class PolicyScorer:
    """Converts evidence into auditable, conservation-checked score terms."""

    def __init__(
        self,
        config: PolicyConfig | None = None,
        *,
        search: SearchConfig | None = None,
    ) -> None:
        self.config = config or PolicyConfig()
        self.search = search or SearchConfig()

    def score(
        self,
        *,
        snapshot: WorldSnapshot,
        action: Action,
        prediction: Prediction,
        evidence: EvidenceScore,
        competence: CompetenceState,
        prior_score: float = 0.0,
        graph_stats: Mapping[str, object] | None = None,
        tap_values: Mapping[str, float] | None = None,
        ego_hazard: float = 0.0,
        hypothesis_signal: float = 0.0,
        imagined: bool = False,
        depth: int = 0,
        path: Sequence[Action] = (),
        source: str = "proposal",
    ) -> Candidate:
        cfg = self.config
        graph = graph_stats or {}
        target = _target_object(snapshot, action)
        terms: list[ScoreTerm] = [
            ScoreTerm(
                "expected_progress",
                _bounded(prediction.progress) * cfg.progress_weight,
                "learned_value",
                source=prediction.source,
            ),
            ScoreTerm(
                "expected_value",
                _bounded(prediction.value) * cfg.value_weight,
                "learned_value",
                source=prediction.source,
            ),
            ScoreTerm(
                "expected_hazard",
                -float(np.clip(prediction.hazard, 0.0, 1.0)) * cfg.hazard_weight,
                "safety",
                source=prediction.source,
            ),
            ScoreTerm(
                "terminal_risk",
                -0.5
                * float(np.clip(prediction.terminal, 0.0, 1.0))
                * cfg.hazard_weight,
                "safety",
                source=prediction.source,
            ),
        ]

        exploration_term_indexes: tuple[int, int] | None = None
        if imagined:
            terms.append(
                ScoreTerm(
                    "model_uncertainty_penalty",
                    -float(prediction.uncertainty)
                    * cfg.imagined_uncertainty_weight,
                    "uncertainty",
                    source=prediction.source,
                )
            )
        else:
            # Reserve the stable trace positions now, then fill their values
            # after every explicit risk sensor has been assembled below.
            first_index = len(terms)
            terms.append(
                ScoreTerm(
                    "epistemic_exploration",
                    0.0,
                    "exploration",
                    source=prediction.source,
                )
            )
            terms.append(
                ScoreTerm(
                    "expected_learning_progress",
                    0.0,
                    "exploration",
                    source="competence",
                )
            )
            exploration_term_indexes = (first_index, first_index + 1)

        terms.extend(
            [
                ScoreTerm(
                    "evidence_positive",
                    evidence.positive * cfg.memory_weight,
                    "memory",
                    source="evidence_store",
                ),
                ScoreTerm(
                    "evidence_negative",
                    -evidence.negative * cfg.memory_weight,
                    "safety",
                    source="evidence_store",
                ),
                ScoreTerm(
                    "risk:evidence_negative",
                    float(np.clip(evidence.negative, 0.0, 1.0)),
                    "safety",
                    influences_score=False,
                    source="evidence_store",
                ),
                ScoreTerm(
                    "evidence_no_change",
                    -0.35 * evidence.no_change * cfg.memory_weight,
                    "memory",
                    source="evidence_store",
                ),
                ScoreTerm(
                    "action_prior",
                    0.15 * _bounded(prior_score),
                    "learned_value",
                    source="outcome_prior",
                ),
                ScoreTerm(
                    "action_cost",
                    -float(cfg.action_cost) * max(1, depth + 1),
                    "efficiency",
                    source="config",
                ),
            ]
        )

        if bool(graph.get("known", False)):
            terms.append(
                ScoreTerm(
                    "exact_graph",
                    float(self.search.exact_graph_bonus),
                    "world_model",
                    source="exact_graph",
                )
            )
        no_change_rate = float(graph.get("no_change_rate", 0.0) or 0.0)
        self_loop_rate = float(graph.get("self_loop_rate", 0.0) or 0.0)
        if no_change_rate:
            terms.append(
                ScoreTerm(
                    "known_no_change",
                    -float(self.search.no_change_penalty)
                    * float(np.clip(no_change_rate, 0.0, 1.0)),
                    "efficiency",
                    source="exact_graph",
                )
            )
        if self_loop_rate:
            terms.append(
                ScoreTerm(
                    "known_loop",
                    -float(self.search.loop_penalty)
                    * float(np.clip(self_loop_rate, 0.0, 1.0)),
                    "efficiency",
                    source="exact_graph",
                )
            )
        distance = graph.get("distance_to_progress")
        if isinstance(distance, (int, float)) and np.isfinite(float(distance)):
            terms.append(
                ScoreTerm(
                    "graph_progress_distance",
                    0.20 / (1.0 + max(float(distance), 0.0)),
                    "world_model",
                    source="exact_graph",
                )
            )

        goal_signal = _bounded(hypothesis_signal, 1.0)
        if goal_signal != 0.0:
            terms.append(
                ScoreTerm(
                    "hypothesis_potential",
                    float(cfg.hypothesis_weight) * goal_signal,
                    "goal",
                    source="hypothesis_engine",
                )
            )

        ego_signal = float(np.clip(_bounded(ego_hazard), 0.0, 1.0))
        terms.extend(
            (
                ScoreTerm(
                    "ego_motion_hazard",
                    -float(cfg.ego_hazard_weight) * ego_signal,
                    "safety",
                    source="control_attribution",
                ),
                ScoreTerm(
                    "risk:ego_motion_hazard",
                    ego_signal,
                    "safety",
                    influences_score=False,
                    source="control_attribution",
                ),
            )
        )

        if target is not None:
            reachable = target.object_id in snapshot.topology.reachable_object_ids
            terms.extend(
                [
                    ScoreTerm(
                        "target_reward_affordance",
                        0.30 * float(np.clip(target.rewarding, 0.0, 1.0)),
                        "affordance",
                        source="object_model",
                    ),
                    ScoreTerm(
                        "target_controllability",
                        0.12 * float(np.clip(target.controllable, 0.0, 1.0)),
                        "affordance",
                        source="object_model",
                    ),
                    ScoreTerm(
                        "target_hazard_affordance",
                        -0.45 * float(np.clip(target.hazard, 0.0, 1.0)),
                        "safety",
                        source="object_model",
                    ),
                    ScoreTerm(
                        "risk:target_hazard_affordance",
                        float(np.clip(target.hazard, 0.0, 1.0)),
                        "safety",
                        influences_score=False,
                        source="object_model",
                    ),
                    ScoreTerm(
                        "target_reachable",
                        0.06 if reachable else -0.03,
                        "topology",
                        source="perception",
                    ),
                ]
            )

        bounded_taps = [
            (str(name), _bounded(float(raw_value), 1.0))
            for name, raw_value in sorted((tap_values or {}).items())
        ]
        total_magnitude = sum(abs(value) for _name, value in bounded_taps)
        tap_scale = 1.0 / max(1.0, total_magnitude)
        for name, raw_value in bounded_taps:
            value = raw_value * tap_scale
            role = name.split(":", 1)[0]
            group = "safety" if role == "hazard" else "tap"
            terms.append(
                ScoreTerm(
                    f"tap:{name}",
                    0.10 * value,
                    group,
                    source="role_tap",
                )
            )
            if role == "hazard":
                terms.append(
                    ScoreTerm(
                        f"risk:tap:{name}",
                        abs(float(raw_value)),
                        "safety",
                        influences_score=False,
                        source="role_tap",
                    )
                )

        candidate = Candidate(
            action=action,
            prediction=prediction,
            terms=tuple(terms),
            depth=int(depth),
            path=tuple(path),
            source=str(source),
        )
        if exploration_term_indexes is not None:
            risk_ok = effective_risk(candidate) <= float(cfg.risk_limit)
            scale = exploration_scale(competence)
            epistemic_index, learning_index = exploration_term_indexes
            terms[epistemic_index] = ScoreTerm(
                "epistemic_exploration",
                (
                    float(prediction.uncertainty)
                    * cfg.exploration_weight
                    * scale
                    if risk_ok
                    else 0.0
                ),
                "exploration",
                source=prediction.source,
            )
            terms[learning_index] = ScoreTerm(
                "expected_learning_progress",
                (
                    float(prediction.uncertainty)
                    * float(competence.expected_learning_gain)
                    * cfg.learning_progress_weight
                    if risk_ok
                    else 0.0
                ),
                "exploration",
                source="competence",
            )
            candidate = Candidate(
                action=action,
                prediction=prediction,
                terms=tuple(terms),
                depth=int(depth),
                path=tuple(path),
                source=str(source),
            )
        # Conservation is a runtime invariant, not merely a test convention.
        recomputed = sum(term.value for term in terms if term.influences_score)
        if not np.isfinite(recomputed) or abs(candidate.score - recomputed) > 1e-9:
            raise RuntimeError("candidate score conservation failure")
        return candidate


class RiskArbiter:
    """Final selector shared by greedy, beam, and epsilon exploration."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    @staticmethod
    def effective_risk(candidate: Candidate) -> float:
        return effective_risk(candidate)

    def choose(
        self,
        candidates: Sequence[Candidate],
        *,
        rng: np.random.Generator,
        epsilon: float | None = None,
    ) -> ArbitrationResult:
        if not candidates:
            raise ValueError("cannot arbitrate an empty candidate set")
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                candidate.action.index,
                candidate.action.y,
                candidate.action.x,
            ),
        )
        safe = [
            candidate
            for candidate in ordered
            if self.effective_risk(candidate) <= float(self.config.risk_limit)
        ]
        pool = safe if safe else sorted(
            ordered,
            key=lambda candidate: (
                self.effective_risk(candidate),
                -candidate.score,
                candidate.action.index,
                candidate.action.y,
                candidate.action.x,
            ),
        )
        eps = (
            float(self.config.exploration_epsilon)
            if epsilon is None
            else float(epsilon)
        )
        if safe and eps > 0.0 and rng.random() < np.clip(eps, 0.0, 1.0):
            index = int(rng.integers(0, len(safe)))
            chosen = safe[index]
            method = "safe_random"
        else:
            chosen = pool[0]
            method = "score" if safe else "least_risk"
        return ArbitrationResult(
            candidate=chosen,
            method=method,
            safe_candidate_count=len(safe),
            effective_risk=self.effective_risk(chosen),
        )


__all__ = [
    "ArbitrationResult",
    "PolicyScorer",
    "RiskArbiter",
    "effective_risk",
    "exploration_scale",
]
