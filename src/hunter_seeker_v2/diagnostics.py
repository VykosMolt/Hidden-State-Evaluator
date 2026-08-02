"""Typed compact traces derived from the runtime's immutable records."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    BoundaryKind,
    Candidate,
    Decision,
    EventKind,
    Transition,
    WorldEvent,
)
from .policy import effective_risk


def _trace_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _trace_sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def _trace_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _trace_int(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _trace_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _trace_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a bool")
    return bool(value)


def _trace_action(value: Any, *, name: str) -> tuple[int, int, int]:
    rows = _trace_sequence(value, name=name)
    if len(rows) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    return (
        _trace_int(rows[0], name=f"{name}[0]"),
        _trace_int(rows[1], name=f"{name}[1]"),
        _trace_int(rows[2], name=f"{name}[2]"),
    )


def _trace_row(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    row = _trace_mapping(value, name=name)
    missing = required.difference(row)
    if missing:
        raise ValueError(f"{name} is missing fields {sorted(missing)!r}")
    unexpected = set(row).difference(required | optional)
    if unexpected:
        raise ValueError(f"{name} has unexpected fields {sorted(unexpected)!r}")
    return row


@dataclass(frozen=True, slots=True)
class CandidateTrace:
    action: tuple[int, int, int]
    score: float
    risk: float
    uncertainty: float
    source: str
    path: tuple[tuple[int, int, int], ...]
    terms: tuple[tuple[str, float, str, bool], ...]


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    decision_id: str
    task_id: str
    stage: int
    step: int
    state_id: str
    chosen_action: tuple[int, int, int]
    chosen_score: float
    method: str
    safe_candidate_count: int
    expanded_nodes: int
    transposition_hits: int
    effective_horizon: int
    candidates: tuple[CandidateTrace, ...]


@dataclass(frozen=True, slots=True)
class TransitionTrace:
    transition_id: str
    decision_id: str
    task_id: str
    stage: int
    step: int
    state_id: str
    successor_id: str
    action: tuple[int, int, int]
    frame_changed: bool
    reward: float
    progress: float
    hazard: float
    terminal: bool
    boundary: str
    events: tuple[str, ...]
    model_loss: float
    prediction_error: float


def _candidate_trace_from_state(
    value: Any,
    *,
    name: str,
) -> CandidateTrace:
    row = _trace_row(
        value,
        name=name,
        required=frozenset(
            {
                "action",
                "score",
                "risk",
                "uncertainty",
                "source",
            }
        ),
        optional=frozenset({"path", "terms"}),
    )
    raw_path = _trace_sequence(
        row.get("path", ()),
        name=f"{name}.path",
    )
    raw_terms = _trace_sequence(
        row.get("terms", ()),
        name=f"{name}.terms",
    )
    terms: list[tuple[str, float, str, bool]] = []
    for term_index, raw_term in enumerate(raw_terms):
        term_name = f"{name}.terms[{term_index}]"
        term = _trace_sequence(raw_term, name=term_name)
        if len(term) != 4:
            raise ValueError(f"{term_name} must contain exactly four values")
        terms.append(
            (
                _trace_string(term[0], name=f"{term_name}[0]"),
                _trace_float(term[1], name=f"{term_name}[1]"),
                _trace_string(term[2], name=f"{term_name}[2]"),
                _trace_bool(term[3], name=f"{term_name}[3]"),
            )
        )
    return CandidateTrace(
        action=_trace_action(row["action"], name=f"{name}.action"),
        score=_trace_float(row["score"], name=f"{name}.score"),
        risk=_trace_float(
            row["risk"],
            name=f"{name}.risk",
            minimum=0.0,
            maximum=2.0,
        ),
        uncertainty=_trace_float(
            row["uncertainty"],
            name=f"{name}.uncertainty",
            minimum=0.0,
            maximum=1.0,
        ),
        source=_trace_string(row["source"], name=f"{name}.source"),
        path=tuple(
            _trace_action(
                action,
                name=f"{name}.path[{action_index}]",
            )
            for action_index, action in enumerate(raw_path)
        ),
        terms=tuple(terms),
    )


def candidate_trace(candidate: Candidate) -> CandidateTrace:
    if not isinstance(candidate, Candidate):
        raise TypeError("candidate_trace requires a Candidate")
    # Keep the audit trace on the exact same aggregate-risk definition as the
    # final arbiter, including explicit non-scoring evidence/tap/affordance
    # risk terms.
    risk = effective_risk(candidate)
    return CandidateTrace(
        action=candidate.action.key,
        score=_trace_float(candidate.score, name="candidate score"),
        risk=risk,
        uncertainty=_trace_float(
            candidate.prediction.uncertainty,
            name="candidate uncertainty",
            minimum=0.0,
            maximum=1.0,
        ),
        source=str(candidate.source),
        path=tuple(action.key for action in candidate.path),
        terms=tuple(
            (
                str(term.name),
                _trace_float(term.value, name="candidate term value"),
                str(term.group),
                _trace_bool(
                    term.influences_score,
                    name="candidate term influences_score",
                ),
            )
            for term in candidate.terms
        ),
    )


class Diagnostics:
    """Bounded trace store; public summaries are derived, never handwritten."""

    def __init__(self, *, max_decisions: int = 10_000, max_transitions: int = 50_000) -> None:
        decision_limit = _trace_int(
            max_decisions,
            name="max_decisions",
            minimum=1,
        )
        transition_limit = _trace_int(
            max_transitions,
            name="max_transitions",
            minimum=1,
        )
        self._decisions: deque[DecisionTrace] = deque(maxlen=decision_limit)
        self._transitions: deque[TransitionTrace] = deque(
            maxlen=transition_limit
        )

    def __deepcopy__(self, memo: dict[int, object]) -> "Diagnostics":
        """Copy the queues while sharing their immutable trace records.

        ``observe`` stages components transactionally on every real action.
        Reconstructing the full nested candidate trace history made that cost
        grow with the lifetime of a persistent agent even though every stored
        row is frozen. Independent deques are sufficient for rollback.
        """

        duplicate = type(self)(
            max_decisions=int(self._decisions.maxlen or 1),
            max_transitions=int(self._transitions.maxlen or 1),
        )
        memo[id(self)] = duplicate
        duplicate._decisions = deque(
            self._decisions,
            maxlen=self._decisions.maxlen,
        )
        duplicate._transitions = deque(
            self._transitions,
            maxlen=self._transitions.maxlen,
        )
        return duplicate

    @property
    def decisions(self) -> tuple[DecisionTrace, ...]:
        return tuple(self._decisions)

    @property
    def transitions(self) -> tuple[TransitionTrace, ...]:
        return tuple(self._transitions)

    def record_decision(
        self,
        decision: Decision,
        *,
        method: str,
        safe_candidate_count: int,
        expanded_nodes: int,
        transposition_hits: int,
        effective_horizon: int,
    ) -> DecisionTrace:
        if not isinstance(decision, Decision):
            raise TypeError("record_decision requires a Decision")
        trace = DecisionTrace(
            decision_id=decision.decision_id,
            task_id=str(decision.metadata.get("task_id", "")),
            stage=_trace_int(
                decision.metadata.get("stage", 1),
                name="decision stage",
                minimum=1,
            ),
            step=_trace_int(decision.step, name="decision step", minimum=0),
            state_id=decision.snapshot_id,
            chosen_action=decision.action.key,
            chosen_score=_trace_float(
                decision.score,
                name="decision chosen_score",
            ),
            method=str(method),
            safe_candidate_count=_trace_int(
                safe_candidate_count,
                name="safe_candidate_count",
                minimum=0,
            ),
            expanded_nodes=_trace_int(
                expanded_nodes,
                name="expanded_nodes",
                minimum=0,
            ),
            transposition_hits=_trace_int(
                transposition_hits,
                name="transposition_hits",
                minimum=0,
            ),
            effective_horizon=_trace_int(
                effective_horizon,
                name="effective_horizon",
                minimum=0,
            ),
            candidates=tuple(candidate_trace(row) for row in decision.candidates),
        )
        self._decisions.append(trace)
        return trace

    def record_transition(
        self,
        transition: Transition,
        *,
        events: Sequence[WorldEvent],
        model_loss: float,
        prediction_error: float,
    ) -> TransitionTrace:
        if not isinstance(transition, Transition):
            raise TypeError("record_transition requires a Transition")
        event_rows = tuple(events)
        if not all(isinstance(event, WorldEvent) for event in event_rows):
            raise TypeError("record_transition events must be WorldEvent values")
        trace = TransitionTrace(
            transition_id=transition.transition_id,
            decision_id=transition.decision_id,
            task_id=transition.task_id,
            stage=_trace_int(
                transition.stage,
                name="transition stage",
                minimum=1,
            ),
            step=_trace_int(
                transition.step,
                name="transition step",
                minimum=0,
            ),
            state_id=transition.before.state_id,
            successor_id=transition.after_state_id,
            action=transition.action.key,
            frame_changed=_trace_bool(
                transition.frame_changed,
                name="transition frame_changed",
            ),
            reward=_trace_float(
                transition.outcome.reward,
                name="transition reward",
            ),
            progress=_trace_float(
                transition.outcome.progress_delta,
                name="transition progress",
            ),
            hazard=_trace_float(
                transition.outcome.hazard,
                name="transition hazard",
                minimum=0.0,
                maximum=1.0,
            ),
            terminal=_trace_bool(
                transition.outcome.terminated,
                name="transition terminal",
            ),
            boundary=transition.outcome.boundary.value,
            events=tuple(event.kind.value for event in event_rows),
            model_loss=_trace_float(model_loss, name="model_loss"),
            prediction_error=_trace_float(
                prediction_error,
                name="prediction_error",
            ),
        )
        self._transitions.append(trace)
        return trace

    def summary(self) -> dict[str, Any]:
        transitions = tuple(self._transitions)
        decisions = tuple(self._decisions)
        event_counts = Counter(
            event
            for transition in transitions
            for event in transition.events
        )
        method_counts = Counter(decision.method for decision in decisions)
        return {
            "decision_count": len(decisions),
            "transition_count": len(transitions),
            "terminal_count": sum(int(row.terminal) for row in transitions),
            "progress_total": float(sum(row.progress for row in transitions)),
            "reward_total": float(sum(row.reward for row in transitions)),
            "hazard_total": float(sum(row.hazard for row in transitions)),
            "change_rate": float(
                np.mean([row.frame_changed for row in transitions])
                if transitions
                else 0.0
            ),
            "mean_model_loss": float(
                np.mean([row.model_loss for row in transitions])
                if transitions
                else 0.0
            ),
            "mean_prediction_error": float(
                np.mean([row.prediction_error for row in transitions])
                if transitions
                else 0.0
            ),
            "event_counts": dict(sorted(event_counts.items())),
            "selection_methods": dict(sorted(method_counts.items())),
        }

    def export_state(self) -> dict[str, Any]:
        return {
            "decisions": [asdict(row) for row in self._decisions],
            "transitions": [asdict(row) for row in self._transitions],
            "max_decisions": self._decisions.maxlen,
            "max_transitions": self._transitions.maxlen,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "Diagnostics":
        if not isinstance(state, Mapping):
            raise ValueError("diagnostics state must be a mapping")
        max_decisions = _trace_int(
            state.get("max_decisions", 10_000),
            name="max_decisions",
            minimum=1,
        )
        max_transitions = _trace_int(
            state.get("max_transitions", 50_000),
            name="max_transitions",
            minimum=1,
        )
        raw_decisions = _trace_sequence(
            state.get("decisions", ()),
            name="diagnostic decisions",
        )
        raw_transitions = _trace_sequence(
            state.get("transitions", ()),
            name="diagnostic transitions",
        )
        if len(raw_decisions) > max_decisions:
            raise ValueError(
                "diagnostic decision rows exceed the declared capacity"
            )
        if len(raw_transitions) > max_transitions:
            raise ValueError(
                "diagnostic transition rows exceed the declared capacity"
            )
        diagnostics = cls(
            max_decisions=max_decisions,
            max_transitions=max_transitions,
        )
        decision_ids: set[str] = set()
        for row_index, raw in enumerate(raw_decisions):
            row_name = f"diagnostic decision[{row_index}]"
            row = _trace_row(
                raw,
                name=row_name,
                required=frozenset(
                    {
                        "decision_id",
                        "task_id",
                        "stage",
                        "step",
                        "state_id",
                        "chosen_action",
                        "chosen_score",
                        "method",
                        "safe_candidate_count",
                        "expanded_nodes",
                        "transposition_hits",
                        "effective_horizon",
                        "candidates",
                    }
                ),
            )
            decision_id = _trace_string(
                row["decision_id"],
                name=f"{row_name}.decision_id",
            )
            if decision_id in decision_ids:
                raise ValueError(
                    f"duplicate diagnostic decision_id {decision_id!r}"
                )
            decision_ids.add(decision_id)
            raw_candidates = _trace_sequence(
                row["candidates"],
                name=f"{row_name}.candidates",
            )
            candidates = tuple(
                _candidate_trace_from_state(
                    candidate,
                    name=f"{row_name}.candidates[{candidate_index}]",
                )
                for candidate_index, candidate in enumerate(raw_candidates)
            )
            diagnostics._decisions.append(
                DecisionTrace(
                    decision_id=decision_id,
                    task_id=_trace_string(
                        row["task_id"],
                        name=f"{row_name}.task_id",
                    ),
                    stage=_trace_int(
                        row["stage"],
                        name=f"{row_name}.stage",
                        minimum=1,
                    ),
                    step=_trace_int(
                        row["step"],
                        name=f"{row_name}.step",
                        minimum=0,
                    ),
                    state_id=_trace_string(
                        row["state_id"],
                        name=f"{row_name}.state_id",
                    ),
                    chosen_action=_trace_action(
                        row["chosen_action"],
                        name=f"{row_name}.chosen_action",
                    ),
                    chosen_score=_trace_float(
                        row["chosen_score"],
                        name=f"{row_name}.chosen_score",
                    ),
                    method=_trace_string(
                        row["method"],
                        name=f"{row_name}.method",
                    ),
                    safe_candidate_count=_trace_int(
                        row["safe_candidate_count"],
                        name=f"{row_name}.safe_candidate_count",
                        minimum=0,
                    ),
                    expanded_nodes=_trace_int(
                        row["expanded_nodes"],
                        name=f"{row_name}.expanded_nodes",
                        minimum=0,
                    ),
                    transposition_hits=_trace_int(
                        row["transposition_hits"],
                        name=f"{row_name}.transposition_hits",
                        minimum=0,
                    ),
                    effective_horizon=_trace_int(
                        row["effective_horizon"],
                        name=f"{row_name}.effective_horizon",
                        minimum=0,
                    ),
                    candidates=candidates,
                )
            )
        transition_ids: set[str] = set()
        for row_index, raw in enumerate(raw_transitions):
            row_name = f"diagnostic transition[{row_index}]"
            row = _trace_row(
                raw,
                name=row_name,
                required=frozenset(
                    {
                        "transition_id",
                        "decision_id",
                        "task_id",
                        "stage",
                        "step",
                        "state_id",
                        "successor_id",
                        "action",
                        "frame_changed",
                        "reward",
                        "progress",
                        "hazard",
                        "terminal",
                        "boundary",
                        "events",
                        "model_loss",
                        "prediction_error",
                    }
                ),
            )
            transition_id = _trace_string(
                row["transition_id"],
                name=f"{row_name}.transition_id",
            )
            if transition_id in transition_ids:
                raise ValueError(
                    f"duplicate diagnostic transition_id {transition_id!r}"
                )
            transition_ids.add(transition_id)
            boundary_raw = _trace_string(
                row["boundary"],
                name=f"{row_name}.boundary",
            )
            try:
                boundary = BoundaryKind(boundary_raw).value
            except ValueError as exc:
                raise ValueError(
                    f"{row_name}.boundary is not a valid BoundaryKind"
                ) from exc
            raw_events = _trace_sequence(
                row["events"],
                name=f"{row_name}.events",
            )
            events: list[str] = []
            for event_index, raw_event in enumerate(raw_events):
                event_name = _trace_string(
                    raw_event,
                    name=f"{row_name}.events[{event_index}]",
                )
                try:
                    events.append(EventKind(event_name).value)
                except ValueError as exc:
                    raise ValueError(
                        f"{row_name}.events[{event_index}] is not a valid EventKind"
                    ) from exc
            diagnostics._transitions.append(
                TransitionTrace(
                    transition_id=transition_id,
                    decision_id=_trace_string(
                        row["decision_id"],
                        name=f"{row_name}.decision_id",
                    ),
                    task_id=_trace_string(
                        row["task_id"],
                        name=f"{row_name}.task_id",
                    ),
                    stage=_trace_int(
                        row["stage"],
                        name=f"{row_name}.stage",
                        minimum=1,
                    ),
                    step=_trace_int(
                        row["step"],
                        name=f"{row_name}.step",
                        minimum=0,
                    ),
                    state_id=_trace_string(
                        row["state_id"],
                        name=f"{row_name}.state_id",
                    ),
                    successor_id=_trace_string(
                        row["successor_id"],
                        name=f"{row_name}.successor_id",
                    ),
                    action=_trace_action(
                        row["action"],
                        name=f"{row_name}.action",
                    ),
                    frame_changed=_trace_bool(
                        row["frame_changed"],
                        name=f"{row_name}.frame_changed",
                    ),
                    reward=_trace_float(
                        row["reward"],
                        name=f"{row_name}.reward",
                    ),
                    progress=_trace_float(
                        row["progress"],
                        name=f"{row_name}.progress",
                    ),
                    hazard=_trace_float(
                        row["hazard"],
                        name=f"{row_name}.hazard",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    terminal=_trace_bool(
                        row["terminal"],
                        name=f"{row_name}.terminal",
                    ),
                    boundary=boundary,
                    events=tuple(events),
                    model_loss=_trace_float(
                        row["model_loss"],
                        name=f"{row_name}.model_loss",
                    ),
                    prediction_error=_trace_float(
                        row["prediction_error"],
                        name=f"{row_name}.prediction_error",
                    ),
                )
            )
        return diagnostics


__all__ = [
    "CandidateTrace",
    "DecisionTrace",
    "Diagnostics",
    "TransitionTrace",
    "candidate_trace",
]
