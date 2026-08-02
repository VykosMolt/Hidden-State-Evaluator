"""Exact state graph and append-only evidence memory.

The ledger is the durable source of truth.  Policy-facing memories are derived
indexes over immutable transition records, so terminal, click-cooldown, recent
replay, and positive affordance evidence cannot disagree about what happened.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    Action,
    BoundaryKind,
    EvidenceScope,
    MemoryConfig,
    Prediction,
    Transition,
    WorldEvent,
)


def _action_key(action: Action | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(action, Action):
        return action.key
    raw = tuple(action)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise ValueError("action key values must be integers")
    values = tuple(int(v) for v in raw)
    if len(values) != 3:
        raise ValueError(f"action key must have three integers, got {values!r}")
    return values


def _state_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _state_float(
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
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _state_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a bool")
    return bool(value)


def _sig_similarity(left: str, right: str) -> float:
    """Cheap deterministic similarity for structured object/effect signatures."""

    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    a = set(left.split("|"))
    b = set(right.split("|"))
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    transition_id: str
    task_id: str
    stage: int
    step: int
    state_id: str
    successor_id: str
    action: tuple[int, int, int]
    object_signature: str
    effect_signature: str
    event_kinds: tuple[str, ...]
    frame_changed: bool
    progress: float
    reward: float
    hazard: float
    terminal: bool
    boundary: str
    uncertainty: float
    confidence: float
    scope: EvidenceScope
    source: str = "online"
    model_version: str = ""

    def __post_init__(self) -> None:
        transition_id = str(self.transition_id).strip()
        task_id = str(self.task_id).strip()
        state_id = str(self.state_id).strip()
        successor_id = str(self.successor_id).strip()
        if not transition_id or not task_id or not state_id or not successor_id:
            raise ValueError("evidence identities must be non-empty")
        object.__setattr__(self, "transition_id", transition_id)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "successor_id", successor_id)
        object.__setattr__(
            self,
            "stage",
            _state_int(self.stage, name="evidence stage", minimum=1),
        )
        object.__setattr__(
            self,
            "step",
            _state_int(self.step, name="evidence step"),
        )
        object.__setattr__(self, "action", _action_key(self.action))
        object.__setattr__(
            self,
            "event_kinds",
            tuple(str(value) for value in self.event_kinds),
        )
        object.__setattr__(
            self,
            "frame_changed",
            _state_bool(self.frame_changed, name="evidence frame_changed"),
        )
        object.__setattr__(
            self,
            "progress",
            _state_float(self.progress, name="evidence progress"),
        )
        object.__setattr__(
            self,
            "reward",
            _state_float(self.reward, name="evidence reward"),
        )
        object.__setattr__(
            self,
            "hazard",
            _state_float(
                self.hazard,
                name="evidence hazard",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "terminal",
            _state_bool(self.terminal, name="evidence terminal"),
        )
        object.__setattr__(
            self,
            "boundary",
            BoundaryKind(self.boundary).value,
        )
        object.__setattr__(
            self,
            "uncertainty",
            _state_float(
                self.uncertainty,
                name="evidence uncertainty",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "confidence",
            _state_float(
                self.confidence,
                name="evidence confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(self, "scope", EvidenceScope(self.scope))
        object.__setattr__(self, "object_signature", str(self.object_signature))
        object.__setattr__(self, "effect_signature", str(self.effect_signature))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "model_version", str(self.model_version))

    @property
    def positive(self) -> bool:
        return self.progress > 0.0 or self.reward > 0.0

    @property
    def completed(self) -> bool:
        return self.boundary in {
            BoundaryKind.LEVEL_COMPLETED.value,
            BoundaryKind.GAME_COMPLETED.value,
        }

    @property
    def adverse_terminal(self) -> bool:
        """Whether termination is safety evidence rather than success.

        Successful level/game completion is structurally terminal, but it is
        not a hazard.  Keeping that distinction here prevents every derived
        memory view from independently (and inconsistently) rediscovering it.
        """

        return bool(self.terminal and not self.completed)

    @property
    def negative(self) -> bool:
        return bool(
            self.hazard > 0.0
            or self.boundary
            in {
                BoundaryKind.DEATH.value,
                BoundaryKind.ENVIRONMENT_ERROR.value,
            }
            or self.adverse_terminal
        )


@dataclass(frozen=True, slots=True)
class EvidenceScore:
    total: float
    positive: float
    negative: float
    no_change: float
    support: int
    negative_support: int
    positive_support: int
    matched_transition_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InverseActionPrediction:
    action: tuple[int, int, int] | None
    confidence: float
    support: int
    distribution: tuple[tuple[tuple[int, int, int], float], ...] = ()


@dataclass(slots=True)
class _EdgeStats:
    action: tuple[int, int, int]
    successor_counts: Counter[str] = field(default_factory=Counter)
    visits: int = 0
    changes: int = 0
    terminal_count: int = 0
    progress_sum: float = 0.0
    value_sum: float = 0.0
    hazard_sum: float = 0.0
    uncertainty_sum: float = 0.0

    def observe(self, record: EvidenceRecord) -> None:
        self.successor_counts[record.successor_id] += 1
        self.visits += 1
        self.changes += int(record.frame_changed)
        # This statistic feeds the policy's terminal-*risk* term.  Successful
        # completion is terminal control flow, not adverse terminal evidence.
        self.terminal_count += int(record.adverse_terminal)
        self.progress_sum += float(record.progress)
        self.value_sum += float(record.reward + record.progress)
        self.hazard_sum += float(record.hazard)
        self.uncertainty_sum += float(record.uncertainty)

    @property
    def dominant_successor(self) -> str | None:
        if not self.successor_counts:
            return None
        # Counter.most_common() breaks ties by insertion order.  JSON
        # ``sort_keys`` can change that order across a checkpoint roundtrip,
        # silently changing exact predictions.  Make the tie break canonical.
        return min(
            self.successor_counts,
            key=lambda state_id: (
                -int(self.successor_counts[state_id]),
                str(state_id),
            ),
        )


@dataclass(slots=True)
class _NodeStats:
    state_id: str
    visits: int = 0
    available_actions: tuple[int, ...] = ()
    latent: np.ndarray | None = None
    object_summary: np.ndarray | None = None
    progress: float = 0.0
    terminal: bool = False
    distance_to_progress: int | None = None
    edges: dict[tuple[int, int, int], _EdgeStats] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphEdgeView:
    action: Action
    successor_id: str
    visits: int
    dominant_fraction: float
    change_rate: float
    hazard: float
    adverse_terminal_rate: float
    uncertainty: float

    @property
    def effective_risk(self) -> float:
        return float(
            np.clip(self.hazard + self.adverse_terminal_rate, 0.0, 2.0)
        )


class StateGraph:
    """Hash-addressed graph of real observed transitions only."""

    def __init__(self) -> None:
        self._nodes: dict[str, _NodeStats] = {}
        self._reverse: dict[str, set[str]] = defaultdict(set)
        self._progress_states: set[str] = set()
        self._transition_ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(node.edges) for node in self._nodes.values())

    @property
    def revisit_count(self) -> int:
        """Committed before-state visits beyond each node's first."""

        return sum(max(0, node.visits - 1) for node in self._nodes.values())

    def node(self, state_id: str) -> _NodeStats | None:
        """Return a detached diagnostic snapshot of one graph node.

        The mutable node/edge accumulators are private durable state.  Returning
        them directly allowed an innocent diagnostic caller to rewrite visit
        counts, successor frequencies, or feature arrays without committing a
        transition.
        """

        node = self._nodes.get(str(state_id))
        return None if node is None else copy.deepcopy(node)

    def observe(
        self,
        transition: Transition,
        *,
        before_latent: np.ndarray,
        after_latent: np.ndarray,
        before_object_summary: np.ndarray,
        after_object_summary: np.ndarray,
        uncertainty: float,
        events: Sequence[WorldEvent] = (),
        object_signature: str = "",
        effect_signature: str = "",
        source: str = "online",
        model_version: str = "",
        before_state_id: str | None = None,
        after_state_id: str | None = None,
    ) -> EvidenceRecord:
        if transition.transition_id in self._transition_ids:
            raise ValueError(f"duplicate transition id: {transition.transition_id}")

        before_id = str(before_state_id or transition.before.state_id)
        after_id = str(after_state_id or transition.after_state_id)
        record = evidence_from_transition(
            transition,
            uncertainty=uncertainty,
            events=events,
            object_signature=object_signature,
            effect_signature=effect_signature,
            source=source,
            model_version=model_version,
            state_id=before_id,
            successor_id=after_id,
        )
        before = self._nodes.setdefault(
            before_id,
            _NodeStats(state_id=before_id),
        )
        after = self._nodes.setdefault(
            after_id,
            _NodeStats(state_id=after_id),
        )
        before.visits += 1
        before.available_actions = tuple(
            int(value) for value in transition.before.observation.available_actions
        )
        before.latent = np.asarray(before_latent, dtype=np.float32).copy()
        before.object_summary = np.asarray(before_object_summary, dtype=np.float32).copy()
        before.progress = float(transition.before.observation.progress)
        after.latent = np.asarray(after_latent, dtype=np.float32).copy()
        after.object_summary = np.asarray(after_object_summary, dtype=np.float32).copy()
        after.progress = float(transition.after_observation.progress)
        # State identity does not include termination.  If the same visible
        # state has ever been observed as terminal, preserve that conservative
        # fact rather than letting a later ambiguous/nonterminal visit make
        # graph search expand through a death screen.
        after.terminal = bool(after.terminal or transition.outcome.terminated)
        after.available_actions = tuple(
            int(value) for value in transition.after_observation.available_actions
        )

        edge = before.edges.setdefault(
            transition.action.key,
            _EdgeStats(action=transition.action.key),
        )
        edge.observe(record)
        self._reverse[after_id].add(before_id)
        self._transition_ids.add(transition.transition_id)
        # Incremental relabeling is exact because the graph only ever gains
        # edges and progress states, so distances are monotone non-increasing.
        if transition.outcome.progress_delta > 0.0 or transition.outcome.completed:
            self._progress_states.add(after_id)
            if after.distance_to_progress is None or after.distance_to_progress > 0:
                after.distance_to_progress = 0
                self._relax_predecessors(after_id)
        if after.distance_to_progress is not None:
            shortcut = int(after.distance_to_progress) + 1
            if (
                before.distance_to_progress is None
                or before.distance_to_progress > shortcut
            ):
                before.distance_to_progress = shortcut
                self._relax_predecessors(before_id)
        return record

    def _relax_predecessors(self, state_id: str) -> None:
        queue: deque[str] = deque((state_id,))
        while queue:
            current = queue.popleft()
            node = self._nodes.get(current)
            if node is None or node.distance_to_progress is None:
                continue
            next_distance = int(node.distance_to_progress) + 1
            for predecessor in sorted(self._reverse.get(current, ())):
                pred_node = self._nodes.get(predecessor)
                if pred_node is None:
                    continue
                if (
                    pred_node.distance_to_progress is None
                    or pred_node.distance_to_progress > next_distance
                ):
                    pred_node.distance_to_progress = next_distance
                    queue.append(predecessor)

    def _back_label_distances(self) -> None:
        for node in self._nodes.values():
            node.distance_to_progress = None
        queue: deque[tuple[str, int]] = deque(
            (state_id, 0) for state_id in sorted(self._progress_states)
        )
        seen: set[str] = set()
        while queue:
            state_id, distance = queue.popleft()
            if state_id in seen:
                continue
            seen.add(state_id)
            node = self._nodes.get(state_id)
            if node is not None:
                node.distance_to_progress = int(distance)
            for predecessor in sorted(self._reverse.get(state_id, ())):
                queue.append((predecessor, distance + 1))

    def exact_prediction(
        self,
        state_id: str,
        action: Action,
        *,
        latent_dim: int,
        object_dim: int,
    ) -> Prediction | None:
        node = self._nodes.get(str(state_id))
        if node is None:
            return None
        edge = node.edges.get(action.key)
        if edge is None or edge.visits <= 0:
            return None

        successor = self._nodes.get(edge.dominant_successor or "")
        if (
            successor is not None
            and node.latent is not None
            and successor.latent is not None
            and node.latent.shape == successor.latent.shape
        ):
            latent_delta = successor.latent - node.latent
        else:
            latent_delta = np.zeros(latent_dim, dtype=np.float32)
        if (
            successor is not None
            and node.object_summary is not None
            and successor.object_summary is not None
            and node.object_summary.shape == successor.object_summary.shape
        ):
            object_delta = successor.object_summary - node.object_summary
        else:
            object_delta = np.zeros(object_dim, dtype=np.float32)

        visits = float(edge.visits)
        successor_total = float(sum(edge.successor_counts.values()))
        dominant = float(edge.successor_counts.most_common(1)[0][1])
        stochasticity = 1.0 - dominant / max(successor_total, 1.0)
        return Prediction(
            change_probability=edge.changes / visits,
            progress=edge.progress_sum / visits,
            value=edge.value_sum / visits,
            hazard=edge.hazard_sum / visits,
            terminal=edge.terminal_count / visits,
            uncertainty=float(np.clip(stochasticity + 1.0 / (visits + 2.0), 0.0, 1.0)),
            latent_delta=latent_delta,
            object_delta=object_delta,
            exact_successor_id=edge.dominant_successor,
            source="exact_graph",
        )

    def action_stats(
        self,
        state_id: str,
        action: Action,
    ) -> Mapping[str, float | bool | int | str | None]:
        node = self._nodes.get(str(state_id))
        edge = node.edges.get(action.key) if node is not None else None
        if edge is None or edge.visits <= 0:
            return {
                "known": False,
                "visits": 0,
                "no_change_rate": 0.0,
                "self_loop_rate": 0.0,
                "distance_to_progress": None,
            }
        self_loops = int(edge.successor_counts.get(str(state_id), 0))
        successor = self._nodes.get(edge.dominant_successor or "")
        return {
            "known": True,
            "visits": int(edge.visits),
            "no_change_rate": float(1.0 - edge.changes / max(edge.visits, 1)),
            "self_loop_rate": float(self_loops / max(edge.visits, 1)),
            "distance_to_progress": (
                successor.distance_to_progress if successor is not None else None
            ),
            "successor_id": edge.dominant_successor,
        }

    def available_action_indices(self, state_id: str) -> tuple[int, ...]:
        node = self._nodes.get(str(state_id))
        return () if node is None else tuple(node.available_actions)

    def is_terminal(self, state_id: str) -> bool:
        node = self._nodes.get(str(state_id))
        return bool(node.terminal) if node is not None else False

    def outgoing(self, state_id: str) -> tuple[GraphEdgeView, ...]:
        """Immutable dominant-edge summaries for deterministic graph search."""

        node = self._nodes.get(str(state_id))
        if node is None:
            return ()
        rows: list[GraphEdgeView] = []
        for key, edge in sorted(node.edges.items()):
            successor_id = edge.dominant_successor
            if successor_id is None or edge.visits <= 0:
                continue
            visits = float(edge.visits)
            dominant = float(edge.successor_counts[successor_id])
            rows.append(
                GraphEdgeView(
                    action=Action(*key),
                    successor_id=successor_id,
                    visits=int(edge.visits),
                    dominant_fraction=dominant / max(visits, 1.0),
                    change_rate=float(edge.changes / max(visits, 1.0)),
                    hazard=float(edge.hazard_sum / max(visits, 1.0)),
                    adverse_terminal_rate=float(
                        edge.terminal_count / max(visits, 1.0)
                    ),
                    uncertainty=float(
                        edge.uncertainty_sum / max(visits, 1.0)
                    ),
                )
            )
        return tuple(rows)

    def export_state(self) -> dict[str, Any]:
        nodes: dict[str, Any] = {}
        for state_id, node in self._nodes.items():
            nodes[state_id] = {
                "visits": node.visits,
                "available_actions": list(node.available_actions),
                "latent": node.latent.tolist() if node.latent is not None else None,
                "object_summary": (
                    node.object_summary.tolist()
                    if node.object_summary is not None
                    else None
                ),
                "progress": node.progress,
                "terminal": node.terminal,
                "distance_to_progress": node.distance_to_progress,
                "edges": {
                    ",".join(str(v) for v in key): {
                        "successor_counts": dict(edge.successor_counts),
                        "visits": edge.visits,
                        "changes": edge.changes,
                        "terminal_count": edge.terminal_count,
                        "progress_sum": edge.progress_sum,
                        "value_sum": edge.value_sum,
                        "hazard_sum": edge.hazard_sum,
                        "uncertainty_sum": edge.uncertainty_sum,
                    }
                    for key, edge in node.edges.items()
                },
            }
        return {
            "nodes": nodes,
            "progress_states": sorted(self._progress_states),
            "transition_ids": sorted(self._transition_ids),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "StateGraph":
        graph = cls()
        raw_nodes = state.get("nodes", {})
        if not isinstance(raw_nodes, Mapping):
            raise ValueError("state graph nodes must be a mapping")
        referenced_successors: set[str] = set()
        total_edge_visits = 0
        for state_id, raw_row in raw_nodes.items():
            state_id = str(state_id)
            if not state_id:
                raise ValueError("state graph node IDs must be non-empty")
            if not isinstance(raw_row, Mapping):
                raise ValueError("state graph node rows must be mappings")
            row = dict(raw_row)
            raw_actions = row.get("available_actions", ())
            if isinstance(raw_actions, (str, bytes)) or not isinstance(
                raw_actions,
                Sequence,
            ):
                raise ValueError("state graph available actions must be a sequence")
            available_actions = tuple(
                _state_int(
                    value,
                    name=f"state graph action inventory for {state_id!r}",
                    minimum=-2**63,
                )
                for value in raw_actions
            )
            if len(set(available_actions)) != len(available_actions):
                raise ValueError("state graph available actions must be unique")
            latent = (
                np.asarray(row["latent"], dtype=np.float32)
                if row.get("latent") is not None
                else None
            )
            object_summary = (
                np.asarray(row["object_summary"], dtype=np.float32)
                if row.get("object_summary") is not None
                else None
            )
            if latent is not None and (
                latent.ndim != 1 or not np.isfinite(latent).all()
            ):
                raise ValueError("state graph latent vectors must be finite and 1D")
            if object_summary is not None and (
                object_summary.ndim != 1
                or not np.isfinite(object_summary).all()
            ):
                raise ValueError(
                    "state graph object summaries must be finite and 1D"
                )
            raw_distance = row.get("distance_to_progress")
            if raw_distance is not None:
                _state_int(
                    raw_distance,
                    name="state graph distance_to_progress",
                )
            node = _NodeStats(
                state_id=state_id,
                visits=_state_int(
                    row.get("visits", 0),
                    name=f"state graph visits for {state_id!r}",
                ),
                available_actions=available_actions,
                latent=latent,
                object_summary=object_summary,
                progress=_state_float(
                    row.get("progress", 0.0),
                    name=f"state graph progress for {state_id!r}",
                ),
                terminal=_state_bool(
                    row.get("terminal", False),
                    name=f"state graph terminal for {state_id!r}",
                ),
                distance_to_progress=raw_distance,
            )
            raw_edges = row.get("edges", {})
            if not isinstance(raw_edges, Mapping):
                raise ValueError("state graph edges must be a mapping")
            node_edge_visits = 0
            for key_text, raw_edge_row in raw_edges.items():
                if not isinstance(raw_edge_row, Mapping):
                    raise ValueError("state graph edge rows must be mappings")
                edge_row = dict(raw_edge_row)
                try:
                    key = _action_key(
                        tuple(int(v) for v in str(key_text).split(","))
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError("state graph action key is malformed") from exc
                edge = _EdgeStats(action=key)
                raw_counts = edge_row.get("successor_counts", {})
                if not isinstance(raw_counts, Mapping) or not raw_counts:
                    raise ValueError(
                        "state graph successor counts must be a non-empty mapping"
                    )
                counts = {
                    str(successor_id): _state_int(
                        count,
                        name="state graph successor count",
                        minimum=1,
                    )
                    for successor_id, count in raw_counts.items()
                }
                if any(not successor_id for successor_id in counts):
                    raise ValueError("state graph successor IDs must be non-empty")
                edge.successor_counts.update(counts)
                edge.visits = _state_int(
                    edge_row.get("visits", 0),
                    name="state graph edge visits",
                    minimum=1,
                )
                if sum(counts.values()) != edge.visits:
                    raise ValueError(
                        "state graph successor counts must sum to edge visits"
                    )
                edge.changes = _state_int(
                    edge_row.get("changes", 0),
                    name="state graph edge changes",
                )
                edge.terminal_count = _state_int(
                    edge_row.get("terminal_count", 0),
                    name="state graph edge terminal_count",
                )
                if edge.changes > edge.visits or edge.terminal_count > edge.visits:
                    raise ValueError(
                        "state graph event counts cannot exceed edge visits"
                    )
                edge.progress_sum = _state_float(
                    edge_row.get("progress_sum", 0.0),
                    name="state graph edge progress_sum",
                )
                edge.value_sum = _state_float(
                    edge_row.get("value_sum", 0.0),
                    name="state graph edge value_sum",
                )
                edge.hazard_sum = _state_float(
                    edge_row.get("hazard_sum", 0.0),
                    name="state graph edge hazard_sum",
                    minimum=0.0,
                    maximum=float(edge.visits),
                )
                edge.uncertainty_sum = _state_float(
                    edge_row.get("uncertainty_sum", 0.0),
                    name="state graph edge uncertainty_sum",
                    minimum=0.0,
                    maximum=float(edge.visits),
                )
                node.edges[key] = edge
                for successor_id in edge.successor_counts:
                    graph._reverse[successor_id].add(state_id)
                    referenced_successors.add(successor_id)
                node_edge_visits += edge.visits
            if node_edge_visits != node.visits:
                raise ValueError(
                    "state graph node visits must equal its edge visit total"
                )
            total_edge_visits += node_edge_visits
            graph._nodes[state_id] = node
        missing_successors = referenced_successors - set(graph._nodes)
        if missing_successors:
            raise ValueError(
                "state graph edges reference missing successor nodes"
            )
        raw_progress_states = state.get("progress_states", ())
        raw_transition_ids = state.get("transition_ids", ())
        for name, rows in (
            ("progress_states", raw_progress_states),
            ("transition_ids", raw_transition_ids),
        ):
            if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
                raise ValueError(f"state graph {name} must be a sequence")
        progress_rows = tuple(str(value) for value in raw_progress_states)
        transition_rows = tuple(str(value) for value in raw_transition_ids)
        if (
            len(set(progress_rows)) != len(progress_rows)
            or len(set(transition_rows)) != len(transition_rows)
        ):
            raise ValueError("state graph identifier sequences must be unique")
        graph._progress_states = set(progress_rows)
        if not graph._progress_states.issubset(graph._nodes):
            raise ValueError("state graph progress states must reference nodes")
        if any(not value for value in transition_rows):
            raise ValueError("state graph transition IDs must be non-empty")
        if len(transition_rows) != total_edge_visits:
            raise ValueError(
                "state graph transition IDs must match committed edge visits"
            )
        graph._transition_ids = set(transition_rows)
        graph._back_label_distances()
        return graph


def evidence_from_transition(
    transition: Transition,
    *,
    uncertainty: float,
    events: Sequence[WorldEvent] = (),
    object_signature: str = "",
    effect_signature: str = "",
    source: str = "online",
    model_version: str = "",
    state_id: str | None = None,
    successor_id: str | None = None,
) -> EvidenceRecord:
    negative = bool(
        transition.outcome.failed
        or transition.outcome.hazard > 0.0
        or transition.outcome.boundary == BoundaryKind.DEATH
    )
    scope = EvidenceScope.TASK if negative else EvidenceScope.TRANSFERABLE
    confidence = float(np.clip(1.0 - float(uncertainty), 0.05, 1.0))
    return EvidenceRecord(
        transition_id=transition.transition_id,
        task_id=transition.task_id,
        stage=transition.stage,
        step=transition.step,
        state_id=str(state_id or transition.before.state_id),
        successor_id=str(successor_id or transition.after_state_id),
        action=transition.action.key,
        object_signature=str(object_signature),
        effect_signature=str(effect_signature),
        event_kinds=tuple(event.kind.value for event in events),
        frame_changed=bool(transition.frame_changed),
        progress=float(transition.outcome.progress_delta),
        reward=float(transition.outcome.reward),
        hazard=float(transition.outcome.hazard),
        terminal=bool(transition.outcome.terminated),
        boundary=transition.outcome.boundary.value,
        uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
        confidence=confidence,
        scope=scope,
        source=str(source),
        model_version=str(model_version),
    )


class EvidenceStore:
    """Bounded retained evidence with durable transition-id tombstones."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self._records: list[EvidenceRecord] = []
        self._ids: set[str] = set()
        self._by_state_action: dict[
            tuple[str, tuple[int, int, int]], list[int]
        ] = defaultdict(list)
        self._by_task_action: dict[
            tuple[str, tuple[int, int, int]], list[int]
        ] = defaultdict(list)
        self._by_action: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    def __deepcopy__(self, memo: dict[int, object]) -> "EvidenceStore":
        """Stage mutable indexes without duplicating frozen evidence rows."""

        duplicate = type(self)(self.config)
        memo[id(self)] = duplicate
        duplicate._records = list(self._records)
        duplicate._ids = set(self._ids)
        duplicate._by_state_action = defaultdict(
            list,
            {key: list(indexes) for key, indexes in self._by_state_action.items()},
        )
        duplicate._by_task_action = defaultdict(
            list,
            {key: list(indexes) for key, indexes in self._by_task_action.items()},
        )
        duplicate._by_action = defaultdict(
            list,
            {key: list(indexes) for key, indexes in self._by_action.items()},
        )
        return duplicate

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records)

    def append(self, record: EvidenceRecord) -> None:
        if record.transition_id in self._ids:
            raise ValueError(f"duplicate evidence transition: {record.transition_id}")
        self._records.append(record)
        self._ids.add(record.transition_id)
        index = len(self._records) - 1
        self._by_state_action[(record.state_id, record.action)].append(index)
        self._by_task_action[(record.task_id, record.action)].append(index)
        self._by_action[record.action].append(index)
        if len(self._records) > int(self.config.max_events):
            self._records = self._records[-int(self.config.max_events) :]
            self._rebuild_indexes()

    def append_transition(
        self,
        transition: Transition,
        *,
        uncertainty: float,
        events: Sequence[WorldEvent] = (),
        object_signature: str = "",
        effect_signature: str = "",
        source: str = "online",
        model_version: str = "",
    ) -> EvidenceRecord:
        record = evidence_from_transition(
            transition,
            uncertainty=uncertainty,
            events=events,
            object_signature=object_signature,
            effect_signature=effect_signature,
            source=source,
            model_version=model_version,
        )
        self.append(record)
        return record

    def _rebuild_indexes(self) -> None:
        self._by_state_action.clear()
        self._by_task_action.clear()
        self._by_action.clear()
        for index, record in enumerate(self._records):
            self._ids.add(record.transition_id)
            self._by_state_action[(record.state_id, record.action)].append(index)
            self._by_task_action[(record.task_id, record.action)].append(index)
            self._by_action[record.action].append(index)

    def score(
        self,
        *,
        task_id: str,
        stage: int,
        state_id: str,
        action: Action,
        object_signature: str = "",
        effect_signature: str = "",
    ) -> EvidenceScore:
        key = action.key
        candidate_indexes: list[int] = []
        candidate_indexes.extend(self._by_state_action.get((state_id, key), ()))
        candidate_indexes.extend(self._by_task_action.get((str(task_id), key), ()))
        candidate_indexes.extend(self._by_action.get(key, ()))
        seen: set[int] = set()
        positive = 0.0
        negative = 0.0
        no_change = 0.0
        positive_support = 0
        negative_support = 0
        task_positive_matched = False
        matched: list[str] = []

        for index in reversed(candidate_indexes):
            if index in seen:
                continue
            seen.add(index)
            record = self._records[index]
            same_state = record.state_id == state_id
            same_task = record.task_id == str(task_id)
            same_stage = same_task and record.stage == int(stage)
            object_sim = _sig_similarity(record.object_signature, object_signature)
            effect_sim = _sig_similarity(record.effect_signature, effect_signature)

            if record.negative:
                # Safety evidence is never transferred to an unrelated task by
                # default.  Exact-state evidence is strongest, then same-stage
                # object-context evidence.
                if not same_task and not self.config.negative_transfer:
                    continue
                if same_state:
                    match = 1.0
                elif same_stage and object_sim >= self.config.evidence_similarity_floor:
                    match = 0.70 * object_sim
                elif same_task and object_sim >= self.config.evidence_similarity_floor:
                    match = 0.45 * object_sim
                elif (
                    not same_task
                    and self.config.negative_transfer
                    and max(object_sim, effect_sim)
                    >= self.config.evidence_similarity_floor
                ):
                    # Cross-task safety transfer is opt-in and still requires
                    # an explicit object/effect match.  Merely sharing an
                    # action index is never enough.
                    match = 0.35 * max(object_sim, effect_sim)
                else:
                    continue
                severity = max(
                    float(record.hazard),
                    1.0 if record.boundary == BoundaryKind.DEATH.value else 0.0,
                    0.6 if record.adverse_terminal else 0.0,
                )
                negative += match * record.confidence * severity
                negative_support += 1
                matched.append(record.transition_id)
                continue

            if record.positive:
                if same_state:
                    match = 1.0
                elif same_task and max(object_sim, effect_sim) >= self.config.evidence_similarity_floor:
                    match = 0.65 * max(object_sim, effect_sim)
                elif (
                    effect_sim >= self.config.evidence_similarity_floor
                    and record.scope == EvidenceScope.TRANSFERABLE
                ):
                    match = 0.35 * effect_sim
                else:
                    continue
                gain = float(np.tanh(max(0.0, record.progress + record.reward)))
                positive += match * record.confidence * gain
                positive_support += 1
                if same_state or same_task:
                    task_positive_matched = True
                matched.append(record.transition_id)
                continue

            if same_state and not record.frame_changed:
                no_change += record.confidence
                matched.append(record.transition_id)

        # Cross-task positive transfer requires repeated effect support; only
        # a positive record that actually matched exempts the current task.
        if (
            positive_support < int(self.config.positive_transfer_min_support)
            and not task_positive_matched
        ):
            positive = 0.0

        # Saturating aggregation keeps many near-duplicates from producing
        # unbounded score pressure.  Counterevidence attenuates but cannot erase
        # repeated terminal evidence.
        pos_term = float(np.tanh(positive))
        neg_term = float(np.tanh(negative))
        noop_term = float(np.tanh(no_change))
        total = pos_term - neg_term - 0.35 * noop_term
        return EvidenceScore(
            total=float(np.clip(total, -1.0, 1.0)),
            positive=pos_term,
            negative=neg_term,
            no_change=noop_term,
            support=positive_support + negative_support,
            negative_support=negative_support,
            positive_support=positive_support,
            matched_transition_ids=tuple(dict.fromkeys(matched[:32])),
        )

    def recent(self, limit: int = 128, *, task_id: str | None = None) -> tuple[EvidenceRecord, ...]:
        rows: Iterable[EvidenceRecord] = reversed(self._records)
        if task_id is not None:
            rows = (row for row in rows if row.task_id == str(task_id))
        result: list[EvidenceRecord] = []
        for row in rows:
            result.append(row)
            if len(result) >= max(0, int(limit)):
                break
        return tuple(reversed(result))

    def infer_action(
        self,
        effect_signature: str,
        *,
        task_id: str | None = None,
    ) -> InverseActionPrediction:
        """Diagnostic inverse-effect model derived from the canonical ledger."""

        weights: dict[tuple[int, int, int], float] = defaultdict(float)
        support = 0
        for record in self._records:
            if task_id is not None and record.task_id != str(task_id):
                continue
            similarity = _sig_similarity(record.effect_signature, effect_signature)
            if similarity < self.config.evidence_similarity_floor:
                continue
            weights[record.action] += similarity * record.confidence
            support += 1
        if not weights:
            return InverseActionPrediction(
                action=None,
                confidence=0.0,
                support=0,
            )
        total = sum(weights.values())
        ordered = sorted(
            (
                (action, float(weight / max(total, 1e-9)))
                for action, weight in weights.items()
            ),
            key=lambda row: (-row[1], row[0]),
        )
        return InverseActionPrediction(
            action=ordered[0][0],
            confidence=ordered[0][1],
            support=support,
            distribution=tuple(ordered),
        )

    def export_state(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            # Retained rows are bounded, but duplicate protection covers the
            # complete lifetime of the store, including evicted transitions.
            "seen_transition_ids": sorted(self._ids),
            "records": [
                {
                    **asdict(record),
                    "scope": record.scope.value,
                }
                for record in self._records
            ],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "EvidenceStore":
        raw_config = state.get("config", {})
        if not isinstance(raw_config, Mapping):
            raise ValueError("evidence configuration must be a mapping")
        config = MemoryConfig(**dict(raw_config))
        store = cls(config)
        raw_records = state.get("records", ())
        if isinstance(raw_records, (str, bytes)) or not isinstance(
            raw_records,
            Sequence,
        ):
            raise ValueError("evidence records must be a sequence")
        if len(raw_records) > int(config.max_events):
            raise ValueError("evidence records exceed configured capacity")
        for row in raw_records:
            if not isinstance(row, Mapping):
                raise ValueError("evidence record rows must be mappings")
            payload = dict(row)
            payload["action"] = _action_key(payload["action"])
            raw_events = payload.get("event_kinds", ())
            if isinstance(raw_events, (str, bytes)) or not isinstance(
                raw_events,
                Sequence,
            ):
                raise ValueError("evidence event kinds must be a sequence")
            payload["event_kinds"] = tuple(str(v) for v in raw_events)
            payload["scope"] = EvidenceScope(payload.get("scope", EvidenceScope.TASK.value))
            store.append(EvidenceRecord(**payload))
        raw_seen = state.get("seen_transition_ids")
        if raw_seen is None:
            # Legacy checkpoints predate durable tombstones.
            seen_rows = tuple(record.transition_id for record in store._records)
        else:
            if isinstance(raw_seen, (str, bytes)) or not isinstance(
                raw_seen,
                Sequence,
            ):
                raise ValueError("seen evidence transition IDs must be a sequence")
            seen_rows = tuple(str(value) for value in raw_seen)
            if len(set(seen_rows)) != len(seen_rows) or any(
                not value for value in seen_rows
            ):
                raise ValueError(
                    "seen evidence transition IDs must be unique and non-empty"
                )
            retained_ids = {
                record.transition_id for record in store._records
            }
            if not retained_ids.issubset(seen_rows):
                raise ValueError(
                    "seen evidence transition IDs must include retained records"
                )
        store._ids.update(seen_rows)
        return store


def object_summary(objects: Sequence[Any]) -> np.ndarray:
    """Fixed small object aggregate used by exact and learned dynamics."""

    if not objects:
        return np.zeros(8, dtype=np.float32)
    areas = np.asarray([float(getattr(obj, "area", 0.0)) for obj in objects])
    hazards = np.asarray([float(getattr(obj, "hazard", 0.0)) for obj in objects])
    rewards = np.asarray([float(getattr(obj, "rewarding", 0.0)) for obj in objects])
    controls = np.asarray([float(getattr(obj, "controllable", 0.0)) for obj in objects])
    moving = np.asarray(
        [
            float(
                np.hypot(
                    float(getattr(obj, "velocity_x", 0.0)),
                    float(getattr(obj, "velocity_y", 0.0)),
                )
                > 0.25
            )
            for obj in objects
        ]
    )
    return np.asarray(
        [
            min(len(objects) / 32.0, 1.0),
            float(np.mean(areas) / max(float(np.max(areas)), 1.0)),
            float(np.std(areas) / max(float(np.max(areas)), 1.0)),
            float(np.mean(hazards)),
            float(np.mean(rewards)),
            float(np.mean(controls)),
            float(np.mean(moving)),
            float(np.max(areas) / max(float(np.sum(areas)), 1.0)),
        ],
        dtype=np.float32,
    )


def combined_object_signature(objects: Sequence[Any], action: Action | None = None) -> str:
    signatures = sorted(
        str(getattr(obj, "signature", ""))
        for obj in objects
        if str(getattr(obj, "signature", ""))
    )
    if action is not None and action.has_position and objects:
        nearest = min(
            objects,
            key=lambda obj: (
                float(getattr(obj, "centroid_x", 0.0)) - action.x
            )
            ** 2
            + (
                float(getattr(obj, "centroid_y", 0.0)) - action.y
            )
            ** 2,
        )
        target = str(getattr(nearest, "signature", ""))
        return f"target:{target}|scene:{','.join(signatures[:12])}"
    return f"scene:{','.join(signatures[:12])}"


__all__ = [
    "EvidenceRecord",
    "EvidenceScore",
    "EvidenceStore",
    "GraphEdgeView",
    "InverseActionPrediction",
    "StateGraph",
    "combined_object_signature",
    "evidence_from_transition",
    "object_summary",
]
