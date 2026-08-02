"""Immutable contracts for the compact Hunter-Seeker runtime.

The module intentionally has no dependency on the legacy Hunter-Seeker package.
Environment-specific objects are converted to :class:`Observation` and
:class:`Outcome` at the adapter boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import blake2b
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


SCHEMA_VERSION = 1


class RuntimeMode(str, Enum):
    AUTONOMOUS = "autonomous"
    STUDENT = "student"
    COMPAT_ASSISTED = "compat_assisted"


class BoundaryKind(str, Enum):
    NONE = "none"
    LEVEL_COMPLETED = "level_completed"
    GAME_COMPLETED = "game_completed"
    DEATH = "death"
    TIME_LIMIT = "time_limit"
    INTERRUPTED = "interrupted"
    ENVIRONMENT_ERROR = "environment_error"


class EventKind(str, Enum):
    MOVED = "moved"
    APPEARED = "appeared"
    DISAPPEARED = "disappeared"
    TRANSFORMED = "transformed"
    CONTACT = "contact"
    FRAME_CHANGED = "frame_changed"
    NO_CHANGE = "no_change"
    PROGRESS = "progress"
    HAZARD = "hazard"
    TERMINAL = "terminal"


class EvidenceScope(str, Enum):
    TASK = "task"
    STAGE = "stage"
    TRANSFERABLE = "transferable"


class TapRole(str, Enum):
    VALUE = "value"
    PROGRESS = "progress"
    HAZARD = "hazard"
    UNCERTAINTY = "uncertainty"
    PAIRWISE = "pairwise"
    SURVIVAL = "survival"
    DIAGNOSTIC = "diagnostic"


def readonly_array(value: np.ndarray | Sequence[Any], *, dtype: Any | None = None) -> np.ndarray:
    """Return a contiguous, immutable array copy.

    Runtime records must not retain mutable references owned by an environment
    or speculative planner.
    """

    arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if arr.dtype.hasobject:
        # Object arrays retain references to arbitrarily mutable Python
        # values, and their raw bytes contain process-local pointer values.
        # They therefore cannot satisfy either deep immutability or stable
        # content identity.
        raise ValueError("object arrays cannot be made deeply immutable")
    # ``setflags(write=False)`` on an owning ndarray is reversible by callers.
    # Rebuild over an immutable ``bytes`` owner so even ``setflags(write=True)``
    # is rejected and pending transactional records cannot be altered later.
    frozen = np.frombuffer(arr.tobytes(order="C"), dtype=arr.dtype).reshape(arr.shape)
    frozen.setflags(write=False)
    return frozen


def _frozen_value(value: Any) -> Any:
    """Recursively detach metadata from caller-owned mutable containers."""

    if isinstance(value, np.ndarray):
        return readonly_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _frozen_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_frozen_value(item) for item in value)
    return value


def frozen_mapping(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Recursively immutable metadata mapping with string keys."""

    if not value:
        return MappingProxyType({})
    return MappingProxyType(
        {str(key): _frozen_value(item) for key, item in value.items()}
    )


def _finite(value: Any, *, fallback: float = 0.0) -> float:
    result = float(value)
    return result if np.isfinite(result) else float(fallback)


def stable_frame_hash(
    frame: np.ndarray,
    *,
    task_id: str = "",
    stage: int = 0,
    available_actions: Sequence[int] | None = None,
    progress: float | None = None,
) -> str:
    arr = np.ascontiguousarray(np.asarray(frame))
    if arr.dtype.hasobject:
        raise ValueError("state identity does not support object arrays")
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        # Identity is content-based: equal integer grids hash equally no
        # matter which storage width recorded them.
        arr = np.ascontiguousarray(arr.astype(np.int64, copy=False))
    digest = blake2b(digest_size=16)
    digest.update(str(task_id).encode("utf-8"))
    digest.update(int(stage).to_bytes(8, "little", signed=True))
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    if available_actions is not None:
        actions = np.asarray(
            sorted(dict.fromkeys(int(value) for value in available_actions)),
            dtype=np.int64,
        )
        digest.update(b"|actions|")
        digest.update(actions.tobytes())
    if progress is not None:
        finite_progress = float(progress)
        if not np.isfinite(finite_progress):
            raise ValueError("state identity progress must be finite")
        digest.update(b"|progress|")
        digest.update(np.asarray([finite_progress], dtype="<f8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Action:
    index: int
    x: int = -1
    y: int = -1
    name: str = ""

    def __post_init__(self) -> None:
        _require_integer("action index", self.index, minimum=0)
        _require_integer("action x", self.x, minimum=-1)
        _require_integer("action y", self.y, minimum=-1)
        index = int(self.index)
        x = int(self.x)
        y = int(self.y)
        if (x == -1) != (y == -1):
            raise ValueError(
                "action coordinates must either both be -1 or both be non-negative"
            )
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "name", str(self.name))

    @property
    def has_position(self) -> bool:
        return self.x >= 0 and self.y >= 0

    @property
    def key(self) -> tuple[int, int, int]:
        return int(self.index), int(self.x), int(self.y)


@dataclass(frozen=True, slots=True)
class Observation:
    frame: np.ndarray
    available_actions: tuple[int, ...]
    task_id: str
    stage: int = 1
    progress: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", readonly_array(self.frame))
        actions_list: list[int] = []
        for action in self.available_actions:
            _require_integer("available action", action, minimum=0)
            actions_list.append(int(action))
        actions = tuple(dict.fromkeys(actions_list))
        object.__setattr__(self, "available_actions", actions)
        object.__setattr__(self, "task_id", str(self.task_id))
        _require_integer("observation stage", self.stage, minimum=1)
        object.__setattr__(self, "stage", int(self.stage))
        _require_number("observation progress", self.progress)
        progress = float(self.progress)
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))

    @property
    def state_id(self) -> str:
        return stable_frame_hash(
            self.frame,
            task_id=self.task_id,
            stage=self.stage,
            available_actions=self.available_actions,
            progress=self.progress,
        )


@dataclass(frozen=True, slots=True)
class Outcome:
    reward: float = 0.0
    progress_delta: float = 0.0
    terminated: bool = False
    truncated: bool = False
    boundary: BoundaryKind = BoundaryKind.NONE
    hazard: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        _require_number("outcome reward", self.reward)
        _require_number("outcome progress_delta", self.progress_delta)
        _require_number(
            "outcome hazard",
            self.hazard,
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "progress_delta", float(self.progress_delta))
        boundary = BoundaryKind(self.boundary)
        _require_bool("outcome terminated", self.terminated)
        _require_bool("outcome truncated", self.truncated)
        terminated = bool(self.terminated)
        truncated = bool(self.truncated)
        if boundary in {
            BoundaryKind.GAME_COMPLETED,
            BoundaryKind.DEATH,
            BoundaryKind.ENVIRONMENT_ERROR,
        }:
            terminated = True
            truncated = False
        elif boundary in {BoundaryKind.TIME_LIMIT, BoundaryKind.INTERRUPTED}:
            terminated = False
            truncated = True
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "truncated", truncated)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "hazard", float(self.hazard))
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))

    @property
    def completed(self) -> bool:
        return self.boundary in {
            BoundaryKind.LEVEL_COMPLETED,
            BoundaryKind.GAME_COMPLETED,
        }

    @property
    def failed(self) -> bool:
        return self.boundary in {
            BoundaryKind.DEATH,
            BoundaryKind.ENVIRONMENT_ERROR,
        }


@dataclass(frozen=True, slots=True)
class ObjectState:
    object_id: int
    track_id: int
    value: int
    area: int
    centroid_x: float
    centroid_y: float
    bbox: tuple[int, int, int, int]
    touches_border: bool = False
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    controllable: float = 0.0
    hazard: float = 0.0
    rewarding: float = 0.0
    confidence: float = 1.0
    signature: str = ""

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("object_id", self.object_id, 0),
            ("track_id", self.track_id, 0),
            ("value", self.value, None),
            ("area", self.area, 1),
        ):
            _require_integer(name, value, minimum=minimum)
            object.__setattr__(self, name, int(value))

        try:
            bbox = tuple(self.bbox)
        except TypeError as exc:
            raise ValueError("bbox must contain four integers") from exc
        if len(bbox) != 4:
            raise ValueError("bbox must contain four integers")
        for index, value in enumerate(bbox):
            _require_integer(f"bbox[{index}]", value, minimum=0)
        x0, y0, x1, y1 = (int(value) for value in bbox)
        if x0 > x1 or y0 > y1:
            raise ValueError("bbox bounds must be ordered")
        if int(self.area) > (x1 - x0 + 1) * (y1 - y0 + 1):
            raise ValueError("object area cannot exceed its bounding-box area")
        object.__setattr__(self, "bbox", (x0, y0, x1, y1))

        for name in ("centroid_x", "centroid_y", "velocity_x", "velocity_y"):
            value = getattr(self, name)
            _require_number(name, value)
            object.__setattr__(self, name, float(value))
        if not x0 <= float(self.centroid_x) <= x1:
            raise ValueError("centroid_x must lie within bbox")
        if not y0 <= float(self.centroid_y) <= y1:
            raise ValueError("centroid_y must lie within bbox")

        for name in ("controllable", "hazard", "rewarding", "confidence"):
            value = getattr(self, name)
            _require_number(name, value, minimum=0.0, maximum=1.0)
            object.__setattr__(self, name, float(value))
        _require_bool("touches_border", self.touches_border)
        object.__setattr__(self, "touches_border", bool(self.touches_border))
        object.__setattr__(self, "signature", str(self.signature))


@dataclass(frozen=True, slots=True)
class WorldEvent:
    kind: EventKind
    subject_track_id: int = -1
    object_signature: str = ""
    magnitude: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EventKind(self.kind))
        _require_integer(
            "event subject_track_id",
            self.subject_track_id,
            minimum=-1,
        )
        _require_number("event magnitude", self.magnitude, minimum=0.0)
        object.__setattr__(self, "subject_track_id", int(self.subject_track_id))
        object.__setattr__(self, "object_signature", str(self.object_signature))
        object.__setattr__(self, "magnitude", float(self.magnitude))
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Topology:
    component_count: int = 0
    largest_component_fraction: float = 0.0
    frontier_fraction: float = 0.0
    object_adjacencies: tuple[tuple[int, int], ...] = ()
    reachable_object_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_integer("component_count", self.component_count, minimum=0)
        _require_number(
            "largest_component_fraction",
            self.largest_component_fraction,
            minimum=0.0,
            maximum=1.0,
        )
        _require_number(
            "frontier_fraction",
            self.frontier_fraction,
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, "component_count", int(self.component_count))
        object.__setattr__(
            self,
            "largest_component_fraction",
            float(self.largest_component_fraction),
        )
        object.__setattr__(self, "frontier_fraction", float(self.frontier_fraction))

        adjacencies: list[tuple[int, int]] = []
        for index, raw_pair in enumerate(self.object_adjacencies):
            try:
                pair = tuple(raw_pair)
            except TypeError as exc:
                raise ValueError(
                    f"object_adjacencies[{index}] must contain two object IDs"
                ) from exc
            if len(pair) != 2:
                raise ValueError(
                    f"object_adjacencies[{index}] must contain two object IDs"
                )
            for side, value in enumerate(pair):
                _require_integer(
                    f"object_adjacencies[{index}][{side}]",
                    value,
                    minimum=0,
                )
            left, right = int(pair[0]), int(pair[1])
            if left == right:
                raise ValueError("an object cannot be adjacent to itself")
            adjacencies.append(tuple(sorted((left, right))))
        if len(adjacencies) != len(set(adjacencies)):
            raise ValueError("object adjacencies must be unique")
        object.__setattr__(self, "object_adjacencies", tuple(adjacencies))

        reachable: list[int] = []
        for index, value in enumerate(self.reachable_object_ids):
            _require_integer(
                f"reachable_object_ids[{index}]",
                value,
                minimum=0,
            )
            reachable.append(int(value))
        if len(reachable) != len(set(reachable)):
            raise ValueError("reachable object IDs must be unique")
        object.__setattr__(self, "reachable_object_ids", tuple(reachable))


@dataclass(frozen=True, slots=True)
class Representation:
    global_vector: np.ndarray
    spatial: np.ndarray
    taps: Mapping[str, np.ndarray] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "global_vector",
            readonly_array(
                np.nan_to_num(
                    np.asarray(self.global_vector, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                dtype=np.float32,
            ),
        )
        object.__setattr__(
            self,
            "spatial",
            readonly_array(
                np.nan_to_num(
                    np.asarray(self.spatial, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                dtype=np.float32,
            ),
        )
        tap_copy = {
            str(name): readonly_array(
                np.nan_to_num(
                    np.asarray(value, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                dtype=np.float32,
            )
            for name, value in self.taps.items()
        }
        object.__setattr__(self, "taps", frozen_mapping(tap_copy))


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    observation: Observation
    objects: tuple[ObjectState, ...]
    events: tuple[WorldEvent, ...]
    topology: Topology
    representation: Representation
    step: int
    memory_state_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Observation):
            raise TypeError("snapshot observation must be an Observation")
        if not isinstance(self.topology, Topology):
            raise TypeError("snapshot topology must be a Topology")
        if not isinstance(self.representation, Representation):
            raise TypeError("snapshot representation must be a Representation")
        objects = tuple(self.objects)
        events = tuple(self.events)
        if any(not isinstance(obj, ObjectState) for obj in objects):
            raise TypeError("snapshot objects must contain only ObjectState values")
        if any(not isinstance(event, WorldEvent) for event in events):
            raise TypeError("snapshot events must contain only WorldEvent values")
        _require_integer("snapshot step", self.step, minimum=0)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "memory_state_id", str(self.memory_state_id))

    @property
    def state_id(self) -> str:
        return self.observation.state_id

    @property
    def memory_id(self) -> str:
        """Identity used by durable memory; exact identity when unfiltered."""

        return self.memory_state_id or self.state_id


@dataclass(frozen=True, slots=True)
class Prediction:
    change_probability: float
    progress: float
    value: float
    hazard: float
    terminal: float
    uncertainty: float
    latent_delta: np.ndarray
    object_delta: np.ndarray
    exact_successor_id: str | None = None
    source: str = "model"

    def __post_init__(self) -> None:
        fallbacks = {
            "change_probability": 0.0,
            "hazard": 1.0,
            "terminal": 1.0,
            "uncertainty": 1.0,
        }
        for name in ("change_probability", "hazard", "terminal", "uncertainty"):
            object.__setattr__(
                self,
                name,
                float(
                    np.clip(
                        _finite(getattr(self, name), fallback=fallbacks[name]),
                        0.0,
                        1.0,
                    )
                ),
            )
        object.__setattr__(self, "progress", _finite(self.progress))
        object.__setattr__(self, "value", _finite(self.value))
        object.__setattr__(
            self,
            "latent_delta",
            readonly_array(
                np.nan_to_num(
                    np.asarray(self.latent_delta, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                dtype=np.float32,
            ),
        )
        object.__setattr__(
            self,
            "object_delta",
            readonly_array(
                np.nan_to_num(
                    np.asarray(self.object_delta, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                dtype=np.float32,
            ),
        )
        object.__setattr__(
            self,
            "exact_successor_id",
            (
                None
                if self.exact_successor_id is None
                else str(self.exact_successor_id)
            ),
        )
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True, slots=True)
class ScoreTerm:
    name: str
    value: float
    group: str
    influences_score: bool = True
    source: str = "core"

    def __post_init__(self) -> None:
        value = float(self.value)
        if not np.isfinite(value):
            value = 0.0
        _require_bool("score term influences_score", self.influences_score)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "group", str(self.group))
        object.__setattr__(
            self,
            "influences_score",
            bool(self.influences_score),
        )
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True, slots=True)
class Candidate:
    action: Action
    prediction: Prediction
    terms: tuple[ScoreTerm, ...] = ()
    depth: int = 0
    path: tuple[Action, ...] = ()
    source: str = "proposal"

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise TypeError("candidate action must be an Action")
        if not isinstance(self.prediction, Prediction):
            raise TypeError("candidate prediction must be a Prediction")
        terms = tuple(self.terms)
        path = tuple(self.path)
        if any(not isinstance(term, ScoreTerm) for term in terms):
            raise TypeError("candidate terms must contain only ScoreTerm values")
        if any(not isinstance(action, Action) for action in path):
            raise TypeError("candidate path must contain only Action values")
        _require_integer("candidate depth", self.depth, minimum=0)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "depth", int(self.depth))
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "source", str(self.source))

    @property
    def score(self) -> float:
        return float(sum(term.value for term in self.terms if term.influences_score))

    def with_terms(self, terms: Sequence[ScoreTerm]) -> "Candidate":
        return Candidate(
            action=self.action,
            prediction=self.prediction,
            terms=tuple(terms),
            depth=self.depth,
            path=self.path,
            source=self.source,
        )


@dataclass(frozen=True, slots=True)
class Decision:
    decision_id: str
    agent_id: str
    snapshot_id: str
    action: Action
    score: float
    candidates: tuple[Candidate, ...]
    mode: RuntimeMode
    step: int
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise TypeError("decision action must be an Action")
        candidates = tuple(self.candidates)
        if any(not isinstance(candidate, Candidate) for candidate in candidates):
            raise TypeError("decision candidates must contain only Candidate values")
        _require_number("decision score", self.score)
        _require_integer("decision step", self.step, minimum=0)
        object.__setattr__(self, "decision_id", str(self.decision_id))
        object.__setattr__(self, "agent_id", str(self.agent_id))
        object.__setattr__(self, "snapshot_id", str(self.snapshot_id))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "mode", RuntimeMode(self.mode))
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Transition:
    transition_id: str
    decision_id: str
    task_id: str
    stage: int
    step: int
    before: WorldSnapshot
    action: Action
    after_observation: Observation
    outcome: Outcome
    frame_changed: bool
    after_state_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.before, WorldSnapshot):
            raise TypeError("transition before must be a WorldSnapshot")
        if not isinstance(self.action, Action):
            raise TypeError("transition action must be an Action")
        if not isinstance(self.after_observation, Observation):
            raise TypeError("transition after_observation must be an Observation")
        if not isinstance(self.outcome, Outcome):
            raise TypeError("transition outcome must be an Outcome")
        _require_integer("transition stage", self.stage, minimum=1)
        _require_integer("transition step", self.step, minimum=0)
        _require_bool("transition frame_changed", self.frame_changed)
        task_id = str(self.task_id)
        after_state_id = str(self.after_state_id)
        if task_id != self.before.observation.task_id:
            raise ValueError("transition task must match its before snapshot")
        if task_id != self.after_observation.task_id:
            raise ValueError("transition task must match its after observation")
        if int(self.stage) != self.before.observation.stage:
            raise ValueError("transition stage must match its before snapshot")
        if int(self.step) != self.before.step:
            raise ValueError("transition step must match its before snapshot")
        if after_state_id != self.after_observation.state_id:
            raise ValueError(
                "transition after_state_id must match its after observation"
            )
        actual_frame_changed = not np.array_equal(
            self.before.observation.frame,
            self.after_observation.frame,
        )
        if bool(self.frame_changed) != actual_frame_changed:
            raise ValueError(
                "transition frame_changed must match its observation frames"
            )
        object.__setattr__(self, "transition_id", str(self.transition_id))
        object.__setattr__(self, "decision_id", str(self.decision_id))
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "stage", int(self.stage))
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "frame_changed", bool(self.frame_changed))
        object.__setattr__(self, "after_state_id", after_state_id)


@dataclass(frozen=True, slots=True)
class CompetenceState:
    dynamics_error_ema: float = 0.0
    hazard_calibration_error: float = 0.0
    predicted_success: float = 0.0
    recent_realized_progress: float = 0.0
    model_disagreement: float = 1.0
    stagnation_count: int = 0
    expected_learning_gain: float = 1.0
    remaining_risk_budget: float = 1.0
    time_pressure: float = 0.0


def _require_integer(
    name: str,
    value: int,
    *,
    minimum: int | None = 0,
) -> None:
    invalid = isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    )
    if not invalid and minimum is not None:
        invalid = int(value) < minimum
    if invalid:
        suffix = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{name} must be an integer{suffix}")


def _require_number(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    beam_width: int = 6
    horizon: int = 2
    max_click_candidates: int = 24
    transposition_limit: int = 4096
    exact_graph_bonus: float = 0.08
    no_change_penalty: float = 0.18
    loop_penalty: float = 0.15
    graph_goal_search_enabled: bool = True
    graph_goal_expansion_limit: int = 512
    graph_goal_depth_limit: int = 64
    graph_goal_bonus_bound: float = 1.0
    graph_goal_heuristic_weight: float = 1.0
    graph_goal_step_cost: float = 0.02
    graph_goal_improvement_epsilon: float = 0.02
    graph_goal_plateau_tolerance: float = 0.03
    graph_goal_min_edge_visits: int = 1
    graph_goal_min_dominant_fraction: float = 0.60
    graph_goal_max_edge_risk: float = 0.70

    def __post_init__(self) -> None:
        _require_integer("beam_width", self.beam_width, minimum=1)
        _require_integer("horizon", self.horizon, minimum=1)
        _require_integer("max_click_candidates", self.max_click_candidates, minimum=1)
        _require_integer("transposition_limit", self.transposition_limit, minimum=1)
        _require_number("exact_graph_bonus", self.exact_graph_bonus, minimum=0.0)
        _require_number("no_change_penalty", self.no_change_penalty, minimum=0.0)
        _require_number("loop_penalty", self.loop_penalty, minimum=0.0)
        _require_bool(
            "graph_goal_search_enabled",
            self.graph_goal_search_enabled,
        )
        _require_integer(
            "graph_goal_expansion_limit",
            self.graph_goal_expansion_limit,
            minimum=1,
        )
        _require_integer(
            "graph_goal_depth_limit",
            self.graph_goal_depth_limit,
            minimum=1,
        )
        _require_number(
            "graph_goal_bonus_bound",
            self.graph_goal_bonus_bound,
            minimum=0.0,
            maximum=4.0,
        )
        _require_number(
            "graph_goal_heuristic_weight",
            self.graph_goal_heuristic_weight,
            minimum=0.0,
        )
        _require_number(
            "graph_goal_step_cost",
            self.graph_goal_step_cost,
            minimum=1e-12,
        )
        _require_number(
            "graph_goal_improvement_epsilon",
            self.graph_goal_improvement_epsilon,
            minimum=0.0,
            maximum=1.0,
        )
        _require_number(
            "graph_goal_plateau_tolerance",
            self.graph_goal_plateau_tolerance,
            minimum=0.0,
            maximum=1.0,
        )
        _require_integer(
            "graph_goal_min_edge_visits",
            self.graph_goal_min_edge_visits,
            minimum=1,
        )
        _require_number(
            "graph_goal_min_dominant_fraction",
            self.graph_goal_min_dominant_fraction,
            minimum=0.0,
            maximum=1.0,
        )
        _require_number(
            "graph_goal_max_edge_risk",
            self.graph_goal_max_edge_risk,
            minimum=0.0,
            maximum=2.0,
        )


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    progress_weight: float = 1.0
    value_weight: float = 0.35
    hazard_weight: float = 1.25
    exploration_weight: float = 0.30
    learning_progress_weight: float = 0.20
    memory_weight: float = 0.50
    action_cost: float = 0.01
    imagined_uncertainty_weight: float = 0.45
    risk_limit: float = 0.70
    exploration_epsilon: float = 0.05
    ego_hazard_weight: float = 0.35
    hypothesis_weight: float = 0.40

    def __post_init__(self) -> None:
        for name in (
            "progress_weight",
            "value_weight",
            "hazard_weight",
            "exploration_weight",
            "learning_progress_weight",
            "memory_weight",
            "action_cost",
            "imagined_uncertainty_weight",
            "ego_hazard_weight",
            "hypothesis_weight",
        ):
            _require_number(name, getattr(self, name), minimum=0.0)
        _require_number("risk_limit", self.risk_limit, minimum=0.0, maximum=2.0)
        _require_number(
            "exploration_epsilon",
            self.exploration_epsilon,
            minimum=0.0,
            maximum=1.0,
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    ensemble_size: int = 5
    learning_rate: float = 0.04
    l2: float = 1e-4
    bootstrap_probability: float = 0.75
    latent_dim: int = 32

    def __post_init__(self) -> None:
        _require_integer("ensemble_size", self.ensemble_size, minimum=1)
        _require_number("learning_rate", self.learning_rate, minimum=1e-15)
        _require_number("l2", self.l2, minimum=0.0)
        _require_number(
            "bootstrap_probability",
            self.bootstrap_probability,
            minimum=0.0,
            maximum=1.0,
        )
        _require_integer("latent_dim", self.latent_dim, minimum=1)


@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    background_value: int | None = None
    min_object_area: int = 1
    track_match_distance: float = 8.0
    track_miss_tolerance: int = 3

    def __post_init__(self) -> None:
        if self.background_value is not None:
            _require_integer("background_value", self.background_value, minimum=None)
        _require_integer("min_object_area", self.min_object_area, minimum=1)
        _require_number("track_match_distance", self.track_match_distance, minimum=0.0)
        _require_integer("track_miss_tolerance", self.track_miss_tolerance)


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    max_events: int = 50_000
    negative_transfer: bool = False
    positive_transfer_min_support: int = 2
    evidence_similarity_floor: float = 0.55

    def __post_init__(self) -> None:
        _require_integer("max_events", self.max_events, minimum=1)
        _require_bool("negative_transfer", self.negative_transfer)
        _require_integer(
            "positive_transfer_min_support",
            self.positive_transfer_min_support,
            minimum=1,
        )
        _require_number(
            "evidence_similarity_floor",
            self.evidence_similarity_floor,
            minimum=0.0,
            maximum=1.0,
        )


@dataclass(frozen=True, slots=True)
class ExogenousConfig:
    """Exogenous observation-change filtering for memory identity.

    A cell is exogenous when its change-event trajectory replays identically
    across episodes whose action histories differ — an intervention-grounded
    independence test.  Masked cells are excluded from the *memory* state
    identity only; transactional identity stays exact.  ``enabled=False``
    must be an exact behavioral no-op.
    """

    enabled: bool = True
    min_confirmations: int = 1
    min_common_horizon: int = 8
    max_events_per_cell: int = 512

    def __post_init__(self) -> None:
        _require_bool("enabled", self.enabled)
        _require_integer("min_confirmations", self.min_confirmations, minimum=1)
        _require_integer("min_common_horizon", self.min_common_horizon, minimum=1)
        _require_integer("max_events_per_cell", self.max_events_per_cell, minimum=1)


@dataclass(frozen=True, slots=True)
class HypothesisConfig:
    """Relational goal hypotheses as bounded potential functions.

    Hypotheses are proposals, never claims: an unverified hypothesis carries
    only exploration-grade weight.  Verification normally requires an actual
    completion observed while the potential is satisfied; the sole hidden-
    completion fallback is a conservative, direct-overlap reach proof from a
    supported controlled-motion model.  ``enabled=False`` must be an exact
    behavioral no-op.
    """

    enabled: bool = True
    max_hypotheses: int = 6
    min_region_area: int = 9
    max_initial_mismatch: float = 0.6
    promotion_epsilon: float = 0.10
    refutation_floor: float = 0.5
    min_support: int = 2
    unverified_scale: float = 0.35
    verified_scale: float = 1.0
    signal_gain: float = 8.0
    # Ordinary solved-frame goal proposal: at a completion, admit a relation
    # as a genuine goal only if it was violated at the level's initial frame
    # (potential >= goal_contrast_floor) and satisfied at the completion
    # frame (potential <= promotion_epsilon).  This is the ordinary evidence
    # path for object-relational ``reach`` and the only path for count targets;
    # the separately gated completion-motion reach fallback covers immediate
    # stage swaps with no visible solved frame.  ``enable_goal_contrast=False``
    # is an exact no-op.
    enable_goal_contrast: bool = True
    goal_contrast_floor: float = 0.3
    # Maximum foreground cells a value may occupy to participate in a
    # ``reach`` pair; larger sets are background-like and skipped.
    reach_cell_cap: int = 400
    # Count/coverage goals are inferred only from a real initial->completion
    # contrast.  They represent a categorical value reaching the count
    # observed at completion (decrease for consumption, increase for
    # coverage/painting), never a hand-authored game rule.
    enable_count_targets: bool = True
    count_min_cells: int = 2
    count_min_change_fraction: float = 0.15
    max_count_targets: int = 3
    goal_frame_cache_limit: int = 2048
    # Some environments replace the solved board with the next stage in the
    # same observation.  In that case there is no old-stage completion frame
    # to contrast.  A reach goal may still be proven when a well-supported
    # ego motion model projects one controlled object directly onto one
    # unchanged, stationary target.  Other relation kinds are deliberately
    # excluded from this fallback.
    enable_completion_motion_reach: bool = True
    completion_motion_min_consistency: float = 0.75

    def __post_init__(self) -> None:
        _require_bool("enabled", self.enabled)
        _require_integer("max_hypotheses", self.max_hypotheses, minimum=1)
        _require_integer("min_region_area", self.min_region_area, minimum=1)
        _require_number(
            "max_initial_mismatch",
            self.max_initial_mismatch,
            minimum=0.0,
            maximum=1.0,
        )
        _require_number(
            "promotion_epsilon",
            self.promotion_epsilon,
            minimum=0.0,
            maximum=1.0,
        )
        _require_number(
            "refutation_floor",
            self.refutation_floor,
            minimum=self.promotion_epsilon,
            maximum=1.0,
        )
        _require_integer("min_support", self.min_support, minimum=1)
        _require_number("unverified_scale", self.unverified_scale, minimum=0.0)
        _require_number("verified_scale", self.verified_scale, minimum=0.0)
        _require_number("signal_gain", self.signal_gain, minimum=0.0)
        _require_number(
            "goal_contrast_floor",
            self.goal_contrast_floor,
            minimum=self.promotion_epsilon,
            maximum=1.0,
        )
        _require_integer("reach_cell_cap", self.reach_cell_cap, minimum=1)
        _require_bool("enable_goal_contrast", self.enable_goal_contrast)
        _require_bool("enable_count_targets", self.enable_count_targets)
        _require_integer("count_min_cells", self.count_min_cells, minimum=1)
        _require_number(
            "count_min_change_fraction",
            self.count_min_change_fraction,
            minimum=0.0,
            maximum=1.0,
        )
        _require_integer("max_count_targets", self.max_count_targets, minimum=1)
        _require_integer(
            "goal_frame_cache_limit",
            self.goal_frame_cache_limit,
            minimum=1,
        )
        _require_bool(
            "enable_completion_motion_reach",
            self.enable_completion_motion_reach,
        )
        _require_number(
            "completion_motion_min_consistency",
            self.completion_motion_min_consistency,
            minimum=0.0,
            maximum=1.0,
        )


@dataclass(frozen=True, slots=True)
class EgoConfig:
    """Control attribution over tracked objects.

    Influence is the count-based analog of causal action influence: the
    normalized mutual information between the executed action and a track's
    quantized displacement response.  ``enabled=False`` must be an exact
    behavioral no-op.
    """

    enabled: bool = True
    min_support: int = 3
    influence_threshold: float = 0.35
    displacement_epsilon: float = 0.25
    hazard_belief_floor: float = 0.15
    signature_bootstrap_scale: float = 0.5
    max_tracked: int = 512

    def __post_init__(self) -> None:
        _require_bool("enabled", self.enabled)
        _require_integer("min_support", self.min_support, minimum=1)
        _require_number(
            "influence_threshold",
            self.influence_threshold,
            minimum=0.0,
            maximum=1.0,
        )
        _require_number("displacement_epsilon", self.displacement_epsilon, minimum=0.0)
        _require_number(
            "hazard_belief_floor",
            self.hazard_belief_floor,
            minimum=0.0,
            maximum=1.0,
        )
        _require_number(
            "signature_bootstrap_scale",
            self.signature_bootstrap_scale,
            minimum=0.0,
            maximum=1.0,
        )
        _require_integer("max_tracked", self.max_tracked, minimum=1)


@dataclass(frozen=True, slots=True)
class LearningConfig:
    replay_capacity: int = 10_000
    replay_every: int = 10
    replay_batch_size: int = 8
    replay_updates: int = 1
    teacher_fraction: float = 0.25

    def __post_init__(self) -> None:
        _require_integer("replay_capacity", self.replay_capacity, minimum=1)
        _require_integer("replay_every", self.replay_every)
        _require_integer("replay_batch_size", self.replay_batch_size, minimum=1)
        _require_integer("replay_updates", self.replay_updates)
        _require_number(
            "teacher_fraction",
            self.teacher_fraction,
            minimum=0.0,
            maximum=1.0,
        )


@dataclass(frozen=True, slots=True)
class AgentConfig:
    runtime_mode: RuntimeMode = RuntimeMode.AUTONOMOUS
    seed: int = 0
    search: SearchConfig = field(default_factory=SearchConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    ego: EgoConfig = field(default_factory=EgoConfig)
    exogenous: ExogenousConfig = field(default_factory=ExogenousConfig)
    hypotheses: HypothesisConfig = field(default_factory=HypothesisConfig)
    enable_online_learning: bool = True
    enable_executable_models: bool = True
    strict_finite: bool = True
    boundary_bridge_change_fraction: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_mode", RuntimeMode(self.runtime_mode))
        _require_integer("seed", self.seed, minimum=None)
        expected = (
            ("search", self.search, SearchConfig),
            ("policy", self.policy, PolicyConfig),
            ("model", self.model, ModelConfig),
            ("perception", self.perception, PerceptionConfig),
            ("memory", self.memory, MemoryConfig),
            ("learning", self.learning, LearningConfig),
            ("ego", self.ego, EgoConfig),
            ("exogenous", self.exogenous, ExogenousConfig),
            ("hypotheses", self.hypotheses, HypothesisConfig),
        )
        for name, value, kind in expected:
            if not isinstance(value, kind):
                raise TypeError(f"{name} must be a {kind.__name__}")
        for name in (
            "enable_online_learning",
            "enable_executable_models",
            "strict_finite",
        ):
            _require_bool(name, getattr(self, name))
        _require_number(
            "boundary_bridge_change_fraction",
            self.boundary_bridge_change_fraction,
            minimum=0.0,
            maximum=1.0,
        )


@runtime_checkable
class ObservationAdapter(Protocol):
    def observation(self, raw: Any, *, task_id: str) -> Observation:
        """Convert an environment observation to the canonical immutable view."""


@runtime_checkable
class ActionAdapter(Protocol):
    def decode(self, action: Action, action_space: Any) -> tuple[Any, Mapping[str, Any]]:
        """Convert a canonical action into an environment action and kwargs."""

    def bootstrap(self, action_space: Any) -> tuple[Any, Mapping[str, Any]]:
        """Return the action used to obtain the first observation."""

    def click_action_index(self) -> int | None:
        """Return the positional action index, or ``None`` for clickless domains."""

    def safe_action_indices(self, observation: Observation | None = None) -> Sequence[int]:
        """Return legal conservative fallbacks."""


@runtime_checkable
class OutcomeAdapter(Protocol):
    def outcome(
        self,
        before: Observation,
        after: Observation,
        *,
        raw_after: Any,
        info: Mapping[str, Any] | None = None,
    ) -> Outcome:
        """Convert environment state/progress into a transition outcome."""


@runtime_checkable
class RepresentationBackend(Protocol):
    """Representation encoder used at both decision and commit time.

    Stateful external implementations should either support ``deepcopy`` or
    expose a matching ``state_dict``/``load_state_dict`` pair so a failed
    observe transaction can restore encoder state.  A backend whose encode is
    genuinely read-only may declare ``transactionally_stateless = True``;
    this avoids copying large frozen models.  An implementation that declares
    ``transactionally_stateful = True`` fails closed if neither rollback path
    is available.
    """

    def encode(
        self,
        observation: Observation,
        objects: Sequence[ObjectState],
    ) -> Representation:
        """Return frozen/grid features and optional named hidden-state taps."""


__all__ = [
    "SCHEMA_VERSION",
    "Action",
    "ActionAdapter",
    "AgentConfig",
    "BoundaryKind",
    "Candidate",
    "CompetenceState",
    "Decision",
    "EgoConfig",
    "EventKind",
    "EvidenceScope",
    "ExogenousConfig",
    "HypothesisConfig",
    "MemoryConfig",
    "LearningConfig",
    "ModelConfig",
    "ObjectState",
    "Observation",
    "ObservationAdapter",
    "Outcome",
    "OutcomeAdapter",
    "PerceptionConfig",
    "PolicyConfig",
    "Prediction",
    "Representation",
    "RepresentationBackend",
    "RuntimeMode",
    "ScoreTerm",
    "SearchConfig",
    "TapRole",
    "Topology",
    "Transition",
    "WorldEvent",
    "WorldSnapshot",
    "frozen_mapping",
    "readonly_array",
    "stable_frame_hash",
]
