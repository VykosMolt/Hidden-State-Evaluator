"""Compact categorical perception and tracking for Hunter-Seeker v2.

The module is deliberately independent of the legacy Hunter-Seeker mixin
stack.  It provides:

* deterministic 4-connected components over categorical 2D frames;
* stable global object tracking with velocity, misses, and pruning;
* events attributed to the transition that just occurred;
* a cheap free-space/topology summary;
* explicit state export/import and isolated speculative forks.

Object ``signature`` values describe value plus translation-invariant shape.
They are deterministic but not required to be unique: two equal-value objects
with the same shape intentionally share a signature while their ``track_id``
values preserve instance identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    EventKind,
    ObjectState,
    Observation,
    PerceptionConfig,
    Topology,
    WorldEvent,
)


PERCEPTION_STATE_VERSION = 1


def _state_int(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    decoded = int(value)
    if minimum is not None and decoded < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return decoded


def _state_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a number")
    decoded = float(value)
    if not math.isfinite(decoded):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and decoded < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return decoded


def _state_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _state_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise ValueError(f"{name} must be a sequence")
    return value


def _state_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class Component:
    """One deterministic frame-local categorical component.

    ``bbox`` uses inclusive ``(x_min, y_min, x_max, y_max)`` coordinates.
    Pixels are sorted ``(x, y)`` pairs so the record is immutable and stable
    across traversal implementations.
    """

    object_id: int
    value: int
    area: int
    centroid_x: float
    centroid_y: float
    bbox: tuple[int, int, int, int]
    touches_border: bool
    signature: str
    pixels: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    """Immutable output of one real or speculative perception update."""

    objects: tuple[ObjectState, ...]
    events: tuple[WorldEvent, ...]
    topology: Topology
    components: tuple[Component, ...]
    background_value: int
    step: int


@dataclass(slots=True)
class _Track:
    track_id: int
    value: int
    area: int
    centroid_x: float
    centroid_y: float
    bbox: tuple[int, int, int, int]
    signature: str
    pixels: tuple[tuple[int, int], ...]
    touches_border: bool
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    misses: int = 0
    age: int = 1
    visible: bool = True
    object_id: int = -1

    def object_state(self) -> ObjectState:
        return ObjectState(
            object_id=int(self.object_id),
            track_id=int(self.track_id),
            value=int(self.value),
            area=int(self.area),
            centroid_x=float(self.centroid_x),
            centroid_y=float(self.centroid_y),
            bbox=tuple(int(v) for v in self.bbox),
            touches_border=bool(self.touches_border),
            velocity_x=float(self.velocity_x),
            velocity_y=float(self.velocity_y),
            confidence=float(min(1.0, 0.5 + 0.1 * self.age)),
            signature=str(self.signature),
        )


def _categorical_frame(frame: np.ndarray | Sequence[Any]) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError(f"categorical perception requires a 2D frame, got {arr.shape}")
    if arr.size == 0:
        raise ValueError("categorical perception requires a non-empty frame")
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        return np.ascontiguousarray(arr.astype(np.int64, copy=False))
    try:
        numeric = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("categorical frame values must be numeric integers") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all():
        raise ValueError("categorical frame values must be finite integers")
    return np.ascontiguousarray(np.rint(numeric).astype(np.int64))


def _background_value(frame: np.ndarray, configured: int | None) -> int:
    if configured is not None:
        return int(configured)
    values, counts = np.unique(frame, return_counts=True)
    max_count = int(counts.max())
    # np.unique is sorted, so the first maximum gives deterministic tie-breaking.
    return int(values[np.flatnonzero(counts == max_count)[0]])


def _component_signature(
    value: int,
    pixels: tuple[tuple[int, int], ...],
    bbox: tuple[int, int, int, int],
) -> str:
    x0, y0, x1, y1 = bbox
    relative = np.asarray(
        [(int(x) - x0, int(y) - y0) for x, y in pixels],
        dtype=np.int32,
    )
    digest = blake2b(digest_size=12)
    digest.update(int(value).to_bytes(8, "little", signed=True))
    digest.update(np.asarray((x1 - x0 + 1, y1 - y0 + 1), dtype=np.int32).tobytes())
    digest.update(relative.tobytes())
    return digest.hexdigest()


def _mask_components(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    """Return deterministic row-major 4-connected pixel components."""

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[tuple[tuple[int, int], ...]] = []
    for y0 in range(h):
        for x0 in range(w):
            if not bool(mask[y0, x0]) or bool(seen[y0, x0]):
                continue
            seen[y0, x0] = True
            stack = [(x0, y0)]
            pixels: list[tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                pixels.append((x, y))
                # Reverse row-major push order because this is a LIFO stack.
                for nx, ny in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)):
                    if (
                        0 <= nx < w
                        and 0 <= ny < h
                        and bool(mask[ny, nx])
                        and not bool(seen[ny, nx])
                    ):
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            out.append(tuple(sorted(pixels, key=lambda p: (p[1], p[0]))))
    return out


def connected_components(
    frame: np.ndarray | Sequence[Any],
    config: PerceptionConfig | None = None,
) -> tuple[int, tuple[Component, ...]]:
    """Parse a categorical frame into deterministic non-background objects."""

    cfg = config or PerceptionConfig()
    arr = _categorical_frame(frame)
    bg = _background_value(arr, cfg.background_value)
    min_area = max(1, int(cfg.min_object_area))
    h, w = arr.shape
    raw: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for value in sorted(int(v) for v in np.unique(arr) if int(v) != bg):
        for pixels in _mask_components(arr == value):
            if len(pixels) >= min_area:
                raw.append((value, pixels))

    sortable: list[
        tuple[
            tuple[int, int, int, int, int, int, str],
            int,
            tuple[tuple[int, int], ...],
            tuple[int, int, int, int],
            str,
        ]
    ] = []
    for value, pixels in raw:
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        signature = _component_signature(value, pixels, bbox)
        key = (
            int(value),
            int(bbox[1]),
            int(bbox[0]),
            int(bbox[3]),
            int(bbox[2]),
            int(len(pixels)),
            signature,
        )
        sortable.append((key, value, pixels, bbox, signature))
    sortable.sort(key=lambda row: row[0])

    components: list[Component] = []
    for object_id, (_key, value, pixels, bbox, signature) in enumerate(sortable):
        xs = np.asarray([p[0] for p in pixels], dtype=np.float64)
        ys = np.asarray([p[1] for p in pixels], dtype=np.float64)
        components.append(
            Component(
                object_id=int(object_id),
                value=int(value),
                area=int(len(pixels)),
                centroid_x=float(xs.mean()),
                centroid_y=float(ys.mean()),
                bbox=bbox,
                touches_border=bool(
                    bbox[0] == 0
                    or bbox[1] == 0
                    or bbox[2] == w - 1
                    or bbox[3] == h - 1
                ),
                signature=signature,
                pixels=pixels,
            )
        )
    return bg, tuple(components)


def _bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    if ix0 > ix1 or iy0 > iy1:
        return 0.0
    intersection = float((ix1 - ix0 + 1) * (iy1 - iy0 + 1))
    left_area = float((lx1 - lx0 + 1) * (ly1 - ly0 + 1))
    right_area = float((rx1 - rx0 + 1) * (ry1 - ry0 + 1))
    return intersection / max(1.0, left_area + right_area - intersection)


def _minimum_cost_matching(
    pairs: Sequence[tuple[float, int, int, bool]],
    track_ids: Sequence[int],
    object_ids: Sequence[int],
) -> dict[int, int]:
    """Lexicographic deterministic bipartite matching.

    Greedily claiming the cheapest individual edge can force a later object
    onto the wrong track even when a lower-cost complete assignment exists.
    A rectangular Hungarian assignment with encoded costs, in order:

    1. maximizes the number of valid matches;
    2. maximizes same-value continuity among equal-cardinality matchings;
    3. minimizes geometric/shape cost.

    The second tier prevents a moving marker from swapping identities with a
    nearby vacated floor component merely because both cross-value matches
    are locally cheap.  Cardinality remains the first tier, so a legitimate
    local transformation is still matched when no same-value assignment can
    achieve the same number of matches.
    """

    tracks = tuple(sorted(int(value) for value in track_ids))
    objects = tuple(sorted(int(value) for value in object_ids))
    if not tracks or not objects or not pairs:
        return {}
    raw_allowed: dict[tuple[int, int], tuple[float, bool]] = {}
    for cost, track_id, object_id, same_value in pairs:
        decoded_cost = float(cost)
        if not math.isfinite(decoded_cost):
            raise ValueError("perception match cost must be finite")
        if not isinstance(same_value, (bool, np.bool_)):
            raise ValueError("perception same-value flag must be boolean")
        raw_allowed[(int(track_id), int(object_id))] = (
            decoded_cost,
            bool(same_value),
        )

    # Affine-normalize base costs before adding the lexicographic continuity
    # tier.  For equal-cardinality assignments this preserves exact cost
    # ordering, while avoiding overflow for large but finite configured gates.
    scale = max(
        1.0,
        max(abs(cost) for cost, _same_value in raw_allowed.values()),
    )
    scaled = {
        key: cost / scale
        for key, (cost, _same_value) in raw_allowed.items()
    }
    scaled_min = min(scaled.values())
    scaled_span = max(scaled.values()) - scaled_min
    maximum_matches = min(len(tracks), len(objects))
    continuity_penalty = maximum_matches * scaled_span + 1.0
    allowed = {
        key: (
            scaled[key]
            - scaled_min
            + (0.0 if same_value else continuity_penalty)
        )
        for key, (_cost, same_value) in raw_allowed.items()
    }
    maximum = max(allowed.values(), default=0.0)
    dummy_cost = (len(tracks) + 1.0) * (maximum + 1.0)
    forbidden_cost = (len(tracks) + 1.0) * dummy_cost
    # One private-capacity dummy column per track permits every row to remain
    # unmatched without ever selecting a forbidden real edge.
    column_count = len(objects) + len(tracks)
    costs = [
        [
            (
                allowed.get((track_id, objects[column]), forbidden_cost)
                if column < len(objects)
                else dummy_cost
            )
            for column in range(column_count)
        ]
        for track_id in tracks
    ]

    # Hungarian algorithm for rows <= columns, using one-based work arrays.
    row_count = len(tracks)
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        p[0] = row
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment: dict[int, int] = {}
    for column in range(1, column_count + 1):
        row = p[column]
        if row == 0 or column > len(objects):
            continue
        track_id = tracks[row - 1]
        object_id = objects[column - 1]
        if (track_id, object_id) in allowed:
            assignment[object_id] = track_id
    return assignment


def _component_adjacencies(
    shape: tuple[int, int],
    components: Sequence[Component],
) -> tuple[tuple[int, int], ...]:
    owner = np.full(shape, -1, dtype=np.int32)
    for component in components:
        for x, y in component.pixels:
            owner[y, x] = int(component.object_id)
    pairs: set[tuple[int, int]] = set()
    if owner.shape[1] > 1:
        left = owner[:, :-1]
        right = owner[:, 1:]
        ys, xs = np.where((left >= 0) & (right >= 0) & (left != right))
        for y, x in zip(ys.tolist(), xs.tolist()):
            pairs.add(tuple(sorted((int(left[y, x]), int(right[y, x])))))
    if owner.shape[0] > 1:
        top = owner[:-1, :]
        bottom = owner[1:, :]
        ys, xs = np.where((top >= 0) & (bottom >= 0) & (top != bottom))
        for y, x in zip(ys.tolist(), xs.tolist()):
            pairs.add(tuple(sorted((int(top[y, x]), int(bottom[y, x])))))
    return tuple(sorted(pairs))


def cheap_topology(
    frame: np.ndarray | Sequence[Any],
    *,
    background_value: int,
    components: Sequence[Component],
) -> Topology:
    """Summarize free-space connectivity and object frontiers cheaply."""

    arr = _categorical_frame(frame)
    free = arr == int(background_value)
    free_components = _mask_components(free)
    total_free = int(free.sum())
    largest = max(free_components, key=len, default=())
    largest_fraction = float(len(largest) / total_free) if total_free else 0.0

    object_adjacencies = _component_adjacencies(arr.shape, components)
    object_pixels = np.zeros_like(free, dtype=bool)
    for component in components:
        for x, y in component.pixels:
            object_pixels[y, x] = True
    frontier = np.zeros_like(free, dtype=bool)
    if arr.shape[0] > 1:
        frontier[:-1, :] |= free[:-1, :] & object_pixels[1:, :]
        frontier[1:, :] |= free[1:, :] & object_pixels[:-1, :]
    if arr.shape[1] > 1:
        frontier[:, :-1] |= free[:, :-1] & object_pixels[:, 1:]
        frontier[:, 1:] |= free[:, 1:] & object_pixels[:, :-1]
    frontier_fraction = float(frontier.sum() / total_free) if total_free else 0.0

    reachable: set[int] = set()
    if largest:
        largest_mask = np.zeros_like(free, dtype=bool)
        for x, y in largest:
            largest_mask[y, x] = True
        contact = largest_mask.copy()
        if arr.shape[0] > 1:
            contact[:-1, :] |= largest_mask[1:, :]
            contact[1:, :] |= largest_mask[:-1, :]
        if arr.shape[1] > 1:
            contact[:, :-1] |= largest_mask[:, 1:]
            contact[:, 1:] |= largest_mask[:, :-1]
        for component in components:
            if any(bool(contact[y, x]) for x, y in component.pixels):
                reachable.add(int(component.object_id))

    return Topology(
        component_count=int(len(free_components)),
        largest_component_fraction=float(largest_fraction),
        frontier_fraction=float(frontier_fraction),
        object_adjacencies=object_adjacencies,
        reachable_object_ids=tuple(sorted(reachable)),
    )


class PerceptionSystem:
    """Stateful categorical tracker with explicit transactional copies."""

    def __init__(self, config: PerceptionConfig | None = None) -> None:
        self.config = config or PerceptionConfig()
        self._tracks: dict[int, _Track] = {}
        self._contacts: set[tuple[int, int]] = set()
        self._next_track_id = 0
        self._step = -1
        self._task_id: str | None = None
        self._stage: int | None = None

    @property
    def step(self) -> int:
        return int(self._step)

    @property
    def track_count(self) -> int:
        return int(len(self._tracks))

    @property
    def visible_track_count(self) -> int:
        return int(sum(1 for track in self._tracks.values() if track.visible))

    def reset(
        self,
        *,
        task_id: str | None = None,
        stage: int | None = None,
    ) -> None:
        self._tracks.clear()
        self._contacts.clear()
        self._next_track_id = 0
        self._step = -1
        self._task_id = None if task_id is None else str(task_id)
        self._stage = None if stage is None else max(1, int(stage))

    def _match_cost(self, track: _Track, component: Component) -> float | None:
        dx = float(component.centroid_x - track.centroid_x)
        dy = float(component.centroid_y - track.centroid_y)
        distance = math.hypot(dx, dy)
        gate = max(0.0, float(self.config.track_match_distance))
        gate *= 1.0 + 0.25 * min(track.misses, max(0, self.config.track_miss_tolerance))
        overlap = _bbox_iou(track.bbox, component.bbox)
        same_value = int(track.value) == int(component.value)
        if distance > gate and overlap <= 0.0:
            return None
        # Cross-value matching is reserved for local transformations rather
        # than allowing a distant new object to steal an old identity.
        if not same_value and distance > max(1.5, 0.5 * gate) and overlap <= 0.0:
            return None
        area_ratio = max(track.area, component.area) / max(1.0, min(track.area, component.area))
        shape_penalty = 0.0 if track.signature == component.signature else 0.35
        value_penalty = 0.0 if same_value else max(0.75, 0.25 * gate)
        miss_penalty = 0.10 * float(track.misses)
        return float(
            distance
            + 0.25 * math.log(max(1.0, area_ratio))
            + shape_penalty
            + value_penalty
            + miss_penalty
            - 0.75 * overlap
        )

    @staticmethod
    def _event_sort_key(event: WorldEvent) -> tuple[int, int, int]:
        order = {
            EventKind.APPEARED: 0,
            EventKind.DISAPPEARED: 1,
            EventKind.MOVED: 2,
            EventKind.TRANSFORMED: 3,
            EventKind.CONTACT: 4,
        }
        other = event.metadata.get("other_track_id", -1)
        return (
            int(order.get(event.kind, 99)),
            int(event.subject_track_id),
            int(other) if isinstance(other, (int, np.integer)) else -1,
        )

    def observe(
        self,
        observation: Observation | np.ndarray | Sequence[Any],
        *,
        step: int | None = None,
    ) -> PerceptionResult:
        """Commit one real observation and return current-transition events."""

        if isinstance(observation, Observation):
            frame = observation.frame
            task_id = str(observation.task_id)
            stage = max(1, int(observation.stage))
            if self._task_id is not None and task_id != self._task_id:
                self.reset(task_id=task_id, stage=stage)
            elif self._stage is not None and stage != self._stage:
                # Track identity is local to one stage geometry.  Carrying it
                # across a boundary creates synthetic MOVED/TRANSFORMED events
                # and can attach the new stage's objects to old causal stats.
                self.reset(task_id=task_id, stage=stage)
            else:
                self._task_id = task_id
                self._stage = stage
        else:
            frame = np.asarray(observation)

        self._step = int(self._step + 1 if step is None else step)
        arr = _categorical_frame(frame)
        background, components = connected_components(arr, self.config)

        tracks_before_visible = {
            track_id: bool(track.visible) for track_id, track in self._tracks.items()
        }
        pairs: list[tuple[float, int, int, bool]] = []
        for track_id, track in sorted(self._tracks.items()):
            for component in components:
                cost = self._match_cost(track, component)
                if cost is not None:
                    pairs.append(
                        (
                            float(cost),
                            int(track_id),
                            int(component.object_id),
                            int(track.value) == int(component.value),
                        )
                    )
        object_to_track = _minimum_cost_matching(
            pairs,
            tuple(self._tracks),
            tuple(component.object_id for component in components),
        )
        assigned_objects = set(object_to_track)
        assigned_tracks = set(object_to_track.values())

        events: list[WorldEvent] = []
        by_object = {component.object_id: component for component in components}
        visible_track_ids_now: set[int] = set(assigned_tracks)
        for object_id in sorted(assigned_objects):
            component = by_object[object_id]
            track = self._tracks[object_to_track[object_id]]
            was_visible = tracks_before_visible.get(track.track_id, False)
            old_value = int(track.value)
            old_signature = str(track.signature)
            old_area = int(track.area)
            dx = float(component.centroid_x - track.centroid_x)
            dy = float(component.centroid_y - track.centroid_y)
            moved = math.hypot(dx, dy)
            transformed = (
                old_value != int(component.value)
                or old_signature != str(component.signature)
            )

            track.value = int(component.value)
            track.area = int(component.area)
            track.centroid_x = float(component.centroid_x)
            track.centroid_y = float(component.centroid_y)
            track.bbox = component.bbox
            track.signature = str(component.signature)
            track.pixels = component.pixels
            track.touches_border = bool(component.touches_border)
            track.velocity_x = dx
            track.velocity_y = dy
            track.misses = 0
            track.age += 1
            track.visible = True
            track.object_id = int(component.object_id)

            if not was_visible:
                events.append(
                    WorldEvent(
                        EventKind.APPEARED,
                        subject_track_id=track.track_id,
                        object_signature=track.signature,
                        magnitude=1.0,
                        metadata={"reappeared": True, "value": track.value},
                    )
                )
            if moved > 1e-6:
                events.append(
                    WorldEvent(
                        EventKind.MOVED,
                        subject_track_id=track.track_id,
                        object_signature=track.signature,
                        magnitude=moved,
                        metadata={"dx": dx, "dy": dy},
                    )
                )
            if transformed:
                area_change = abs(track.area - old_area) / max(1.0, float(old_area))
                events.append(
                    WorldEvent(
                        EventKind.TRANSFORMED,
                        subject_track_id=track.track_id,
                        object_signature=track.signature,
                        magnitude=float(min(1.0, (old_value != track.value) + area_change)),
                        metadata={
                            "old_value": old_value,
                            "new_value": track.value,
                            "old_signature": old_signature,
                            "new_signature": track.signature,
                        },
                    )
                )

        for component in components:
            if component.object_id in assigned_objects:
                continue
            track_id = int(self._next_track_id)
            self._next_track_id += 1
            track = _Track(
                track_id=track_id,
                value=int(component.value),
                area=int(component.area),
                centroid_x=float(component.centroid_x),
                centroid_y=float(component.centroid_y),
                bbox=component.bbox,
                signature=str(component.signature),
                pixels=component.pixels,
                touches_border=bool(component.touches_border),
                object_id=int(component.object_id),
            )
            self._tracks[track_id] = track
            object_to_track[int(component.object_id)] = track_id
            visible_track_ids_now.add(track_id)
            events.append(
                WorldEvent(
                    EventKind.APPEARED,
                    subject_track_id=track_id,
                    object_signature=track.signature,
                    magnitude=1.0,
                    metadata={"reappeared": False, "value": track.value},
                )
            )

        for track_id, track in sorted(tuple(self._tracks.items())):
            if track_id in visible_track_ids_now:
                continue
            if track.visible:
                events.append(
                    WorldEvent(
                        EventKind.DISAPPEARED,
                        subject_track_id=track.track_id,
                        object_signature=track.signature,
                        magnitude=1.0,
                        metadata={"value": track.value, "misses": track.misses + 1},
                    )
                )
            track.visible = False
            track.object_id = -1
            track.velocity_x = 0.0
            track.velocity_y = 0.0
            track.misses += 1

        tolerance = max(0, int(self.config.track_miss_tolerance))
        for track_id in [
            tid for tid, track in self._tracks.items() if int(track.misses) > tolerance
        ]:
            del self._tracks[track_id]

        adjacency_ids = _component_adjacencies(arr.shape, components)
        contacts_now: set[tuple[int, int]] = set()
        for left_object, right_object in adjacency_ids:
            left_track = object_to_track.get(int(left_object))
            right_track = object_to_track.get(int(right_object))
            if left_track is None or right_track is None or left_track == right_track:
                continue
            contacts_now.add(tuple(sorted((int(left_track), int(right_track)))))
        for left_track, right_track in sorted(contacts_now - self._contacts):
            subject = self._tracks.get(left_track)
            events.append(
                WorldEvent(
                    EventKind.CONTACT,
                    subject_track_id=left_track,
                    object_signature=subject.signature if subject is not None else "",
                    magnitude=1.0,
                    metadata={"other_track_id": right_track},
                )
            )
        self._contacts = contacts_now

        objects = tuple(
            track.object_state()
            for track in sorted(
                (track for track in self._tracks.values() if track.visible),
                key=lambda item: (item.object_id, item.track_id),
            )
        )
        topology = cheap_topology(
            arr,
            background_value=background,
            components=components,
        )
        return PerceptionResult(
            objects=objects,
            events=tuple(sorted(events, key=self._event_sort_key)),
            topology=topology,
            components=components,
            background_value=int(background),
            step=int(self._step),
        )

    def export_state(self) -> dict[str, Any]:
        """Return a durable, JSON-compatible tracking state."""

        return {
            "version": PERCEPTION_STATE_VERSION,
            "config": {
                "background_value": self.config.background_value,
                "min_object_area": int(self.config.min_object_area),
                "track_match_distance": float(self.config.track_match_distance),
                "track_miss_tolerance": int(self.config.track_miss_tolerance),
            },
            "task_id": self._task_id,
            "stage": self._stage,
            "step": int(self._step),
            "next_track_id": int(self._next_track_id),
            "contacts": [list(pair) for pair in sorted(self._contacts)],
            "tracks": [
                {
                    "track_id": int(track.track_id),
                    "value": int(track.value),
                    "area": int(track.area),
                    "centroid_x": float(track.centroid_x),
                    "centroid_y": float(track.centroid_y),
                    "bbox": list(track.bbox),
                    "signature": str(track.signature),
                    "pixels": [list(pixel) for pixel in track.pixels],
                    "touches_border": bool(track.touches_border),
                    "velocity_x": float(track.velocity_x),
                    "velocity_y": float(track.velocity_y),
                    "misses": int(track.misses),
                    "age": int(track.age),
                    "visible": bool(track.visible),
                    "object_id": int(track.object_id),
                }
                for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
            ],
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Replace current state from :meth:`export_state` output."""

        if not isinstance(state, Mapping):
            raise ValueError("perception state must be a mapping")
        version = _state_int(
            state.get("version"),
            "perception state version",
        )
        if version != PERCEPTION_STATE_VERSION:
            raise ValueError(
                f"unsupported perception state version {version}; "
                f"expected {PERCEPTION_STATE_VERSION}"
            )
        config_raw = state.get("config")
        if not isinstance(config_raw, Mapping):
            raise ValueError("perception state config must be a mapping")
        required_config = {
            "background_value",
            "min_object_area",
            "track_match_distance",
            "track_miss_tolerance",
        }
        if not required_config.issubset(config_raw):
            missing = sorted(required_config - set(config_raw))
            raise ValueError(
                f"perception state config is missing fields {missing!r}"
            )
        background_raw = config_raw["background_value"]
        decoded_config = PerceptionConfig(
            background_value=(
                None
                if background_raw is None
                else _state_int(
                    background_raw,
                    "perception background_value",
                )
            ),
            min_object_area=_state_int(
                config_raw["min_object_area"],
                "perception min_object_area",
                minimum=1,
            ),
            track_match_distance=_state_number(
                config_raw["track_match_distance"],
                "perception track_match_distance",
                minimum=0.0,
            ),
            track_miss_tolerance=_state_int(
                config_raw["track_miss_tolerance"],
                "perception track_miss_tolerance",
                minimum=0,
            ),
        )
        tracks: dict[int, _Track] = {}
        raw_tracks = _state_sequence(
            state.get("tracks"),
            "perception state tracks",
        )
        visible_object_ids: set[int] = set()
        for index, raw in enumerate(raw_tracks):
            if not isinstance(raw, Mapping):
                raise ValueError("perception track state must be a mapping")
            required_track = {
                "track_id",
                "value",
                "area",
                "centroid_x",
                "centroid_y",
                "bbox",
                "signature",
                "pixels",
                "touches_border",
                "velocity_x",
                "velocity_y",
                "misses",
                "age",
                "visible",
                "object_id",
            }
            if not required_track.issubset(raw):
                missing = sorted(required_track - set(raw))
                raise ValueError(
                    f"perception track {index} is missing fields {missing!r}"
                )
            prefix = f"perception track {index}"
            track_id = _state_int(
                raw["track_id"],
                f"{prefix} track_id",
                minimum=0,
            )
            if track_id in tracks:
                raise ValueError(f"duplicate perception track id {track_id}")
            bbox_values = _state_sequence(raw["bbox"], f"{prefix} bbox")
            if len(bbox_values) != 4:
                raise ValueError("track bbox must contain four coordinates")
            bbox_raw = tuple(
                _state_int(
                    value,
                    f"{prefix} bbox coordinate",
                    minimum=0,
                )
                for value in bbox_values
            )
            x0, y0, x1, y1 = bbox_raw
            if x0 > x1 or y0 > y1:
                raise ValueError(f"{prefix} bbox is inverted")
            raw_pixels = _state_sequence(raw["pixels"], f"{prefix} pixels")
            decoded_pixels: list[tuple[int, int]] = []
            for pixel_index, pixel in enumerate(raw_pixels):
                values = _state_sequence(
                    pixel,
                    f"{prefix} pixel {pixel_index}",
                )
                if len(values) != 2:
                    raise ValueError(
                        f"{prefix} pixel {pixel_index} must contain [x, y]"
                    )
                decoded_pixels.append(
                    (
                        _state_int(
                            values[0],
                            f"{prefix} pixel x",
                            minimum=0,
                        ),
                        _state_int(
                            values[1],
                            f"{prefix} pixel y",
                            minimum=0,
                        ),
                    )
                )
            pixels = tuple(decoded_pixels)
            if not pixels:
                raise ValueError(f"{prefix} pixels must not be empty")
            if len(set(pixels)) != len(pixels):
                raise ValueError(f"{prefix} pixels contain duplicates")
            if pixels != tuple(sorted(pixels, key=lambda point: (point[1], point[0]))):
                raise ValueError(f"{prefix} pixels must be in row-major order")
            expected_bbox = (
                min(point[0] for point in pixels),
                min(point[1] for point in pixels),
                max(point[0] for point in pixels),
                max(point[1] for point in pixels),
            )
            if bbox_raw != expected_bbox:
                raise ValueError(
                    f"{prefix} bbox does not bound its pixels exactly"
                )
            value = _state_int(raw["value"], f"{prefix} value")
            area = _state_int(raw["area"], f"{prefix} area", minimum=1)
            if area != len(pixels):
                raise ValueError(f"{prefix} area does not equal pixel count")
            centroid_x = _state_number(
                raw["centroid_x"],
                f"{prefix} centroid_x",
            )
            centroid_y = _state_number(
                raw["centroid_y"],
                f"{prefix} centroid_y",
            )
            expected_centroid_x = sum(point[0] for point in pixels) / area
            expected_centroid_y = sum(point[1] for point in pixels) / area
            if not math.isclose(
                centroid_x,
                expected_centroid_x,
                rel_tol=0.0,
                abs_tol=1e-9,
            ) or not math.isclose(
                centroid_y,
                expected_centroid_y,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{prefix} centroid does not equal its pixel centroid"
                )
            signature = _state_string(
                raw["signature"],
                f"{prefix} signature",
            )
            expected_signature = _component_signature(
                value,
                pixels,
                bbox_raw,
            )
            if signature != expected_signature:
                raise ValueError(
                    f"{prefix} signature does not match value and pixels"
                )
            misses = _state_int(
                raw["misses"],
                f"{prefix} misses",
                minimum=0,
            )
            if misses > decoded_config.track_miss_tolerance:
                raise ValueError(
                    f"{prefix} misses exceeds track_miss_tolerance"
                )
            age = _state_int(raw["age"], f"{prefix} age", minimum=1)
            visible = _state_bool(raw["visible"], f"{prefix} visible")
            object_id = _state_int(
                raw["object_id"],
                f"{prefix} object_id",
                minimum=-1,
            )
            if visible:
                if misses != 0 or object_id < 0:
                    raise ValueError(
                        f"{prefix} visible state requires misses=0 and object_id>=0"
                    )
                if object_id in visible_object_ids:
                    raise ValueError(
                        f"duplicate visible perception object id {object_id}"
                    )
                visible_object_ids.add(object_id)
            elif misses < 1 or object_id != -1:
                raise ValueError(
                    f"{prefix} hidden state requires misses>=1 and object_id=-1"
                )
            tracks[track_id] = _Track(
                track_id=track_id,
                value=value,
                area=area,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                bbox=bbox_raw,
                signature=signature,
                pixels=pixels,
                touches_border=_state_bool(
                    raw["touches_border"],
                    f"{prefix} touches_border",
                ),
                velocity_x=_state_number(
                    raw["velocity_x"],
                    f"{prefix} velocity_x",
                ),
                velocity_y=_state_number(
                    raw["velocity_y"],
                    f"{prefix} velocity_y",
                ),
                misses=misses,
                age=age,
                visible=visible,
                object_id=object_id,
            )
        if visible_object_ids != set(range(len(visible_object_ids))):
            raise ValueError(
                "visible perception object ids must be contiguous from zero"
            )
        raw_contacts = _state_sequence(
            state.get("contacts"),
            "perception state contacts",
        )
        contacts: set[tuple[int, int]] = set()
        for index, pair in enumerate(raw_contacts):
            values = _state_sequence(pair, f"perception contact {index}")
            if len(values) != 2:
                raise ValueError(
                    f"perception contact {index} must contain two track ids"
                )
            left = _state_int(
                values[0],
                f"perception contact {index} left id",
                minimum=0,
            )
            right = _state_int(
                values[1],
                f"perception contact {index} right id",
                minimum=0,
            )
            if left >= right:
                raise ValueError(
                    "perception contacts must be canonical distinct pairs"
                )
            contact = (left, right)
            if contact in contacts:
                raise ValueError(f"duplicate perception contact {contact!r}")
            if (
                left not in tracks
                or right not in tracks
                or not tracks[left].visible
                or not tracks[right].visible
            ):
                raise ValueError(
                    "perception contacts must reference visible tracks"
                )
            contacts.add(contact)
        next_track_id = _state_int(
            state.get("next_track_id"),
            "perception next_track_id",
            minimum=0,
        )
        if tracks and next_track_id <= max(tracks):
            raise ValueError(
                "perception next_track_id must exceed every existing track id"
            )
        step = _state_int(
            state.get("step"),
            "perception step",
            minimum=-1,
        )
        if tracks and step < 0:
            raise ValueError("perception tracks require a nonnegative step")
        task_id = state.get("task_id")
        decoded_task_id = (
            None
            if task_id is None
            else _state_string(task_id, "perception task_id")
        )
        stage = state.get("stage")
        decoded_stage = (
            None
            if stage is None
            else _state_int(stage, "perception stage", minimum=1)
        )

        # Commit only after the entire replacement has been validated.
        self.config = decoded_config
        self._tracks = tracks
        self._contacts = contacts
        self._next_track_id = next_track_id
        self._step = step
        self._task_id = decoded_task_id
        self._stage = decoded_stage

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "PerceptionSystem":
        system = cls()
        system.import_state(state)
        return system

    def fork(self) -> "PerceptionSystem":
        """Return a fully independent planning copy."""

        return self.from_state(self.export_state())

    def preview(
        self,
        observation: Observation | np.ndarray | Sequence[Any],
        *,
        step: int | None = None,
    ) -> PerceptionResult:
        """Observe on an isolated fork, leaving durable state untouched."""

        return self.fork().observe(observation, step=step)


__all__ = [
    "PERCEPTION_STATE_VERSION",
    "Component",
    "PerceptionResult",
    "PerceptionSystem",
    "cheap_topology",
    "connected_components",
]
