"""Domain-general control attribution (ego/controlled-set localization).

A track is controlled to the degree that its observed displacement response
*depends on* the executed action.  The influence score is the count-based
discrete analog of causal action influence (Seitzer et al., NeurIPS 2021):
normalized mutual information between the action index and the track's
quantized displacement bucket, scaled by support trust.  Static objects and
action-independent drifters both collapse to a single response distribution
and score zero by construction.

The model consumes only committed real transitions (MOVED/DISAPPEARED events
plus the executed action), never speculative rollouts, and never selects
actions itself.  Its outputs are:

* ``enrich``: grounds the existing ``ObjectState.controllable`` belief;
* ``controlled_objects``: the current controlled set for outcome attribution;
* ``motion_hazard``: a bounded [0, 1] reading for one named score term.

Directional attribution is restricted to non-positional actions; click
attribution stays with the affordance click-target route.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    Action,
    EgoConfig,
    ObjectState,
    WorldEvent,
    WorldSnapshot,
)


_BUCKETS = 9  # none + 8 octants


def _state_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    decoded = int(value)
    if decoded < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return decoded


def _state_key_int(value: Any, name: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical integer string")
    if (
        not value.isascii()
        or not value.isdecimal()
        or (len(value) > 1 and value.startswith("0"))
    ):
        raise ValueError(f"{name} must be a canonical nonnegative integer")
    return int(value)


def _state_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    integral: bool = False,
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
    if integral and not decoded.is_integer():
        raise ValueError(f"{name} must be integer-valued")
    return decoded


def _state_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise ValueError(f"{name} must be a sequence")
    return value


def _bucket(dx: float, dy: float, epsilon: float) -> int:
    if math.hypot(float(dx), float(dy)) <= max(0.0, float(epsilon)):
        return 0
    octant = int(round(math.atan2(float(dy), float(dx)) / (math.pi / 4.0))) % 8
    return 1 + octant


def _normalized_mutual_information(joint: Mapping[int, Sequence[float]]) -> float:
    """Plug-in normalized MI between action index and response bucket."""

    actions = [np.asarray(row, dtype=np.float64) for row in joint.values()]
    if len(actions) < 2:
        return 0.0
    table = np.stack(actions)
    total = float(table.sum())
    if total <= 0.0:
        return 0.0
    probabilities = table / total
    action_marginal = probabilities.sum(axis=1)
    bucket_marginal = probabilities.sum(axis=0)
    support_actions = int(np.count_nonzero(action_marginal))
    support_buckets = int(np.count_nonzero(bucket_marginal))
    if support_actions < 2 or support_buckets < 2:
        return 0.0
    mutual_information = 0.0
    for a in range(probabilities.shape[0]):
        for b in range(probabilities.shape[1]):
            p = probabilities[a, b]
            if p <= 0.0:
                continue
            mutual_information += p * math.log(
                p / (action_marginal[a] * bucket_marginal[b])
            )
    normalizer = math.log(min(support_actions, support_buckets))
    if normalizer <= 0.0:
        return 0.0
    return float(np.clip(mutual_information / normalizer, 0.0, 1.0))


class ControlAttribution:
    """Count-based action-to-motion attribution over tracks and signatures."""

    def __init__(self, config: EgoConfig | None = None) -> None:
        self.config = config or EgoConfig()
        # track_id -> action_index -> bucket counts
        self._track_buckets: dict[int, dict[int, list[float]]] = {}
        # track_id -> action_index -> [sum_dx, sum_dy, count]
        self._track_motion: dict[int, dict[int, list[float]]] = {}
        self._track_signatures: dict[int, str] = {}
        # signature-level aggregates bootstrap reappearing/respawned instances
        self._signature_buckets: dict[str, dict[int, list[float]]] = {}
        self._signature_motion: dict[str, dict[int, list[float]]] = {}
        self._active_task_id: str | None = None
        self.update_count = 0

    def __len__(self) -> int:
        return len(self._track_buckets)

    def begin_episode(self, task_id: str) -> None:
        """Start a fresh track namespace while retaining signature evidence.

        Perception track ids are episode-local and routinely restart at zero.
        Track-keyed causal counts therefore cannot survive a run boundary.
        Signature aggregates are deliberately retained: callers opt into that
        transferable bootstrap by supplying an object's signature to
        :meth:`influence`/``enrich``.
        """

        self.reset_tracks()
        self._active_task_id = str(task_id)

    def reset_tracks(self) -> None:
        """Drop evidence keyed by perception-local track ids only."""

        self._track_buckets.clear()
        self._track_motion.clear()
        self._track_signatures.clear()

    @staticmethod
    def _row(
        store: dict[Any, dict[int, list[float]]],
        key: Any,
        action_index: int,
        width: int,
    ) -> list[float]:
        per_action = store.setdefault(key, {})
        row = per_action.get(int(action_index))
        if row is None:
            row = [0.0] * width
            per_action[int(action_index)] = row
        return row

    def _prune(self) -> None:
        limit = max(1, int(self.config.max_tracked))
        while len(self._track_buckets) > limit:
            oldest = next(iter(self._track_buckets))
            self._track_buckets.pop(oldest, None)
            self._track_motion.pop(oldest, None)
            self._track_signatures.pop(oldest, None)
        while len(self._signature_buckets) > limit:
            oldest = next(iter(self._signature_buckets))
            self._signature_buckets.pop(oldest, None)
            self._signature_motion.pop(oldest, None)

    def observe(
        self,
        *,
        action: Action,
        events: Sequence[WorldEvent],
        visible_objects: Sequence[ObjectState],
    ) -> int:
        """Attribute one committed transition; returns updated track rows."""

        if not self.config.enabled or action.has_position:
            return 0
        moved: dict[int, tuple[float, float]] = {}
        disappeared: set[int] = set()
        for event in events:
            kind = getattr(event.kind, "value", event.kind)
            if kind == "moved":
                dx = _state_number(
                    event.metadata.get("dx", 0.0),
                    "ego movement dx",
                )
                dy = _state_number(
                    event.metadata.get("dy", 0.0),
                    "ego movement dy",
                )
                moved[int(event.subject_track_id)] = (dx, dy)
            elif kind == "disappeared":
                disappeared.add(int(event.subject_track_id))

        updated = 0
        epsilon = self.config.displacement_epsilon
        for obj in visible_objects:
            track_id = int(obj.track_id)
            if track_id in disappeared:
                continue
            signature = str(obj.signature)
            known_signature = self._track_signatures.get(track_id)
            if known_signature is not None and known_signature != signature:
                # A tracker may recycle an id after pruning/transformation.
                # Never carry instance-level causal counts across identities.
                self._track_buckets.pop(track_id, None)
                self._track_motion.pop(track_id, None)
            self._track_signatures[track_id] = signature
            dx, dy = moved.get(track_id, (0.0, 0.0))
            bucket = _bucket(dx, dy, epsilon)
            bucket_row = self._row(
                self._track_buckets, track_id, action.index, _BUCKETS
            )
            bucket_row[bucket] += 1.0
            motion_row = self._row(self._track_motion, track_id, action.index, 3)
            motion_row[0] += dx
            motion_row[1] += dy
            motion_row[2] += 1.0
            if signature:
                self._row(
                    self._signature_buckets, signature, action.index, _BUCKETS
                )[bucket] += 1.0
                sig_motion = self._row(
                    self._signature_motion, signature, action.index, 3
                )
                sig_motion[0] += dx
                sig_motion[1] += dy
                sig_motion[2] += 1.0
            updated += 1
        if updated:
            self.update_count += 1
            self._prune()
        return updated

    def _influence_from(
        self,
        joint: Mapping[int, Sequence[float]] | None,
    ) -> float:
        if not joint:
            return 0.0
        support = float(sum(sum(row) for row in joint.values()))
        trust = support / (support + float(max(1, self.config.min_support)))
        return _normalized_mutual_information(joint) * trust

    def influence(self, track_id: int, signature: str = "") -> float:
        """Influence for a live track, with signature-level bootstrap."""

        stored_signature = self._track_signatures.get(int(track_id))
        track_level = (
            0.0
            if signature and stored_signature not in (None, str(signature))
            else self._influence_from(self._track_buckets.get(int(track_id)))
        )
        signature_level = 0.0
        if signature:
            signature_level = float(
                self.config.signature_bootstrap_scale
            ) * self._influence_from(self._signature_buckets.get(str(signature)))
        return float(np.clip(max(track_level, signature_level), 0.0, 1.0))

    def controlled_objects(
        self,
        objects: Sequence[ObjectState],
    ) -> tuple[ObjectState, ...]:
        if not self.config.enabled:
            return ()
        threshold = float(self.config.influence_threshold)
        return tuple(
            obj
            for obj in objects
            if self.influence(obj.track_id, obj.signature) >= threshold
        )

    def enrich(self, objects: Sequence[ObjectState]) -> tuple[ObjectState, ...]:
        """Ground ``controllable`` with attribution evidence; never lowers it."""

        if not self.config.enabled or not objects:
            return tuple(objects)
        enriched: list[ObjectState] = []
        for obj in objects:
            influence = self.influence(obj.track_id, obj.signature)
            if influence > float(obj.controllable):
                enriched.append(replace(obj, controllable=float(influence)))
            else:
                enriched.append(obj)
        return tuple(enriched)

    def _predicted_motion(
        self,
        track_id: int,
        signature: str,
        action_index: int,
    ) -> tuple[float, float, float] | None:
        """Mean displacement and modal-bucket consistency for one action."""

        motion = self._track_motion.get(int(track_id), {}).get(int(action_index))
        buckets = self._track_buckets.get(int(track_id), {}).get(int(action_index))
        if (motion is None or motion[2] < self.config.min_support) and signature:
            motion = self._signature_motion.get(str(signature), {}).get(
                int(action_index)
            )
            buckets = self._signature_buckets.get(str(signature), {}).get(
                int(action_index)
            )
        if motion is None or buckets is None or motion[2] < self.config.min_support:
            return None
        count = float(motion[2])
        consistency = float(max(buckets) / max(sum(buckets), 1.0))
        return motion[0] / count, motion[1] / count, consistency

    def controlled_motion_prediction(
        self,
        obj: ObjectState,
        action: Action,
    ) -> tuple[float, float, float] | None:
        """Return a supported motion estimate for a controlled object.

        The estimate is read-only evidence from committed transitions:
        ``(mean_dx, mean_dy, modal_direction_consistency)``.  Positional
        actions and objects below the configured causal-control threshold
        deliberately return ``None``.  This narrow public view lets other
        reasoning components use the ego model without reaching into its
        mutable statistics.
        """

        if not self.config.enabled or action.has_position:
            return None
        if (
            self.influence(obj.track_id, obj.signature)
            < float(self.config.influence_threshold)
        ):
            return None
        prediction = self._predicted_motion(
            obj.track_id,
            obj.signature,
            action.index,
        )
        if prediction is None:
            return None
        dx, dy, consistency = prediction
        if not all(math.isfinite(value) for value in (dx, dy, consistency)):
            return None
        return float(dx), float(dy), float(np.clip(consistency, 0.0, 1.0))

    def motion_hazard(self, snapshot: WorldSnapshot, action: Action) -> float:
        """Bounded [0, 1] reading: predicted ego motion into believed hazard."""

        if not self.config.enabled or action.has_position:
            return 0.0
        controlled = self.controlled_objects(snapshot.objects)
        if not controlled:
            return 0.0
        hazard_floor = float(self.config.hazard_belief_floor)
        hazards = [
            obj
            for obj in snapshot.objects
            if float(obj.hazard) >= hazard_floor
        ]
        if not hazards:
            return 0.0
        best = 0.0
        for ego in controlled:
            prediction = self.controlled_motion_prediction(ego, action)
            if prediction is None:
                continue
            dx, dy, consistency = prediction
            if math.hypot(dx, dy) <= self.config.displacement_epsilon:
                continue
            shift_x, shift_y = int(round(dx)), int(round(dy))
            x0, y0, x1, y1 = ego.bbox
            predicted = (x0 + shift_x, y0 + shift_y, x1 + shift_x, y1 + shift_y)
            influence = self.influence(ego.track_id, ego.signature)
            for hazard_object in hazards:
                if hazard_object.track_id == ego.track_id:
                    continue
                hx0, hy0, hx1, hy1 = hazard_object.bbox
                gap_x = max(hx0 - predicted[2], predicted[0] - hx1, 0)
                gap_y = max(hy0 - predicted[3], predicted[1] - hy1, 0)
                if max(gap_x, gap_y) > 1:
                    continue
                signal = influence * consistency * float(hazard_object.hazard)
                best = max(best, signal)
        return float(np.clip(best, 0.0, 1.0))

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "updates": int(self.update_count),
            "tracked": len(self._track_buckets),
            "signatures": len(self._signature_buckets),
        }

    def state_dict(self) -> dict[str, Any]:
        def _encode(store: Mapping[Any, Mapping[int, Sequence[float]]]) -> dict:
            return {
                str(key): {
                    str(action): [float(v) for v in row]
                    for action, row in per_action.items()
                }
                for key, per_action in store.items()
            }

        return {
            "track_buckets": _encode(self._track_buckets),
            "track_motion": _encode(self._track_motion),
            "track_signatures": {
                str(track_id): signature
                for track_id, signature in self._track_signatures.items()
            },
            "signature_buckets": _encode(self._signature_buckets),
            "signature_motion": _encode(self._signature_motion),
            "active_task_id": self._active_task_id,
            "update_count": int(self.update_count),
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        config: EgoConfig | None = None,
    ) -> "ControlAttribution":
        if not isinstance(state, Mapping):
            raise ValueError("ego state must be a mapping")
        model = cls(config)
        required = {
            "track_buckets",
            "track_motion",
            "track_signatures",
            "signature_buckets",
            "signature_motion",
            "active_task_id",
            "update_count",
        }
        if not required.issubset(state):
            missing = sorted(required - set(state))
            raise ValueError(f"ego state is missing fields {missing!r}")

        def _decode_store(
            raw: Any,
            *,
            name: str,
            string_keys: bool,
            width: int,
            bucket_counts: bool,
        ) -> dict[Any, dict[int, list[float]]]:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{name} must be a mapping")
            decoded: dict[Any, dict[int, list[float]]] = {}
            for raw_key, raw_actions in raw.items():
                if string_keys:
                    if not isinstance(raw_key, str) or not raw_key:
                        raise ValueError(
                            f"{name} keys must be nonempty strings"
                        )
                    key: Any = raw_key
                else:
                    key = _state_key_int(raw_key, f"{name} key")
                if key in decoded:
                    raise ValueError(f"duplicate {name} key {key!r}")
                if not isinstance(raw_actions, Mapping):
                    raise ValueError(
                        f"{name}[{raw_key!r}] must be a mapping"
                    )
                actions: dict[int, list[float]] = {}
                for raw_action, raw_row in raw_actions.items():
                    action = _state_key_int(
                        raw_action,
                        f"{name} action key",
                    )
                    if action in actions:
                        raise ValueError(
                            f"duplicate {name} action {action}"
                        )
                    row_values = _state_sequence(
                        raw_row,
                        f"{name}[{raw_key!r}][{raw_action!r}]",
                    )
                    if len(row_values) != width:
                        raise ValueError(
                            f"{name} rows must have width {width}"
                        )
                    row: list[float] = []
                    for index, value in enumerate(row_values):
                        if bucket_counts or index == width - 1:
                            row.append(
                                _state_number(
                                    value,
                                    f"{name} count",
                                    minimum=0.0,
                                    integral=True,
                                )
                            )
                        else:
                            row.append(
                                _state_number(
                                    value,
                                    f"{name} motion",
                                )
                            )
                    if row[-1] <= 0.0 and not bucket_counts:
                        raise ValueError(
                            f"{name} motion support must be positive"
                        )
                    if bucket_counts and sum(row) <= 0.0:
                        raise ValueError(
                            f"{name} bucket support must be positive"
                        )
                    actions[action] = row
                if not actions:
                    raise ValueError(
                        f"{name}[{raw_key!r}] must contain an action"
                    )
                decoded[key] = actions
            return decoded

        track_buckets = _decode_store(
            state["track_buckets"],
            name="ego track_buckets",
            string_keys=False,
            width=_BUCKETS,
            bucket_counts=True,
        )
        track_motion = _decode_store(
            state["track_motion"],
            name="ego track_motion",
            string_keys=False,
            width=3,
            bucket_counts=False,
        )
        signature_buckets = _decode_store(
            state["signature_buckets"],
            name="ego signature_buckets",
            string_keys=True,
            width=_BUCKETS,
            bucket_counts=True,
        )
        signature_motion = _decode_store(
            state["signature_motion"],
            name="ego signature_motion",
            string_keys=True,
            width=3,
            bucket_counts=False,
        )

        def _validate_store_pair(
            buckets: Mapping[Any, Mapping[int, Sequence[float]]],
            motion: Mapping[Any, Mapping[int, Sequence[float]]],
            name: str,
        ) -> None:
            if set(buckets) != set(motion):
                raise ValueError(f"{name} bucket and motion keys differ")
            for key, action_buckets in buckets.items():
                action_motion = motion[key]
                if set(action_buckets) != set(action_motion):
                    raise ValueError(
                        f"{name} action keys differ for {key!r}"
                    )
                for action, bucket_row in action_buckets.items():
                    support = float(sum(bucket_row))
                    motion_support = float(action_motion[action][2])
                    if support != motion_support:
                        raise ValueError(
                            f"{name} support differs for {key!r}, action {action}"
                        )

        _validate_store_pair(
            track_buckets,
            track_motion,
            "ego track",
        )
        _validate_store_pair(
            signature_buckets,
            signature_motion,
            "ego signature",
        )

        raw_track_signatures = state["track_signatures"]
        if not isinstance(raw_track_signatures, Mapping):
            raise ValueError("ego track_signatures must be a mapping")
        track_signatures: dict[int, str] = {}
        for raw_track_id, raw_signature in raw_track_signatures.items():
            track_id = _state_key_int(
                raw_track_id,
                "ego track_signatures key",
            )
            if not isinstance(raw_signature, str):
                raise ValueError("ego track signature must be a string")
            track_signatures[track_id] = raw_signature
        if set(track_signatures) != set(track_buckets):
            raise ValueError(
                "ego track signatures must exactly cover tracked ids"
            )
        if len(track_buckets) > int(model.config.max_tracked):
            raise ValueError("ego track state exceeds max_tracked")
        if len(signature_buckets) > int(model.config.max_tracked):
            raise ValueError("ego signature state exceeds max_tracked")

        active_task_id = state.get("active_task_id")
        if active_task_id is not None and not isinstance(active_task_id, str):
            raise ValueError("ego active_task_id must be a string or null")
        update_count = _state_int(
            state["update_count"],
            "ego update_count",
            minimum=0,
        )
        if update_count == 0 and (track_buckets or signature_buckets):
            raise ValueError("ego statistics require a positive update_count")

        model._track_buckets = track_buckets
        model._track_motion = track_motion
        model._track_signatures = track_signatures
        model._signature_buckets = signature_buckets
        model._signature_motion = signature_motion
        model._active_task_id = active_task_id
        model.update_count = update_count
        return model


__all__ = [
    "ControlAttribution",
]
