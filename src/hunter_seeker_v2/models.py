"""Compact representation, online dynamics ensemble, priors, and competence.

The default implementation is deliberately small and CPU-friendly.  It is a
real trainable route with observable targets, while the representation backend
can later be replaced by frozen Ouro loop-state spatial taps without changing search
or runtime contracts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    Action,
    CompetenceState,
    ModelConfig,
    ObjectState,
    Observation,
    Prediction,
    Representation,
    RepresentationBackend,
    Topology,
    Transition,
    WorldSnapshot,
)
from .memory import object_summary


_OBJECT_DIM = 8
_ACTION_HASH_DIM = 8
_COORD_DIM = 4
_TOPOLOGY_DIM = 5
_SCALAR_OUTPUTS = 5


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _finite_vector(value: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)
    if arr.size == size:
        return arr
    result = np.zeros(size, dtype=np.float32)
    result[: min(size, arr.size)] = arr[: min(size, arr.size)]
    return result


def _loaded_float(
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


def _loaded_int(
    value: Any,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _loaded_action_index(value: Any, *, name: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical integer string")
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical integer string") from exc
    if result < 0 or value != str(result):
        raise ValueError(f"{name} must be a canonical non-negative integer string")
    return result


def _loaded_action_key(value: Any, *, name: str) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical action key")
    pieces = value.split(",")
    if len(pieces) != 3:
        raise ValueError(f"{name} must be a canonical action key")
    try:
        action = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical action key") from exc
    if value != ",".join(str(item) for item in action) or action[0] < 0:
        raise ValueError(f"{name} must be a canonical action key")
    return action


def _loaded_array(value: Any, *, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(
            item,
            (int, float, np.integer, np.floating),
        )
        for item in raw.reshape(-1)
    ):
        raise ValueError(f"{name} must contain only numbers")
    result = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _loaded_pair(value: Any, *, name: str) -> tuple[float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ValueError(f"{name} must contain count and total")
    count = _loaded_float(value[0], name=f"{name}.count", minimum=0.0)
    total = _loaded_float(value[1], name=f"{name}.total")
    if count == 0.0 and total != 0.0:
        raise ValueError(f"{name} cannot have a total without support")
    return count, total


def _loaded_affordance_row(
    value: Any,
    *,
    name: str,
    transferable: bool,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    row = {
        "count": _loaded_float(
            value.get("count", 0.0),
            name=f"{name}.count",
            minimum=0.0,
        ),
        "changed": _loaded_float(
            value.get("changed", 0.0),
            name=f"{name}.changed",
            minimum=0.0,
        ),
        "hazard": _loaded_float(
            value.get("hazard", 0.0),
            name=f"{name}.hazard",
            minimum=0.0,
        ),
        "reward": _loaded_float(
            value.get("reward", 0.0),
            name=f"{name}.reward",
        ),
        "terminal": _loaded_float(
            value.get("terminal", 0.0),
            name=f"{name}.terminal",
            minimum=0.0,
        ),
    }
    if row["count"] == 0.0 and any(
        row[field] != 0.0
        for field in ("changed", "hazard", "reward", "terminal")
    ):
        raise ValueError(f"{name} cannot contain outcomes without support")
    tolerance = 1e-9 * max(1.0, row["count"])
    for field in ("changed", "hazard", "terminal"):
        if row[field] > row["count"] + tolerance:
            raise ValueError(f"{name}.{field} cannot exceed count")
    if transferable and (row["hazard"] != 0.0 or row["terminal"] != 0.0):
        raise ValueError(f"{name} contains task-scoped safety evidence")
    if transferable and row["reward"] < 0.0:
        raise ValueError(f"{name} contains negative transferable reward")
    return row


def topology_vector(topology: Topology) -> np.ndarray:
    return np.asarray(
        [
            min(float(topology.component_count) / 16.0, 1.0),
            float(np.clip(topology.largest_component_fraction, 0.0, 1.0)),
            float(np.clip(topology.frontier_fraction, 0.0, 1.0)),
            min(len(topology.object_adjacencies) / 32.0, 1.0),
            min(len(topology.reachable_object_ids) / 32.0, 1.0),
        ],
        dtype=np.float32,
    )


class GridFeatureBackend(RepresentationBackend):
    """Deterministic categorical-grid representation.

    It preserves a small spatial map and a fixed global vector.  This backend is
    not presented as a universal vision model; it is the compact, fully
    testable fallback for ARC-like categorical domains.
    """

    transactionally_stateless = True

    # 32x32 is a measured resolution choice, not aesthetics: on the trusted
    # trio, 8x8 pooling of 64x64 boards averaged away decision-relevant cells
    # (student recorded-state fit ls20 48->50/54, tr87 87->89/106, wa30
    # 749->917/1556 when raised to 32x32, no game regressing), while full
    # 64x64 resolution regressed ls20 to 37/54 because timer pixels dominate
    # unpooled distances even though it lifted tr87 to 104/106.
    def __init__(
        self,
        *,
        latent_dim: int = 32,
        spatial_shape: tuple[int, int] = (32, 32),
        histogram_bins: int = 16,
    ) -> None:
        self.latent_dim = max(8, int(latent_dim))
        self.spatial_shape = (
            max(1, int(spatial_shape[0])),
            max(1, int(spatial_shape[1])),
        )
        self.histogram_bins = max(4, int(histogram_bins))

    def _spatial_pool(self, frame: np.ndarray) -> np.ndarray:
        grid = np.asarray(frame, dtype=np.float32)
        if grid.ndim != 2:
            raise ValueError(f"categorical frame must be 2D, got {grid.shape}")
        row_groups = np.array_split(np.arange(grid.shape[0]), self.spatial_shape[0])
        col_groups = np.array_split(np.arange(grid.shape[1]), self.spatial_shape[1])
        scale = max(float(np.max(np.abs(grid))) if grid.size else 0.0, 1.0)
        pooled = np.zeros(self.spatial_shape, dtype=np.float32)
        for row_index, rows in enumerate(row_groups):
            for col_index, cols in enumerate(col_groups):
                if rows.size and cols.size:
                    pooled[row_index, col_index] = float(
                        np.mean(grid[np.ix_(rows, cols)]) / scale
                    )
        return pooled

    def encode(
        self,
        observation: Observation,
        objects: Sequence[ObjectState],
    ) -> Representation:
        frame = np.asarray(observation.frame)
        if frame.ndim != 2:
            raise ValueError(f"categorical frame must be 2D, got {frame.shape}")
        flat = frame.reshape(-1).astype(np.int64, copy=False)
        clipped = np.clip(flat, 0, self.histogram_bins - 1)
        histogram = np.bincount(clipped, minlength=self.histogram_bins).astype(np.float32)
        histogram /= max(float(histogram.sum()), 1.0)
        spatial = self._spatial_pool(frame)
        height, width = frame.shape
        if flat.size:
            values, counts = np.unique(flat, return_counts=True)
            dominant_fraction = float(np.max(counts) / flat.size)
            unique_fraction = float(min(len(values) / self.histogram_bins, 1.0))
        else:
            dominant_fraction = 1.0
            unique_fraction = 0.0
        gradients_x = (
            float(np.mean(frame[:, 1:] != frame[:, :-1])) if width > 1 else 0.0
        )
        gradients_y = (
            float(np.mean(frame[1:, :] != frame[:-1, :])) if height > 1 else 0.0
        )
        shape_and_texture = np.asarray(
            [
                min(height / 64.0, 2.0),
                min(width / 64.0, 2.0),
                float(height / max(width, 1)),
                dominant_fraction,
                unique_fraction,
                gradients_x,
                gradients_y,
                float(np.std(frame) / max(float(np.max(np.abs(frame))), 1.0)),
            ],
            dtype=np.float32,
        )
        features = np.concatenate(
            [
                histogram,
                shape_and_texture,
                object_summary(objects),
                spatial.reshape(-1),
            ]
        )
        # Deterministic folding preserves contributions from the full vector
        # without introducing a learned encoder into the baseline.
        latent = np.zeros(self.latent_dim, dtype=np.float32)
        for index, value in enumerate(features):
            latent[index % self.latent_dim] += float(value)
        counts = np.bincount(
            np.arange(features.size) % self.latent_dim,
            minlength=self.latent_dim,
        )
        latent /= np.maximum(counts, 1)
        return Representation(
            global_vector=latent,
            spatial=spatial,
            taps={},
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "latent_dim": self.latent_dim,
            "spatial_shape": list(self.spatial_shape),
            "histogram_bins": self.histogram_bins,
        }


class ActionPrior:
    """Small empirical prior used for proposals, never as a teacher lookup.

    ``_global`` contains *only* explicitly transferable positive support.
    Task rows contain the complete signed outcome history.  This split is
    important: a death attached to action ``1`` in one game must not make
    action ``1`` unattractive in every other game merely because canonical
    action indexes happen to coincide.
    """

    def __init__(self) -> None:
        self._global: dict[int, list[float]] = {}
        self._task: dict[tuple[str, int], list[float]] = {}

    @staticmethod
    def _update_row(row: list[float], target: float) -> None:
        row[0] += 1.0
        row[1] += float(target)

    def observe(
        self,
        transition: Transition,
        *,
        empirical: bool = True,
        transferable: bool = False,
    ) -> None:
        if not empirical:
            return
        target = float(
            transition.outcome.reward
            + transition.outcome.progress_delta
            - transition.outcome.hazard
            - float(transition.outcome.failed)
        )
        task_row = self._task.setdefault(
            (transition.task_id, int(transition.action.index)),
            [0.0, 0.0],
        )
        self._update_row(task_row, target)
        if transferable and target > 0.0:
            global_row = self._global.setdefault(
                int(transition.action.index), [0.0, 0.0]
            )
            self._update_row(global_row, target)

    def observe_label(
        self,
        *,
        task_id: str,
        action_index: int,
        weight: float = 1.0,
        transferable: bool = False,
    ) -> None:
        """Distill a teacher label into the actual runtime prior."""

        weight = float(max(0.0, weight))
        if weight == 0.0:
            return
        task_row = self._task.setdefault(
            (str(task_id), int(action_index)),
            [0.0, 0.0],
        )
        task_row[0] += weight
        task_row[1] += weight
        if transferable:
            global_row = self._global.setdefault(int(action_index), [0.0, 0.0])
            global_row[0] += weight
            global_row[1] += weight

    def score(self, task_id: str, action_index: int) -> float:
        global_count, global_total = self._global.get(int(action_index), [0.0, 0.0])
        task_count, task_total = self._task.get(
            (str(task_id), int(action_index)),
            [0.0, 0.0],
        )
        global_mean = global_total / max(global_count, 1.0)
        task_mean = task_total / max(task_count, 1.0)
        if task_count <= 0.0:
            return float(np.tanh(global_mean))
        # Local adverse evidence is authoritative.  Transferable positive
        # support may bootstrap an unseen/neutral task, but cannot dilute a
        # task's own observed failure.
        if task_mean < 0.0:
            return float(np.tanh(task_mean))
        trust = task_count / (task_count + 3.0)
        return float(np.tanh(trust * task_mean + (1.0 - trust) * global_mean))

    def state_dict(self) -> dict[str, Any]:
        return {
            "scope_semantics": 2,
            "global": {str(k): list(v) for k, v in self._global.items()},
            "task": {
                f"{task}\u241f{action}": list(value)
                for (task, action), value in self._task.items()
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ActionPrior":
        prior = cls()
        semantics = _loaded_int(
            state.get("scope_semantics", 1),
            name="action prior scope_semantics",
            minimum=1,
            maximum=2,
        )
        raw_global = state.get("global", {})
        raw_task = state.get("task", {})
        if not isinstance(raw_global, Mapping) or not isinstance(raw_task, Mapping):
            raise ValueError("action prior rows must be mappings")
        for key, value in raw_global.items():
            count, total = _loaded_pair(
                value,
                name=f"action prior global row {key!r}",
            )
            if semantics < 2:
                # Legacy checkpoints mixed scoped outcomes into the global
                # row and did not record transfer authorization.  Task rows
                # retain the evidence; the unsafe global projection is
                # intentionally discarded.
                continue
            action_index = _loaded_action_index(
                key,
                name="action prior global action key",
            )
            if count > 0.0 and total <= 0.0:
                raise ValueError(
                    "transferable action prior rows require positive total support"
                )
            if count > 0.0:
                prior._global[action_index] = [count, total]
        for key, value in raw_task.items():
            if not isinstance(key, str):
                raise ValueError("action prior task row key is malformed")
            key_text = key
            if "\u241f" not in key_text:
                raise ValueError("action prior task row key is malformed")
            task, action = key_text.rsplit("\u241f", 1)
            count, total = _loaded_pair(
                value,
                name=f"action prior task row {key!r}",
            )
            action_index = _loaded_action_index(
                action,
                name="action prior task action key",
            )
            prior._task[(task, action_index)] = [count, total]
        return prior


class AffordanceModel:
    """Outcome-grounded object affordances keyed by shape/value signature.

    The legacy eight-head objectivity module is compressed to the three
    policy-relevant calibrated quantities that have direct environment labels:
    controllability/change, hazard, and progress/reward.  Raw counters remain
    available for diagnostics and can later supervise a richer head.
    """

    def __init__(self) -> None:
        # Transferable rows intentionally contain positive/change evidence
        # only.  Hazard and adverse-terminal evidence lives in task rows.
        self._rows: dict[str, dict[str, float]] = {}
        self._task_rows: dict[tuple[str, str], dict[str, float]] = {}

    @staticmethod
    def _new_row() -> dict[str, float]:
        return {
            "count": 0.0,
            "changed": 0.0,
            "hazard": 0.0,
            "reward": 0.0,
            "terminal": 0.0,
        }

    @staticmethod
    def _accumulate(
        row: dict[str, float],
        *,
        changed: float,
        hazard: float,
        reward: float,
        terminal: float,
        weight: float,
    ) -> None:
        row["count"] += weight
        row["changed"] += weight * changed
        row["hazard"] += weight * hazard
        row["reward"] += weight * reward
        row["terminal"] += weight * terminal

    def observe_signature(
        self,
        signature: str,
        *,
        task_id: str | None = None,
        changed: float = 0.0,
        hazard: float = 0.0,
        reward: float = 0.0,
        terminal: float = 0.0,
        weight: float = 1.0,
        empirical: bool = True,
        transferable_positive: bool = False,
    ) -> None:
        """Attribute one outcome observation to an object signature.

        This is the single grounding route shared by click-target attribution
        and ego-contact attribution.
        """

        if not empirical or not signature or float(weight) <= 0.0:
            return
        signature = str(signature)
        weight = float(weight)
        task_row = self._task_rows.setdefault(
            ("" if task_id is None else str(task_id), signature),
            self._new_row(),
        )
        self._accumulate(
            task_row,
            changed=float(changed),
            hazard=float(hazard),
            reward=float(reward),
            terminal=float(terminal),
            weight=weight,
        )
        if transferable_positive and (float(changed) > 0.0 or float(reward) > 0.0):
            positive_row = self._rows.setdefault(signature, self._new_row())
            self._accumulate(
                positive_row,
                changed=max(0.0, float(changed)),
                hazard=0.0,
                reward=max(0.0, float(reward)),
                terminal=0.0,
                weight=weight,
            )

    def observe(
        self,
        *,
        object_signature: str,
        transition: Transition,
        empirical: bool = True,
        transferable_positive: bool = False,
    ) -> None:
        self.observe_signature(
            object_signature,
            task_id=transition.task_id,
            changed=float(transition.frame_changed),
            hazard=float(
                max(
                    transition.outcome.hazard,
                    1.0 if transition.outcome.failed else 0.0,
                )
            ),
            reward=float(
                max(
                    0.0,
                    transition.outcome.reward + transition.outcome.progress_delta,
                )
            ),
            terminal=float(
                transition.outcome.terminated and not transition.outcome.completed
            ),
            empirical=empirical,
            transferable_positive=transferable_positive,
        )

    @staticmethod
    def _belief(row: Mapping[str, float] | None) -> tuple[float, float, float, float]:
        if row is None:
            return 0.0, 0.0, 0.0, 0.0
        count = max(float(row["count"]), 1.0)
        trust = float(row["count"] / (row["count"] + 2.0))
        controllable = trust * float(row["changed"] / count)
        hazard = trust * float(row["hazard"] / count)
        rewarding = trust * float(np.tanh(row["reward"] / count))
        terminal = trust * float(row["terminal"] / count)
        return controllable, hazard, rewarding, terminal

    def belief(
        self,
        signature: str,
        *,
        task_id: str | None = None,
        allow_positive_transfer: bool = True,
    ) -> tuple[float, float, float, float]:
        signature = str(signature)
        local = self._belief(
            self._task_rows.get(("" if task_id is None else str(task_id), signature))
        )
        transferable = (
            self._belief(self._rows.get(signature))
            if allow_positive_transfer
            else (0.0, 0.0, 0.0, 0.0)
        )
        return (
            max(local[0], transferable[0]),
            local[1],
            max(local[2], transferable[2]),
            local[3],
        )

    def enrich(
        self,
        objects: Sequence[ObjectState],
        *,
        task_id: str | None = None,
        allow_positive_transfer: bool = True,
    ) -> tuple[ObjectState, ...]:
        enriched: list[ObjectState] = []
        for obj in objects:
            controllable, hazard, rewarding, _terminal = self.belief(
                obj.signature,
                task_id=task_id,
                allow_positive_transfer=allow_positive_transfer,
            )
            enriched.append(
                replace(
                    obj,
                    controllable=float(controllable),
                    hazard=float(hazard),
                    rewarding=float(rewarding),
                )
            )
        return tuple(enriched)

    def state_dict(self) -> dict[str, Any]:
        return {
            "scope_semantics": 2,
            "transferable_positive": {
                str(signature): {str(k): float(v) for k, v in row.items()}
                for signature, row in self._rows.items()
            },
            "task": {
                f"{task}\u241f{signature}": {
                    str(k): float(v) for k, v in row.items()
                }
                for (task, signature), row in self._task_rows.items()
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "AffordanceModel":
        model = cls()
        semantics = _loaded_int(
            state.get("scope_semantics", 1),
            name="affordance scope_semantics",
            minimum=1,
            maximum=2,
        )
        if semantics >= 2:
            raw_positive = state.get("transferable_positive", {})
            raw_task = state.get("task", {})
            if not isinstance(raw_positive, Mapping) or not isinstance(
                raw_task,
                Mapping,
            ):
                raise ValueError("affordance rows must be mappings")
            model._rows = {
                str(signature): _loaded_affordance_row(
                    row,
                    name=f"transferable affordance {signature!r}",
                    transferable=True,
                )
                for signature, row in raw_positive.items()
            }
            for key, row in raw_task.items():
                if not isinstance(key, str):
                    raise ValueError("affordance task row key is malformed")
                key_text = key
                if "\u241f" not in key_text:
                    raise ValueError("affordance task row key is malformed")
                task, signature = key_text.rsplit("\u241f", 1)
                model._task_rows[(task, signature)] = _loaded_affordance_row(
                    row,
                    name=f"task affordance {key_text!r}",
                    transferable=False,
                )
            return model

        # Legacy rows were globally scoped.  Preserve their positive/change
        # information as transferable, but quarantine hazard/terminal values
        # in the legacy unscoped view rather than leaking them to named tasks.
        for signature, raw in state.items():
            if str(signature) == "scope_semantics":
                continue
            row = _loaded_affordance_row(
                raw,
                name=f"legacy affordance {signature!r}",
                transferable=False,
            )
            model._task_rows[("", str(signature))] = dict(row)
            positive = model._new_row()
            positive["count"] = float(row.get("count", 0.0))
            positive["changed"] = max(0.0, float(row.get("changed", 0.0)))
            positive["reward"] = max(0.0, float(row.get("reward", 0.0)))
            if positive["changed"] > 0.0 or positive["reward"] > 0.0:
                model._rows[str(signature)] = positive
        return model


class DynamicsEnsemble:
    """Bootstrap linear ensemble over planning-relevant transition targets."""

    object_dim = _OBJECT_DIM

    def __init__(self, config: ModelConfig | None = None, *, seed: int = 0) -> None:
        self.config = config or ModelConfig()
        self.latent_dim = int(self.config.latent_dim)
        self.feature_dim = (
            self.latent_dim
            + _OBJECT_DIM
            + _ACTION_HASH_DIM
            + _COORD_DIM
            + _TOPOLOGY_DIM
            + 1
        )
        self.output_dim = _SCALAR_OUTPUTS + self.latent_dim + _OBJECT_DIM
        self._rng = np.random.default_rng(int(seed))
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []
        for _ in range(max(1, int(self.config.ensemble_size))):
            self._weights.append(
                self._rng.normal(
                    0.0,
                    0.015,
                    size=(self.output_dim, self.feature_dim),
                ).astype(np.float32)
            )
            bias = np.zeros(self.output_dim, dtype=np.float32)
            bias[0] = 0.0
            bias[3] = -2.2
            bias[4] = -2.6
            self._biases.append(bias)
        self._support: dict[tuple[str, tuple[int, int, int]], int] = {}
        self.update_count = 0
        self.last_loss = 0.0
        self.learning_progress_ema = 0.0

    def _features(
        self,
        latent: np.ndarray,
        obj_summary: np.ndarray,
        action: Action,
        frame_shape: tuple[int, int],
        topology: Topology,
    ) -> np.ndarray:
        action_hash = np.zeros(_ACTION_HASH_DIM, dtype=np.float32)
        index = int(action.index)
        action_hash[index % _ACTION_HASH_DIM] = 1.0
        action_hash[(index * 5 + 3) % _ACTION_HASH_DIM] += 0.5
        height, width = frame_shape
        if action.has_position:
            x_norm = float(action.x / max(width - 1, 1))
            y_norm = float(action.y / max(height - 1, 1))
            coords = np.asarray(
                [x_norm, y_norm, x_norm * x_norm, y_norm * y_norm],
                dtype=np.float32,
            )
        else:
            coords = np.zeros(_COORD_DIM, dtype=np.float32)
        result = np.concatenate(
            [
                _finite_vector(latent, self.latent_dim),
                _finite_vector(obj_summary, _OBJECT_DIM),
                action_hash,
                coords,
                topology_vector(topology),
                np.ones(1, dtype=np.float32),
            ]
        )
        return result.astype(np.float32, copy=False)

    @staticmethod
    def _decode(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scalars = np.asarray(raw[:_SCALAR_OUTPUTS], dtype=np.float32).copy()
        scalars[0] = float(_sigmoid(scalars[0]))
        scalars[3] = float(_sigmoid(scalars[3]))
        scalars[4] = float(_sigmoid(scalars[4]))
        return (
            scalars,
            raw[_SCALAR_OUTPUTS : -_OBJECT_DIM],
            raw[-_OBJECT_DIM:],
        )

    def predict_from_features(
        self,
        *,
        latent: np.ndarray,
        obj_summary: np.ndarray,
        action: Action,
        frame_shape: tuple[int, int],
        topology: Topology,
        state_id: str = "",
    ) -> Prediction:
        features = self._features(latent, obj_summary, action, frame_shape, topology)
        decoded: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for weights, bias in zip(self._weights, self._biases):
            decoded.append(self._decode(weights @ features + bias))
        scalars = np.stack([row[0] for row in decoded])
        latent_deltas = np.stack([row[1] for row in decoded])
        object_deltas = np.stack([row[2] for row in decoded])
        mean_scalars = np.mean(scalars, axis=0)
        support = self._support.get((str(state_id), action.key), 0)
        disagreement = float(
            np.mean(np.std(scalars, axis=0))
            + 0.25 * np.mean(np.std(latent_deltas, axis=0))
        )
        uncertainty = float(
            np.clip(disagreement + 1.0 / np.sqrt(float(support) + 1.0), 0.0, 1.0)
        )
        return Prediction(
            change_probability=float(mean_scalars[0]),
            progress=float(mean_scalars[1]),
            value=float(mean_scalars[2]),
            hazard=float(mean_scalars[3]),
            terminal=float(mean_scalars[4]),
            uncertainty=uncertainty,
            latent_delta=np.mean(latent_deltas, axis=0),
            object_delta=np.mean(object_deltas, axis=0),
            source="ensemble",
        )

    def predict(self, snapshot: WorldSnapshot, action: Action) -> Prediction:
        return self.predict_from_features(
            latent=snapshot.representation.global_vector,
            obj_summary=object_summary(snapshot.objects),
            action=action,
            frame_shape=tuple(int(v) for v in snapshot.observation.frame.shape),
            topology=snapshot.topology,
            state_id=snapshot.state_id,
        )

    def _target(
        self,
        before: WorldSnapshot,
        after: WorldSnapshot,
        transition: Transition,
    ) -> np.ndarray:
        target = np.zeros(self.output_dim, dtype=np.float32)
        target[0] = float(transition.frame_changed)
        target[1] = float(transition.outcome.progress_delta)
        target[2] = float(
            transition.outcome.reward
            + transition.outcome.progress_delta
            - transition.outcome.hazard
        )
        target[3] = float(
            max(
                transition.outcome.hazard,
                1.0 if transition.outcome.failed else 0.0,
            )
        )
        target[4] = float(transition.outcome.terminated)
        if transition.outcome.completed:
            # Completion ends control flow but is not terminal *risk*.
            target[4] = 0.0
        before_latent = _finite_vector(
            before.representation.global_vector,
            self.latent_dim,
        )
        after_latent = _finite_vector(
            after.representation.global_vector,
            self.latent_dim,
        )
        target[_SCALAR_OUTPUTS : -_OBJECT_DIM] = after_latent - before_latent
        target[-_OBJECT_DIM:] = object_summary(after.objects) - object_summary(before.objects)
        return target

    @staticmethod
    def _prediction_loss(decoded: tuple[np.ndarray, np.ndarray, np.ndarray], target: np.ndarray) -> float:
        prediction = np.concatenate(decoded)
        return float(np.mean(np.square(prediction - target)))

    def update(
        self,
        before: WorldSnapshot,
        after: WorldSnapshot,
        transition: Transition,
        *,
        empirical: bool = True,
    ) -> float:
        features = self._features(
            before.representation.global_vector,
            object_summary(before.objects),
            transition.action,
            tuple(int(v) for v in before.observation.frame.shape),
            before.topology,
        )
        target = self._target(before, after, transition)
        losses_before: list[float] = []
        losses_after: list[float] = []
        updated = 0
        for member, (weights, bias) in enumerate(zip(self._weights, self._biases)):
            if (
                member > 0
                and self._rng.random() > float(self.config.bootstrap_probability)
            ):
                continue
            raw = weights @ features + bias
            decoded = self._decode(raw)
            losses_before.append(self._prediction_loss(decoded, target))
            prediction = np.concatenate(decoded)
            error = prediction - target
            # Probability outputs are decoded through sigmoid.
            for probability_index in (0, 3, 4):
                probability = prediction[probability_index]
                error[probability_index] *= probability * (1.0 - probability)
            error = np.clip(error, -5.0, 5.0)
            lr = float(self.config.learning_rate)
            weights -= lr * (
                np.outer(error, features)
                + float(self.config.l2) * weights
            ).astype(np.float32)
            bias -= lr * error.astype(np.float32)
            losses_after.append(
                self._prediction_loss(
                    self._decode(weights @ features + bias),
                    target,
                )
            )
            updated += 1
        if not losses_before:
            return 0.0
        before_loss = float(np.mean(losses_before))
        after_loss = float(np.mean(losses_after)) if losses_after else before_loss
        improvement = max(0.0, before_loss - after_loss)
        self.learning_progress_ema = (
            0.95 * self.learning_progress_ema + 0.05 * improvement
        )
        self.last_loss = after_loss
        self.update_count += int(updated > 0)
        if empirical:
            support_key = (before.state_id, transition.action.key)
            self._support[support_key] = self._support.get(support_key, 0) + 1
        return after_loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "weights": [value.tolist() for value in self._weights],
            "biases": [value.tolist() for value in self._biases],
            "support": {
                f"{state_id}\u241f{action[0]},{action[1]},{action[2]}": count
                for (state_id, action), count in self._support.items()
            },
            "update_count": self.update_count,
            "last_loss": self.last_loss,
            "learning_progress_ema": self.learning_progress_ema,
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any], *, seed: int = 0) -> "DynamicsEnsemble":
        model = cls(ModelConfig(**dict(state.get("config", {}))), seed=seed)
        weights = [
            _loaded_array(value, name="dynamics weights")
            for value in state.get("weights", ())
        ]
        biases = [
            _loaded_array(value, name="dynamics biases")
            for value in state.get("biases", ())
        ]
        if len(weights) != len(model._weights) or len(biases) != len(model._biases):
            raise ValueError("dynamics ensemble member count does not match checkpoint")
        for expected, loaded in zip(model._weights, weights):
            if expected.shape != loaded.shape:
                raise ValueError(
                    f"dynamics weight shape mismatch: {loaded.shape} != {expected.shape}"
                )
        for expected, loaded in zip(model._biases, biases):
            if expected.shape != loaded.shape:
                raise ValueError(
                    f"dynamics bias shape mismatch: {loaded.shape} != {expected.shape}"
                )
        model._weights = weights
        model._biases = biases
        raw_support = state.get("support", {})
        if not isinstance(raw_support, Mapping):
            raise ValueError("dynamics support must be a mapping")
        for key, count in raw_support.items():
            if not isinstance(key, str):
                raise ValueError("dynamics support key is malformed")
            key_text = key
            if "\u241f" not in key_text:
                raise ValueError("dynamics support key is malformed")
            state_id, action_text = key_text.rsplit("\u241f", 1)
            action = _loaded_action_key(
                action_text,
                name="dynamics support action key",
            )
            model._support[(state_id, action)] = _loaded_int(
                count,
                name=f"dynamics support {key_text!r}",
            )
        model.update_count = _loaded_int(
            state.get("update_count", 0),
            name="dynamics update_count",
        )
        model.last_loss = _loaded_float(
            state.get("last_loss", 0.0),
            name="dynamics last_loss",
            minimum=0.0,
        )
        model.learning_progress_ema = _loaded_float(
            state.get("learning_progress_ema", 0.0),
            name="dynamics learning_progress_ema",
            minimum=0.0,
        )
        rng_state = state.get("rng_state")
        if isinstance(rng_state, Mapping):
            try:
                model._rng.bit_generator.state = deepcopy(dict(rng_state))
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid dynamics RNG state") from exc
        return model


class CompetenceMonitor:
    """Operational self-state calibrated only against observed outcomes."""

    def __init__(self, *, initial_risk_budget: float = 1.0) -> None:
        self.state = CompetenceState(remaining_risk_budget=float(initial_risk_budget))
        self._updates = 0

    def update(
        self,
        *,
        prediction: Prediction,
        transition: Transition,
        model_loss: float,
        learning_progress: float,
        time_pressure: float = 0.0,
    ) -> CompetenceState:
        old = self.state
        realized_hazard = float(
            max(
                transition.outcome.hazard,
                1.0 if transition.outcome.failed else 0.0,
            )
        )
        hazard_error = abs(float(prediction.hazard) - realized_hazard)
        progress = float(transition.outcome.progress_delta)
        predicted_success = float(
            np.clip(
                prediction.progress
                + 0.5 * prediction.value
                - prediction.hazard,
                0.0,
                1.0,
            )
        )
        stagnation = 0 if progress > 0.0 else int(old.stagnation_count) + 1
        remaining_budget = float(
            np.clip(
                old.remaining_risk_budget
                - 0.15 * realized_hazard
                + 0.05 * float(transition.outcome.completed),
                0.0,
                1.0,
            )
        )
        alpha = 0.10 if self._updates else 1.0
        self.state = CompetenceState(
            dynamics_error_ema=(1.0 - alpha) * old.dynamics_error_ema
            + alpha * float(model_loss),
            hazard_calibration_error=(1.0 - alpha) * old.hazard_calibration_error
            + alpha * hazard_error,
            predicted_success=(1.0 - alpha) * old.predicted_success
            + alpha * predicted_success,
            recent_realized_progress=(1.0 - alpha) * old.recent_realized_progress
            + alpha * progress,
            model_disagreement=(1.0 - alpha) * old.model_disagreement
            + alpha * float(prediction.uncertainty),
            stagnation_count=stagnation,
            expected_learning_gain=(1.0 - alpha) * old.expected_learning_gain
            + alpha * float(np.clip(learning_progress, 0.0, 1.0)),
            remaining_risk_budget=remaining_budget,
            # A direct reading, not an EMA: the clock does not average.
            time_pressure=float(np.clip(time_pressure, 0.0, 1.0)),
        )
        self._updates += 1
        return self.state

    def begin_episode(self, *, initial_risk_budget: float = 1.0) -> CompetenceState:
        """Reset only episode-local operational readings.

        Calibration and learned-model EMAs remain durable across episodes;
        stagnation, spent risk, realized-progress recency, and clock pressure
        describe the active episode and therefore must not leak into the next.
        """

        old = self.state
        self.state = replace(
            old,
            recent_realized_progress=0.0,
            stagnation_count=0,
            remaining_risk_budget=float(np.clip(initial_risk_budget, 0.0, 1.0)),
            time_pressure=0.0,
        )
        return self.state

    def planning_horizon_adjustment(self) -> int:
        state = self.state
        if state.model_disagreement > 0.65 or state.dynamics_error_ema > 0.5:
            return -1
        if state.model_disagreement < 0.20 and state.predicted_success > 0.35:
            return 1
        return 0

    def state_dict(self) -> dict[str, Any]:
        return {"state": asdict(self.state), "updates": self._updates}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CompetenceMonitor":
        monitor = cls()
        raw = state.get("state", {})
        if not isinstance(raw, Mapping):
            raise ValueError("competence state must be a mapping")
        defaults = asdict(CompetenceState())
        unknown = set(raw) - set(defaults)
        if unknown:
            raise ValueError(
                f"competence state contains unknown fields: {sorted(unknown)!r}"
            )
        merged = {**defaults, **dict(raw)}
        monitor.state = CompetenceState(
            dynamics_error_ema=_loaded_float(
                merged["dynamics_error_ema"],
                name="competence dynamics_error_ema",
                minimum=0.0,
            ),
            hazard_calibration_error=_loaded_float(
                merged["hazard_calibration_error"],
                name="competence hazard_calibration_error",
                minimum=0.0,
                maximum=1.0,
            ),
            predicted_success=_loaded_float(
                merged["predicted_success"],
                name="competence predicted_success",
                minimum=0.0,
                maximum=1.0,
            ),
            recent_realized_progress=_loaded_float(
                merged["recent_realized_progress"],
                name="competence recent_realized_progress",
            ),
            model_disagreement=_loaded_float(
                merged["model_disagreement"],
                name="competence model_disagreement",
                minimum=0.0,
                maximum=1.0,
            ),
            stagnation_count=_loaded_int(
                merged["stagnation_count"],
                name="competence stagnation_count",
            ),
            expected_learning_gain=_loaded_float(
                merged["expected_learning_gain"],
                name="competence expected_learning_gain",
                minimum=0.0,
                maximum=1.0,
            ),
            remaining_risk_budget=_loaded_float(
                merged["remaining_risk_budget"],
                name="competence remaining_risk_budget",
                minimum=0.0,
                maximum=1.0,
            ),
            time_pressure=_loaded_float(
                merged["time_pressure"],
                name="competence time_pressure",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        monitor._updates = _loaded_int(
            state.get("updates", 0),
            name="competence updates",
        )
        return monitor


__all__ = [
    "ActionPrior",
    "AffordanceModel",
    "CompetenceMonitor",
    "DynamicsEnsemble",
    "GridFeatureBackend",
    "topology_vector",
]
