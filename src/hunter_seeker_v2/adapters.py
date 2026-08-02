"""Environment adapters for the compact Hunter-Seeker runtime.

The adapters convert environment-owned mutable objects into the immutable
contracts in :mod:`hunter_seeker_v2.contracts`.  This module deliberately does
not import the legacy Hunter-Seeker agent or any of its mixins.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contracts import Action, BoundaryKind, Observation, Outcome


_MISSING = object()


class AdapterError(ValueError):
    """Raised when an environment value cannot satisfy a canonical contract."""


class ObservationUnavailable(AdapterError):
    """Raised when no initial frame is available for an observation stream."""


def _field(raw: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(raw, Mapping):
        if name in raw:
            return raw[name]
    elif hasattr(raw, name):
        return getattr(raw, name)
    if default is _MISSING:
        raise AdapterError(f"observation has no {name!r} field")
    return default


def _optional_bool(raw: Any, name: str, default: bool = False) -> bool:
    value = _field(raw, name, default)
    return _strict_bool(value, name=name)


def _action_index(value: Any) -> int:
    native = getattr(value, "value", value)
    if isinstance(native, (bool, np.bool_)) or not isinstance(
        native,
        (int, np.integer),
    ):
        raise AdapterError(f"action value {value!r} is not an integer index")
    return int(native)


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise AdapterError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise AdapterError(f"{name} must be a bool, got {value!r}")
    return bool(value)


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{name} must be numeric, got {value!r}") from exc
    if not np.isfinite(result):
        raise AdapterError(f"{name} must be finite, got {value!r}")
    return result


def _categorical_frame(
    value: Any,
    *,
    n_values: int,
    pad_value: int,
) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise AdapterError(
            f"categorical frame must have shape [H, W], got {arr.shape!r}"
        )
    if arr.size == 0:
        raise AdapterError("categorical frame must not be empty")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.issubdtype(arr.dtype, np.floating):
            raise AdapterError(f"categorical frame dtype must be numeric, got {arr.dtype}")
        if not np.isfinite(arr).all() or not np.equal(arr, np.rint(arr)).all():
            raise AdapterError("categorical frame contains non-integral values")
        arr = np.rint(arr).astype(np.int64)
    if int(arr.min()) < 0:
        raise AdapterError("categorical frame contains a negative label")
    invalid_mask = (arr >= int(n_values)) & (arr != int(pad_value))
    if bool(np.any(invalid_mask)):
        invalid = int(arr[invalid_mask][0])
        raise AdapterError(
            f"categorical frame label {invalid} exceeds the configured "
            f"regular labels [0, {int(n_values)}) and is not pad value "
            f"{int(pad_value)}"
        )
    return np.ascontiguousarray(arr)


def _state_text(raw: Any, state_field: str) -> str:
    return str(_field(raw, state_field, "") or "")


@dataclass(slots=True)
class CategoricalObservationAdapter:
    """Adapter for a two-dimensional categorical/symbolic observation.

    Field names and action ontology are configurable.  No spatial dimension,
    action index, or environment-specific terminal convention is embedded in
    this generic path.
    """

    n_values: int
    pad_value: int
    n_actions: int
    frame_field: str = "grid"
    actions_field: str = "available_actions"
    progress_field: str = "progress"
    stage_field: str | None = "stage"
    state_field: str = "state"
    default_actions: tuple[int, ...] = ()
    frame_is_sequence: bool = False
    _last_frames: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.n_values = _strict_int(self.n_values, name="n_values")
        self.pad_value = _strict_int(self.pad_value, name="pad_value")
        self.n_actions = _strict_int(self.n_actions, name="n_actions")
        if self.n_values <= 0:
            raise AdapterError("n_values must be positive")
        if self.pad_value < 0:
            raise AdapterError("pad_value must be non-negative")
        if self.n_actions <= 0:
            raise AdapterError("n_actions must be positive")
        self.default_actions = tuple(_action_index(a) for a in self.default_actions)
        self._validate_actions(self.default_actions)

    def _validate_actions(self, actions: Sequence[int]) -> tuple[int, ...]:
        normalized = tuple(dict.fromkeys(_action_index(a) for a in actions))
        invalid = [a for a in normalized if a < 0 or a >= self.n_actions]
        if invalid:
            raise AdapterError(
                f"available action indices {invalid!r} fall outside "
                f"[0, {self.n_actions})"
            )
        return normalized

    def _extract_frame(self, raw: Any, task_id: str) -> tuple[np.ndarray, bool]:
        value = _field(raw, self.frame_field, None)
        if self.frame_is_sequence:
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray, np.ndarray)
            ):
                value = value[-1] if value else None
            elif value is not None:
                raise AdapterError(
                    f"{self.frame_field!r} must be a sequence of frames"
                )
        if value is None:
            cached = self._last_frames.get(str(task_id))
            if cached is None:
                raise ObservationUnavailable(
                    f"no frame is available yet for task {task_id!r}"
                )
            return cached, False
        frame = _categorical_frame(
            value,
            n_values=self.n_values,
            pad_value=self.pad_value,
        )
        return frame, True

    def observation(self, raw: Any, *, task_id: str) -> Observation:
        frame, frame_available = self._extract_frame(raw, str(task_id))
        raw_actions = _field(raw, self.actions_field, None)
        if raw_actions is None:
            actions = self.default_actions
        else:
            try:
                actions = tuple(_action_index(a) for a in raw_actions)
            except TypeError as exc:
                raise AdapterError(
                    f"{self.actions_field!r} must be an iterable of action indices"
                ) from exc
        actions = self._validate_actions(actions)

        progress = _finite_float(
            _field(raw, self.progress_field, 0.0) or 0.0,
            name=self.progress_field,
        )
        if self.stage_field is None:
            stage = int(progress) + 1
        else:
            raw_stage = _field(raw, self.stage_field, None)
            if raw_stage is None:
                stage = int(progress) + 1
            else:
                stage = _strict_int(
                    raw_stage,
                    name=f"{self.stage_field!r}",
                )
                if stage < 1:
                    raise AdapterError(
                        f"{self.stage_field!r} must be at least 1"
                    )
        state = _state_text(raw, self.state_field)
        observation = Observation(
            frame=frame,
            available_actions=actions,
            task_id=str(task_id),
            stage=max(1, stage),
            progress=progress,
            metadata={
                "state": state,
                "frame_available": bool(frame_available),
                "n_values": self.n_values,
                "pad_value": self.pad_value,
            },
        )
        if frame_available:
            # Commit the fallback cache only after every field has passed
            # validation and the immutable Observation has been constructed.
            self._last_frames[str(task_id)] = np.array(frame, copy=True)
        return observation

    def clear(self, task_id: str | None = None) -> None:
        if task_id is None:
            self._last_frames.clear()
        else:
            self._last_frames.pop(str(task_id), None)


class ArcObservationAdapter(CategoricalObservationAdapter):
    """ARC-AGI-3 observation adapter with no fixed frame dimensions."""

    def __init__(self) -> None:
        super().__init__(
            n_values=16,
            pad_value=16,
            n_actions=8,
            frame_field="frame",
            actions_field="available_actions",
            progress_field="levels_completed",
            stage_field=None,
            state_field="state",
            default_actions=(1, 2, 3, 4),
            frame_is_sequence=True,
        )


class MockObservationAdapter(CategoricalObservationAdapter):
    """Small clickless categorical adapter used by unit/integration tests."""

    def __init__(
        self,
        *,
        n_values: int = 8,
        n_actions: int = 4,
        pad_value: int | None = None,
    ) -> None:
        super().__init__(
            n_values=n_values,
            pad_value=n_values if pad_value is None else pad_value,
            n_actions=n_actions,
            default_actions=tuple(range(int(n_actions))),
        )


@dataclass(slots=True)
class CategoricalActionAdapter:
    """Map contiguous canonical action indices to environment-owned values."""

    action_values: Sequence[Any] | Mapping[int, Any]
    action_names: Sequence[str] | None = None
    positional_action_index: int | None = None
    bootstrap_index: int = 0
    fallback_indices: Sequence[int] | None = None
    position_kwargs: Callable[[Action], Mapping[str, Any]] | None = None
    _values: tuple[Any, ...] = field(init=False, repr=False)
    _names: tuple[str, ...] = field(init=False, repr=False)
    _fallback: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.action_values, Mapping):
            rows: dict[int, Any] = {}
            for raw_index, value in self.action_values.items():
                index = _action_index(raw_index)
                if index in rows:
                    raise AdapterError(
                        f"duplicate categorical action mapping index {index}"
                    )
                rows[index] = value
            expected = list(range(len(rows)))
            if sorted(rows) != expected:
                raise AdapterError(
                    "categorical action mapping keys must be contiguous from zero"
                )
            self._values = tuple(rows[idx] for idx in expected)
        else:
            if isinstance(self.action_values, (str, bytes, bytearray)):
                raise AdapterError("action_values must be a non-string sequence")
            try:
                self._values = tuple(self.action_values)
            except TypeError as exc:
                raise AdapterError("action_values must be a sequence") from exc
        if not self._values:
            raise AdapterError("action_values must not be empty")

        if self.action_names is None:
            self._names = tuple(str(value) for value in self._values)
        else:
            if isinstance(self.action_names, (str, bytes, bytearray)):
                raise AdapterError("action_names must be a non-string sequence")
            self._names = tuple(str(name) for name in self.action_names)
            if len(self._names) != len(self._values):
                raise AdapterError("action_names must match action_values length")

        self.bootstrap_index = _action_index(self.bootstrap_index)
        self._require_index(self.bootstrap_index)
        if self.positional_action_index is not None:
            self.positional_action_index = _action_index(
                self.positional_action_index
            )
            self._require_index(self.positional_action_index)

        fallback = (
            tuple(range(len(self._values)))
            if self.fallback_indices is None
            else tuple(
                dict.fromkeys(_action_index(v) for v in self.fallback_indices)
            )
        )
        for idx in fallback:
            self._require_index(idx)
        if not fallback:
            raise AdapterError("fallback_indices must not be empty")
        self._fallback = fallback

    @property
    def n_actions(self) -> int:
        return len(self._values)

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def _require_index(self, index: int) -> None:
        if index < 0 or index >= len(self._values):
            raise AdapterError(
                f"action index {index} falls outside [0, {len(self._values)})"
            )

    def _safe_index(self, requested: int) -> int:
        requested = _action_index(requested)
        self._require_index(requested)
        return requested

    def decode(
        self,
        action: Action,
        action_space: Any = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        del action_space
        if not isinstance(action, Action):
            raise AdapterError("decode requires a canonical Action")
        index = self._safe_index(action.index)
        kwargs: Mapping[str, Any] = {}
        if (
            self.positional_action_index is not None
            and index == self.positional_action_index
        ):
            if not action.has_position:
                raise AdapterError("positional action requires explicit x/y coordinates")
            x = _strict_int(action.x, name="action x coordinate")
            y = _strict_int(action.y, name="action y coordinate")
            if self.position_kwargs is None:
                kwargs = {"x": x, "y": y}
            else:
                raw_kwargs = self.position_kwargs(action)
                if not isinstance(raw_kwargs, Mapping):
                    raise AdapterError("position_kwargs must return a mapping")
                kwargs = dict(raw_kwargs)
        return self._values[index], kwargs

    def bootstrap(self, action_space: Any = None) -> tuple[Any, Mapping[str, Any]]:
        return self.decode(Action(self.bootstrap_index), action_space)

    def click_action_index(self) -> int | None:
        return self.positional_action_index

    def safe_action_indices(
        self,
        observation: Observation | None = None,
    ) -> Sequence[int]:
        if observation is not None and observation.available_actions:
            available = tuple(
                idx
                for idx in observation.available_actions
                if 0 <= int(idx) < len(self._values)
            )
            if available:
                return available
        return self._fallback


class MockActionAdapter(CategoricalActionAdapter):
    """Four-action clickless integer adapter."""

    def __init__(self, n_actions: int = 4) -> None:
        count = _strict_int(n_actions, name="n_actions")
        names = ("UP", "DOWN", "LEFT", "RIGHT")
        action_names = (
            names if count == len(names) else tuple(f"A{idx}" for idx in range(count))
        )
        super().__init__(
            action_values=tuple(range(count)),
            action_names=action_names,
            positional_action_index=None,
            bootstrap_index=0,
        )


class ArcActionAdapter:
    """ARC action adapter preserving the SDK's enum-value and click payload."""

    n_actions = 8
    action_names = (
        "RESET",
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
        "INTERACT",
        "CLICK",
        "UNDO",
    )

    def _value_map(self, action_space: Any) -> tuple[dict[int, Any], Any]:
        if action_space is None:
            values = {idx: idx for idx in range(self.n_actions)}
            return values, 0
        try:
            members = list(action_space)
        except TypeError as exc:
            raise AdapterError("ARC action_space must be an iterable enum class") from exc
        if not members:
            raise AdapterError("ARC action_space must expose at least one action")
        first = members[0] if members else None
        values: dict[int, Any] = {}
        for member in members:
            index = _action_index(member)
            if index < 0 or index >= self.n_actions:
                raise AdapterError(
                    f"ARC action index {index} falls outside [0, {self.n_actions})"
                )
            if index in values:
                raise AdapterError(f"duplicate ARC action index {index}")
            values[index] = member
        return values, first

    def decode(
        self,
        action: Action,
        action_space: Any,
    ) -> tuple[Any, Mapping[str, Any]]:
        if not isinstance(action, Action):
            raise AdapterError("decode requires a canonical Action")
        values, _first = self._value_map(action_space)
        index = _action_index(action.index)
        if index not in values:
            raise AdapterError(f"ARC action index {index} is unavailable")
        env_action = values[index]
        kwargs: Mapping[str, Any] = {}
        if index == 6:
            if not action.has_position:
                raise AdapterError("ARC click action requires explicit x/y coordinates")
            kwargs = {
                "data": {
                    "x": _strict_int(action.x, name="action x coordinate"),
                    "y": _strict_int(action.y, name="action y coordinate"),
                }
            }
        return env_action, kwargs

    def bootstrap(self, action_space: Any) -> tuple[Any, Mapping[str, Any]]:
        return self.decode(Action(0, name="RESET"), action_space)

    def click_action_index(self) -> int | None:
        return 6

    def safe_action_indices(
        self,
        observation: Observation | None = None,
    ) -> Sequence[int]:
        if observation is not None and observation.available_actions:
            available = tuple(
                idx
                for idx in observation.available_actions
                if 0 <= int(idx) < self.n_actions
            )
            if available:
                return available
        return (1, 2, 3, 4)


@dataclass(slots=True)
class CategoricalOutcomeAdapter:
    """Derive incremental outcome truth from two canonical observations."""

    state_field: str = "state"
    terminated_field: str = "terminated"
    truncated_field: str = "truncated"
    reward_field: str = "reward"
    hazard_field: str = "hazard"
    progress_scale: float = 100.0
    # Visual change alone is not an outcome. This shaping bonus is admitted
    # only when the same transition has an independent positive progress
    # signal (see ``outcome`` below).
    change_bonus: float = 10.0
    completion_bonus: float = 5.0
    terminal_penalty: float = 100.0
    terminal_tokens: tuple[str, ...] = ("GAME_OVER", "DEATH", "DEAD")
    success_tokens: tuple[str, ...] = (
        "GAME_COMPLETED",
        "VICTORY",
        "SUCCESS",
        "WIN",
    )

    def outcome(
        self,
        before: Observation,
        after: Observation,
        *,
        raw_after: Any,
        info: Mapping[str, Any] | None = None,
    ) -> Outcome:
        if info is None:
            info = {}
        elif not isinstance(info, Mapping):
            raise AdapterError("outcome info must be a mapping")
        state = _state_text(raw_after, self.state_field)
        upper_state = state.upper()
        progress_delta = max(0.0, float(after.progress) - float(before.progress))
        frame_changed = not np.array_equal(before.frame, after.frame)

        explicit_boundary = info.get("boundary")
        boundary: BoundaryKind | None = None
        if explicit_boundary is not None:
            try:
                boundary = (
                    explicit_boundary
                    if isinstance(explicit_boundary, BoundaryKind)
                    else BoundaryKind(str(explicit_boundary).lower())
                )
            except (TypeError, ValueError) as exc:
                raise AdapterError(
                    f"boundary must be a valid BoundaryKind, got "
                    f"{explicit_boundary!r}"
                ) from exc
        if boundary == BoundaryKind.NONE:
            boundary = None

        truncated = (
            _strict_bool(info["truncated"], name="truncated")
            if "truncated" in info
            else _optional_bool(raw_after, self.truncated_field, False)
        )
        explicit_terminated = (
            _strict_bool(info["terminated"], name="terminated")
            if "terminated" in info
            else _optional_bool(raw_after, self.terminated_field, False)
        )
        state_tokens = set(re.findall(r"[A-Z][A-Z0-9_]*", upper_state))
        success = any(str(token).upper() in state_tokens for token in self.success_tokens)
        terminal_match = any(
            str(token).upper() in state_tokens for token in self.terminal_tokens
        )
        terminated = bool(explicit_terminated or success or terminal_match)

        if boundary is None:
            if truncated:
                boundary = BoundaryKind.TIME_LIMIT
            elif success:
                boundary = BoundaryKind.GAME_COMPLETED
            elif terminated:
                boundary = BoundaryKind.DEATH
            elif progress_delta > 0.0:
                boundary = BoundaryKind.LEVEL_COMPLETED
            else:
                boundary = BoundaryKind.NONE

        # An authoritative boundary and the terminal flag must describe the
        # same action.  Environments commonly report only one of the two.
        if boundary in {
            BoundaryKind.GAME_COMPLETED,
            BoundaryKind.DEATH,
            BoundaryKind.ENVIRONMENT_ERROR,
        }:
            terminated = True
        if boundary == BoundaryKind.TIME_LIMIT:
            truncated = True
        if boundary == BoundaryKind.INTERRUPTED:
            truncated = True

        raw_reward = info.get("reward", _field(raw_after, self.reward_field, None))
        if raw_reward is None:
            reward = self.progress_scale * progress_delta
            # A raw pixel change is not evidence of progress on its own.  The
            # configured shaping bonus is admitted only when an independent
            # progress signal grounds the same transition.
            if frame_changed and progress_delta > 0.0:
                reward += self.change_bonus
            if progress_delta > 0.0:
                reward += self.completion_bonus
            if boundary in {BoundaryKind.DEATH, BoundaryKind.ENVIRONMENT_ERROR}:
                reward -= self.terminal_penalty
        else:
            reward = _finite_float(raw_reward, name=self.reward_field)

        raw_hazard = info.get("hazard", _field(raw_after, self.hazard_field, None))
        hazard = (
            _finite_float(raw_hazard, name=self.hazard_field)
            if raw_hazard is not None
            else (1.0 if boundary == BoundaryKind.DEATH else 0.0)
        )
        if not 0.0 <= hazard <= 1.0:
            raise AdapterError(
                f"{self.hazard_field} must be in [0, 1], got {hazard!r}"
            )
        return Outcome(
            reward=reward,
            progress_delta=progress_delta,
            terminated=terminated,
            truncated=truncated,
            boundary=boundary,
            hazard=hazard,
            metadata={
                "state": state,
                "frame_changed": bool(frame_changed),
                "level_completed": bool(progress_delta > 0.0),
                "progress_before": float(before.progress),
                "progress_after": float(after.progress),
            },
        )


class ArcOutcomeAdapter(CategoricalOutcomeAdapter):
    """ARC boundary/outcome adapter."""

    def __init__(self) -> None:
        super().__init__(
            state_field="state",
            terminated_field="terminated",
            truncated_field="truncated",
            reward_field="reward",
            hazard_field="hazard",
            terminal_tokens=("GAME_OVER", "DEATH", "DEAD"),
            success_tokens=("GAME_COMPLETED", "VICTORY", "SUCCESS", "WIN"),
        )


class MockOutcomeAdapter(CategoricalOutcomeAdapter):
    """Generic outcome adapter for the mock categorical environment."""


# Explicit generic aliases make the intended public vocabulary discoverable.
GenericCategoricalObservationAdapter = CategoricalObservationAdapter
GenericCategoricalActionAdapter = CategoricalActionAdapter
GenericCategoricalOutcomeAdapter = CategoricalOutcomeAdapter


__all__ = [
    "AdapterError",
    "ArcActionAdapter",
    "ArcObservationAdapter",
    "ArcOutcomeAdapter",
    "CategoricalActionAdapter",
    "CategoricalObservationAdapter",
    "CategoricalOutcomeAdapter",
    "GenericCategoricalActionAdapter",
    "GenericCategoricalObservationAdapter",
    "GenericCategoricalOutcomeAdapter",
    "MockActionAdapter",
    "MockObservationAdapter",
    "MockOutcomeAdapter",
    "ObservationUnavailable",
]
