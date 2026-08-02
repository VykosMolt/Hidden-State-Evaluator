"""Exogenous observation-change filtering for durable memory identity.

Exact frame hashing makes every observation containing an autonomously
evolving component (a clock, a depleting bar, a background animation) a
brand-new state, so hash-addressed memory never sees a revisit.  This module
learns which observation cells are *exogenous* — outside the agent's control
— and provides a memory identity that masks exactly those cells.

The test is intervention-grounded, in the spirit of the exogenous-state
decomposition literature (Dietterich et al., ICML 2018; Efroni et al., ICLR
2022; Lamb et al., 2022): a cell is confirmed exogenous only when its
change-event trajectory replays identically across episodes whose executed
action sequences differ.  Different action histories are real interventions;
an identical cell trajectory under them is direct evidence of
action-independence.  No assumption is made about geometry, values, tick
rates, or any particular environment family.

Scope and safety:

* only the *memory* identity is filtered; transactional identity, teacher
  lookup, and executable-model state stay exact;
* evidence and masks are scoped by task, visual stage, and frame shape;
* the mask is frozen within a task/stage segment and recomputed only when a
  comparable segment is finalized, so memory keys stay stable within it;
* a later contradiction unmasks the cell (conservative), and nodes recorded
  under an older mask are simply orphaned rather than rewritten;
* a deterministic time-driven hazard animation is also exogenous by this
  test and will be merged in memory — outcome attribution through object
  beliefs and evidence remains the safety carrier for such cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    Action,
    ExogenousConfig,
    Observation,
    stable_frame_hash,
)


_MASK_SENTINEL = -8
EXOGENOUS_STATE_VERSION = 2


def _shape_key(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(v)) for v in shape)


def _shape_tuple(shape: Sequence[Any]) -> tuple[int, int]:
    values = tuple(int(v) for v in shape)
    if len(values) != 2 or any(v <= 0 for v in values):
        raise ValueError(f"exogenous frame shape must be positive 2-D, got {values!r}")
    return values


def _stage(value: Any) -> int:
    return max(1, int(value))


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


def _state_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _state_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _state_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise ValueError(f"{name} must be a sequence")
    return value


def _state_shape(value: Any, name: str) -> tuple[int, int]:
    values = _state_sequence(value, name)
    if len(values) != 2:
        raise ValueError(f"{name} must contain [height, width]")
    return (
        _state_int(values[0], f"{name} height", minimum=1),
        _state_int(values[1], f"{name} width", minimum=1),
    )


@dataclass(slots=True)
class _EpisodeBuffer:
    task_id: str
    stage: int
    shape_key: str | None = None
    steps: int = 0
    actions: list[tuple[int, int, int]] = field(default_factory=list)
    # (y, x) -> [(step, new_value), ...]
    events: dict[tuple[int, int], list[tuple[int, int]]] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class _TaskShapeStats:
    # Previous finished episode, kept for the pairwise comparison.
    previous_steps: int = 0
    previous_actions: list[tuple[int, int, int]] = field(default_factory=list)
    previous_events: dict[tuple[int, int], list[tuple[int, int]]] = field(
        default_factory=dict
    )
    has_previous: bool = False
    # (y, x) -> consecutive cross-episode confirmations.
    confirmations: dict[tuple[int, int], int] = field(default_factory=dict)
    masked: set[tuple[int, int]] = field(default_factory=set)
    episodes: int = 0


class ExogenousChangeFilter:
    """Cross-episode, intervention-grounded exogenous-cell detection."""

    def __init__(self, config: ExogenousConfig | None = None) -> None:
        self.config = config or ExogenousConfig()
        self._stats: dict[tuple[str, int, str], _TaskShapeStats] = {}
        self._active: _EpisodeBuffer | None = None
        self.finalized_episodes = 0
        # V1 checkpoints keyed evidence by only (task, shape).  Such masks
        # cannot be assigned to a stage without risking cross-stage aliasing,
        # so the loader records and ignores them rather than fanning them out.
        self._quarantined_legacy_scopes = 0
        self._quarantined_legacy_active = False

    # ------------------------------------------------------------------ run

    def begin_episode(self, task_id: str, stage: int = 1) -> None:
        """Finalize any active segment, then record ``(task_id, stage)``."""

        if not self.config.enabled:
            return
        self._finalize_active()
        self._active = _EpisodeBuffer(
            task_id=str(task_id),
            stage=_stage(stage),
        )

    def end_episode(self) -> bool:
        """Explicitly finalize the current episode.

        Returns ``True`` when a non-empty comparable stage segment was committed.
        Relying on the next ``begin_episode`` loses the final run of an
        experiment and any active buffer saved at shutdown.
        """

        if not self.config.enabled:
            return False
        return self._finalize_active()

    def observe_transition(
        self,
        *,
        task_id: str,
        before_frame: np.ndarray,
        after_frame: np.ndarray,
        action: Action,
        stage: int = 1,
        after_stage: int | None = None,
    ) -> int:
        """Record one committed real transition; returns recorded change cells.

        ``stage`` belongs to ``before_frame`` and ``after_stage`` belongs to
        ``after_frame``.  A differing pair is a boundary transition and is
        intentionally not admitted as evidence for either stage.
        """

        if not self.config.enabled:
            return 0
        before_stage = _stage(stage)
        successor_stage = (
            before_stage if after_stage is None else _stage(after_stage)
        )
        buffer = self._active
        if (
            buffer is None
            or buffer.task_id != str(task_id)
            or buffer.stage != before_stage
        ):
            self.begin_episode(task_id, before_stage)
            buffer = self._active
        before = np.asarray(before_frame)
        after = np.asarray(after_frame)
        if before.ndim != 2 or after.ndim != 2:
            raise ValueError(
                "exogenous transition frames must both be two-dimensional"
            )
        if before_stage != successor_stage:
            # A boundary image is not comparable with either stage's ordinary
            # within-stage dynamics.  Commit the old segment, skip this visual
            # transition, and prepare an empty buffer for the successor stage.
            self._finalize_active()
            self._active = _EpisodeBuffer(
                task_id=str(task_id),
                stage=successor_stage,
            )
            return 0
        if before.shape != after.shape:
            # A shape change (stage switch) ends comparable recording.
            self._finalize_active()
            self._active = _EpisodeBuffer(
                task_id=str(task_id),
                stage=successor_stage,
            )
            return 0
        key = _shape_key(_shape_tuple(after.shape))
        if buffer.shape_key is None:
            buffer.shape_key = key
        elif buffer.shape_key != key:
            self._finalize_active()
            self._active = _EpisodeBuffer(
                task_id=str(task_id),
                stage=before_stage,
                shape_key=key,
            )
            buffer = self._active
        step = buffer.steps
        buffer.steps += 1
        buffer.actions.append(action.key)
        changed = np.argwhere(before != after)
        limit = int(self.config.max_events_per_cell)
        for y, x in changed:
            cell = (int(y), int(x))
            rows = buffer.events.setdefault(cell, [])
            if len(rows) < limit:
                rows.append((step, int(after[cell[0], cell[1]])))
        return int(len(changed))

    def _finalize_active(self) -> bool:
        buffer = self._active
        self._active = None
        if buffer is None or buffer.shape_key is None or buffer.steps <= 0:
            return False
        stats = self._stats.setdefault(
            (buffer.task_id, buffer.stage, buffer.shape_key),
            _TaskShapeStats(),
        )
        stats.episodes += 1
        self.finalized_episodes += 1
        if stats.has_previous:
            self._compare(stats, buffer)
        stats.previous_steps = buffer.steps
        stats.previous_actions = list(buffer.actions)
        stats.previous_events = {
            cell: list(rows) for cell, rows in buffer.events.items()
        }
        stats.has_previous = True
        return True

    def _compare(self, stats: _TaskShapeStats, current: _EpisodeBuffer) -> None:
        horizon = min(stats.previous_steps, current.steps)
        if horizon < int(self.config.min_common_horizon):
            return
        previous_actions = stats.previous_actions[:horizon]
        current_actions = current.actions[:horizon]
        if previous_actions == current_actions:
            # Identical action histories are not an intervention; the
            # comparison carries no evidence either way.
            return

        def _within(rows: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
            return [row for row in rows if row[0] < horizon]

        cells = set(stats.previous_events) | set(current.events)
        threshold = int(self.config.min_confirmations)
        for cell in cells:
            previous_rows = _within(stats.previous_events.get(cell, ()))
            current_rows = _within(current.events.get(cell, ()))
            # The intervention must occur on a transition where this cell
            # exhibits the matched event.  A differing action after the
            # cell's final change (or only on unrelated transitions) is not
            # evidence that the cell is action-independent.
            intervened_on_event = bool(
                previous_rows
                and any(
                    previous_actions[step] != current_actions[step]
                    for step, _value in previous_rows
                )
            )
            if previous_rows and previous_rows == current_rows and intervened_on_event:
                count = stats.confirmations.get(cell, 0) + 1
                stats.confirmations[cell] = count
                if count >= threshold:
                    stats.masked.add(cell)
            else:
                # Contradiction or absence: reset and conservatively unmask.
                if cell in stats.confirmations:
                    stats.confirmations[cell] = 0
                stats.masked.discard(cell)

    # ---------------------------------------------------------------- reads

    def mask_cells(
        self,
        task_id: str,
        shape: tuple[int, ...],
        stage: int = 1,
    ) -> frozenset[tuple[int, int]]:
        stats = self._stats.get(
            (str(task_id), _stage(stage), _shape_key(_shape_tuple(shape)))
        )
        if stats is None:
            return frozenset()
        return frozenset(stats.masked)

    def fuse_estimate(
        self,
        task_id: str,
        shape: tuple[int, ...],
        stage: int = 1,
    ) -> int | None:
        """Estimated episode budget implied by the exogenous cells.

        Masked cells are reproducible, action-independent schedules; the last
        step at which any of them changed in the previous episode bounds the
        clock's observed span.  Returns ``None`` when no mask exists.
        """

        stats = self._stats.get(
            (str(task_id), _stage(stage), _shape_key(_shape_tuple(shape)))
        )
        if stats is None or not stats.masked or not stats.has_previous:
            return None
        last_steps = [
            rows[-1][0]
            for cell in stats.masked
            if (rows := stats.previous_events.get(cell))
        ]
        if not last_steps:
            return None
        return int(max(last_steps)) + 1

    def time_pressure(
        self,
        task_id: str,
        shape: tuple[int, ...],
        step: int,
        stage: int = 1,
    ) -> float:
        """Fraction of the estimated exogenous budget already consumed."""

        fuse = self.fuse_estimate(task_id, shape, stage)
        if fuse is None or fuse <= 0:
            return 0.0
        return float(np.clip(float(step) / float(fuse), 0.0, 1.0))

    def masked_state_id(self, observation: Observation) -> str:
        """Memory identity; equals the exact identity while no mask exists."""

        if not self.config.enabled:
            return ""
        frame = np.asarray(observation.frame)
        cells = self.mask_cells(
            observation.task_id,
            frame.shape,
            observation.stage,
        )
        if not cells:
            return ""
        masked = frame.astype(np.int64, copy=True)
        for y, x in cells:
            if 0 <= y < masked.shape[0] and 0 <= x < masked.shape[1]:
                masked[y, x] = _MASK_SENTINEL
        return stable_frame_hash(
            masked,
            task_id=observation.task_id,
            stage=observation.stage,
            available_actions=observation.available_actions,
            progress=observation.progress,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "task_shapes": len(self._stats),
            "task_stage_shapes": len(self._stats),
            "finalized_episodes": int(self.finalized_episodes),
            "quarantined_legacy_scopes": int(
                self._quarantined_legacy_scopes
            ),
            "quarantined_legacy_active": bool(
                self._quarantined_legacy_active
            ),
            "masked_cells": int(
                sum(len(stats.masked) for stats in self._stats.values())
            ),
        }

    # ---------------------------------------------------------- persistence

    def state_dict(self) -> dict[str, Any]:
        def _events(
            rows: Mapping[tuple[int, int], Sequence[tuple[int, int]]],
        ) -> list[dict[str, Any]]:
            return [
                {
                    "cell": [int(cell[0]), int(cell[1])],
                    "events": [
                        [int(step), int(value)] for step, value in events
                    ],
                }
                for cell, events in sorted(rows.items())
            ]

        scopes = []
        for (task_id, stage, shape_key), stats in sorted(
            self._stats.items(),
            key=lambda item: item[0],
        ):
            shape = _shape_tuple(shape_key.split("x"))
            scopes.append(
                {
                    "task_id": task_id,
                    "stage": int(stage),
                    "shape": list(shape),
                    "previous_steps": int(stats.previous_steps),
                    "previous_actions": [
                        [int(v) for v in key] for key in stats.previous_actions
                    ],
                    "previous_events": _events(stats.previous_events),
                    "has_previous": bool(stats.has_previous),
                    "confirmations": [
                        {
                            "cell": [int(cell[0]), int(cell[1])],
                            "count": int(count),
                        }
                        for cell, count in sorted(stats.confirmations.items())
                    ],
                    "masked": [
                        [int(y), int(x)] for y, x in sorted(stats.masked)
                    ],
                    "episodes": int(stats.episodes),
                }
            )
        active = None
        if self._active is not None:
            active_shape = (
                None
                if self._active.shape_key is None
                else list(_shape_tuple(self._active.shape_key.split("x")))
            )
            active = {
                "task_id": self._active.task_id,
                "stage": int(self._active.stage),
                "shape": active_shape,
                "steps": int(self._active.steps),
                "actions": [
                    [int(v) for v in key] for key in self._active.actions
                ],
                "events": _events(self._active.events),
            }
        return {
            "version": EXOGENOUS_STATE_VERSION,
            "scopes": scopes,
            "active": active,
            "finalized_episodes": int(self.finalized_episodes),
            "legacy_quarantine": {
                "scopes": int(self._quarantined_legacy_scopes),
                "active": bool(self._quarantined_legacy_active),
            },
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        config: ExogenousConfig | None = None,
    ) -> "ExogenousChangeFilter":
        if not isinstance(state, Mapping):
            raise ValueError("exogenous state must be a mapping")
        instance = cls(config)

        version = (
            1
            if "version" not in state
            else _state_int(
                state["version"],
                "exogenous state version",
                minimum=1,
            )
        )
        if version == 1:
            # V1 used (task, shape) scopes.  Applying those masks to any one
            # stage—or all stages—would preserve the aliasing this version
            # fixes, so retain only transparent quarantine counts and relearn.
            legacy_stats = state.get("stats", {})
            if not isinstance(legacy_stats, Mapping):
                raise ValueError("legacy exogenous stats must be a mapping")
            legacy_active = state.get("active")
            if legacy_active is not None and not isinstance(
                legacy_active,
                Mapping,
            ):
                raise ValueError(
                    "legacy exogenous active state must be a mapping or null"
                )
            instance._quarantined_legacy_scopes = (
                len(legacy_stats)
            )
            instance._quarantined_legacy_active = legacy_active is not None
            instance.finalized_episodes = _state_int(
                state.get("finalized_episodes", 0),
                "legacy exogenous finalized_episodes",
                minimum=0,
            )
            return instance
        if version != EXOGENOUS_STATE_VERSION:
            raise ValueError(
                f"unsupported exogenous state version {version}; "
                f"expected {EXOGENOUS_STATE_VERSION}"
            )

        def _cell(
            raw: Any,
            *,
            shape: tuple[int, int],
            name: str,
        ) -> tuple[int, int]:
            values = _state_sequence(raw, name)
            if len(values) != 2:
                raise ValueError(
                    f"exogenous cell must contain [y, x], got {raw!r}"
                )
            cell = (
                _state_int(values[0], f"{name} y", minimum=0),
                _state_int(values[1], f"{name} x", minimum=0),
            )
            if not (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]):
                raise ValueError(
                    f"exogenous cell {cell!r} falls outside shape {shape!r}"
                )
            return cell

        def _actions(raw: Any, *, name: str) -> list[tuple[int, int, int]]:
            encoded = _state_sequence(raw, name)
            actions: list[tuple[int, int, int]] = []
            for index, action in enumerate(encoded):
                action_values = _state_sequence(
                    action,
                    f"{name} action {index}",
                )
                values = tuple(
                    _state_int(
                        value,
                        f"{name} action {index} value",
                    )
                    for value in action_values
                )
                if len(values) != 3:
                    raise ValueError(
                        f"exogenous action key must contain three values, "
                        f"got {values!r}"
                    )
                action_index, x, y = values
                if action_index < 0:
                    raise ValueError(
                        "exogenous action index must be nonnegative"
                    )
                if not (
                    (x == -1 and y == -1)
                    or (x >= 0 and y >= 0)
                ):
                    raise ValueError(
                        "exogenous action coordinates must both be -1 "
                        "or both be nonnegative"
                    )
                actions.append(values)
            return actions

        def _events(
            raw: Any,
            *,
            shape: tuple[int, int],
            steps: int,
            name: str,
        ) -> dict[tuple[int, int], list[tuple[int, int]]]:
            encoded = _state_sequence(raw, name)
            decoded: dict[tuple[int, int], list[tuple[int, int]]] = {}
            for event_index, event_row in enumerate(encoded):
                if not isinstance(event_row, Mapping):
                    raise ValueError("exogenous event row must be a mapping")
                if "cell" not in event_row or "events" not in event_row:
                    raise ValueError(
                        f"{name} event row is missing cell or events"
                    )
                cell = _cell(
                    event_row["cell"],
                    shape=shape,
                    name=f"{name} event {event_index} cell",
                )
                if cell in decoded:
                    raise ValueError(f"duplicate exogenous event cell {cell!r}")
                raw_rows = _state_sequence(
                    event_row["events"],
                    f"{name} event {event_index} rows",
                )
                if len(raw_rows) > int(instance.config.max_events_per_cell):
                    raise ValueError(
                        f"{name} event history exceeds max_events_per_cell"
                    )
                rows: list[tuple[int, int]] = []
                previous_step = -1
                for row_index, raw_row in enumerate(raw_rows):
                    values = _state_sequence(
                        raw_row,
                        f"{name} event {event_index} row {row_index}",
                    )
                    if len(values) != 2:
                        raise ValueError(
                            "exogenous event rows must contain [step, value]"
                        )
                    step = _state_int(
                        values[0],
                        f"{name} event step",
                        minimum=0,
                    )
                    if step >= steps:
                        raise ValueError(
                            f"{name} event step {step} exceeds horizon {steps}"
                        )
                    if step <= previous_step:
                        raise ValueError(
                            f"{name} event steps must be strictly increasing"
                        )
                    previous_step = step
                    rows.append(
                        (
                            step,
                            _state_int(
                                values[1],
                                f"{name} event value",
                            ),
                        )
                    )
                if not rows:
                    raise ValueError(
                        f"{name} event histories must not be empty"
                    )
                decoded[cell] = rows
            return decoded

        required_top = {
            "scopes",
            "active",
            "finalized_episodes",
            "legacy_quarantine",
        }
        if not required_top.issubset(state):
            missing = sorted(required_top - set(state))
            raise ValueError(
                f"exogenous state is missing fields {missing!r}"
            )
        raw_scopes = _state_sequence(
            state["scopes"],
            "exogenous scopes",
        )
        current_episode_total = 0
        for scope_index, row in enumerate(raw_scopes):
            if not isinstance(row, Mapping):
                raise ValueError("exogenous scope row must be a mapping")
            required_scope = {
                "task_id",
                "stage",
                "shape",
                "previous_steps",
                "previous_actions",
                "previous_events",
                "has_previous",
                "confirmations",
                "masked",
                "episodes",
            }
            if not required_scope.issubset(row):
                missing = sorted(required_scope - set(row))
                raise ValueError(
                    f"exogenous scope {scope_index} is missing {missing!r}"
                )
            scope_name = f"exogenous scope {scope_index}"
            task_id = _state_string(row["task_id"], f"{scope_name} task_id")
            stage = _state_int(
                row["stage"],
                f"{scope_name} stage",
                minimum=1,
            )
            shape = _state_shape(row["shape"], f"{scope_name} shape")
            scope = (task_id, stage, _shape_key(shape))
            if scope in instance._stats:
                raise ValueError(f"duplicate exogenous scope {scope!r}")
            previous_steps = _state_int(
                row["previous_steps"],
                f"{scope_name} previous_steps",
                minimum=1,
            )
            previous_actions = _actions(
                row["previous_actions"],
                name=f"{scope_name} previous_actions",
            )
            if len(previous_actions) != previous_steps:
                raise ValueError(
                    f"{scope_name} action count must equal previous_steps"
                )
            previous_events = _events(
                row["previous_events"],
                shape=shape,
                steps=previous_steps,
                name=f"{scope_name} previous_events",
            )
            has_previous = _state_bool(
                row["has_previous"],
                f"{scope_name} has_previous",
            )
            if not has_previous:
                raise ValueError(
                    f"{scope_name} must contain previous evidence"
                )
            confirmations: dict[tuple[int, int], int] = {}
            raw_confirmations = _state_sequence(
                row["confirmations"],
                f"{scope_name} confirmations",
            )
            for confirmation_index, confirmation in enumerate(
                raw_confirmations
            ):
                if not isinstance(confirmation, Mapping):
                    raise ValueError(
                        "exogenous confirmation row must be a mapping"
                    )
                if "cell" not in confirmation or "count" not in confirmation:
                    raise ValueError(
                        "exogenous confirmation is missing cell or count"
                    )
                cell = _cell(
                    confirmation["cell"],
                    shape=shape,
                    name=(
                        f"{scope_name} confirmation "
                        f"{confirmation_index} cell"
                    ),
                )
                if cell in confirmations:
                    raise ValueError(
                        f"duplicate exogenous confirmation cell {cell!r}"
                    )
                confirmations[cell] = _state_int(
                    confirmation["count"],
                    f"{scope_name} confirmation count",
                    minimum=0,
                )
            raw_masked = _state_sequence(
                row["masked"],
                f"{scope_name} masked",
            )
            masked_list = [
                _cell(
                    cell,
                    shape=shape,
                    name=f"{scope_name} masked cell",
                )
                for cell in raw_masked
            ]
            if len(set(masked_list)) != len(masked_list):
                raise ValueError(f"{scope_name} masked cells contain duplicates")
            masked = set(masked_list)
            for cell in masked:
                if (
                    confirmations.get(cell, 0)
                    < int(instance.config.min_confirmations)
                    or cell not in previous_events
                ):
                    raise ValueError(
                        f"{scope_name} masked cell lacks confirming evidence"
                    )
            episodes = _state_int(
                row["episodes"],
                f"{scope_name} episodes",
                minimum=1,
            )
            if any(
                count > max(0, episodes - 1)
                for count in confirmations.values()
            ):
                raise ValueError(
                    f"{scope_name} confirmation count exceeds episode evidence"
                )
            current_episode_total += episodes
            stats = _TaskShapeStats(
                previous_steps=previous_steps,
                previous_actions=previous_actions,
                previous_events=previous_events,
                has_previous=has_previous,
                confirmations=confirmations,
                masked=masked,
                episodes=episodes,
            )
            instance._stats[scope] = stats
        instance.finalized_episodes = _state_int(
            state["finalized_episodes"],
            "exogenous finalized_episodes",
            minimum=0,
        )
        if instance.finalized_episodes < current_episode_total:
            raise ValueError(
                "exogenous finalized_episodes is below scoped episode total"
            )
        quarantine = state["legacy_quarantine"]
        if not isinstance(quarantine, Mapping):
            raise ValueError("exogenous legacy_quarantine must be a mapping")
        if "scopes" not in quarantine or "active" not in quarantine:
            raise ValueError(
                "exogenous legacy_quarantine is missing fields"
            )
        instance._quarantined_legacy_scopes = _state_int(
            quarantine["scopes"],
            "exogenous quarantined legacy scopes",
            minimum=0,
        )
        instance._quarantined_legacy_active = _state_bool(
            quarantine["active"],
            "exogenous quarantined legacy active",
        )
        if (
            instance._quarantined_legacy_scopes == 0
            and not instance._quarantined_legacy_active
            and instance.finalized_episodes != current_episode_total
        ):
            raise ValueError(
                "exogenous finalized_episodes does not equal scoped episodes"
            )
        active = state["active"]
        if active is not None and not isinstance(active, Mapping):
            raise ValueError("exogenous active state must be a mapping or null")
        if isinstance(active, Mapping):
            required_active = {
                "task_id",
                "stage",
                "shape",
                "steps",
                "actions",
                "events",
            }
            if not required_active.issubset(active):
                missing = sorted(required_active - set(active))
                raise ValueError(
                    f"exogenous active state is missing {missing!r}"
                )
            active_shape_raw = active.get("shape")
            active_shape = (
                None
                if active_shape_raw is None
                else _state_shape(
                    active_shape_raw,
                    "exogenous active shape",
                )
            )
            active_steps = _state_int(
                active["steps"],
                "exogenous active steps",
                minimum=0,
            )
            active_actions = _actions(
                active["actions"],
                name="exogenous active actions",
            )
            if len(active_actions) != active_steps:
                raise ValueError(
                    "exogenous active action count must equal steps"
                )
            if active_shape is None:
                raw_active_events = _state_sequence(
                    active["events"],
                    "exogenous active events",
                )
                if active_steps != 0 or active_actions or raw_active_events:
                    raise ValueError(
                        "shapeless exogenous active state must be empty"
                    )
                active_events: dict[
                    tuple[int, int],
                    list[tuple[int, int]],
                ] = {}
            else:
                if active_steps <= 0:
                    raise ValueError(
                        "shaped exogenous active state requires steps"
                    )
                active_events = _events(
                    active["events"],
                    shape=active_shape,
                    steps=active_steps,
                    name="exogenous active events",
                )
            instance._active = _EpisodeBuffer(
                task_id=_state_string(
                    active["task_id"],
                    "exogenous active task_id",
                ),
                stage=_state_int(
                    active["stage"],
                    "exogenous active stage",
                    minimum=1,
                ),
                shape_key=(
                    None
                    if active_shape is None
                    else _shape_key(active_shape)
                ),
                steps=active_steps,
                actions=active_actions,
                events=active_events,
            )
        return instance


__all__ = [
    "EXOGENOUS_STATE_VERSION",
    "ExogenousChangeFilter",
]
