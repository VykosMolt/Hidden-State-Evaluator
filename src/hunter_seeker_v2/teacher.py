"""Single audited boundary for all demonstration and teacher access.

The compact runtime does not import phase templates or solved-prefix logic
directly.  Any implementation that can provide demonstrations or action advice
must satisfy :class:`Teacher` and be accessed through
:class:`TeacherAccessGuard`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .contracts import Action, RuntimeMode, frozen_mapping, readonly_array
from .contracts import stable_frame_hash


def _strict_integer(name: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _strict_bool(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool")
    return bool(value)


def _strict_number(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _strict_string(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _canonical_candidates(
    values: Sequence[Action],
    *,
    name: str,
) -> tuple[Action, ...]:
    try:
        candidates = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of Action values") from exc
    if any(not isinstance(action, Action) for action in candidates):
        raise TypeError(f"{name} must contain only Action values")
    keys = tuple(action.key for action in candidates)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} must not contain duplicate action keys")
    return candidates


def _categorical_trajectory_frames(value: Any, *, name: str) -> np.ndarray:
    """Validate and canonicalize recorded frames like the live adapter."""

    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"trajectory {name} must have shape [N,H,W]")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError(f"trajectory {name} must contain numeric labels")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or not np.equal(array, np.rint(array)).all():
            raise ValueError(
                f"trajectory {name} must contain finite integer labels"
            )
        array = np.rint(array)
    if array.size:
        info = np.iinfo(np.int64)
        if int(array.min()) < info.min or int(array.max()) > info.max:
            raise ValueError(f"trajectory {name} labels exceed int64 range")
    return np.ascontiguousarray(array.astype(np.int64, copy=False))


class TeacherAccessPhase(str, Enum):
    ACTION_SELECTION = "action_selection"
    OFFLINE_TRAINING = "offline_training"
    DATA_IMPORT = "data_import"
    EVALUATION = "evaluation"


@dataclass(frozen=True, slots=True)
class TeacherRequest:
    """Current-state-only request for optional teacher advice."""

    task_id: str
    stage: int
    state_id: str
    candidates: tuple[Action, ...]
    purpose: str = "action_advice"
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_id",
            _strict_string("teacher request task_id", self.task_id),
        )
        object.__setattr__(
            self,
            "stage",
            _strict_integer("teacher request stage", self.stage, minimum=1),
        )
        object.__setattr__(
            self,
            "state_id",
            _strict_string("teacher request state_id", self.state_id),
        )
        object.__setattr__(
            self,
            "candidates",
            _canonical_candidates(
                self.candidates,
                name="teacher request candidates",
            ),
        )
        object.__setattr__(
            self,
            "purpose",
            _strict_string("teacher request purpose", self.purpose),
        )
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class TeacherQuery:
    """Query for offline demonstration examples."""

    task_id: str
    stage: int | None = None
    limit: int = 1_000
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        limit = _strict_integer("teacher query limit", self.limit, minimum=1)
        object.__setattr__(
            self,
            "task_id",
            _strict_string("teacher query task_id", self.task_id),
        )
        object.__setattr__(
            self,
            "stage",
            (
                None
                if self.stage is None
                else _strict_integer(
                    "teacher query stage",
                    self.stage,
                    minimum=1,
                )
            ),
        )
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class TeacherResponse:
    """Teacher advice; ``action=None`` is an explicit abstention."""

    action: Action | None
    confidence: float
    source: str
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        if self.action is not None and not isinstance(self.action, Action):
            raise TypeError("teacher response action must be an Action or None")
        object.__setattr__(
            self,
            "confidence",
            _strict_number(
                "teacher response confidence",
                self.confidence,
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "source",
            _strict_string("teacher response source", self.source),
        )
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class TeacherExample:
    """Offline demonstration record used to train a student."""

    task_id: str
    stage: int
    state_id: str
    action: Action
    successor_state_id: str | None = None
    weight: float = 1.0
    source: str = "teacher"
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            raise TypeError("teacher example action must be an Action")
        weight = _strict_number(
            "teacher example weight",
            self.weight,
            minimum=0.0,
        )
        object.__setattr__(
            self,
            "task_id",
            _strict_string("teacher example task_id", self.task_id),
        )
        object.__setattr__(
            self,
            "stage",
            _strict_integer("teacher example stage", self.stage, minimum=1),
        )
        object.__setattr__(
            self,
            "state_id",
            _strict_string("teacher example state_id", self.state_id),
        )
        object.__setattr__(
            self,
            "successor_state_id",
            (
                None
                if self.successor_state_id is None
                else _strict_string(
                    "teacher example successor_state_id",
                    self.successor_state_id,
                )
            ),
        )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(
            self,
            "source",
            _strict_string("teacher example source", self.source),
        )
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))


@runtime_checkable
class Teacher(Protocol):
    """Protocol implemented by every source of demonstrations or advice."""

    teacher_id: str

    def suggest(self, request: TeacherRequest) -> TeacherResponse | None:
        """Return current-state advice or abstain."""

    def examples(self, query: TeacherQuery) -> Sequence[TeacherExample]:
        """Return offline examples matching ``query``."""


@dataclass(frozen=True, slots=True)
class TeacherAccessRecord:
    sequence: int
    teacher_id: str
    method: str
    phase: TeacherAccessPhase
    mode: RuntimeMode
    allowed: bool
    task_id: str
    stage: int | None
    state_id: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            _strict_integer(
                "teacher access sequence",
                self.sequence,
                minimum=1,
            ),
        )
        object.__setattr__(self, "teacher_id", str(self.teacher_id))
        object.__setattr__(self, "method", str(self.method))
        object.__setattr__(self, "phase", TeacherAccessPhase(self.phase))
        object.__setattr__(self, "mode", RuntimeMode(self.mode))
        object.__setattr__(
            self,
            "allowed",
            _strict_bool("teacher access allowed", self.allowed),
        )
        object.__setattr__(self, "task_id", str(self.task_id))
        object.__setattr__(
            self,
            "stage",
            (
                None
                if self.stage is None
                else _strict_integer(
                    "teacher access stage",
                    self.stage,
                    minimum=1,
                )
            ),
        )
        object.__setattr__(self, "state_id", str(self.state_id))
        object.__setattr__(self, "reason", str(self.reason))


class TeacherAccessAudit:
    """Bounded audit trail that counts all attempted accesses."""

    def __init__(self, *, max_records: int = 4_096) -> None:
        capacity = _strict_integer(
            "teacher audit max_records",
            max_records,
            minimum=1,
        )
        self._records: deque[TeacherAccessRecord] = deque(maxlen=capacity)
        self._attempts = 0
        self._allowed = 0
        self._denied = 0
        self._action_selection_reads = 0
        self._next_sequence = 1

    @property
    def records(self) -> tuple[TeacherAccessRecord, ...]:
        return tuple(self._records)

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def allowed(self) -> int:
        return self._allowed

    @property
    def denied(self) -> int:
        return self._denied

    @property
    def action_selection_reads(self) -> int:
        return self._action_selection_reads

    def record(
        self,
        *,
        teacher_id: str,
        method: str,
        phase: TeacherAccessPhase,
        mode: RuntimeMode,
        allowed: bool,
        task_id: str,
        stage: int | None,
        state_id: str,
        reason: str,
    ) -> TeacherAccessRecord:
        row = TeacherAccessRecord(
            sequence=self._next_sequence,
            teacher_id=str(teacher_id),
            method=str(method),
            phase=TeacherAccessPhase(phase),
            mode=RuntimeMode(mode),
            allowed=_strict_bool("teacher access allowed", allowed),
            task_id=str(task_id),
            stage=(
                None
                if stage is None
                else _strict_integer(
                    "teacher access stage",
                    stage,
                    minimum=1,
                )
            ),
            state_id=str(state_id),
            reason=str(reason),
        )
        self._next_sequence += 1
        self._attempts += 1
        if row.allowed:
            self._allowed += 1
            if row.phase is TeacherAccessPhase.ACTION_SELECTION:
                self._action_selection_reads += 1
        else:
            self._denied += 1
        self._records.append(row)
        return row

    def summary(self) -> Mapping[str, int]:
        return frozen_mapping(
            {
                "attempts": self.attempts,
                "allowed": self.allowed,
                "denied": self.denied,
                "retained_records": len(self._records),
                "action_selection_reads": self.action_selection_reads,
            }
        )


class TeacherAccessViolation(PermissionError):
    """Raised before a forbidden teacher implementation is invoked."""


@dataclass(frozen=True, slots=True)
class TeacherAccessView:
    """A phase-bound view that prevents call sites from omitting the phase."""

    guard: "TeacherAccessGuard"
    phase: TeacherAccessPhase

    def suggest(self, request: TeacherRequest) -> TeacherResponse | None:
        return self.guard.suggest(request, phase=self.phase)

    def examples(self, query: TeacherQuery) -> tuple[TeacherExample, ...]:
        return self.guard.examples(query, phase=self.phase)


class TeacherAccessGuard:
    """Audit and enforce teacher access under one runtime mode.

    Autonomous and student modes may consume examples during explicit offline
    training/import phases, but any teacher read during action selection is a
    hard error.  Only ``compat_assisted`` permits such runtime advice.
    """

    def __init__(
        self,
        teacher: Teacher,
        *,
        mode: RuntimeMode,
        audit: TeacherAccessAudit | None = None,
    ) -> None:
        if not isinstance(teacher, Teacher):
            raise TypeError("teacher does not satisfy the Teacher protocol")
        if audit is not None and not isinstance(audit, TeacherAccessAudit):
            raise TypeError("audit must be a TeacherAccessAudit")
        self._teacher = teacher
        self._mode = RuntimeMode(mode)
        self.audit = audit if audit is not None else TeacherAccessAudit()

    @property
    def teacher_id(self) -> str:
        return str(self._teacher.teacher_id)

    @property
    def mode(self) -> RuntimeMode:
        return self._mode

    def for_phase(self, phase: TeacherAccessPhase) -> TeacherAccessView:
        return TeacherAccessView(self, TeacherAccessPhase(phase))

    def action_selection(self) -> TeacherAccessView:
        return self.for_phase(TeacherAccessPhase.ACTION_SELECTION)

    def offline_training(self) -> TeacherAccessView:
        return self.for_phase(TeacherAccessPhase.OFFLINE_TRAINING)

    def _authorize(
        self,
        *,
        method: str,
        phase: TeacherAccessPhase,
        task_id: str,
        stage: int | None,
        state_id: str,
    ) -> None:
        phase = TeacherAccessPhase(phase)
        allowed = not (
            phase is TeacherAccessPhase.ACTION_SELECTION
            and self.mode is not RuntimeMode.COMPAT_ASSISTED
        )
        reason = (
            "compatibility assistance explicitly enabled"
            if allowed and phase is TeacherAccessPhase.ACTION_SELECTION
            else "offline teacher access"
            if allowed
            else f"{self.mode.value} mode forbids teacher reads during action selection"
        )
        self.audit.record(
            teacher_id=self.teacher_id,
            method=method,
            phase=phase,
            mode=self.mode,
            allowed=allowed,
            task_id=task_id,
            stage=stage,
            state_id=state_id,
            reason=reason,
        )
        if not allowed:
            raise TeacherAccessViolation(reason)

    def suggest(
        self,
        request: TeacherRequest,
        *,
        phase: TeacherAccessPhase,
    ) -> TeacherResponse | None:
        if not isinstance(request, TeacherRequest):
            raise TypeError("teacher suggest requires a TeacherRequest")
        self._authorize(
            method="suggest",
            phase=phase,
            task_id=request.task_id,
            stage=request.stage,
            state_id=request.state_id,
        )
        response = self._teacher.suggest(request)
        if response is None:
            return response
        if not isinstance(response, TeacherResponse):
            raise TypeError("teacher suggest must return a TeacherResponse or None")
        if response.action is None:
            return response
        legal = {action.key for action in request.candidates}
        if response.action.key not in legal:
            raise ValueError("teacher suggested an action outside the request candidates")
        return response

    def examples(
        self,
        query: TeacherQuery,
        *,
        phase: TeacherAccessPhase,
    ) -> tuple[TeacherExample, ...]:
        if not isinstance(query, TeacherQuery):
            raise TypeError("teacher examples requires a TeacherQuery")
        self._authorize(
            method="examples",
            phase=phase,
            task_id=query.task_id,
            stage=query.stage,
            state_id="",
        )
        returned = self._teacher.examples(query)
        try:
            rows = tuple(returned)
        except TypeError as exc:
            raise TypeError(
                "teacher examples must be an iterable of TeacherExample rows"
            ) from exc
        for index, row in enumerate(rows):
            if not isinstance(row, TeacherExample):
                raise TypeError(
                    f"teacher example row {index} is not a TeacherExample"
                )
            if not isinstance(row.action, Action):
                raise TypeError(
                    f"teacher example row {index} has a non-Action action"
                )
            if row.task_id != query.task_id:
                raise ValueError(
                    f"teacher example row {index} does not match query task"
                )
            if query.stage is not None and row.stage != query.stage:
                raise ValueError(
                    f"teacher example row {index} does not match query stage"
                )
        return rows[: query.limit]

    def assert_no_action_selection_reads(self) -> None:
        if self.audit.action_selection_reads:
            raise AssertionError(
                f"teacher {self.teacher_id!r} was read during action selection"
            )


class NullTeacher:
    """Explicit no-data implementation useful for uniform wiring."""

    teacher_id = "null"

    def suggest(self, request: TeacherRequest) -> TeacherResponse | None:
        return None

    def examples(self, query: TeacherQuery) -> Sequence[TeacherExample]:
        return ()


class TrajectoryTeacher:
    """Exact-state trajectory teacher for explicit compatibility/student use.

    It replaces the legacy package's several hidden demonstration paths with
    one inspectable mapping.  Merely constructing this object cannot influence
    action selection; all reads still pass through :class:`TeacherAccessGuard`.
    """

    def __init__(
        self,
        examples: Sequence[TeacherExample],
        *,
        teacher_id: str = "trajectory",
    ) -> None:
        self.teacher_id = str(teacher_id)
        try:
            self._examples = tuple(examples)
        except TypeError as exc:
            raise TypeError(
                "trajectory examples must be an iterable of TeacherExample rows"
            ) from exc
        for index, example in enumerate(self._examples):
            if not isinstance(example, TeacherExample):
                raise TypeError(
                    f"trajectory example row {index} is not a TeacherExample"
                )
        self._by_state: dict[
            tuple[str, int, str],
            list[TeacherExample],
        ] = {}
        # Stage numbering is recorder bookkeeping and can disagree with the
        # live environment by one step around level boundaries.  Content is a
        # fallback only within that narrow boundary window; identical frames
        # from unrelated stages must never alias teacher actions.
        self._by_content: dict[tuple[str, str], list[TeacherExample]] = {}
        for example in self._examples:
            self._by_state.setdefault(
                (example.task_id, example.stage, example.state_id),
                [],
            ).append(example)
            frame = example.metadata.get("frame")
            if frame is not None:
                content_id = stable_frame_hash(
                    frame,
                    task_id=example.task_id,
                    stage=0,
                )
                self._by_content.setdefault(
                    (example.task_id, content_id),
                    [],
                ).append(example)

    def suggest(self, request: TeacherRequest) -> TeacherResponse | None:
        rows = self._by_state.get(
            (request.task_id, request.stage, request.state_id),
            (),
        )
        exact_match = bool(rows)
        if not rows:
            content_id = request.metadata.get("content_state_id")
            if content_id:
                rows = tuple(
                    row
                    for row in self._by_content.get(
                        (request.task_id, str(content_id)),
                        (),
                    )
                    if abs(int(row.stage) - int(request.stage)) <= 1
                )
                exact_match = False
        legal = {action.key for action in request.candidates}
        supported = [row for row in rows if row.action.key in legal]
        if not supported:
            return TeacherResponse(
                action=None,
                confidence=0.0,
                source=self.teacher_id,
                metadata={"reason": "exact_state_abstention"},
            )
        by_action: dict[tuple[int, int, int], list[TeacherExample]] = {}
        for row in supported:
            by_action.setdefault(row.action.key, []).append(row)
        weighted = [
            (
                sum(max(0.0, row.weight) for row in action_rows),
                action_key,
                action_rows,
            )
            for action_key, action_rows in by_action.items()
        ]
        weighted.sort(key=lambda item: (-item[0], item[1]))
        total = sum(item[0] for item in weighted)
        winning_weight, _winning_key, winning_rows = weighted[0]
        if winning_weight <= 0.0 or total <= 0.0:
            return TeacherResponse(
                action=None,
                confidence=0.0,
                source=self.teacher_id,
                metadata={"reason": "zero_weight_support"},
            )
        best = min(
            winning_rows,
            key=lambda row: (
                row.action.index,
                row.action.y,
                row.action.x,
                row.source,
            ),
        )
        confidence = winning_weight / total
        return TeacherResponse(
            action=best.action,
            confidence=confidence,
            source=self.teacher_id,
            metadata={
                "support": len(supported),
                "action_support": len(winning_rows),
                "action_weight": winning_weight,
                "total_weight": total,
                "exact_state": exact_match,
            },
        )

    def examples(self, query: TeacherQuery) -> Sequence[TeacherExample]:
        rows = [
            row
            for row in self._examples
            if row.task_id == query.task_id
            and (query.stage is None or row.stage == query.stage)
        ]
        return tuple(rows[: query.limit])

    @classmethod
    def from_npz(
        cls,
        path: str,
        *,
        task_id: str,
        teacher_id: str = "trajectory_npz",
        available_actions: Sequence[int] | None = None,
    ) -> "TrajectoryTeacher":
        """Load a recorded trajectory as an exact-state teacher.

        Recorded NPZ files carry only frames, actions, and levels, while a
        live ``Observation.state_id`` also hashes the available action set
        and progress.  ``available_actions`` therefore declares the
        recording's action inventory; when omitted it is inferred as the set
        of action indices the demonstrator actually used.  The inventory
        serves two roles: state ids become live-comparable (exact-index
        matches are reachable whenever the live action set equals the
        declared one; otherwise the content fallback still applies), and
        distillation receives the true legal alternative set — an action
        missing from it never receives negative updates, which measurably
        collapses the student head onto that action (wa30: the one action
        outside the default provider set accumulated 158 positive and zero
        negative examples and won every argmax).  Progress is recovered from
        ``levels``.
        """
        with np.load(path, allow_pickle=False) as data:
            required = {"frames", "actions"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"trajectory NPZ is missing keys: {missing}")
            frames = _categorical_trajectory_frames(
                data["frames"],
                name="frames",
            )
            actions = np.asarray(data["actions"])
            levels = (
                np.asarray(data["levels"]).reshape(-1)
                if "levels" in data.files
                else np.zeros(len(frames), dtype=np.int64)
            )
            after_frames = (
                _categorical_trajectory_frames(
                    data["frames_after"],
                    name="frames_after",
                )
                if "frames_after" in data.files
                else None
            )
        if actions.ndim != 2 or actions.shape[1] != 3:
            raise ValueError("trajectory actions must have shape [N,3]")
        if (
            np.issubdtype(actions.dtype, np.bool_)
            or not np.issubdtype(actions.dtype, np.integer)
        ):
            raise ValueError("trajectory actions must contain integer values")
        int64_info = np.iinfo(np.int64)
        if actions.size and (
            int(actions.min()) < int64_info.min
            or int(actions.max()) > int64_info.max
        ):
            raise ValueError("trajectory action values exceed int64 range")
        actions = np.ascontiguousarray(actions.astype(np.int64, copy=False))
        if len(frames) != len(actions):
            raise ValueError("trajectory frames/actions length mismatch")
        if (
            np.issubdtype(levels.dtype, np.bool_)
            or not np.issubdtype(levels.dtype, np.integer)
        ):
            raise ValueError("trajectory levels must contain integer values")
        if levels.size and (
            int(levels.min()) < int64_info.min
            or int(levels.max()) > int64_info.max
        ):
            raise ValueError("trajectory level values exceed int64 range")
        levels = np.ascontiguousarray(levels.astype(np.int64, copy=False))
        if len(levels) != len(frames):
            raise ValueError("trajectory levels length mismatch")
        if len(levels) and int(levels.min()) < 0:
            raise ValueError("trajectory levels must be non-negative")
        if len(levels) > 1 and bool(np.any(np.diff(levels) < 0)):
            raise ValueError("trajectory levels must be non-decreasing")
        if after_frames is not None and after_frames.shape != frames.shape:
            raise ValueError("frames_after must match frames shape")
        used_actions = tuple(
            sorted({int(value) for value in actions[:, 0]})
        )
        if any(action_index < 0 for action_index in used_actions):
            raise ValueError(
                "trajectory contains a negative demonstrated action index"
        )
        if available_actions is not None:
            normalized_actions: list[int] = []
            for index, value in enumerate(available_actions):
                normalized_actions.append(
                    _strict_integer(
                        f"available_actions[{index}]",
                        value,
                        minimum=0,
                    )
                )
            declared_actions = tuple(dict.fromkeys(normalized_actions))
            omitted = tuple(
                sorted(set(used_actions) - set(declared_actions))
            )
            if omitted:
                raise ValueError(
                    "declared available_actions omit demonstrated action "
                    f"indices: {list(omitted)}"
                )
        else:
            declared_actions = used_actions
        if not declared_actions:
            raise ValueError("trajectory declares no usable action indices")
        examples: list[TeacherExample] = []
        for index in range(len(frames)):
            stage = int(levels[index]) + 1
            progress_delta = (
                max(0, int(levels[index + 1]) - int(levels[index]))
                if index + 1 < len(levels)
                else 0
            )
            successor = (
                stable_frame_hash(
                    after_frames[index],
                    task_id=str(task_id),
                    stage=stage + progress_delta,
                    available_actions=declared_actions,
                    progress=float(levels[index]) + progress_delta,
                )
                if after_frames is not None
                else None
            )
            examples.append(
                TeacherExample(
                    task_id=str(task_id),
                    stage=stage,
                    state_id=stable_frame_hash(
                        frames[index],
                        task_id=str(task_id),
                        stage=stage,
                        available_actions=declared_actions,
                        progress=float(levels[index]),
                    ),
                    action=Action(
                        int(actions[index, 0]),
                        int(actions[index, 1]),
                        int(actions[index, 2]),
                    ),
                    successor_state_id=successor,
                    source=teacher_id,
                    metadata={
                        "row": index,
                        "available_actions": declared_actions,
                        "frame": readonly_array(frames[index]),
                        "frame_after": (
                            readonly_array(after_frames[index])
                            if after_frames is not None
                            else None
                        ),
                        "progress": float(levels[index]),
                        "progress_delta": float(progress_delta),
                    },
                )
            )
        return cls(examples, teacher_id=teacher_id)


__all__ = [
    "NullTeacher",
    "Teacher",
    "TeacherAccessAudit",
    "TeacherAccessGuard",
    "TeacherAccessPhase",
    "TeacherAccessRecord",
    "TeacherAccessViolation",
    "TeacherAccessView",
    "TeacherExample",
    "TeacherQuery",
    "TeacherRequest",
    "TeacherResponse",
    "TrajectoryTeacher",
]
