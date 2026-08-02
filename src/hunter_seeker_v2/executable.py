"""Replay-verified executable world models.

Executable models are compact rule systems that can predict and plan from the
current state.  They receive :class:`ExecutableState`, never a ``Transition``
or future observation.  The registry alone performs replay verification and
will not expose prediction or planning from a model until its declared version
passes the configured coverage and error gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import blake2b
import marshal
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .contracts import (
    Action,
    ObjectState,
    Representation,
    Topology,
    Transition,
    WorldSnapshot,
    frozen_mapping,
    readonly_array,
)


def _finite(value: float, *, fallback: float = 0.0) -> float:
    result = float(value)
    return result if np.isfinite(result) else float(fallback)


def _required_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _digest_value(value: Any) -> str:
    """Deterministically fingerprint behavior-bearing Python/model state."""

    digest = blake2b(digest_size=32)
    active: set[int] = set()

    def update(item: Any) -> None:
        if item is None:
            digest.update(b"none;")
            return
        if isinstance(item, bool):
            digest.update(b"bool:1;" if item else b"bool:0;")
            return
        if isinstance(item, (int, str, bytes)):
            payload = item if isinstance(item, bytes) else str(item).encode("utf-8")
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b":")
            digest.update(str(len(payload)).encode("ascii"))
            digest.update(b":")
            digest.update(payload)
            digest.update(b";")
            return
        if isinstance(item, float):
            digest.update(f"float:{item.hex()};".encode("ascii"))
            return
        if isinstance(item, np.generic):
            update(item.item())
            return
        if isinstance(item, np.ndarray):
            array = np.asarray(item)
            digest.update(b"ndarray:")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(b":")
            digest.update(repr(tuple(int(v) for v in array.shape)).encode("ascii"))
            digest.update(b":")
            if array.dtype.hasobject:
                update(array.tolist())
            else:
                digest.update(np.ascontiguousarray(array).tobytes())
            digest.update(b";")
            return

        identity = id(item)
        if identity in active:
            digest.update(f"cycle:{_qualified_type(item)};".encode("utf-8"))
            return
        active.add(identity)
        try:
            if isinstance(item, Mapping):
                digest.update(b"mapping{")
                ordered = sorted(
                    item.items(),
                    key=lambda row: _digest_value(row[0]),
                )
                for key, child in ordered:
                    update(key)
                    update(child)
                digest.update(b"}")
                return
            if isinstance(item, (list, tuple)):
                digest.update(type(item).__name__.encode("ascii") + b"[")
                for child in item:
                    update(child)
                digest.update(b"]")
                return
            if isinstance(item, (set, frozenset)):
                digest.update(type(item).__name__.encode("ascii") + b"{")
                for child_digest in sorted(_digest_value(child) for child in item):
                    digest.update(child_digest.encode("ascii"))
                digest.update(b"}")
                return
            if is_dataclass(item) and not isinstance(item, type):
                digest.update(f"dataclass:{_qualified_type(item)}{{".encode("utf-8"))
                for row in fields(item):
                    update(row.name)
                    update(getattr(item, row.name))
                digest.update(b"}")
                return

            function = getattr(item, "__func__", item)
            code = getattr(function, "__code__", None)
            if code is not None:
                digest.update(
                    f"callable:{getattr(function, '__module__', '')}:"
                    f"{getattr(function, '__qualname__', '')}:".encode("utf-8")
                )
                digest.update(marshal.dumps(code))
                update(getattr(function, "__defaults__", None))
                update(getattr(function, "__kwdefaults__", None))
                closure = getattr(function, "__closure__", None)
                if closure:
                    for cell in closure:
                        try:
                            update(cell.cell_contents)
                        except ValueError:
                            digest.update(b"empty-cell;")
                return

            attributes = getattr(item, "__dict__", None)
            if isinstance(attributes, Mapping):
                digest.update(f"object:{_qualified_type(item)}:".encode("utf-8"))
                update(attributes)
                return
            digest.update(f"stateless:{_qualified_type(item)};".encode("utf-8"))
        finally:
            active.remove(identity)

    update(value)
    return digest.hexdigest()


def _transient_behavior_fields(model: Any) -> tuple[str, ...]:
    raw = getattr(model, "verification_transient_fields", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("verification_transient_fields must be a sequence of names")
    return tuple(sorted(dict.fromkeys(str(name) for name in raw)))


def _default_behavior_state(
    model: Any,
    *,
    transient_fields: tuple[str, ...],
) -> Mapping[str, Any]:
    """Best-effort state projection for models without an explicit signature.

    Stateful models with caches or instrumentation should expose
    ``verification_signature`` (or a behavior-only ``state_dict``). As a
    narrow fallback they may explicitly name non-behavioral attributes in
    ``verification_transient_fields``; undeclared state is fail-closed.
    """

    attributes = getattr(model, "__dict__", None)
    if not isinstance(attributes, Mapping):
        return {}
    excluded = frozenset(transient_fields)
    return {
        str(name): value
        for name, value in attributes.items()
        if str(name)
        not in {
            "verification_signature",
            "verification_transient_fields",
            *excluded,
        }
    }


def _model_behavior_signature(model: Any) -> str:
    transient_fields = _transient_behavior_fields(model)
    declared = getattr(model, "verification_signature", None)
    if declared is not None:
        state = declared() if callable(declared) else declared
        state_source = "verification_signature"
    else:
        state_fn = getattr(model, "state_dict", None)
        if callable(state_fn):
            state = state_fn()
            state_source = "state_dict"
        else:
            state = _default_behavior_state(
                model,
                transient_fields=transient_fields,
            )
            state_source = "attributes"
    return _digest_value(
        {
            "type": _qualified_type(model),
            "model_id": str(model.model_id).strip(),
            "model_version": str(model.model_version).strip(),
            "complexity": float(model.complexity),
            "state_source": state_source,
            "state": state,
            "transient_fields": transient_fields,
            "predict": getattr(model, "predict"),
            "plan": getattr(model, "plan"),
        }
    )


@dataclass(frozen=True, slots=True)
class ExecutableState:
    """Current-state-only input exposed to executable models."""

    state_id: str
    task_id: str
    stage: int
    step: int
    frame: np.ndarray
    available_actions: tuple[int, ...]
    objects: tuple[ObjectState, ...]
    topology: Topology
    representation: Representation
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", str(self.state_id))
        object.__setattr__(self, "task_id", str(self.task_id))
        object.__setattr__(self, "stage", max(1, int(self.stage)))
        object.__setattr__(self, "step", max(0, int(self.step)))
        object.__setattr__(self, "frame", readonly_array(self.frame))
        object.__setattr__(
            self,
            "available_actions",
            tuple(dict.fromkeys(int(action) for action in self.available_actions)),
        )
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))

    @classmethod
    def from_snapshot(cls, snapshot: WorldSnapshot) -> "ExecutableState":
        """Strip a durable snapshot to the current-state executable view."""

        return cls(
            state_id=snapshot.state_id,
            task_id=snapshot.observation.task_id,
            stage=snapshot.observation.stage,
            step=snapshot.step,
            frame=snapshot.observation.frame,
            available_actions=snapshot.observation.available_actions,
            objects=snapshot.objects,
            topology=snapshot.topology,
            representation=snapshot.representation,
            metadata={
                "observation_progress": snapshot.observation.progress,
            },
        )


@dataclass(frozen=True, slots=True)
class ExecutablePrediction:
    """One rule-model successor prediction."""

    applicable: bool
    successor_state_id: str | None = None
    change_probability: float = 0.0
    progress_delta: float = 0.0
    hazard: float = 0.0
    terminal: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        applicable = bool(self.applicable)
        successor = (
            None if self.successor_state_id is None else str(self.successor_state_id)
        )
        if applicable and not successor:
            raise ValueError(
                "an applicable executable prediction requires a successor_state_id"
            )
        object.__setattr__(self, "applicable", applicable)
        object.__setattr__(self, "successor_state_id", successor)
        object.__setattr__(
            self,
            "change_probability",
            float(np.clip(_finite(self.change_probability), 0.0, 1.0)),
        )
        object.__setattr__(self, "progress_delta", _finite(self.progress_delta))
        object.__setattr__(
            self,
            "hazard",
            float(np.clip(_finite(self.hazard, fallback=1.0), 0.0, 1.0)),
        )
        object.__setattr__(
            self,
            "terminal",
            float(np.clip(_finite(self.terminal, fallback=1.0), 0.0, 1.0)),
        )
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ExecutablePlan:
    """Plan returned by a verified executable model."""

    model_id: str
    model_version: str
    actions: tuple[Action, ...]
    score: float = 0.0
    predicted_state_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        model_id = str(self.model_id).strip()
        model_version = str(self.model_version).strip()
        if not model_id or not model_version:
            raise ValueError("plan must identify its model and version")
        actions = tuple(self.actions)
        predicted = tuple(str(state_id) for state_id in self.predicted_state_ids)
        if predicted and len(predicted) != len(actions):
            raise ValueError(
                "predicted_state_ids must be empty or align one-to-one with actions"
            )
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "score", _finite(self.score))
        object.__setattr__(self, "predicted_state_ids", predicted)
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))


@runtime_checkable
class ExecutableModel(Protocol):
    """Current-state-only rule model interface.

    Implementations with mutable behavior should expose a behavior-only
    ``verification_signature`` (or ``state_dict``).  Instrumentation and
    caches may instead be named in ``verification_transient_fields`` so they
    do not revoke an otherwise unchanged verification.
    """

    model_id: str
    model_version: str
    complexity: float

    def predict(
        self,
        state: ExecutableState,
        action: Action,
    ) -> ExecutablePrediction:
        """Predict from current state and action only."""

    def plan(
        self,
        state: ExecutableState,
        actions: Sequence[Action],
        *,
        horizon: int,
    ) -> ExecutablePlan | None:
        """Plan from current state without access to replay successors."""


@dataclass(frozen=True, slots=True)
class ReplayCase:
    """Observed successor known to the verifier, but never passed to a model."""

    case_id: str
    before: ExecutableState
    action: Action
    after_state_id: str
    frame_changed: bool
    progress_delta: float = 0.0
    hazard: float = 0.0
    terminal: bool = False

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        if not case_id:
            raise ValueError("case_id must be non-empty")
        after_state_id = str(self.after_state_id).strip()
        if not after_state_id:
            raise ValueError("after_state_id must be non-empty")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "after_state_id", after_state_id)
        object.__setattr__(self, "frame_changed", bool(self.frame_changed))
        object.__setattr__(self, "progress_delta", _finite(self.progress_delta))
        object.__setattr__(
            self,
            "hazard",
            float(np.clip(_finite(self.hazard, fallback=1.0), 0.0, 1.0)),
        )
        object.__setattr__(self, "terminal", bool(self.terminal))

    @classmethod
    def from_transition(cls, transition: Transition) -> "ReplayCase":
        return cls(
            case_id=transition.transition_id,
            before=ExecutableState.from_snapshot(transition.before),
            action=transition.action,
            after_state_id=transition.after_state_id,
            frame_changed=transition.frame_changed,
            progress_delta=transition.outcome.progress_delta,
            hazard=transition.outcome.hazard,
            terminal=(
                transition.outcome.terminated
                and not transition.outcome.completed
            ),
        )


@dataclass(frozen=True, slots=True)
class ReplayVerificationPolicy:
    minimum_cases: int = 3
    minimum_coverage: float = 0.50
    maximum_error_rate: float = 0.0
    maximum_safety_error: float = 0.0
    maximum_auxiliary_error: float = 0.0

    def __post_init__(self) -> None:
        minimum_cases = int(self.minimum_cases)
        minimum_coverage = _required_finite(
            "minimum_coverage",
            self.minimum_coverage,
        )
        maximum_error_rate = _required_finite(
            "maximum_error_rate",
            self.maximum_error_rate,
        )
        maximum_safety_error = _required_finite(
            "maximum_safety_error",
            self.maximum_safety_error,
        )
        maximum_auxiliary_error = _required_finite(
            "maximum_auxiliary_error",
            self.maximum_auxiliary_error,
        )
        if minimum_cases <= 0:
            raise ValueError("minimum_cases must be positive")
        if not 0.0 <= minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must lie within [0, 1]")
        if not 0.0 <= maximum_error_rate <= 1.0:
            raise ValueError("maximum_error_rate must lie within [0, 1]")
        if not 0.0 <= maximum_safety_error <= 1.0:
            raise ValueError("maximum_safety_error must lie within [0, 1]")
        if not 0.0 <= maximum_auxiliary_error <= 1.0:
            raise ValueError("maximum_auxiliary_error must lie within [0, 1]")
        object.__setattr__(self, "minimum_cases", minimum_cases)
        object.__setattr__(self, "minimum_coverage", minimum_coverage)
        object.__setattr__(self, "maximum_error_rate", maximum_error_rate)
        object.__setattr__(self, "maximum_safety_error", maximum_safety_error)
        object.__setattr__(
            self,
            "maximum_auxiliary_error",
            maximum_auxiliary_error,
        )


@dataclass(frozen=True, slots=True)
class ReplayVerification:
    model_id: str
    model_version: str
    cases: int
    covered: int
    matches: int
    mismatches: int
    coverage: float
    error_rate: float
    mean_auxiliary_error: float
    maximum_safety_error: float
    complexity: float
    verified: bool
    reason: str

    @property
    def rank_key(self) -> tuple[float, float, float, float, float, str]:
        """Prefer coverage, safety, exactness, calibration, then simplicity."""

        return (
            -self.coverage,
            self.maximum_safety_error,
            self.error_rate,
            self.mean_auxiliary_error,
            self.complexity,
            self.model_id,
        )


@dataclass(slots=True)
class _RegistryEntry:
    model: ExecutableModel
    model_id: str
    verification: ReplayVerification | None = None
    behavior_signature: str | None = None


class ExecutableModelError(RuntimeError):
    """Base registry/model error."""


class ExecutableModelNotVerified(ExecutableModelError):
    """Raised when prediction or planning is requested before verification."""


class ExecutableModelRegistry:
    """Register, replay-verify, rank, and safely expose executable models."""

    def __init__(
        self,
        *,
        policy: ReplayVerificationPolicy | None = None,
    ) -> None:
        self.policy = policy or ReplayVerificationPolicy()
        self._entries: dict[str, _RegistryEntry] = {}

    def register(self, model: ExecutableModel) -> None:
        if not isinstance(model, ExecutableModel):
            raise TypeError("model does not satisfy the ExecutableModel protocol")
        model_id = str(model.model_id).strip()
        model_version = str(model.model_version).strip()
        complexity = _finite(model.complexity, fallback=-1.0)
        if not model_id or not model_version:
            raise ValueError("model_id and model_version must be non-empty")
        if complexity < 0.0:
            raise ValueError("model complexity must be finite and non-negative")
        if model_id in self._entries:
            raise ValueError(f"executable model {model_id!r} is already registered")
        self._entries[model_id] = _RegistryEntry(
            model=model,
            model_id=model_id,
        )

    def registered_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def invalidate(self, model_id: str) -> None:
        entry = self._entry(model_id)
        entry.verification = None
        entry.behavior_signature = None

    def verification(self, model_id: str) -> ReplayVerification | None:
        return self._current_verification(self._entry(model_id))

    def verify(
        self,
        model_id: str,
        cases: Sequence[ReplayCase],
    ) -> ReplayVerification:
        entry = self._entry(model_id)
        # Re-verification is fail-closed.  No exception or rejected replay set
        # may leave a previously trusted report available to live planning.
        entry.verification = None
        entry.behavior_signature = None
        model = entry.model
        current_model_id = str(model.model_id).strip()
        current_version = str(model.model_version).strip()
        complexity = _finite(model.complexity, fallback=-1.0)
        if current_model_id != entry.model_id:
            raise ValueError(
                f"registered executable model_id {entry.model_id!r} changed "
                f"to {current_model_id!r}"
            )
        if not current_version:
            raise ValueError("model_version must remain non-empty during verification")
        if complexity < 0.0:
            raise ValueError(
                "model complexity must remain finite and non-negative during verification"
            )
        case_rows = tuple(cases)
        case_ids: set[str] = set()
        for case in case_rows:
            if not isinstance(case, ReplayCase):
                raise TypeError("replay verification requires ReplayCase values")
            if case.case_id in case_ids:
                raise ValueError(
                    f"duplicate replay case_id {case.case_id!r} is not independent coverage"
                )
            case_ids.add(case.case_id)
        covered = 0
        matches = 0
        mismatches = 0
        auxiliary_errors: list[float] = []
        safety_errors: list[float] = []

        for case in case_rows:
            # The model receives only the current-state view and action.
            prediction = model.predict(case.before, case.action)
            if not isinstance(prediction, ExecutablePrediction):
                raise TypeError("executable predict() returned an invalid type")
            if not prediction.applicable:
                continue

            covered += 1
            if prediction.successor_state_id == case.after_state_id:
                matches += 1
            else:
                mismatches += 1
            change_error = abs(
                prediction.change_probability - float(case.frame_changed)
            )
            progress_error = min(
                1.0,
                abs(prediction.progress_delta - case.progress_delta),
            )
            hazard_error = abs(prediction.hazard - case.hazard)
            terminal_error = abs(prediction.terminal - float(case.terminal))
            auxiliary_errors.append(
                float(
                    np.mean(
                        [
                            change_error,
                            progress_error,
                            hazard_error,
                            terminal_error,
                        ]
                    )
                )
            )
            # A single missed hazard or terminal boundary is enough to make a
            # rule model unsafe.  Do not dilute it by other cases or outputs.
            safety_errors.append(float(max(hazard_error, terminal_error)))

        count = len(case_rows)
        coverage = covered / count if count else 0.0
        error_rate = mismatches / covered if covered else 1.0
        mean_auxiliary_error = (
            float(np.mean(auxiliary_errors)) if auxiliary_errors else 1.0
        )
        maximum_safety_error = max(safety_errors, default=1.0)
        # Prediction may be stateful for harmless diagnostics, but the
        # identity/version/complexity that define the verification report
        # cannot change while the replay gate is running.
        if str(model.model_id).strip() != entry.model_id:
            raise ValueError("model_id changed during replay verification")
        if str(model.model_version).strip() != current_version:
            raise ValueError("model_version changed during replay verification")
        if _finite(model.complexity, fallback=-1.0) != complexity:
            raise ValueError("model complexity changed during replay verification")
        if count < self.policy.minimum_cases:
            verified = False
            reason = (
                f"insufficient replay cases: {count} < {self.policy.minimum_cases}"
            )
        elif coverage < self.policy.minimum_coverage:
            verified = False
            reason = (
                f"insufficient replay coverage: {coverage:.6f} "
                f"< {self.policy.minimum_coverage:.6f}"
            )
        elif error_rate > self.policy.maximum_error_rate:
            verified = False
            reason = (
                f"replay error rate too high: {error_rate:.6f} "
                f"> {self.policy.maximum_error_rate:.6f}"
            )
        elif maximum_safety_error > self.policy.maximum_safety_error:
            verified = False
            reason = (
                f"replay safety error too high: {maximum_safety_error:.6f} "
                f"> {self.policy.maximum_safety_error:.6f}"
            )
        elif mean_auxiliary_error > self.policy.maximum_auxiliary_error:
            verified = False
            reason = (
                f"replay auxiliary error too high: {mean_auxiliary_error:.6f} "
                f"> {self.policy.maximum_auxiliary_error:.6f}"
            )
        else:
            verified = True
            reason = "replay verification passed"

        report = ReplayVerification(
            model_id=entry.model_id,
            model_version=current_version,
            cases=count,
            covered=covered,
            matches=matches,
            mismatches=mismatches,
            coverage=coverage,
            error_rate=error_rate,
            mean_auxiliary_error=mean_auxiliary_error,
            maximum_safety_error=maximum_safety_error,
            complexity=complexity,
            verified=verified,
            reason=reason,
        )
        behavior_signature = (
            _model_behavior_signature(model)
            if report.verified
            else None
        )
        entry.verification = report
        entry.behavior_signature = behavior_signature
        return report

    def ranked_verifications(self) -> tuple[ReplayVerification, ...]:
        reports: list[ReplayVerification] = []
        for entry in self._entries.values():
            report = self._current_verification(entry)
            if report is not None and report.verified:
                reports.append(report)
        reports.sort(key=lambda report: report.rank_key)
        return tuple(reports)

    def eligible_models(self) -> tuple[ExecutableModel, ...]:
        return tuple(
            self._entries[report.model_id].model
            for report in self.ranked_verifications()
        )

    def predict(
        self,
        model_id: str,
        state: ExecutableState,
        action: Action,
    ) -> ExecutablePrediction:
        entry = self._verified_entry(model_id)
        try:
            prediction = entry.model.predict(state, action)
        except BaseException:
            entry.verification = None
            entry.behavior_signature = None
            raise
        if not isinstance(prediction, ExecutablePrediction):
            entry.verification = None
            entry.behavior_signature = None
            raise TypeError("executable predict() returned an invalid type")
        self._assert_behavior_unchanged(entry)
        return prediction

    def plan(
        self,
        state: ExecutableState,
        actions: Sequence[Action],
        *,
        horizon: int,
        model_id: str | None = None,
    ) -> ExecutablePlan | None:
        candidate_actions = tuple(actions)
        if int(horizon) <= 0:
            raise ValueError("planning horizon must be positive")
        if not candidate_actions:
            return None

        if model_id is None:
            model_ids = tuple(
                report.model_id for report in self.ranked_verifications()
            )
            if not model_ids:
                return None
        else:
            model_ids = (str(model_id),)

        for candidate_model_id in model_ids:
            entry = self._verified_entry(candidate_model_id)
            try:
                plan = entry.model.plan(
                    state,
                    candidate_actions,
                    horizon=int(horizon),
                )
            except BaseException:
                entry.verification = None
                entry.behavior_signature = None
                raise
            self._assert_behavior_unchanged(entry)
            if plan is None:
                # A ranked model may be verified globally yet inapplicable to
                # this state.  Continue to the next applicable specialist.
                continue
            try:
                return self._validate_plan(
                    entry,
                    plan,
                    candidate_actions=candidate_actions,
                    horizon=int(horizon),
                )
            except BaseException:
                entry.verification = None
                entry.behavior_signature = None
                raise
        return None

    @staticmethod
    def _validate_plan(
        entry: _RegistryEntry,
        plan: ExecutablePlan,
        *,
        candidate_actions: tuple[Action, ...],
        horizon: int,
    ) -> ExecutablePlan:
        if not isinstance(plan, ExecutablePlan):
            raise TypeError("executable plan() returned an invalid type")
        if plan.model_id != entry.model_id:
            raise ExecutableModelError("plan model_id does not match registry model")
        report = entry.verification
        if report is None or plan.model_version != report.model_version:
            raise ExecutableModelError("plan model_version does not match verified model")
        if not plan.actions:
            raise ExecutableModelError("executable plan must contain an action")
        if len(plan.actions) > int(horizon):
            raise ExecutableModelError("executable plan exceeds requested horizon")
        legal_first_actions = {action.key for action in candidate_actions}
        if plan.actions[0].key not in legal_first_actions:
            raise ExecutableModelError(
                "executable plan begins with an action outside the candidate set"
            )
        return plan

    def _entry(self, model_id: str) -> _RegistryEntry:
        key = str(model_id).strip()
        try:
            return self._entries[key]
        except KeyError as exc:
            raise KeyError(f"unknown executable model {key!r}") from exc

    def _current_verification(
        self,
        entry: _RegistryEntry,
    ) -> ReplayVerification | None:
        report = entry.verification
        if report is None:
            return None
        try:
            current_model_id = str(entry.model.model_id).strip()
            current_version = str(entry.model.model_version).strip()
            current_complexity = _finite(
                entry.model.complexity,
                fallback=-1.0,
            )
        except BaseException:
            entry.verification = None
            entry.behavior_signature = None
            return None
        if (
            current_model_id != entry.model_id
            or report.model_id != entry.model_id
            or report.model_version != current_version
            or current_complexity != report.complexity
        ):
            entry.verification = None
            entry.behavior_signature = None
            return None
        if not report.verified:
            return report
        try:
            current_signature = _model_behavior_signature(entry.model)
        except BaseException:
            entry.verification = None
            entry.behavior_signature = None
            return None
        if (
            entry.behavior_signature is None
            or current_signature != entry.behavior_signature
        ):
            entry.verification = None
            entry.behavior_signature = None
            return None
        return report

    def _assert_behavior_unchanged(self, entry: _RegistryEntry) -> None:
        if self._current_verification(entry) is None:
            raise ExecutableModelNotVerified(
                f"executable model {entry.model_id!r} changed after replay "
                "verification"
            )

    def _verified_entry(self, model_id: str) -> _RegistryEntry:
        entry = self._entry(model_id)
        report = self._current_verification(entry)
        if report is None or not report.verified:
            raise ExecutableModelNotVerified(
                f"executable model {model_id!r} has not passed replay verification"
            )
        return entry


__all__ = [
    "ExecutableModel",
    "ExecutableModelError",
    "ExecutableModelNotVerified",
    "ExecutableModelRegistry",
    "ExecutablePlan",
    "ExecutablePrediction",
    "ExecutableState",
    "ReplayCase",
    "ReplayVerification",
    "ReplayVerificationPolicy",
]
