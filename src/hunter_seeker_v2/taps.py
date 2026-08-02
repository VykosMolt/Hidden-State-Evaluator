"""Role-scoped sensor taps for Hunter-Seeker v2.

Taps are observations about candidates, not autonomous controllers.  The
public wrappers in this module make calibration, score bounds, and authority
explicit:

* pointwise readings are finite and clipped to declared calibration bounds;
* pairwise readings score both input orders and expose only their
  antisymmetric component;
* survival readings may rank and retain branches, but never choose the final
  action.

The module is intentionally independent of the legacy evaluator package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .contracts import (
    Action,
    Candidate,
    ScoreTerm,
    TapRole,
    WorldSnapshot,
    frozen_mapping,
)
from .policy import effective_risk


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool")
    return bool(value)


def _confidence(value: Any, *, name: str) -> float:
    result = _finite(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie within [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class TapCalibration:
    """Auditable affine calibration followed by a hard score bound.

    ``calibration_id`` identifies the calibration dataset/artifact and
    ``sample_count`` prevents an uncalibrated tap from being represented as a
    policy-capable sensor.  Pairwise wrappers additionally require a symmetric
    bound and zero offset so antisymmetry cannot be broken by calibration.
    """

    calibration_id: str
    sample_count: int
    lower_bound: float
    upper_bound: float
    scale: float = 1.0
    offset: float = 0.0
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        calibration_id = str(self.calibration_id).strip()
        if not calibration_id:
            raise ValueError("calibration_id must name a calibration artifact")
        sample_count = _integer(
            self.sample_count,
            name="sample_count",
            minimum=1,
        )

        lower = _finite(self.lower_bound, name="lower_bound")
        upper = _finite(self.upper_bound, name="upper_bound")
        scale = _finite(self.scale, name="calibration scale")
        offset = _finite(self.offset, name="calibration offset")
        if lower >= upper:
            raise ValueError("lower_bound must be smaller than upper_bound")
        if scale <= 0.0:
            raise ValueError("calibration scale must be positive")

        object.__setattr__(self, "calibration_id", calibration_id)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(
            self,
            "confidence",
            _confidence(self.confidence, name="calibration confidence"),
        )
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))

    def apply(self, raw_value: float) -> float:
        calibrated = (
            self.scale * _finite(raw_value, name="raw tap value")
            + self.offset
        )
        return float(np.clip(calibrated, self.lower_bound, self.upper_bound))

    def apply_antisymmetric(self, raw_margin: float) -> float:
        """Calibrate a relational margin without introducing a bias term."""

        if self.offset != 0.0:
            raise ValueError("pairwise calibration offset must be exactly zero")
        if self.lower_bound != -self.upper_bound:
            raise ValueError("pairwise calibration bounds must be symmetric")
        calibrated = self.scale * _finite(
            raw_margin,
            name="raw pairwise margin",
        )
        bound = self.upper_bound
        return float(min(max(calibrated, -bound), bound))


@dataclass(frozen=True, slots=True)
class PointwiseReading:
    """Finite, bounded and provenance-carrying candidate reading."""

    tap_id: str
    role: TapRole
    raw_value: float
    value: float
    lower_bound: float
    upper_bound: float
    confidence: float
    calibration_id: str
    calibration_samples: int
    influences_score: bool
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        tap_id = str(self.tap_id).strip()
        if not tap_id:
            raise ValueError("tap_id must be non-empty")
        role = TapRole(self.role)
        if role is TapRole.PAIRWISE:
            raise ValueError("pairwise taps must use PairwiseReading")
        influences_score = _boolean(
            self.influences_score,
            name="influences_score",
        )
        if role in {TapRole.SURVIVAL, TapRole.DIAGNOSTIC} and influences_score:
            raise ValueError(f"{role.value} taps cannot directly influence score")

        lower = _finite(self.lower_bound, name="reading lower_bound")
        upper = _finite(self.upper_bound, name="reading upper_bound")
        if lower >= upper:
            raise ValueError("reading bounds are invalid")
        if role in {TapRole.HAZARD, TapRole.UNCERTAINTY, TapRole.SURVIVAL}:
            if lower < 0.0 or upper > 1.0:
                raise ValueError(f"{role.value} readings must be bounded within [0, 1]")

        object.__setattr__(self, "tap_id", tap_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "raw_value",
            _finite(self.raw_value, name="reading raw_value"),
        )
        object.__setattr__(
            self,
            "value",
            float(
                np.clip(
                    _finite(self.value, name="reading value"),
                    lower,
                    upper,
                )
            ),
        )
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(
            self,
            "confidence",
            _confidence(self.confidence, name="reading confidence"),
        )
        calibration_id = str(self.calibration_id).strip()
        calibration_samples = _integer(
            self.calibration_samples,
            name="calibration_samples",
        )
        if influences_score and (not calibration_id or calibration_samples <= 0):
            raise ValueError(
                "score-influencing pointwise readings require calibration provenance"
            )
        object.__setattr__(self, "calibration_id", calibration_id)
        object.__setattr__(self, "calibration_samples", calibration_samples)
        object.__setattr__(self, "influences_score", influences_score)
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))


@runtime_checkable
class PointwiseRawScorer(Protocol):
    def __call__(
        self,
        snapshot: WorldSnapshot,
        candidate: Candidate,
    ) -> float:
        """Return an uncalibrated scalar for one candidate."""


@runtime_checkable
class PointwiseTap(Protocol):
    tap_id: str
    role: TapRole

    def read(
        self,
        snapshot: WorldSnapshot,
        candidate: Candidate,
    ) -> PointwiseReading:
        """Read one candidate without selecting an action."""


@dataclass(frozen=True, slots=True)
class CalibratedPointwiseTap:
    """Small adapter that turns any pure scorer into a bounded tap."""

    tap_id: str
    role: TapRole
    calibration: TapCalibration
    scorer: PointwiseRawScorer
    influences_score: bool = True
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)

    def __post_init__(self) -> None:
        tap_id = str(self.tap_id).strip()
        if not tap_id:
            raise ValueError("tap_id must be non-empty")
        role = TapRole(self.role)
        if role is TapRole.PAIRWISE:
            raise ValueError("use AntisymmetricPairwiseTap for pairwise sensors")
        if not callable(self.scorer):
            raise TypeError("scorer must be callable")
        if not isinstance(self.calibration, TapCalibration):
            raise TypeError("calibration must be a TapCalibration")
        influences_score = _boolean(
            self.influences_score,
            name="influences_score",
        )
        if role in {TapRole.SURVIVAL, TapRole.DIAGNOSTIC} and influences_score:
            raise ValueError(f"{role.value} taps cannot directly influence score")
        if role in {TapRole.HAZARD, TapRole.UNCERTAINTY, TapRole.SURVIVAL}:
            if (
                self.calibration.lower_bound < 0.0
                or self.calibration.upper_bound > 1.0
            ):
                raise ValueError(f"{role.value} calibration must lie within [0, 1]")

        object.__setattr__(self, "tap_id", tap_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "influences_score", influences_score)
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))

    def read(
        self,
        snapshot: WorldSnapshot,
        candidate: Candidate,
    ) -> PointwiseReading:
        raw_value = _finite(
            self.scorer(snapshot, candidate),
            name=f"tap {self.tap_id!r} raw value",
        )
        return PointwiseReading(
            tap_id=self.tap_id,
            role=self.role,
            raw_value=raw_value,
            value=self.calibration.apply(raw_value),
            lower_bound=self.calibration.lower_bound,
            upper_bound=self.calibration.upper_bound,
            confidence=self.calibration.confidence,
            calibration_id=self.calibration.calibration_id,
            calibration_samples=self.calibration.sample_count,
            influences_score=self.influences_score,
            metadata=self.metadata,
        )


@runtime_checkable
class PairwiseRawScorer(Protocol):
    def __call__(
        self,
        snapshot: WorldSnapshot,
        left: Candidate,
        right: Candidate,
    ) -> float:
        """Return a directional uncalibrated pair score."""


@dataclass(frozen=True, slots=True)
class PairwiseReading:
    """Public relational reading after strict two-order antisymmetrization."""

    tap_id: str
    left_action: Action
    right_action: Action
    forward_raw: float
    reverse_raw: float
    value: float
    bound: float
    confidence: float
    calibration_id: str
    calibration_samples: int
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)
    role: TapRole = TapRole.PAIRWISE

    def __post_init__(self) -> None:
        tap_id = str(self.tap_id).strip()
        if not tap_id:
            raise ValueError("tap_id must be non-empty")
        if not isinstance(self.left_action, Action) or not isinstance(
            self.right_action,
            Action,
        ):
            raise TypeError("pairwise reading actions must be Action values")
        bound = _finite(self.bound, name="pairwise bound")
        if bound <= 0.0:
            raise ValueError("pairwise bound must be positive")
        object.__setattr__(self, "tap_id", tap_id)
        object.__setattr__(self, "role", TapRole.PAIRWISE)
        object.__setattr__(
            self,
            "forward_raw",
            _finite(self.forward_raw, name="pairwise forward_raw"),
        )
        object.__setattr__(
            self,
            "reverse_raw",
            _finite(self.reverse_raw, name="pairwise reverse_raw"),
        )
        object.__setattr__(
            self,
            "value",
            float(
                np.clip(
                    _finite(self.value, name="pairwise value"),
                    -bound,
                    bound,
                )
            ),
        )
        object.__setattr__(self, "bound", bound)
        object.__setattr__(
            self,
            "confidence",
            _confidence(self.confidence, name="pairwise confidence"),
        )
        calibration_id = str(self.calibration_id).strip()
        calibration_samples = _integer(
            self.calibration_samples,
            name="calibration_samples",
            minimum=1,
        )
        if not calibration_id or calibration_samples <= 0:
            raise ValueError("pairwise readings require calibration provenance")
        object.__setattr__(self, "calibration_id", calibration_id)
        object.__setattr__(self, "calibration_samples", calibration_samples)
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))

    def flipped(self) -> "PairwiseReading":
        """Return the exact public reading for the reversed ordering."""

        return PairwiseReading(
            tap_id=self.tap_id,
            left_action=self.right_action,
            right_action=self.left_action,
            forward_raw=self.reverse_raw,
            reverse_raw=self.forward_raw,
            value=-self.value,
            bound=self.bound,
            confidence=self.confidence,
            calibration_id=self.calibration_id,
            calibration_samples=self.calibration_samples,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class AntisymmetricPairwiseTap:
    """Evaluate both directions and expose ``(s(a,b)-s(b,a))/2`` only."""

    tap_id: str
    calibration: TapCalibration
    scorer: PairwiseRawScorer
    metadata: Mapping[str, Any] = field(default_factory=frozen_mapping)
    role: TapRole = TapRole.PAIRWISE

    def __post_init__(self) -> None:
        tap_id = str(self.tap_id).strip()
        if not tap_id:
            raise ValueError("tap_id must be non-empty")
        if not callable(self.scorer):
            raise TypeError("scorer must be callable")
        if not isinstance(self.calibration, TapCalibration):
            raise TypeError("calibration must be a TapCalibration")
        # Validate symmetry eagerly rather than discovering it during policy use.
        self.calibration.apply_antisymmetric(0.0)
        object.__setattr__(self, "tap_id", tap_id)
        object.__setattr__(self, "role", TapRole.PAIRWISE)
        object.__setattr__(self, "metadata", frozen_mapping(self.metadata))

    def compare(
        self,
        snapshot: WorldSnapshot,
        left: Candidate,
        right: Candidate,
    ) -> PairwiseReading:
        forward = _finite(
            self.scorer(snapshot, left, right),
            name=f"tap {self.tap_id!r} forward value",
        )
        reverse = _finite(
            self.scorer(snapshot, right, left),
            name=f"tap {self.tap_id!r} reverse value",
        )
        raw_margin = 0.5 * (forward - reverse)
        value = self.calibration.apply_antisymmetric(raw_margin)
        return PairwiseReading(
            tap_id=self.tap_id,
            left_action=left.action,
            right_action=right.action,
            forward_raw=forward,
            reverse_raw=reverse,
            value=value,
            bound=self.calibration.upper_bound,
            confidence=self.calibration.confidence,
            calibration_id=self.calibration.calibration_id,
            calibration_samples=self.calibration.sample_count,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class SurvivalRank:
    candidate: Candidate
    reading: PointwiseReading
    input_index: int


@dataclass(frozen=True, slots=True)
class SurvivalRetention:
    """A pruned candidate set with no final-action field or selection method."""

    retained: tuple[Candidate, ...]
    dropped: tuple[Candidate, ...]
    ranking: tuple[SurvivalRank, ...]
    top_k: int


@dataclass(frozen=True, slots=True)
class SurvivalRetainer:
    """Use a survival tap only as a branch-retention prefilter.

    Retaining a single branch from a multi-candidate set would be equivalent to
    final selection, so that configuration is rejected.
    """

    tap: PointwiseTap

    def __post_init__(self) -> None:
        if TapRole(self.tap.role) is not TapRole.SURVIVAL:
            raise ValueError("SurvivalRetainer requires a survival-role tap")

    def rank(
        self,
        snapshot: WorldSnapshot,
        candidates: Sequence[Candidate],
    ) -> tuple[SurvivalRank, ...]:
        rows = [
            SurvivalRank(
                candidate=candidate,
                reading=self.tap.read(snapshot, candidate),
                input_index=index,
            )
            for index, candidate in enumerate(candidates)
        ]
        for row in rows:
            if row.reading.role is not TapRole.SURVIVAL:
                raise ValueError("survival tap returned a non-survival reading")
            if row.reading.influences_score:
                raise ValueError("survival readings cannot influence final score")
        rows.sort(key=lambda row: (-row.reading.value, row.input_index))
        return tuple(rows)

    def retain(
        self,
        snapshot: WorldSnapshot,
        candidates: Sequence[Candidate],
        *,
        top_k: int,
    ) -> SurvivalRetention:
        candidate_tuple = tuple(candidates)
        requested = _integer(top_k, name="top_k", minimum=1)
        if len(candidate_tuple) > 1 and requested < 2:
            raise ValueError(
                "a survival tap cannot retain only one branch from multiple candidates"
            )

        ranking = self.rank(snapshot, candidate_tuple)
        retained_count = min(requested, len(ranking))
        retained = tuple(row.candidate for row in ranking[:retained_count])
        dropped = tuple(row.candidate for row in ranking[retained_count:])
        return SurvivalRetention(
            retained=retained,
            dropped=dropped,
            ranking=ranking,
            top_k=retained_count,
        )


@dataclass(frozen=True, slots=True)
class TapBundle:
    """Role-aware tap wiring used by search.

    Pointwise and pairwise readings contribute only small bounded terms.
    Survival remains a separate top-k retention operation and never exposes a
    final selector.
    """

    pointwise: tuple[PointwiseTap, ...] = ()
    pairwise: tuple[AntisymmetricPairwiseTap, ...] = ()
    survival: SurvivalRetainer | None = None
    pairwise_weight: float = 0.05

    def __post_init__(self) -> None:
        weight = _finite(self.pairwise_weight, name="pairwise_weight")
        if not 0.0 <= weight <= 0.25:
            raise ValueError("pairwise_weight must be finite and within [0, 0.25]")
        pointwise = tuple(self.pointwise)
        pairwise = tuple(self.pairwise)
        if not all(isinstance(tap, PointwiseTap) for tap in pointwise):
            raise TypeError("pointwise taps must satisfy the PointwiseTap protocol")
        if not all(
            isinstance(tap, AntisymmetricPairwiseTap)
            for tap in pairwise
        ):
            raise TypeError("pairwise taps must be AntisymmetricPairwiseTap values")
        if self.survival is not None and not isinstance(
            self.survival,
            SurvivalRetainer,
        ):
            raise TypeError("survival must be a SurvivalRetainer")
        tap_ids = [
            str(tap.tap_id)
            for tap in pointwise
        ] + [
            str(tap.tap_id)
            for tap in pairwise
        ]
        if self.survival is not None:
            tap_ids.append(str(self.survival.tap.tap_id))
        duplicate_ids = sorted(
            tap_id for tap_id in set(tap_ids) if tap_ids.count(tap_id) > 1
        )
        if duplicate_ids:
            raise ValueError(f"tap ids must be unique within a bundle: {duplicate_ids}")
        object.__setattr__(self, "pointwise", pointwise)
        object.__setattr__(self, "pairwise", pairwise)
        object.__setattr__(self, "pairwise_weight", weight)

    def pointwise_values(
        self,
        snapshot: WorldSnapshot,
        candidate: Candidate,
    ) -> Mapping[str, float]:
        values: dict[str, float] = {}
        for tap in self.pointwise:
            reading = tap.read(snapshot, candidate)
            if not isinstance(reading, PointwiseReading):
                raise TypeError("pointwise taps must return PointwiseReading values")
            if reading.tap_id != tap.tap_id or reading.role is not TapRole(tap.role):
                raise ValueError("pointwise reading provenance does not match its tap")
            if not reading.influences_score:
                continue
            sign = -1.0 if reading.role is TapRole.HAZARD else 1.0
            values[f"{reading.role.value}:{reading.tap_id}"] = (
                sign
                * float(reading.value)
                * float(reading.confidence)
            )
        return frozen_mapping(values)

    def apply_pairwise(
        self,
        snapshot: WorldSnapshot,
        candidates: Sequence[Candidate],
    ) -> tuple[Candidate, ...]:
        rows = tuple(candidates)
        if not self.pairwise or len(rows) < 2:
            return rows
        margins = np.zeros(len(rows), dtype=np.float64)
        comparisons = np.zeros(len(rows), dtype=np.float64)
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                for tap in self.pairwise:
                    reading = tap.compare(
                        snapshot,
                        rows[left_index],
                        rows[right_index],
                    )
                    contribution = (
                        float(reading.value) / float(reading.bound)
                        * float(reading.confidence)
                        * float(self.pairwise_weight)
                    )
                    margins[left_index] += contribution
                    margins[right_index] -= contribution
                    comparisons[left_index] += 1.0
                    comparisons[right_index] += 1.0
        result: list[Candidate] = []
        for index, candidate in enumerate(rows):
            margin = (
                float(margins[index] / comparisons[index])
                if comparisons[index] > 0
                else 0.0
            )
            result.append(
                Candidate(
                    action=candidate.action,
                    prediction=candidate.prediction,
                    terms=candidate.terms
                    + (
                        ScoreTerm(
                            "tap:pairwise",
                            margin,
                            "tap",
                            source="antisymmetric_pair_tap",
                        ),
                    ),
                    depth=candidate.depth,
                    path=candidate.path,
                    source=candidate.source,
                )
            )
        return tuple(result)

    def retain(
        self,
        snapshot: WorldSnapshot,
        candidates: Sequence[Candidate],
        *,
        top_k: int,
    ) -> tuple[Candidate, ...]:
        rows = tuple(candidates)
        requested = _integer(top_k, name="top_k", minimum=1)
        if self.survival is None or len(rows) <= max(2, requested):
            return rows
        retained = self.survival.retain(
            snapshot,
            rows,
            top_k=max(2, requested),
        ).retained
        # A survival sensor is a branch-efficiency hint, not a safety gate.
        # Preserve the candidate the final arbiter would regard as globally
        # least risky; otherwise a high survival rank can delete every safe
        # action before the one authoritative risk boundary sees the set.
        least_risk = min(
            rows,
            key=lambda candidate: (
                effective_risk(candidate),
                -candidate.score,
                candidate.action.index,
                candidate.action.y,
                candidate.action.x,
            ),
        )
        if any(candidate is least_risk for candidate in retained):
            return retained
        return retained[:-1] + (least_risk,)


__all__ = [
    "AntisymmetricPairwiseTap",
    "CalibratedPointwiseTap",
    "PairwiseRawScorer",
    "PairwiseReading",
    "PointwiseRawScorer",
    "PointwiseReading",
    "PointwiseTap",
    "SurvivalRank",
    "SurvivalRetainer",
    "SurvivalRetention",
    "TapBundle",
    "TapCalibration",
]
