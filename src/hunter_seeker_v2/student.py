"""Compact state-conditioned student policy for Hunter-Seeker v2.

The head in this module is deliberately a learned function of the current
representation and a candidate action.  It never stores or queries exact state
identifiers and it has no reference to a :class:`~hunter_seeker_v2.teacher.Teacher`.
Teacher examples may update it during an explicit offline phase; runtime reads
need only a :class:`~hunter_seeker_v2.contracts.WorldSnapshot` and an action.

The learning API accepts a normalized representation trajectory.  The default
runtime supplies one connector output; future integration may supply early,
middle, and late representations. Labels and realized outcome targets update
every weighted trajectory element, while runtime behavior is read from the
designated terminal representation only. This is merely a trajectory-credit
interface -- the one-step default is not described as RLTT.

Negative updates are task-local.  Positive updates additionally train a small
global row whose bounded contribution is controlled by
``positive_transfer_scale``.  This makes transfer policy explicit instead of
allowing task failures to leak through a global action prior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import Action, Outcome, Representation, Transition, WorldSnapshot
from .teacher import TeacherExample


STUDENT_STATE_VERSION = 1
_CANDIDATE_DIM = 8

# Per-task bandwidth fit (see _fit_task_scale): place the RBF kernel argument
# _SCALE_FIT_KERNEL_ARG at the _SCALE_FIT_QUANTILE quantile of latent
# distances between differently-labeled demonstration states, bounded to
# _SCALE_FIT_BOUNDS.  Calibrated on the trusted trio (2026-07-19): the rule
# reproduces the empirically-best fixed scales per game (ls20 ~10, tr87 ~64)
# from geometry alone.
_SCALE_FIT_QUANTILE = 0.05
_SCALE_FIT_KERNEL_ARG = 2.0
_SCALE_FIT_BOUNDS = (2.0, 120.0)

# Process-wide memoized RFF bases keyed by (state_dim, input width).  Content
# is a pure function of the key and a fixed seed, so sharing across policy
# instances changes no output; keeping it at module level also keeps the
# multi-megabyte bases out of the per-observe transaction deep copy.
_RFF_BASIS_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _finite(value: Any, *, fallback: float = 0.0) -> float:
    result = float(value)
    return result if np.isfinite(result) else float(fallback)


def _loaded_number(
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


def _loaded_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _loaded_action_index(value: Any, *, name: str) -> int:
    """Parse the canonical JSON object key emitted by ``state_dict``."""

    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical integer string")
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical integer string") from exc
    if result < 0 or value != str(result):
        raise ValueError(f"{name} must be a canonical non-negative integer string")
    return result


@dataclass(frozen=True, slots=True)
class StudentPolicyConfig:
    """Configuration for :class:`StateConditionedStudentPolicy`.

    ``score_bound`` bounds the raw head score before ``runtime_weight`` is
    applied; their product bounds the absolute ``student_policy`` score term
    exposed to the controller.  ``positive_transfer_scale=0`` disables all
    cross-task use; negative labels and outcomes are task-scoped for every
    setting.

    ``fit_state_feature_scale`` lets each distillation batch replace
    ``state_feature_scale`` with a per-task bandwidth fitted from that task's
    measured latent geometry (see :meth:`_fit_task_scale`); the config value
    remains the default for tasks without a fit.  ``online_target_weighting``
    scales committed-outcome updates by the magnitude of their outcome
    target, so neutral steps (target zero) no longer erode distilled teacher
    margins by regressing trained rows toward zero.
    """

    # RFF width is a measured capacity lever, not a tuning knob: on the
    # trusted routes, 128 -> 1024 lifted recorded-state argmax agreement on
    # every game (ls20 42->48/54, tr87 84->87/106, wa30 532->749/1556) at
    # ~1.6ms per candidate score, and wa30's residual retention gap scales
    # monotonically with this width while being flat in feature content
    # (gate-4 ablation).
    state_dim: int = 1024
    state_feature_scale: float = 8.0
    learning_rate: float = 0.25
    l2: float = 1e-4
    score_bound: float = 1.0
    runtime_weight: float = 3.0
    confidence_prior: float = 2.0
    max_weight_norm: float = 8.0
    positive_transfer_scale: float = 0.25
    transfer_teacher_positives: bool = True
    transfer_online_positives: bool = True
    teacher_epochs: int = 4
    balance_teacher_actions: bool = True
    fit_state_feature_scale: bool = True
    online_target_weighting: bool = True

    def __post_init__(self) -> None:
        state_dim = _loaded_integer(
            self.state_dim,
            name="state_dim",
            minimum=1,
        )
        teacher_epochs = _loaded_integer(
            self.teacher_epochs,
            name="teacher_epochs",
            minimum=1,
        )
        numeric_fields = {
            "state_feature_scale": _loaded_number(
                self.state_feature_scale,
                name="state_feature_scale",
                minimum=np.finfo(np.float64).tiny,
            ),
            "learning_rate": _loaded_number(
                self.learning_rate,
                name="learning_rate",
                minimum=np.finfo(np.float64).tiny,
            ),
            "l2": _loaded_number(self.l2, name="l2", minimum=0.0),
            "score_bound": _loaded_number(
                self.score_bound,
                name="score_bound",
                minimum=np.finfo(np.float64).tiny,
                maximum=1.0,
            ),
            "runtime_weight": _loaded_number(
                self.runtime_weight,
                name="runtime_weight",
                minimum=0.0,
                maximum=4.0,
            ),
            "confidence_prior": _loaded_number(
                self.confidence_prior,
                name="confidence_prior",
                minimum=np.finfo(np.float64).tiny,
            ),
            "max_weight_norm": _loaded_number(
                self.max_weight_norm,
                name="max_weight_norm",
                minimum=np.finfo(np.float64).tiny,
            ),
            "positive_transfer_scale": _loaded_number(
                self.positive_transfer_scale,
                name="positive_transfer_scale",
                minimum=0.0,
                maximum=1.0,
            ),
        }
        for name in (
            "transfer_teacher_positives",
            "transfer_online_positives",
            "balance_teacher_actions",
            "fit_state_feature_scale",
            "online_target_weighting",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be a bool")
            object.__setattr__(self, name, bool(getattr(self, name)))
        object.__setattr__(self, "state_dim", state_dim)
        object.__setattr__(self, "teacher_epochs", teacher_epochs)
        for name, value in numeric_fields.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class StudentRepresentationTrajectory:
    """Current-state representations with normalized trajectory credit.

    ``terminal_index`` designates the representation used for runtime action
    scoring.  All representations with nonzero ``loop_weights`` receive the
    same teacher label or online outcome target during learning.
    """

    representations: tuple[Representation, ...]
    loop_weights: tuple[float, ...] = ()
    terminal_index: int = -1

    def __post_init__(self) -> None:
        representations = tuple(self.representations)
        if not representations:
            raise ValueError("student representation trajectory cannot be empty")
        if not all(isinstance(row, Representation) for row in representations):
            raise TypeError("trajectory elements must be Representation values")
        raw_weights = (
            np.ones(len(representations), dtype=np.float64)
            if not self.loop_weights
            else np.asarray(self.loop_weights, dtype=np.float64).reshape(-1)
        )
        if raw_weights.size != len(representations):
            raise ValueError("loop_weights must align with representations")
        if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0):
            raise ValueError("loop_weights must be finite and non-negative")
        total = float(raw_weights.sum())
        if total <= 0:
            raise ValueError("at least one loop weight must be positive")
        normalized = tuple(float(value / total) for value in raw_weights)
        terminal_index = int(self.terminal_index)
        if terminal_index < 0:
            terminal_index += len(representations)
        if not 0 <= terminal_index < len(representations):
            raise ValueError("terminal_index is outside the representation trajectory")
        object.__setattr__(self, "representations", representations)
        object.__setattr__(self, "loop_weights", normalized)
        object.__setattr__(self, "terminal_index", terminal_index)

    @property
    def terminal(self) -> Representation:
        return self.representations[self.terminal_index]

    @classmethod
    def current(cls, snapshot: WorldSnapshot) -> "StudentRepresentationTrajectory":
        """One-step trajectory used by the currently integrated runtime."""

        return cls((snapshot.representation,), (1.0,), terminal_index=0)


@dataclass(frozen=True, slots=True)
class StudentTeacherSample:
    """One represented offline label for balanced batch distillation."""

    snapshot: WorldSnapshot
    example: TeacherExample
    candidates: tuple[Action, ...]
    trajectory: StudentRepresentationTrajectory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, WorldSnapshot):
            raise TypeError("student sample snapshot must be a WorldSnapshot")
        if not isinstance(self.example, TeacherExample):
            raise TypeError("student sample example must be a TeacherExample")
        candidates = tuple(self.candidates)
        if not candidates or not all(isinstance(row, Action) for row in candidates):
            raise ValueError("student sample requires canonical action candidates")
        if self.trajectory is not None and not isinstance(
            self.trajectory,
            StudentRepresentationTrajectory,
        ):
            raise TypeError("student sample trajectory has the wrong type")
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True, slots=True)
class StudentPolicyScore:
    """Bounded candidate preference with auditable support and confidence."""

    value: float
    raw_value: float
    task_value: float
    transfer_value: float
    support: float
    task_support: float
    transfer_support: float
    confidence: float


@dataclass(frozen=True, slots=True)
class StudentPolicyDistribution:
    """Normalized preference over one explicitly supplied legal candidate set."""

    candidates: tuple[Action, ...]
    probabilities: tuple[float, ...]
    scores: tuple[StudentPolicyScore, ...]


@dataclass(slots=True)
class _LinearRow:
    weights: np.ndarray
    support: float = 0.0
    positive_support: float = 0.0
    negative_support: float = 0.0

    @classmethod
    def empty(cls, size: int) -> "_LinearRow":
        return cls(weights=np.zeros(int(size), dtype=np.float64))

    def state_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "support": float(self.support),
            "positive_support": float(self.positive_support),
            "negative_support": float(self.negative_support),
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        size: int,
        max_weight_norm: float,
    ) -> "_LinearRow":
        try:
            raw_weights = np.asarray(state.get("weights", ()), dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError("student row weights must be a numeric array") from exc
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            )
            for value in raw_weights.reshape(-1)
        ):
            raise ValueError("student row weights must contain only numbers")
        weights = np.asarray(raw_weights, dtype=np.float64).reshape(-1)
        if weights.size != int(size):
            raise ValueError(
                f"student row width {weights.size} does not match expected {size}"
            )
        if not np.isfinite(weights).all():
            raise ValueError("student row contains nonfinite weights")
        norm = float(np.linalg.norm(weights))
        if norm > float(max_weight_norm) + 1e-9:
            raise ValueError("student row weight norm exceeds configured maximum")
        support = _loaded_number(
            state.get("support", 0.0),
            name="student row support",
            minimum=0.0,
        )
        positive = _loaded_number(
            state.get("positive_support", 0.0),
            name="student row positive_support",
            minimum=0.0,
        )
        negative = _loaded_number(
            state.get("negative_support", 0.0),
            name="student row negative_support",
            minimum=0.0,
        )
        support_tolerance = 1e-9 * max(1.0, support)
        if positive + negative > support + support_tolerance:
            raise ValueError(
                "student positive/negative support cannot exceed total support"
            )
        return cls(
            weights=weights.copy(),
            support=support,
            positive_support=positive,
            negative_support=negative,
        )


class StateConditionedStudentPolicy:
    """Deterministic candidate-conditioned linear student head.

    Rows are keyed only by task and action *type* (the integer action index).
    The state and click coordinates remain continuous feature inputs, so there
    is no exact-state action table hidden in the implementation.
    """

    def __init__(self, config: StudentPolicyConfig | None = None) -> None:
        self.config = config or StudentPolicyConfig()
        self._feature_dim = (
            1
            + int(self.config.state_dim)
            + _CANDIDATE_DIM
            + int(self.config.state_dim) * _CANDIDATE_DIM
        )
        self._task_rows: dict[tuple[str, int], _LinearRow] = {}
        self._transfer_rows: dict[int, _LinearRow] = {}
        self._task_scales: dict[str, float] = {}
        self.teacher_updates = 0
        self.online_updates = 0

    @property
    def feature_dim(self) -> int:
        return int(self._feature_dim)

    @staticmethod
    def _dedupe_actions(actions: Sequence[Action]) -> tuple[Action, ...]:
        rows: dict[tuple[int, int, int], Action] = {}
        for action in actions:
            if not isinstance(action, Action):
                raise TypeError("student candidates must be Action values")
            rows.setdefault(action.key, action)
        return tuple(rows.values())

    @staticmethod
    def _trajectory(
        snapshot: WorldSnapshot,
        trajectory: StudentRepresentationTrajectory | None,
    ) -> StudentRepresentationTrajectory:
        return trajectory or StudentRepresentationTrajectory.current(snapshot)

    @staticmethod
    def _state_values(representation: Representation) -> np.ndarray:
        """Flattened, clipped latent input consumed by the feature map."""

        values = np.concatenate(
            (
                np.asarray(
                    representation.global_vector,
                    dtype=np.float64,
                ).reshape(-1),
                np.asarray(
                    representation.spatial,
                    dtype=np.float64,
                ).reshape(-1),
            )
        )
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(values, -8.0, 8.0)

    def _state_features(
        self,
        representation: Representation,
        *,
        scale: float | None = None,
    ) -> np.ndarray:
        """Deterministic nonlinear features of the represented state.

        A folded linear summary could not express short minority turns in a
        route: on the trusted ``tr87`` states it collapsed back to the 75.5%
        dominant-action classifier.  This fixed random-Fourier map consumes
        flattened spatial positions and supplies a compact nonlinear basis
        while keeping the only trained parameters in the auditable student
        rows.  The map is regenerated deterministically from input width; no
        state id, teacher record, or lookup table is stored.  It does not
        guarantee that every represented state is separable.

        The scale is an RBF bandwidth: the implied kernel between states at
        latent distance ``d`` is ``exp(-scale^2 d^2 / 2)``, so the scale must
        sit near the reciprocal of typical inter-state distances or the map
        degenerates.  On the trusted ``ls20`` states the median pairwise
        latent distance is ~0.39 (adjacent ~0.13); at the original scale of
        80 even adjacent states decorrelated (feature cosine -0.11), turning
        the head into a 4-row hash table that fit only 31/54 of its own
        training states and steered zero autonomous route steps.  The right
        bandwidth is game-geometry dependent (trusted ``tr87`` states sit two
        orders of magnitude closer together than ``ls20`` states), so
        distillation fits a per-task scale from measured label-separation
        distances (:meth:`_fit_task_scale`); ``state_feature_scale`` is the
        default for tasks without a fit.
        """

        values = self._state_values(representation)
        dim = int(self.config.state_dim)
        effective_scale = (
            float(self.config.state_feature_scale) if scale is None else float(scale)
        )
        basis_key = (dim, int(values.size))
        basis = _RFF_BASIS_CACHE.get(basis_key)
        if basis is None:
            # PCG64 is instantiated locally from a width-derived seed, making
            # this a pure function even during action selection; the basis is
            # memoized because regenerating it dominated scoring cost at
            # large widths.
            generator = np.random.default_rng(9_173 + int(values.size))
            projection = generator.normal(
                0.0,
                1.0,
                size=(dim, int(values.size)),
            )
            phase = generator.uniform(0.0, 2.0 * np.pi, size=dim)
            projection.setflags(write=False)
            phase.setflags(write=False)
            basis = (projection, phase)
            _RFF_BASIS_CACHE[basis_key] = basis
        projection, phase = basis
        result = np.sqrt(2.0 / dim) * np.cos(
            effective_scale * (projection @ values) + phase
        )
        return result.astype(np.float64, copy=False)

    def _task_scale(self, task_id: str) -> float | None:
        """Fitted per-task bandwidth, or ``None`` for the config default."""

        if not self.config.fit_state_feature_scale:
            return None
        return self._task_scales.get(str(task_id))

    def _fit_task_scale(
        self,
        task_id: str,
        values_rows: Sequence[np.ndarray],
        labels: Sequence[int],
    ) -> float | None:
        """Fit one task's RBF bandwidth from demonstration geometry.

        The states a policy must tell apart are those that demand different
        actions, so the bandwidth is anchored to the low quantile of latent
        distances between differently-labeled states: the kernel argument
        ``scale * d`` is set to ``_SCALE_FIT_KERNEL_ARG`` at that separation
        distance, making such pairs distinguishable while closer
        same-decision revisits stay correlated.  Purely geometric — no state
        ids or actions are stored — and deterministic for a given batch.
        """

        task_id = str(task_id)
        if task_id in self._task_scales:
            return self._task_scales[task_id]
        # A scale is part of the feature map under which existing rows were
        # trained.  Refitting it in a later import silently reinterprets every
        # stored weight, including action rows untouched by the new batch.
        if any(row_task == task_id for row_task, _action in self._task_rows):
            return None
        label_array = np.asarray(labels, dtype=np.int64)
        if len(values_rows) < 2 or np.unique(label_array).size < 2:
            return None
        stacked = np.stack(values_rows)
        # Gram-matrix distances: the broadcasted difference tensor is
        # O(n^2 * d) memory and fails outright at wide representations
        # (1,556 x 4,128 values would need ~75 GiB); this identity needs
        # only the n x n matrix.
        squared_norms = np.einsum("ij,ij->i", stacked, stacked)
        squared = (
            squared_norms[:, None]
            + squared_norms[None, :]
            - 2.0 * (stacked @ stacked.T)
        )
        distances = np.sqrt(np.maximum(squared, 0.0))
        upper = np.triu_indices(len(stacked), 1)
        different = label_array[:, None] != label_array[None, :]
        separations = distances[upper][different[upper]]
        separations = separations[separations > 1e-9]
        if not separations.size:
            return None
        anchor = float(np.quantile(separations, _SCALE_FIT_QUANTILE))
        if anchor <= 0:
            return None
        low, high = _SCALE_FIT_BOUNDS
        scale = float(np.clip(_SCALE_FIT_KERNEL_ARG / anchor, low, high))
        self._task_scales[task_id] = scale
        return scale

    @staticmethod
    def _local_spatial(
        snapshot: WorldSnapshot,
        representation: Representation,
        action: Action,
    ) -> tuple[float, float]:
        if not action.has_position:
            return 0.0, 0.0
        spatial = np.asarray(representation.spatial, dtype=np.float64)
        frame = np.asarray(snapshot.observation.frame)
        if spatial.ndim != 2 or not spatial.size or frame.ndim < 2:
            return 0.0, 0.0
        frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
        row = int(
            round(
                np.clip(action.y, 0, max(frame_height - 1, 0))
                * max(spatial.shape[0] - 1, 0)
                / max(frame_height - 1, 1)
            )
        )
        column = int(
            round(
                np.clip(action.x, 0, max(frame_width - 1, 0))
                * max(spatial.shape[1] - 1, 0)
                / max(frame_width - 1, 1)
            )
        )
        local = _finite(spatial[row, column])
        row_lo, row_hi = max(0, row - 1), min(spatial.shape[0], row + 2)
        col_lo, col_hi = max(0, column - 1), min(spatial.shape[1], column + 2)
        neighborhood = _finite(np.mean(spatial[row_lo:row_hi, col_lo:col_hi]))
        return float(np.tanh(local)), float(np.tanh(neighborhood))

    def _candidate_features(
        self,
        snapshot: WorldSnapshot,
        representation: Representation,
        action: Action,
    ) -> np.ndarray:
        if not action.has_position:
            return np.zeros(_CANDIDATE_DIM, dtype=np.float64)
        frame = np.asarray(snapshot.observation.frame)
        height = int(frame.shape[0]) if frame.ndim >= 1 else 1
        width = int(frame.shape[1]) if frame.ndim >= 2 else 1
        x = float(np.clip(2.0 * action.x / max(width - 1, 1) - 1.0, -1.0, 1.0))
        y = float(np.clip(2.0 * action.y / max(height - 1, 1) - 1.0, -1.0, 1.0))
        local, neighborhood = self._local_spatial(snapshot, representation, action)
        return np.asarray(
            [1.0, x, y, x * x, y * y, x * y, local, neighborhood],
            dtype=np.float64,
        )

    def features(
        self,
        snapshot: WorldSnapshot,
        action: Action,
        *,
        representation: Representation | None = None,
    ) -> np.ndarray:
        """Return the deterministic bounded joint feature vector."""

        if not isinstance(action, Action):
            raise TypeError("student scoring requires an Action")
        current = representation or snapshot.representation
        if not isinstance(current, Representation):
            raise TypeError("student features require a Representation")
        state = self._state_features(
            current,
            scale=self._task_scale(snapshot.observation.task_id),
        )
        candidate = self._candidate_features(snapshot, current, action)
        features = np.concatenate(
            (
                np.ones(1, dtype=np.float64),
                state,
                candidate,
                np.outer(state, candidate).reshape(-1),
            )
        )
        norm = float(np.linalg.norm(features))
        if norm > 1.0:
            features /= norm
        features.setflags(write=False)
        return features

    def _row(
        self,
        *,
        task_id: str,
        action_index: int,
        create: bool,
    ) -> _LinearRow | None:
        key = (str(task_id), int(action_index))
        row = self._task_rows.get(key)
        if row is None and create:
            row = _LinearRow.empty(self.feature_dim)
            self._task_rows[key] = row
        return row

    def _transfer_row(self, action_index: int, *, create: bool) -> _LinearRow | None:
        key = int(action_index)
        row = self._transfer_rows.get(key)
        if row is None and create:
            row = _LinearRow.empty(self.feature_dim)
            self._transfer_rows[key] = row
        return row

    def _update_row(
        self,
        row: _LinearRow,
        features: np.ndarray,
        *,
        target: float,
        weight: float,
        support_weight: float | None = None,
    ) -> None:
        sample_weight = _finite(weight)
        if sample_weight <= 0:
            return
        bounded_target = float(np.clip(_finite(target), -1.0, 1.0))
        prediction = float(np.tanh(float(np.dot(row.weights, features))))
        error = bounded_target - prediction
        rate = float(self.config.learning_rate) * sample_weight
        decay = max(0.0, 1.0 - rate * float(self.config.l2))
        row.weights *= decay
        row.weights += rate * error * features
        norm = float(np.linalg.norm(row.weights))
        limit = float(self.config.max_weight_norm)
        if norm > limit:
            row.weights *= limit / norm
        empirical_support = (
            sample_weight
            if support_weight is None
            else max(0.0, _finite(support_weight))
        )
        row.support += empirical_support
        if bounded_target > 0:
            row.positive_support += empirical_support
        elif bounded_target < 0:
            row.negative_support += empirical_support

    def observe_teacher(
        self,
        snapshot: WorldSnapshot,
        example: TeacherExample,
        *,
        candidates: Sequence[Action] = (),
        trajectory: StudentRepresentationTrajectory | None = None,
    ) -> int:
        """Learn one weighted demonstration across its representation trajectory.

        Legal alternatives receive task-local contrastive labels.  Only the
        demonstrated positive action may update the explicit transfer row.
        Returns the number of parameter rows updated.
        """

        legal, credit = self._validate_teacher_example(
            snapshot,
            example,
            candidates=candidates,
            trajectory=trajectory,
        )
        weight = float(example.weight)
        if weight <= 0.0:
            return 0
        updates = self._observe_teacher_weighted(
            snapshot,
            example,
            legal=legal,
            credit=credit,
            optimization_weight=weight,
            support_weight=weight,
        )
        self.teacher_updates += 1
        return updates

    def _validate_teacher_example(
        self,
        snapshot: WorldSnapshot,
        example: TeacherExample,
        *,
        candidates: Sequence[Action],
        trajectory: StudentRepresentationTrajectory | None,
    ) -> tuple[tuple[Action, ...], StudentRepresentationTrajectory]:
        if not isinstance(example, TeacherExample):
            raise TypeError("student teacher updates require TeacherExample values")
        if snapshot.observation.task_id != example.task_id:
            raise ValueError("teacher example task does not match its representation")
        if snapshot.observation.stage != example.stage:
            raise ValueError("teacher example stage does not match its representation")
        legal = self._dedupe_actions(candidates) if candidates else (example.action,)
        if example.action.key not in {action.key for action in legal}:
            raise ValueError("teacher action is absent from the legal candidate set")
        return legal, self._trajectory(snapshot, trajectory)

    def _observe_teacher_weighted(
        self,
        snapshot: WorldSnapshot,
        example: TeacherExample,
        *,
        legal: Sequence[Action],
        credit: StudentRepresentationTrajectory,
        optimization_weight: float,
        support_weight: float,
    ) -> int:
        alternatives = tuple(
            action for action in legal if action.key != example.action.key
        )
        updates = 0
        for representation, loop_weight in zip(
            credit.representations,
            credit.loop_weights,
            strict=True,
        ):
            step_weight = float(optimization_weight) * loop_weight
            step_support = float(support_weight) * loop_weight
            if step_weight <= 0:
                continue
            chosen_features = self.features(
                snapshot,
                example.action,
                representation=representation,
            )
            chosen_row = self._row(
                task_id=example.task_id,
                action_index=example.action.index,
                create=True,
            )
            assert chosen_row is not None
            self._update_row(
                chosen_row,
                chosen_features,
                target=1.0,
                weight=step_weight,
                support_weight=step_support,
            )
            updates += 1
            alternative_weight = step_weight / max(len(alternatives), 1)
            alternative_support = step_support / max(len(alternatives), 1)
            for action in alternatives:
                row = self._row(
                    task_id=example.task_id,
                    action_index=action.index,
                    create=True,
                )
                assert row is not None
                self._update_row(
                    row,
                    self.features(
                        snapshot,
                        action,
                        representation=representation,
                    ),
                    target=-1.0,
                    weight=alternative_weight,
                    support_weight=alternative_support,
                )
                updates += 1

            if self.config.transfer_teacher_positives:
                transfer = self._transfer_row(example.action.index, create=True)
                assert transfer is not None
                self._update_row(
                    transfer,
                    chosen_features,
                    target=1.0,
                    weight=step_weight,
                    support_weight=step_support,
                )
                updates += 1
        return updates

    def observe_teacher_batch(
        self,
        samples: Sequence[StudentTeacherSample],
    ) -> int:
        """Fit a balanced represented-state policy without frequency collapse.

        Action classes receive equal total optimization weight while empirical
        support is counted exactly once.  Repeated epochs therefore improve
        the trained function without pretending that one demonstration became
        several independent observations.
        """

        prepared: list[
            tuple[
                StudentTeacherSample,
                tuple[Action, ...],
                StudentRepresentationTrajectory,
            ]
        ] = []
        action_weight: dict[int, float] = {}
        for sample in samples:
            if not isinstance(sample, StudentTeacherSample):
                raise TypeError("student teacher batch requires StudentTeacherSample values")
            legal, credit = self._validate_teacher_example(
                sample.snapshot,
                sample.example,
                candidates=sample.candidates,
                trajectory=sample.trajectory,
            )
            weight = float(sample.example.weight)
            if weight <= 0.0:
                continue
            prepared.append((sample, legal, credit))
            action_index = int(sample.example.action.index)
            action_weight[action_index] = (
                action_weight.get(action_index, 0.0) + weight
            )
        if not prepared:
            return 0
        if self.config.fit_state_feature_scale:
            by_task: dict[str, tuple[list[np.ndarray], list[int]]] = {}
            for sample, _legal, credit in prepared:
                rows, task_labels = by_task.setdefault(
                    str(sample.example.task_id),
                    ([], []),
                )
                rows.append(self._state_values(credit.terminal))
                task_labels.append(int(sample.example.action.index))
            for task_id, (rows, task_labels) in by_task.items():
                self._fit_task_scale(task_id, rows, task_labels)
        mean_action_weight = float(
            sum(action_weight.values()) / max(len(action_weight), 1)
        )
        updates = 0
        epochs = int(self.config.teacher_epochs)
        for epoch in range(epochs):
            for sample, legal, credit in prepared:
                example = sample.example
                base_weight = float(example.weight)
                balance = (
                    mean_action_weight
                    / max(action_weight[int(example.action.index)], 1e-12)
                    if self.config.balance_teacher_actions
                    else 1.0
                )
                updates += self._observe_teacher_weighted(
                    sample.snapshot,
                    example,
                    legal=legal,
                    credit=credit,
                    optimization_weight=base_weight * balance,
                    support_weight=(base_weight if epoch == 0 else 0.0),
                )
        self.teacher_updates += len(prepared)
        return updates

    @staticmethod
    def _outcome_target(outcome: Outcome) -> float:
        hazard = max(float(outcome.hazard), 1.0 if outcome.failed else 0.0)
        utility = (
            float(outcome.reward)
            + float(outcome.progress_delta)
            + float(outcome.completed)
            - hazard
            - float(outcome.failed)
        )
        return float(np.tanh(_finite(utility)))

    def observe_outcome(
        self,
        snapshot: WorldSnapshot,
        action: Action,
        outcome: Outcome,
        *,
        weight: float = 1.0,
        trajectory: StudentRepresentationTrajectory | None = None,
    ) -> int:
        """Update every representation from one committed action outcome.

        Negative and neutral outcomes update only the task row.  Positive
        outcomes may additionally update the global positive-transfer row.

        With ``online_target_weighting`` (default) each update is scaled by
        the outcome-target magnitude.  A neutral step carries no preference
        information, yet regressing the taken action toward its target of
        zero erodes whatever distilled teacher margin that state region
        holds; measured live, a distilled ls20 head that completed a level in
        episode 1 degenerated into a single-action loop by episode 2 purely
        from accumulated neutral updates.  Deaths and completions keep full
        weight; partial hazards scale proportionally.
        """

        sample_weight = _finite(weight)
        if sample_weight <= 0:
            return 0
        target = self._outcome_target(outcome)
        if self.config.online_target_weighting:
            sample_weight *= abs(target)
            if sample_weight <= 0:
                self.online_updates += 1
                return 0
        credit = self._trajectory(snapshot, trajectory)
        updates = 0
        for representation, loop_weight in zip(
            credit.representations,
            credit.loop_weights,
            strict=True,
        ):
            step_weight = sample_weight * loop_weight
            if step_weight <= 0:
                continue
            features = self.features(
                snapshot,
                action,
                representation=representation,
            )
            task_row = self._row(
                task_id=snapshot.observation.task_id,
                action_index=action.index,
                create=True,
            )
            assert task_row is not None
            self._update_row(
                task_row,
                features,
                target=target,
                weight=step_weight,
            )
            updates += 1
            if target > 0 and self.config.transfer_online_positives:
                transfer = self._transfer_row(action.index, create=True)
                assert transfer is not None
                self._update_row(
                    transfer,
                    features,
                    target=target,
                    weight=step_weight,
                )
                updates += 1
        self.online_updates += 1
        return updates

    def observe_transition(
        self,
        transition: Transition,
        *,
        weight: float = 1.0,
        trajectory: StudentRepresentationTrajectory | None = None,
    ) -> int:
        """Convenience wrapper for the canonical committed transition."""

        return self.observe_outcome(
            transition.before,
            transition.action,
            transition.outcome,
            weight=weight,
            trajectory=trajectory,
        )

    def score(
        self,
        snapshot: WorldSnapshot,
        action: Action,
        *,
        trajectory: StudentRepresentationTrajectory | None = None,
    ) -> StudentPolicyScore:
        """Score only the designated terminal/current representation."""

        credit = self._trajectory(snapshot, trajectory)
        features = self.features(
            snapshot,
            action,
            representation=credit.terminal,
        )
        task_row = self._row(
            task_id=snapshot.observation.task_id,
            action_index=action.index,
            create=False,
        )
        transfer_row = self._transfer_row(action.index, create=False)
        task_value = (
            float(np.dot(task_row.weights, features)) if task_row is not None else 0.0
        )
        transfer_logit = (
            float(np.dot(transfer_row.weights, features))
            if transfer_row is not None
            else 0.0
        )
        # Cross-task evidence is deliberately one-way: a row may be fitted only
        # by positive labels, and a feature mismatch cannot turn it into a
        # cross-task penalty.  Negative pressure always remains task-local.
        transfer_value = max(0.0, transfer_logit)
        transfer_scale = float(self.config.positive_transfer_scale)
        raw_value = task_value + transfer_scale * transfer_value
        task_support = float(task_row.support) if task_row is not None else 0.0
        transfer_support = (
            float(transfer_row.positive_support) if transfer_row is not None else 0.0
        )
        support = task_support + transfer_scale * transfer_support
        confidence = support / (support + float(self.config.confidence_prior))
        value = (
            float(self.config.score_bound)
            * float(np.tanh(raw_value))
            * float(np.clip(confidence, 0.0, 1.0))
        )
        return StudentPolicyScore(
            value=float(
                np.clip(value, -self.config.score_bound, self.config.score_bound)
            ),
            raw_value=_finite(raw_value),
            task_value=_finite(task_value),
            transfer_value=_finite(transfer_value),
            support=_finite(support),
            task_support=_finite(task_support),
            transfer_support=_finite(transfer_support),
            confidence=float(np.clip(_finite(confidence), 0.0, 1.0)),
        )

    def distribution(
        self,
        snapshot: WorldSnapshot,
        candidates: Sequence[Action],
        *,
        trajectory: StudentRepresentationTrajectory | None = None,
    ) -> StudentPolicyDistribution:
        """Return a deterministic normalized distribution over legal candidates.

        The bounded, confidence-adjusted values are used as logits.  Consequently
        an unseen head is uniform and probability order exactly follows the score
        order used by runtime arbitration.
        """

        legal = self._dedupe_actions(candidates)
        if not legal:
            raise ValueError("student distribution requires legal candidates")
        scores = tuple(
            self.score(snapshot, action, trajectory=trajectory) for action in legal
        )
        logits = np.asarray([row.value for row in scores], dtype=np.float64)
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(np.sum(probabilities))
        return StudentPolicyDistribution(
            candidates=legal,
            probabilities=tuple(float(value) for value in probabilities),
            scores=scores,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": STUDENT_STATE_VERSION,
            "config": asdict(self.config),
            "feature_dim": self.feature_dim,
            "task_rows": {
                task_id: {
                    str(action_index): row.state_dict()
                    for (row_task, action_index), row in sorted(self._task_rows.items())
                    if row_task == task_id
                }
                for task_id in sorted({key[0] for key in self._task_rows})
            },
            "transfer_rows": {
                str(action_index): row.state_dict()
                for action_index, row in sorted(self._transfer_rows.items())
            },
            "task_scales": {
                task_id: float(scale)
                for task_id, scale in sorted(self._task_scales.items())
            },
            "teacher_updates": int(self.teacher_updates),
            "online_updates": int(self.online_updates),
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        config: StudentPolicyConfig | None = None,
    ) -> "StateConditionedStudentPolicy":
        if not isinstance(state, Mapping):
            raise ValueError("student checkpoint must be a mapping")
        version = _loaded_integer(
            state.get("version", -1),
            name="student checkpoint version",
        )
        if version != STUDENT_STATE_VERSION:
            raise ValueError(
                f"unsupported student state version {version}; "
                f"expected {STUDENT_STATE_VERSION}"
            )
        raw_config = state.get("config", {})
        if not isinstance(raw_config, Mapping):
            raise ValueError("student checkpoint config must be a mapping")
        saved_config = StudentPolicyConfig(**dict(raw_config))
        if config is not None and config != saved_config:
            raise ValueError("student checkpoint configuration does not match")
        instance = cls(config or saved_config)
        feature_dim = _loaded_integer(
            state.get("feature_dim", -1),
            name="student checkpoint feature_dim",
        )
        if feature_dim != instance.feature_dim:
            raise ValueError("student checkpoint feature dimension does not match")
        task_rows = state.get("task_rows", {})
        if not isinstance(task_rows, Mapping):
            raise ValueError("student task_rows must be a mapping")
        for task_id, action_rows in task_rows.items():
            if not isinstance(task_id, str):
                raise ValueError("student task row keys must be strings")
            if not isinstance(action_rows, Mapping):
                raise ValueError("student per-task rows must be mappings")
            for action_index, row_state in action_rows.items():
                if not isinstance(row_state, Mapping):
                    raise ValueError("student row state must be a mapping")
                parsed_action = _loaded_action_index(
                    action_index,
                    name="student task action key",
                )
                instance._task_rows[(task_id, parsed_action)] = (
                    _LinearRow.from_state(
                        row_state,
                        size=instance.feature_dim,
                        max_weight_norm=instance.config.max_weight_norm,
                    )
                )
        transfer_rows = state.get("transfer_rows", {})
        if not isinstance(transfer_rows, Mapping):
            raise ValueError("student transfer_rows must be a mapping")
        for action_index, row_state in transfer_rows.items():
            if not isinstance(row_state, Mapping):
                raise ValueError("student row state must be a mapping")
            parsed_action = _loaded_action_index(
                action_index,
                name="student transfer action key",
            )
            row = _LinearRow.from_state(
                row_state,
                size=instance.feature_dim,
                max_weight_norm=instance.config.max_weight_norm,
            )
            tolerance = 1e-9 * max(1.0, row.support)
            if (
                row.negative_support > tolerance
                or abs(row.positive_support - row.support) > tolerance
            ):
                raise ValueError(
                    "student transfer rows must contain positive support only"
                )
            instance._transfer_rows[parsed_action] = row
        task_scales = state.get("task_scales", {})
        if not isinstance(task_scales, Mapping):
            raise ValueError("student task_scales must be a mapping")
        for task_id, scale in task_scales.items():
            if not isinstance(task_id, str):
                raise ValueError("student task scale keys must be strings")
            instance._task_scales[task_id] = _loaded_number(
                scale,
                name="student task scale",
                minimum=np.finfo(np.float64).tiny,
            )
        instance.teacher_updates = _loaded_integer(
            state.get("teacher_updates", 0),
            name="student teacher_updates",
        )
        instance.online_updates = _loaded_integer(
            state.get("online_updates", 0),
            name="student online_updates",
        )
        return instance

    def summary(self) -> dict[str, Any]:
        return {
            "task_rows": len(self._task_rows),
            "transfer_rows": len(self._transfer_rows),
            "task_scales": {
                task_id: float(scale)
                for task_id, scale in sorted(self._task_scales.items())
            },
            "teacher_updates": int(self.teacher_updates),
            "online_updates": int(self.online_updates),
            "task_support": float(sum(row.support for row in self._task_rows.values())),
            "transfer_positive_support": float(
                sum(row.positive_support for row in self._transfer_rows.values())
            ),
        }


__all__ = [
    "STUDENT_STATE_VERSION",
    "StateConditionedStudentPolicy",
    "StudentPolicyDistribution",
    "StudentPolicyConfig",
    "StudentPolicyScore",
    "StudentRepresentationTrajectory",
    "StudentTeacherSample",
]
