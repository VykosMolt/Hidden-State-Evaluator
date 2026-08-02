"""Explicit online/replay learning with no action-selection read path."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import LearningConfig, Transition, WorldSnapshot
from .models import ActionPrior, AffordanceModel, DynamicsEnsemble


@dataclass(frozen=True, slots=True)
class ReplayItem:
    before: WorldSnapshot
    after: WorldSnapshot
    transition: Transition
    target_object_signature: str = ""
    source: str = "online"
    teacher: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.before, WorldSnapshot):
            raise TypeError("replay before must be a WorldSnapshot")
        if not isinstance(self.after, WorldSnapshot):
            raise TypeError("replay after must be a WorldSnapshot")
        if not isinstance(self.transition, Transition):
            raise TypeError("replay transition must be a Transition")
        if self.transition.before is not self.before:
            raise ValueError("replay transition must reference its before snapshot")
        if self.transition.after_state_id != self.after.state_id:
            raise ValueError("replay transition must match its after snapshot")
        if self.transition.after_observation.state_id != self.after.observation.state_id:
            raise ValueError("replay after observation does not match its transition")
        if not isinstance(self.teacher, (bool, np.bool_)):
            raise TypeError("replay teacher must be a bool")
        object.__setattr__(
            self,
            "target_object_signature",
            str(self.target_object_signature),
        )
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "teacher", bool(self.teacher))


class ReplayBuffer:
    """Bounded transition replay separated by provenance.

    This object has deliberately no candidate-scoring or lookup method.  Runtime
    action selection cannot query demonstrations through it.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        if isinstance(capacity, (bool, np.bool_)) or not isinstance(
            capacity,
            (int, np.integer),
        ):
            raise ValueError("replay capacity must be an integer")
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self._items: deque[ReplayItem] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, memo: dict[int, object]) -> "ReplayBuffer":
        """Copy the mutable queue while sharing immutable replay records.

        ``ReplayItem`` and the snapshots/transitions it references are frozen;
        transactional staging needs an independent deque, not duplicate frame
        arrays for every historical transition.
        """

        duplicate = type(self)(self.capacity)
        memo[id(self)] = duplicate
        duplicate._items = deque(self._items, maxlen=self.capacity)
        return duplicate

    @property
    def items(self) -> tuple[ReplayItem, ...]:
        return tuple(self._items)

    def push(self, item: ReplayItem) -> None:
        if not isinstance(item, ReplayItem):
            raise TypeError("replay buffer accepts only ReplayItem values")
        self._items.append(item)

    def sample(
        self,
        batch_size: int,
        *,
        teacher_fraction: float,
        rng: np.random.Generator,
    ) -> tuple[ReplayItem, ...]:
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
            batch_size,
            (int, np.integer),
        ):
            raise ValueError("replay batch_size must be an integer")
        if int(batch_size) < 0:
            raise ValueError("replay batch_size must be non-negative")
        if isinstance(teacher_fraction, (bool, np.bool_)) or not isinstance(
            teacher_fraction,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError("teacher_fraction must be a number")
        teacher_fraction = float(teacher_fraction)
        if not np.isfinite(teacher_fraction) or not 0.0 <= teacher_fraction <= 1.0:
            raise ValueError("teacher_fraction must be finite and in [0, 1]")
        size = min(int(batch_size), len(self._items))
        if size <= 0:
            return ()
        teacher = [item for item in self._items if item.teacher]
        online = [item for item in self._items if not item.teacher]
        desired_teacher = min(
            len(teacher),
            int(round(size * teacher_fraction)),
        )
        desired_online = min(len(online), size - desired_teacher)
        remainder = size - desired_teacher - desired_online
        if remainder:
            extra_teacher = min(remainder, len(teacher) - desired_teacher)
            desired_teacher += extra_teacher
            remainder -= extra_teacher
        if remainder:
            desired_online += min(remainder, len(online) - desired_online)

        selected: list[ReplayItem] = []
        if desired_teacher:
            indexes = rng.choice(
                len(teacher),
                size=desired_teacher,
                replace=False,
            )
            selected.extend(teacher[int(index)] for index in np.atleast_1d(indexes))
        if desired_online:
            indexes = rng.choice(
                len(online),
                size=desired_online,
                replace=False,
            )
            selected.extend(online[int(index)] for index in np.atleast_1d(indexes))
        rng.shuffle(selected)
        return tuple(selected)

    def expert_fraction(self) -> float:
        if not self._items:
            return 0.0
        return float(np.mean([item.teacher for item in self._items]))

    def change_rate(self) -> float:
        if not self._items:
            return 0.0
        return float(
            np.mean([item.transition.frame_changed for item in self._items])
        )

    def click_change_rate(self) -> float:
        clicks = [
            item
            for item in self._items
            if item.transition.action.has_position
        ]
        if not clicks:
            return 0.0
        return float(np.mean([item.transition.frame_changed for item in clicks]))


class CompactLearner:
    """Owns the only trainable-update calls for the default compact models."""

    def __init__(
        self,
        *,
        dynamics: DynamicsEnsemble,
        prior: ActionPrior,
        affordances: AffordanceModel,
        config: LearningConfig | None = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(dynamics, DynamicsEnsemble):
            raise TypeError("dynamics must be a DynamicsEnsemble")
        if not isinstance(prior, ActionPrior):
            raise TypeError("prior must be an ActionPrior")
        if not isinstance(affordances, AffordanceModel):
            raise TypeError("affordances must be an AffordanceModel")
        if config is not None and not isinstance(config, LearningConfig):
            raise TypeError("config must be a LearningConfig")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed,
            (int, np.integer),
        ):
            raise ValueError("learner seed must be an integer")
        self.dynamics = dynamics
        self.prior = prior
        self.affordances = affordances
        self.config = config or LearningConfig()
        self.replay = ReplayBuffer(self.config.replay_capacity)
        self._rng = np.random.default_rng(int(seed))
        self.real_updates = 0
        self.replay_updates = 0
        self.last_replay_loss = 0.0

    def observe(
        self,
        item: ReplayItem,
        *,
        learn_immediately: bool = True,
    ) -> float:
        if not isinstance(learn_immediately, (bool, np.bool_)):
            raise TypeError("learn_immediately must be a bool")
        self.replay.push(item)
        if not learn_immediately:
            return 0.0
        loss = self._update(item, empirical=True)
        self.real_updates += 1
        return loss

    def _update(self, item: ReplayItem, *, empirical: bool) -> float:
        loss = self.dynamics.update(
            item.before,
            item.after,
            item.transition,
            empirical=empirical,
        )
        # Priors, affordances, and ensemble support are empirical counts.
        # Replay may optimize trainable dynamics weights, but it must not turn
        # one real observation into many independent pieces of evidence.
        self.prior.observe(
            item.transition,
            empirical=empirical,
            transferable=item.teacher,
        )
        self.affordances.observe(
            object_signature=item.target_object_signature,
            transition=item.transition,
            empirical=empirical,
            transferable_positive=item.teacher,
        )
        return float(loss)

    def replay_step(self) -> float:
        losses: list[float] = []
        for _ in range(max(0, int(self.config.replay_updates))):
            batch = self.replay.sample(
                self.config.replay_batch_size,
                teacher_fraction=self.config.teacher_fraction,
                rng=self._rng,
            )
            for item in batch:
                losses.append(self._update(item, empirical=False))
                self.replay_updates += 1
        self.last_replay_loss = float(np.mean(losses)) if losses else 0.0
        return self.last_replay_loss

    def maybe_replay(self, transition_count: int) -> float:
        cadence = int(self.config.replay_every)
        if isinstance(transition_count, (bool, np.bool_)) or not isinstance(
            transition_count,
            (int, np.integer),
        ):
            raise ValueError("transition_count must be an integer")
        transition_count = int(transition_count)
        if cadence <= 0 or transition_count <= 0:
            return 0.0
        if transition_count % cadence != 0:
            return 0.0
        return self.replay_step()


__all__ = [
    "CompactLearner",
    "ReplayBuffer",
    "ReplayItem",
]
