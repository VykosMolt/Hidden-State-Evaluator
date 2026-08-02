"""Transactional environment harness for compact Hunter-Seeker.

The generic :func:`run_episode` loop is intentionally small:

``act -> environment.step -> adapt outcome -> observe -> callbacks``.

ARC-specific imports are lazy, so the adapter and runner tests do not require
the ARC SDK.  No legacy Hunter-Seeker agent or mixin is imported here.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .adapters import (
    AdapterError,
    ArcActionAdapter,
    ArcObservationAdapter,
    ArcOutcomeAdapter,
)
from .contracts import (
    ActionAdapter,
    BoundaryKind,
    Decision,
    Observation,
    ObservationAdapter,
    Outcome,
    OutcomeAdapter,
    Transition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARC_ENVIRONMENTS_DIR = PROJECT_ROOT / "data" / "arc_agi3" / "environment_files"
ARC_RECORDINGS_DIR = PROJECT_ROOT / "artifacts" / "recordings" / "arc_agi3"


def _result_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise AdapterError(f"{name} must be a bool, got {value!r}")
    return bool(value)


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


@runtime_checkable
class TransactionalAgent(Protocol):
    """Minimal agent surface consumed by the harness."""

    def begin_run(
        self,
        task_id: str,
        observation: Observation | None = None,
    ) -> None:
        ...

    def act(self, observation: Observation) -> Decision:
        ...

    def observe(
        self,
        decision: Decision,
        next_observation: Observation,
        outcome: Outcome,
    ) -> Transition:
        ...

    def end_run(self, outcome: Outcome | None = None) -> None:
        ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Immutable summary of one environment run."""

    task_id: str
    initial_observation: Observation
    final_observation: Observation
    final_outcome: Outcome
    transitions: tuple[Transition, ...]

    @property
    def steps(self) -> int:
        return len(self.transitions)

    @property
    def completed(self) -> bool:
        return self.final_outcome.completed

    @property
    def failed(self) -> bool:
        return self.final_outcome.failed


def make_arcade() -> Any:
    """Construct an ARC Arcade backed by repository-local data paths."""

    import arc_agi

    ARC_ENVIRONMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ARC_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return arc_agi.Arcade(
        environments_dir=str(ARC_ENVIRONMENTS_DIR),
        recordings_dir=str(ARC_RECORDINGS_DIR),
    )


def _normalize_environment_result(
    result: Any,
) -> tuple[Any, dict[str, Any]]:
    """Accept raw, ARC, Gymnasium, and classic Gym step return shapes."""

    if not isinstance(result, tuple):
        return result, {}
    if len(result) == 2 and isinstance(result[1], Mapping):
        raw, info = result
        return raw, dict(info)
    if len(result) == 5 and isinstance(result[4], Mapping):
        raw, reward, terminated, truncated, info = result
        merged = dict(info)
        # Gymnasium's tuple fields are authoritative.  A stale/conflicting
        # duplicate in info must not suppress termination or change reward.
        merged["reward"] = reward
        merged["terminated"] = _result_bool(
            terminated,
            name="Gymnasium terminated",
        )
        merged["truncated"] = _result_bool(
            truncated,
            name="Gymnasium truncated",
        )
        return raw, merged
    if len(result) == 4 and isinstance(result[3], Mapping):
        raw, reward, done, info = result
        merged = dict(info)
        raw_truncated = merged.get(
            "TimeLimit.truncated",
            merged.get("truncated", False),
        )
        truncated = _result_bool(
            raw_truncated,
            name="Gym truncated",
        )
        done_flag = _result_bool(done, name="Gym done")
        merged["reward"] = reward
        merged["truncated"] = truncated
        merged["terminated"] = done_flag and not truncated
        return raw, merged
    return result, {}


def _call_if_present(agent: TransactionalAgent, name: str, *args: Any) -> None:
    callback = getattr(agent, name, None)
    if callable(callback):
        callback(*args)


def _dispatch_boundary_callbacks(
    agent: TransactionalAgent,
    transition: Transition,
) -> None:
    """Run optional compatibility callbacks only after commit."""

    outcome = transition.outcome
    if outcome.completed:
        completed_stage = max(
            int(transition.before.observation.stage),
            int(transition.after_observation.stage) - 1,
        )
        _call_if_present(agent, "on_level_complete", completed_stage)
    if outcome.boundary == BoundaryKind.DEATH:
        _call_if_present(agent, "on_game_over")
    if outcome.boundary != BoundaryKind.NONE:
        _call_if_present(agent, "on_boundary", transition)


def _environment_error_outcome(
    exc: Exception | None = None,
    *,
    returned_none: bool = False,
) -> Outcome:
    metadata: dict[str, Any] = {"returned_none": bool(returned_none)}
    if exc is not None:
        metadata.update(
            {
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
        )
    return Outcome(
        reward=-100.0,
        terminated=True,
        boundary=BoundaryKind.ENVIRONMENT_ERROR,
        metadata=metadata,
    )


def _time_limit_outcome(outcome: Outcome, *, max_steps: int) -> Outcome:
    """Attach the runner's local step limit to the causal final action.

    Reward, progress, hazard, and adapter metadata still belong to the
    environment step that consumed the remaining budget.  A non-terminal
    environment boundary is retained as provenance because the committed
    transition can expose only one authoritative boundary.
    """

    metadata = dict(outcome.metadata)
    metadata["max_steps"] = int(max_steps)
    if outcome.boundary != BoundaryKind.NONE:
        metadata["environment_boundary"] = outcome.boundary.value
    return Outcome(
        reward=outcome.reward,
        progress_delta=outcome.progress_delta,
        truncated=True,
        boundary=BoundaryKind.TIME_LIMIT,
        hazard=outcome.hazard,
        metadata=metadata,
    )


def _initial_raw_observation(
    environment: Any,
    *,
    action_adapter: ActionAdapter,
    action_space: Any,
    bootstrap: bool,
) -> Any:
    if bootstrap:
        action, kwargs = action_adapter.bootstrap(action_space)
        result = environment.step(action, **dict(kwargs))
    else:
        reset = getattr(environment, "reset", None)
        if not callable(reset):
            raise TypeError(
                "bootstrap=False requires an environment with reset()"
            )
        result = reset()
    raw, _info = _normalize_environment_result(result)
    if raw is None:
        raise RuntimeError("environment returned no initial observation")
    return raw


def run_episode(
    environment: Any,
    agent: TransactionalAgent,
    *,
    task_id: str,
    observation_adapter: ObservationAdapter,
    action_adapter: ActionAdapter,
    outcome_adapter: OutcomeAdapter,
    action_space: Any = None,
    max_steps: int = 500,
    initial_raw: Any | None = None,
    bootstrap: bool = True,
) -> EpisodeResult:
    """Run one transactionally correct episode.

    Every environment action attempted after :meth:`TransactionalAgent.act`
    is committed exactly once with :meth:`TransactionalAgent.observe`,
    including a returned-``None`` or raised environment error.  Completion
    and death callbacks therefore cannot steal attribution from the action
    that caused them.
    """

    task_id = str(task_id)
    max_steps = _nonnegative_int(max_steps, name="max_steps")
    clear = getattr(observation_adapter, "clear", None)
    if callable(clear):
        clear(task_id)

    raw = (
        initial_raw
        if initial_raw is not None
        else _initial_raw_observation(
            environment,
            action_adapter=action_adapter,
            action_space=action_space,
            bootstrap=bootstrap,
        )
    )
    observation = observation_adapter.observation(raw, task_id=task_id)
    if not isinstance(observation, Observation):
        raise TypeError("observation adapter must return an Observation")
    initial_observation = observation
    transitions: list[Transition] = []
    final_outcome = Outcome(
        truncated=True,
        boundary=BoundaryKind.TIME_LIMIT,
        metadata={"max_steps": max_steps},
    )

    agent.begin_run(task_id, observation)
    episode_error: BaseException | None = None
    try:
        for step_index in range(max_steps):
            before = observation
            decision = agent.act(before)
            if not isinstance(decision, Decision):
                raise TypeError("agent.act() must return a Decision")

            raw_after: Any = None
            info: dict[str, Any] = {}
            step_error: Exception | None = None
            after = before
            outcome: Outcome | None = None
            try:
                env_action, raw_kwargs = action_adapter.decode(
                    decision.action,
                    action_space,
                )
                if not isinstance(raw_kwargs, Mapping):
                    raise TypeError("action adapter kwargs must be a mapping")
                kwargs = dict(raw_kwargs)
            except Exception:
                # No environment action was issued, so this is not a real
                # transition and must not become negative learning evidence.
                cancel = getattr(agent, "cancel_decision", None)
                if callable(cancel):
                    cancel(decision, reason="action_decode_failed")
                raise
            try:
                result = environment.step(env_action, **kwargs)
                raw_after, info = _normalize_environment_result(result)
                if raw_after is not None:
                    adapted_after = observation_adapter.observation(
                        raw_after,
                        task_id=task_id,
                    )
                    if not isinstance(adapted_after, Observation):
                        raise TypeError(
                            "observation adapter must return an Observation"
                        )
                    after = adapted_after
                    adapted_outcome = outcome_adapter.outcome(
                        before,
                        after,
                        raw_after=raw_after,
                        info=info,
                    )
                    if not isinstance(adapted_outcome, Outcome):
                        raise TypeError("outcome adapter must return an Outcome")
                    outcome = adapted_outcome
            except Exception as exc:
                step_error = exc

            if outcome is None:
                outcome = _environment_error_outcome(
                    step_error,
                    returned_none=raw_after is None and step_error is None,
                )

            # The runner owns this truncation condition, so put it on the
            # final causal transition before committing it.  Environment
            # termination/truncation remains authoritative when it coincides
            # with the local budget boundary.
            if (
                step_index + 1 == max_steps
                and not outcome.terminated
                and not outcome.truncated
            ):
                outcome = _time_limit_outcome(outcome, max_steps=max_steps)

            # The commit precedes every callback and termination check.
            transition = agent.observe(decision, after, outcome)
            if not isinstance(transition, Transition):
                raise TypeError("agent.observe() must return a Transition")
            if transition.decision_id != decision.decision_id:
                raise RuntimeError(
                    "agent.observe() returned a transition for another decision"
                )
            transitions.append(transition)
            observation = after
            final_outcome = transition.outcome
            _dispatch_boundary_callbacks(agent, transition)

            if final_outcome.terminated or final_outcome.truncated:
                break
        else:
            final_outcome = Outcome(
                truncated=True,
                boundary=BoundaryKind.TIME_LIMIT,
                metadata={"max_steps": max_steps},
            )
    except BaseException as exc:
        episode_error = exc
        raise
    finally:
        try:
            agent.end_run(final_outcome)
        except Exception:
            # Cleanup must not replace the causal environment/adapter/commit
            # exception.  With no primary failure, end_run errors still surface.
            if episode_error is None:
                raise

    return EpisodeResult(
        task_id=task_id,
        initial_observation=initial_observation,
        final_observation=observation,
        final_outcome=final_outcome,
        transitions=tuple(transitions),
    )


def make_default_arc_agent() -> TransactionalAgent:
    """Create a compact autonomous agent configured for ARC actions."""

    from .agent import CompactHunterSeeker

    actions = ArcActionAdapter()
    return CompactHunterSeeker(
        click_action_index=actions.click_action_index(),
        safe_action_provider=actions.safe_action_indices,
    )


def run_arc_game(
    game_id: str,
    agent: TransactionalAgent | None = None,
    *,
    arcade: Any | None = None,
    max_steps: int = 500,
    render: bool = False,
) -> EpisodeResult:
    """Run one ARC-AGI-3 game through the compact transactional harness."""

    from arcengine import GameAction

    arcade = arcade or make_arcade()
    agent = agent or make_default_arc_agent()
    environment = arcade.make(
        str(game_id),
        render_mode="terminal" if render else None,
    )
    try:
        return run_episode(
            environment,
            agent,
            task_id=str(game_id),
            observation_adapter=ArcObservationAdapter(),
            action_adapter=ArcActionAdapter(),
            outcome_adapter=ArcOutcomeAdapter(),
            action_space=GameAction,
            max_steps=max_steps,
            bootstrap=True,
        )
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_id")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    result = run_arc_game(
        args.game_id,
        max_steps=args.max_steps,
        render=args.render,
    )
    print(
        json.dumps(
            {
                "task_id": result.task_id,
                "steps": result.steps,
                "boundary": result.final_outcome.boundary.value,
                "progress": result.final_observation.progress,
                "reward": sum(row.outcome.reward for row in result.transitions),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "ARC_ENVIRONMENTS_DIR",
    "ARC_RECORDINGS_DIR",
    "EpisodeResult",
    "TransactionalAgent",
    "make_arcade",
    "make_default_arc_agent",
    "run_arc_game",
    "run_episode",
]
