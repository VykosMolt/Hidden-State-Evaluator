"""Live causal validation of tu93 completion-motion reach + graph guidance.

Acquisition is deliberately assisted only by an explicit, fixed 18-action
arbiter.  It selects among the agent's ordinary candidates, so every action
still follows the native ``act -> environment.step -> observe`` transaction.
No teacher object, hidden trajectory lookup, or source-level goal rule exists.

Two isolated donors acquire the same route under configs differing only in
``graph_goal_search_enabled``.  Each is checkpointed and loaded into a fresh,
teacher-free autonomous agent.  The enabled arm retains its acquired graph.
The disabled arm removes acquired graph state after load as the explicit
intervention; disabling only goal search is not a valid graph-off control
because the base policy also consumes graph distance-to-progress.

Pass gate (fixed local tu93 seed 0, 25 autonomous actions):

* acquisition yields only verified ``completion_motion reach|4|14`` goals;
* the graph-enabled arm exactly replays and completes the acquired route;
* the graph-disabled arm records no graph plan and does not complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hunter_seeker_v2.adapters import (  # noqa: E402
    ArcActionAdapter,
    ArcObservationAdapter,
    ArcOutcomeAdapter,
)
from hunter_seeker_v2.agent import CompactHunterSeeker  # noqa: E402
from hunter_seeker_v2.contracts import (  # noqa: E402
    AgentConfig,
    Decision,
    Observation,
    Outcome,
    PolicyConfig,
    RuntimeMode,
    SearchConfig,
    Transition,
)
from hunter_seeker_v2.memory import StateGraph  # noqa: E402
from hunter_seeker_v2.persistence import load_checkpoint, save_checkpoint  # noqa: E402
from hunter_seeker_v2.policy import (  # noqa: E402
    ArbitrationResult,
    RiskArbiter,
)
from hunter_seeker_v2.run_arc import make_arcade  # noqa: E402


EXPERIMENT_ID = "hs_v2_tu93_graph_goal_v1"
SCHEMA_VERSION = 1
GAME_ID = "tu93"
ENVIRONMENT_SEED = 0
AGENT_SEED = 0
EVALUATION_BUDGET = 25
ACQUISITION_ROUTE = (
    4,
    2,
    2,
    4,
    1,
    4,
    2,
    2,
    3,
    3,
    2,
    4,
    4,
    2,
    4,
    1,
    4,
    2,
)
GOAL_ORIGINS = frozenset({"goal_contrast", "completion_motion"})
COUNT_KINDS = frozenset({"count_at_most", "count_at_least"})
REGION_KINDS = frozenset(
    {
        "equal",
        "equal_canonical",
        "mirror_h",
        "mirror_v",
        "self_mirror_h",
        "self_mirror_v",
        "uniform",
    }
)


@dataclass(frozen=True, slots=True)
class RunTrace:
    initial_observation: Observation
    final_observation: Observation
    final_outcome: Outcome
    decisions: tuple[Decision, ...]
    transitions: tuple[Transition, ...]
    stopped_reason: str


class ExplicitRouteArbiter:
    """Acquisition-only selector over the agent's generated candidates."""

    def __init__(self, route: Sequence[int]) -> None:
        self.route = tuple(int(index) for index in route)
        self.position = 0
        self.records: list[dict[str, Any]] = []

    def choose(
        self,
        candidates: Sequence[Any],
        *,
        rng: np.random.Generator,
        epsilon: float | None = None,
    ) -> ArbitrationResult:
        del rng, epsilon
        if self.position >= len(self.route):
            raise AssertionError("acquisition requested an action beyond the fixed route")
        expected = self.route[self.position]
        matches = [
            candidate
            for candidate in candidates
            if int(candidate.action.index) == expected
            and not candidate.action.has_position
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"route step {self.position} expected action {expected}, "
                f"found {len(matches)} matching candidates"
            )
        chosen = matches[0]
        effective_risk = RiskArbiter.effective_risk(chosen)
        self.records.append(
            {
                "route_index": self.position,
                "forced_action": expected,
                "available_candidate_actions": [
                    int(candidate.action.index) for candidate in candidates
                ],
                "effective_risk": effective_risk,
            }
        )
        self.position += 1
        return ArbitrationResult(
            candidate=chosen,
            method="explicit_fixed_route_acquisition",
            safe_candidate_count=len(candidates),
            effective_risk=effective_risk,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_digest(frame: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(frame))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _config(*, graph_enabled: bool) -> AgentConfig:
    # Everything except the graph-plan term is neutralized.  Online learning
    # is off, so the disabled arm cannot inherit an accidental student route.
    return AgentConfig(
        runtime_mode=RuntimeMode.AUTONOMOUS,
        seed=AGENT_SEED,
        search=SearchConfig(
            horizon=1,
            beam_width=4,
            exact_graph_bonus=0.0,
            no_change_penalty=0.0,
            loop_penalty=0.0,
            graph_goal_search_enabled=bool(graph_enabled),
            graph_goal_bonus_bound=1.0,
        ),
        policy=PolicyConfig(
            progress_weight=0.0,
            value_weight=0.0,
            hazard_weight=0.0,
            exploration_weight=0.0,
            learning_progress_weight=0.0,
            memory_weight=0.0,
            action_cost=0.0,
            imagined_uncertainty_weight=0.0,
            # Keep the ordinary production risk threshold active.
            risk_limit=0.70,
            exploration_epsilon=0.0,
            ego_hazard_weight=0.0,
            hypothesis_weight=0.0,
        ),
        enable_online_learning=False,
        enable_executable_models=False,
    )


def _agent(config: AgentConfig) -> CompactHunterSeeker:
    actions = ArcActionAdapter()
    result = CompactHunterSeeker(
        config=config,
        click_action_index=actions.click_action_index(),
        safe_action_provider=actions.safe_action_indices,
        teacher=None,
    )
    if result.teacher is not None:
        raise AssertionError("this experiment must never attach a teacher")
    return result


def _run_until_completion_or_budget(
    *,
    arcade: Any,
    agent: CompactHunterSeeker,
    budget: int,
) -> RunTrace:
    """Run native transactions, stopping immediately after first completion."""

    random.seed(ENVIRONMENT_SEED)
    np.random.seed(ENVIRONMENT_SEED)
    environment = arcade.make(
        GAME_ID,
        seed=ENVIRONMENT_SEED,
        render_mode=None,
    )
    if environment is None:
        raise RuntimeError(f"ARC could not construct {GAME_ID!r}")

    # ARC dynamically loads its engine with the environment.  Resolve the enum
    # afterwards so the action instances belong to the live module generation.
    from arcengine import GameAction

    action_adapter = ArcActionAdapter()
    observation_adapter = ArcObservationAdapter()
    outcome_adapter = ArcOutcomeAdapter()
    action_values = {int(member.value): member for member in GameAction}
    raw = environment.step(action_values[0])
    observation = observation_adapter.observation(raw, task_id=GAME_ID)
    initial_observation = observation
    decisions: list[Decision] = []
    transitions: list[Transition] = []
    final_outcome = Outcome()
    stopped_reason = "budget_exhausted"
    agent.begin_run(GAME_ID, observation)
    try:
        for _step in range(int(budget)):
            decision = agent.act(observation)
            env_action, kwargs = action_adapter.decode(decision.action, GameAction)
            raw_after = environment.step(env_action, **dict(kwargs))
            next_observation = observation_adapter.observation(
                raw_after,
                task_id=GAME_ID,
            )
            outcome = outcome_adapter.outcome(
                observation,
                next_observation,
                raw_after=raw_after,
                info={},
            )
            transition = agent.observe(decision, next_observation, outcome)
            decisions.append(decision)
            transitions.append(transition)
            observation = next_observation
            final_outcome = transition.outcome
            if final_outcome.completed:
                agent.on_level_complete(max(1, int(observation.stage) - 1))
                stopped_reason = "first_completion"
                break
            if final_outcome.terminated or final_outcome.truncated:
                stopped_reason = final_outcome.boundary.value
                break
    finally:
        agent.end_run(final_outcome)
        close = getattr(environment, "close", None)
        if callable(close):
            close()

    return RunTrace(
        initial_observation=initial_observation,
        final_observation=observation,
        final_outcome=final_outcome,
        decisions=tuple(decisions),
        transitions=tuple(transitions),
        stopped_reason=stopped_reason,
    )


def _hypothesis_rows(agent: CompactHunterSeeker) -> list[dict[str, Any]]:
    return [
        {
            "hypothesis_id": row.hypothesis_id,
            "kind": row.kind,
            "origin": row.origin,
            "verified": bool(row.verified),
            "refuted": bool(row.refuted),
            "initial_potential": float(row.initial_potential),
            "value_a": row.value_a,
            "value_b": row.value_b,
        }
        for row in agent.hypotheses.active(GAME_ID, 1)
    ]


def _goal_rows(agent: CompactHunterSeeker) -> list[dict[str, Any]]:
    return [
        row
        for row in _hypothesis_rows(agent)
        if row["origin"] in GOAL_ORIGINS
    ]


def _semantic_transition_rows(trace: RunTrace) -> list[dict[str, Any]]:
    return [
        {
            "before_frame": _frame_digest(transition.before.observation.frame),
            "before_stage": int(transition.before.observation.stage),
            "action": list(transition.action.key),
            "after_frame": _frame_digest(transition.after_observation.frame),
            "after_stage": int(transition.after_observation.stage),
            "progress_delta": float(transition.outcome.progress_delta),
            "boundary": transition.outcome.boundary.value,
        }
        for transition in trace.transitions
    ]


def _trace_payload(trace: RunTrace) -> dict[str, Any]:
    graph_plans = [
        {
            "step": index,
            "action": int(decision.action.index),
            "goal_id": str(decision.metadata.get("graph_goal_id", "")),
            "goal_kind": str(decision.metadata.get("graph_goal_kind", "")),
            "target_id": str(decision.metadata.get("graph_goal_target_id", "")),
            "path_length": int(
                decision.metadata.get("graph_goal_path_length", 0)
            ),
            "phi_delta": float(
                decision.metadata.get("graph_goal_phi_delta", 0.0)
            ),
        }
        for index, decision in enumerate(trace.decisions)
        if str(decision.metadata.get("graph_goal_id", ""))
    ]
    return {
        "steps": len(trace.transitions),
        "action_sequence": [
            int(decision.action.index) for decision in trace.decisions
        ],
        "completion_count": sum(
            int(transition.outcome.completed)
            for transition in trace.transitions
        ),
        "final_progress": float(trace.final_observation.progress),
        "final_stage": int(trace.final_observation.stage),
        "final_boundary": trace.final_outcome.boundary.value,
        "stopped_reason": trace.stopped_reason,
        "graph_plan_count": len(graph_plans),
        "graph_plans": graph_plans,
    }


def _acquire(
    *,
    arcade: Any,
    config: AgentConfig,
    checkpoint_path: Path,
) -> tuple[CompactHunterSeeker, RunTrace, ExplicitRouteArbiter]:
    donor = _agent(config)
    route_arbiter = ExplicitRouteArbiter(ACQUISITION_ROUTE)
    donor.arbiter = route_arbiter
    trace = _run_until_completion_or_budget(
        arcade=arcade,
        agent=donor,
        budget=len(ACQUISITION_ROUTE),
    )
    # The intervention is acquisition-only and is not serialized.
    donor.arbiter = RiskArbiter(donor.config.policy)
    save_checkpoint(donor, str(checkpoint_path))
    return donor, trace, route_arbiter


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="new artifact directory (must not already exist)",
    )
    args = parser.parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (
        args.out_root.expanduser().resolve()
        if args.out_root is not None
        else (
            PROJECT_ROOT
            / "artifacts"
            / "reports"
            / "hunter_seeker_v2"
            / f"tu93_graph_goal_v1_{run_id}"
        ).resolve()
    )
    configs = {
        "graph_enabled": _config(graph_enabled=True),
        "graph_disabled": _config(graph_enabled=False),
    }
    normalized_enabled = replace(
        configs["graph_enabled"],
        search=replace(
            configs["graph_enabled"].search,
            graph_goal_search_enabled=False,
        ),
    )

    # Resolve optional runtime dependencies before reserving an artifact path,
    # so import/setup failures cannot leave a directory resembling a report.
    arcade = make_arcade()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()

    acquisitions: dict[str, dict[str, Any]] = {}
    acquisition_traces: dict[str, RunTrace] = {}
    checkpoint_paths: dict[str, Path] = {}
    for arm, config in configs.items():
        checkpoint_path = checkpoints / f"{arm}.json"
        donor, trace, route_arbiter = _acquire(
            arcade=arcade,
            config=config,
            checkpoint_path=checkpoint_path,
        )
        goals = _goal_rows(donor)
        acquisitions[arm] = {
            **_trace_payload(trace),
            "provenance": {
                "kind": "explicit_fixed_route_arbiter",
                "route": list(ACQUISITION_ROUTE),
                "forced_decisions": route_arbiter.position,
                "route_decisions": route_arbiter.records,
                "max_effective_risk": max(
                    row["effective_risk"] for row in route_arbiter.records
                ),
                "teacher_attached": False,
                "online_learning_enabled": False,
                "transaction_path": "agent.act -> environment.step -> agent.observe",
            },
            "active_hypotheses": _hypothesis_rows(donor),
            "goals": goals,
            "graph_nodes": len(donor.graph),
            "graph_edges": donor.graph.edge_count,
            "checkpoint": str(checkpoint_path.relative_to(output)),
            "checkpoint_sha256": _sha256(checkpoint_path),
        }
        acquisition_traces[arm] = trace
        checkpoint_paths[arm] = checkpoint_path

    evaluations: dict[str, dict[str, Any]] = {}
    for arm, config in configs.items():
        evaluation_agent = _agent(config)
        load_checkpoint(evaluation_agent, str(checkpoint_paths[arm]))
        if evaluation_agent.teacher is not None:
            raise AssertionError("evaluation must be teacher-free")
        intervention: dict[str, Any]
        if arm == "graph_disabled":
            acquired_nodes = len(evaluation_agent.graph)
            acquired_edges = evaluation_agent.graph.edge_count
            evaluation_agent.graph = StateGraph()
            evaluation_agent.search_engine = evaluation_agent._new_search_engine()
            intervention = {
                "kind": "remove_acquired_state_graph",
                "reason": (
                    "graph_goal_search_enabled=False alone still exposes "
                    "generic graph distance-to-progress to the base scorer"
                ),
                "removed_nodes": acquired_nodes,
                "removed_edges": acquired_edges,
            }
        else:
            intervention = {
                "kind": "retain_acquired_state_graph",
                "retained_nodes": len(evaluation_agent.graph),
                "retained_edges": evaluation_agent.graph.edge_count,
            }
        trace = _run_until_completion_or_budget(
            arcade=arcade,
            agent=evaluation_agent,
            budget=EVALUATION_BUDGET,
        )
        evaluations[arm] = {
            **_trace_payload(trace),
            "teacher_attached": False,
            "forced_action_count": 0,
            "intervention": intervention,
            "goal_rows_before_evaluation": acquisitions[arm]["goals"],
        }

    desired_goal = [
        {
            "hypothesis_id": "reach|4|14",
            "kind": "reach",
            "origin": "completion_motion",
            "verified": True,
            "refuted": False,
            "initial_potential": 0.875,
            "value_a": 4,
            "value_b": 14,
        }
    ]
    enabled_eval = evaluations["graph_enabled"]
    disabled_eval = evaluations["graph_disabled"]
    semantic_enabled = _semantic_transition_rows(
        acquisition_traces["graph_enabled"]
    )
    semantic_disabled = _semantic_transition_rows(
        acquisition_traces["graph_disabled"]
    )
    no_count_or_region_goals = all(
        row["kind"] not in COUNT_KINDS | REGION_KINDS
        for arm in acquisitions.values()
        for row in arm["goals"]
    )
    checks = {
        "configs_differ_only_by_graph_goal_flag": (
            normalized_enabled == configs["graph_disabled"]
        ),
        "acquisition_routes_exact": all(
            row["action_sequence"] == list(ACQUISITION_ROUTE)
            for row in acquisitions.values()
        ),
        "acquisition_forcing_exact": all(
            row["provenance"]["forced_decisions"]
            == len(ACQUISITION_ROUTE)
            for row in acquisitions.values()
        ),
        "acquisition_route_within_risk_limit": all(
            row["provenance"]["max_effective_risk"]
            <= configs[arm].policy.risk_limit
            for arm, row in acquisitions.items()
        ),
        "acquisition_transactions_equivalent": (
            semantic_enabled == semantic_disabled
        ),
        "acquisition_each_completed_once": all(
            row["completion_count"] == 1 for row in acquisitions.values()
        ),
        "acquisition_goal_exact": all(
            row["goals"] == desired_goal for row in acquisitions.values()
        ),
        "acquisition_no_count_or_region_goals": no_count_or_region_goals,
        "evaluation_teacher_free": all(
            not row["teacher_attached"] for row in evaluations.values()
        ),
        "enabled_follows_exact_route": (
            enabled_eval["action_sequence"] == list(ACQUISITION_ROUTE)
        ),
        "enabled_completes_once_within_budget": (
            enabled_eval["completion_count"] == 1
            and enabled_eval["steps"] <= EVALUATION_BUDGET
        ),
        "enabled_graph_plan_on_every_route_step": (
            enabled_eval["graph_plan_count"] == len(ACQUISITION_ROUTE)
        ),
        "disabled_has_no_graph_plans": (
            disabled_eval["graph_plan_count"] == 0
        ),
        "disabled_does_not_complete_within_budget": (
            disabled_eval["completion_count"] == 0
            and disabled_eval["steps"] == EVALUATION_BUDGET
        ),
        "disabled_action_sequence_exact": (
            disabled_eval["action_sequence"] == [1] * EVALUATION_BUDGET
        ),
    }
    passed = all(checks.values())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "game_id": GAME_ID,
        "environment_seed": ENVIRONMENT_SEED,
        "agent_seed": AGENT_SEED,
        "evaluation_budget": EVALUATION_BUDGET,
        "acquisition_route": list(ACQUISITION_ROUTE),
        "design": {
            "acquisition": (
                "two isolated, equivalent native-transaction acquisitions "
                "with an explicit fixed-route arbiter and no teacher"
            ),
            "evaluation": (
                "fresh checkpoint-loaded autonomous agents; enabled retains "
                "the acquired graph, disabled removes it before play"
            ),
            "contamination_control": (
                "separate donors, checkpoints, agents, and seeded environment "
                "instances for both arms"
            ),
        },
        "effective_config": {
            arm: {
                "runtime_mode": config.runtime_mode.value,
                "graph_goal_search_enabled": (
                    config.search.graph_goal_search_enabled
                ),
                "online_learning_enabled": config.enable_online_learning,
                "executable_models_enabled": config.enable_executable_models,
                "policy_exploration_epsilon": (
                    config.policy.exploration_epsilon
                ),
                "policy_risk_limit": config.policy.risk_limit,
            }
            for arm, config in configs.items()
        },
        "acquisition": acquisitions,
        "evaluation": evaluations,
        "checks": checks,
        "passed": passed,
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    summary_path = output / "summary.json"
    with summary_path.open("x", encoding="utf-8") as target:
        json.dump(summary, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print(f"artifact: {summary_path}")
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"tu93 causal validation failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
