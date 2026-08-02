"""Causal validation of acquired ``count_at_most`` graph guidance on ls20.

The acquisition route is explicit and intervention-only.  It selects among
the agent's ordinary candidates, so every action still follows the native
``act -> environment.step -> observe`` transaction.  Evaluation uses fresh
agents loaded from one immutable checkpoint:

* ``count_enabled`` retains the verified count goal and graph;
* ``count_ablated`` retains the graph but refutes only the verified count
  planning goal;
* ``graph_disabled`` removes the acquired graph and disables graph guidance.

The result is deliberately limited to causal use of acquired committed count
knowledge.  It is not a from-scratch discovery or generalization claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hunter_seeker_v2.agent import CompactHunterSeeker  # noqa: E402
from hunter_seeker_v2.contracts import AgentConfig, RuntimeMode, SearchConfig  # noqa: E402
from hunter_seeker_v2.memory import StateGraph  # noqa: E402
from hunter_seeker_v2.persistence import load_checkpoint, save_checkpoint  # noqa: E402
from hunter_seeker_v2.policy import RiskArbiter  # noqa: E402
import run_hs_v2_tu93_graph_goal_v1 as _native  # noqa: E402


EXPERIMENT_ID = "hs_v2_ls20_count_graph_goal_v1"
SCHEMA_VERSION = 1
GAME_ID = "ls20"
ENVIRONMENT_SEED = 0
AGENT_SEED = 0
EVALUATION_BUDGET = 60

# This is the recorded level-0 route that exposes the delayed solved frame.
# It is only used by the acquisition arbiter; it is never serialized as a
# teacher or consulted by evaluation agents.
ACQUISITION_ROUTE = (3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_native() -> None:
    """Reuse only the native harness transaction helpers for this game."""

    _native.GAME_ID = GAME_ID
    _native.ENVIRONMENT_SEED = ENVIRONMENT_SEED
    _native.AGENT_SEED = AGENT_SEED
    _native.ACQUISITION_ROUTE = ACQUISITION_ROUTE
    _native.EVALUATION_BUDGET = EVALUATION_BUDGET
    _native.GOAL_ORIGINS = frozenset({"goal_contrast", "completion_motion"})


def _config() -> AgentConfig:
    return AgentConfig(
        runtime_mode=RuntimeMode.AUTONOMOUS,
        seed=AGENT_SEED,
        search=SearchConfig(
            horizon=1,
            beam_width=4,
            exact_graph_bonus=0.0,
            no_change_penalty=0.0,
            loop_penalty=0.0,
            graph_goal_search_enabled=True,
            graph_goal_bonus_bound=1.0,
        ),
        enable_online_learning=False,
        enable_executable_models=False,
    )


def _agent(config: AgentConfig) -> CompactHunterSeeker:
    return _native._agent(config)


def _count_goals(agent: CompactHunterSeeker) -> list[dict[str, Any]]:
    rows = _native._goal_rows(agent)
    return [row for row in rows if row["kind"] in {"count_at_most", "count_at_least"}]


def _count_goal_objects(agent: CompactHunterSeeker) -> list[Any]:
    scope = (GAME_ID, 1)
    return [
        row
        for row in agent.hypotheses._by_scope.get(scope, ())
        if row.origin in {"goal_contrast", "completion_motion"}
        and row.kind in {"count_at_most", "count_at_least"}
    ]


def _acquisition_evidence(
    trace: Any,
    *,
    boundary_bridge_change_fraction: float,
) -> dict[str, Any]:
    transitions = []
    count_trace = []
    value_counts: list[dict[int, int]] = []
    for transition in trace.transitions:
        before = np.asarray(transition.before.observation.frame)
        after = np.asarray(transition.after_observation.frame)
        changed_fraction = float(np.mean(before != after))
        before_count = int(np.count_nonzero(before == 11))
        after_count = int(np.count_nonzero(after == 11))
        before_values = {
            int(value): int(np.count_nonzero(before == value))
            for value in np.unique(before)
        }
        after_values = {
            int(value): int(np.count_nonzero(after == value))
            for value in np.unique(after)
        }
        if not value_counts:
            value_counts.append(before_values)
        value_counts.append(after_values)
        count_trace.append(
            {
                "before_stage": int(transition.before.observation.stage),
                "after_stage": int(transition.after_observation.stage),
                "before_count_value_11": before_count,
                "after_count_value_11": after_count,
                "completed": bool(transition.outcome.completed),
            }
        )
        transitions.append(
            {
                "before_frame_sha256": _native._frame_digest(before),
                "after_frame_sha256": _native._frame_digest(after),
                "before_stage": int(transition.before.observation.stage),
                "after_stage": int(transition.after_observation.stage),
                "action": list(transition.action.key),
                "changed_fraction": changed_fraction,
                "boundary": transition.outcome.boundary.value,
                "protocol": (
                    "delayed_visual_bridge"
                    if changed_fraction <= float(boundary_bridge_change_fraction)
                    else "immediate_stage_swap"
                ),
            }
        )
    same_stage_rows = [
        row
        for row in count_trace
        if row["before_stage"] == row["after_stage"]
    ]
    candidate_values = sorted(
        set().union(*(set(row) for row in value_counts))
        if value_counts
        else set()
    )
    directional_counts: dict[str, dict[str, Any]] = {}
    for value in candidate_values:
        counts = [int(row.get(value, 0)) for row in value_counts]
        deltas = [
            int(row.get(value, 0) - before.get(value, 0))
            for before, row in zip(value_counts, value_counts[1:])
        ]
        same_stage_deltas = [
            int(row["after_count_value_11"] - row["before_count_value_11"])
            for row in same_stage_rows
        ] if value == 11 else []
        directional_deltas = same_stage_deltas or deltas
        directional_counts[str(value)] = {
            "initial_count": counts[0],
            "pre_completion_count": counts[-2] if len(counts) >= 2 else counts[0],
            "returned_successor_count": counts[-1],
            "observed_deltas": deltas,
            "same_stage_value_11_deltas": same_stage_deltas,
            "direction": (
                "decreasing"
                if directional_deltas and all(delta <= 0 for delta in directional_deltas)
                else "increasing"
                if directional_deltas and all(delta >= 0 for delta in directional_deltas)
                else "mixed"
            ),
            "same_stage_delta_values": sorted(set(same_stage_deltas)),
            "step_index_aligned": bool(
                value == 11
                and same_stage_deltas
                and set(same_stage_deltas).issubset({-2, 0})
            ),
            "target": None,
            "diagnostic_only": True,
            "policy_influence": False,
        }
    return {
        "count_value_11_trace": count_trace,
        "semantic_transitions": transitions,
        "boundary_bridge_change_fraction": float(boundary_bridge_change_fraction),
        "shadow_count_diagnostics": {
            "provenance": "immediate_stage_swap_preflight",
            "scope": "old_stage_only_until_boundary",
            "values": directional_counts,
            "target_inference": "abstain",
            "exogenous_status": "unclassified_by_single_episode",
            "policy_influence": False,
        },
    }


def _evaluation_intervention(
    agent: CompactHunterSeeker,
    arm: str,
) -> dict[str, Any]:
    if arm == "count_ablated":
        goals = _count_goal_objects(agent)
        if not goals:
            raise AssertionError("count ablation found no verified count goal")
        for goal in goals:
            goal.refuted = True
        agent.search_engine = agent._new_search_engine()
        return {
            "kind": "refute_verified_count_planning_goals",
            "goal_ids": [str(goal.hypothesis_id) for goal in goals],
            "retained_nodes": len(agent.graph),
            "retained_edges": agent.graph.edge_count,
        }
    if arm == "graph_disabled":
        nodes = len(agent.graph)
        edges = agent.graph.edge_count
        agent.graph = StateGraph()
        agent.search_engine = agent._new_search_engine()
        return {
            "kind": "remove_acquired_state_graph",
            "reason": (
                "disabling goal search alone leaves generic graph distance-to-progress "
                "available to the base scorer"
            ),
            "removed_nodes": nodes,
            "removed_edges": edges,
        }
    return {
        "kind": "retain_acquired_count_goal_and_state_graph",
        "retained_nodes": len(agent.graph),
        "retained_edges": agent.graph.edge_count,
    }


def _run(output: Path) -> dict[str, Any]:
    _configure_native()
    arcade = _native.make_arcade()
    config = _config()
    checkpoint = output / "acquisition_checkpoint.json"

    donor = _agent(config)
    route_arbiter = _native.ExplicitRouteArbiter(ACQUISITION_ROUTE)
    donor.arbiter = route_arbiter
    acquisition_trace = _native._run_until_completion_or_budget(
        arcade=arcade,
        agent=donor,
        budget=len(ACQUISITION_ROUTE),
    )
    donor.arbiter = RiskArbiter(donor.config.policy)
    save_checkpoint(donor, str(checkpoint))

    acquisition = {
        **_native._trace_payload(acquisition_trace),
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
        "active_hypotheses": _native._hypothesis_rows(donor),
        "count_goals": _count_goals(donor),
        "graph_nodes": len(donor.graph),
        "graph_edges": donor.graph.edge_count,
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "evidence": _acquisition_evidence(
            acquisition_trace,
            boundary_bridge_change_fraction=donor.config.boundary_bridge_change_fraction,
        ),
    }

    count_goals = acquisition["count_goals"]
    preflight_checks = {
        "acquisition_route_exact": acquisition["action_sequence"] == list(ACQUISITION_ROUTE),
        "acquisition_forcing_exact": (
            acquisition["provenance"]["forced_decisions"] == len(ACQUISITION_ROUTE)
        ),
        "acquisition_within_risk_limit": (
            acquisition["provenance"]["max_effective_risk"] <= config.policy.risk_limit
        ),
        "acquisition_completed_once": acquisition["completion_count"] == 1,
        "count_goal_exactly_value_11": all(
            row["kind"] == "count_at_most" and row["value_a"] == 11
            for row in count_goals
        ) and bool(count_goals),
        "count_goal_origin_verified": all(
            row["origin"] == "goal_contrast" and row["verified"]
            for row in count_goals
        ) and bool(count_goals),
    }
    # A current environment may legitimately use the immediate-swap protocol.
    # Do not continue into ablations when the old solved frame was unavailable:
    # that would turn a failed preflight into an invalid causal claim.
    if not (
        preflight_checks["count_goal_exactly_value_11"]
        and preflight_checks["count_goal_origin_verified"]
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "game_id": GAME_ID,
            "environment_seed": ENVIRONMENT_SEED,
            "agent_seed": AGENT_SEED,
            "evaluation_budget": EVALUATION_BUDGET,
            "acquisition_route": list(ACQUISITION_ROUTE),
            "acquisition": acquisition,
            "evaluation": {},
            "preflight_failed": True,
            "preflight_failure_reason": (
                "live acquisition did not expose the preregistered old-stage "
                "value-11 count_at_most goal; no causal ablation was run"
            ),
            "checks": preflight_checks,
            "passed": False,
            "script_sha256": _sha256(Path(__file__).resolve()),
        }

    evaluations: dict[str, dict[str, Any]] = {}
    for arm in ("count_enabled", "count_ablated", "graph_disabled"):
        evaluator = _agent(config)
        load_checkpoint(evaluator, str(checkpoint))
        if evaluator.teacher is not None:
            raise AssertionError("evaluation must be teacher-free")
        intervention = _evaluation_intervention(evaluator, arm)
        trace = _native._run_until_completion_or_budget(
            arcade=arcade,
            agent=evaluator,
            budget=EVALUATION_BUDGET,
        )
        plans = [
            {
                "step": index,
                "goal_id": str(decision.metadata.get("graph_goal_id", "")),
                "goal_kind": str(decision.metadata.get("graph_goal_kind", "")),
                "path_length": int(decision.metadata.get("graph_goal_path_length", 0)),
            }
            for index, decision in enumerate(trace.decisions)
            if decision.metadata.get("graph_goal_id")
        ]
        evaluations[arm] = {
            **_native._trace_payload(trace),
            "graph_plans": plans,
            "teacher_attached": False,
            "forced_action_count": 0,
            "intervention": intervention,
        }

    count_goal_ids = [row["hypothesis_id"] for row in count_goals]
    enabled = evaluations["count_enabled"]
    ablated = evaluations["count_ablated"]
    graph_off = evaluations["graph_disabled"]
    checks = {
        **preflight_checks,
        "enabled_count_graph_plans": (
            enabled["graph_plan_count"] > 0
            and all(plan["goal_kind"] == "count_at_most" for plan in enabled["graph_plans"])
        ),
        "enabled_has_expected_goal_id": all(
            plan["goal_id"] in count_goal_ids for plan in enabled["graph_plans"]
        ) and bool(enabled["graph_plans"]),
        "enabled_completes_once": enabled["completion_count"] == 1,
        "enabled_within_budget": enabled["steps"] <= EVALUATION_BUDGET,
        "count_ablated_no_count_plans": not any(
            plan["goal_kind"] == "count_at_most" for plan in ablated["graph_plans"]
        ),
        "graph_off_no_plans": graph_off["graph_plan_count"] == 0,
        "ablated_does_not_complete": ablated["completion_count"] == 0,
        "graph_off_does_not_complete": graph_off["completion_count"] == 0,
        "all_evaluations_teacher_free": all(
            not row["teacher_attached"] for row in evaluations.values()
        ),
        "no_forced_evaluation_actions": all(
            row["forced_action_count"] == 0 for row in evaluations.values()
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "game_id": GAME_ID,
        "environment_seed": ENVIRONMENT_SEED,
        "agent_seed": AGENT_SEED,
        "evaluation_budget": EVALUATION_BUDGET,
        "acquisition_route": list(ACQUISITION_ROUTE),
        "count_goal_ids": count_goal_ids,
        "acquisition": acquisition,
        "evaluation": evaluations,
        "checks": checks,
        "passed": all(checks.values()),
        "script_sha256": _sha256(Path(__file__).resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.out_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = _run(output)
    summary_path = output / "summary.json"
    with summary_path.open("x", encoding="utf-8") as target:
        json.dump(summary, target, indent=2, sort_keys=True, allow_nan=False)
        target.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print(f"artifact: {summary_path}")
    if not summary["passed"]:
        failed = [name for name, value in summary["checks"].items() if not value]
        raise AssertionError(f"ls20 count causal validation failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
