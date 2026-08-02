"""Causally decomposed fresh-discovery experiment on seeded ``tu93``.

The experiment has one acquisition-only positive control and three isolated
evaluation arms:

* ``acquired_route_positive`` loads the acquired graph and may use graph-goal
  guidance;
* ``acquired_graph_off`` loads the same acquisition, then removes its graph;
* ``fresh_autonomous`` starts with an empty agent and no route, checkpoint,
  teacher, or compatibility mode;
* ``fresh_goal_promotion_off`` is the same fresh construction with both
  supported completion-to-goal promotion paths disabled.

The acquisition route is intervention-only.  It selects the agent's ordinary
candidate set and still commits every action through the native transaction
loop.  A non-completing fresh arm is a valid experimental result, not a test
failure.  The report therefore separates structural validity from empirical
fresh discovery and never turns a failed or blocked run into a positive claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hunter_seeker_v2.agent import CompactHunterSeeker  # noqa: E402
from hunter_seeker_v2.contracts import (  # noqa: E402
    AgentConfig,
    HypothesisConfig,
    RuntimeMode,
)
from hunter_seeker_v2.memory import StateGraph  # noqa: E402
from hunter_seeker_v2.persistence import load_checkpoint, save_checkpoint  # noqa: E402
from hunter_seeker_v2.policy import RiskArbiter  # noqa: E402
from hs_v2_experiment_provenance import (  # noqa: E402
    git_state,
    jsonable,
    runtime_environment,
    sha256_manifest,
    utc_now,
    write_summary,
)
import run_hs_v2_tu93_graph_goal_v1 as _native  # noqa: E402


EXPERIMENT_ID = "hs_v2_fresh_discovery_v1"
SCHEMA_VERSION = 1
GAME_ID = "tu93"
ENVIRONMENT_SEED = 0
AGENT_SEED = 0
EVALUATION_BUDGET = 25
ACQUISITION_ROUTE = _native.ACQUISITION_ROUTE

# This is evidence from the repository state before this task began.  The
# report also records the live status at experiment start; the two records are
# intentionally distinct because this task itself adds files.
PRE_TASK_GIT_BASELINE = {
    "head": "ea40a48f7f41fdf60866eb2067033c62dd14be0f",
    "status_entry_count": 9882,
    "staged_entry_count": 0,
    "unstaged_tracked_entry_count": 39,
    "untracked_entry_count": 9843,
    "status_sha256": "35928c5fd875731ba9c30ea64b7157b8259b790501b6bb27beb6f63c85cd1c5a",
    "capture_note": "Captured before implementation; no reset, cleanup, staging, or push was performed.",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_manifest() -> dict[str, Any]:
    """Hash the exact V2 inputs used by this experiment, by category."""

    source_root = PROJECT_ROOT / "src" / "hunter_seeker_v2"
    manual_root = PROJECT_ROOT / "utilities" / "tests" / "manual"
    unit_root = PROJECT_ROOT / "utilities" / "tests" / "unit"
    docs_root = PROJECT_ROOT / "docs" / "hunter_seeker_v2"
    env_root = PROJECT_ROOT / "data" / "arc_agi3" / "environment_files" / "tu93"
    categories = {
        "v2_source": sorted(source_root.glob("*.py")),
        "v2_docs": [
            docs_root / name
            for name in (
                "ARCHITECTURE.md",
                "RESEARCH.md",
                "CAPABILITY_PARITY.md",
                "FRESH_DISCOVERY_EXPERIMENT.md",
            )
        ],
        "focused_tests": [
            unit_root / "test_hunter_seeker_v2_fresh_discovery.py",
            unit_root / "test_hunter_seeker_v2_search_security_regressions.py",
            unit_root / "test_hunter_seeker_v2_transaction_regressions.py",
        ],
        "manual_runners": [
            manual_root / "hs_v2_experiment_provenance.py",
            manual_root / "run_hs_v2_tu93_graph_goal_v1.py",
            Path(__file__).resolve(),
        ],
        "tu93_environment": sorted(env_root.glob("*/metadata.json"))
        + sorted(env_root.glob("*/tu93.py")),
    }
    manifests = {
        name: sha256_manifest(paths, project_root=PROJECT_ROOT)
        for name, paths in categories.items()
    }
    all_paths = [path for paths in categories.values() for path in paths]
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "categories": manifests,
        "aggregate": sha256_manifest(all_paths, project_root=PROJECT_ROOT),
        "pre_task_git_baseline": PRE_TASK_GIT_BASELINE,
        "experiment_start_git": git_state(PROJECT_ROOT),
        "environment": runtime_environment(),
    }


def acquired_config() -> AgentConfig:
    """Use the existing graph-goal control configuration unchanged."""

    return _native._config(graph_enabled=True)


def fresh_config(*, goal_promotion_enabled: bool = True) -> AgentConfig:
    """Build an autonomous configuration with no compatibility assistance."""

    hypotheses = HypothesisConfig(
        enable_goal_contrast=bool(goal_promotion_enabled),
        enable_completion_motion_reach=bool(goal_promotion_enabled),
    )
    return AgentConfig(
        runtime_mode=RuntimeMode.AUTONOMOUS,
        seed=AGENT_SEED,
        hypotheses=hypotheses,
        enable_online_learning=True,
        enable_executable_models=False,
    )


def build_fresh_agent(config: AgentConfig | None = None) -> CompactHunterSeeker:
    """Construct a fresh agent and assert the required isolation boundary."""

    agent = _native._agent(config or fresh_config())
    if agent.runtime_mode is not RuntimeMode.AUTONOMOUS:
        raise AssertionError("fresh arm is not autonomous")
    if agent.teacher is not None:
        raise AssertionError("fresh arm unexpectedly has a teacher")
    if not isinstance(agent.arbiter, RiskArbiter):
        raise AssertionError("fresh arm does not use the native risk arbiter")
    if len(agent.graph) != 0 or agent.graph.edge_count != 0 or len(agent.evidence) != 0:
        raise AssertionError("fresh arm was constructed with durable state")
    return agent


def freshness_audit(
    agent: CompactHunterSeeker,
    *,
    arm: str,
    checkpoint_loaded: bool,
    route_used: bool,
    forced_action_count: int,
    initial_graph_nodes: int | None = None,
    initial_graph_edges: int | None = None,
    initial_evidence_records: int | None = None,
) -> dict[str, Any]:
    """Return a machine-checkable construction and execution leakage audit."""

    graph_nodes = len(agent.graph) if initial_graph_nodes is None else int(initial_graph_nodes)
    graph_edges = agent.graph.edge_count if initial_graph_edges is None else int(initial_graph_edges)
    evidence_records = (
        len(agent.evidence)
        if initial_evidence_records is None
        else int(initial_evidence_records)
    )
    return {
        "arm": str(arm),
        "runtime_mode": agent.runtime_mode.value,
        "teacher_attached": agent.teacher is not None,
        "checkpoint_loaded": bool(checkpoint_loaded),
        "route_used": bool(route_used),
        "forced_action_count": int(forced_action_count),
        "compatibility_mode": agent.runtime_mode is RuntimeMode.COMPAT_ASSISTED,
        "arbiter_class": type(agent.arbiter).__name__,
        "initial_graph_nodes": graph_nodes,
        "initial_graph_edges": graph_edges,
        "initial_evidence_records": evidence_records,
        "fresh_isolation_pass": bool(
            agent.runtime_mode is RuntimeMode.AUTONOMOUS
            and agent.teacher is None
            and not checkpoint_loaded
            and not route_used
            and int(forced_action_count) == 0
            and type(agent.arbiter) is RiskArbiter
            and graph_nodes == 0
            and graph_edges == 0
            and evidence_records == 0
        ),
    }


def _failure_class(trace: Any, *, budget: int) -> str:
    if trace.final_outcome.completed:
        return "completed"
    if trace.final_outcome.terminated:
        return f"terminated:{trace.final_outcome.boundary.value}"
    if len(trace.transitions) >= int(budget):
        return "budget_exhausted"
    return f"stopped:{trace.stopped_reason}"


def _run_payload(
    *,
    arm: str,
    agent: CompactHunterSeeker,
    trace: Any,
    budget: int,
    checkpoint_loaded: bool,
    route_used: bool,
    forced_action_count: int,
    intervention: Mapping[str, Any],
    initial_state: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    decisions = []
    for index, decision in enumerate(trace.decisions):
        decisions.append(
            {
                "step": index,
                "action": list(decision.action.key),
                "score": float(decision.score),
                "selection_method": str(decision.metadata.get("selection_method", "")),
                "graph_goal_id": str(decision.metadata.get("graph_goal_id", "")),
                "graph_goal_kind": str(decision.metadata.get("graph_goal_kind", "")),
                "graph_goal_path_length": int(
                    decision.metadata.get("graph_goal_path_length", 0)
                ),
                "candidate_count": len(decision.candidates),
            }
        )
    transitions = []
    milestones = [
        {
            "kind": "initial_observation",
            "step": 0,
            "frame_sha256": _native._frame_digest(trace.initial_observation.frame),
            "stage": int(trace.initial_observation.stage),
        }
    ]
    for index, transition in enumerate(trace.transitions, start=1):
        transitions.append(
            {
                "step": index,
                "action": list(transition.action.key),
                "before_frame_sha256": _native._frame_digest(
                    transition.before.observation.frame
                ),
                "after_frame_sha256": _native._frame_digest(
                    transition.after_observation.frame
                ),
                "before_stage": int(transition.before.observation.stage),
                "after_stage": int(transition.after_observation.stage),
                "progress_delta": float(transition.outcome.progress_delta),
                "reward": float(transition.outcome.reward),
                "boundary": transition.outcome.boundary.value,
                "completed": bool(transition.outcome.completed),
            }
        )
        milestones.append(
            {
                "kind": "transition_committed",
                "step": index,
                "boundary": transition.outcome.boundary.value,
            }
        )
        if transition.outcome.completed:
            milestones.append({"kind": "level_completed", "step": index})
    goals = _native._goal_rows(agent)
    if goals:
        milestones.append(
            {
                "kind": "verified_goal_present",
                "step": len(trace.transitions),
                "goal_ids": [str(row["hypothesis_id"]) for row in goals],
            }
        )
    audit = freshness_audit(
        agent,
        arm=arm,
        checkpoint_loaded=checkpoint_loaded,
        route_used=route_used,
        forced_action_count=forced_action_count,
        initial_graph_nodes=(initial_state or {}).get("graph_nodes"),
        initial_graph_edges=(initial_state or {}).get("graph_edges"),
        initial_evidence_records=(initial_state or {}).get("evidence_records"),
    )
    return {
        "arm": arm,
        "budget": int(budget),
        "steps": len(trace.transitions),
        "checkpoint_loaded": bool(checkpoint_loaded),
        "route_used": bool(route_used),
        "forced_action_count": int(forced_action_count),
        "action_sequence": [list(decision.action.key) for decision in trace.decisions],
        "completion_count": sum(
            int(transition.outcome.completed) for transition in trace.transitions
        ),
        "final_stage": int(trace.final_observation.stage),
        "final_progress": float(trace.final_observation.progress),
        "final_boundary": trace.final_outcome.boundary.value,
        "stopped_reason": trace.stopped_reason,
        "failure_classification": _failure_class(trace, budget=budget),
        "decisions": decisions,
        "transitions": transitions,
        "milestones": milestones,
        "verified_goals": goals,
        "graph_plan_count": sum(bool(row["graph_goal_id"]) for row in decisions),
        "graph_nodes": len(agent.graph),
        "graph_edges": agent.graph.edge_count,
        "measurement_summary": agent.measurement_summary(),
        "intervention": jsonable(intervention),
        "leakage_audit": audit,
    }


def _write_json(path: Path, payload: Any) -> None:
    write_summary(path, payload, overwrite=False)


def _run_experiment(output: Path, *, args: argparse.Namespace) -> dict[str, Any]:
    manifest = _bounded_manifest()
    arcade = _native.make_arcade()
    config = acquired_config()
    checkpoint = output / "acquisition_graph_enabled_checkpoint.json"
    graph_off_checkpoint = output / "acquisition_graph_disabled_checkpoint.json"

    donor = _native._agent(config)
    route_arbiter = _native.ExplicitRouteArbiter(ACQUISITION_ROUTE)
    donor.arbiter = route_arbiter
    acquisition_trace = _native._run_until_completion_or_budget(
        arcade=arcade,
        agent=donor,
        budget=len(ACQUISITION_ROUTE),
    )
    donor.arbiter = RiskArbiter(donor.config.policy)
    save_checkpoint(donor, str(checkpoint))
    acquisition = _run_payload(
        arm="acquisition",
        agent=donor,
        trace=acquisition_trace,
        budget=len(ACQUISITION_ROUTE),
        checkpoint_loaded=False,
        route_used=True,
        forced_action_count=route_arbiter.position,
        intervention={
            "kind": "explicit_fixed_route_acquisition_only",
            "route": list(ACQUISITION_ROUTE),
            "transaction_path": "agent.act -> environment.step -> agent.observe",
            "teacher_attached": False,
            "online_learning_enabled": False,
        },
    )
    acquisition["route_decisions"] = route_arbiter.records
    acquisition["checkpoint_sha256"] = _sha256(checkpoint)

    graph_off_config = _native._config(graph_enabled=False)
    graph_off_donor = _native._agent(graph_off_config)
    graph_off_route = _native.ExplicitRouteArbiter(ACQUISITION_ROUTE)
    graph_off_donor.arbiter = graph_off_route
    graph_off_acquisition_trace = _native._run_until_completion_or_budget(
        arcade=arcade,
        agent=graph_off_donor,
        budget=len(ACQUISITION_ROUTE),
    )
    graph_off_donor.arbiter = RiskArbiter(graph_off_donor.config.policy)
    save_checkpoint(graph_off_donor, str(graph_off_checkpoint))
    graph_off_acquisition_payload = _run_payload(
        arm="acquisition_graph_off",
        agent=graph_off_donor,
        trace=graph_off_acquisition_trace,
        budget=len(ACQUISITION_ROUTE),
        checkpoint_loaded=False,
        route_used=True,
        forced_action_count=graph_off_route.position,
        intervention={
            "kind": "matched_explicit_fixed_route_acquisition_only",
            "route": list(ACQUISITION_ROUTE),
            "graph_goal_search_enabled": False,
        },
        initial_state={
            "graph_nodes": 0,
            "graph_edges": 0,
            "evidence_records": 0,
        },
    )
    acquisition["graph_off_matched_acquisition"] = {
        **graph_off_acquisition_payload,
        "checkpoint_sha256": _sha256(graph_off_checkpoint),
    }

    evaluations: dict[str, dict[str, Any]] = {}
    positive = _native._agent(config)
    load_checkpoint(positive, str(checkpoint))
    positive_initial_state = {
        "graph_nodes": len(positive.graph),
        "graph_edges": positive.graph.edge_count,
        "evidence_records": len(positive.evidence),
    }
    positive_trace = _native._run_until_completion_or_budget(
        arcade=arcade, agent=positive, budget=EVALUATION_BUDGET
    )
    evaluations["acquired_route_positive"] = _run_payload(
        arm="acquired_route_positive",
        agent=positive,
        trace=positive_trace,
        budget=EVALUATION_BUDGET,
        checkpoint_loaded=True,
        route_used=False,
        forced_action_count=0,
        intervention={
            "kind": "retain_acquired_graph_and_verified_goals",
            "checkpoint": checkpoint.name,
        },
        initial_state=positive_initial_state,
    )

    graph_off = _native._agent(graph_off_config)
    load_checkpoint(graph_off, str(graph_off_checkpoint))
    retained_nodes = len(graph_off.graph)
    retained_edges = graph_off.graph.edge_count
    graph_off.graph = StateGraph()
    graph_off.search_engine = graph_off._new_search_engine()
    graph_off_trace = _native._run_until_completion_or_budget(
        arcade=arcade, agent=graph_off, budget=EVALUATION_BUDGET
    )
    evaluations["acquired_graph_off"] = _run_payload(
        arm="acquired_graph_off",
        agent=graph_off,
        trace=graph_off_trace,
        budget=EVALUATION_BUDGET,
        checkpoint_loaded=True,
        route_used=False,
        forced_action_count=0,
        intervention={
            "kind": "remove_acquired_state_graph",
            "removed_nodes": retained_nodes,
            "removed_edges": retained_edges,
            "reason": (
                "graph_goal_search_enabled=False alone leaves the base graph-distance "
                "scorer available, so the graph is removed as in the existing V2 control"
            ),
        },
        initial_state={
            "graph_nodes": 0,
            "graph_edges": 0,
            "evidence_records": len(graph_off.evidence),
        },
    )

    fresh = build_fresh_agent()
    fresh_initial_state = {
        "graph_nodes": len(fresh.graph),
        "graph_edges": fresh.graph.edge_count,
        "evidence_records": len(fresh.evidence),
    }
    fresh_trace = _native._run_until_completion_or_budget(
        arcade=arcade, agent=fresh, budget=EVALUATION_BUDGET
    )
    evaluations["fresh_autonomous"] = _run_payload(
        arm="fresh_autonomous",
        agent=fresh,
        trace=fresh_trace,
        budget=EVALUATION_BUDGET,
        checkpoint_loaded=False,
        route_used=False,
        forced_action_count=0,
        intervention={
            "kind": "none",
            "construction": "new autonomous agent; no teacher/checkpoint/route/compatibility",
        },
        initial_state=fresh_initial_state,
    )

    fresh_off = build_fresh_agent(fresh_config(goal_promotion_enabled=False))
    fresh_off_initial_state = {
        "graph_nodes": len(fresh_off.graph),
        "graph_edges": fresh_off.graph.edge_count,
        "evidence_records": len(fresh_off.evidence),
    }
    fresh_off_trace = _native._run_until_completion_or_budget(
        arcade=arcade, agent=fresh_off, budget=EVALUATION_BUDGET
    )
    evaluations["fresh_goal_promotion_off"] = _run_payload(
        arm="fresh_goal_promotion_off",
        agent=fresh_off,
        trace=fresh_off_trace,
        budget=EVALUATION_BUDGET,
        checkpoint_loaded=False,
        route_used=False,
        forced_action_count=0,
        intervention={
            "kind": "disable_goal_promotion_paths",
            "enable_goal_contrast": False,
            "enable_completion_motion_reach": False,
        },
        initial_state=fresh_off_initial_state,
    )

    checks = {
        "acquisition_route_exact": [
            int(action[0]) for action in acquisition["action_sequence"]
        ]
        == [int(action) for action in ACQUISITION_ROUTE],
        "acquisition_completed_once": acquisition["completion_count"] == 1,
        "acquisition_forcing_exact": acquisition["forced_action_count"]
        == len(ACQUISITION_ROUTE),
        "graph_off_acquisition_route_exact": [
            int(action[0])
            for action in acquisition["graph_off_matched_acquisition"][
                "action_sequence"
            ]
        ]
        == [int(action) for action in ACQUISITION_ROUTE],
        "graph_off_acquisition_completed_once": acquisition[
            "graph_off_matched_acquisition"
        ]["completion_count"]
        == 1,
        "positive_control_teacher_free": not evaluations["acquired_route_positive"][
            "leakage_audit"
        ]["teacher_attached"],
        "graph_off_has_no_graph_plans": evaluations["acquired_graph_off"][
            "graph_plan_count"
        ]
        == 0,
        "fresh_autonomous_isolation": evaluations["fresh_autonomous"][
            "leakage_audit"
        ]["fresh_isolation_pass"],
        "fresh_goal_off_isolation": evaluations["fresh_goal_promotion_off"][
            "leakage_audit"
        ]["fresh_isolation_pass"],
        "fresh_goal_off_switches_disabled": (
            fresh_off.config.hypotheses.enable_goal_contrast is False
            and fresh_off.config.hypotheses.enable_completion_motion_reach is False
        ),
    }
    fresh_completed = evaluations["fresh_autonomous"]["completion_count"] > 0
    if not all(checks.values()):
        verdict = "INVALID_STRUCTURAL_CHECK_FAILURE"
        verdict_class = "invalid"
    elif fresh_completed:
        verdict = "VALID_FRESH_DISCOVERY_COMPLETION_OBSERVED"
        verdict_class = "positive_observation"
    else:
        verdict = "VALID_NO_FRESH_DISCOVERY_WITHIN_BUDGET"
        verdict_class = "negative_or_inconclusive"
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "run_id": output.name,
        "created_at_utc": utc_now(),
        "game_id": GAME_ID,
        "environment_seed": ENVIRONMENT_SEED,
        "agent_seed": AGENT_SEED,
        "evaluation_budget": EVALUATION_BUDGET,
        "acquisition_route": list(ACQUISITION_ROUTE),
        "design": {
            "comparison_scope": (
                "acquired graph use versus acquired graph removal, plus a separately "
                "reported default-policy fresh autonomous arm"
            ),
            "acquisition": "explicit route selects ordinary candidates only; no teacher or online learning",
            "freshness": "fresh arms are constructed, not checkpoint-loaded, and do not receive the acquisition route",
            "goal_switch": "goal promotion is disabled by both supported completion promotion flags",
        },
        "provenance_manifest": manifest,
        "acquisition": acquisition,
        "evaluations": evaluations,
        "checks": checks,
        "verdict": {"code": verdict, "class": verdict_class, "fresh_completion": fresh_completed},
        "blocked": False,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "arguments": jsonable(vars(args)),
    }


def _blocked_summary(output: Path, *, args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    try:
        manifest: Any = _bounded_manifest()
    except Exception as manifest_error:
        manifest = {
            "blocked": True,
            "error": {
                "type": type(manifest_error).__name__,
                "message": str(manifest_error),
            },
            "pre_task_git_baseline": PRE_TASK_GIT_BASELINE,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "run_id": output.name,
        "created_at_utc": utc_now(),
        "game_id": GAME_ID,
        "blocked": True,
        "failure_classification": "runtime_setup_or_execution_error",
        "error": {"type": type(error).__name__, "message": str(error)},
        "verdict": {"code": "BLOCKED_RUNTIME_ERROR", "class": "blocked"},
        "provenance_manifest": manifest,
        "arguments": jsonable(vars(args)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="new artifact directory; it must not already exist",
    )
    args = parser.parse_args(argv)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (
        args.out_root.expanduser().resolve()
        if args.out_root is not None
        else PROJECT_ROOT
        / "artifacts"
        / "reports"
        / "hunter_seeker_v2"
        / f"fresh_discovery_v1_{run_id}"
    )
    output.mkdir(parents=True, exist_ok=False)
    try:
        summary = _run_experiment(output, args=args)
    except Exception as error:
        summary = _blocked_summary(output, args=args, error=error)
    _write_json(output / "summary.json", summary)
    _write_json(output / "provenance_manifest.json", summary["provenance_manifest"])
    _write_json(output / "verdict.json", summary["verdict"])
    for arm, payload in summary.get("evaluations", {}).items():
        _write_json(output / f"trace_{arm}.json", payload)
    _write_json(output / "trace_acquisition.json", summary.get("acquisition", {}))
    matched = summary.get("acquisition", {}).get("graph_off_matched_acquisition")
    if matched is not None:
        _write_json(output / "trace_acquisition_graph_off.json", matched)
    print(json.dumps(summary["verdict"], indent=2, sort_keys=True, allow_nan=False))
    print(f"artifact: {output / 'summary.json'}")
    return 0 if not summary.get("blocked") else 2


if __name__ == "__main__":
    raise SystemExit(main())
