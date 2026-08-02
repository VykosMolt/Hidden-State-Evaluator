"""Authoritative Sol-frozen TU93 fresh-retention experiment.

This runner is intentionally independent of the historical acquisition and
graph-goal runners.  It compares only two autonomous, teacher-free learners:

* ``persistent_fresh`` keeps one newly constructed agent across attempts;
* ``episode_isolated`` constructs a constructor-baseline agent for every
  attempt.

Both agents may learn online inside an episode.  No acquisition, route,
checkpoint, teacher, compatibility mode, task trajectory, or graph-specific
intervention is available in this module.  The experiment fails closed unless
the exact local TU93 variant and source hashes are present.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACT_ROOT = (PROJECT_ROOT / "artifacts" / "reports" / "hunter_seeker_v2").resolve()
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hunter_seeker_v2.adapters import (  # noqa: E402
    ArcActionAdapter,
    ArcObservationAdapter,
    ArcOutcomeAdapter,
)
from hunter_seeker_v2.agent import CompactHunterSeeker  # noqa: E402
from hunter_seeker_v2.contracts import (  # noqa: E402
    AgentConfig,
    BoundaryKind,
    Decision,
    Observation,
    Outcome,
    RuntimeMode,
    Transition,
)
from hunter_seeker_v2.models import GridFeatureBackend  # noqa: E402
from hunter_seeker_v2.policy import RiskArbiter  # noqa: E402


EXPERIMENT_ID = "hs_v2_tu93_fresh_retention_v1"
SCHEMA_VERSION = 1
GAME_ID = "tu93"
ENVIRONMENT_VERSION = "0768757b"
EXACT_GAME_ID = f"{GAME_ID}-{ENVIRONMENT_VERSION}"
ENVIRONMENT_DIR = (
    PROJECT_ROOT / "data" / "arc_agi3" / "environment_files" / GAME_ID / ENVIRONMENT_VERSION
)
SOURCE_PATH = ENVIRONMENT_DIR / "tu93.py"
METADATA_PATH = ENVIRONMENT_DIR / "metadata.json"
EXPECTED_SOURCE_SHA256 = "80e41888f9f7b1a0c03e02c0aff3814e0fd68eb5b35ef22bb3649c87fc60a23f"
EXPECTED_METADATA_SHA256 = "ad29d072e6977cb914b729c0f461157d971f4182656adb0f811c77bf14faa20f"

REPLICATE_COUNT = 32
ATTEMPT_LIMIT = 20
ACTION_LIMIT = 50
MCMEMAR_ALPHA = 0.05
ARM_PERSISTENT = "persistent_fresh"
ARM_ISOLATED = "episode_isolated"
ARMS = (ARM_PERSISTENT, ARM_ISOLATED)
EXPECTED_PAIR_IDS = tuple(f"replicate-{index:02d}" for index in range(REPLICATE_COUNT))
CONFIG_PATHS = (
    PROJECT_ROOT / "barbados" / "pyproject.toml",
    PROJECT_ROOT / "pytest.ini",
    PROJECT_ROOT / "requirements" / "requirements.txt",
)
REQUIRED_STATE_COMPONENTS = (
    "action_prior",
    "affordance",
    "competence",
    "diagnostics",
    "evidence",
    "exogenous",
    "graph",
    "hypotheses",
    "learner",
    "model",
    "perception",
    "policy_state",
    "private_rng_states",
    "replay",
    "rng_state",
    "runtime_fields",
    "student",
    "treatment_state",
)

# This is deliberately process-local and never enters any JSON record.  A
# serialized summary can describe a run, but it cannot prove that this process
# performed the exact-environment execution which made the causal result
# eligible.
_LIVE_RUN_TOKENS: set[object] = set()


class ExperimentContractError(RuntimeError):
    """The requested experiment cannot be run under its frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert runtime records to strict JSON without losing array identity."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return repr(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    files = []
    for path in sorted({Path(item).resolve() for item in paths}):
        if not path.is_file():
            raise ExperimentContractError(f"manifest input is not a file: {path}")
        files.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {"algorithm": "sha256", "file_count": len(files), "files": files, "aggregate_sha256": _digest(files)}


def _package_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "arc-agi", "arcengine"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ExperimentContractError(f"required package is unavailable: {name}") from exc
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": packages,
    }


def _schema_manifest() -> dict[str, Any]:
    schema = {
        "schema_version": SCHEMA_VERSION,
        "summary_required": [
            "schema_version", "experiment", "authoritative", "blocked", "design",
            "provenance_manifest", "finalization_marker", "pairs", "checks", "verdict",
        ],
        "pair_required": [
            "pair_id", "agent_seed", "attempts", "persistent_completed",
            "isolated_completed", "valid_pair", "finalized", "valid_finalization",
        ],
        "attempt_required": [
            "pair_id", "attempt_id", "attempt_index", "schedule", "arm_order", "arms",
            "finalized", "valid_pair",
        ],
        "row_required": [
            "pair_id", "attempt_id", "arm", "agent_seed", "status",
            "bootstrap_environment_step", "decisions", "committed_transitions",
            "environment_steps", "state_store_before", "state_store_after",
            "executed_action_evidence", "completed", "completion_count",
            "completion_lineage", "final_stage", "final_boundary",
        ],
        "state_components": list(REQUIRED_STATE_COMPONENTS),
    }
    return {
        "algorithm": "sha256",
        "schema_version": SCHEMA_VERSION,
        "definition_sha256": _digest(schema),
        "required_fields": schema,
    }


def _retire_live_run_token(token: object | None) -> None:
    if token is not None:
        _LIVE_RUN_TOKENS.discard(token)


def verify_environment_contract() -> dict[str, Any]:
    """Verify the exact local/offline environment before importing its code."""

    if not ENVIRONMENT_DIR.is_dir():
        raise ExperimentContractError(f"required local environment directory is absent: {ENVIRONMENT_DIR}")
    if not SOURCE_PATH.is_file() or not METADATA_PATH.is_file():
        raise ExperimentContractError("required TU93 source or metadata is absent")
    source_sha = _sha256(SOURCE_PATH)
    metadata_sha = _sha256(METADATA_PATH)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ExperimentContractError(
            f"TU93 source hash mismatch: expected {EXPECTED_SOURCE_SHA256}, got {source_sha}"
        )
    if metadata_sha != EXPECTED_METADATA_SHA256:
        raise ExperimentContractError(
            f"TU93 metadata hash mismatch: expected {EXPECTED_METADATA_SHA256}, got {metadata_sha}"
        )
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentContractError("TU93 metadata cannot be read as JSON") from exc
    if not isinstance(metadata, Mapping) or metadata.get("game_id") != EXACT_GAME_ID:
        raise ExperimentContractError("TU93 metadata does not identify the exact frozen variant")
    declared_dir = Path(str(metadata.get("local_dir", ""))).resolve()
    if declared_dir != ENVIRONMENT_DIR.resolve():
        raise ExperimentContractError("TU93 metadata local_dir does not match the selected local variant")
    return {
        "game_id": EXACT_GAME_ID,
        "variant": ENVIRONMENT_VERSION,
        "verified": True,
        "source": {"path": str(SOURCE_PATH.relative_to(PROJECT_ROOT)), "expected_sha256": EXPECTED_SOURCE_SHA256, "observed_sha256": source_sha, "verified": True},
        "metadata": {"path": str(METADATA_PATH.relative_to(PROJECT_ROOT)), "expected_sha256": EXPECTED_METADATA_SHA256, "observed_sha256": metadata_sha, "verified": True},
        "selection": "local_offline_exact_version",
    }


def make_exact_arcade() -> Any:
    """Construct one offline loader after the exact input gate passes."""

    verify_environment_contract()
    try:
        import arc_agi
        from arc_agi.base import OperationMode
    except ImportError as exc:
        raise ExperimentContractError("arc_agi is unavailable; refusing non-local fallback") from exc
    return arc_agi.Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENVIRONMENT_DIR.parent.parent),
        recordings_dir=str(PROJECT_ROOT / "artifacts" / "recordings" / "arc_agi3"),
    )


def make_exact_environment(seed: int, *, arcade: Any | None = None) -> Any:
    """Construct only the pinned local variant; never download or select latest."""

    verify_environment_contract()
    arcade = arcade or make_exact_arcade()
    environment = arcade.make(EXACT_GAME_ID, seed=int(seed), render_mode=None)
    if environment is None:
        raise ExperimentContractError(f"offline Arcade could not construct {EXACT_GAME_ID}")
    return environment


def default_agent_config(seed: int) -> AgentConfig:
    """The default autonomous config, with only seed and executable models changed."""

    return AgentConfig(seed=int(seed), enable_executable_models=False)


def build_constructor_baseline_agent(seed: int) -> CompactHunterSeeker:
    agent = CompactHunterSeeker(
        config=default_agent_config(seed),
        click_action_index=ArcActionAdapter().click_action_index(),
        safe_action_provider=ArcActionAdapter().safe_action_indices,
        teacher=None,
    )
    if agent.runtime_mode is not RuntimeMode.AUTONOMOUS:
        raise ExperimentContractError("constructor baseline is not autonomous")
    if agent.teacher is not None:
        raise ExperimentContractError("constructor baseline has a teacher")
    if type(agent.arbiter) is not RiskArbiter:
        raise ExperimentContractError("constructor baseline does not use ordinary RiskArbiter")
    if not isinstance(agent.representation_backend, GridFeatureBackend):
        raise ExperimentContractError("constructor baseline does not use default GridFeature")
    if agent.config.enable_executable_models:
        raise ExperimentContractError("executable models must be disabled")
    if len(agent.graph) != 0 or agent.graph.edge_count != 0 or len(agent.evidence) != 0:
        raise ExperimentContractError("constructor baseline contains durable state")
    return agent


def _observation_fingerprint(observation: Observation) -> str:
    return _digest(
        {
            "state_id": observation.state_id,
            "frame": np.asarray(observation.frame),
            "available_actions": observation.available_actions,
            "stage": observation.stage,
            "progress": observation.progress,
            "metadata": observation.metadata,
        }
    )


def _component_state(agent: CompactHunterSeeker) -> dict[str, Any]:
    graph = agent.graph.export_state()
    evidence = agent.evidence.export_state()
    hypotheses = agent.hypotheses.state_dict()
    student = agent.student_policy.state_dict()
    dynamics = agent.dynamics.state_dict()
    diagnostics = agent.diagnostics.export_state()
    replay_items = tuple(agent.learner.replay.items)
    replay_ids = [item.transition.transition_id for item in replay_items]
    replay_payloads = [_jsonable(item) for item in replay_items]
    policy_state = {
        "action_prior": agent.prior.state_dict(),
        "affordance": agent.affordances.state_dict(),
        "competence": agent.competence.state_dict(),
        "exogenous": agent.exogenous.state_dict(),
        "perception": agent.perception.export_state(),
        "ego": agent.ego.state_dict(),
    }
    retained_rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "agent_private": agent._rng.bit_generator.state,
        "model": agent.dynamics._rng.bit_generator.state,
        "learner": agent.learner._rng.bit_generator.state,
    }
    learner = {
        "real_updates": agent.learner.real_updates,
        "replay_updates": agent.learner.replay_updates,
        "last_replay_loss": agent.learner.last_replay_loss,
        "replay_ids": replay_ids,
        "replay_payloads": replay_payloads,
    }
    runtime_fields = {
        name: {"supported": True, "value": _jsonable(getattr(agent, name))}
        for name in (
            "_total_transition_count",
            "_step",
            "_decision_counter",
            "_awaiting_visual_reset",
            "_task_id",
            "_current_snapshot",
            "_pending",
            "_last_transition",
            "_committed_decision_ids",
            "_run_active",
            "_resume_ready",
            "_run_transition_count",
        )
        if hasattr(agent, name)
    }
    for name in ("_external_runtime_state", "_environment_internal_state"):
        runtime_fields.setdefault(
            name,
            {"supported": False, "value": None, "status": "unsupported_runtime_field"},
        )
    return {
        "graph": graph,
        "evidence": evidence,
        "hypotheses": hypotheses,
        "student": student,
        "learner": learner,
        "runtime_fields": runtime_fields,
        "model": dynamics,
        "diagnostics": diagnostics,
        "policy_state": policy_state,
        "treatment_state": {
            "label": "retained_rng",
            "rng_state": retained_rng_state,
        },
    }


def _private_rng_states(agent: CompactHunterSeeker) -> dict[str, Any]:
    """Capture discoverable private RNG states and label unsupported gaps."""

    states: dict[str, Any] = {
        "python_global": {"supported": True, "value": _jsonable(random.getstate())},
        "numpy_global": {"supported": True, "value": _jsonable(np.random.get_state())},
    }
    roots: dict[str, Any] = {"agent": agent}
    for name, value in vars(agent).items():
        if name.startswith("_") and value is not None:
            roots[f"agent.{name}"] = value
    for component_name in (
        "dynamics", "learner", "prior", "affordances", "competence",
        "exogenous", "perception", "student_policy", "ego", "hypotheses",
        "graph", "evidence",
    ):
        component = getattr(agent, component_name, None)
        if component is None:
            continue
        for name, value in getattr(component, "__dict__", {}).items():
            if name.startswith("_"):
                roots[f"{component_name}.{name}"] = value
    for path, value in roots.items():
        if "rng" not in path.lower() and "random" not in path.lower():
            continue
        try:
            if hasattr(value, "bit_generator"):
                state = value.bit_generator.state
            elif hasattr(value, "getstate"):
                state = value.getstate()
            elif hasattr(value, "state"):
                state = value.state
            else:
                states[path] = {"supported": False, "value": None, "status": "private_rng_state_not_exposed"}
                continue
            states[path] = {"supported": True, "value": _jsonable(state)}
        except BaseException as exc:
            states[path] = {
                "supported": False,
                "value": None,
                "status": "private_rng_state_read_failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
    for path in ("agent._rng", "dynamics._rng", "learner._rng"):
        states.setdefault(path, {"supported": False, "value": None, "status": "private_rng_not_available"})
    return states


def state_store_fingerprint(agent: CompactHunterSeeker) -> dict[str, Any]:
    """Fingerprint actual mutable stores and expose their runtime counts."""

    state = _component_state(agent)
    hypothesis_summary = agent.hypotheses.summary()
    student_summary = agent.student_policy.summary()
    diagnostic_summary = agent.diagnostics.summary()
    private_rng_states = _private_rng_states(agent)
    components = {
        "graph": {"fingerprint": _digest(state["graph"]), "nodes": len(agent.graph), "edges": agent.graph.edge_count, "revisits": agent.graph.revisit_count},
        "evidence": {"fingerprint": _digest(state["evidence"]), "records": len(agent.evidence)},
        "replay": {
            "fingerprint": _digest(state["learner"]["replay_payloads"]),
            "items": len(agent.learner.replay),
            "payloads": state["learner"]["replay_payloads"],
        },
        "hypotheses": {"fingerprint": _digest(state["hypotheses"]), "count": hypothesis_summary["hypotheses"], "verified": hypothesis_summary["verified"], "goals": hypothesis_summary["goals"]},
        "student": {"fingerprint": _digest(state["student"]), "task_rows": student_summary["task_rows"], "transfer_rows": student_summary["transfer_rows"], "online_updates": student_summary["online_updates"]},
        "learner": {"fingerprint": _digest(state["learner"]), "real_updates": agent.learner.real_updates, "replay_updates": agent.learner.replay_updates},
        "model": {"fingerprint": _digest(state["model"]), "dynamics_updates": agent.dynamics.update_count},
        "diagnostics": {"fingerprint": _digest(state["diagnostics"]), "decisions": diagnostic_summary["decision_count"], "transitions": diagnostic_summary["transition_count"]},
        "policy_state": {"fingerprint": _digest(state["policy_state"])},
        "action_prior": {"fingerprint": _digest(state["policy_state"]["action_prior"]), "global_rows": len(agent.prior._global), "task_rows": len(agent.prior._task)},
        "affordance": {"fingerprint": _digest(state["policy_state"]["affordance"]), "transferable_rows": len(agent.affordances._rows), "task_rows": len(agent.affordances._task_rows)},
        "competence": {"fingerprint": _digest(state["policy_state"]["competence"]), "updates": int(agent.competence._updates)},
        "exogenous": {"fingerprint": _digest(state["policy_state"]["exogenous"]), "finalized_episodes": int(agent.exogenous.finalized_episodes)},
        "perception": {"fingerprint": _digest(state["policy_state"]["perception"]), "tracks": agent.perception.track_count},
        "rng_state": {"fingerprint": _digest(state["treatment_state"]["rng_state"])},
        "runtime_fields": {
            "fingerprint": _digest(state["runtime_fields"]),
            "fields": state["runtime_fields"],
        },
        "private_rng_states": {
            "fingerprint": _digest(private_rng_states),
            "states": private_rng_states,
        },
        "treatment_state": {
            "label": state["treatment_state"]["label"],
            "fingerprint": _digest(state["treatment_state"]),
            "retained_rng": True,
        },
    }
    return {"overall_fingerprint": _digest(components), "components": components}


def runtime_leakage_audit(
    agent: CompactHunterSeeker,
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    arm: str,
    constructor_baseline_required: bool,
) -> dict[str, Any]:
    """Audit runtime state rather than trusting arm labels supplied by a caller."""

    static_audit = _static_forbidden_path_audit()
    no_forbidden_imports = bool(static_audit["pass"])
    empty_constructor_state = bool(
        before["components"]["graph"]["nodes"] == 0
        and before["components"]["graph"]["edges"] == 0
        and before["components"]["evidence"]["records"] == 0
        and before["components"]["replay"]["items"] == 0
        and before["components"].get("action_prior", {}).get("global_rows") == 0
        and before["components"].get("action_prior", {}).get("task_rows") == 0
        and before["components"].get("affordance", {}).get("transferable_rows") == 0
        and before["components"].get("affordance", {}).get("task_rows") == 0
        and before["components"].get("competence", {}).get("updates") == 0
        and before["components"].get("exogenous", {}).get("finalized_episodes") == 0
        and before["components"].get("perception", {}).get("tracks") == 0
    )
    runtime_boundary = {
        "supported": True,
        "status": "supported_constructor_and_runtime_boundary",
        "pass": bool(
            agent.runtime_mode is RuntimeMode.AUTONOMOUS
            and agent.teacher is None
            and type(agent.arbiter) is RiskArbiter
            and isinstance(agent.representation_backend, GridFeatureBackend)
            and not agent.config.enable_executable_models
        ),
        "checkpoint_loaded": False,
        "route_used": False,
        "teacher_attached": agent.teacher is not None,
    }
    base_pass = bool(
        agent.runtime_mode is RuntimeMode.AUTONOMOUS
        and agent.teacher is None
        and type(agent.arbiter) is RiskArbiter
        and isinstance(agent.representation_backend, GridFeatureBackend)
        and not agent.config.enable_executable_models
        and no_forbidden_imports
        and runtime_boundary["pass"]
    )
    supported_not_applicable = {
        "supported": True,
        "value": None,
        "status": "not_applicable_static_source_and_runtime_boundary_verified",
    }
    counts = {
        "checkpoint_load_calls": dict(supported_not_applicable),
        "teacher_reads": dict(supported_not_applicable),
        "route_forcing_calls": dict(supported_not_applicable),
        "graph_specific_causal_interventions": dict(supported_not_applicable),
    }
    return {
        "arm": arm,
        "constructor_baseline_required": bool(constructor_baseline_required),
        "constructor_state_empty": empty_constructor_state,
        "persistent_state_retention_expected": bool(arm == ARM_PERSISTENT and not constructor_baseline_required),
        "persistent_state_retained": bool(arm == ARM_PERSISTENT and not empty_constructor_state),
        "runtime_mode": agent.runtime_mode.value,
        "teacher_attached": agent.teacher is not None,
        "arbiter_class": type(agent.arbiter).__name__,
        "representation_class": type(agent.representation_backend).__name__,
        "checkpoint_or_route_or_teacher_state": runtime_boundary,
        "forbidden_call_counts": counts,
        "forbidden_call_counter_status": "not_applicable_static_source_and_runtime_boundary_supported",
        "forbidden_call_instrumentation": {
            "supported": True,
            "status": "not_applicable_static_source_and_runtime_boundary_supported",
            "pass": True,
        },
        "static_source_verification": static_audit,
        "runtime_boundary_verification": runtime_boundary,
        "no_forbidden_imports_in_authoritative_runner": no_forbidden_imports,
        "fresh_constructor_state": {key: before["components"][key] for key in ("graph", "evidence", "replay")},
        "after_state": {key: after["components"][key] for key in after["components"]},
        "fresh_isolation_pass": bool(
            base_pass
            and static_audit["supported"]
            and static_audit["pass"]
            and runtime_boundary["supported"]
            and runtime_boundary["pass"]
            and (not constructor_baseline_required or empty_constructor_state)
        ),
    }


def _static_forbidden_path_audit() -> dict[str, Any]:
    """Prove the runner has no forbidden import or execution path.

    Dynamic call counters are deliberately not fabricated.  This runner has a
    supported static/source boundary and a supported constructor/runtime
    boundary; both must pass before the isolation result can be causal.
    """

    source_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden_import_tokens = {
        "load_checkpoint",
        "save_checkpoint",
        "Teacher",
        "ExplicitRouteArbiter",
        "run_hs_v2_fresh_discovery_v1",
        "run_hs_v2_tu93_graph_goal_v1",
    }
    forbidden_call_tokens = {
        "load_checkpoint",
        "save_checkpoint",
        "force_route",
        "route_forcing",
        "graph_goal_intervention",
    }
    imported_names = {
        alias.name.rsplit(".", 1)[-1]
        for node in ast.walk(source_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    call_names = {
        node.func.id
        for node in ast.walk(source_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    import_hits = sorted(imported_names.intersection(forbidden_import_tokens))
    call_hits = sorted(call_names.intersection(forbidden_call_tokens))
    return {
        "supported": True,
        "status": "supported_static_source_audit",
        "pass": not import_hits and not call_hits,
        "forbidden_import_hits": import_hits,
        "forbidden_call_hits": call_hits,
    }


def _decision_payload(decision: Decision) -> dict[str, Any]:
    return _jsonable(decision)


def _transition_payload(transition: Transition) -> dict[str, Any]:
    outcome_payload = _jsonable(transition.outcome)
    outcome_payload["completed"] = bool(transition.outcome.completed)
    outcome_payload["failed"] = bool(transition.outcome.failed)
    return {
        "transition_id": transition.transition_id,
        "decision_id": transition.decision_id,
        "task_id": transition.task_id,
        "stage": transition.stage,
        "step": transition.step,
        "before_state_id": transition.before.state_id,
        "after_state_id": transition.after_state_id,
        "after_stage": transition.after_observation.stage,
        "before_observation_fingerprint": _observation_fingerprint(transition.before.observation),
        "after_observation_fingerprint": _observation_fingerprint(transition.after_observation),
        "action": list(transition.action.key),
        "frame_changed": transition.frame_changed,
        "outcome": outcome_payload,
        "metadata": _jsonable(transition.after_observation.metadata),
    }


def _seed_schedule(pair_index: int, attempt_index: int, agent_seed: int) -> dict[str, Any]:
    prefix = f"{EXPERIMENT_ID}|replicate={pair_index}|attempt={attempt_index}|agent={agent_seed}"
    def derive(label: str) -> int:
        return int(hashlib.sha256(f"{prefix}|{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    order_hash = hashlib.sha256(f"{prefix}|arm-order".encode("utf-8")).hexdigest()
    return {
        "agent_seed": int(agent_seed),
        "environment_seed": derive("paired-environment"),
        "python_seed": derive("python-and-numpy"),
        "numpy_seed": derive("numpy"),
        "arm_order_hash_sha256": order_hash,
        "arm_order": [ARM_PERSISTENT, ARM_ISOLATED] if int(order_hash[:2], 16) % 2 == 0 else [ARM_ISOLATED, ARM_PERSISTENT],
    }


def paired_arm_order(pair_index: int, attempt_index: int, agent_seed: int | None = None) -> tuple[str, str]:
    schedule = _seed_schedule(pair_index, attempt_index, pair_index if agent_seed is None else agent_seed)
    return tuple(schedule["arm_order"])


def _run_attempt(
    agent: CompactHunterSeeker,
    environment: Any,
    *,
    pair_id: str,
    attempt_id: str,
    arm: str,
    agent_seed: int,
    action_limit: int,
    action_adapter: Any,
    observation_adapter: Any,
    outcome_adapter: Any,
    action_space: Any = None,
    bootstrap: bool = True,
    constructor_baseline_required: bool = True,
    attempt_progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one real attempt through act -> environment -> observe commits."""

    if action_limit < 1:
        raise ValueError("action_limit must be positive")
    capture = attempt_progress if attempt_progress is not None else {}
    capture.update(
        {
            "schema_version": SCHEMA_VERSION,
            "pair_id": pair_id,
            "attempt_id": attempt_id,
            "arm": arm,
            "agent_seed": int(agent_seed),
            "status": "blocked_by_runtime_exception",
            "environment_steps": [],
            "executed_action_evidence": None,
            "bootstrap_environment_step": {"attempted": False, "returned": False},
        }
    )
    task_id = EXACT_GAME_ID if arm in ARMS else pair_id
    decisions: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    completion_lineage: list[dict[str, Any]] = []
    final_outcome = Outcome(truncated=True, boundary=BoundaryKind.TIME_LIMIT, metadata={"max_actions": action_limit})
    capture["decisions"] = decisions
    capture["committed_transitions"] = transitions
    stopped_reason = "action_limit"
    invalid = False
    aborted = False
    error: dict[str, Any] | None = None
    executed_action_evidence: dict[str, Any] | None = None
    run_started = False
    try:
        observation_adapter.clear(task_id)
        if bootstrap:
            bootstrap_action, bootstrap_kwargs = action_adapter.bootstrap(action_space)
            capture["bootstrap_environment_step"]["attempted"] = True
            try:
                initial_raw = environment.step(bootstrap_action, **dict(bootstrap_kwargs))
            except BaseException as exc:
                capture["bootstrap_environment_step"]["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                raise
            capture["bootstrap_environment_step"]["returned"] = True
        else:
            initial_raw = environment.reset()
        initial = observation_adapter.observation(initial_raw, task_id=task_id)
        before_state = state_store_fingerprint(agent)
        initial_fingerprint = _observation_fingerprint(initial)
        observation = initial
        capture["initial_observation_fingerprint"] = initial_fingerprint
        capture["initial_state_fingerprint"] = _digest({"state_id": initial.state_id, "observation_fingerprint": initial_fingerprint})
        capture["initial_agent_store_fingerprint"] = before_state["overall_fingerprint"]
        capture["initial_state_id"] = initial.state_id
        capture["initial_stage"] = initial.stage
        agent.begin_run(task_id, observation)
        run_started = True
    except BaseException as exc:
        capture["error"] = {"type": type(exc).__name__, "message": str(exc)}
        capture["status"] = "blocked_by_runtime_exception"
        close = getattr(environment, "close", None)
        if callable(close):
            close()
        raise
    try:
        if observation.stage != 1:
            invalid = True
            stopped_reason = "invalid_initial_stage"
        for step_index in range(action_limit):
            if invalid:
                break
            if observation.stage != 1:
                invalid = True
                stopped_reason = "stage_2_action_would_be_attempted"
                break
            decision = agent.act(observation)
            decisions.append({"step": step_index, "full": _decision_payload(decision), "metadata": _jsonable(decision.metadata)})
            env_action, kwargs = action_adapter.decode(decision.action, action_space)
            executed_action_evidence = {
                "step": step_index,
                "decision_id": decision.decision_id,
                "action": list(decision.action.key),
                "environment_step_attempted": True,
                "environment_step_returned": False,
                "environment_step_completed": False,
            }
            capture["executed_action_evidence"] = executed_action_evidence
            step_evidence = {
                **executed_action_evidence,
                "environment_step_attempted": True,
                "environment_step_returned": False,
            }
            capture["environment_steps"].append(step_evidence)
            raw_after = environment.step(env_action, **dict(kwargs))
            executed_action_evidence["environment_step_returned"] = True
            executed_action_evidence["environment_step_completed"] = True
            step_evidence["environment_step_returned"] = True
            step_evidence["environment_step_completed"] = True
            next_observation = observation_adapter.observation(raw_after, task_id=task_id)
            outcome = outcome_adapter.outcome(observation, next_observation, raw_after=raw_after, info={})
            if step_index + 1 == action_limit and not outcome.completed and not outcome.terminated and not outcome.truncated:
                outcome = dataclasses.replace(outcome, truncated=True, boundary=BoundaryKind.TIME_LIMIT, metadata={**dict(outcome.metadata), "max_actions": action_limit})
            transition = agent.observe(decision, next_observation, outcome)
            transition_payload = _transition_payload(transition)
            transition_payload.update(
                {
                    "environment_step_attempted": True,
                    "environment_step_returned": True,
                }
            )
            transitions.append(transition_payload)
            observation = next_observation
            capture["final_stage"] = observation.stage
            final_outcome = transition.outcome
            if final_outcome.completed:
                if transition.stage != 1:
                    invalid = True
                    stopped_reason = "completion_not_stage_1"
                    break
                agent.on_level_complete(1)
                completion_lineage.append({"transition_id": transition.transition_id, "decision_id": transition.decision_id, "completed_stage": 1, "before_stage": transition.stage, "after_stage": transition.after_observation.stage, "agent_id": decision.agent_id})
                stopped_reason = "first_stage_1_completion"
                break
            if final_outcome.terminated or final_outcome.truncated:
                stopped_reason = final_outcome.boundary.value
                break
            if next_observation.stage != 1:
                invalid = True
                stopped_reason = "stage_2_observed_without_completion"
                break
    except BaseException as exc:
        capture["error"] = {"type": type(exc).__name__, "message": str(exc)}
        capture["status"] = "blocked_by_runtime_exception"
        pending = agent.pending_decision
        step_was_attempted = bool(
            executed_action_evidence
            and executed_action_evidence.get("environment_step_attempted")
        )
        if pending is not None and not step_was_attempted:
            agent.cancel_decision(
                pending,
                reason="attempt_aborted_before_environment_step",
            )
        raise
    finally:
        # Once environment.step has been attempted, the pending decision is
        # evidence and must remain visible to the top-level blocked artifact.
        # end_run would reject that state, so only finalize a clean attempt.
        if run_started and agent.pending_decision is None:
            agent.end_run(final_outcome)
        close = getattr(environment, "close", None)
        if callable(close):
            close()
    after_state = state_store_fingerprint(agent)
    capture.update(
        {
            "initial_observation_fingerprint": initial_fingerprint,
            "initial_state_fingerprint": _digest({"state_id": initial.state_id, "observation_fingerprint": initial_fingerprint}),
            "initial_agent_store_fingerprint": before_state["overall_fingerprint"],
            "initial_state_id": initial.state_id,
            "initial_stage": initial.stage,
            "final_stage": observation.stage,
            "committed_transitions": transitions,
            "decisions": decisions,
            "state_store_before": before_state,
            "state_store_after": after_state,
            "measurement_summary": _jsonable(agent.measurement_summary()),
            "runtime_counts": after_state["components"],
        }
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair_id,
        "attempt_id": attempt_id,
        "arm": arm,
        "agent_seed": int(agent_seed),
        "status": "invalid" if invalid else "valid",
        "stopped_reason": stopped_reason,
        "initial_observation_fingerprint": initial_fingerprint,
        "initial_state_fingerprint": _digest({"state_id": initial.state_id, "observation_fingerprint": initial_fingerprint}),
        "initial_agent_store_fingerprint": before_state["overall_fingerprint"],
        "initial_state_id": initial.state_id,
        "initial_stage": initial.stage,
        "final_stage": observation.stage,
        "final_boundary": final_outcome.boundary.value,
        "completed": bool(final_outcome.completed and not invalid),
        "completion_count": sum(int(row["outcome"]["completed"]) for row in transitions),
        "committed_action_count": len(transitions),
        "decisions": decisions,
        "committed_transitions": transitions,
        "completion_lineage": completion_lineage,
        "state_store_before": before_state,
        "state_store_after": after_state,
        "measurement_summary": _jsonable(agent.measurement_summary()),
        "runtime_counts": after_state["components"],
        "executed_action_evidence": executed_action_evidence,
        "bootstrap_environment_step": capture.get("bootstrap_environment_step"),
        "environment_steps": capture["environment_steps"],
        "leakage_audit": runtime_leakage_audit(
            agent,
            before=before_state,
            after=after_state,
            arm=arm,
            constructor_baseline_required=constructor_baseline_required,
        ),
        "error": error,
    }
    capture.update(result)
    return result


def _exact_mcnemar_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    live_run_token: object | None = None,
) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid_pair")]
    persistent_only = sum(int(row["persistent_completed"] and not row["isolated_completed"]) for row in valid)
    isolated_only = sum(int(row["isolated_completed"] and not row["persistent_completed"]) for row in valid)
    discordant = persistent_only + isolated_only
    tail = sum(comb(discordant, k) for k in range(persistent_only, discordant + 1)) / (2**discordant) if discordant else 1.0
    return {
        "name": "one-sided_exact_mcnemar",
        "alternative": "persistent_fresh > episode_isolated",
        "alpha": MCMEMAR_ALPHA,
        "valid_replicates": len(valid),
        "persistent_only": persistent_only,
        "isolated_only": isolated_only,
        "discordant_pairs": discordant,
        "p_value": tail,
        "pass": bool(valid and tail <= MCMEMAR_ALPHA),
        "causal_eligible": live_run_token in _LIVE_RUN_TOKENS,
        "preregistered": True,
    }


def _aggregate_from_raw_pairs(
    pairs: Sequence[Mapping[str, Any]],
    checks: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Derive completion aggregates only from raw transition outcomes."""

    validity = list((checks or {}).get("pair_validity", ()))
    aggregates = []
    for index, pair in enumerate(pairs):
        attempts = pair.get("attempts", ()) if isinstance(pair, Mapping) else ()
        persistent_completed = any(
            _raw_completion(attempt.get("arms", {}).get(ARM_PERSISTENT, {}))
            for attempt in attempts
            if isinstance(attempt, Mapping) and isinstance(attempt.get("arms"), Mapping)
        )
        isolated_completed = any(
            _raw_completion(attempt.get("arms", {}).get(ARM_ISOLATED, {}))
            for attempt in attempts
            if isinstance(attempt, Mapping) and isinstance(attempt.get("arms"), Mapping)
        )
        aggregates.append(
            {
                "pair_id": pair.get("pair_id") if isinstance(pair, Mapping) else None,
                "valid_pair": bool(validity[index]) if index < len(validity) else False,
                "persistent_completed": persistent_completed,
                "isolated_completed": isolated_completed,
            }
        )
    return aggregates


def _provenance_manifest(script_path: Path) -> dict[str, Any]:
    source_files = sorted((SRC_ROOT / "hunter_seeker_v2").glob("*.py")) + [script_path]
    if script_path.resolve() != Path(__file__).resolve():
        raise ExperimentContractError("runner manifest must identify this authoritative runner")
    if not source_files or any(not path.is_file() for path in (*source_files, *CONFIG_PATHS)):
        raise ExperimentContractError("source or configuration manifest input is missing")
    return {
        "schema_version": SCHEMA_VERSION,
        "authoritative": True,
        "exact_environment_verification": True,
        "source_manifest": _file_manifest(source_files),
        "config_manifest": _file_manifest(CONFIG_PATHS),
        "package_manifest": _package_manifest(),
        "runner_manifest": _file_manifest((Path(__file__).resolve(),)),
        "schema_manifest": _schema_manifest(),
        "environment_manifest": verify_environment_contract(),
    }


def _best_effort_provenance_manifest(error: BaseException | None = None) -> dict[str, Any]:
    """Return a schema-valid manifest even when exact verification is blocked."""

    try:
        return _provenance_manifest(Path(__file__).resolve())
    except BaseException as manifest_error:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked_exact_environment_manifest_unavailable",
            "authoritative": True,
            "error": {
                "type": type(manifest_error).__name__,
                "message": str(manifest_error),
            },
            "source_manifest": {"status": "unavailable"},
            "config_manifest": {"status": "unavailable"},
            "package_manifest": {"status": "unavailable"},
            "runner_manifest": {"status": "unavailable"},
            "schema_manifest": {"status": "unavailable"},
            "environment_manifest": {
                "game_id": EXACT_GAME_ID,
                "variant": ENVIRONMENT_VERSION,
                "selection": "local_offline_exact_version",
                "source": {"expected_sha256": EXPECTED_SOURCE_SHA256, "verified": False},
                "metadata": {"expected_sha256": EXPECTED_METADATA_SHA256, "verified": False},
                "verified": False,
            },
            "exact_environment_verification": False,
            "requested_exact_environment": {
                "game_id": EXACT_GAME_ID,
                "source_sha256": EXPECTED_SOURCE_SHA256,
                "metadata_sha256": EXPECTED_METADATA_SHA256,
            },
            "blocking_error": None
            if error is None
            else {"type": type(error).__name__, "message": str(error)},
        }


def _blocked_summary(
    *,
    pair_count: int,
    attempt_limit: int,
    action_limit: int,
    error: BaseException,
    partial_pairs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    design = _requested_design(
        pair_count=pair_count,
        attempt_limit=attempt_limit,
        action_limit=action_limit,
    )
    checks = _structural_checks(partial_pairs)
    aggregate_rows = _aggregate_from_raw_pairs(partial_pairs, checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "authoritative": True,
        "blocked": True,
        "failure_classification": "blocked_runtime_or_contract_error",
        "error": {"type": type(error).__name__, "message": str(error)},
        "requested_design": design,
        "design": design,
        "provenance_manifest": _best_effort_provenance_manifest(error),
        "finalization_marker": {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "complete": False,
        },
        "pairs": list(partial_pairs),
        "aggregate_paired_result": {
            "replicates": aggregate_rows,
            "descriptive_only_unless_gate_passes": True,
        },
        "checks": {**checks, "blocked": True},
        "verdict": {
            "code": "BLOCKED_RUNTIME_ERROR",
            "causal_claim": False,
            "incomplete_design": True,
            "claim_scope": "no causal claim; run blocked before contract completion",
        },
    }


def _requested_design(*, pair_count: int, attempt_limit: int, action_limit: int) -> dict[str, Any]:
    return {
        "arms": ARMS,
        "replicates_requested": int(pair_count),
        "replicate_agent_seeds": list(range(int(pair_count))),
        "attempt_limit": int(attempt_limit),
        "action_limit": int(action_limit),
        "online_learning": True,
        "stop_rule": "first stage-1 completion",
        "primary_trace_stage_rule": "no action with before-stage != 1",
        "environment_seed_rule": "paired deterministic seed per replicate/attempt",
        "order_rule": "alternating arm order from SHA-256 preregistered hash",
        "exact_contract_design": bool(
            pair_count == REPLICATE_COUNT
            and attempt_limit == ATTEMPT_LIMIT
            and action_limit == ACTION_LIMIT
        ),
        "smoke": not (
            pair_count == REPLICATE_COUNT
            and attempt_limit == ATTEMPT_LIMIT
            and action_limit == ACTION_LIMIT
        ),
    }


def _stage_fields_pass(row: Mapping[str, Any]) -> bool:
    if type(row.get("initial_stage")) is not int or type(row.get("final_stage")) is not int:
        return False
    if row["initial_stage"] != 1 or row["final_stage"] < 1:
        return False
    transitions = row.get("committed_transitions")
    if not isinstance(transitions, list):
        return False
    for transition in transitions:
        if not isinstance(transition, Mapping):
            return False
        if type(transition.get("stage")) is not int or type(transition.get("after_stage")) is not int:
            return False
        if transition["stage"] < 1 or transition["after_stage"] < 1:
            return False
    return True


def _no_stage_2_primary_actions(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if row.get("status") != "valid" or not _stage_fields_pass(row):
            return False
        if any(transition.get("stage") != 1 for transition in row["committed_transitions"]):
            return False
    return True


def _pair_attempt_match(rows: Mapping[str, Mapping[str, Any]]) -> bool:
    if set(rows) != set(ARMS):
        return False
    if any(rows[arm].get("status") != "valid" for arm in ARMS):
        return False
    observation_fingerprints = [rows[arm].get("initial_observation_fingerprint") for arm in ARMS]
    state_ids = [rows[arm].get("initial_state_id") for arm in ARMS]
    return bool(
        all(isinstance(value, str) and value for value in observation_fingerprints)
        and all(isinstance(value, str) and value for value in state_ids)
        and len(set(observation_fingerprints)) == 1
        and len(set(state_ids)) == 1
    )


def _blocked_attempt_record(
    *,
    pair_id: str,
    attempt_id: str,
    arm: str,
    agent_seed: int,
    capture: Mapping[str, Any] | None,
    error: BaseException,
) -> dict[str, Any]:
    """Keep partial action evidence in a schema-shaped blocked row."""

    captured = dict(capture or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair_id,
        "attempt_id": attempt_id,
        "arm": arm,
        "agent_seed": int(agent_seed),
        "status": "blocked_by_runtime_exception",
        "stopped_reason": "runtime_exception",
        "initial_observation_fingerprint": captured.get("initial_observation_fingerprint"),
        "bootstrap_environment_step": captured.get("bootstrap_environment_step"),
        "initial_state_fingerprint": captured.get("initial_state_fingerprint"),
        "initial_agent_store_fingerprint": captured.get("initial_agent_store_fingerprint"),
        "initial_state_id": captured.get("initial_state_id"),
        "initial_stage": captured.get("initial_stage"),
        "final_stage": captured.get("final_stage"),
        "final_boundary": BoundaryKind.ENVIRONMENT_ERROR.value,
        "completed": False,
        "completion_count": 0,
        "committed_action_count": len(captured.get("committed_transitions", ())),
        "decisions": list(captured.get("decisions", ())),
        "committed_transitions": list(captured.get("committed_transitions", ())),
        "completion_lineage": [],
        "state_store_before": captured.get("state_store_before"),
        "state_store_after": captured.get("state_store_after"),
        "measurement_summary": captured.get("measurement_summary"),
        "runtime_counts": captured.get("runtime_counts"),
        "executed_action_evidence": captured.get("executed_action_evidence"),
        "environment_steps": list(captured.get("environment_steps", ())),
        "leakage_audit": {"fresh_isolation_pass": False, "blocked": True},
        "error": {"type": type(error).__name__, "message": str(error)},
        "finalized": False,
    }


def _pair_attempt_record(
    pair_id: str,
    attempt_index: int,
    schedule: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    attempt_id: str | None = None,
) -> dict[str, Any]:
    pair_match = _pair_attempt_match(rows)
    skipped_after_completion = bool(
        rows
        and any(row.get("status") == "skipped_after_first_completion" for row in rows.values())
    )
    valid_pair = bool(
        all(row.get("status") in {"valid", "skipped_after_first_completion"} for row in rows.values())
        and (skipped_after_completion or pair_match)
    )
    return {
        "pair_id": pair_id,
        "attempt_id": attempt_id or next((row.get("attempt_id") for row in rows.values() if row.get("attempt_id")), ""),
        "attempt_index": int(attempt_index),
        "schedule": dict(schedule),
        "arm_order": list(schedule["arm_order"]),
        "initial_observation_fingerprints": {arm: rows.get(arm, {}).get("initial_observation_fingerprint") for arm in ARMS},
        "initial_state_ids": {arm: rows.get(arm, {}).get("initial_state_id") for arm in ARMS},
        "initial_state_fingerprints": {arm: rows.get(arm, {}).get("initial_state_fingerprint") for arm in ARMS},
        "initial_agent_store_fingerprints": {arm: rows.get(arm, {}).get("initial_agent_store_fingerprint") for arm in ARMS},
        "matching_initial_observation_state": pair_match,
        "skipped_after_completion": skipped_after_completion,
        "finalized": valid_pair,
        "valid_pair": valid_pair,
        "arms": dict(rows),
    }


def _raw_completion(row: Mapping[str, Any]) -> bool:
    transitions = row.get("committed_transitions")
    return bool(
        isinstance(transitions, list)
        and any(
            isinstance(transition, Mapping)
            and isinstance(transition.get("outcome"), Mapping)
            and transition["outcome"].get("completed") is True
            for transition in transitions
        )
    )


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _state_store_evidence_pass(row: Mapping[str, Any]) -> bool:
    """Require a complete before/after store capture, not summary counters."""

    for name in ("state_store_before", "state_store_after"):
        store = row.get(name)
        if not isinstance(store, Mapping) or not _nonempty_text(store.get("overall_fingerprint")):
            return False
        components = store.get("components")
        if not isinstance(components, Mapping) or set(components) != set(REQUIRED_STATE_COMPONENTS):
            return False
        if _digest(components) != store.get("overall_fingerprint"):
            return False
        for component_name in REQUIRED_STATE_COMPONENTS:
            component = components.get(component_name)
            if not isinstance(component, Mapping) or not _nonempty_text(component.get("fingerprint")):
                return False
    runtime_counts = row.get("runtime_counts")
    return isinstance(runtime_counts, Mapping) and set(runtime_counts) == set(REQUIRED_STATE_COMPONENTS)


def _bootstrap_evidence_pass(row: Mapping[str, Any]) -> bool:
    bootstrap = row.get("bootstrap_environment_step")
    return bool(
        isinstance(bootstrap, Mapping)
        and bootstrap.get("attempted") is True
        and bootstrap.get("returned") is True
    )


def _action_evidence_pass(row: Mapping[str, Any]) -> bool:
    evidence = row.get("executed_action_evidence")
    return bool(
        isinstance(evidence, Mapping)
        and type(evidence.get("step")) is int
        and _nonempty_text(evidence.get("decision_id"))
        and evidence.get("environment_step_attempted") is True
        and evidence.get("environment_step_returned") is True
        and evidence.get("environment_step_completed") is True
    )


def _row_identity_evidence_pass(row: Mapping[str, Any]) -> bool:
    if not all(
        _nonempty_text(row.get(name))
        for name in ("initial_observation_fingerprint", "initial_state_id", "initial_state_fingerprint", "initial_agent_store_fingerprint")
    ):
        return False
    if row.get("initial_state_fingerprint") != _digest(
        {
            "state_id": row.get("initial_state_id"),
            "observation_fingerprint": row.get("initial_observation_fingerprint"),
        }
    ):
        return False
    before = row.get("state_store_before")
    return isinstance(before, Mapping) and before.get("overall_fingerprint") == row.get("initial_agent_store_fingerprint")


def _transition_trace_pass(
    row: Mapping[str, Any],
    seen_decision_ids: set[str],
    seen_transition_ids: set[str],
) -> bool:
    decisions = row.get("decisions")
    transitions = row.get("committed_transitions")
    environment_steps = row.get("environment_steps")
    if not isinstance(decisions, list) or not isinstance(transitions, list) or not isinstance(environment_steps, list):
        return False
    if type(row.get("committed_action_count")) is not int or type(row.get("completion_count")) is not int:
        return False
    if row.get("committed_action_count") != len(decisions) or len(decisions) != len(transitions) or len(transitions) != len(environment_steps):
        return False
    if not decisions or not transitions or not environment_steps:
        return False
    if not _action_evidence_pass(row):
        return False
    prior_state_id = row.get("initial_state_id")
    prior_observation_fingerprint = row.get("initial_observation_fingerprint")
    for index, (decision, transition, step) in enumerate(zip(decisions, transitions, environment_steps, strict=True)):
        if not isinstance(decision, Mapping) or not isinstance(transition, Mapping) or not isinstance(step, Mapping):
            return False
        full = decision.get("full")
        if not isinstance(full, Mapping):
            return False
        decision_id = full.get("decision_id")
        transition_id = transition.get("transition_id")
        if not isinstance(decision_id, str) or not decision_id or decision_id in seen_decision_ids:
            return False
        if not isinstance(transition_id, str) or not transition_id or transition_id in seen_transition_ids:
            return False
        seen_decision_ids.add(decision_id)
        seen_transition_ids.add(transition_id)
        if decision.get("step") != index or full.get("step") != index:
            return False
        if transition.get("step") != index or transition.get("decision_id") != decision_id:
            return False
        if transition.get("before_state_id") != prior_state_id or transition.get("before_observation_fingerprint") != prior_observation_fingerprint:
            return False
        if not _nonempty_text(transition.get("after_state_id")) or not _nonempty_text(transition.get("after_observation_fingerprint")):
            return False
        if transition.get("action") != [full.get("action", {}).get("index"), full.get("action", {}).get("x"), full.get("action", {}).get("y")]:
            return False
        if transition.get("environment_step_attempted") is not True or transition.get("environment_step_returned") is not True:
            return False
        if step.get("step") != index or step.get("decision_id") != decision_id:
            return False
        if step.get("environment_step_attempted") is not True or step.get("environment_step_returned") is not True:
            return False
        outcome = transition.get("outcome")
        if not isinstance(outcome, Mapping):
            return False
        if any(type(outcome.get(name)) is not bool for name in ("completed", "failed", "terminated", "truncated")):
            return False
        if outcome["completed"] and outcome["failed"]:
            return False
        prior_state_id = transition.get("after_state_id")
        prior_observation_fingerprint = transition.get("after_observation_fingerprint")
    return True


def _structural_checks(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute every causal structural condition from raw recorded rows."""

    pair_ids = [pair.get("pair_id") for pair in pairs if isinstance(pair, Mapping)]
    pair_seeds = [pair.get("agent_seed") for pair in pairs if isinstance(pair, Mapping)]
    pair_identity_pass = bool(
        len(pairs) == REPLICATE_COUNT
        and len(pair_ids) == len(pairs)
        and tuple(pair_ids) == EXPECTED_PAIR_IDS
        and all(type(value) is int for value in pair_seeds)
        and pair_seeds == list(range(REPLICATE_COUNT))
    )
    attempts = [
        attempt
        for pair in pairs
        if isinstance(pair, Mapping)
        for attempt in pair.get("attempts", ())
        if isinstance(attempt, Mapping)
    ]
    attempt_records_pass = bool(pairs)
    attempt_identity_pass = True
    environment_seeds: list[int] = []
    attempt_ids: set[str] = set()
    seen_decision_ids: set[str] = set()
    seen_transition_ids: set[str] = set()
    row_status_pass = True
    pair_matching_pass = True
    leakage_pass = True
    stage_fields_pass = True
    no_stage_2_primary_actions = True
    environment_step_trace_pass = True
    completion_consistency_pass = True
    finalization_pass = True
    pair_validity: list[bool] = []
    completed_arms: dict[str, set[str]] = {
        str(pair.get("pair_id")) if isinstance(pair, Mapping) else "<malformed>": set()
        for pair in pairs
    }
    for pair_index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            pair_validity.append(False)
            attempt_records_pass = False
            pair_identity_pass = False
            finalization_pass = False
            continue
        pair_id = pair.get("pair_id")
        seen_attempt_indices: set[int] = set()
        pair_valid = isinstance(pair, Mapping)
        attempts_for_pair = pair.get("attempts") if isinstance(pair, Mapping) else None
        if not isinstance(attempts_for_pair, list):
            attempt_records_pass = False
            pair_valid = False
            attempts_for_pair = []
        pair_stopped = pair.get("stopped_after_first_completion") is True
        pair_finalized = pair.get("valid_finalization") is True and pair.get("finalized") is True
        if len(attempts_for_pair) != ATTEMPT_LIMIT or not pair_finalized or pair_stopped:
            attempt_records_pass = False
            pair_valid = False
        for attempt in attempts_for_pair:
            if not isinstance(attempt, Mapping):
                attempt_identity_pass = False
                pair_valid = False
                continue
            attempt_index = attempt.get("attempt_index")
            if type(attempt_index) is not int or attempt_index not in range(ATTEMPT_LIMIT) or attempt_index in seen_attempt_indices:
                attempt_identity_pass = False
                pair_valid = False
            seen_attempt_indices.add(attempt_index)
            attempt_id = attempt.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id or attempt_id in attempt_ids:
                attempt_identity_pass = False
                pair_valid = False
            else:
                attempt_ids.add(attempt_id)
            if attempt.get("pair_id") != pair_id or set(attempt.get("arms", {})) != set(ARMS):
                attempt_identity_pass = False
                pair_valid = False
            schedules = attempt.get("schedule", {})
            expected_schedule = None
            if type(pair.get("agent_seed")) is int and type(attempt_index) is int:
                expected_schedule = _seed_schedule(pair_index, attempt_index, pair.get("agent_seed"))
            if isinstance(expected_schedule, Mapping) and attempt.get("arm_order") != expected_schedule.get("arm_order"):
                attempt_identity_pass = False
                pair_valid = False
            if not isinstance(schedules, Mapping) or expected_schedule is None or any(schedules.get(key) != value for key, value in expected_schedule.items()):
                attempt_identity_pass = False
                pair_valid = False
            else:
                environment_seeds.append(int(schedules["environment_seed"]))
            rows = attempt.get("arms", {})
            for arm in ARMS:
                row = rows.get(arm)
                if not isinstance(row, Mapping):
                    row_status_pass = False
                    continue
                status = row.get("status")
                if row.get("pair_id") != pair_id or row.get("attempt_id") != attempt_id or row.get("arm") != arm or row.get("agent_seed") != pair.get("agent_seed"):
                    row_status_pass = False
                if status == "skipped_after_first_completion":
                    valid_skip = bool(
                        row.get("completed") is True
                        and row.get("aborted") is not True
                        and row.get("pair_id") == pair_id
                        and row.get("attempt_id") == attempt_id
                        and row.get("arm") == arm
                        and row.get("agent_seed") == pair.get("agent_seed")
                        and row.get("committed_action_count") == 0
                        and row.get("decisions", []) == []
                        and row.get("committed_transitions", []) == []
                        and row.get("environment_steps", []) == []
                        and row.get("stopped_after_first_completion") is True
                        and row.get("valid_finalization") is True
                        and arm in completed_arms.get(str(pair_id), set())
                    )
                    row_status_pass = row_status_pass and valid_skip
                    pair_valid = pair_valid and valid_skip
                    continue
                if status != "valid":
                    row_status_pass = False
                    pair_valid = False
                    continue
                row_valid = bool(
                    not row.get("error")
                    and row.get("aborted") is not True
                    and _bootstrap_evidence_pass(row)
                    and _row_identity_evidence_pass(row)
                    and _state_store_evidence_pass(row)
                    and _stage_fields_pass(row)
                    and _transition_trace_pass(row, seen_decision_ids, seen_transition_ids)
                )
                if not row_valid:
                    row_status_pass = False
                    stage_fields_pass = False
                    pair_valid = False
                    if not _stage_fields_pass(row):
                        stage_fields_pass = False
                raw_completed = _raw_completion(row)
                if row.get("completed") is not raw_completed:
                    completion_consistency_pass = False
                    pair_valid = False
                if raw_completed:
                    completed_arms.setdefault(str(pair_id), set()).add(arm)
                elif arm in completed_arms.get(str(pair_id), set()):
                    row_status_pass = False
                    pair_valid = False
                transitions = row.get("committed_transitions", ())
                if any(transition.get("stage") != 1 for transition in transitions if isinstance(transition, Mapping)):
                    no_stage_2_primary_actions = False
                for decision in row.get("decisions", ()):
                    metadata = decision.get("full", {}).get("metadata", {}) if isinstance(decision, Mapping) else {}
                    if isinstance(metadata, Mapping) and metadata.get("stage") not in (None, 1):
                        no_stage_2_primary_actions = False
                if not isinstance(row.get("environment_steps"), list):
                    environment_step_trace_pass = False
                    pair_valid = False
                else:
                    for step in row["environment_steps"]:
                        if not isinstance(step, Mapping) or step.get("environment_step_attempted") is not True or step.get("environment_step_returned") is not True:
                            environment_step_trace_pass = False
                            pair_valid = False
                completion_rows = [transition for transition in transitions if isinstance(transition, Mapping) and isinstance(transition.get("outcome"), Mapping) and transition["outcome"].get("completed") is True]
                lineage = row.get("completion_lineage")
                if row.get("completion_count") != len(completion_rows) or not isinstance(lineage, list) or len(lineage) != len(completion_rows) or len(completion_rows) > 1:
                    completion_consistency_pass = False
                    pair_valid = False
                elif completion_rows:
                    last = completion_rows[0]
                    lineage_row = lineage[0]
                    if not isinstance(lineage_row, Mapping) or lineage_row.get("transition_id") != last.get("transition_id") or lineage_row.get("decision_id") != last.get("decision_id") or lineage_row.get("completed_stage") != 1 or lineage_row.get("before_stage") != last.get("stage") or lineage_row.get("after_stage") != last.get("after_stage") or row.get("stopped_reason") != "first_stage_1_completion":
                        completion_consistency_pass = False
                        pair_valid = False
                elif lineage or row.get("stopped_reason") == "first_stage_1_completion":
                    completion_consistency_pass = False
                    pair_valid = False
                if transitions:
                    if row.get("final_stage") != transitions[-1].get("after_stage") or row.get("final_boundary") != transitions[-1].get("outcome", {}).get("boundary"):
                        completion_consistency_pass = False
                        pair_valid = False
                elif row.get("final_stage") != row.get("initial_stage"):
                    completion_consistency_pass = False
                    pair_valid = False
            if attempt.get("skipped_after_completion"):
                if not any(row.get("status") == "skipped_after_first_completion" for row in rows.values() if isinstance(row, Mapping)):
                    pair_matching_pass = False
                    pair_valid = False
            elif not _pair_attempt_match(rows):
                pair_matching_pass = False
                pair_valid = False
            if attempt.get("finalized") is not True or attempt.get("valid_pair") is not True:
                finalization_pass = False
                pair_valid = False
        if seen_attempt_indices and seen_attempt_indices != set(range(len(seen_attempt_indices))):
            attempt_identity_pass = False
            pair_valid = False
        raw_pair_completion = {
            arm: arm in completed_arms.get(str(pair_id), set())
            for arm in ARMS
        }
        if len(attempts_for_pair) != ATTEMPT_LIMIT:
            attempt_records_pass = False
            pair_valid = False
        if pair.get("persistent_completed") is not raw_pair_completion[ARM_PERSISTENT] or pair.get("isolated_completed") is not raw_pair_completion[ARM_ISOLATED]:
            completion_consistency_pass = False
            pair_valid = False
        if pair.get("valid_pair") is not True or pair.get("finalized") is not True or pair.get("valid_finalization") is not True:
            finalization_pass = False
            pair_valid = False
        if pair.get("valid_pair") is not bool(pair_valid):
            finalization_pass = False
            pair_valid = False
        pair_validity.append(bool(pair_valid))
    if len(environment_seeds) != len(set(environment_seeds)):
        attempt_identity_pass = False
    if len(attempt_ids) != len(attempts):
        attempt_identity_pass = False
    executed_rows = [
        row
        for attempt in attempts
        for row in attempt.get("arms", {}).values()
        if isinstance(row, Mapping) and row.get("status") == "valid"
    ]
    leakage_pass = bool(executed_rows) and all(
        row.get("leakage_audit", {}).get("fresh_isolation_pass", False)
        for row in executed_rows
    )
    stage_fields_pass = bool(executed_rows) and stage_fields_pass and all(_stage_fields_pass(row) for row in executed_rows)
    no_stage_2_primary_actions = bool(executed_rows) and no_stage_2_primary_actions and _no_stage_2_primary_actions(executed_rows)
    structural_pass = bool(
        pairs
        and pair_identity_pass
        and attempt_records_pass
        and attempt_identity_pass
        and row_status_pass
        and pair_matching_pass
        and leakage_pass
        and stage_fields_pass
        and no_stage_2_primary_actions
        and environment_step_trace_pass
        and completion_consistency_pass
        and finalization_pass
    )
    return {
        "pair_identity_pass": pair_identity_pass,
        "attempt_records_pass": attempt_records_pass,
        "attempt_identity_pass": attempt_identity_pass,
        "row_status_pass": row_status_pass,
        "pair_matching_pass": pair_matching_pass,
        "leakage_pass": leakage_pass,
        "stage_fields_pass": stage_fields_pass,
        "environment_step_trace_pass": environment_step_trace_pass,
        "no_stage_2_primary_actions": no_stage_2_primary_actions,
        "completion_consistency_pass": completion_consistency_pass,
        "finalization_pass": finalization_pass,
        "pair_validity": pair_validity,
        "structural_pass": structural_pass,
    }


def _exact_pinned_manifest(manifest: Mapping[str, Any] | None) -> bool:
    if not isinstance(manifest, Mapping):
        return False
    if manifest.get("authoritative") is not True or manifest.get("exact_environment_verification") is not True:
        return False
    environment = manifest.get("environment_manifest")
    if not isinstance(environment, Mapping) or environment.get("game_id") != EXACT_GAME_ID or environment.get("variant") != ENVIRONMENT_VERSION or environment.get("verified") is not True:
        return False
    source = environment.get("source", {})
    metadata = environment.get("metadata", {})
    if not isinstance(source, Mapping) or not isinstance(metadata, Mapping):
        return False
    try:
        required_manifests = (
            ("source_manifest", sorted((SRC_ROOT / "hunter_seeker_v2").glob("*.py")) + [Path(__file__).resolve()]),
            ("config_manifest", list(CONFIG_PATHS)),
            ("runner_manifest", [Path(__file__).resolve()]),
        )
        manifests_pass = all(
            isinstance(manifest.get(name), Mapping)
            and manifest.get(name) == _file_manifest(paths)
            for name, paths in required_manifests
        )
        package_pass = manifest.get("package_manifest") == _package_manifest()
    except (ExperimentContractError, OSError, ValueError):
        manifests_pass = False
        package_pass = False
    try:
        schema_pass = manifest.get("schema_manifest") == _schema_manifest()
    except (ExperimentContractError, OSError, ValueError):
        schema_pass = False
    return bool(
        source.get("verified") is True
        and metadata.get("verified") is True
        and source.get("path") == str(SOURCE_PATH.relative_to(PROJECT_ROOT))
        and metadata.get("path") == str(METADATA_PATH.relative_to(PROJECT_ROOT))
        and source.get("expected_sha256") == EXPECTED_SOURCE_SHA256
        and metadata.get("expected_sha256") == EXPECTED_METADATA_SHA256
        and source.get("observed_sha256") == EXPECTED_SOURCE_SHA256
        and metadata.get("observed_sha256") == EXPECTED_METADATA_SHA256
        and environment.get("selection") == "local_offline_exact_version"
        and manifests_pass
        and package_pass
        and schema_pass
    )


def _causal_design_eligible(
    *,
    pair_count: int,
    attempt_limit: int,
    action_limit: int,
    pairs: Sequence[Mapping[str, Any]],
    checks: Mapping[str, Any],
    authoritative: bool = False,
    provenance_manifest: Mapping[str, Any] | None = None,
    finalization_marker: Mapping[str, Any] | None = None,
    live_run_token: object | None = None,
    smoke: bool = False,
    blocked: bool = False,
) -> bool:
    del checks  # Eligibility is recomputed from the recorded rows below.
    internal_checks = _structural_checks(pairs)
    return bool(
        not smoke
        and not blocked
        and authoritative is True
        and live_run_token in _LIVE_RUN_TOKENS
        and _exact_pinned_manifest(provenance_manifest)
        and isinstance(finalization_marker, Mapping)
        and finalization_marker.get("status") == "finalized"
        and finalization_marker.get("complete") is True
        and finalization_marker.get("raw_evidence_sha256") == _digest(pairs)
        and finalization_marker.get("checks_sha256") == _digest(internal_checks)
        and pair_count == REPLICATE_COUNT
        and attempt_limit == ATTEMPT_LIMIT
        and action_limit == ACTION_LIMIT
        and len(pairs) == REPLICATE_COUNT
        and internal_checks.get("structural_pass") is True
        and internal_checks.get("no_stage_2_primary_actions") is True
    )


def _run_experiment(
    *,
    pair_count: int,
    attempt_limit: int,
    action_limit: int,
    environment_factory: Callable[[int], Any] | None,
    authoritative: bool,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shared execution core; only the public authoritative wrapper is contractual."""

    if pair_count < 1 or pair_count > REPLICATE_COUNT:
        raise ValueError(f"pair_count must be in [1, {REPLICATE_COUNT}]")
    if attempt_limit < 1 or attempt_limit > ATTEMPT_LIMIT:
        raise ValueError(f"attempt_limit must be in [1, {ATTEMPT_LIMIT}]")
    if action_limit < 1 or action_limit > ACTION_LIMIT:
        raise ValueError(f"action_limit must be in [1, {ACTION_LIMIT}]")
    manifest = (
        _provenance_manifest(Path(__file__).resolve())
        if authoritative
        else {
            "schema_version": SCHEMA_VERSION,
            "authoritative": False,
            "status": "non_authoritative_test_helper",
            "exact_environment_verification": False,
        }
    )
    offline_arcade = make_exact_arcade() if authoritative else None
    live_run_token = None
    if authoritative:
        # The only construction site is after exact manifest verification and
        # exact offline Arcade setup.  The object never enters serialized data.
        live_run_token = object()
        _LIVE_RUN_TOKENS.add(live_run_token)
    pairs: list[dict[str, Any]] = []
    if progress is not None:
        progress["pairs"] = pairs
    for pair_index in range(pair_count):
        agent_seed = pair_index
        pair_id = f"replicate-{pair_index:02d}"
        persistent_agent = build_constructor_baseline_agent(agent_seed)
        arm_completed = {arm: False for arm in ARMS}
        pair_attempts: list[dict[str, Any]] = []
        pair_result: dict[str, Any] = {
            "pair_id": pair_id,
            "agent_seed": agent_seed,
            "attempts": pair_attempts,
            "persistent_completed": False,
            "isolated_completed": False,
            "valid_pair": True,
            "finalized": False,
            "valid_finalization": False,
            "stopped_after_first_completion": False,
        }
        pairs.append(pair_result)
        for attempt_index in range(attempt_limit):
            schedule = _seed_schedule(pair_index, attempt_index, agent_seed)
            rows: dict[str, dict[str, Any]] = {}
            for arm in schedule["arm_order"]:
                attempt_id = f"{pair_id}-attempt-{attempt_index:02d}"
                if arm_completed[arm]:
                    rows[arm] = {
                        "pair_id": pair_id,
                        "attempt_id": attempt_id,
                        "arm": arm,
                        "agent_seed": int(agent_seed),
                        "status": "skipped_after_first_completion",
                        "completed": True,
                        "stopped_after_first_completion": True,
                        "valid_finalization": True,
                        "initial_observation_fingerprint": None,
                        "committed_action_count": 0,
                    }
                    continue
                if arm == ARM_PERSISTENT:
                    agent = persistent_agent
                else:
                    agent = build_constructor_baseline_agent(agent_seed)
                random.seed(schedule["python_seed"])
                np.random.seed(schedule["numpy_seed"])
                environment = (
                    environment_factory(schedule["environment_seed"])
                    if not authoritative
                    else make_exact_environment(schedule["environment_seed"], arcade=offline_arcade)
                )
                attempt_capture: dict[str, Any] = {}
                try:
                    rows[arm] = _run_attempt(
                        agent,
                        environment,
                        pair_id=pair_id,
                        attempt_id=attempt_id,
                        arm=arm,
                        agent_seed=agent_seed,
                        action_limit=action_limit,
                        action_adapter=ArcActionAdapter() if authoritative else environment.action_adapter,
                        observation_adapter=ArcObservationAdapter() if authoritative else environment.observation_adapter,
                        outcome_adapter=ArcOutcomeAdapter() if authoritative else environment.outcome_adapter,
                        action_space=_game_action_enum() if authoritative else environment.action_space,
                        bootstrap=True,
                        constructor_baseline_required=(arm == ARM_ISOLATED or attempt_index == 0),
                        attempt_progress=attempt_capture,
                    )
                except BaseException as exc:
                    rows[arm] = _blocked_attempt_record(
                        pair_id=pair_id,
                        attempt_id=attempt_id,
                        arm=arm,
                        agent_seed=agent_seed,
                        capture=attempt_capture,
                        error=exc,
                    )
                    for missing_arm in ARMS:
                        rows.setdefault(
                            missing_arm,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "pair_id": pair_id,
                                "attempt_id": attempt_id,
                                "arm": missing_arm,
                                "agent_seed": int(agent_seed),
                                "status": "not_started_due_to_blocked_attempt",
                                "finalized": False,
                            },
                        )
                    pair_attempts.append(_pair_attempt_record(pair_id, attempt_index, schedule, rows, attempt_id))
                    pair_result["valid_pair"] = False
                    pair_result["finalized"] = False
                    raise
                arm_completed[arm] = bool(rows[arm].get("completed"))
            attempt_record = _pair_attempt_record(pair_id, attempt_index, schedule, rows, attempt_id)
            pair_attempts.append(attempt_record)
            pair_result["valid_pair"] = bool(pair_result["valid_pair"] and attempt_record["valid_pair"])
        pair_result["persistent_completed"] = any(
            _raw_completion(row["arms"][ARM_PERSISTENT])
            for row in pair_attempts
            if isinstance(row, Mapping) and isinstance(row.get("arms"), Mapping)
        )
        pair_result["isolated_completed"] = any(
            _raw_completion(row["arms"][ARM_ISOLATED])
            for row in pair_attempts
            if isinstance(row, Mapping) and isinstance(row.get("arms"), Mapping)
        )
        pair_result["stopped_after_first_completion"] = False
        pair_result["valid_finalization"] = True
        pair_result["finalized"] = bool(
            len(pair_attempts) == attempt_limit
            and all(attempt.get("finalized") for attempt in pair_attempts)
        )
    checks = _structural_checks(pairs)
    aggregate_rows = _aggregate_from_raw_pairs(pairs, checks)
    gate = _exact_mcnemar_gate(aggregate_rows, live_run_token=live_run_token)
    finalization_marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "finalized",
        "complete": True,
        "raw_evidence_sha256": _digest(pairs),
        "checks_sha256": _digest(checks),
    }
    eligible = _causal_design_eligible(
        pair_count=pair_count,
        attempt_limit=attempt_limit,
        action_limit=action_limit,
        pairs=pairs,
        checks=checks,
        authoritative=authoritative,
        provenance_manifest=manifest,
        finalization_marker=finalization_marker,
        live_run_token=live_run_token,
        smoke=not (
            pair_count == REPLICATE_COUNT
            and attempt_limit == ATTEMPT_LIMIT
            and action_limit == ACTION_LIMIT
        ),
    )
    _retire_live_run_token(live_run_token)
    verdict_code = (
        "CAUSAL_RETENTION_GATE_PASS"
        if eligible and gate["pass"]
        else ("INCOMPLETE_DESIGN" if not _requested_design(pair_count=pair_count, attempt_limit=attempt_limit, action_limit=action_limit)["exact_contract_design"] else "DESCRIPTIVE_ONLY_NO_CAUSAL_GATE")
    )
    design = _requested_design(pair_count=pair_count, attempt_limit=attempt_limit, action_limit=action_limit)
    design["causal_claim"] = verdict_code == "CAUSAL_RETENTION_GATE_PASS"
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "authoritative": bool(authoritative),
        "blocked": False,
        "historical_runner": {"path": "utilities/tests/manual/run_hs_v2_fresh_discovery_v1.py", "status": "historical_intermediate_not_used"},
        "requested_design": design,
        "design": design,
        "provenance_manifest": manifest,
        "finalization_marker": finalization_marker,
        "pairs": pairs,
        "aggregate_paired_result": {"replicates": aggregate_rows, "persistent_completions": sum(int(row["persistent_completed"]) for row in aggregate_rows), "isolated_completions": sum(int(row["isolated_completed"]) for row in aggregate_rows), "exact_mcnemar_gate": gate, "descriptive_only_unless_gate_passes": True},
        "checks": checks,
        "verdict": {"code": verdict_code, "causal_claim": verdict_code == "CAUSAL_RETENTION_GATE_PASS", "incomplete_design": not design["exact_contract_design"], "claim_scope": "TU93 pinned local variant and this bounded design only" if authoritative else "non-authoritative injected-environment test only; no TU93 provenance"},
    }


def run_experiment(*, pair_count: int = 1, attempt_limit: int = 2, action_limit: int = 10) -> dict[str, Any]:
    """Run the authoritative exact-environment experiment."""

    return _run_experiment(
        pair_count=pair_count,
        attempt_limit=attempt_limit,
        action_limit=action_limit,
        environment_factory=None,
        authoritative=True,
    )


def run_non_authoritative_smoke(
    environment_factory: Callable[[int], Any],
    *,
    pair_count: int = 1,
    attempt_limit: int = 1,
    action_limit: int = 1,
) -> dict[str, Any]:
    """Use an injected environment only for non-authoritative adapter tests."""

    return _run_experiment(
        pair_count=pair_count,
        attempt_limit=attempt_limit,
        action_limit=action_limit,
        environment_factory=environment_factory,
        authoritative=False,
    )


def _game_action_enum() -> Any:
    try:
        from arcengine import GameAction
    except ImportError as exc:
        raise ExperimentContractError("arcengine is unavailable; refusing to execute") from exc
    return GameAction


def _validate_authoritative_output_path(output: Path) -> Path:
    """Allow only a new direct child of the dedicated artifact root."""

    resolved = output.expanduser().resolve(strict=False)
    if resolved == ARTIFACT_ROOT or resolved.parent != ARTIFACT_ROOT:
        raise ExperimentContractError(
            f"--out-root must be a new direct artifact directory under {ARTIFACT_ROOT}"
        )
    if resolved.exists() or resolved.is_symlink():
        raise ExperimentContractError(
            f"refusing to replace existing or unsafe artifact output: {resolved}"
        )
    return resolved


def write_artifacts(output: Path, summary: Mapping[str, Any]) -> None:
    """Publish a complete artifact directory atomically.

    The destination must be new.  Existing live or historical directories are
    never moved, deleted, or replaced.
    """

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        payloads = [
            ("summary.json", summary),
            ("provenance_manifest.json", summary.get("provenance_manifest", {})),
            ("verdict.json", summary.get("verdict", {})),
        ]
        if summary.get("blocked"):
            payloads.append(("blocked.json", summary))
        payloads.append(
            (
                "historical_status.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked_artifact" if summary.get("blocked") else "live_artifact",
                    "historical": False,
                    "stale": False,
                    "immutable": True,
                },
            )
        )
        for pair in summary.get("pairs", ()):
            if isinstance(pair, Mapping) and pair.get("pair_id"):
                payloads.append((f"trace_{pair['pair_id']}.json", pair))
        for name, payload in payloads:
            encoded = (json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            with (staging / name).open("wb") as target:
                target.write(encoded)
                target.flush()
                os.fsync(target.fileno())
        marker = {
            "schema_version": SCHEMA_VERSION,
            "artifact_complete": True,
            "immutable": True,
            "summary_sha256": _digest(summary),
            "files": [name for name, _payload in payloads],
        }
        marker_encoded = (json.dumps(marker, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        with (staging / "COMPLETION_MARKER.json").open("wb") as target:
            target.write(marker_encoded)
            target.flush()
            os.fsync(target.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to replace existing artifact output: {output}")
        os.replace(staging, output)
        staging = None  # type: ignore[assignment]
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=1, help=f"replicate pairs (1..{REPLICATE_COUNT}; --full uses {REPLICATE_COUNT})")
    parser.add_argument("--attempts", type=int, default=2, help=f"attempts per arm (1..{ATTEMPT_LIMIT}; --full uses {ATTEMPT_LIMIT})")
    parser.add_argument("--actions", type=int, default=10, help=f"committed actions per attempt (1..{ACTION_LIMIT}; --full uses {ACTION_LIMIT})")
    parser.add_argument("--full", action="store_true", help="run the full 32-pair, 20-attempt, 50-action contract")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args(argv)
    limits = (REPLICATE_COUNT, ATTEMPT_LIMIT, ACTION_LIMIT) if args.full else (args.pairs, args.attempts, args.actions)
    requested_output = args.out_root or (
        ARTIFACT_ROOT
        / f"fresh_retention_v1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
    )
    output = _validate_authoritative_output_path(requested_output)
    progress: dict[str, Any] = {"pairs": []}
    try:
        summary = _run_experiment(
            pair_count=limits[0],
            attempt_limit=limits[1],
            action_limit=limits[2],
            environment_factory=None,
            authoritative=True,
            progress=progress,
        )
    except BaseException as exc:
        summary = _blocked_summary(
            pair_count=limits[0],
            attempt_limit=limits[1],
            action_limit=limits[2],
            error=exc,
            partial_pairs=progress.get("pairs", ()),
        )
    write_artifacts(output.resolve(), summary)
    print(json.dumps(summary["verdict"], sort_keys=True))
    print(f"artifact: {output.resolve() / 'summary.json'}")
    return 2 if summary.get("blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
