from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest


MANUAL_ROOT = Path(__file__).resolve().parents[1] / "manual"
if str(MANUAL_ROOT) not in sys.path:
    sys.path.insert(0, str(MANUAL_ROOT))

import run_hs_v2_tu93_fresh_retention_v1 as experiment  # noqa: E402
from hunter_seeker_v2.adapters import (  # noqa: E402
    MockActionAdapter,
    MockObservationAdapter,
    MockOutcomeAdapter,
)


class FakeEpisode:
    """Small real-adapter environment used to execute the retention helper."""

    action_adapter = MockActionAdapter(n_actions=8)
    observation_adapter = MockObservationAdapter(n_actions=8)
    outcome_adapter = MockOutcomeAdapter()
    action_space = None

    def __init__(self, *, complete_on_action: int | None = None, seed: int = 0, completion_state: str = "GAME_COMPLETED") -> None:
        self.complete_on_action = complete_on_action
        self.seed = seed
        self.completion_state = completion_state
        self.calls = 0

    def _raw(self, *, state: str = "RUNNING", progress: float = 0.0) -> dict:
        frame = np.zeros((5, 5), dtype=np.uint8)
        frame[2, (self.calls + self.seed) % 5] = 3
        return {
            "grid": frame,
            "available_actions": [0, 1, 2, 3],
            "progress": progress,
            "state": state,
        }

    def reset(self) -> dict:
        self.calls = 0
        return self._raw()

    def step(self, _action: int, **_kwargs: object) -> dict:
        self.calls += 1
        if self.complete_on_action is not None and self.calls > self.complete_on_action:
            return self._raw(state=self.completion_state, progress=1.0)
        return self._raw()

    def close(self) -> None:
        pass


def _factory(seed: int) -> FakeEpisode:
    return FakeEpisode(seed=seed)


class FailAfterEnvironmentStepAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def clear(self, task_id: str) -> None:
        FakeEpisode.observation_adapter.clear(task_id)

    def observation(self, raw: object, *, task_id: str):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("observation failed after environment step")
        return FakeEpisode.observation_adapter.observation(raw, task_id=task_id)


class FailOnBootstrap(FakeEpisode):
    def step(self, _action: int, **_kwargs: object) -> dict:
        raise RuntimeError("bootstrap environment step failed")


def test_exact_environment_hashes_are_verified() -> None:
    manifest = experiment.verify_environment_contract()
    assert manifest["game_id"] == "tu93-0768757b"
    assert manifest["source"]["observed_sha256"] == experiment.EXPECTED_SOURCE_SHA256
    assert manifest["metadata"]["observed_sha256"] == experiment.EXPECTED_METADATA_SHA256
    assert manifest["source"]["verified"] is True
    assert manifest["metadata"]["verified"] is True


def test_authoritative_runner_has_no_forbidden_runner_or_loader_imports() -> None:
    source_path = Path(experiment.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
    assert not any("run_hs_v2_tu93_graph_goal_v1" in name for name in imported_modules)
    assert not any("run_hs_v2_fresh_discovery_v1" in name for name in imported_modules)
    assert "load_checkpoint" not in imported_modules
    assert "save_checkpoint" not in imported_modules
    assert "teacher" not in {name.rsplit(".", 1)[-1].lower() for name in imported_modules}


def test_reset_vs_persistent_state_behavior_uses_real_attempt_helper() -> None:
    persistent = experiment.build_constructor_baseline_agent(0)
    first = experiment._run_attempt(
        persistent,
        FakeEpisode(seed=11),
        pair_id="p0",
        attempt_id="p0-a0",
        arm=experiment.ARM_PERSISTENT,
        agent_seed=0,
        action_limit=2,
        action_adapter=FakeEpisode.action_adapter,
        observation_adapter=FakeEpisode.observation_adapter,
        outcome_adapter=FakeEpisode.outcome_adapter,
    )
    second = experiment._run_attempt(
        persistent,
        FakeEpisode(seed=11),
        pair_id="p0",
        attempt_id="p0-a1",
        arm=experiment.ARM_PERSISTENT,
        agent_seed=0,
        action_limit=2,
        action_adapter=FakeEpisode.action_adapter,
        observation_adapter=FakeEpisode.observation_adapter,
        outcome_adapter=FakeEpisode.outcome_adapter,
    )
    isolated = experiment.build_constructor_baseline_agent(0)
    isolated_attempt = experiment._run_attempt(
        isolated,
        FakeEpisode(seed=11),
        pair_id="p0",
        attempt_id="p0-b0",
        arm=experiment.ARM_ISOLATED,
        agent_seed=0,
        action_limit=2,
        action_adapter=FakeEpisode.action_adapter,
        observation_adapter=FakeEpisode.observation_adapter,
        outcome_adapter=FakeEpisode.outcome_adapter,
    )
    assert first["status"] == second["status"] == isolated_attempt["status"] == "valid"
    assert first["state_store_before"]["components"]["graph"]["nodes"] == 0
    assert second["state_store_before"]["components"]["graph"]["nodes"] > 0
    assert isolated_attempt["state_store_before"]["components"]["graph"]["nodes"] == 0


def test_stage_one_completion_stops_primary_trace_before_stage_two_action() -> None:
    agent = experiment.build_constructor_baseline_agent(0)
    result = experiment._run_attempt(
        agent,
        FakeEpisode(complete_on_action=1),
        pair_id="p0",
        attempt_id="p0-a0",
        arm=experiment.ARM_PERSISTENT,
        agent_seed=0,
        action_limit=5,
        action_adapter=FakeEpisode.action_adapter,
        observation_adapter=FakeEpisode.observation_adapter,
        outcome_adapter=FakeEpisode.outcome_adapter,
    )
    assert result["status"] == "valid"
    assert result["completed"] is True
    assert result["stopped_reason"] == "first_stage_1_completion"
    assert result["committed_action_count"] == 1
    assert all(row["stage"] == 1 for row in result["committed_transitions"])
    assert len(result["completion_lineage"]) == 1


def test_completion_on_exact_final_allowed_action_keeps_level_completed() -> None:
    agent = experiment.build_constructor_baseline_agent(0)
    result = experiment._run_attempt(
        agent,
        FakeEpisode(complete_on_action=2, completion_state="RUNNING"),
        pair_id="p0",
        attempt_id="p0-final",
        arm=experiment.ARM_PERSISTENT,
        agent_seed=0,
        action_limit=2,
        action_adapter=FakeEpisode.action_adapter,
        observation_adapter=FakeEpisode.observation_adapter,
        outcome_adapter=FakeEpisode.outcome_adapter,
    )
    assert result["status"] == "valid"
    assert result["completed"] is True
    assert result["final_boundary"] == experiment.BoundaryKind.LEVEL_COMPLETED.value
    assert result["committed_action_count"] == 2
    assert result["committed_transitions"][-1]["outcome"]["completed"] is True


def test_smoke_experiment_has_paired_schema_and_descriptive_verdict() -> None:
    summary = experiment.run_non_authoritative_smoke(
        _factory,
        pair_count=1,
        attempt_limit=1,
        action_limit=1,
    )
    assert summary["experiment"] == experiment.EXPERIMENT_ID
    assert summary["authoritative"] is False
    assert summary["provenance_manifest"]["exact_environment_verification"] is False
    assert summary["design"]["arms"] == experiment.ARMS
    assert len(summary["pairs"]) == 1
    pair = summary["pairs"][0]
    assert pair["agent_seed"] == 0
    attempt = pair["attempts"][0]
    assert attempt["valid_pair"] is True
    assert attempt["matching_initial_observation_state"] is True
    assert set(attempt["arms"]) == set(experiment.ARMS)
    assert "aggregate_paired_result" in summary
    assert "exact_mcnemar_gate" in summary["aggregate_paired_result"]
    assert summary["verdict"]["causal_claim"] is False
    assert summary["verdict"]["code"] == "INCOMPLETE_DESIGN"


def test_authoritative_runner_has_no_environment_factory_bypass() -> None:
    assert "environment_factory" not in inspect.signature(experiment.run_experiment).parameters
    assert inspect.signature(experiment.run_non_authoritative_smoke).parameters["environment_factory"].default is inspect.Parameter.empty


def test_causal_gate_requires_exact_design_and_all_structural_conditions() -> None:
    pair = {"valid_pair": True, "finalized": True}
    checks = {"structural_pass": True, "no_stage_2_primary_actions": True}
    pairs = [pair] * experiment.REPLICATE_COUNT
    assert experiment._causal_design_eligible(
        pair_count=experiment.REPLICATE_COUNT,
        attempt_limit=experiment.ATTEMPT_LIMIT,
        action_limit=experiment.ACTION_LIMIT,
        pairs=pairs,
        checks=checks,
        authoritative=True,
        provenance_manifest={"environment_manifest": {"game_id": experiment.EXACT_GAME_ID}},
    ) is False
    for kwargs in (
        {"pair_count": 31, "attempt_limit": 20, "action_limit": 50},
        {"pair_count": 32, "attempt_limit": 19, "action_limit": 50},
        {"pair_count": 32, "attempt_limit": 20, "action_limit": 49},
    ):
        assert experiment._causal_design_eligible(
            **kwargs, pairs=pairs, checks=checks
        ) is False
    assert experiment._causal_design_eligible(
        pair_count=32,
        attempt_limit=20,
        action_limit=50,
        pairs=[*pairs[:-1], {"valid_pair": False, "finalized": True}],
        checks=checks,
    ) is False
    assert experiment._causal_design_eligible(
        pair_count=32,
        attempt_limit=20,
        action_limit=50,
        pairs=pairs,
        checks={"structural_pass": True, "no_stage_2_primary_actions": False},
    ) is False


def test_structural_checks_fail_closed_for_missing_or_malformed_stage_fields() -> None:
    valid_row = {
        "status": "valid",
        "initial_stage": 1,
        "final_stage": 1,
        "committed_transitions": [{"stage": 1, "after_stage": 1}],
        "leakage_audit": {"fresh_isolation_pass": True},
    }
    pair = {
        "valid_pair": True,
        "finalized": True,
        "attempts": [{
            "finalized": True,
            "skipped_after_completion": False,
            "arms": {
                experiment.ARM_PERSISTENT: {**valid_row, "initial_observation_fingerprint": "o", "initial_state_id": "s"},
                experiment.ARM_ISOLATED: {**valid_row, "initial_observation_fingerprint": "o", "initial_state_id": "s"},
            },
        }],
    }
    assert experiment._structural_checks([pair])["structural_pass"] is False
    malformed = {**valid_row, "committed_transitions": [{"stage": 1}]}
    bad_pair = {**pair, "attempts": [{**pair["attempts"][0], "arms": {experiment.ARM_PERSISTENT: malformed, experiment.ARM_ISOLATED: valid_row}}]}
    assert experiment._structural_checks([bad_pair])["stage_fields_pass"] is False
    assert experiment._structural_checks([bad_pair])["structural_pass"] is False


def test_matching_keeps_observation_state_id_and_agent_store_fingerprints_distinct() -> None:
    agent = experiment.build_constructor_baseline_agent(0)
    result = experiment._run_attempt(
        agent,
        FakeEpisode(seed=11),
        pair_id="p0",
        attempt_id="p0-a0",
        arm=experiment.ARM_PERSISTENT,
        agent_seed=0,
        action_limit=1,
        action_adapter=FakeEpisode.action_adapter,
        observation_adapter=FakeEpisode.observation_adapter,
        outcome_adapter=FakeEpisode.outcome_adapter,
    )
    assert result["initial_state_fingerprint"] != result["initial_observation_fingerprint"]
    assert result["initial_agent_store_fingerprint"] == result["state_store_before"]["overall_fingerprint"]
    assert result["initial_state_id"]


def test_runtime_audit_reports_supported_static_and_boundary_verification() -> None:
    agent = experiment.build_constructor_baseline_agent(0)
    state = experiment.state_store_fingerprint(agent)
    audit = experiment.runtime_leakage_audit(
        agent,
        before=state,
        after=state,
        arm=experiment.ARM_ISOLATED,
        constructor_baseline_required=True,
    )
    assert audit["no_forbidden_imports_in_authoritative_runner"] is True
    assert audit["forbidden_call_counter_status"] == "not_applicable_static_source_and_runtime_boundary_supported"
    assert audit["static_source_verification"]["supported"] is True
    assert audit["runtime_boundary_verification"]["supported"] is True
    assert audit["forbidden_call_instrumentation"]["pass"] is True
    assert all(value["supported"] is True for value in audit["forbidden_call_counts"].values())


def test_structural_checks_recompute_schedule_and_transition_identity() -> None:
    pair = {
        "pair_id": "replicate-00",
        "agent_seed": 0,
        "attempts": [],
        "persistent_completed": False,
        "isolated_completed": False,
        "valid_pair": True,
        "finalized": True,
        "valid_finalization": True,
        "stopped_after_first_completion": False,
    }
    pairs = [dict(pair, pair_id=f"replicate-{index:02d}", agent_seed=index) for index in range(experiment.REPLICATE_COUNT)]
    checks = experiment._structural_checks(pairs)
    assert checks["pair_identity_pass"] is True
    assert checks["attempt_records_pass"] is False

    malformed = dict(pairs[0])
    malformed["attempts"] = [{
        "pair_id": malformed["pair_id"],
        "attempt_id": "duplicate-attempt",
        "attempt_index": 0,
        "schedule": experiment._seed_schedule(0, 0, 0),
        "arm_order": experiment._seed_schedule(0, 0, 0)["arm_order"],
        "finalized": True,
        "valid_pair": True,
        "arms": {
            arm: {"status": "aborted", "pair_id": malformed["pair_id"], "arm": arm}
            for arm in experiment.ARMS
        },
    }]
    assert experiment._structural_checks([*pairs[1:], malformed])["row_status_pass"] is False


def test_causal_gate_requires_finalization_marker_and_exact_raw_evidence_digest() -> None:
    pairs = [{"pair_id": f"replicate-{index:02d}", "agent_seed": index} for index in range(experiment.REPLICATE_COUNT)]
    assert experiment._causal_design_eligible(
        pair_count=experiment.REPLICATE_COUNT,
        attempt_limit=experiment.ATTEMPT_LIMIT,
        action_limit=experiment.ACTION_LIMIT,
        pairs=pairs,
        checks={},
        authoritative=True,
        provenance_manifest={"authoritative": True, "exact_environment_verification": True, "environment_manifest": {
            "game_id": experiment.EXACT_GAME_ID,
            "selection": "local_offline_exact_version",
            "source": {"verified": True, "observed_sha256": experiment.EXPECTED_SOURCE_SHA256},
            "metadata": {"verified": True, "observed_sha256": experiment.EXPECTED_METADATA_SHA256},
        }},
        finalization_marker={"status": "finalized", "complete": True, "raw_evidence_sha256": "wrong"},
    ) is False


def test_serialized_full_evidence_cannot_authorize_without_process_local_token() -> None:
    pairs = [{"pair_id": f"replicate-{index:02d}", "agent_seed": index} for index in range(experiment.REPLICATE_COUNT)]
    checks = experiment._structural_checks(pairs)
    manifest = experiment._provenance_manifest(Path(experiment.__file__).resolve())
    marker = {
        "status": "finalized",
        "complete": True,
        "raw_evidence_sha256": experiment._digest(pairs),
        "checks_sha256": experiment._digest(checks),
    }
    assert experiment._causal_design_eligible(
        pair_count=experiment.REPLICATE_COUNT,
        attempt_limit=experiment.ATTEMPT_LIMIT,
        action_limit=experiment.ACTION_LIMIT,
        pairs=pairs,
        checks=checks,
        authoritative=True,
        provenance_manifest=manifest,
        finalization_marker=marker,
        live_run_token=None,
    ) is False
    assert experiment._causal_design_eligible(
        pair_count=experiment.REPLICATE_COUNT,
        attempt_limit=experiment.ATTEMPT_LIMIT,
        action_limit=experiment.ACTION_LIMIT,
        pairs=pairs,
        checks=checks,
        authoritative=True,
        provenance_manifest=manifest,
        finalization_marker={**marker, "live_run_token": "serialized-looking-token"},
    ) is False


def test_structural_checks_reject_missing_bootstrap_or_state_store_evidence() -> None:
    agent = experiment.build_constructor_baseline_agent(0)
    row = experiment._run_attempt(
        agent,
        FakeEpisode(seed=11),
        pair_id="replicate-00",
        attempt_id="replicate-00-attempt-00",
        arm=experiment.ARM_PERSISTENT,
        agent_seed=0,
        action_limit=1,
        action_adapter=FakeEpisode.action_adapter,
        observation_adapter=FakeEpisode.observation_adapter,
        outcome_adapter=FakeEpisode.outcome_adapter,
    )
    assert experiment._bootstrap_evidence_pass(row)
    assert experiment._state_store_evidence_pass(row)
    assert not experiment._bootstrap_evidence_pass({**row, "bootstrap_environment_step": {}})
    assert not experiment._state_store_evidence_pass({**row, "state_store_after": {}})


def test_main_rejects_workspace_ancestor_and_existing_output_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(experiment, "ARTIFACT_ROOT", tmp_path / "dedicated")
    experiment.ARTIFACT_ROOT.mkdir()
    with pytest.raises(experiment.ExperimentContractError, match="new direct artifact"):
        experiment.main(["--pairs", "1", "--attempts", "1", "--actions", "1", "--out-root", str(tmp_path)])
    existing = experiment.ARTIFACT_ROOT / "existing"
    existing.mkdir()
    with pytest.raises(experiment.ExperimentContractError, match="existing"):
        experiment.main(["--pairs", "1", "--attempts", "1", "--actions", "1", "--out-root", str(existing)])


def test_blocked_aggregate_uses_raw_rows_not_fabricated_pair_flags() -> None:
    pair = {
        "pair_id": "replicate-00",
        "persistent_completed": True,
        "isolated_completed": True,
        "attempts": [{
            "arms": {
                experiment.ARM_PERSISTENT: {"committed_transitions": [{"outcome": {"completed": False}}]},
                experiment.ARM_ISOLATED: {"committed_transitions": [{"outcome": {"completed": False}}]},
            }
        }],
    }
    summary = experiment._blocked_summary(
        pair_count=1,
        attempt_limit=1,
        action_limit=1,
        error=RuntimeError("blocked"),
        partial_pairs=[pair],
    )
    assert summary["aggregate_paired_result"]["replicates"] == [{
        "pair_id": "replicate-00",
        "valid_pair": False,
        "persistent_completed": False,
        "isolated_completed": False,
    }]


def test_observation_failure_after_step_records_executed_action_and_invalidates_attempt() -> None:
    agent = experiment.build_constructor_baseline_agent(0)
    progress = {}
    with pytest.raises(RuntimeError, match="observation failed"):
        experiment._run_attempt(
            agent,
            FakeEpisode(seed=11),
            pair_id="p0",
            attempt_id="p0-error",
            arm=experiment.ARM_PERSISTENT,
            agent_seed=0,
            action_limit=1,
            action_adapter=FakeEpisode.action_adapter,
            observation_adapter=FailAfterEnvironmentStepAdapter(),
            outcome_adapter=FakeEpisode.outcome_adapter,
            attempt_progress=progress,
        )
    assert progress["executed_action_evidence"]["environment_step_attempted"] is True
    assert progress["executed_action_evidence"]["environment_step_returned"] is True
    assert agent.pending_decision is not None


def test_bootstrap_failure_preserves_attempted_evidence() -> None:
    agent = experiment.build_constructor_baseline_agent(0)
    progress = {}
    with pytest.raises(RuntimeError, match="bootstrap environment step failed"):
        experiment._run_attempt(
            agent,
            FailOnBootstrap(seed=11),
            pair_id="p0",
            attempt_id="p0-bootstrap-error",
            arm=experiment.ARM_PERSISTENT,
            agent_seed=0,
            action_limit=1,
            action_adapter=FakeEpisode.action_adapter,
            observation_adapter=FakeEpisode.observation_adapter,
            outcome_adapter=FakeEpisode.outcome_adapter,
            attempt_progress=progress,
        )
    assert progress["bootstrap_environment_step"]["attempted"] is True
    assert progress["bootstrap_environment_step"]["returned"] is False
    blocked = experiment._blocked_attempt_record(
        pair_id="p0",
        attempt_id="p0-bootstrap-error",
        arm=experiment.ARM_PERSISTENT,
        agent_seed=0,
        capture=progress,
        error=RuntimeError("bootstrap environment step failed"),
    )
    assert blocked["bootstrap_environment_step"]["attempted"] is True
    assert blocked["bootstrap_environment_step"]["returned"] is False


def test_fingerprint_includes_runtime_fields_full_replay_and_rng_support_status() -> None:
    state = experiment.state_store_fingerprint(experiment.build_constructor_baseline_agent(0))
    runtime = state["components"]["runtime_fields"]["fields"]
    assert all(name in runtime for name in (
        "_total_transition_count", "_step", "_decision_counter",
        "_awaiting_visual_reset", "_task_id", "_current_snapshot",
    ))
    assert "replay_payloads" in state["components"]["replay"] or state["components"]["replay"]["items"] == 0
    assert state["components"]["private_rng_states"]["states"]["agent._rng"]["supported"] is True
    assert runtime["_environment_internal_state"]["supported"] is False


class FailOnSecondPrimaryStep(FakeEpisode):
    def step(self, action: int, **kwargs: object) -> dict:
        if self.calls >= 3:
            raise RuntimeError("second primary environment step failed")
        return super().step(action, **kwargs)


def test_later_environment_step_failure_preserves_current_action_evidence() -> None:
    agent = experiment.build_constructor_baseline_agent(0)
    progress = {}
    with pytest.raises(RuntimeError, match="second primary"):
        experiment._run_attempt(
            agent,
            FailOnSecondPrimaryStep(seed=11),
            pair_id="p0",
            attempt_id="p0-later-error",
            arm=experiment.ARM_PERSISTENT,
            agent_seed=0,
            action_limit=3,
            action_adapter=FakeEpisode.action_adapter,
            observation_adapter=FakeEpisode.observation_adapter,
            outcome_adapter=FakeEpisode.outcome_adapter,
            attempt_progress=progress,
        )
    assert [row["environment_step_returned"] for row in progress["environment_steps"]] == [True, True, False]
    assert progress["executed_action_evidence"]["step"] == 2
    assert progress["executed_action_evidence"]["environment_step_returned"] is False
    assert len(progress["committed_transitions"]) == 2
    assert agent.pending_decision is not None


def test_structural_checks_reject_aborted_rows_and_duplicate_pair_identity() -> None:
    bad = [{"pair_id": "duplicate", "agent_seed": 0, "attempts": [{"attempt_index": 0, "pair_id": "duplicate", "arms": {experiment.ARM_PERSISTENT: {"status": "aborted"}, experiment.ARM_ISOLATED: {"status": "valid"}}}]}] * 2
    checks = experiment._structural_checks(bad)
    assert checks["pair_identity_pass"] is False
    assert checks["row_status_pass"] is False
    assert checks["structural_pass"] is False


def test_write_artifacts_rejects_existing_directory_and_publishes_immutable_tree(tmp_path) -> None:
    summary = {
        "schema_version": experiment.SCHEMA_VERSION,
        "experiment": experiment.EXPERIMENT_ID,
        "blocked": True,
        "provenance_manifest": {"schema_version": experiment.SCHEMA_VERSION},
        "verdict": {"code": "BLOCKED_RUNTIME_ERROR"},
        "pairs": [],
    }
    output = tmp_path / "artifact"
    experiment.write_artifacts(output, summary)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        experiment.write_artifacts(output, summary)
    assert (output / "summary.json").is_file()
    assert (output / "blocked.json").is_file()
    assert (output / "COMPLETION_MARKER.json").is_file()
    assert json.loads((output / "historical_status.json").read_text(encoding="utf-8"))["immutable"] is True
    assert not list(tmp_path.glob("artifact.historical_stale_*/summary.json"))


def test_write_artifacts_never_moves_existing_output(tmp_path) -> None:
    summary = {
        "schema_version": experiment.SCHEMA_VERSION,
        "experiment": experiment.EXPERIMENT_ID,
        "blocked": True,
        "provenance_manifest": {},
        "verdict": {"code": "OLD"},
        "pairs": [],
    }
    output = tmp_path / "rollback"
    experiment.write_artifacts(output, summary)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        experiment.write_artifacts(output, summary)
    assert json.loads((output / "verdict.json").read_text(encoding="utf-8"))["code"] == "OLD"
    assert not list(tmp_path.glob("rollback.historical_stale_*/summary.json"))


def test_blocked_main_writes_schema_valid_metadata_on_keyboard_interrupt(tmp_path, monkeypatch) -> None:
    def interrupt(**_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(experiment, "_run_experiment", interrupt)
    monkeypatch.setattr(experiment, "ARTIFACT_ROOT", tmp_path)
    output = tmp_path / "blocked"
    assert experiment.main(["--pairs", "1", "--attempts", "1", "--actions", "1", "--out-root", str(output)]) == 2
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["blocked"] is True
    assert summary["requested_design"]["replicates_requested"] == 1
    assert "provenance_manifest" in summary
    assert summary["pairs"] == []
    assert json.loads((output / "provenance_manifest.json").read_text(encoding="utf-8")) == summary["provenance_manifest"]


def test_runtime_attempt_exception_becomes_blocked_top_level_artifact(tmp_path, monkeypatch) -> None:
    def fail_after_partial(**kwargs):
        progress = kwargs["progress"]
        pair = {
            "pair_id": "replicate-00",
            "agent_seed": 0,
            "attempts": [{
                "pair_id": "replicate-00",
                "attempt_index": 0,
                "arms": {
                    experiment.ARM_PERSISTENT: {
                        "status": "blocked_by_runtime_exception",
                        "executed_action_evidence": {
                            "step": 0,
                            "environment_step_attempted": True,
                            "environment_step_returned": True,
                        },
                    },
                    experiment.ARM_ISOLATED: {"status": "not_started_due_to_blocked_attempt"},
                },
            }],
        }
        progress["pairs"].append(pair)
        raise RuntimeError("ordinary attempt failure")

    monkeypatch.setattr(experiment, "_run_experiment", fail_after_partial)
    monkeypatch.setattr(experiment, "ARTIFACT_ROOT", tmp_path)
    output = tmp_path / "blocked-runtime"
    assert experiment.main(["--pairs", "1", "--attempts", "1", "--actions", "1", "--out-root", str(output)]) == 2
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["blocked"] is True
    assert summary["verdict"]["code"] == "BLOCKED_RUNTIME_ERROR"
    assert summary["pairs"][0]["attempts"][0]["arms"][experiment.ARM_PERSISTENT]["executed_action_evidence"]["environment_step_attempted"] is True
    assert json.loads((output / "blocked.json").read_text(encoding="utf-8"))["blocked"] is True
