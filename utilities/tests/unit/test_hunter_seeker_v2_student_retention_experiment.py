from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hunter_seeker_v2.contracts import (
    Action,
    Observation,
    Outcome,
    Representation,
    RuntimeMode,
    Topology,
    Transition,
    WorldSnapshot,
)
from hunter_seeker_v2.teacher import (
    TeacherAccessGuard,
    TeacherAccessViolation,
    TeacherRequest,
)
from utilities.tests.manual import run_hs_v2_student_retention_v1 as experiment


def _snapshot(frame: np.ndarray, *, stage: int = 1, step: int = 0) -> WorldSnapshot:
    observation = Observation(
        frame=frame,
        available_actions=(1, 2),
        task_id="route-game",
        stage=stage,
    )
    return WorldSnapshot(
        observation=observation,
        objects=(),
        events=(),
        topology=Topology(),
        representation=Representation(
            global_vector=np.zeros(4, dtype=np.float32),
            spatial=np.zeros(frame.shape, dtype=np.float32),
        ),
        step=step,
    )


def _transition(
    position: int,
    before: np.ndarray,
    after: np.ndarray,
    action: Action,
    *,
    stage: int = 1,
) -> Transition:
    snapshot = _snapshot(before, stage=stage, step=position)
    after_observation = Observation(
        frame=after,
        available_actions=(1, 2),
        task_id="route-game",
        stage=stage,
    )
    return Transition(
        transition_id=f"transition-{position}",
        decision_id=f"decision-{position}",
        task_id="route-game",
        stage=stage,
        step=position,
        before=snapshot,
        action=action,
        after_observation=after_observation,
        outcome=Outcome(),
        frame_changed=not np.array_equal(before, after),
        after_state_id=after_observation.state_id,
    )


def _route(tmp_path: Path) -> experiment.TrustedTrajectory:
    frames = np.asarray(
        [
            [[0, 1], [1, 0]],
            [[2, 0], [0, 2]],
        ],
        dtype=np.uint8,
    )
    frames_after = np.asarray(
        [
            [[1, 1], [1, 0]],
            [[2, 2], [0, 2]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "route-game_run0_traj.npz"
    np.savez(
        path,
        frames=frames,
        frames_after=frames_after,
        actions=np.asarray([[1, -1, -1], [2, -1, -1]], dtype=np.int64),
        levels=np.asarray([0, 0], dtype=np.int64),
    )
    return experiment._load_trusted(tmp_path, "route-game", 0)


def _ambiguous_route(tmp_path: Path) -> experiment.TrustedTrajectory:
    before = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    after_one = np.asarray([[1, 1], [1, 0]], dtype=np.uint8)
    after_two = np.asarray([[0, 2], [1, 0]], dtype=np.uint8)
    path = tmp_path / "ambiguous-game_run0_traj.npz"
    np.savez(
        path,
        frames=np.stack((before, before)),
        frames_after=np.stack((after_one, after_two)),
        actions=np.asarray([[1, -1, -1], [2, -1, -1]], dtype=np.int64),
        levels=np.asarray([0, 0], dtype=np.int64),
    )
    return experiment._load_trusted(tmp_path, "ambiguous-game", 0)


def test_state_aligned_metrics_reject_frequency_only_route_imitation(
    tmp_path: Path,
) -> None:
    route = _route(tmp_path)
    transitions = (
        _transition(0, route.frames[0], route.frames_after[0], Action(2)),
        _transition(1, route.frames[1], route.frames_after[1], Action(1)),
    )
    result = SimpleNamespace(transitions=transitions)

    metrics, trace = experiment._analyze_episode(result, route)

    assert metrics["action_index_frequency_similarity"] == pytest.approx(1.0)
    assert metrics["position_action_index_overlap"] == pytest.approx(0.0)
    assert metrics["live_state_alignment_rate"] == pytest.approx(1.0)
    assert metrics["state_aligned_action_index_accuracy"] == pytest.approx(0.0)
    assert metrics["state_aligned_full_action_accuracy"] == pytest.approx(0.0)
    assert metrics["state_action_successor_accuracy"] == pytest.approx(0.0)
    assert metrics["strict_route_prefix_fraction"] == pytest.approx(0.0)
    assert all(not row["state_aligned_full_action_match"] for row in trace)


def test_state_action_successor_metric_requires_the_route_transition(
    tmp_path: Path,
) -> None:
    route = _route(tmp_path)
    correct = (
        _transition(0, route.frames[0], route.frames_after[0], Action(1)),
        _transition(1, route.frames[1], route.frames_after[1], Action(2)),
    )
    metrics, _trace = experiment._analyze_episode(
        SimpleNamespace(transitions=correct),
        route,
    )
    assert metrics["state_aligned_full_action_accuracy"] == pytest.approx(1.0)
    assert metrics["state_action_successor_accuracy"] == pytest.approx(1.0)
    assert metrics["trusted_row_coverage"] == pytest.approx(1.0)
    assert metrics["strict_route_prefix_fraction"] == pytest.approx(1.0)

    wrong_successor = (
        _transition(
            0,
            route.frames[0],
            np.full_like(route.frames_after[0], 7),
            Action(1),
        ),
    )
    metrics, _trace = experiment._analyze_episode(
        SimpleNamespace(transitions=wrong_successor),
        route,
    )
    assert metrics["state_aligned_full_action_accuracy"] == pytest.approx(1.0)
    assert metrics["state_action_successor_accuracy"] == pytest.approx(0.0)
    assert metrics["ordered_state_action_accuracy"] == pytest.approx(1.0)
    assert metrics["ordered_state_action_successor_accuracy"] == pytest.approx(0.0)

    adjacent_boundary_stage = (
        _transition(
            0,
            route.frames[0],
            route.frames_after[0],
            Action(1),
            stage=2,
        ),
    )
    metrics, _trace = experiment._analyze_episode(
        SimpleNamespace(transitions=adjacent_boundary_stage),
        route,
    )
    assert metrics["frame_only_aligned_steps"] == 1
    assert metrics["state_aligned_steps"] == 1
    assert metrics["exact_stage_state_aligned_steps"] == 0
    assert metrics["state_action_successor_accuracy"] == pytest.approx(1.0)
    assert metrics["strict_route_prefix_fraction"] == pytest.approx(0.5)
    assert metrics["exact_stage_strict_route_prefix_fraction"] == pytest.approx(0.0)

    unrelated_stage = (
        _transition(
            0,
            route.frames[0],
            route.frames_after[0],
            Action(1),
            stage=3,
        ),
    )
    metrics, _trace = experiment._analyze_episode(
        SimpleNamespace(transitions=unrelated_stage),
        route,
    )
    assert metrics["frame_only_aligned_steps"] == 1
    assert metrics["state_aligned_steps"] == 0
    assert metrics["exact_stage_state_aligned_steps"] == 0


def test_frame_digest_is_integer_width_independent() -> None:
    values = np.asarray([[0, 1], [15, 3]], dtype=np.uint8)
    expected = experiment._frame_digest(values)

    assert experiment._frame_digest(values.astype(np.int8)) == expected
    assert experiment._frame_digest(values.astype(np.int32)) == expected
    assert experiment._frame_digest(values.astype(np.int64)) == expected


def test_state_only_ordered_alignment_cannot_cherry_pick_repeated_route_rows(
    tmp_path: Path,
) -> None:
    route = _ambiguous_route(tmp_path)
    reversed_transitions = (
        _transition(0, route.frames[0], route.frames_after[1], Action(2)),
        _transition(1, route.frames[1], route.frames_after[0], Action(1)),
    )

    metrics, trace = experiment._analyze_episode(
        SimpleNamespace(transitions=reversed_transitions),
        route,
    )

    assert metrics["action_index_frequency_similarity"] == pytest.approx(1.0)
    # The intentionally optimistic occurrence metric can select a different
    # duplicate after seeing each action and successor.
    assert metrics["state_action_successor_accuracy"] == pytest.approx(1.0)
    # The primary mapping is fixed from state order alone: route rows 0 then 1.
    assert [row["ordered_state_route_row"] for row in trace] == [0, 1]
    assert metrics["ordered_state_action_accuracy"] == pytest.approx(0.0)
    assert metrics["ordered_state_action_successor_accuracy"] == pytest.approx(0.0)
    assert metrics["ordered_transition_shared_horizon_rate"] == pytest.approx(0.0)
    assert metrics["strict_route_prefix_shared_horizon_fraction"] == pytest.approx(
        0.0
    )


class _NullProvider:
    teacher_id = "audit-test"

    def suggest(self, _request):
        return None

    def examples(self, _query):
        return ()


def test_teacher_audit_rejects_even_a_denied_action_selection_attempt() -> None:
    guard = TeacherAccessGuard(_NullProvider(), mode=RuntimeMode.STUDENT)
    clean = experiment._teacher_audit_payload(guard)
    assert clean["action_selection_attempts"] == 0

    request = TeacherRequest(
        task_id="route-game",
        stage=1,
        state_id="state",
        candidates=(Action(1),),
    )
    with pytest.raises(TeacherAccessViolation):
        guard.action_selection().suggest(request)
    # The provider was protected, so the legacy allowed-read counter is zero;
    # the experiment deliberately applies the stronger attempted-read audit.
    assert guard.audit.action_selection_reads == 0
    with pytest.raises(AssertionError, match="access attempt"):
        experiment._teacher_audit_payload(guard)


def test_student_scoring_guard_requires_the_head_term() -> None:
    action = (1, -1, -1)
    missing = SimpleNamespace(
        decision_id="missing",
        chosen_action=action,
        candidates=(SimpleNamespace(action=action, terms=()),),
    )
    with pytest.raises(AssertionError, match="scoring contract"):
        experiment._assert_student_scoring((missing,))

    present = SimpleNamespace(
        decision_id="present",
        chosen_action=action,
        candidates=(
            SimpleNamespace(
                action=action,
                terms=(("student_policy", 0.2, "learned_value", True),),
            ),
        ),
    )
    audit = experiment._assert_student_scoring((present,), require_nonzero=True)
    assert audit["nonzero_terms"] == 1


def test_head_off_counterfactual_matches_risk_arbiter_positional_tie_break() -> None:
    # CandidateTrace serializes actions as (index, x, y), but the runtime
    # arbiter deliberately tie-breaks on (index, y, x).
    candidates = (
        SimpleNamespace(
            action=(5, 9, 1),
            score=0.2,
            risk=0.1,
            terms=(("student_policy", 0.2, "learned_value", True),),
        ),
        SimpleNamespace(
            action=(5, 1, 9),
            score=0.2,
            risk=0.1,
            terms=(("student_policy", 0.2, "learned_value", True),),
        ),
    )

    chosen = experiment._without_student_choice(
        SimpleNamespace(candidates=candidates),
        risk_limit=0.7,
    )

    assert chosen == (5, 9, 1)


def test_output_and_json_writes_are_collision_safe(tmp_path: Path) -> None:
    requested = tmp_path / "run"
    assert experiment._exclusive_output_directory(requested, "unused") == requested
    with pytest.raises(FileExistsError):
        experiment._exclusive_output_directory(requested, "unused")

    target = requested / "payload.json"
    experiment._write_json_exclusive(target, {"ok": True})
    with pytest.raises(FileExistsError):
        experiment._write_json_exclusive(target, {"ok": False})


def test_arc_environment_receives_the_explicit_paired_seed(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _Environment:
        def close(self) -> None:
            calls["closed"] = True

    class _Arcade:
        def make(self, game: str, **kwargs):
            calls["game"] = game
            calls["make_kwargs"] = kwargs
            return _Environment()

    sentinel = object()

    def fake_run_episode(environment, agent, **kwargs):
        calls["environment"] = environment
        calls["agent"] = agent
        calls["run_kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(experiment, "run_episode", fake_run_episode)
    agent = object()
    result = experiment._run_seeded_arc_episode(
        "tr87",
        agent,
        arcade=_Arcade(),
        environment_seed=123456,
        max_steps=17,
    )

    assert result is sentinel
    assert calls["make_kwargs"] == {"seed": 123456, "render_mode": None}
    assert calls["run_kwargs"]["max_steps"] == 17
    assert calls["closed"] is True


def test_exact_trajectory_run_is_required_and_defaults_match_gate(tmp_path: Path) -> None:
    route = _route(tmp_path)
    assert len(route) == 2
    with pytest.raises(FileNotFoundError, match="no fallback run"):
        experiment._load_trusted(tmp_path, "route-game", 1)

    args = experiment._parser().parse_args([])
    assert args.episodes == 3
    assert args.max_steps == 500
    assert tuple(name for name, _distilled in experiment.ARM_SPECS) == (
        "teacher_distilled_stack",
        "undistilled_stack",
    )
