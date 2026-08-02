"""Four-arm causal attribution for the state-conditioned student stack.

The v2/v3/v4 retention experiments compare a fully distilled stack against an
undistilled baseline, so broad behavioral differences belong to the distilled
*stack* (prior + dynamics + replay + head together).  This experiment
decomposes that attribution with four persistent-agent arms under the paired
seeded-episode protocol of ``run_hs_v2_student_retention_v1``:

- ``full_distilled_stack``: everything distilled, head term live (the v4 arm);
- ``distilled_head_off``: everything distilled identically, but the head's
  ``runtime_weight`` is zero, so its score term is constitutively 0.0 while
  the distilled prior/dynamics/replay remain active — isolates how much of
  the full arm's behavior needs head *pressure* rather than stack priors;
- ``head_only_distilled``: a donor agent is distilled offline and only its
  trained head object is transplanted into an otherwise fresh agent —
  isolates how much the head alone carries without the distilled stack;
- ``undistilled_stack``: nothing distilled.

All action selection remains teacher-free in every arm (guard-audited); the
donor's offline read is reported separately.  Metrics, alignment rules, and
seeding are imported from the retention runner so results are directly
comparable with the v4 artifact.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_hs_v2_student_retention_v1 as retention

from hunter_seeker_v2.adapters import ArcActionAdapter
from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import RuntimeMode
from hunter_seeker_v2.run_arc import make_arcade
from hunter_seeker_v2.student import (
    StateConditionedStudentPolicy,
    StudentPolicyConfig,
)
from hunter_seeker_v2.teacher import TeacherAccessGuard, TeacherQuery, TrajectoryTeacher

EXPERIMENT_ID = "hs_v2_student_attribution_v1"
SCHEMA_VERSION = 1
ARM_NAMES = (
    "full_distilled_stack",
    "distilled_head_off",
    "head_only_distilled",
    "undistilled_stack",
)


def _weightless_head_fingerprint(head: StateConditionedStudentPolicy) -> str:
    state = head.state_dict()
    state.pop("config", None)
    return retention._fingerprint(state)


def _build_arm(
    arm_name: str,
    *,
    game: str,
    route: retention.TrustedTrajectory,
    config: Any,
    student_config: StudentPolicyConfig,
) -> dict[str, Any]:
    def fresh_teacher() -> TrajectoryTeacher:
        if retention._sha256_file(route.path) != route.sha256:
            raise RuntimeError(
                f"trusted trajectory changed before {arm_name} construction"
            )
        return TrajectoryTeacher.from_npz(
            str(route.path),
            task_id=game,
            teacher_id=f"trusted:{game}:{route.sha256[:12]}",
        )

    def fresh_agent(
        head: StateConditionedStudentPolicy,
        guard: TeacherAccessGuard,
    ) -> CompactHunterSeeker:
        adapter = ArcActionAdapter()
        return CompactHunterSeeker(
            config=config,
            click_action_index=adapter.click_action_index(),
            safe_action_provider=adapter.safe_action_indices,
            teacher=guard,
            student_policy=head,
        )

    donor_audit: dict[str, Any] | None = None
    if arm_name == "full_distilled_stack":
        head = StateConditionedStudentPolicy(student_config)
        initial_fingerprint = _weightless_head_fingerprint(head)
        guard = TeacherAccessGuard(fresh_teacher(), mode=RuntimeMode.STUDENT)
        agent = fresh_agent(head, guard)
        distilled = agent.distill_teacher(
            TeacherQuery(task_id=game, limit=max(len(route), 1))
        )
    elif arm_name == "distilled_head_off":
        head = StateConditionedStudentPolicy(
            dataclasses.replace(student_config, runtime_weight=0.0)
        )
        initial_fingerprint = _weightless_head_fingerprint(head)
        guard = TeacherAccessGuard(fresh_teacher(), mode=RuntimeMode.STUDENT)
        agent = fresh_agent(head, guard)
        distilled = agent.distill_teacher(
            TeacherQuery(task_id=game, limit=max(len(route), 1))
        )
    elif arm_name == "head_only_distilled":
        head = StateConditionedStudentPolicy(student_config)
        initial_fingerprint = _weightless_head_fingerprint(head)
        donor_guard = TeacherAccessGuard(fresh_teacher(), mode=RuntimeMode.STUDENT)
        donor = fresh_agent(head, donor_guard)
        distilled = donor.distill_teacher(
            TeacherQuery(task_id=game, limit=max(len(route), 1))
        )
        if donor.student_policy is not head:
            raise AssertionError("donor did not train the transplant head")
        donor_audit = retention._teacher_audit_payload(donor_guard)
        # Transplant: only the trained head object crosses; the runtime agent
        # is otherwise fresh (empty graph/evidence/prior/dynamics/replay).
        guard = TeacherAccessGuard(fresh_teacher(), mode=RuntimeMode.STUDENT)
        agent = fresh_agent(head, guard)
        if len(agent.graph) != 0 or len(agent.evidence) != 0:
            raise AssertionError("head-only recipient is not fresh")
        if len(agent.buffer) != 0:
            raise AssertionError("head-only recipient inherited replay")
    elif arm_name == "undistilled_stack":
        head = StateConditionedStudentPolicy(student_config)
        initial_fingerprint = _weightless_head_fingerprint(head)
        guard = TeacherAccessGuard(fresh_teacher(), mode=RuntimeMode.STUDENT)
        agent = fresh_agent(head, guard)
        distilled = 0
    else:  # pragma: no cover - defended by ARM_NAMES
        raise ValueError(f"unknown arm {arm_name!r}")

    expects_distillation = arm_name != "undistilled_stack"
    if expects_distillation:
        if distilled != len(route):
            raise AssertionError(
                f"{arm_name}: distilled {distilled} examples, expected {len(route)}"
            )
        if agent.student_policy.teacher_updates <= 0:
            raise AssertionError(f"{arm_name}: head received no teacher updates")
    elif agent.student_policy.teacher_updates != 0:
        raise AssertionError("undistilled head received teacher updates")
    if agent.student_policy is not head:
        raise AssertionError(f"{arm_name}: agent did not retain the explicit head")
    return {
        "agent": agent,
        "guard": guard,
        "distilled_examples": distilled,
        "initial_head_fingerprint": initial_fingerprint,
        "donor_audit": donor_audit,
        # The head-off arm's term is constitutively zero; every other
        # distilled arm must show live nonzero student scoring.
        "require_nonzero_scoring": arm_name
        in ("full_distilled_stack", "head_only_distilled"),
        "pre_schedule_student_summary": agent.student_policy.summary(),
    }


def _execution_order(seed: int, episode: int) -> tuple[str, ...]:
    rotation = (int(seed) + int(episode)) % len(ARM_NAMES)
    return ARM_NAMES[rotation:] + ARM_NAMES[:rotation]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=["ls20", "tr87", "wa30"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=retention.DEFAULT_TRAJECTORY_ROOT,
    )
    parser.add_argument("--out-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    retention._validate_args(
        argparse.Namespace(
            games=args.games,
            seeds=args.seeds,
            episodes=args.episodes,
            max_steps=args.max_steps,
            run_index=args.run_index,
        )
    )
    script_path = Path(__file__).resolve()
    routes = {
        game: retention._load_trusted(args.trajectory_root, game, args.run_index)
        for game in args.games
    }
    retention_main_src = retention._source_manifest(Path(retention.__file__))
    from hunter_seeker_v2.contracts import AgentConfig, PolicyConfig

    # Same agent configuration as the retention protocol.
    common_config = AgentConfig(
        runtime_mode=RuntimeMode.STUDENT,
        seed=0,
        policy=PolicyConfig(exploration_epsilon=0.0),
    )
    student_config = StudentPolicyConfig()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "_"
        + uuid4().hex[:12]
    )
    output = (
        args.out_root
        if args.out_root is not None
        else retention.DEFAULT_ARTIFACT_PARENT / f"student_attribution_{run_id}"
    )
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "started_at_utc": retention._utc_now(),
        "args": {
            "games": list(args.games),
            "seeds": [int(seed) for seed in args.seeds],
            "episodes": int(args.episodes),
            "max_steps": int(args.max_steps),
            "run_index": int(args.run_index),
        },
        "arms": list(ARM_NAMES),
        "design": {
            "pairing": "identical per-(seed,episode) environment seeds across arms",
            "arm_rotation": "execution order rotates by (seed+episode) mod 4",
            "head_only_transplant": (
                "donor distills offline; only the trained head object moves to a "
                "fresh agent (empty graph/evidence/prior/dynamics/replay)"
            ),
            "head_off_mechanism": "StudentPolicyConfig.runtime_weight=0.0",
            "comparability": (
                "metrics, alignment and seeding imported from "
                "run_hs_v2_student_retention_v1 (v4 protocol)"
            ),
        },
        "agent_config_sha256": retention._fingerprint(common_config),
        "student_config": dataclasses.asdict(student_config),
        "attribution_script_sha256": retention._sha256_file(script_path),
        "retention_module_manifest": retention_main_src,
        "trajectories": {
            game: retention._trajectory_manifest(route)
            for game, route in routes.items()
        },
        "package_versions": retention._package_versions(),
    }
    retention._write_json_exclusive(output / "run_started.json", provenance)
    rows: list[dict[str, Any]] = []
    arcade = make_arcade()
    rows_path = output / "rows.jsonl"
    with rows_path.open("x", encoding="utf-8") as rows_file:
        for seed in args.seeds:
            config = dataclasses.replace(common_config, seed=int(seed))
            for game in args.games:
                route = routes[game]
                arms = {
                    arm_name: _build_arm(
                        arm_name,
                        game=game,
                        route=route,
                        config=config,
                        student_config=student_config,
                    )
                    for arm_name in ARM_NAMES
                }
                fingerprints = {
                    arm["initial_head_fingerprint"] for arm in arms.values()
                }
                if len(fingerprints) != 1:
                    raise AssertionError(
                        "arms did not begin with identical (weightless) heads"
                    )
                for episode in range(1, args.episodes + 1):
                    paired_seed = retention._episode_seed(seed, episode)
                    paired_initial_signature: dict[str, Any] | None = None
                    for position, arm_name in enumerate(
                        _execution_order(seed, episode),
                        start=1,
                    ):
                        arm = arms[arm_name]
                        agent = arm["agent"]
                        applied = retention._set_episode_seed(seed, episode)
                        if applied != paired_seed:
                            raise AssertionError("paired episode seed drifted")
                        decision_offset = len(agent.diagnostics.decisions)
                        started = time.monotonic()
                        result = retention._run_seeded_arc_episode(
                            game,
                            agent,
                            arcade=arcade,
                            environment_seed=paired_seed,
                            max_steps=args.max_steps,
                        )
                        elapsed = time.monotonic() - started
                        decisions = agent.diagnostics.decisions[decision_offset:]
                        scoring_audit = retention._assert_student_scoring(
                            decisions,
                            require_nonzero=arm["require_nonzero_scoring"],
                        )
                        initial_signature = retention._observation_signature(
                            result.initial_observation
                        )
                        if paired_initial_signature is None:
                            paired_initial_signature = initial_signature
                        elif initial_signature != paired_initial_signature:
                            raise AssertionError(
                                "paired seeded environments diverged for "
                                f"{game} seed={seed} episode={episode}"
                            )
                        metrics, _trace = retention._analyze_episode(
                            result,
                            route,
                            decisions,
                            risk_limit=float(config.policy.risk_limit),
                        )
                        row = {
                            "experiment": EXPERIMENT_ID,
                            "game": game,
                            "seed": int(seed),
                            "episode": int(episode),
                            "arm": arm_name,
                            "arm_execution_position": position,
                            "paired_episode_seed": paired_seed,
                            "elapsed_seconds": elapsed,
                            "distilled_examples": arm["distilled_examples"],
                            "student_scoring_audit": scoring_audit,
                            "teacher_audit": retention._teacher_audit_payload(
                                arm["guard"]
                            ),
                            "donor_teacher_audit": arm["donor_audit"],
                            "student_summary": agent.student_policy.summary(),
                            **metrics,
                        }
                        rows_file.write(retention._stable_json(row) + "\n")
                        rows_file.flush()
                        rows.append(row)
    for row in rows:
        audit = row["teacher_audit"]
        if audit["summary"]["action_selection_reads"] != 0:
            raise AssertionError("an arm performed teacher action-selection reads")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "run_id": run_id,
        "completed_at_utc": retention._utc_now(),
        "provenance_file": "run_started.json",
        "rows": rows,
        "arm_totals": {
            arm_name: {
                "level_completions": sum(
                    row["level_completion_transitions"]
                    for row in rows
                    if row["arm"] == arm_name
                ),
                "strict_prefix_steps": sum(
                    row["strict_route_prefix_state_action_successor_steps"]
                    for row in rows
                    if row["arm"] == arm_name
                ),
                "ordered_matches": sum(
                    row["ordered_state_action_successor_matches"]
                    for row in rows
                    if row["arm"] == arm_name
                ),
            }
            for arm_name in ARM_NAMES
        },
    }
    retention._write_json_exclusive(output / "summary.json", summary)
    print(retention._stable_json(summary["arm_totals"]))
    print(f"artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
