"""Named experiment: checkpoint-carry of assisted knowledge into teacher-free play.

Every mechanism validated so far (student head, hypothesis promotion,
exogenous masking, fitted bandwidths) has been exercised in replay-adjacent
settings.  This experiment asks whether they compose into new capability:
an agent acquires knowledge under compat-assisted information conditions
(offline distillation plus one assisted episode whose completions fire
verified hypothesis promotions), is checkpointed, and the checkpoint is
loaded into an agent with **no teacher attached** — teacher-free by
construction, not merely by audit — which then plays fresh episodes.

Arms per (game, seed), paired seeded episodes:

- ``carried``: fresh agent + phase-A checkpoint, no teacher object exists;
- ``fresh``: identical agent, no checkpoint, no teacher object.

Both phases share one AgentConfig (``compat_assisted``, epsilon zero)
because checkpoint loading validates full config equality; the teacher-free
property of phase B rests on the absence of any teacher, which is stronger
than a mode flag.  Games: ls20 (run0 route) and tr87 (merged runs) — the
two games where assisted play completed levels and promoted hypotheses.
wa30 is excluded: assisted play never completed it, so there is no
acquired knowledge to carry.

Readouts: carried-vs-fresh level completions and progress; whether carried
play reaches states beyond the walked levels (ls20 level 2 is unwalked —
the run0 route covers level 1 only); hypothesis and student summaries
before and after every episode.
"""

from __future__ import annotations

import argparse
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
from hunter_seeker_v2.contracts import AgentConfig, PolicyConfig, RuntimeMode
from hunter_seeker_v2.persistence import load_checkpoint, save_checkpoint
from hunter_seeker_v2.run_arc import make_arcade
from hunter_seeker_v2.teacher import TeacherQuery, TrajectoryTeacher

EXPERIMENT_ID = "hs_v2_checkpoint_carry_v1"
SCHEMA_VERSION = 1
TRAJECTORY_ROOT = PROJECT_ROOT / "data" / "trajectories" / "trusted_topology_trio_20260513"
GAME_ROUTES = {
    # The assisted-parity recipe: ls20 needs run0 (starts at reset); tr87
    # benefits from merging every recorded run.
    "ls20": ("ls20_run0_traj.npz",),
    "tr87": ("tr87_run0_traj.npz", "tr87_run1_traj.npz"),
}


def _merged_teacher(game: str) -> tuple[TrajectoryTeacher, list[dict[str, Any]]]:
    examples: list = []
    manifest: list[dict[str, Any]] = []
    for name in GAME_ROUTES[game]:
        path = TRAJECTORY_ROOT / name
        teacher = TrajectoryTeacher.from_npz(str(path), task_id=game)
        examples.extend(teacher._examples)
        manifest.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": retention._sha256_file(path),
                "examples": len(teacher._examples),
            }
        )
    return (
        TrajectoryTeacher(tuple(examples), teacher_id=f"trusted:{game}"),
        manifest,
    )


def _build_agent(
    config: AgentConfig,
    *,
    teacher: TrajectoryTeacher | None,
) -> CompactHunterSeeker:
    adapter = ArcActionAdapter()
    return CompactHunterSeeker(
        config=config,
        click_action_index=adapter.click_action_index(),
        safe_action_provider=adapter.safe_action_indices,
        teacher=teacher,
    )


def _knowledge_summary(agent: CompactHunterSeeker) -> dict[str, Any]:
    summary = agent.measurement_summary()
    return {
        "hypotheses": summary.get("hypotheses"),
        "student_policy": summary.get("student_policy"),
        "graph_nodes": len(agent.graph),
        "evidence_records": len(agent.evidence),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=["ls20", "tr87"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--reset-head-before-distill",
        action="store_true",
        help=(
            "re-distill the carried head from a clean state after the assisted "
            "episode (isolates online-update perturbation of the head)"
        ),
    )
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args(argv)
    unknown = sorted(set(args.games) - set(GAME_ROUTES))
    if unknown:
        raise SystemExit(f"no assisted knowledge recipe for games: {unknown}")
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "_"
        + uuid4().hex[:12]
    )
    output = (
        args.out_root
        if args.out_root is not None
        else retention.DEFAULT_ARTIFACT_PARENT / f"checkpoint_carry_{run_id}"
    )
    output.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output / "checkpoints"
    checkpoints_dir.mkdir()
    teachers = {game: _merged_teacher(game) for game in args.games}
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "started_at_utc": retention._utc_now(),
        "args": {
            "games": list(args.games),
            "seeds": [int(seed) for seed in args.seeds],
            "episodes": int(args.episodes),
            "max_steps": int(args.max_steps),
        },
        "design": {
            "phase_a": (
                "offline distill_teacher over the merged trusted route, then "
                "one compat-assisted episode; completions fire hypothesis "
                "promotion; checkpoint saved afterwards"
            ),
            "phase_b_teacher_free_by_construction": (
                "phase-B agents are built with teacher=None; no teacher "
                "object exists to read"
            ),
            "shared_config_reason": (
                "checkpoint loading validates full config equality, so both "
                "phases run the same compat_assisted config; phase-B teacher "
                "absence is structural"
            ),
            "pairing": "identical per-(seed,episode) environment seeds across arms",
            "excluded_games": {
                "wa30": "assisted play never completed it; nothing to carry"
            },
        },
        "trajectories": {
            game: manifest for game, (_teacher, manifest) in teachers.items()
        },
        "script_sha256": retention._sha256_file(Path(__file__).resolve()),
        "source_manifest": retention._source_manifest(Path(__file__).resolve()),
        "package_versions": retention._package_versions(),
    }
    retention._write_json_exclusive(output / "run_started.json", provenance)
    arcade = make_arcade()
    rows: list[dict[str, Any]] = []
    rows_path = output / "rows.jsonl"
    with rows_path.open("x", encoding="utf-8") as rows_file:

        def emit(row: dict[str, Any]) -> None:
            rows_file.write(retention._stable_json(row) + "\n")
            rows_file.flush()
            rows.append(row)

        for seed in args.seeds:
            config = AgentConfig(
                runtime_mode=RuntimeMode.COMPAT_ASSISTED,
                seed=int(seed),
                policy=PolicyConfig(exploration_epsilon=0.0),
            )
            for game in args.games:
                teacher, _manifest = _merged_teacher(game)
                # --- Phase A: acquire knowledge under assisted conditions.
                # Ordering matters: the assisted episode runs FIRST with an
                # untrained head so the live teacher drives completions that
                # promote hypotheses and populate graph/evidence.  Distilling
                # first trains a head that fights the teacher on games it fits
                # imperfectly (tr87: a distilled-first head overrode the
                # teacher and dropped acquisition from 3 levels to 0), which
                # would leave nothing to carry.  The head is distilled after,
                # so the checkpoint carries both knowledge sources.
                donor = _build_agent(config, teacher=teacher)
                retention._set_episode_seed(seed, 0)
                started = time.monotonic()
                assisted = retention._run_seeded_arc_episode(
                    game,
                    donor,
                    arcade=arcade,
                    environment_seed=retention._episode_seed(seed, 0),
                    max_steps=args.max_steps,
                )
                assisted_elapsed = time.monotonic() - started
                if args.reset_head_before_distill:
                    # The assisted episode's online updates perturb the head
                    # relative to a pure distillation; re-distilling into a
                    # clean head keeps the graph/hypotheses/evidence acquired
                    # during the episode while giving the carried head the
                    # same pristine state that retention v5 completed from.
                    donor.student_policy = type(donor.student_policy)(
                        donor.student_policy.config
                    )
                distilled = donor.distill_teacher(
                    TeacherQuery(task_id=game, limit=100_000)
                )
                phase_a_row = {
                    "phase": "A_assisted_acquisition",
                    "game": game,
                    "seed": int(seed),
                    "distilled_examples": distilled,
                    "assisted_first": True,
                    "steps": assisted.steps,
                    "progress": float(assisted.final_observation.progress),
                    "levels_completed": sum(
                        1
                        for transition in assisted.transitions
                        if transition.outcome.progress_delta > 0
                    ),
                    "boundary": assisted.final_outcome.boundary.value,
                    "elapsed_seconds": assisted_elapsed,
                    "knowledge": _knowledge_summary(donor),
                }
                emit(phase_a_row)
                checkpoint_path = checkpoints_dir / f"{game}_seed{seed}.json"
                save_checkpoint(donor, str(checkpoint_path))
                phase_a_row_sha = retention._sha256_file(checkpoint_path)

                # --- Phase B: paired carried-vs-fresh teacher-free episodes.
                arms: dict[str, CompactHunterSeeker] = {}
                for arm_name in ("carried", "fresh"):
                    agent = _build_agent(config, teacher=None)
                    if agent.teacher is not None:
                        raise AssertionError("phase-B agent must have no teacher")
                    if arm_name == "carried":
                        load_checkpoint(agent, str(checkpoint_path))
                        if agent.teacher is not None:
                            raise AssertionError(
                                "loading a checkpoint must not attach a teacher"
                            )
                        if agent.student_policy.teacher_updates <= 0:
                            raise AssertionError(
                                "carried arm lost its distilled student head"
                            )
                    arms[arm_name] = agent
                for episode in range(1, args.episodes + 1):
                    order = (
                        ("carried", "fresh")
                        if (seed + episode) % 2
                        else ("fresh", "carried")
                    )
                    for arm_name in order:
                        agent = arms[arm_name]
                        retention._set_episode_seed(seed, episode)
                        started = time.monotonic()
                        result = retention._run_seeded_arc_episode(
                            game,
                            agent,
                            arcade=arcade,
                            environment_seed=retention._episode_seed(seed, episode),
                            max_steps=args.max_steps,
                        )
                        emit(
                            {
                                "phase": "B_teacher_free",
                                "game": game,
                                "seed": int(seed),
                                "episode": int(episode),
                                "arm": arm_name,
                                "checkpoint_sha256": (
                                    phase_a_row_sha
                                    if arm_name == "carried"
                                    else None
                                ),
                                "steps": result.steps,
                                "progress": float(
                                    result.final_observation.progress
                                ),
                                "final_stage": int(result.final_observation.stage),
                                "levels_completed": sum(
                                    1
                                    for transition in result.transitions
                                    if transition.outcome.progress_delta > 0
                                ),
                                "boundary": result.final_outcome.boundary.value,
                                "elapsed_seconds": time.monotonic() - started,
                                "knowledge": _knowledge_summary(agent),
                            }
                        )
    arm_totals: dict[str, dict[str, float]] = {}
    for arm_name in ("carried", "fresh"):
        arm_rows = [
            row
            for row in rows
            if row.get("phase") == "B_teacher_free" and row.get("arm") == arm_name
        ]
        arm_totals[arm_name] = {
            "episodes": len(arm_rows),
            "levels_completed": sum(row["levels_completed"] for row in arm_rows),
            "max_progress": max(
                (row["progress"] for row in arm_rows),
                default=0.0,
            ),
            "mean_steps": (
                sum(row["steps"] for row in arm_rows) / len(arm_rows)
                if arm_rows
                else 0.0
            ),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_ID,
        "run_id": run_id,
        "completed_at_utc": retention._utc_now(),
        "provenance_file": "run_started.json",
        "rows": rows,
        "phase_b_arm_totals": arm_totals,
    }
    retention._write_json_exclusive(output / "summary.json", summary)
    print(retention._stable_json(arm_totals))
    print(f"artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
