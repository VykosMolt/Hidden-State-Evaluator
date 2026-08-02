"""Named experiment: v2 compat-assisted trio under v1's information conditions.

The legacy strict topology-trio baseline (1/2/0 levels) ran with trusted
trajectories loaded; its clears followed the trusted continuation sequence.
This experiment gives v2 the same information through its audited teacher
boundary: `TrajectoryTeacher` over the same
`data/trajectories/trusted_topology_trio_20260513` NPZ files, agent in
``compat_assisted`` mode, epsilon 0, 500-step budget as in the legacy runs.

Ledger gates addressed: CAPABILITY_PARITY.md remaining-work items 1 and 3.

Usage:

    venv/bin/python utilities/tests/manual/run_hs_v2_assisted_trio_v1.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from hunter_seeker_v2.adapters import ArcActionAdapter
from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import AgentConfig, PolicyConfig, RuntimeMode
from hunter_seeker_v2.run_arc import PROJECT_ROOT, make_arcade, run_arc_game
from hunter_seeker_v2.teacher import TrajectoryTeacher
from hs_v2_experiment_provenance import (
    prepare_summary_target,
    start_provenance,
    utc_now,
    write_summary,
)

TRAJECTORY_ROOT = PROJECT_ROOT / "data" / "trajectories" / "trusted_topology_trio_20260513"


class CountingTeacher:
    """Transparent proxy recording hit/abstention counts."""

    def __init__(self, inner: TrajectoryTeacher) -> None:
        self.inner = inner
        self.teacher_id = inner.teacher_id
        self.hits = 0
        self.abstentions = 0

    def suggest(self, request):
        response = self.inner.suggest(request)
        if response is None or response.action is None:
            self.abstentions += 1
        else:
            self.hits += 1
        return response

    def examples(self, query):
        return self.inner.examples(query)


def _trajectory_paths(
    game: str,
    *,
    single_run: bool = False,
    run_index: int | None = None,
) -> list[Path]:
    paths = sorted(TRAJECTORY_ROOT.glob(f"{game}_run*_traj.npz"))
    if not paths:
        raise FileNotFoundError(f"no trusted trajectories for {game}")
    if run_index is not None:
        wanted = [p for p in paths if p.name == f"{game}_run{run_index}_traj.npz"]
        paths = wanted or paths
    elif single_run:
        # One coherent continuation, like the legacy trusted-route runs;
        # merged runs can disagree at a shared state and split confidence.
        paths = [max(paths, key=lambda p: p.stat().st_size)]
    return paths


def _load_teacher(game: str, paths: list[Path]) -> CountingTeacher:
    examples: list = []
    for path in paths:
        teacher = TrajectoryTeacher.from_npz(str(path), task_id=game)
        examples.extend(teacher._examples)
    return CountingTeacher(TrajectoryTeacher(tuple(examples), teacher_id=f"trusted:{game}"))


def _first_frame_diagnostic(path: Path, result) -> dict:
    recorded = np.asarray(np.load(str(path))["frames"][0], dtype=np.int64)
    live = np.asarray(result.initial_observation.frame, dtype=np.int64)
    if recorded.shape != live.shape:
        return {"first_frame_match": False, "reason": "shape"}
    diff = int((recorded != live).sum())
    return {"first_frame_match": diff == 0, "differing_cells": diff}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=["ls20", "tr87", "wa30"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing summary.json",
    )
    parser.add_argument(
        "--out-root",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "reports"
            / "hunter_seeker_v2"
            / f"assisted_trio_{date.today().strftime('%Y%m%d')}"
        ),
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    target = prepare_summary_target(out_root, overwrite=args.overwrite)
    paths_by_game = {
        game: _trajectory_paths(
            game,
            single_run=args.single_run,
            run_index=args.run_index,
        )
        for game in args.games
    }
    provenance = start_provenance(
        args=args,
        project_root=PROJECT_ROOT,
        script_path=Path(__file__),
        input_paths=(
            path
            for game in args.games
            for path in paths_by_game[game]
        ),
    )
    arcade = make_arcade()
    rows: list[dict] = []
    effective_configs: dict[str, AgentConfig] = {}
    for seed in args.seeds:
        for game in args.games:
            selected_paths = paths_by_game[game]
            teacher = _load_teacher(game, selected_paths)
            actions = ArcActionAdapter()
            agent = CompactHunterSeeker(
                config=AgentConfig(
                    runtime_mode=RuntimeMode.COMPAT_ASSISTED,
                    seed=seed,
                    policy=PolicyConfig(exploration_epsilon=0.0),
                ),
                click_action_index=actions.click_action_index(),
                safe_action_provider=actions.safe_action_indices,
                teacher=teacher,
            )
            effective_configs[f"seed={seed}|arm=compat_assisted"] = agent.config
            started_at = utc_now()
            started = time.perf_counter()
            result = run_arc_game(
                game,
                agent,
                arcade=arcade,
                max_steps=args.max_steps,
            )
            completed_at = utc_now()
            duration = round(time.perf_counter() - started, 6)
            summary = agent.measurement_summary()
            row = {
                "hypotheses": summary.get("hypotheses"),
                "game": game,
                "seed": seed,
                "mode": "compat_assisted",
                "steps": result.steps,
                "boundary": result.final_outcome.boundary.value,
                "progress": result.final_observation.progress,
                "stage": result.final_observation.stage,
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "duration_seconds": duration,
                "seconds": round(duration, 1),
                "teacher_hits": teacher.hits,
                "teacher_abstentions": teacher.abstentions,
                "teacher_examples": len(teacher.inner._examples),
                "trajectory_files": [str(path) for path in selected_paths],
                **_first_frame_diagnostic(selected_paths[0], result),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "experiment": "hs_v2_assisted_trio_v1",
        "max_steps": args.max_steps,
        "legacy_reference": {
            "ls20": "1 level, game over step 91",
            "tr87": "2 levels, game over step 192",
            "wa30": "0 levels, game over step 200",
        },
        "rows": rows,
        "provenance": provenance.finish(
            effective_agent_configs=effective_configs,
            ordering={
                "loop_nesting": ["seed", "game", "arm"],
                "seed_order": args.seeds,
                "game_order": args.games,
                "arm_order": ["compat_assisted"],
            },
        ),
    }
    write_summary(target, payload, overwrite=args.overwrite)
    print(f"wrote {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
