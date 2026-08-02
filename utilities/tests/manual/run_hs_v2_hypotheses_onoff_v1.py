"""Named experiment: relational goal hypotheses on/off on template games.

Matched action budget, identical seed and configuration apart from
``HypothesisConfig.enabled``.  Games are chosen so the engine's generic
relation templates plausibly match the displayed goal (sc25 copy-grid,
lf52 uniformity, m0r0/ka59 mirror) plus tr87 as a template-mismatch
control (its true goal is legend translation, which no template covers).

Usage:

    venv/bin/python utilities/tests/manual/run_hs_v2_hypotheses_onoff_v1.py \
        --games sc25 lf52 m0r0 ka59 tr87 --seeds 0 1 --episodes 2
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from hunter_seeker_v2.adapters import ArcActionAdapter
from hunter_seeker_v2.agent import CompactHunterSeeker
from hunter_seeker_v2.contracts import AgentConfig, HypothesisConfig
from hunter_seeker_v2.run_arc import PROJECT_ROOT, make_arcade, run_arc_game
from hs_v2_experiment_provenance import (
    prepare_summary_target,
    start_provenance,
    utc_now,
    write_summary,
)


def _build_agent(seed: int, enabled: bool) -> CompactHunterSeeker:
    actions = ArcActionAdapter()
    return CompactHunterSeeker(
        config=AgentConfig(
            seed=seed,
            hypotheses=HypothesisConfig(enabled=enabled),
        ),
        click_action_index=actions.click_action_index(),
        safe_action_provider=actions.safe_action_indices,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games",
        nargs="+",
        default=["sc25", "lf52", "m0r0", "ka59", "tr87"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
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
            / f"hypotheses_onoff_{date.today().strftime('%Y%m%d')}"
        ),
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    target = prepare_summary_target(out_root, overwrite=args.overwrite)
    provenance = start_provenance(
        args=args,
        project_root=PROJECT_ROOT,
        script_path=Path(__file__),
    )
    arcade = make_arcade()
    rows: list[dict] = []
    effective_configs: dict[str, AgentConfig] = {}
    for seed in args.seeds:
        for game in args.games:
            for enabled in (True, False):
                agent = _build_agent(seed, enabled)
                arm = "hyp_on" if enabled else "hyp_off"
                effective_configs[f"seed={seed}|arm={arm}"] = agent.config
                for episode in range(1, max(1, args.episodes) + 1):
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
                        "game": game,
                        "seed": seed,
                        "arm": arm,
                        "episode": episode,
                        "steps": result.steps,
                        "boundary": result.final_outcome.boundary.value,
                        "progress": result.final_observation.progress,
                        "stage": result.final_observation.stage,
                        "started_at_utc": started_at,
                        "completed_at_utc": completed_at,
                        "duration_seconds": duration,
                        "seconds": round(duration, 1),
                        "hypotheses": summary.get("hypotheses"),
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "experiment": "hs_v2_hypotheses_onoff_v1",
        "max_steps": args.max_steps,
        "rows": rows,
        "provenance": provenance.finish(
            effective_agent_configs=effective_configs,
            ordering={
                "loop_nesting": ["seed", "game", "arm", "episode"],
                "seed_order": args.seeds,
                "game_order": args.games,
                "arm_order": ["hyp_on", "hyp_off"],
                "episode_order": list(range(1, max(1, args.episodes) + 1)),
            },
        ),
    }
    write_summary(target, payload, overwrite=args.overwrite)
    print(f"wrote {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
