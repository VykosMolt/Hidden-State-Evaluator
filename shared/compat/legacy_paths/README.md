# Legacy Path Map

The old symlinks were removed so the tree has a single canonical filesystem
shape. Use these replacements when replaying historical commands.

- `checkpoints_pairwise/` -> `rpe/checkpoints/evaluator/`
- `checkpoints_running/` -> `artifacts/checkpoints/running/`
- `environment_files/` -> `data/arc_agi3/environment_files/`
- `solved_sequences/` -> `data/trajectories/solved_sequences/`
- `trusted_trajs/` -> `data/trajectories/trusted_trajs/`

Prefer the canonical `artifacts/` and `data/` paths for new commands.
