# Artifacts

Run records that belong to no single publication line.

- `logs/`: training and evaluation logs (`branch_training_v1`, `branch_training_v2`,
  `corecontent_v2`, `mpn_s0`, and the dated 2026-06 run logs).
- `ops/`: `run_downstream_v3.sh`, `thermal_guard.sh`.
- `cleanup_*`, `reports_large_*`: the 2026-07-02 storage cleanup audit, decision
  table and deletion log.

Reports, checkpoints and models moved to `opi/`, `rpe/` and `shared/` on 2026-09-15
(see `../MOVED_PATHS.md`). The Hunter-Seeker ARC data that used to live here
(`trajectories/`, `debug_frames/`, `recordings/`, `event_dumps/`) was purged 2026-09-20.
