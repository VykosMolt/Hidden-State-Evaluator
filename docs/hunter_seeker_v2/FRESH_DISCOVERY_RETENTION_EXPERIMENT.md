# TU93 fresh-discovery retention experiment

`hs_v2_tu93_fresh_retention_v1` is the authoritative Sol-frozen experiment
for whether within-episode online learning is retained across fresh attempts.
Its runner is
`utilities/tests/manual/run_hs_v2_tu93_fresh_retention_v1.py`.

The earlier
`run_hs_v2_fresh_discovery_v1.py` four-arm runner and its artifacts are retained
as historical/intermediate evidence. They contain acquisition, graph-retained,
graph-off, and goal-promotion controls and must not be used for the retention
claim. The earlier graph-specific runner is likewise historical for this
question. Neither is imported or called by the authoritative runner.

## Frozen contract

Only local/offline TU93 variant `0768757b` is eligible. The runner verifies the
metadata and source before constructing an environment:

| Input | Required SHA-256 |
| --- | --- |
| `data/arc_agi3/environment_files/tu93/0768757b/tu93.py` | `80e41888f9f7b1a0c03e02c0aff3814e0fd68eb5b35ef22bb3649c87fc60a23f` |
| `data/arc_agi3/environment_files/tu93/0768757b/metadata.json` | `ad29d072e6977cb914b729c0f461157d971f4182656adb0f811c77bf14faa20f` |

Environment construction uses the exact versioned game ID in ARC offline
mode. Missing dependencies, a changed source or metadata file, or a variant
selection failure blocks the run; there is no download/latest-version
fallback.

There are exactly two arms:

* `persistent_fresh`: one newly constructed autonomous agent is retained for
  all of a replicate's attempts;
* `episode_isolated`: a new constructor-baseline autonomous agent is built
  before each attempt.

Both arms use `RuntimeMode.AUTONOMOUS`, the default `AgentConfig` with the
replicate seed and executable models disabled, the default `GridFeature`
backend, ordinary `RiskArbiter`, `teacher=None`, and online learning enabled.
Neither arm loads a checkpoint, uses a route or acquisition helper, reads a
teacher, enters compatibility mode, consumes task-specific artifacts, or
receives a graph-specific causal intervention.

The full design has 32 paired replicates with agent seeds `0..31`, at most 20
attempts per arm, and at most 50 committed actions per attempt. An arm stops
at its first stage-1 completion. The primary trace records no action whose
before-state is stage 2. Each pair uses the same deterministic environment
seed and requires matching initial observation/state fingerprints. Arm order
alternates according to a preregistered SHA-256 schedule, and Python, NumPy,
agent, and environment seed derivation is recorded in every pair.

## Evidence schema and verdict

Each attempt records the full decisions and decision metadata, committed
transitions, stage/boundary information, completion lineage, pair/attempt IDs,
initial and final state-store fingerprints, and runtime counts/fingerprints for
graph, evidence, replay, hypotheses, student, learner, model, and diagnostics.
Freshness/leakage audits inspect actual agent state and the authoritative
runner's import surface. Invalid and aborted attempts remain explicit records.
Source, config, package, and exact-environment manifests are included in the
summary and provenance artifact.

The causal gate is recomputed from raw pair and attempt rows at finalization.
It requires exactly the unique pair IDs `replicate-00` through `replicate-31`,
agent seeds `0..31`, exactly 20 scheduled attempts per pair, unique
attempt/decision/transition IDs, stage-1 starts, non-empty bootstrap/action and
complete before/after state-store evidence, expected schedules, and complete
environment-step and outcome/lineage consistency. Persistent and isolated
completion values are derived from transition outcomes; summary flags are
checked for consistency but are never used as causal evidence. Exact source,
config, package, runner, schema, and environment manifests are independently
verified. A causal result additionally requires an opaque process-local live-run
token created only after authoritative exact-environment setup; no serialized
summary or trace can recreate it, so non-authoritative helpers and direct or
synthetic gate calls remain descriptive-only. Every causal artifact carries an
exact pinned manifest, `authoritative: true`, and a complete finalization
marker.

Freshness verification reports supported static/source and constructor/runtime
boundary checks. This runner has no checkpoint, route, teacher, or
graph-intervention path, so forbidden-call counters are explicitly marked
not-applicable rather than treated as unsupported passes. If either supported
boundary check fails, the verdict is descriptive-only. Bootstrap, initial
observation, begin-run, and cleanup failures keep their attempted evidence in
the blocked artifact; blocked aggregates are recomputed from raw rows.
Publication accepts only a new direct child of
`artifacts/reports/hunter_seeker_v2`, writes machine-readable
`historical_status.json`/`stale` metadata and `COMPLETION_MARKER.json`, and
never replaces or moves an existing live output.

The aggregate includes a preregistered one-sided exact McNemar calculation
(`alpha = 0.05`, alternative `persistent_fresh > episode_isolated`). A causal
retention claim is emitted only when the structural checks and this gate pass;
otherwise the verdict is descriptive-only. No result from this experiment
supports held-out generalization or graph-specific causality.

The CLI defaults to a safe smoke configuration (`1` pair, `2` attempts, `10`
actions). Use `--full` for the frozen `32 x 20 x 50` design, or pass explicit
limits for a bounded smoke run. Artifacts are written to a new direct child of
`artifacts/reports/hunter_seeker_v2/`; `--out-root` rejects the repository or
workspace ancestors, arbitrary directories, existing outputs, and unsafe
replacement paths.

The earlier attempted full live run on 2026-08-02 was blocked by local
execution time, not by the pinned environment or a contract violation. It
produced no full-design result and was not rerun during the final hardening.
The hardened blocked metadata is recorded in
`artifacts/reports/hunter_seeker_v2/fresh_retention_v1_full_blocked_final3_20260802/`;
the earlier blocked directory remains preserved historical evidence.
