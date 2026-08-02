# Fresh-discovery experiment

This note defines the bounded `tu93` V2 experiment implemented by
`utilities/tests/manual/run_hs_v2_fresh_discovery_v1.py`.

## Preregistered design

The experiment uses the seeded native V2 transaction loop and ARC adapters. It
has one explicit-route acquisition, followed by isolated arms:

| Arm | Acquired state | Route | Teacher | Purpose |
| --- | --- | --- | --- | --- |
| `acquired_route_positive` | checkpoint graph and goals retained | no | no | positive control for committed graph guidance |
| `acquired_graph_off` | acquired graph removed | no | no | graph-off acquired control |
| `fresh_autonomous` | new empty agent | no | no | truly fresh autonomous discovery |
| `fresh_goal_promotion_off` | new empty agent | no | no | fresh switch control with both supported completion-promotion paths disabled |

The acquisition arbiter only selects an action already present in the ordinary
candidate set. Every action uses `agent.act -> environment.step ->
agent.observe`. The fresh arms are constructed directly in `AUTONOMOUS` mode;
they do not load the acquisition checkpoint, receive its route, attach a
teacher, or enter `compat_assisted` mode. A checkpoint is allowed only in the
acquired controls.

The goal-promotion control uses the existing public configuration switches
`enable_goal_contrast=False` and `enable_completion_motion_reach=False`. A
graph-write-disabled control is not claimed because V2 has no public graph
write switch and the experiment does not introduce a substitute runtime API.

## Evidence and verdicts

Each run writes a JSON trace with decisions, committed transitions, frame
hashes, stage/boundary milestones, graph-plan metadata, measurement summary,
failure classification, and a leakage audit. The output also contains:

- `provenance_manifest.json`: exact V2 source, selected docs, focused tests,
  manual runners, and both selected `tu93` environment variants, with SHA-256
  hashes;
- `summary.json`: design, acquisition, evaluations, checks, and verdict;
- `verdict.json`: machine-readable verdict only.

The structural verdicts distinguish `VALID_FRESH_DISCOVERY_COMPLETION_OBSERVED`
from `VALID_NO_FRESH_DISCOVERY_WITHIN_BUDGET`. A missing runtime dependency or
execution exception produces `BLOCKED_RUNTIME_ERROR` and remains a blocked
artifact. No positive result is inferred from a timeout, death, or blocked run.

The pre-task repository was already dirty. Its preserved baseline was HEAD
`ea40a48f7f41fdf60866eb2067033c62dd14be0f`, 0 staged entries, 39 unstaged
tracked entries, 9,843 untracked entries, 9,882 total status entries, and
status digest
`35928c5fd875731ba9c30ea64b7157b8259b790501b6bb27beb6f63c85cd1c5a`.
The runner records that baseline alongside the live experiment-start status;
it never stages, resets, cleans, or pushes repository state.

## Results

Run completed at `2026-08-01T22:08:56.333Z` with artifact
`artifacts/reports/hunter_seeker_v2/fresh_discovery_v1_20260802T002000Z/` and
provenance aggregate SHA-256
`a6c9d74742c6046b2d7fbdc0e0e2728138d7ae706568b930e15e88142681c3e0`.

| Arm | Steps | Completions | Graph plans | Failure classification |
| --- | ---: | ---: | ---: | --- |
| `acquired_route_positive` | 18 | 1 | 18 | completed |
| `acquired_graph_off` | 25 | 0 | 0 | budget exhausted |
| `fresh_autonomous` | 25 | 0 | 0 | budget exhausted |
| `fresh_goal_promotion_off` | 25 | 0 | 0 | budget exhausted |

The acquisition route was exact and completed once in both matched acquisition
donors. The fresh-arm leakage audits passed: both began with zero graph nodes,
zero graph edges, and zero evidence records; both were autonomous, teacher-free,
checkpoint-free, route-free, and used the native `RiskArbiter`. The goal switch
control recorded both promotion flags as disabled and admitted no verified goal.

Machine verdict: `VALID_NO_FRESH_DISCOVERY_WITHIN_BUDGET`. This is a valid
bounded negative/inconclusive result for from-scratch discovery, not evidence
that fresh discovery succeeded or failed beyond this seeded 25-step budget. It
does establish that the substrate can distinguish retained acquired knowledge
from graph-off and genuinely fresh arms without silently converting the
acquisition route into a fresh result.
