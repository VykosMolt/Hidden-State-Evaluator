# Hunter-Seeker v2

Hunter-Seeker v2 is the compact, transactional rebuild of the experimental
Hunter-Seeker agent.  It is side-by-side with `hunter_seeker_core`; the legacy
package remains available as a reference oracle.

The v2 production package is composition-based.  Its core invariants are:

- `act` never consumes an unobserved predecessor transition;
- `observe` commits the action that actually caused completion or death;
- autonomous and student action selection perform no teacher lookup;
- hidden-state taps are bounded sensors rather than rewards or controllers;
- imagined uncertainty is penalized, while safe real uncertainty can be
  explored;
- negative memory is task-scoped by default;
- all policy pressure is visible as named score terms.

## Audited goal-directed planning (2026-07-26)

Verified completion-grounded `reach` and monotonic count/coverage goals now
guide a bounded weighted-A* search over committed state-graph edges.  The
heuristic is deliberately not claimed to be admissible through walls.  When a
known edge has exact progress evidence, that causal edge outranks both a
heuristic-potential decrease and an untried frontier; its first action still
passes through the ordinary legality, support, risk, and final arbitration
gates.

ARC stage changes use two protocols.  With the default 5% changed-cell
threshold, a small completion change is treated as a delayed board reveal and
the subsequent visual-reset action is excluded from ordinary dynamics and
goal evidence.  A larger change is an immediate next-board swap: generic
cross-stage goal contrast is suppressed because no old-stage solved frame is
visible.  The only inference allowed in that case is an unambiguous
ego-motion-projected `reach` relation whose controlled source lands directly
on one stationary target (`origin="completion_motion"`).

The reproducible July 26 live `tu93` acquisition in
`utilities/tests/manual/run_hs_v2_tu93_graph_goal_v1.py` uses an explicit
fixed-route arbiter over ordinary agent candidates and native transactions;
it attaches no teacher.  Two isolated acquisitions of the exact 18-edge route
each produced only `reach|4|14` through this fallback, with no false
count/equality goals.  A fresh autonomous graph-enabled replay followed all 18
actions exactly and completed stage 1 at step 18.  The matched graph-off
control disabled goal search and removed the acquired state graph (preventing
base-scorer graph-distance leakage); it chose action 1 on all 25 steps and
completed nothing.  Observed progress paths receive the full bounded graph
term; the named `graph_goal_detour_reconciliation` term cancels only a
contradictory one-step hypothesis penalty so the verified path can detour
around walls.  Autonomous evaluation kept the ordinary 0.70 risk threshold;
the maximum acquired-route effective risk was 0.134.

## Run a public ARC game

```bash
venv/bin/python -m hunter_seeker_v2.run_arc ls20 --max-steps 100
```

The runner uses the correct order:

```python
decision = agent.act(observation)
raw_next = environment.step(...)
next_observation = observation_adapter.observation(raw_next, task_id=task_id)
outcome = outcome_adapter.outcome(observation, next_observation, raw_after=raw_next)
transition = agent.observe(decision, next_observation, outcome)
```

## Construct the agent

```python
from hunter_seeker_v2.adapters import MockActionAdapter
from hunter_seeker_v2.agent import CompactHunterSeeker

actions = MockActionAdapter()
agent = CompactHunterSeeker(
    click_action_index=actions.click_action_index(),
    safe_action_provider=actions.safe_action_indices,
)
```

The agent consumes immutable `Observation` values from
`hunter_seeker_v2.contracts`.

## Runtime modes

- `autonomous`: no teacher or direct demonstration read in `act`.
- `student`: teacher data may update models offline, but `act` has no teacher
  access.
- `compat_assisted`: exact-state trajectory advice may add a bounded score term
  and is always audited.

Use `TrajectoryTeacher.from_npz(...)` for the existing
`frames / frames_after / actions / levels` trajectory format.

`distill_teacher(...)` trains the state-independent prior, dynamics, and the
state-conditioned student policy.  The latter reads only the current
representation during `act`; its named `student_policy` term is visible in the
decision trace and still passes through risk arbitration.

## Optional Ouro loop features

`OuroLoopRepresentationBackend` wraps an already-loaded frozen GridEncoder and
Ouro looped model.  It compresses early/middle/late loop states through
`TapConnector` into one representation; it does not import the legacy agent or
give Ouro authority over the controller.  This is a representation plugin, not
RLTT training.  Genuine RLTT additionally requires all-loop policy-gradient
credit from terminal-loop sampled outcomes.

```python
from hunter_seeker_v2.representation import (
    OuroLoopRepresentationBackend,
    TapConnector,
)

backend = OuroLoopRepresentationBackend(
    encoder,
    ouro_model,
    connector=TapConnector(latent_dim=32),
    device="cuda",
)
```

## Verification

Focused v2 tests:

```bash
venv/bin/python -m pytest -q \
  utilities/tests/unit/test_hunter_seeker_v2_*.py \
  utilities/tests/integration/test_hunter_seeker_v2_integration.py
```

Design and migration notes:

- `docs/hunter_seeker_v2/ARCHITECTURE.md`
- `docs/hunter_seeker_v2/RESEARCH.md`
- `docs/hunter_seeker_v2/CAPABILITY_PARITY.md`
