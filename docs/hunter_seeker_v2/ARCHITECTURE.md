# Hunter-Seeker v2

Hunter-Seeker v2 is a side-by-side compact rebuild.  The legacy
`hunter_seeker_core` package remains intact as a behavioral reference and
compatibility oracle; v2 does not inherit its mixin stack.

The rebuild preserves demonstrated capabilities, not historical mechanisms.
In particular, it does not treat the legacy CLT evaluator, affect vocabulary,
phase templates, or a zero-initialized context token as architectural
foundations.

## Empirical boundary

The rebuild follows the corrected evidence in the 2026 Erratum and full draft:

- Preference is modestly readable both pointwise and relationally.
- Pairwise readers require source/task-disjoint evaluation and strict
  antisymmetrization.
- Role-specialized taps are sensors, not controllers.
- Readability does not establish steering, internal use, consciousness, or
  improved capability.
- External environment outcomes and independent verifiers remain authoritative.
- Branch survival and final selection are separate roles.

Consequently:

- The fixed-order `FrozenCLTAnchor` is not ported.
- No hidden-state reader is a reward source by default.
- Pointwise progress, hazard, value, and uncertainty estimates are legitimate.
- Pairwise taps are optional, antisymmetric by construction, and cannot bypass
  the controller or external outcome labels.
- Self-state is operational competence telemetry, not a claim about affective
  experience.

## Runtime invariant

The public interaction loop is transactional:

```python
decision = agent.act(observation)
next_observation = environment.step(decision.action)
transition = agent.observe(decision, next_observation, outcome)
```

Exactly one successful environment action creates exactly one immutable
transition before any completion, death, reset, or run-end handling.  A
decision cannot be observed twice and a different agent instance cannot commit
it.  Completion and death belong to the action that caused them, never to its
predecessor.

### Stage-boundary protocol

A numeric stage change does not by itself say whether the returned pixels are
the solved old board or the new board.  The runtime therefore measures the
changed-cell fraction against `boundary_bridge_change_fraction` (default
`0.05`):

- a completing, nonterminal change at or below 5% is the delayed-reveal
  protocol.  The old-stage completion frame remains usable as goal evidence;
  one deterministic bridge decision then reveals the next board, and that
  screen replacement is excluded from ordinary dynamics, ego, exogenous, and
  goal learning;
- a larger completing change is an immediate stage swap.  Perception starts a
  new track namespace and generic old-frame/new-frame contrast is suppressed,
  because the new board is not evidence about the solved old-stage geometry.

For an immediate swap, the only hidden-completion inference is a conservative
`completion_motion` reach proof: a supported controlled-motion prediction must
move exactly one source colour into direct overlap with exactly one unchanged,
stationary target.  Adjacency, moving targets, multiple plausible pairs, count
goals, and geometric equality all cause abstention.

## Compact data flow

```text
environment observation
        |
        v
canonical adapter view
        |
        +--> exact frame/state graph
        |
        +--> connected components, tracks, events, topology
        |
        +--> frozen/grid representation + optional role taps
                         |
                         v
             action-conditioned dynamics ensemble
              effect / progress / hazard / terminal
                         |
                         v
             candidates + bounded receding-horizon search
                         |
        verified reach/count goals + observed-graph guidance
                         |
              state-conditioned student preferences
                         |
                         v
          explicit score terms + one safety/risk arbitration
                         |
                         v
                       action
```

Root observations receive the full object/event/topology pass.  Speculative
nodes chain exact graph successors where known (carrying memory state ids
through the beam) and fall back to compact learned predictions otherwise;
goal-hypothesis potentials apply along both paths.  Verified reach/count goals
also run bounded deterministic weighted-A* guidance over committed graph
edges.  Speculation cannot mutate durable state.

## Exploration and imagination

One epistemic uncertainty estimate replaces several loosely coupled
Hunter/Seeker heuristics.

For real actions:

```text
score =
    expected_progress
  + expected_value
  - hazard_weight * expected_hazard
  + exploration_weight * epistemic_uncertainty
  + learning_progress
  + state_conditioned_student_policy
  + scoped_memory
  - action_cost
```

For imagined continuations, uncertainty changes sign:

```text
imagined_score =
    expected_progress
  + expected_value
  - hazard_weight * expected_hazard
  - model_doubt_weight * accumulated_uncertainty
  - action_cost
```

Real uncertainty is attractive only after the candidate passes the configured
risk constraint.  Imagined uncertainty is never exploitable optimism.

## State graph and world models

The observed graph is hash-addressed and records:

- exact state/action/successor transitions;
- no-change actions;
- cycles and merged paths;
- terminal and progress boundaries;
- visit counts and back-labelled distance to observed progress.

Verified completion-grounded `reach`, `count_at_most`, and `count_at_least`
goals can guide bounded weighted-A* traversal of those observed edges.  Their
potentials are useful heuristics rather than admissible distances through
walls, so no optimality claim is made.  Target precedence is evidential:

1. an exact committed edge whose successor is progress-labelled;
2. a cached state with improved goal potential;
3. a non-worsening untried frontier;
4. an older uncached node.

The exact progress edge deliberately outranks frontier and potential
improvement.  Every edge must still meet visit support, dominant-successor,
legality, and risk thresholds, and the resulting first-action bonus remains
bounded and passes through the ordinary risk arbiter.

A committed path ending in observed positive progress receives the full
configured graph bonus rather than a second path-length discount; weighted
A*'s accumulated edge cost already ranks path length.  If its first move is a
necessary geometric detour, the audit-visible
`graph_goal_detour_reconciliation` term cancels only the contradictory
one-step `hypothesis_potential` penalty.  It does not cancel hazard, legality,
support, dominance, or any other policy evidence.

The learned ensemble predicts decision-relevant quantities rather than
reconstructing every pixel:

- global/spatial latent delta;
- object additions, removals, and motion summary;
- probability of any change;
- progress distribution;
- hazard/death probability;
- completion/terminal probability.

A diagnostic decoder may be added independently.  It is not required by the
planner.

An optional executable-model registry accepts compact rule models that expose
`predict`, `plan`, and replay verification.  Models are ranked by verified
coverage, mismatch rate, and complexity.  They cannot read recorded future
observations during prediction.

## One evidence store

The append-only transition/event ledger is the source of truth.  Derived views
replace overlapping engram, terminal, recent-window, and click-cooldown stores.
Each record carries task/stage scope, provenance, confidence, polarity, object
signature, action relation, outcome, model version, and uncertainty.

- Negative hazard/terminal evidence is task-scoped by default.
- Positive transfer requires matching action-effect evidence.
- Counterevidence attenuates a belief but cannot silently erase repeated
  terminal evidence.
- Beliefs and indexes are rebuildable from the ledger.

## Teacher boundary

Runtime modes are mutually exclusive:

- `autonomous`: no solved prefix, phase policy, exact demonstration lookup, or
  teacher read during `act`.
- `student`: may have trained from teacher data, but performs no teacher read
  during `act`.
- `compat_assisted`: explicitly labelled historical-assistance mode.

All direct demonstration access implements one `Teacher` protocol.  Autonomous
and student modes install a guard that raises if any teacher method is called
during action selection.

`student.py` is the legal retained-knowledge route.  Offline examples train a
candidate-conditioned function of the current representation.  Balanced
teacher batches counteract collapse onto the demonstrated action histogram;
the action-conditioned rows can therefore express minority turns, although
neither balancing nor the deterministic nonlinear feature map guarantees that
all route states are separable.  The map consumes the flattened global and
spatial representation and stores neither teacher state ids nor a teacher
reference.  Committed online outcomes update the same task rows, and only
explicit positive rows may transfer.  Its bounded, weighted `student_policy`
term is visible in every root candidate and still passes through the final
risk gate.

The optional `StudentRepresentationTrajectory` applies normalized label or
outcome credit to every supplied loop representation and designates one
terminal representation for runtime scoring.  This multi-representation API
is unit-tested but is not wired into `HunterSeekerV2Agent`: both the default
GridFeature backend and the Ouro connector currently supply one collapsed
`Representation`.  The runtime therefore uses a one-element trajectory and is
not described as RLTT.  Genuine RLTT would require an action-policy analogue
of the Williams–Tureci objective: terminal-loop sampling, one group-normalized
outcome advantage applied to every loop's action distribution with normalized
loop weights, and terminal-loop-only KL regularization.

## Taps

Tap providers return typed, calibrated readings:

- pointwise progress/value/hazard/uncertainty;
- pairwise candidate preference;
- branch survival;
- diagnostic-only readings.

Pairwise adapters evaluate both orders and expose only the antisymmetrized
component.  Survival taps can retain top-k branches but cannot silently become
the final selector.  A tap can influence policy only when its configuration
names the role, calibration data, and score bound.

## Exogenous memory identity

Exact frame hashing makes every observation containing an autonomously
evolving component — a countdown bar, a step counter, a background
animation — a permanently novel state, silently disabling the state graph,
no-change penalties, distance labels, and cross-episode reuse.
`exogenous.py` learns which observation cells are exogenous with an
intervention-grounded test: a cell whose change-event trajectory replays
identically across episodes whose executed action histories differ is
outside the agent's control.  Masked cells are excluded from the *memory*
identity (`WorldSnapshot.memory_id`) used by the state graph and evidence
keys; transactional identity, the teacher boundary, and executable-model
state remain exact.  Evidence is scoped by task, visual stage, and frame
shape; legacy task/shape-only masks are quarantined rather than copied across
stages.  The mask is frozen within a comparable task/stage segment,
cross-stage visual swaps are excluded from evidence, contradictions unmask
conservatively, and `ExogenousConfig.enabled = False` is an exact no-op.
Stated limitations: first-episode memory is inherently unfiltered, a
never-resetting lifetime clock evades the test, truncated or delayed causal
event histories remain inference limitations, and a time-deterministic hazard
animation will be merged in memory, leaving object beliefs and evidence as
the safety carrier.

## Relational goal hypotheses

`hypotheses.py` addresses goal inference with region-geometric relations
(cellwise and lattice-canonical equality, mirror equality, self-symmetry,
interior uniformity), object `reach`, and one-sided monotonic count/coverage
targets (`count_at_most` and `count_at_least`).  A hypothesis is a proposal,
never a claim:

- policy pressure is one named bounded term, `hypothesis_potential`, built
  from realized potential deltas per action (with exact successor lookups
  when the state graph knows the outcome) — potential-difference shaping,
  removable by construction;
- unverified structural invariants carry exploration-grade weight.  Genuine
  goals normally enter through an observed initial-to-completion
  violated→satisfied contrast and are tagged `origin="goal_contrast"`;
- count goals require a material monotonic trace and use a one-sided
  potential: reaching or passing the completion count remains satisfied;
- if an immediate stage swap hides the completion frame, only the narrow
  direct-overlap reach proof described above may enter with
  `origin="completion_motion"`;
- only verified `reach` and count goals participate in graph guidance.
  Historical observed frames backfill their potentials; missing historical
  geometry stays unknown rather than becoming false zero;
- fully masked relations have no observable support and cannot be proposed or
  promoted as proven;
- exogenous-masked cells are excluded from potentials so environment clocks
  cannot masquerade as goal mismatch;
- `HypothesisConfig.enabled = False` is an exact behavioral no-op.

The reproducible July 26 live `tu93` experiment in
`utilities/tests/manual/run_hs_v2_tu93_graph_goal_v1.py` used an explicit
fixed-route acquisition arbiter over native agent transactions, with no
teacher.  Two isolated acquisitions of the exact 18-edge route each admitted
only `reach|4|14` with `origin="completion_motion"` and no false
count/equality relation.  A fresh autonomous graph-enabled replay selected
the entire 18-action route exactly and completed stage 1 at step 18.  The
matched graph-off control disabled goal search and removed the acquired state
graph to prevent base-scorer graph-distance leakage; it selected action 1 for
all 25 steps and completed zero stages.  Autonomous evaluation retained the
ordinary 0.70 risk threshold (maximum acquired-route effective risk 0.134).
This closes the load-bearing graph-search gate for acquired, committed
knowledge; it does not establish from-scratch discovery.

## Control attribution

`ego.py` restores the legacy ego/control-set capability as one compact,
count-based mechanism.  A track's influence is the normalized mutual
information between the executed action and the track's quantized
displacement response, scaled by support trust — a causal-influence proxy,
not an intervention proof.  Static objects and genuinely action-independent
drifters collapse toward zero under action-diverse evidence; an exogenous
process correlated with action choice can still confound this statistic.
Directional attribution applies only to non-positional actions; click
attribution stays with the affordance click-target route.

The component has exactly three outputs:

- it grounds the existing `ObjectState.controllable` belief (never lowering
  it), which makes the object-summary features live in clickless domains;
- it defines the controlled set used for ego-protected contact attribution:
  on a hazardous or fatal outcome, objects adjacent to the controlled set are
  blamed through the ordinary affordance route while controlled signatures
  are protected;
- it emits one named bounded score term, `ego_motion_hazard`, when the
  predicted ego displacement under a candidate action lands on or adjacent to
  an object with a believed hazard.

`EgoConfig.enabled = False` is an exact behavioral no-op.  The component
consumes only committed real transitions and never selects actions.  Because
contact attribution fires at hazard/terminal outcomes, its behavioral value
appears across repeated attempts at the same task, not within a single
first-death episode.

## Operational competence state

The compact self-state contains quantities with observable targets:

- dynamics error EMA;
- hazard calibration error;
- predicted success;
- recent realized progress;
- ensemble disagreement;
- stagnation count;
- expected learning gain;
- remaining risk budget.

These values tune exploration weight, planning horizon, retrieval, updating,
and recovery.  They are direct normalized inputs, not injected through an
untrained projector.

## Package map

```text
hunter_seeker_v2/
  contracts.py       immutable public types and configuration
  adapters.py        ARC, legacy, and synthetic environment boundaries
  perception.py      components, global track assignment, events, topology
  representation.py  frozen multi-tap and optional Ouro loop connector
  memory.py          state graph and append-only evidence ledger
  models.py          compact online priors, dynamics ensemble, and competence
  ego.py             count-based control attribution over tracks/signatures
  exogenous.py       intervention-grounded exogenous-cell memory filtering
  hypotheses.py      relational goal hypotheses as bounded potentials
  taps.py            role-specific sensor contracts and antisymmetric wrappers
  teacher.py         the only demonstration-access boundary
  executable.py      verified executable-world-model registry
  search.py          proposals, exact/model rollouts, and transpositions
  policy.py          score assembly and risk arbitration
  learning.py        online and provenance-separated replay updates
  student.py         state-conditioned and trajectory-credit student policy
  agent.py           transactional act/observe runtime and online learning
  persistence.py     strict versioned checkpoints
  diagnostics.py     typed traces and derived reports
  run_arc.py         terminal-correct ARC harness
```

## Acceptance gates

The implementation is not considered complete until tests establish:

1. `N` successful environment steps create exactly `N` transitions.
2. Fatal and completing actions own their outcomes.
3. Re-observing a decision is rejected without duplicate learning.
4. Odd-shaped categorical grids and clickless action spaces work.
5. Speculative search leaves durable perception and memory unchanged.
6. Final score equals the sum of active score terms.
7. Risk arbitration applies equally to greedy, beam, and random candidates.
8. Unrelated negative evidence is exactly neutral.
9. Autonomous action selection performs zero teacher reads.
10. Pair taps are exactly antisymmetric at their public boundary.
11. Save/load reproduces the first resumed decision and scoped evidence, while
    full loads reject mismatched backends, taps, or executable registries.
12. Empty masks/actions, nonfinite sensor values, and unseen states remain
    finite and safe.
13. Every learned route has a parameter/state change test and a crafted
    decision-causality test; a head-specific live-performance claim additionally
    requires a matched head-on/head-off rollout.
14. Executable models are replay-verified before they may plan real actions.
15. Copied or altered decisions cannot commit, and a failed observe rolls all
    mutable knowledge back before an exact retry.
16. Imbalanced demonstrations recover minority state-conditioned actions;
    optimization epochs do not inflate empirical support.
17. Delayed and immediate stage changes follow distinct protocols, and a
    next-stage board cannot become generic old-stage goal evidence.
18. Exact progress edges outrank heuristic/frontier graph targets while still
    passing support and risk gates.
19. Malformed component, ego, and exogenous checkpoint state is rejected
    without coercion, clamping, or partial replacement.

## Deliberate compatibility status

The following remain outside the autonomous core:

- solved-prefix following and runtime phase templates;
- the invalid fixed-order evaluator anchor;
- affective/consciousness interpretations;
- the legacy post-veto module that currently does not veto;
- coarse episode memory that is written but not read by policy;
- context/temporal projectors without demonstrated gradient and decision paths.

Compatibility plugins may expose historical assisted behavior, but reports must
never mix assisted and autonomous results.
