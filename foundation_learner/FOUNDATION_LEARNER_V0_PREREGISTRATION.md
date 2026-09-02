# FOUNDATION LEARNER V0 — PREREGISTRATION

Frozen: 2026-08-09, before any accelerator access, before any training, before
any sealed-test generation was readable. Companion authority for exact
engineering values: `docs/IMPLEMENTATION_CONTRACT.md` (+ amendments ledger).
Where this document and the contract state the same quantity, they agree; the
contract is the tie-breaker for engineering detail, this document for
scientific claims and analysis policy.

This is Foundation Learner **Pilot 0**. A null result is a useful result.

## 1. Research question

Can Ouro-RLTT 2.6B be trained over complete learning histories so that it
becomes better at **learning a previously unseen task family from attempts and
feedback**, rather than merely becoming better at the training tasks
themselves? The object of study is the within-episode learning trajectory
R_0..R_6, not static accuracy.

## 2. Backbone identity (frozen)

Single checkpoint for every arm: `models/ouro_rltt_local`, tree SHA-256
`a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889` (identical
byte-for-byte to the O1 v2.1 binding), OuroForCausalLM, 48 physical layers,
`total_ut_steps=4`, bf16, tokenizer.json SHA-256 `fcb808fe…c8fa8d`,
transformers exactly 4.54.1. No other model, no proxy model in any scientific
path. Every arm begins from a fresh load of this checkpoint. No O1 axis,
calibration result, or O1-mutated state enters training.

## 3. Task ecology and split (frozen)

Twelve computationally distinct exact-verifier generator families
(contract §4): boolean_rule, propositional_transform, modular_arithmetic,
sequence_transform, string_rewrite, finite_state_transducer,
permutation_composition, set_operations, graph_edge_semantics, dsl_execution,
constraint_rules, grammar_classification. Fresh latent rule per episode;
exact string-grammar verifiers; no LLM judge anywhere in principal outcomes.

Split rule (public, deterministic, frozen before computation):
`h(fid) = SHA-256("FOUNDATION_LEARNER_V0_FAMILY_SPLIT" ‖ 0x00 ‖ fid)`, sort
ascending by hex, positions 0–5 TRAIN, 6–8 DEVELOPMENT, 9–11 SEALED TEST.
Computed once (Amendment 2):

- TRAIN: grammar_classification, set_operations, dsl_execution,
  string_rewrite, finite_state_transducer, modular_arithmetic
- DEVELOPMENT: sequence_transform, graph_edge_semantics, boolean_rule
- SEALED TEST: constraint_rules, permutation_composition,
  propositional_transform

Zero family / exact-task / latent-rule overlap across splits (mechanically
enforced and hostile-tested). Sealed shards enciphered at generation; the only
decipher path is gated on a frozen development-decisions record and a
single-use opening ledger. Sealed outcomes are opened once, after all
development decisions freeze, and never justify further model modification.

## 4. Episode structure (frozen: EPISODE_STRUCTURE_V0)

One latent rule per episode. Interaction indices: 0 = attempt on P1,
1 = revision of P1 after feedback, 2 = attempt on P2, 3 = revision of P2,
4 = related problem P3, 5 = mean over queries Q1–Q3, 6 = mean over transfers
T1–T2 (K = 6). Feedback channels: certified `FEEDBACK` (always truthful;
correctness or certified structured), uncertified `HINT` (structured; may be
poisoned in poison conditions; rendering never distinguishes poisoned from
truthful hints), `REVEAL` (support answers only, only in declared supervised
conditions). Hidden query/transfer labels are never exposed. Training
histories are off-policy, from frozen scripted attempt policies (error rates
0.7 / 0.25 / 0.30; declared v0 design decision). Evaluation episodes are
online: real greedy generations, real verifier feedback. Plain-text protocol;
no chat template; strict final `ANSWER:` grammar.

## 5. Arms (frozen definitions; contract §7 for exact values)

- FL0 base evaluation (no training).
- FL1 static training on isolated task→answer pairs (no histories).
- FL2 imitation of complete successful histories (all model-span NLL).
- FL3 ordered-feedback meta-training: weighted NLL only on post-feedback
  competence spans — query 1.0, transfer 1.0, revisions 0.25, attempt-0 and
  all context 0. Stated plainly: this is direct query-loss optimization over
  learning histories; no "learning-progress reward" novelty is claimed.
- FL4 future-competence head: predicts realized Δ in query log-likelihood
  from including vs ablating a feedback item (targets from TRAIN families
  only, computed with the final FL3 checkpoint; ranking + MSE loss).
- FL5 persistent fast state: 1024-d GRU state over feedback events, injected
  as 8 norm-clamped prefix embeddings; FAST_STATE_ON vs FAST_STATE_OFF on the
  identical FL3 objective; survives textual context reset; reset at episode
  boundaries.
- FL6 value-gated adaptation: incorporate item iff FL4 prediction > 0
  (threshold frozen at 0), vs unconditional; tested under poisoned feedback.
- FL7 fast parameter adaptation: custom low-rank adapter (r=8, α=16, all
  attention q/v projections), 4 inner SGD steps (lr 1e-3) per revealed
  support item, global Frobenius clip 1.0, per-episode reset; ungated and
  value-gated variants. Runs only if predecessors justify it and time
  remains; never silently replaces FL5.
- FL8 consolidation: merge value-selected fast deltas into a slow adapter
  bank (scale 0.5); none vs indiscriminate vs value-gated; A→B→A retention,
  interference, unrelated-family degradation. Prepared, not required.

## 6. Primary comparison and headline rule

FL3 vs FL1 vs FL2 on whole-family-held-out learning curves. FL4–FL8 are
mechanistic extensions; an extension result can never replace a failed core
comparison in the headline. Unseen-instance and unseen-family generalization
are always reported separately; instance-level generalization is never
described as transferable learning. The principal Foundation Learner claim
requires improvement on the sealed whole-family holdout.

## 7. Metrics (frozen)

(1) macro-AULC over interaction indices 0–6, family-macro;
(2) ΔAULC vs FL1; (3) ΔAULC vs FL2; (4) R_0; (5) R_K; (6) improvement slope;
(7) interactions-to-threshold (0.5); (8) within-rule fresh-instance
generalization (R_4) [renamed by Amendment 17; NOT transfer];
(9) whole-family transfer; (10) context-reset persistence; (11) A→B→A
retention/interference; (12) poisoned-feedback robustness; (13) surface-remap
robustness; (14) FL4/FL6 value ranking, calibration, top-choice regret vs
oracle/random/surface heuristics. Family-clustered bootstrap (families, then
episodes; 10,000 replicates; shared resample indices for paired differences).
No row-independent intervals; no single-family domination of aggregates.

## 8. Compute matching (frozen policy)

Core arms FL1/FL2/FL3: identical optimizer-update count U and identical
per-update token budget (FLOP-matched); loss-token totals differ by objective
construction and are reported explicitly (declared option B). Full per-arm
compute ledgers (updates, loss/forward/backward tokens, wall time, GPU time,
examples, episodes) are published. The same trainable-parameter scope
(PEFT_MODE or FULL_MODEL_MODE, contract §8) for all core arms — never mixed;
scope chosen by the mechanical affordability rule, never by outcomes.

## 9. Development grid and selection (frozen)

Grid: exactly 2 learning rates ({1e-4, 3e-4} PEFT / {1e-5, 3e-5} FULL) on
FL3 at 25% step count; one optimizer family (AdamW 0.9/0.95, wd 0.01), one
scheduler family (cosine, 3% warmup); one interaction horizon
(EPISODE_STRUCTURE_V0). Selection: higher DEV macro-AULC, tie → lower LR;
winner locked for FL1/FL2/FL3. TRAIN+DEV families only. No new candidates
after seeing dev performance. Root seed 20260809; second predeclared seed
20260810 only if affordable.

## 10. Promotion, fallback, scheduling (frozen)

Promotion rules, fallback work, and the time-aware scheduler are frozen in
contract §10–§11 (FL3→extensions gate: stability + DEV macro-AULC ≥ FL1 +
0.02 + positive slope evidence; FL6 needs FL4 dev pairwise ≥ 0.55; FL8 needs
nonzero persistence evidence; SEALED TEST never used for promotion; a failed
rung triggers predeclared fallback work only — no live objective invention;
safety factor 1.25; final transfer reserve 1200 s; checkpoint every 600 s or
200 steps). O1 calibration has first claim on the accelerator; FL runs only
after O1 records are verified, transferred, and the O1 process closed; FL
never reads O1 scientific outputs; USD 45 total budget authoritative; rental
confirmation NOT AUTHORIZED at freeze time.

## 11. Allowed conclusion vocabulary

NO_META_LEARNING_SIGNAL · STATIC_CAPABILITY_GAIN_ONLY ·
HISTORY_IMITATION_GAIN · WITHIN_EPISODE_LEARNING_GAIN ·
WHOLE_FAMILY_TRANSFER_GAIN · CONTEXT_ONLY_ADAPTATION ·
PERSISTENT_FAST_STATE_GAIN · LEARNING_VALUE_PREDICTIVE · VALUE_GATING_GAIN ·
FAST_PARAMETER_GAIN · CONSOLIDATION_GAIN · INTERFERENCE_FAILURE ·
POISON_ROBUSTNESS_FAILURE · INCONCLUSIVE_UNDER_COMPUTE_BUDGET.
Never "recursive self-improvement"; never "generally self-improving system".

## 12. Deliberately unresolved (B200-derived, mechanical only)

1. Measured B200 training/eval throughput (BENCH stage output).
2. U (updates per core arm) — largest of {600, 1200, 2400, 4800} passing the
   frozen affordability inequality.
3. FULL_MODEL_MODE vs PEFT_MODE — outcome of the frozen affordability rule.
4. `available_foundation_learner_seconds` — computed at session time after O1
   closes, from remaining authorized budget.
5. Evaluation batch size (post equivalence-gate) and resulting eval episode
   counts per stage (frozen per-stage maxima in stage_definitions).
6. `o1_entry_command` — operator-bound (the sealed O1 package's pod
   entrypoint is currently a refusing stub; recorded, out of FL scope).
7. Container registry digest reference — operator-bound until the GHCR
   push (see §13.11).

No conceptual decision is left open for the accelerator session.

## 13. Pre-run amendments after adversarial review (frozen 2026-08-09, before
any accelerator use, before any training run, sealed set still unopened)

An independent adversarial review and an independent verification of the
complete package produced findings that are repaired in code and/or recorded
here as claim-scope constraints. Everything in this section is frozen BEFORE
any experiment ran; no outcome data existed when it was written.

1. **Statistical power and claim scope (few clusters).** With 3 development
   and 3 sealed families, the frozen family-clustered bootstrap's nominal-95%
   intervals cover the *population* ("an unseen family in general") estimand
   at only ≈74–83% (simulated under realistic between-family spread), and a
   CI-excludes-0 rule has ≈17–26% type-I error. Therefore: all sealed
   results are reported with the number of family clusters, per-family
   effects, and a sign statement; intervals are explicitly conditional on
   the three specific sealed families; the allowed conclusion
   WHOLE_FAMILY_TRANSFER_GAIN is always qualified "on the three sealed
   holdout families" and never presented as a population-level unseen-family
   claim. The frozen +0.02 development promotion margin is acknowledged to
   sit near this noise floor; it gates extension *spending*, not scientific
   claims.
2. **Headline decomposition (binary answer-flip shortcut).** Binary-answer
   families admit a "repeat the other label after INCORRECT" heuristic worth
   far more than the promotion margin at indices 1 and 3. The headline
   ΔAULC is therefore always reported alongside (a) AULC restricted to
   interaction indices {4,5,6} (fresh items, flip-immune) and (b) a
   transcript-computed flip-attributable success rate. A positive headline
   not supported by the {4,5,6}-restricted contrast is reported as
   shortcut-suspect, not as meta-learning.
3. **FL0 format floor.** The base model is expected to frequently fail the
   strict `ANSWER:` grammar within 64 new tokens. `answer_line_rate` is
   reported per arm × family × interaction index; any macro cell with rate
   < 0.5 is flagged FORMAT_NONCOMPLIANT next to its value; persistence
   ratios with zero-valued denominators are reported UNDEFINED, never 0.
4. **Feedback information taxonomy.** Structured hints are legitimately
   informative — probabilistic evidence is the point of feedback. The
   defect class is *deterministic decodability* of the pending answer from
   hint features. THREE such defects were found and FIXED pre-run, before
   any training or sealed access: (a) graph_edge_semantics hint-NODE
   SELECTION encoded reachability (generator → 1.1.0, answer-independent
   selection); (b) constraint_rules hint CONTENT ("number of violated
   constraints": zero ⟺ SAT) decoded the label with P=1.0 — replaced by a
   candidate-constraint probe whose probed candidate need not be active in
   the hidden rule (generator → 1.1.0+); (c) grammar_classification hint
   content ("which conjunct fails": none ⟺ IN) likewise — replaced by a
   pool-predicate probe (generator → 1.1.0). A permanent power-asserted
   fixture (≈6,500 items/family) now forbids deterministic hint branches in
   ALL twelve families with an EMPTY allow-list, and proves its own
   non-vacuity by reconstructing each of the three defective 1.0.0 rules
   and requiring the violation to reappear. Measured probabilistic lifts
   are recorded in the fixture's report (e.g. boolean_rule pivot-ABSENT
   ≈0.89 vs 0.63 prior; modular_arithmetic residue-band hints reduce the
   candidate set to ≈3.5 after one hint and fully determine ≈48% of items
   after two). The repaired probes' status is computable from the displayed
   prompt; their value is that they point at and pre-evaluate a hypothesis
   from the family's own pool, so rule identification comes from
   accumulating (probe, status, verdict) evidence across rounds.
   Interpretation rule: R_1/R_3 gains may reflect hint exploitation; the
   leak-robust quantities are the {4,5,6}-restricted metrics and fresh-item
   transfer.
4b. **Label balance and constant-answer floors.** constraint_rules item
   sampling was measured 89.3% UNSAT — a constant-answer policy would score
   0.893 on that sealed family. Fixed pre-run: its sampler is conditioned
   to approximately balanced labels (target P(SAT) ∈ [0.4, 0.6]); no other
   family's distribution changed (boolean_rule's 0.63 majority rate is
   recorded and accepted). Every family-level result is reported against a
   per-family constant-answer baseline column so that no majority-class
   floor can be read as competence.
5. **FL2 comparison confounds.** FL2's successful-history data forces
   attempt-0 wrong ≈70% of the time and imitates it with weight 1.0, so FL2
   is trained to be wrong at R_0 by construction; FL2 and FL3 also train on
   different history variants (not merely different objectives). Metric 3
   (ΔAULC vs FL2) is therefore reported both including and excluding
   interaction index 0, and is described as measuring objective + data
   jointly.
6. **FL5 claim scope.** FL5 arms use a segmented pipeline and are NOT
   FL3-comparable; every FL5 result carries `comparable_to_fl3: false`. The
   evaluation triad is frozen as ON / OFF / ON_S0 (trained ON module with
   state pinned to zero). A persistent-fast-state claim requires ON >
   ON_S0 under context reset — separating the recurrent state's
   contribution from the learned static prefix; failing that, the result is
   reported as prefix-tuning-equivalent (CONTEXT_ONLY_ADAPTATION /
   PERSISTENT_FAST_STATE_GAIN not granted).
7. **Off-policy scripted histories — named threat to validity.** Training
   histories use hand-written plausible-error samplers; evaluation attempts
   are the model's own. The error-style distribution shift could mute or
   mimic treatment effects. Planned diagnostic (post-run, local): compare
   scripted vs realized attempt/error distributions on TRAIN families from
   FL0/FL3 transcripts; conclusions are restricted accordingly.
8. **Eval-time online context allowance.** Online evaluation contexts
   (scripted text + real generations) may exceed the 2048-token
   *data-generation* budget; the frozen eval-time allowance is 4096 tokens
   (model limit 65536), with per-episode overflow isolation (recorded and
   excluded, never silently dropped, never aborting the batch).
9. **Sealed-opening robustness.** The single sealed opening is two-phase:
   evaluation must produce records before the opening commits; an aborted
   attempt is permanently ledgered and permits exactly one retry. The
   sealed evaluation runs the development-selected promoted arm from its
   recorded checkpoint (never the untrained base), and requires the
   completed core comparison as an entry condition.
10. **Diagnostics.** Surface-remap, A→B→A interference, and
    poisoned-feedback diagnostics run unconditionally after the core
    comparison (metrics 9/11/12/13 are produced in V0), including under the
    FL3-null fallback.
11. **Seventh operator-bound unresolved field.** The B200 container
    registry digest reference remains unresolved until the operator's
    registry push (mirroring the O1 record); it joins the declared
    mechanical unresolved set.

## 17. Amendment 17 — post-review repairs (2026-08-28, PRE-RUN, sealed set still unopened)

Three independent scientific reviews of the frozen design (two adversarial
model reviews plus the maintainer's own pass) were run before any accelerator
use. **No outcome data existed when this section was written**; `STATUS` was
`PRE_RENTAL_BUILD` and the sealed shards were unopened. Findings are repaired
in code and/or recorded here as claim-scope constraints.

### 1. The sealed opening evaluates EVERY core arm, not only the promoted one

**The defect.** Section 6 makes the principal claim "improvement on the sealed
whole-family holdout", and section 7 designates metrics (2) ΔAULC vs FL1 and
(3) ΔAULC vs FL2. But `campaign/stage_definitions.sealed_eval_work` evaluated
only the promoted arm, so **those two metrics did not exist on the sealed
set**. A positive FL3 trajectory there is fully compatible with base-model
in-context learning, because FL0's context-only baseline was measured on
DEVELOPMENT only. The single-use opening would have been spent producing a
number that cannot support the preregistered principal claim.

**The repair.** `SEALED_EVAL_ARMS = (FL0, FL1, FL2, FL3)`. Every arm walks the
**same** sealed episode set inside the **same** single opening, which is what
makes the contrasts paired rather than two independent samples. Every arm is fully LOADED and its checkpoint binding verified *before* the
seal is touched — a dry run that is then discarded — because file existence
alone is not enough: `load_checkpoint` sha256-verifies a multi-GB payload,
`_assert_checkpoint_binding` raises on arm-config drift, and `bundle_factory`
can OOM. Those failures are DETERMINISTIC, so inside the opening they would
recur on the retry and exhaust both permanent attempts, losing the sealed set
for good. `resolve_answer_parser()` is forced in the same preflight, because
it now raises rather than falling back and `run_episodes` resolves it lazily. Inside the opening there is no fallback to the base model for a
trained arm: evaluating the base and labelling it FL1 would fabricate the
contrast, so it refuses instead.

**Amendment 12.7 is unchanged and not weakened.** The promoted arm is still
never *replaced* by the untrained base — that was the R-C4 defect. FL0 is now
evaluated **in addition**, as the comparator.

**Paired contrasts are computed on the INTERSECTION** of episodes scoreable in
every arm of the contrast, with the per-arm exclusion counts reported. This is
frozen here because it must be: `clustered_sample_from_records` silently drops
an episode whose AULC is `None` (an aborted or online-budget-exceeded episode)
and `paired_clustered_bootstrap` then REFUSES arms whose episode-id sequences
differ. Prompt length depends on each arm's own generated history, so FL0 —
the untrained base, expected by 13.3 to ramble past the token allowance — will
drop a different set from FL3. Unhandled, that turns the designated PRIMARY
into "unavailable" *after* the single-use opening is spent; choosing the
handling then would be an unpreregistered post-hoc decision.

Cost: `STAGE_EVAL_MULTIPLIER["SEALED_EVAL"] = 4` and
`STAGE_BUNDLE_LOADS["SEALED_EVAL"] = 8` (one load per arm in the preflight dry
run, one in the opening), both tied to `len(SEALED_EVAL_ARMS)` by import-time
checks that raise — not `assert`, which `python -O` strips.

### 2. The frozen decision rule (this section did not previously exist)

Section 11 listed fourteen conclusion labels with **no criteria**, and section
6 required "improvement" without defining it. That left the final labelling
step as post-hoc judgment over ~18 metrics × several index restrictions. The
rule below is frozen now, before any data.

**PRIMARY (one, designated):** ΔAULC(FL3 − FL1) on the **{4,5,6}-restricted**
sealed contrast. Indices {4,5,6} are the post-feedback fresh items: immune to
the binary answer-flip shortcut (13.2) and to index-0 answer-format effects.
FL1 is the static-capability control at matched compute.

**"Improvement" in section 6 means all three of:** the primary point estimate
is > 0; the conditional 95% family-clustered interval excludes 0; and the sign
is the same in **at least 2 of the 3** sealed families.

**The achievable-inference floor, stated explicitly.** With F = 3 sealed
families, any family-level distribution-free test bottoms out at one-sided
p = 0.5³ = **0.125**. No configuration of results can reach p < 0.05 by a
family-level test. Every interval is **conditional on these three named
families** and is never a population-level unseen-family claim. This is a
property of the design, not of the outcome, and it is why the sealed test is
descriptive of three named families by construction.

**Everything else in section 7 is SECONDARY and DESCRIPTIVE.** No secondary is
an inferential test, so no multiplicity adjustment is claimed for them; a
secondary may motivate future work and may never carry the headline. The
`excludes_zero` boolean emitted by `analysis/report.py` is a descriptive
interval property, NOT a decision, and must not be reported as one for any
comparison other than the designated primary.

**Named contrasts.** All are ΔAULC on the {4,5,6} restriction, on the paired
intersection, with conditional 95% family-clustered intervals:
`P = FL3−FL1` (the PRIMARY), `S1 = FL3−FL0`, `S2 = FL2−FL1`, `S3 = FL1−FL0`.
"Meets" means the three-clause improvement rule above applied to that contrast.

**Label precedence.** Several conditions can hold at once, so labels are
evaluated in the order listed and the FIRST match is the headline; any others
that also hold are reported as additional findings, never as the headline.

**Criteria for the previously-undefined labels:**

1. `INCONCLUSIVE_UNDER_COMPUTE_BUDGET` — the sealed stage did not run, or any
   arm required by the contrast being reported is absent, or the paired
   intersection is empty. Checked FIRST: nothing below is meaningful otherwise.
2. `WHOLE_FAMILY_TRANSFER_GAIN` — `P` is met. Always written "on the three
   sealed holdout families". If item 3's grid-spread annotation applies, it is
   stated alongside; it annotates, it does not veto.
3. `HISTORY_IMITATION_GAIN` — `S2` is met and its point estimate is ≥ `P`'s,
   i.e. undifferentiated history imitation accounts for the gain.
4. `STATIC_CAPABILITY_GAIN_ONLY` — `S3` is met but `P` is not.
5. `CONTEXT_ONLY_ADAPTATION` — `S1` is NOT met (the trained promoted arm does
   not separate from the untrained base) while the raw R curve nevertheless
   rises for FL0, i.e. the trajectory is context-driven rather than trained.
6. `NO_META_LEARNING_SIGNAL` — none of `P`, `S1`, `S2`, `S3` is met.

`WITHIN_EPISODE_LEARNING_GAIN` is **NOT ASSIGNABLE in this campaign** and is
withdrawn from the vocabulary for Pilot 0. It requires the reset-vs-history
contrast, and `context_reset` is produced only for FL0 and FL5 — no stage
generates a reset condition for FL3, on development or sealed. Claiming it
would require adding that condition to the trained-arm evaluation, which is a
change to the campaign, not to this document.

### 3. Learning-rate selection asymmetry (declared, not repaired)

`campaign/dev_selector.GRID_ARM = "FL3"`: the two-point learning-rate grid runs
on **FL3 only** and the winner is imposed on FL1 and FL2. The comparison is
FLOP-matched but **not tuning-matched**, and the direction of the bias
systematically favours the treatment arm on the headline contrast. Declared as
a claim-scope constraint: **a FL3 − FL1 margin smaller than FL3's own grid
spread is not interpretable as an arm effect**, and the grid spread is reported
next to the primary.

### 4. Format acquisition gets a DIAGNOSTIC, not a correction

A rising R_0..R_6 curve can be produced with zero rule learning by the model
acquiring the strict `ANSWER:` grammar from the scaffolding and its own earlier
attempts — section 13.3 already expects the base model to fail that grammar
frequently. `FORMAT_NONCOMPLIANT` only flags cells below 0.5, so a cell moving
0.6 → 0.95 passes clean. Accuracy **among format-compliant attempts** is
therefore reported per index alongside AULC, and the `answer_line_rate` curve
is printed next to the R curve.

**It is a diagnostic and must not be read as a format-corrected accuracy.**
Compliance is a post-treatment variable that the arm affects and that episode
difficulty affects jointly with correctness, so conditioning on it opens a
collider path: it substitutes a selection bias of unknown sign for the format
channel rather than removing it, and the bias favours whichever arm is least
compliant — which is expected to be FL0, shrinking the trained-vs-base
contrast. It is read ONLY jointly with `answer_line_rate` and never as an arm
comparison. The comparable format-robust quantity remains the headline itself,
which already scores a non-compliant attempt as incorrect (fixed denominator,
no collider).

### 5. Metric 8 renamed

"related-task transfer (R_4)" → **"within-rule fresh-instance generalization
(R_4)"**. The related item is the same latent rule at the same difficulty,
drawn from an identical item distribution to R_5's queries — no family branches
on `KIND_RELATED`. It is not transfer in any sense, and the old name invites
double-counting it with metric 9.

### 6. "Sealed" is PROCEDURAL, not cryptographic

`K_seal = sha256("FL_V0_SEALED_KEY\0" + split_manifest_sha256)` derives from a
**public, recomputable** digest, as `data/shards.py` and `campaign/sealed_gate.py`
both state unprompted. The integrity that is real comes from the write-ahead
intent record, the hash-chained append-only single-use ledger, and
`campaign/promotion.py` structurally refusing sealed evidence. Write-ups must
describe the protection as procedural and must not imply cryptographic sealing.

### 7. OPEN RISK introduced by this amendment — sealed-stage scheduling

`SEALED_EVAL` is priority 13, **last of the seventeen stages**, and its own note
gives the reason: "LAST: it can never inform a development decision because
every such decision is already frozen." That rationale is sound as far as it
goes — but the structural protection against contamination is
`campaign/promotion.py` refusing sealed evidence in the `DevMetrics`
constructor (hostile-tested against five smuggling routes), **not** the
ordering. The ordering is belt-and-braces on top of it.

The consequence is a scheduling tension that item 1 of this amendment
sharpens, and it is recorded here rather than silently repaired:

- `SEALED_EVAL` produces the **principal claim** (section 6).
- It is scheduled behind every mechanistic extension (FL4-FL8) and every
  diagnostic, although section 6 also says "an extension result can never
  replace a failed core comparison in the headline".
- Under budget pressure it is therefore the **first work to be dropped**, and
  item 1 raised its cost from 300 to 1200 episode walks (+900, about +10% of
  the campaign's total evaluation walks).

So the amendment that made the sealed result *interpretable* also made it
*more likely to be skipped*. Both halves are true and both are recorded.

**Not repaired here**, deliberately: moving a stage in the priority ladder is a
design change whose downstream effects (SECOND_SEED, the fallback-work chain,
the overrun watchdog) have not been traced, and the campaign's own projection
already refuses to start work it cannot finish. **The decision required before
the rental** is one of:

1. reserve the projected `SEALED_EVAL` seconds up front, so the extensions
   consume only what remains after the principal claim is funded; or
2. promote `SEALED_EVAL` to run immediately after `CORE_MATCHING` (priority 5),
   relying on `promotion.py`'s structural refusal — which is what actually
   enforces non-contamination — rather than on ordering; or
3. accept explicitly that a budget-truncated campaign yields
   `INCONCLUSIVE_UNDER_COMPUTE_BUDGET` with no sealed result, which the
   conclusion vocabulary in item 2 already provides for.

4. **predeclare a degraded arm set** via the `fallback_work` machinery the
   stage definition already carries (contract §10): if the projection does not
   fit, run `(FL0, FL1, FL3)`. The PRIMARY is ΔAULC(FL3−FL1) and FL0 is the
   in-context comparator, so only secondary metric (3), ΔAULC vs FL2, is lost.
   That cuts the increase from +900 to +600 walks, preserves the principal
   claim intact, and touches neither the priority ladder nor the budget
   accounting. It converts "the headline stage is dropped entirely" into "the
   headline stage runs, one secondary is unavailable."

**Two independent reviews split on this**, and both arguments are recorded
because they are about different risks:

- One favours **option 4**: it is the only option that keeps the principal
  claim computable under budget pressure without changing scheduling.
- One favours **option 1 + moving the sealed stage off interruptible
  capacity**, and raises a specific objection to option 2 that the other did
  not: after `unlock.revoke()` the sealed report sits in
  `ctx.results["SEALED_EVAL"]`, readable by every later stage's *work
  function*. Only `DevMetrics` construction is structurally guarded, so
  running the sealed stage at priority 5 with FL4–FL8 still to come genuinely
  widens the contamination surface rather than being a no-op. **Option 2 is
  therefore withdrawn.** The same review notes the opening is now ~4× longer
  and carries four model loads, so on interruptible capacity a mid-opening
  eviction — which counts against the two permanent attempts — is roughly 4×
  more likely.

Options 1 and 4 are complementary and both are recommended: reserve the
projected seconds up front (which requires the corrected
`STAGE_BUNDLE_LOADS`, or the reservation is itself an under-estimate), AND
predeclare the degraded arm set as the fallback. Option 3 remains the status
quo and is defensible for a pilot only as an explicit recorded choice.

## 18. Amendment 18 — session budget follows O1 amendment 2; sealed evaluation reserved (2026-09-02, PRE-RUN, sealed set still unopened)

The "USD 45 total budget" statements in sections 12 and 13 are superseded:
the combined session budget is USD 35.00 total / 30.00 compute / 5.00 reserve
(O1 budget amendment 2, 2026-09-02; contract Amendment 17). The change is a
resourcing fact, not a design choice: no metric, threshold, seed, split,
family, episode structure, arm, or promotion rule changes with it. Rental
confirmation remains NOT AUTHORIZED.

Amendment 17 option 1 is implemented: the scheduler reserves the projected
sealed-evaluation seconds (times the frozen safety factor) before admitting
any rung other than BENCH, DEV_GRID, FL1-FL3, CORE_MATCHING and SEALED_EVAL
itself, so a short ladder degrades by dropping mechanism rungs and
diagnostics, never the sealed evaluation.
