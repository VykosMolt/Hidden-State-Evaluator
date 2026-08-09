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
(7) interactions-to-threshold (0.5); (8) related-task transfer (R_4);
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

No conceptual decision is left open for the accelerator session.
