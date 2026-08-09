# Contract amendments

## Amendment 1 — sealed-shard key derivation (2026-08-09, pre-implementation)

Original §12 derived the sealed-shard cipher key from
`dev_decisions_sha256`. That is impossible: sealed shards are enciphered at
local pre-generation time, before any development decisions exist.

Amended rule: `K_seal = sha256(b"FL_V0_SEALED_KEY\0" +
split_manifest_sha256_hex_utf8)`. Enciphering happens at shard-write time in
`data/shards.py` using the shared util in `ecology/base.py`. Deciphering code
exists ONLY in `campaign/sealed_gate.py` and refuses to run unless
`DEV_DECISIONS_FROZEN.json` exists and the single-use opening ledger entry has
been written. Enforcement of the sealed-test policy is therefore procedural
(gate + ledger + hostile fixtures) with the cipher preventing accidental or
casual plaintext access; this is recorded honestly rather than claiming
cryptographic unopenability.

Recorded before any implementation began. No data or results existed at
amendment time.

## Amendment 2 — family split computed (2026-08-09, pre-implementation)

The §5 rule was executed once, mechanically, immediately after freezing:
TRAIN = {grammar_classification, set_operations, dsl_execution,
string_rewrite, finite_state_transducer, modular_arithmetic};
DEV = {sequence_transform, graph_edge_semantics, boolean_rule};
SEALED_TEST = {constraint_rules, permutation_composition,
propositional_transform}. `ecology/split.py` must reproduce exactly this
assignment from the frozen rule; a regression test pins it. No re-rolls
occurred; no performance information existed at computation time.

## Amendment 3 — no peft at runtime; custom LoRA (2026-08-09, pre-implementation)

The B200 container lock (`o1_b200/deploy/requirements.b200.lock`) contains no
`peft`. Rather than adding runtime dependencies or modifying the O1 image,
all low-rank adapter functionality (PEFT_MODE for core arms, FL7 fast
adapter, FL8 slow bank) is implemented in `training/lora.py` (custom,
norm-controllable, resettable — properties FL7 needs anyway). peft 0.19.1
(present in the local venv only) serves as a numerical oracle in a local unit
test (lazily imported; skipped where absent). Recorded before W2 began.

## Amendment 4 — §21 renumbered to §21b after §22/§23 insertion (editorial).

## Amendment 5 — training-core interface reality vs §23 (2026-08-09, W2)

Recorded by W2 (training core) after reading the implemented
`episodes/schema.py` and `episodes/render.py`. Per the §23 drift rule the
CONSUMER adapted; no producer file was edited. Nothing here changes an
objective, a weight, a metric, or a gate.

1. **FL1 is one episode to MANY examples.** §7 defines FL1 data as *isolated*
   (TASK -> ANSWER) pairs, so a single episode yields one example per
   task-bearing item. The §23 signature
   `episode_to_training_example(ep, tokenizer, arm)` still exists and behaves
   exactly as specified (it gains a keyword-only `item_index=0` selecting one
   FL1 pair, and remains the whole-episode example for FL2/FL3);
   `episode_to_training_examples(...)` (plural) returns all of them. The pool
   path `isolated_pair_to_training_example(prompt, answer, tokenizer)` is the
   primary FL1 route when items come from `data/pools.py` rather than episodes.

2. **FL3 weight for interaction index 4 (the related-task attempt).** §7 lists
   QUERY 1.0, TRANSFER 1.0, revisions (1, 3) 0.25, attempt0 0.0 and "all
   TASK/FEEDBACK/HINT/REVEAL context 0.0" but does not name index 4. It is
   frozen at **0.0** ("everything else 0.0"). Recorded before any training run.

3. **`Event` has no `meta` field.** §23 lists `meta: dict` on `Event`;
   `episodes/schema.py` carries `slot` instead and keeps per-item data in
   `Episode.items[item_id]["instance"]`. `training/tokenization.py` reads the
   item table first and per-event `meta` second, so both shapes work, and FL1
   still refuses (hard error) any item whose canonical answer cannot be
   established.

4. **`render_episode` signature and span shape.** §23 specifies
   `render_episode(ep, include_event_indices=None, upto_event=None)` returning
   `.spans` as `(char_start, char_end)` pairs. The implementation is
   `render_episode(episode, include_header=True)` plus
   `render_subset(episode, keep_indices, include_header=True)`, returning
   `EventSpan(event_index, role, item_id, interaction_index, start, end)`
   objects. The training core normalises all of these shapes and needs no
   subsetting, so no producer change is required.

5. **Answer spans are located through the event's own text.** The renderer
   emits `ATTEMPT: ANSWER: <payload>` on ONE line, i.e. the `ANSWER:` marker is
   not at a line start of the rendered text (the verifier parses the event
   text, where it is). The loss-mask builder therefore locates the event text
   inside its rendered block (it must occur exactly once) and applies the §4
   line grammar to that text. The resulting character span can be straddled by
   a single byte-level token (`ĠANSWER`); the frozen policy
   `on_straddle="expand_whitespace"` includes such a token when everything it
   contributes outside the span is whitespace — which is exactly the separator
   the model must emit — and hard-fails on any other straddle. No mask is ever
   approximated silently.

## Amendment 6 — stability thresholds left open by §17 (2026-08-09, W2)

§17 fixes the grad-norm rule (>50 for 3 consecutive steps), the divergence
rule (>2x trailing median over 200 steps) and the throughput rule (<40 % of
BENCH for 5 min), but states no threshold for "memory growth", "repeated zero
grad" or "fast-state norm explosion". `training/stability.py` declares them
now, before any run, as IMPLEMENTATION CHOICES exposed on `StabilityConfig`
and recorded inside every failure record:

- memory growth: allocated memory > 1.25x its value 200 steps earlier
  (severity OBSERVATION, not a scientific failure);
- repeated zero grad: grad norm exactly 0.0 on 10 consecutive steps;
- fast-state norm: ||s|| > 1000.0 (FL5 owns the tighter mechanism-specific
  bound and may pass its own value).

These are detector thresholds only. They can stop an arm and write a failure
record; they can never change a hyperparameter, an objective, or a gate.

## Amendment 7 — ecology / episodes / data implementation decisions (2026-08-09, W1)

Recorded by W1 (task ecology, episode format, data pre-generation) after
implementing §4, §5, §6 and §16. None of these changes an objective, a weight,
a metric, a gate, a pool size, the split, or the sealed-test policy.

1. **Instance kind lives in the seed.** `TaskInstance` has exactly the nine §4
   fields, so support / related / query / transfer cannot be a tenth field. The
   kind is carried in the two low bits of `seed`
   (`ecology/base.encode_seed` / `decode_seed`). Consequence, and the reason for
   the choice: every instance is a pure function of
   `(rule, seed, difficulty, surface_map_id)`, so `verify`,
   `structured_feedback` and `surface_remap` can reconstruct the full latent
   state of an item from the instance alone, with no side channel.

2. **`TaskFamily` gains `plausible_error` (and the `wrong_answer` wrapper).**
   §6 requires "a family-specific plausible-error sampler" for the scripted
   attempt policy; only the family knows what a plausible wrong answer is. The
   §4 interface is implemented in full and unchanged; this is an addition.
   `wrong_answer` wraps it and hard-fails rather than ever returning the truth.

3. **Structured feedback is deterministic; its `rng` argument is not consumed.**
   §4 passes `rng` to `structured_feedback` / `corrupted_feedback`, but a
   stochastic hint makes the §4 guarantee "corrupted feedback differs from the
   true feedback" unverifiable across calls (an early smoke run produced exactly
   that failure). Feedback content is therefore a pure function of
   `(rule, instance, attempt)`; any family tie-break draws from a seed derived
   from the instance. The `rng` parameter is retained for signature
   compatibility and deliberately unused.

4. **Hints are opaque, digit-free letter codes.** Every structured hint renders
   as `TAG_<CODE>` fields, contains no digits, and identifies items by their
   index in an ordering the prompt displays (registers, nodes, predicates). This
   is what makes "feedback never contains the pending canonical answer"
   machine-checkable (`ecology/base.answer_leak`, asserted on every call and
   hostile-tested). Two structural preconditions are enforced in code:
   `MAX_HINT_OPTIONS = 128` bounds the code alphabet, and no answer label may
   coincide with a code (a `("F","T")` boolean label variant was found and
   removed by the hostile fixture for exactly this reason).

5. **Episode identity excludes the poison condition.** `make_episode_id`
   covers family, split, rule, seed, difficulty, mode, structure, structured,
   reveal and arm tag but NOT the §9 poison condition, and a mode-independent
   `make_episode_key` derives items and attempt draws. The five poison
   conditions are therefore alternative HINT bodies over one and the same
   episode, and FL2's `successful` and FL3's `scripted` variants share items and
   first-attempt draws. Both contrasts are controlled rather than confounded
   with a different item sample.

6. **Certified feedback is correctness-only.** §6 permits "`CORRECT`/`INCORRECT`
   or certified structured feedback"; the implementation always uses the former,
   so all structured content lives in the uncertified HINT channel where poison
   is defined. Truthful and poisoned hints render identically.

7. **REVEAL is supported but not pre-generated.** §6 makes REVEAL present only
   in conditions that define supervised feedback (FL7). `assemble_episode`
   builds REVEAL events on demand from the stored support answers
   (`reveal=True`); the standard pools are generated with `reveal=False` rather
   than doubling every pool. The correctness-only condition is likewise the
   standard episode rendered without its HINT events
   (`episodes/render.render_subset`).

8. **§23 reconciliation for W1-owned interfaces.** §23 was added to the
   contract after W1 began; the W1 modules were brought to it ADDITIVELY, with
   no signature removed and no consumer broken:
   - `Event` gains the §23 `meta: dict` side table (env-only, never rendered,
     defaulting to `{}`; `Event.from_dict` tolerates records without it). The
     rendering label `slot` is retained, so Amendment 5 item 3 remains valid.
   - `render_episode` gains the §23 `include_event_indices` / `upto_event`
     selectors (composable) and `Rendered.char_spans` returns the §23
     `(char_start, char_end)` pairs; the richer `EventSpan` objects and
     `render_subset` remain.
   - `render_reset_query(episode, query_event_index)` is added and REFUSES any
     event that is not a query-style task, so a context-reset prompt cannot
     silently retain history.
   - `read_shard` gains the §23 `sealed_key` parameter and
     `episode_from_json(d)` is added. `read_shard` never DERIVES a key: without
     an explicit key it still refuses enciphered bytes. Because §23 places a
     decipher parameter in `data/shards.py`, the hostile fixture's invariant is
     restated where it now bites — no module under `ecology/`, `episodes/`,
     `data/` or `scripts/` performs a sealed read, derives a key for reading, or
     ships a decipher CLI — and Amendment 1's honest position is unchanged: the
     protection is procedural, the key comes from a public digest, and no
     cryptographic unopenability is claimed.

9. **Uniqueness rule made precise.** §5 requires that no `instance_id` crosses
   train/dev/test and no `rule_id` is shared across splits. Generation enforces
   the stronger operational rule that an `instance_id` may be claimed only by
   one `(split, family, episode key)` — which permits exactly the intended
   sharing between the two attempt-policy modes of one episode and nothing else
   — and that a `rule_id` is fresh within a family (deterministic redraw, hard
   failure after `RULE_DRAW_ATTEMPTS = 256`; measured worst case over both
   predeclared seeds and the full pools is 12).

## Amendment 8 — evaluation-side interface reality vs §23 (2026-08-09, W4; renumbered from a duplicate "7" by the integrator, content untouched)

The evaluation and analysis modules consume the producers' ACTUAL APIs (the §23
drift rule: consumers adapt, nobody edits another worker's files). The
deviations from the §23 sketch, and how they are absorbed, are:

1. **`render_reset_query(ep, query_event_index)` does not exist.**
   `episodes/render.py` provides the same capability as
   `render_subset(episode, keep_indices)` (its docstring names the
   context-reset evaluation as a motivating caller).
   `evaluation/learning_curve.resolve_reset_renderer` therefore accepts
   `render_reset_query` FIRST and falls back to `render_subset(ep, [idx])`,
   recording the resolved spelling in every episode record
   (`reset_render_source`). The reset prompt is header + the current query
   block + the renderer's own attempt cue, and is additionally VERIFIED
   against every prior event text (`context_reset.LeakCheckingWalker`).

2. **`render_episode` takes `include_header`, not `include_event_indices` /
   `upto_event`; spans are `EventSpan` objects, not pairs.** The walker builds
   its prompts by rendering a LIVE episode (the plan with the model's real
   attempts and the real computed feedback substituted) and locating the
   pending attempt slot with a sentinel, so it needs no `upto_event`
   parameter; `rendered_spans` accepts `.start`/`.end` objects, dicts and
   pairs.

3. **`data/shards.read_shard(path)` takes no sealed key and there is no
   `episode_from_json`.** `evaluation/fl0_base.load_episodes` calls
   `read_shard(path)` and rebuilds episodes with `Episode.from_dict`. Sealed
   shards remain unreadable from the evaluation package by construction:
   deciphering lives only in `campaign/sealed_gate.py`.

4. **Poison-condition ids.** §9 spells the untouched condition
   "correct-informative" and metric 12 calls it "clean". The data layer's
   frozen ids (`ecology/poison.POISON_CONDITIONS`) are
   `("clean", "correct-redundant", "irrelevant", "partially-misleading",
   "corrupted")`. `evaluation/metrics` imports that tuple and aliases the §9
   spellings onto it; `CLEAN_CONDITION = "clean"`. No condition is added,
   removed or renamed.

5. **`greedy_generate` and `answer_logprob` keep their §23 signatures
   exactly**; the extra capabilities they need (batch policy, detailed
   records, batch scoring) are keyword-only additions and separate functions.

Two ONLINE-evaluator policies that §6 leaves open are frozen here, and are
documented in `evaluation/learning_curve.py`:

- **Channel policy.** In BOTH the `structured` and `correctness_only`
  conditions the certified `FEEDBACK` event carries the verifier's verdict
  (`CORRECT`/`INCORRECT`) computed from the model's REAL attempt, and the
  uncertified `HINT` event carries the family's structured content, selected
  per slot by the data layer's poison schedule through
  `ecology.poison.hint_for_condition`. `correctness_only` drops HINT events
  entirely (§6: hints exist "in structured conditions"). The certified channel
  is never corrupted, and a certified event flagged `poison` is a hard error.
- **Leak-check formulation.** The context-reset check removes exactly ONE
  occurrence of the current query's own text before probing the prompt for
  prior-event text, because two items of one family can carry byte-identical
  surfaces (a 4-variable Boolean assignment repeats within an episode) and a
  naive substring probe would flag the query itself. A renderer that really
  keeps the history is still caught; a hostile fixture proves it.

## Amendment 9 — FL4–FL8 mechanism decisions (2026-08-09, W3)

Recorded by W3 (mechanisms) after implementing §7's FL4–FL8 against the
producers' ACTUAL modules (§23 drift rule: consumers adapt, nobody edits
another worker's files). Numbering starts at 9 because §23's amendment stream
already contains two independently authored Amendment 7s (W1 and W4). None of
the items below changes an objective, a weight, a metric, a gate, a threshold,
a pool size, the split, or the sealed-test policy.

1. **Final-loop hidden states come from `return_per_loop_hidden_states`.**
   §7 binds "final-layer, final-loop hidden state" without naming an API.
   `mechanisms/hidden_states.py` uses
   `OuroForCausalLM(..., return_per_loop_hidden_states=True)
   .per_loop_hidden_states[-1]`, which is the ONLY way to obtain that tensor
   AND the logits from one differentiable forward (FL5 training needs both).
   `modeling_ouro.OuroModel.forward` appends the post-`self.norm` state of each
   recurrent loop to `hidden_states_list` and returns
   `last_hidden_state = hidden_states_list[-1]`, so this tap is IDENTICAL to
   the one `evaluation/learning_curve._final_loop_hidden_states` uses online;
   a unit test pins the bitwise equality of the two. The inner-module fallback
   is used only when a wrapper hides the RLTT fields, and it REFUSES to serve a
   request that also needs logits rather than substituting another quantity.

2. **FL4 head-input contexts end at item j.** `head_input_hidden` renders
   events `0..j` with `render_subset` and refuses any context containing a
   `QUERY_TASK`/`TRANSFER_TASK` block or a query/transfer attempt. This is the
   structural form of "the head never sees future answers"; a hostile fixture
   attacks it with a feedback event moved behind the query block.

3. **FL4 target span convention.** The target is the mean per-token logprob of
   the correct QUERY answers, averaged over the three query ITEMS (each item's
   own per-token mean), both halves computed with W4's
   `evaluation.scoring.answer_logprob_batch`. Because that function requires a
   STRICTLY token-aligned character span while W2's loss masks use the frozen
   `expand_whitespace` straddle policy (`ĠANSWER` carries the separator space),
   `value_head.token_aligned_answer_span` selects exactly the tokens W2 would
   put loss on and hands the scorer THEIR character boundaries. The FL4 target
   therefore measures the log-likelihood of exactly the tokens FL3 trains on,
   with no re-alignment inside the scorer. Events after the last query attempt
   are dropped from the scored rendering: under a causal LM they cannot change
   the query log-probabilities, so the score is numerically identical and the
   forward is shorter.

4. **FL4 constants not fixed by §7** (all frozen here, before any run, and
   documented in `mechanisms/value_head.py`): ranking-hinge margin 1.0 (the
   `torch.nn.MarginRankingLoss` default; predictions are unbounded, so the
   margin is a scale convention and the auxiliary z-scored MSE fixes the
   scale); targets z-scored with the TRAINING-set mean/std, stored in the head
   as buffers so DEV predictions share the scale (calibration slope/intercept
   are therefore reported against the z-scored target); head optimizer AdamW
   with the §8 campaign constants and LR 1e-3 (§8's grid binds the BACKBONE
   arms); default 40 epochs, 8 episode-groups per batch (both caller-settable
   and recorded per run).

5. **FL5 module parameterisation.** `W_in` and `W_p` are BIAS-FREE (matching
   §7's equations and its ~25 M parameter estimate); `GRUCell` keeps its
   standard biases. `hidden_size` is a constructor parameter (2048 for the
   frozen checkpoint, 64 for the tiny mechanics model). `rho` must be supplied
   EXACTLY ONCE — either as an embedding matrix (rho = 2 x median row L2 norm,
   computed once and recorded in `config()`/`config_hash()`) or as an explicit
   value for checkpoint reload — so it can never be silently defaulted. The
   per-vector clamp is `P_i * min(1, rho/||P_i||)`, differentiable and exact at
   the boundary. FL5's mechanism-specific fast-state norm bound (Amendment 6
   left the value to FL5) is 64.0, passed through W2's `fast_state_norm_max`.

6. **FL5 training is a segmented forward, and BOTH arms use it.** §7 says
   FAST_STATE_ON and FAST_STATE_OFF share the FL3 objective, data order and
   seeds and that OFF "disables injection AND state usage". `fl5_training.py`
   renders and tokenises the episode ONCE, partitions its tokens into segments
   cut after each run of feedback events (each token belongs to the segment
   containing its start offset, so the partition is exact), and forwards each
   segment as `[prefix; segment tokens]`. Consequence, and the reason for the
   choice: the ONLY channel from an earlier interaction to a later segment is
   the state `s`, so the ON/OFF contrast is exactly "the state carried the
   learning". The FL3-weighted NLL is accumulated over ALL segments (revision
   spans contribute from their own segments) and reduced as `OBJ_EPISODE_MEAN`,
   i.e. the objective is identical to FL3's, not merely similar. A weighted
   token at segment-local position 0 is a hard error, as in W2.

7. **FL5 plugs into W2 by reuse, not by copy.** `run_training_arm` is built
   around `examples_to_batch` plus one whole-sequence forward and cannot express
   a stateful segmented forward through its external step hook, so FL5 runs its
   own loop and REUSES W2's `build_optimizer`, `build_scheduler`, `epoch_order`,
   `ComputeLedger`, `StabilityMonitor` and `save_checkpoint`. `FL5TrainConfig`
   is duck-typed for `build_optimizer`/`build_scheduler` (they read only
   `learning_rate`, `betas`, `eps`, `weight_decay`, `warmup_steps`, `updates`)
   rather than registering a fake core arm in `training/arms.py`.

8. **FL7 inner-loop supervision is enforced, not assumed.** The supervised
   example is the SUPPORT task event plus an answer event carrying the REVEALED
   canonical answer, rendered through W1's renderer and masked by W2's
   `episode_to_training_examples(..., arm="FL1")` (loss on the `ANSWER:` line
   only). `supervised_example` REFUSES any item without a `REVEAL` event or
   with a non-support role, so query/transfer labels cannot enter the inner
   loop. Because the fast delta lives in the model, exactly one FL7 episode may
   be in flight at a time (`attach_lora` refuses a second attachment, so the
   mistake cannot be made silently).

9. **FL8's "separate slow bank" is W2's merged-component list.** `merge_scaled_`
   appends a FROZEN low-rank component to a layer; `zero_lora_` explicitly does
   not touch those components. "Merge into a separate slow bank, then clear the
   fast state" is therefore `merge_scaled_` followed by `zero_lora_`, and
   `slow_bank_report` reports the bank alone. A recorded per-episode delta is
   carried by `FastDelta`, which exposes exactly the three attributes
   `merge_scaled_` reads from a source handle; restoring a delta with
   `load_lora_state_` instead would CLEAR the destination's merged components
   (i.e. wipe the bank), which is why the delta is carried separately. The
   merge arithmetic itself is W2's function, unmodified.

10. **FL8 value-gated selection rule.** §7 says "selection = value gate" but
    not how per-item gate decisions become a per-episode selection. Frozen
    rule: an episode's delta is merged iff the gate ADMITTED AT LEAST ONE
    update in that episode (`n_admitted > 0`). A delta from zero admitted
    updates is exactly the zero delta, so the rule only makes the bookkeeping
    explicit. Sealed families are refused at every consolidation entry point,
    including after a record has already been accepted.

11. **Numerical tolerance in the FL7 norm-bound fixture.**
    `clip_lora_frobenius_` rescales float32 `B` factors in place, so the
    re-measured global norm carries float32 rounding (~1e-8 relative; float32
    eps is 1.2e-7). The hostile fixture allows `1e-6 * beta` for that rounding
    and separately asserts that the PRE-clip norms exceed beta by orders of
    magnitude, so a real violation cannot hide inside the tolerance. The bound
    itself is not relaxed.

## Amendment 10 — campaign / deploy / scripts implementation decisions (2026-08-09, W5)

Recorded by W5 (campaign layer, deploy, packaging scripts) after implementing
§10–§13, §15, §18 and §19 against the producers' ACTUAL modules (§23 drift
rule: consumers adapt, nobody edits another worker's files). Numbering
continues at 10 because the amendment stream already contains two
independently authored Amendment 7s (W1 and W4). None of the items below
changes an objective, a weight, a metric, a promotion threshold, a pool size,
the split, the safety factor, the transfer reserve, or the sealed-test policy.

1. **The stage table contains two operational stages that are not §7 rungs.**
   `BENCH` is §11's "runtime/throughput validation" (priority 1) and
   `DEV_GRID` is §8's mandatory two-learning-rate FL3 grid, which must complete
   before any core arm because it is what DEFINES the core learning rate. Both
   sit inside the core-comparison allocation that §11's affordability rule
   already projects ("grid + 3 arms + evals"). A `SECOND_SEED` stage carries
   §11 priority 9 (additional predeclared seeds), and `SEALED_EVAL` — the
   single §12 opening — is LAST, so it cannot precede a development decision.

2. **BENCH is admitted against a DECLARED budget, not a measured one.** Every
   other stage is projected from BENCH's measurements; BENCH cannot be, because
   it IS the measurement. `FL_BUDGET_POLICY.json` therefore carries
   `bench_declared_budget_seconds = 900` and BENCH is admitted against it under
   the same `× 1.25 + reserve` inequality. This is the ONE projection in the
   campaign that is declared rather than measured, and it is labelled
   `BENCH_DECLARED` in the journal.

3. **BENCH measures the evaluation cost on TRAIN episodes.** §11 requires the
   throughput benchmark to run "on a non-evaluation training shard". The
   affordability projection also needs a per-episode EVALUATION cost, so BENCH
   times the real W4 walker on TRAIN episodes as well. No DEVELOPMENT or
   SEALED_TEST episode is consumed by the benchmark. The evaluation cost is
   scope-independent (the walker only runs greedy inference), so it is measured
   once and attached to every scope's measurement.

4. **FULL_MODEL_MODE is INELIGIBLE when unmeasured.** §11's rule needs a FULL
   projection to decide. BENCH measures both scopes when both are requested; if
   a FULL measurement is absent, the rule returns "ineligible" rather than
   estimating one, so an unmeasured scope can never be selected.

5. **Per-stage evaluation maxima are frozen in `stage_definitions.py`.** The
   preregistration leaves "evaluation batch size and resulting eval episode
   counts" B200-derived. Only the BATCH SIZE is genuinely B200-derived (it is
   the output of the §22 equivalence gate); the AMOUNT of evidence is frozen
   here (`EVAL_EPISODES_PER_STAGE`) so it cannot drift with available time.

6. **`max_new_tokens` is frozen at 64 and the override is rehearsal-only.**
   `_generation_config` refuses a shortened decode budget unless the context is
   explicitly marked `rehearsal`; the dress rehearsal uses 8 and records both
   values in its report.

7. **The dress rehearsal uses a labelled miniature ladder.** §18 sanctions
   "tiny model, tiny pools, seconds-scale budgets", which the frozen U ladder
   {600, 1200, 2400, 4800} cannot express on a CPU tiny model.
   `plan_core_comparison(ladder=…)` REFUSES any non-frozen ladder unless
   `rehearsal=True` is passed explicitly, stamps the plan
   `REHEARSAL_LADDER_OVERRIDE`, and every rehearsal arm configuration is tagged
   `STAGE_SMOKE` ("local mechanics only; never a scientific result"). The
   frozen ladder and the frozen learning-rate grid remain fully enforced on the
   CORE/GRID stage tags, which the unit suite exercises directly. The
   rehearsal's nominal authorized budget is 3600 s rather than "seconds-scale"
   because the 1200 s transfer reserve is NOT scaled down; the wall-clock cost
   of the rehearsal itself is ~4 minutes.

8. **`VERIFY_O1_RECORDS` runs through a separate hash-only custodian.** §13
   requires FL to refuse every O1 path AND to verify O1's records against O1's
   own manifests. Those are in tension, so `session_supervisor.O1RecordCustodian`
   is a deliberately crippled reader: it accepts ONLY paths under the declared
   O1 roots, exposes no method that returns file content, and parses nothing
   but path/digest pairs. The FL isolation guard continues to refuse those
   paths for all FL work. Hashing bytes is not reading outcomes; every path the
   custodian touches is journalled, and a unit test pins the absence of a
   content-returning method.

9. **The sealed opening ledger carries an audit nonce.** Two openings in the
   same wall-clock second would otherwise produce byte-identical entries, so a
   deleted-and-recreated ledger would be indistinguishable from the original.
   `opening_nonce` (16 random bytes) makes the opening-entry hash recorded
   INSIDE every read-only sealed result a real cross-check. It is an audit
   value only; no code path branches on it, so §20's ban on nondeterminism in
   scientific paths is untouched. Amendment 1's honest position is unchanged:
   the protection is procedural, and no cryptographic unopenability is claimed.

10. **Checkpoint best-DEV selection lives in `campaign/promotion.py`.** §15
    fixes the rule (max DEV macro-AULC at scheduled evaluations) but not its
    home. `select_best_dev_checkpoint` consumes the same `DevMetrics` object as
    every promotion rule, which structurally cannot hold a sealed record, and
    breaks ties toward the EARLIER evaluation so the choice is deterministic.

11. **The secret deny-list scrub covers the whole "tokeniz…" word family.** The
    O1 packager scrubs the word `tokenizer` before matching (a tokenizer is a
    model asset, not a credential); this package also contains
    `tokenization.py`, so the scrub is `tokeniz(er|ation|e|ing|…)`. A bare
    `token` in a file name (`hf_token`, `api_token`) still trips the deny-list.

12. **Pregenerated data is git-ignored; its manifests are not.** The raw shards
    (~360 MB) are bitwise reproducible from `scripts/pregenerate_all.py`, so the
    repository-root `.gitignore` ignores `artifacts_fl/pregen*/*` and
    re-includes `artifacts_fl/pregen*/MANIFESTS/`.
    `scripts/package_release.py` mirrors `PREGEN_MANIFEST.json`,
    `SHARD_SUMS.json` and `family_split_manifest.json` into that directory,
    hash-verified on every run, so packaging, manifest filling and review can
    always find them. `foundation_learner/.gitignore` cannot express this
    itself (the data lives outside the package directory) and says so.

13. **`SHA256SUMS` covers the whole content manifest, rooted at the repository
    root.** It lists `foundation_learner/**` and `artifacts_fl/pregen/**`
    (minus `reports/local_runs/`, `__pycache__/`, `*.pyc`, `*.tmp` and the
    release artefacts) in the O1 two-space format, with an EXACT-coverage
    bijection check in both directions. When the data is absent from a checkout,
    `scripts/run_all_tests.py` reports the checksum suite as SKIPPED with that
    reason rather than as a pass.

14. **`scripts/run_all_tests.py --fast` names what it omits.** The measured
    slow modules (the long model walkers) are listed in `SLOW_NODE_IDS` and are
    recorded in `TEST_REPORT.json` as `skipped_modules`, so a fast run can never
    be mistaken for a full one. No test is weakened or deleted to obtain speed.

15. **`campaign/stage_definitions.py` names the FL4–FL8 entry points as dotted
    path strings** (`foundation_learner.mechanisms.<module>:run_fl<N>_stage`),
    resolved lazily at stage time. A MISSING mechanisms module is recorded as
    `SKIPPED_MECHANISMS`; a PRESENT module without the named entry point is an
    integration ERROR, never a silent skip. This keeps the campaign layer
    independent of the mechanism API while it is still being written, at the
    cost of binding those five entry-point names — which W3 may satisfy with a
    thin adapter if its internal API differs.

## Amendment 11 — FL4–FL8 campaign stage adapters (2026-08-09, W3)

Recorded by W3 after wiring the mechanism rungs into W5's declarative stage
table. The table resolves each rung lazily through a dotted entry point
(`foundation_learner.mechanisms.<module>:run_fl<N>_stage`) with the signature
`run_flN_stage(ctx: StageContext, stage: StageDefinition) -> dict`; W3 supplies
those five adapters plus `mechanisms/stage_support.py`, which holds everything
they share. No campaign file was edited. None of the items below changes an
objective, a weight, a metric, a promotion threshold, the split, or the
sealed-test policy.

1. **FL4's DEVELOPMENT evaluation needs DEVELOPMENT targets.** §7 fixes the
   head's TRAINING targets as "TRAIN families only" and, in the same paragraph,
   requires a DEV evaluation of the head (Spearman, pairwise ranking accuracy,
   calibration, top-1 regret, versus surface heuristics) — which is only
   computable if the same realized-value quantity is computed on DEVELOPMENT
   episodes. `assert_target_split` therefore replaces the previous all-or-
   nothing check: the default admits TRAIN only, the FL4 stage's DEV evaluation
   passes `allowed_splits=TARGET_SPLITS_DEV_EVAL`, and SEALED_TEST is filtered
   out of ANY caller-supplied list, so it is refused under every setting. This
   strictly tightens the old `require_train_split=False` path, which performed
   no split check at all.

2. **FL4 targets are computed with the restored FL3 checkpoint.** §7 says the
   targets are computed "with the final FL3 checkpoint", but `StageContext`
   hands out a FRESH BASE bundle by design (§7's per-arm fresh-load rule).
   `stage_support.arm_checkpoint_state` restores the arm's `final` checkpoint
   into that fresh bundle with W2's own `load_checkpoint` +
   `load_lora_state_` / `load_state_dict`. When no FL3 checkpoint exists in the
   session (a rehearsal, or FL3 skipped) the base checkpoint is used and the
   payload records `target_model = BASE_CHECKPOINT_NO_TRAINED_ARM` with the
   reason — it is never passed off as the FL3 model.

3. **Payload → promotion field mapping (frozen).** `stage_definitions.fl6_entry`
   reads `ctx.results["FL4"]["pairwise_ranking_accuracy"]`, so FL4 surfaces that
   exact key at the TOP LEVEL of its payload; `fl7_entry` reads
   `status == "COMPLETE"`; `promote_fl4`'s data-sufficiency evidence is
   published as `ctx.extra["fl4_scoreable_items_per_episode"]`; `fl8_entry`
   reads `ctx.extra["persistence_evidence"]["point"]` / `["ci"]`, which FL5
   (and FL7, when FL5 published nothing defined) writes.

4. **The FL8 persistence estimand.** §10 admits FL8 on "FL5 or FL7 context-reset
   persistence > 0 with CI excluding 0 on DEV" without naming the statistic.
   Frozen here: the per-episode retained improvement
   `mean(R_k, k >= reset_from_index) - R_0` under the context-reset condition,
   compared between the mechanism arm and its MATCHED no-mechanism arm on the
   SAME episodes, with §14's paired family-clustered bootstrap (one shared
   resample plan, 10,000 replicates). The difference form is what makes the
   claim about the mechanism rather than about the task.

5. **FL7's supervised episodes are re-assembled with `reveal=True`.** The
   standard pools are pre-generated with `reveal=False` (Amendment 7 item 7),
   but FL7's inner loop is REVEAL-supervised. `stage_support.reveal_variant`
   calls W1's `assemble_episode` with the episode's own family, rule spec,
   seed, difficulty, mode, condition and structured flag and `reveal=True`;
   since assembly is a pure function of exactly those, the result carries the
   SAME rule, items and scripted attempts plus the REVEAL events. Nothing is
   fabricated.

6. **A missing FL4 head is a recorded skip, never a fabricated gate.** FL6 (and
   FL7's gated variant, and FL8's `value_gated` mode) needs the FROZEN FL4
   head. FL4 publishes it in-process (`ctx.extra["fl4_value_head"]`) and
   persists its tensors next to the report through the guard. When it is
   absent, FL6 returns `status = SKIPPED_MISSING_INPUT`, FL7 runs its ungated
   variant only, and FL8's `value_gated` mode records `gate_available = false`;
   an untrained head is never substituted, because a random gate presented as a
   mechanism would be a fabricated capability.

7. **Rehearsal-scale reductions are declared, never silent.** Every adapter
   takes its scale from `ctx.eval_episode_cap` / `ctx.train_example_cap` /
   `ctx.rehearsal`, and every payload carries `rehearsal` plus the note "DRESS
   REHEARSAL: mechanics only, never a scientific result". Two rehearsal
   defaults are declared in code: mechanism training arms use 2 updates
   (`stage_support.REHEARSAL_UPDATES`), and FL6 walks the two ENDS of the frozen
   §9 condition list (`clean`, `corrupted`) instead of all five
   (`value_gating.REHEARSAL_POISON_CONDITIONS`). A real run uses `ctx.updates`
   (required; it is never guessed) and every frozen condition present in the
   data. `ctx.extra` keys (`fl4_train_episodes`, `fl5_updates`,
   `fl6_conditions`, `fl8_chains`, `fl8_modes`, ...) let the campaign set any of
   these explicitly, and the chosen values are recorded in the payload.

8. **FL7 runs one episode at a time.** The fast delta lives IN the model, so
   several FL7 episodes cannot share a bundle; `attach_lora` refuses a second
   attachment, so the mistake cannot be made silently. FL7 therefore evaluates
   episode by episode and uses one fresh bundle per variant plus one for the
   matched no-adapter baseline. This is a real throughput property of
   parameter-level adaptation, not an implementation shortcut.

9. **Frozen campaign conventions are imported, not re-implemented.** The
   adapters use `campaign.stage_definitions`'s guarded episode loader, its
   exact-verifier `env_factory` and its `_generation_config` (the frozen
   64-token decode budget, rehearsal-overridable). Duplicating those rules
   would let two copies of a FROZEN constant drift; if the campaign renames the
   helper the stage raises loudly instead of decoding with a different budget.
   All imports are made INSIDE the adapter functions, so importing
   `mechanisms` never pulls in the campaign package and the lazy stage table
   stays lazy.
