# O1 v2.1 — independent scientific review findings and decisions (2026-08-28)

Four independent adversarial passes over the sealed package: two by an external
model (OpenAI gpt-5.6 "Sol", xhigh) covering intervention/transport/parser and
axis/cohorts/seeds, one by a Claude Opus specialist that adjudicated them and
reproduced one finding executably, and the maintainer's own pass.

**Timing.** `STATUS` was pre-rental, no calibration outcome and no materialized
confirmatory manifest existed, and `EXTERNAL_CHRONOLOGY.json` shows the
precommit externally anchored (`ls-remote` advertised + blobs fetched back).
**Every finding below is prospective.** Nothing here retracts a result, because
no result exists.

**Scope note.** The paid accelerator session runs CALIBRATION only. Nothing in
`o1_b200/runner` calls `run_o1_primary`; the confirmatory analysis is a
separate offline stage. Findings 1, 4 and 5 therefore gate that later stage,
not the rental.

---

## 1. CLOSED — confirmatory cohort selection freedom (was MATERIAL)

Nothing verified the confirmatory manifest against
`cohort_allocation.materialize()`. `gate_G23` checked only task id / content
hash / setting; `gate_G11` checked stratum and eligible setting but neither the
per-setting quota nor the sealed-rank prefix; `load_task_manifest` did not even
require `sealed_rank_within_generator_setting`. An arbitrary eligible-pool
subset with arbitrary stratum ranks passed every gate — **reproduced with a
working proof of concept**. This defeated the property PREREGISTRATION §14.4
claims: "materializes the cohort with zero operator discretion".

The selection is outcome-blind (no model outcome exists for any pool task when
the cohort is built), so it is not selection on measured difficulty. It is free
selection on visible task content, plus free choice of the stratum ranks that
drive the G13 Latin-square action-to-stream map.

**Closed by** `o1_b200/runner/verify_confirmatory_cohort.py` (gate
`G25_COHORT_DERIVATION`) — re-derives the cohort using the SEALED
`allocate()`/`materialize()` (imported, never reimplemented) and requires exact
identity on `(task_id, stratum, sealed_rank_within_stratum)`. It lives OUTSIDE
the sealed package deliberately: the v2.1 zip is byte-pinned and externally
chronology-anchored, so editing it would mint a new package identity and
invalidate the precommit that anchors the design's priority. This is the same
seam the batched calibration backend uses — sealed arithmetic stays
authoritative, verification wraps it.

**It is a REQUIRED precondition of the confirmatory analysis.** Regression
tests: `o1_b200/tests/test_confirmatory_cohort_gate.py` (7 checks, registered
in `run_all_b200_tests.MODULES`), including the reproduced substitution and a
rank permutation.

## 2. DECISION REQUIRED — the max-of-8 estimand is asymmetric (MATERIAL)

`H_reach = mean_g[max_i R_structured − max_i R_baseline]`. Verified in code
(`o1_analysis.oracle_per_task`, `run_o1_v2_orchestrator`): the structured `max`
ranges over **8 distinct interventions** (4 axes × 2 signs, Latin-square
assigned), while the baseline `max` ranges over **8 stochastic draws of one
null**. `max` is variance-sensitive, so any perturbation that merely widens the
output distribution raises `E[max]` **with no steering content whatsoever**.
`selection_headroom` is precisely that diversity quantity, and it is a declared
secondary.

The identifying control is the `random` arm — same 4×2 geometry, same alpha,
Gram-matched random basis. It is **first on the budget drop list**
(`o1_analysis.budget_fallback` → `dropped_arms=["random", …]`; §5 "an optional
budget arm"; §10 fixes the drop order as random, then secondary magnitude).

**Consequence, stated exactly:** with `random` dropped, `H_reach > 0` licenses
only *"an 8-fold structured perturbation bank raises oracle reachability"* — it
does **not** license *"the reconstructed axes steer the model toward correct
answers"*. The second reading is the scientifically interesting one and the one
the axis-reconstruction framing invites, and it is **unidentified** without
`random`.

**Decision recorded:** the drop order is preregistered and lives in sealed
code, so it is not changed here. Instead: **if the `random` arm is dropped, the
report must not use axis-steering language**, and `H_reach` must be written as
a perturbation-bank reachability result. Promoting `random` ahead of
`structured_secondary` in the drop order would require a v2.2 package and a
re-anchored precommit; the secondary magnitude answers a dose question while
`random` answers the identification question, so if a future re-seal happens
for any other reason, that reordering should ride along.

## 3. NON-ISSUE — McNemar and generator-setting clustering

One reviewer argued the exact McNemar test is anti-conservative because there
are only three generator settings. **Refuted on adjudication:** tasks are drawn
from a fixed, seeded, parameter-free generator, so within a setting there is no
estimated shared parameter through which two tasks could co-vary; per-task
stream seeds are independent SHA-256 derivations; and the exact test conditions
on `n_disc` and tests `n10 ~ Bin(n_disc, ½)` under the sharp null. Validity
holds. Cross-setting heterogeneity costs power, not level.

**What IS true is a generalization limit:** every interval resamples tasks, so
all intervals are conditional on these three settings, and no per-setting
breakdown of `H_reach` is produced. **Required in the report:** `endpoint()`
restricted to each of `rules_2/3/4` as a declared secondary, the three
per-setting `(n10, n01)` cells next to the pooled result, and claim scope
written as "the sealed cohort drawn from these three settings".

## 4. RESIDUAL, recorded — `logits_finite` is an unverifiable input

It is type-checked but never recomputed, and it hard-fails alpha coherence.
Unlike the repetition flag it is **not recomputable from any stored artifact** —
it is a property of the forward pass — so recomputation is not an available
fix. Exposure is bounded: the exploitable direction is forging `True`, but
generation breaks with `finish_reason="nonfinite_logits"` the moment logits go
non-finite, and empty text independently hard-fails, so a forged `True` also
requires fabricating plausible text (i.e. wholesale row fabrication, a
different threat model). It is dropped from confirmatory records entirely, so
it touches only alpha selection.

**Decision:** reclassified as an unverifiable runtime attestation, alongside
`elapsed_wall_seconds`. If a v2.2 package is ever minted, require
`finish_reason` in the calibration schema and cross-check
`logits_finite == (finish_reason != "nonfinite_logits")`.

## 5. RESIDUAL, recorded — calibration does not enforce the Latin square

Confirmatory enforces it (`gate_G13`); calibration validation checks only that
streams are a permutation of 0..7 with the four antipodal signed pairs present.
**Mitigation verified:** calibration seeds are NOT free — every seed is
recomputed with `derive_stream_seed(master_seed, task_id, stream_index)` and
mismatches are rejected, so each signed action occupies a distinct seed-pinned
stream. The only unverified freedom is the *pairing*, and exploiting it requires
physically regenerating calibration under a different permutation at 8×-plus
compute against a precommit and an attempt ledger — a garden of forking paths,
not a cheap edit. Affects per-direction diagnostics, not the aggregate endpoint.

## 6. RESIDUAL, recorded — the injectable `backend=` seam

Both orchestration entry points accept `backend=`; `RealBackend` — where the
runtime-version assertions and the sealed model load live — is constructed only
when `backend is None`. The CLI never exposes it, so this is a Python-API seam.

**The live use argues the other way:** `o1_b200/runner/calibration_backend.py`
refuses to load a model without `manifest_raw`, then calls the SEALED
`assert_runtime_versions` and the SEALED `_load_model`. The one deliberate use
reproduces `RealBackend`'s checks rather than bypassing them. The residual is
that no record attests which backend produced it, and `RUN_PROVENANCE.json` is
an operator declaration rather than a runtime attestation.

**Recommended if a v2.2 is ever minted:** replay a preregistered pseudorandom
2% subsample through a freshly constructed `RealBackend` and require exact
token-id equality — the baseline-bank resume path already does exactly this
kind of byte-exact reproduction check.

## 7. Confirmed sound (adversarial pass found no defect)

Alpha selection is correctness-blind (`_coherence_for_alpha` reads only
`logits_finite`, `catastrophic_repetition`, empty text and `well_formed`; no
`verifier_correct`). Transport is diagnostic-only and selects nothing — which
is why the known bf16 rho inflation corrupts a REPORTED DIAGNOSTIC, not the
endpoint and not any decision. Axis construction is disjoint from evaluation by
item AND distribution (axis references are ARC-Challenge/MMLU; evaluation tasks
are synthetic propositional logic). Scoring is genuinely recomputed from stored
text, never trusted. `first_divergence` ignores upstream-supplied values.
Coupling survival filters nothing. `paired_arrays` refuses unequal task sets
rather than silently changing G. `_verdict` refuses to read a nonsignificant
test as a negative oracle. The seal is single-write and hash-chained. The
zero-alpha parity gate is fully armed in the frozen manifest
(`token_identity_required: true`, `min_token_identity_rate: 1.0`), and
`power.alpha_level` is 0.05, matching `endpoint()`.
