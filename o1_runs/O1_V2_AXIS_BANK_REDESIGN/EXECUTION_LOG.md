# O1_V2_AXIS_BANK_REDESIGN execution log

## 2026-07-29 — immutable predecessors and isolated worktree

- Began from accepted diagnosis commit
  `38c474ebea3c6173d48d659d3edc0317c36acfd4` in isolated worktree
  `/tmp/o1_v2_axis_bank_redesign`.
- Verified preservation manifest for failed `O1_REAL_001`.
- Did not modify the dirty original repository or any failed tensor/capture.

## 2026-07-29 — real axis reconstruction

- Copied A1/A2 and both exact induced-update matrices byte-for-byte into a new
  source bundle.
- Computed A3/A4 as uncentered raw means followed by unit-RMS normalization.
- Preserved the eight-task order and exact capture parity/checkpoint/adapter
  bindings.
- Built and verified the four-axis tensor and deterministic Gram-matched
  random bank. Axis verdict: `SEALABLE`.

## 2026-07-29 — D2_OUTCOME_AXIS_V2

- Audited exact s3b2 source, task/domain class support, task-balanced direction,
  clustered bootstrap, leave-one-positive-task-out, domain-held-out stability,
  and per-task contributions.
- Verdict: `INSUFFICIENT_STABILITY`.
- After freezing prospective cohorts, task-ID leakage audit passed with zero
  overlap. D2 remains excluded from the primary bank.

## 2026-07-29 — package/runtime/preregistration

- Generalized package semantics to v2.0 and removed the shared-PC gate.
- Added exact mean-update/source-matrix gates and adversarial fixtures.
- Resolved physical L3_24 as module index 23 from accepted capture code;
  corrected the carried manifest ambiguity before precommit.
- Added frozen symbolic Horizon task generator, prompt, strict parser,
  independent truth-table verifier, generation runtime, and cohort builder.
- Froze 96 calibration tasks and a disjoint 2400-task confirmatory candidate
  population. No model outcome was generated or inspected.
- Full package suite passed: 53/53 core, 17/17 calibration, 15/15 v2
  integration, 7/7 CLI, 15/15 axis adversarial; exact power
  324/817/2238/337; no package bytecode.

## Current chronology status

No O1 calibration or confirmation has occurred. The next permitted step is to
commit the complete design, create the calibration precommit against that
commit and exact runtime artifacts, externally push the precommit commit, and
stop before calibration.

## 2026-07-29 — calibration precommit created

- Package/design commit:
  `47e7ed5b5d4d07235522e714c3970db6ec0887cb`.
- Frozen precommit-input commit:
  `945095843290ff0039bfd0986861c970516f8e75`.
- Created `CALIBRATION_PRECOMMIT.json` while the worktree was clean and while
  `calibration_records.jsonl` did not exist.
- Precommit SHA-256:
  `910176b666a47da75273b72bf5bfef8484faeb223ddab823ab02ccd35f47def2`.
- Verified the precommit against all 61 bound calibration configuration paths.
- External Git commitment is the remaining chronology gate. No calibration
  command has been launched.

## 2026-07-29 — external chronology commitment verified

- Committed the calibration precommit in Git commit
  `70651d529ad4ecf204c81dc88854f6f376c420c6`.
- Pushed branch `o1-v2-axis-bank-redesign` to the authenticated `origin`
  remote. The server reports that the repository moved to
  `VykosMolt/Hidden-State-Evaluator` and accepted the branch there.
- Independently queried `refs/heads/o1-v2-axis-bank-redesign` with
  `git ls-remote`; it advertised the exact commit above.
- Local upstream-tracking metadata could not be written because the shared
  main repository `.git` directory is read-only; this did not affect the
  successful remote push or remote-ref verification.
- Stopped before calibration. `calibration_records.jsonl` remains absent.

## 2026-07-29 — verified package archive

- Created `o1_packages/O1_oracle_reachability_v2.0.0_verified.zip` from the
  already committed and verified `o1_v200` source tree.
- Archive SHA-256:
  `3efe914c41d5fe185e0a557cf2bc75b47fc41633b178599083701b661d87f6d0`.
- Adjacent `.sha256` created; `unzip -t` reported no errors.
- No package source, precommit field, axis tensor, cohort, or calibration state
  changed.

## 2026-07-31 — v2.0.0 calibration precommit superseded before any calibration

- An independent hostile audit (GO_AFTER_MECHANICAL_FIXES) found defects in
  sealed, precommit-bound artifacts: the lax commitment parser, the unsealed
  generation-to-record orchestration and transport computation, the
  unimplemented pool-capped confirmatory-cohort construction, the medium
  band-edge inclusivity contradiction, the degenerate budget_mde return, and
  the loss of the /tmp worktree behind RUNTIME_ARTIFACT_PATHS.json.
- No calibration record was ever generated under the superseded precommit.
- The v2.0.0 precommit was moved to
  `CALIBRATION_PRECOMMIT.v2.0.0.superseded.json` and the supersession recorded
  in `ATTEMPT_LEDGER.jsonl`.
- Package v2.1.0 (`o1_packages/O1_oracle_reachability_v2.1.0_source/o1_v210`)
  implements the fixes with new fixture suites (53/53 core, 22/22 calibration,
  15/15 integration, 7/7 CLI, 58/58 parser adversarial, 11/11 orchestrator,
  16/16 cohort builder, 15/15 axis verifier) and unchanged scientific design:
  bank bytes, endpoint, K=8, CRN, alpha grid/rule, bands, delta target, power
  machinery, cohorts, and drop order are identical.
- `O1_V2_INTERPRETATION_ADDENDUM.md` binds the interpretation of A3/A4, null
  scope, and positive-primary scope before any outcome exists.
- `RUNTIME_ARTIFACT_PATHS.json` regenerated with durable-worktree paths; the
  axis package, cohorts, and seed matrix are byte-identical to their sealed
  v2 values.

## 2026-07-31 — replacement v2.1 calibration precommit created

- Design/package commit `4a4bd3344362b92471cae97f8211e4998882addd`;
  frozen-inputs commit `d5c8c39` (clean worktree at creation).
- `CALIBRATION_PRECOMMIT.json` binds 68 calibration configuration paths,
  including the new sealed orchestrator, transport module, runtime library
  versions, candidate-pool hash, and throughput plausibility ceiling.
- `calibration_records.jsonl` did not exist at creation.
- Axis package, calibration cohort, candidate pool, and checkpoint all hash to
  their unchanged v2 values.
- External Git commitment is the remaining chronology gate; no calibration
  command has been launched.
