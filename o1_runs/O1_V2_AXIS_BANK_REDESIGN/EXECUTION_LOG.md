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

