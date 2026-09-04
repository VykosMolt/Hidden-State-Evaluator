# JLens operational handoff

Status: `SUPPORTED_LOCAL_OBSERVATIONAL_RESULT_UNFROZEN`.

This handoff supersedes the Opus-era handoff preserved under
`history/2026-09-04-opus/`. It records operational state only; scientific
values and claim statuses live in `artifacts/jlens/final/analysis.json`.

## Established now

- Current-source M1–M5 validation passes on the local RTX 5070 Ti Laptop GPU.
  The 18 required equality comparisons are bit-exact, and M4 includes human
  loop 1 (`ut=0`).
- The CPU regression suites and the five CUDA index-map tests pass.
- The 648-prompt hidden-state cache and J/logit-lens score arrays were
  regenerated against current model/source bytes. Every numerical array is
  bit-identical to the preserved pre-repair copy.
- Five-fold probe fitting is deterministic across two complete runs. The fair
  population is 576 prompts from 39 unordered-pair clusters.
- Main counts, readout contrasts, exit agreement, transport moments, and probe
  point estimates pass a separate arithmetic verifier that does not import the
  report generators.
- The original documents and peer notes are byte-preserved and hash-checked.
- Fit, merge, evaluate, report, and publication paths now use atomic outputs and
  hash-bound sidecars/receipts. Unsafe or incomplete resume state fails closed.

## Scientific endpoint

The retained n=80 final-exit average Jacobian lens is below the logit lens for
task-defined multihop intermediate-token readout at loops 1–3 and arithmetic at
loop 1. This is local, estimator-specific evidence. It is not evidence that the
intermediate is absent or causally erased.

The local/eventual convergence prediction is `REFUTED_PRE_FINAL`. Cross-loop
transfer is row-dominated on multihop at n=32 and n=80, but the cause remains
unidentified. Transport scatter is modeled from nested averages, not directly
measured per-prompt Jacobians.

## Deliberately open

- 1000-prompt or replicated fixed-n fitting: `NOT_RUN`.
- Matched position-by-reduction 2x2 estimator control: `NOT_RUN`.
- Direct per-prompt sufficient statistics: `NOT_RETAINED`.
- Matched recurrent-model/ablation evidence for architecture causality:
  `NOT_ESTABLISHED`.
- B300 result from the failed paid run: `NOT_RETAINED`.
- Submission readiness: `NOT_ESTABLISHED`.

None of these states may be upgraded from filenames, elapsed time, or a future
pod transcript.

## Rental boundary

No cloud pod was launched during this repair. The old recipe is rejected. The
replacement controller and immutable publisher are covered by offline fault
injection, but paid execution remains a separate authorized campaign requiring
a positive balance, explicit budget/runtime limits, detached supervision, and
verified termination. `HANDOFF_B300.md` is a blocked-state notice, not a launch
guide.

## Safe next command

```bash
venv/bin/python src/ouro_jlens/manifest.py verify
```

Then read `RESULTS.md` and `REPRODUCE.md`. If further empirical work is desired,
freeze a new experiment contract and budget first; do not revive the archived
commands.
