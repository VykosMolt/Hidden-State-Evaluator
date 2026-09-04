# Ouro JLens status

This directory describes a local observational study of prompt-averaged Jacobian-lens readout in the fixed Ouro-2.6B snapshot. It is not a completed mechanism study and it is not submission-ready.

## Current conclusion

On the retained 80-prompt final-exit lens and evaluation stimuli, task-defined multihop intermediate-token readout is substantially lower than the logit lens at loops 1–3. Arithmetic shows a resolved relative deficit only at loop 1. This establishes a limitation of this average-Jacobian readout under this estimator and population. It does not show that an intermediate is absent, causally erased, or unavailable to other monitors.

Mechanism, estimator-independence, 1000-prompt robustness, and matched-model generalisation are `INCONCLUSIVE`. The B300 validation and fit are `NOT_RETAINED`.

## Authority

- `artifacts/jlens/final/analysis.json`: current machine-readable result and claim statuses.
- `artifacts/jlens/MANIFEST.json`: deterministic byte-custody inventory; it does not upgrade scientific status.
- `artifacts/jlens/final/fit_size.json`: nested fit-size sensitivity.
- `artifacts/jlens/final/transport.json`: fitted-map norms and explicitly modeled scatter.
- `artifacts/jlens/probe/cv_all648/summary.json`: cross-fitted arithmetic-probe report.
- `HANDOFF.md`: operational state and safe next action.
- `METHODS.md`: populations and estimands.
- `REPRODUCE.md`: validation and regeneration commands.
- `INCIDENT_2026-09-04.md`: paid-run postmortem.

The pre-repair Opus documents and peer notes are preserved byte-for-byte under `history/2026-09-04-opus/`. They are historical evidence, not current authority.
