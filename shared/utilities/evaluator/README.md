# Evaluator Utilities

Measurement scripts for the evaluator project live here.

- `run_post_rltt_probe_bundle.py`: standard post-RLTT diagnostic bundle.
- `probes/`: long-running evaluator, layer-tap, branch-selection, spatial,
  ARC-transfer, and math-domain probes.

Default outputs go to `../../rpe/evaluator/`. The canonical
pairwise checkpoint is `../../rpe/checkpoints/evaluator/pairwise_epoch2.pt`.
