# Ouro Loop-State Pairwise Evaluator (superseded)

> **The headline numbers of version 1 were evaluation errors.** This repository is kept as the snapshot of the
> original code. Its claims are superseded by version 3 of the paper:
>
> Jan Kirin, *Relational Preference Encoding in Looped Transformer Internal States*, version 3 (October 2026),
> replacing versions 1 and 2 of [arXiv:2604.09870](https://arxiv.org/abs/2604.09870).
>
> Analysis scripts, per-pair outputs, fitted probe weights and the input manifest for version 3 are in [`v3/`](v3/);
> start with [`v3/README.md`](v3/README.md).

## What changed

| Claim | Version 1 (this repository) | Version 3 | Cause of the error |
|---|---:|---:|---|
| Pairwise nonlinear evaluator, HH-RLHF test split | 95.2% | 64.7% (scored in both orders) | position bias: tested with the preferred reply always first |
| Linear relational probe | 84.5% | 61.5% (last loop) | leakage: tested on rows built from its own training pairs |
| Linear single-response probe, per reply | 21.75% | 54.9% (last loop) | leakage, as above |
| "Preference is encoded relationally" | claimed | not testable with linear probes: a probe on the difference of two replies is a per-reply score taken twice, and the probe trained on single replies ranks pairs within two points of it | version 1's evidence came from the leakage above |

Version 2 (July 2026) prepended an erratum to the paper; its corrected relational probe number (0.5653) was itself
lowered by a third error: right truncation cut off the differing replies, so 8.1% of held-out pairs became identical
inputs and were scored as errors. Version 3 replaces both and
lists every changed number in its Appendix A. The correction notice previously at the top of this README is
superseded by this one.

## Status of the original text

Everything below this line in earlier versions of this README (the "Main result" table, the 95.2% figure, the
interpretation that loop-state trajectories expose a strong relational evaluative structure, and the follow-up
claims built on it) is **withdrawn**. It remains in the git history as a record of what was reported.

## Contents

The original code is unchanged from the version 1 experiments and is provided as is:

| File | Purpose |
|---|---|
| `evaluator_pairwise.py` | Pairwise evaluator architecture. |
| `train_pairwise_fast.py` | Training script for the pairwise evaluator. |
| `evaluate_pairwise.py` | Evaluation script for trained checkpoints. |
| `extract_features.py` | Feature extraction from frozen Ouro loop states. |
| `ouro_inspect.py` | Utilities for inspecting Ouro hidden states. |
| `probe_test.py` | Linear probe experiments of version 1. |
| `modeling_ouro_patched.py` | Patched Ouro modeling file used during the experiments. |
| `papers/` | Earlier manuscripts, kept for the record. |
| `v3/` | Version 3's analysis package: portable scripts, results, per-pair outputs, fitted probe weights, figures, `MANIFEST.sha256` and `INPUTS_OFFLINE.sha256`. |

If you reuse this code: score every pair in both orders and report accuracy on s(c, r) - s(r, c) > 0; split
probe data by pair, not by row; and check how many pairs truncation makes identical (version 3, Sec. 4).

## License

Apache-2.0.
