# Ouro Loop-State Pairwise Evaluator (superseded)

> **The headline numbers of version 1 were evaluation errors.** This repository is kept as the snapshot of the
> original code. Its claims are superseded by version 3 of the paper:
>
> Jan Kirin, *Reading Human Preference from Looped Transformer Internal States*, version 3
> (versions 1 and 2 were titled *Relational Preference Encoding in Looped Transformer Internal States*),
> Replacement manuscript for [arXiv:2604.09870](https://arxiv.org/abs/2604.09870) (November 2026).
>
> Analysis scripts, per-pair outputs, fitted probe weights and the input manifest for version 3 are in [`v3/`](v3/);
> start with [`v3/README.md`](v3/README.md).

## What changed

| Claim | Version 1 (this repository) | Version 3 | Cause of the error |
|---|---:|---:|---|
| Pairwise nonlinear evaluator, HH-RLHF test split | 95.2% | 64.7% (scored in both orders) | position bias: tested with the preferred reply always first |
| Linear relational probe | 84.5% | 61.5% (last loop) | leakage: tested on rows built from its own training pairs |
| Linear single-response probe, per reply | 21.75% | 54.9% (last loop) | leakage, as above |
| "Preference is encoded relationally" | claimed | linear probes cannot test this; the primary nonlinear joint-reader comparison is unresolved | the linear probes rank pairs with the same family of functions |

Version 2 (July 2026) prepended an erratum to the paper; its corrected relational probe number (0.5653) was itself
lowered by a third error: right truncation hid the differing replies in 7.7% of held-out pairs. Together with
34 pairs whose texts were identical, 8.1% of pairs became identical inputs and were scored as errors.
Version 3 replaces both and
lists every changed number in its Appendix A. The correction notice previously at the top of this README is
superseded by this one.

The primary nonlinear comparison estimates +0.7 percentage points for the joint reader over a per-reply scorer
(95% CI [0.0, 1.4]); this establishes neither a positive gain nor an upper bound below one point. The all-loop
and RLTT secondary comparisons do meet the below-one-point rule for the tested networks.

## Status of the original text

Everything below this line in earlier versions of this README (the "Main result" table, the 95.2% figure, the
interpretation that loop-state trajectories expose a strong relational evaluative structure, and the follow-up
claims built on it) is **withdrawn**. It remains in the git history as a record of what was reported.

## Contents

The code is unchanged from the original experiments and is provided as is:

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

If you reuse this code: score every pair in both orders and report accuracy on s(c, r) - s(r, c) > 0; split
probe data by pair, not by row; and check how many pairs truncation makes identical (version 3, Sec. 4).

## License

Apache-2.0.
