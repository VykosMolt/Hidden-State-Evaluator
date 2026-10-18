# Relational test: a pair model against a decomposable model of matched capacity

Written 2026-10-18, before any model in this test was fit. Its SHA-256 is in `PROTOCOL.md.sha256`.

**This test is post hoc.** It was designed after the linear-probe results of RPE v3 had been seen. A linear difference
probe w.(h_A - h_B) = w.h_A - w.h_B is decomposable, so it cannot separate a relational encoding from a per-reply one.
This test asks whether a model that sees both replies jointly ranks held-out pairs better than the best decomposable
scorer of matched capacity, on identical data.

## Data
- The 40,000 HH-RLHF training-split pairs of the published probes, with the published pair-disjoint split of 32,000
  training and 8,000 held-out pairs (train/eval positions from `hh_thinking_40000_pair_disjoint_probe.pt`, the split
  used for every backbone).
- Features: the left-truncated, 384-token, mean-pooled loop states of the published probes
  (`$RPE_RUNS/thinking_left`, `$RPE_RUNS/rltt_left_cloud`), stored in float16 and read as float32.
- Primary analysis: Ouro-Thinking, last loop (2,048 dimensions). Secondary analyses (descriptive, same rule):
  Ouro-Thinking, all four loops concatenated (8,192); Ouro-RLTT, last loop (2,048).
- Validation set: the training pairs whose HH-RLHF train index i satisfies
  `int.from_bytes(sha256(str(i).encode()).digest()[:8], "big") % 10 == 0` (about 10%). The remaining training pairs
  are the fit set. Validation is used only for selecting settings and for early stopping. The 8,000 held-out pairs
  are scored once, after all choices are made.

## Preprocessing (identical for both models)
1. Standardize each feature with the mean and standard deviation (plus 1e-6) of the fit-set replies (chosen and
   rejected).
2. Project onto the top 512 principal components of the standardized fit-set replies, computed in float64, without
   whitening. The dimension 512 is fixed here for all three analyses.

## Models
Every hidden layer is Linear, then ReLU, then Dropout. The output layer is Linear to one unit. Weights use PyTorch's
default initialization, seeded per run.
- **Model D (decomposable):** s(A, B) = f(h_A) - f(h_B), with f an MLP from R^512 to R.
- **Model P (pair):** s(A, B) = 0.5 [g(A, B) - g(B, A)], with g an MLP on [h_A, h_B, h_A - h_B, h_A * h_B]
  (elementwise product; input 2,048). The score is exactly antisymmetric but not decomposable.
- Both models use the same grid of hidden widths and depths. Parameter counts are reported for both; P's first layer
  is larger.

## Training (identical for both models)
- Loss: mean softplus(-s(chosen, rejected)), the logistic loss on the preferred reply. Both scores are antisymmetric,
  so presentation order needs no randomization.
- Optimizer: AdamW (decoupled weight decay), learning rate 1e-3, batches of 256 pairs, reshuffled every epoch with a
  seeded generator.
- At most 40 epochs. After each epoch, validation accuracy (ties 1/2) is computed. Training stops after 5 epochs
  without improvement, and the weights of the best epoch are restored.
- Grid (24 settings): hidden width {64, 256} x hidden layers {1, 2} x weight decay {1e-4, 1e-3, 1e-2} x dropout
  {0, 0.2}.
- Computation: float32 on CPU, 8 threads.

## Selection and final fits
1. For each model, train every grid setting with seeds 0, 1, 2. Select the setting with the best mean validation
   accuracy. On an exact tie, take the setting that comes first in grid order (width, then depth, then weight decay,
   then dropout, each ascending).
2. Train the selected setting of each model with 5 fresh seeds (100 to 104).
3. A model's held-out score for a pair is the mean of its 5 seeds' scores. Means of antisymmetric scores stay
   antisymmetric, and means of decomposable scores stay decomposable.
4. Capacity-matched sensitivity check: P with its selected depth, weight decay and dropout, at the largest width whose
   parameter count does not exceed that of the selected D. Trained with the same 5 seeds 100 to 104.

## Endpoint and verdict (primary analysis)
- Primary endpoint: held-out accuracy of P minus held-out accuracy of D, on the 8,000 held-out pairs, with ties
  counted as 1/2. The 95% percentile interval comes from a paired bootstrap over pairs (10,000 draws,
  `numpy.random.default_rng(20261018)`).
- Verdict:
  - **RELATIONAL_ADVANTAGE** if the lower bound is above 0.
  - **NO_RELATIONAL_ADVANTAGE_ABOVE_1PT** if the upper bound is below +1 point.
  - **UNRESOLVED** otherwise.
- The secondary analyses apply the same rule, descriptively.

## Also reported
- Each model's held-out accuracy with a bootstrap interval (same draws).
- Each model against the published converged linear relational probe on the same pairs (`probe_pairs_*_converged.csv`;
  relational_final for the last-loop analyses, relational_all4 for all loops), as paired differences.
- A linear f on the preprocessed (512-dimensional) features, fit with the converged linear relational objective
  (`fit_probes.py --converged`, Newton). This is descriptive and shows how much the preprocessing changes the linear
  probe.
- Per-seed held-out accuracies and their spread, the full selection table, parameter counts, the capacity-matched P,
  and run times.

## Gates (run before the real fits)
1. On a subset of pairs, with randomly initialized models in evaluation mode:
   - D must satisfy s(A, B) = -s(B, A) exactly and s(A, B) = f(A) - f(B) exactly, and
     s(A, B) + s(B, C) - s(A, C) = 0 up to float32 rounding.
   - P must satisfy s(A, B) = -s(B, A) exactly. Its s(A, B) + s(B, C) - s(A, C) is reported, to show that P is not
     decomposable.
2. The data loading and split of this test's code, given raw last-loop Ouro-Thinking features and the converged linear
   relational fit, must reproduce the held-out relational last-loop accuracy of `probes_thinking_converged.json`
   (0.615375) exactly.

If a gate fails, the test stops and the failure is reported.

## Outputs
Per analysis:
- a results JSON with the verdict;
- per-pair held-out scores for D, P, capacity-matched P, every seed, and the linear references;
- fitted weights (preprocessing and every final model);
- the configuration;
- logs.

Every departure from this protocol is listed in the results.
