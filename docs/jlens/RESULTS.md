# JLens results

Status: `SUPPORTED_LOCAL_OBSERVATIONAL_RESULT_UNFROZEN`.

The machine-readable authority is `artifacts/jlens/final/analysis.json`. Values below are descriptive of the fixed local Ouro-2.6B snapshot and retained artifacts; they are not a general mechanism claim.

## Main result

Any-layer excess hit@10 for task-defined intermediate tokens:

| population | readout | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---|---:|---:|---:|---:|
| multihop, 90 items | final-exit average J-lens, n=80 | +0.003 | -0.025 | +0.084 | +0.451 |
| | logit lens | +0.358 | +0.458 | +0.500 | +0.483 |
| | paired difference, J minus logit | -0.355 | -0.483 | -0.416 | -0.032 |
| order-operations numeric, 51 items | final-exit average J-lens, n=80 | +0.054 | +0.273 | +0.181 | +0.193 |
| | logit lens | +0.306 | +0.152 | +0.131 | +0.213 |
| | paired difference, J minus logit | -0.252 | +0.121 | +0.050 | -0.020 |

Unadjusted item-bootstrap 95% intervals for J minus logit:

- Multihop: `[-0.460,-0.250]`, `[-0.583,-0.382]`, `[-0.519,-0.313]`, `[-0.105,+0.039]`.
- Arithmetic: `[-0.354,-0.136]`, `[-0.005,+0.246]`, `[-0.081,+0.186]`, `[-0.069,+0.023]`.

The supported statement is relative: the retained final-exit average J-lens is below the logit lens at multihop loops 1–3 and arithmetic loop 1. Loop 4 multihop is unresolved; arithmetic loops 2–4 are unresolved. The J-lens is not absolutely “blind”: arithmetic candidate-set top-1 is 0.608 at loop 1, and multihop excess is positive at loops 3–4.

## Current instrumentation validation

The current source passes M1–M5 on the local RTX 5070 Ti Laptop GPU. The retained validator report covers hook non-interference, equality of virtual recurrent exits to native exits, recurrent firing/carry identity, VJPs from all four recurrent loops including human loop 1 (`ut=0`), and consistency with the stock final-loop Jacobian. All 18 required equality comparisons were bit-exact. Numerical acceptance and bit-exact status are recorded separately in `artifacts/jlens/validation/milestones.json`.

## Lens-free recurrent exits

Top-1 agreement between each recurrent exit and the final exit, over all raw task stimuli and weighted once per item:

| population | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---:|---:|---:|---:|
| multihop, 93 items | 0.409 | 0.656 | 0.785 | 1.000 |
| order-operations, 55 items | 0.527 | 0.855 | 0.964 | 1.000 |

Earlier figures silently weighted items once per clean intermediate slot. Those figures are superseded.

Boundary-aware answer matching gives 72 correct targets among 148 stimuli. The three old false positives were `11` for `1`, `daytime` for `day`, and `16` for `1`. Model-correct robustness subsets from the old report are superseded unless regenerated with this definition.

## Local-exit versus eventual-exit

The contemporaneous local pre-analysis specification expected the local/eventual gap to shrink over pre-final loops. It does not:

| loop | mean KL(local readout || eventual readout) | last-eight-layer top-1 agreement | multihop local-minus-eventual excess |
|---|---:|---:|---:|
| 1 | 4.903 | 0.002 | +0.065 |
| 2 | 4.567 | 0.018 | +0.114 |
| 3 | 6.526 | 0.008 | +0.151 |

Status: `REFUTED_PRE_FINAL`. Equality at loop 4 is definitional. This comparison uses 24-prompt local lenses and a 32-prompt eventual lens, so it is not mixed into the n=80 headline.

## Cross-loop association

At both n=32 and n=80, the multihop transfer matrix is row-dominated: the readout changes mainly with the loop where the Jacobian was fitted, rather than the loop producing the state. The fit-loop variance share is 0.98 at n=32 and 0.99 at n=80; the state-loop share is 0.01 at both sizes. Arithmetic is not comparably stable (fit-loop shares 0.10 and 0.44, with substantial interaction).

This is a replicated descriptive association. “Remaining horizon” is not established as its cause because fit loop also changes estimation difficulty and the learned-map population.

## Fit-size sensitivity

The 8, 32, 56, and 80-prompt fits are nested prefixes. Thresholded multihop loop-1 excess remains near zero. Under that floor, mean improvement in `log10(rank+1)` from n=8 to n=80 is:

| population | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---:|---:|---:|---:|
| multihop | +0.096 | +0.227 | +0.244 | +0.019 |
| order-operations numeric | +0.308 | +0.387 | +0.315 | +0.038 |

Multihop loop 1 and both loop-4 cells have intervals crossing zero; the other cells improve over this nested range. This does not establish fixed-n variance or behavior at n=1000. Status: `LARGE_FIT_INCONCLUSIVE`.

## Transport norms and scatter

The raw norm of the fitted average map at n=80, averaged over source locations within loop, is `[0.159, 0.178, 0.272, 0.810]`. The loop-4/loop-1 descriptive ratio is 5.09. It is not a formal population lower bound.

The least-squares model across fit sizes estimates mean-map norms `[0.134, 0.173, 0.269, 0.807]`. Valid-decomposition modeled RMS scatter is `[0.708, 0.324, 0.321, 0.747]`. Two of 47 loop-4 source locations have negative fitted scatter moments and are excluded from that last aggregate. The implied total RMS, where `mu² + sigma²` is nonnegative, is `[0.721, 0.368, 0.419, 1.103]`. The old `0.715` loop-4 scatter value resulted from silently replacing the two invalid scatter moments with zero.

These are modeled moments from nested prompt averages. They are not direct single-prompt Jacobian measurements. The pattern is consistent with large relative prompt-to-prompt variation at long horizon, but prompt-specific rewriting and cancellation remain `INCONCLUSIVE`.

## Arithmetic probe

The fair comparison uses 576 fold-trainable prompts from 39 unordered-pair clusters, not all 648 prompts. Each method’s layer is selected on other folds and scored on the held-out fold. Point estimates are:

| readout | loop 1 | loop 2 | loop 3 | loop 4 |
|---|---:|---:|---:|---:|
| supervised probe | 0.615 | 0.590 | 0.337 | 0.236 |
| logit lens | 0.764 | 0.500 | 0.179 | 0.311 |
| final-exit average J-lens | 0.392 | 0.500 | 0.148 | 0.120 |

Uniform 17-way chance is 0.059. The empirical majority baseline is 0.111 on all 648 prompts and 0.125 on the fair 576-prompt population. The J-lens loop-4 point estimate does not exceed that majority baseline.

Pointwise pair-cluster bootstrap intervals put J minus logit below zero at loop 1 `[-0.548,-0.221]` and loop 4 `[-0.284,-0.105]`; loops 2–3 are unresolved. Supervised probe minus logit is below zero at loop 1 `[-0.329,-0.032]`, above zero at loops 2 `[+0.003,+0.283]` and 3 `[+0.041,+0.308]`, and unresolved at loop 4. These are not multiplicity-adjusted family-wide claims. Absence of a resolved difference is not equivalence.

## Checkpoints

The retained local checkpoint run already includes KL, Jensen–Shannon divergence, entropy, final-token rank, and top-1 agreement for base, Thinking, and RLTT. Distributional divergence decreases from base to Thinking to RLTT while categorical top-1 agreement does not improve. Cause remains unresolved. These artifacts predate the repaired custody pipeline and retain `SUPPORTED_LOCAL_UNFROZEN` status.

## Claim boundary

Supported: a local, estimator-specific multihop readout deficit relative to the logit lens, plus an arithmetic loop-1 deficit.

Not established: absence of intermediate information, causal erasure, direct single-prompt transport equality, estimator independence, 1000-prompt robustness, architecture as cause, matched-model generalisation, B300 validation, or submission readiness.
