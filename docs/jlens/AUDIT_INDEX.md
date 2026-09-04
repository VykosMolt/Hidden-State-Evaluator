# Audit disposition

The peer notes are retained unchanged. This index records the current disposition rather than repeating their prose.

| finding | disposition |
|---|---|
| recurrent virtual-index mapping | current M1–M5 validation passes; equality comparisons bit-exact |
| long-horizon VJP includes human loop 1 | repaired and passed with sources `[16,64,112,160]` |
| any-layer control Jensen bug | repaired and regression-tested |
| item-versus-slot weighting of exit agreement | repaired and regression-tested |
| unsafe target-prefix correctness | repaired; current count is 72/148 |
| `0.708/0.715` called direct single-prompt norms | retracted; these were modeled scatter, with two invalid loop-4 moments |
| mechanism established by cancellation | `INCONCLUSIVE`; wording reduced to “consistent with” |
| fit-size artifact excluded | retracted; large-fit robustness remains `INCONCLUSIVE` |
| estimator reduction excluded | retracted; matched 2x2 remains unrun |
| probe uses all 648 prompts | corrected to 576 fold-trainable prompts / 39 clusters for the fair table; current caches regenerated |
| test-selected probe layers | replaced by cross-fitted layer selection and pair-cluster resampling |
| majority baseline omitted | repaired: 0.111 on 648, 0.125 on 576 |
| H2 local/eventual gap shrinks | `REFUTED_PRE_FINAL` |
| checkpoint JS/entropy rerun open | stale; retained local rerun had completed |
| 20 CPU tests | corrected: 15 CPU-runnable plus 5 CUDA/model tests |
| B300 validation and fit | `NOT_RETAINED` |
| old rental recipe | rejected and replaced; paid execution remains prohibited pending safety gates |
| submission readiness | `NOT_ESTABLISHED` |

No historical peer verdict authorizes the current bytes: final wording changed after that audit, and the old tree had no immutable manifest. Fresh verification and review are required for the repaired tree.
