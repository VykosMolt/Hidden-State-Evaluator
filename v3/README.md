# RPE v3: analysis package

Scripts, result files and per-pair outputs for version 3 of Jan Kirin, *Relational Preference Encoding in Looped
Transformer Internal States* ([arXiv:2604.09870](https://arxiv.org/abs/2604.09870)). Version 3 replaces versions 1
and 2, whose headline numbers came from evaluation errors.

The package is small: it does not contain model weights, feature shards or evaluator checkpoints. Those are listed,
with SHA-256 and size, in `INPUTS_OFFLINE.sha256`. Every file in the package is listed in `MANIFEST.sha256`.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | All analysis scripts. `config.py` holds the pinned revisions and reads every data location from environment variables. |
| `scripts/evaluator_core/` | Version 1's evaluator architecture (needed to load its checkpoints). |
| `scripts/paper_style/` | Plot style used by `make_figures.py`. |
| `results/` | Every result file the paper's numbers and figures are read from, including per-pair outputs. |
| `figures/` | The four figures, as produced by `make_figures.py`. |
| `relational_test_20261018/` | The post hoc relational test: protocol and its hash, gates, results, per-pair scores, weights, logs. |
| `config.example.sh` | Template for the environment variables. |
| `requirements.txt` | Exact package versions, and where each run was made. |
| `INPUTS_OFFLINE.sha256` | Required inputs that are not in the package: `sha256  size_bytes  VARIABLE/relative_path`. |
| `MANIFEST.sha256` | SHA-256 of every file in the package (`sha256sum -c MANIFEST.sha256`). |

## Setup

```bash
pip install -r requirements.txt
cp config.example.sh config.sh   # edit the paths
source config.sh
cd scripts
```

Check the offline inputs against the manifest:

```bash
grep -v -e '^#' -e '  HF:' ../INPUTS_OFFLINE.sha256 | while read h s p; do v=${p%%/*}; f="${!v}/${p#*/}"
  [ "$(sha256sum < "$f" | cut -d' ' -f1)" = "$h" ] || echo "MISMATCH $p"; done
```

(HH-RLHF lines, prefixed `HF:`, are skipped by this loop; `config.load_hh` downloads that exact revision.)

## Run recipe

Re-running writes into `$RPE_RESULTS` (default `results/`). To compare against the shipped files, first copy them:
`cp -r ../results /tmp/rpe_results && export RPE_RESULTS=/tmp/rpe_results`.

**GPU steps** (their outputs are the offline inputs in `$RPE_RUNS`; skip them to start from the shards):

```bash
python extract_hh_states.py --backbone thinking --mode left  --out $RPE_RUNS     # probe features, left truncation
python extract_hh_states.py --backbone thinking --mode repro --out $RPE_RUNS     # exact-zero check -> repro.json
python extract_hh_states.py --backbone rltt     --mode left  --out $RPE_RUNS     # (published run: rltt_left_cloud/)
python extract_hh_states.py --backbone rltt     --mode repro --out $RPE_RUNS
python epochs_and_surface.py --backbone thinking --max-length 1024 --early-exit 0.87 --out $RPE_RUNS/A_thinking_1024
python epochs_and_surface.py --backbone rltt --check-shards --out $RPE_RUNS/A_rltt                  # 768 tokens
```

**CPU steps**, in this order:

```bash
python summarize_epochs.py $RPE_RUNS/A_thinking_1024 $RPE_RESULTS/epochs_thinking_1024.json
python summarize_epochs.py $RPE_RUNS/A_rltt          $RPE_RESULTS/epochs_rltt.json
python fit_probes.py --converged --features $RPE_RUNS/thinking_left --tag thinking_converged \
       --out $RPE_RESULTS/probes_thinking_converged.json --save-dir $RPE_RESULTS
python fit_probes.py --converged --features $RPE_RUNS/rltt_left_cloud --tag rltt_left_converged \
       --out $RPE_RESULTS/probes_rltt_left_converged.json --save-dir $RPE_RESULTS
python length_vs_evaluator.py $RPE_RESULTS/length_vs_evaluator.json
python length_probe_pairs.py  $RPE_RESULTS/length_probe_pairs.json
python figure_data.py                         # evaluator_pair_scores_thinking.npz, first_difference_heldout.json
python identical_tokens_evaluator.py thinking # identical_at_1024_tokens_thinking.json
python identical_tokens_evaluator.py rltt     # identical_at_768_tokens_rltt.json
python paper_numbers.py                       # prints the derived numbers, asserting them against saved records
python make_figures.py                        # ../figures/*.pdf
```

`fit_probes.py --converged` fits every probe to convergence: the same objective, regularization, features and split
as the published fits, solved in float64 by Newton's method to max |gradient| <= 1e-10 (each fit takes 3–6 steps).
Without `--converged` it uses the published settings (float32 L-BFGS, 200 iterations), which reproduce the published
`probes_thinking.json` / `probes_rltt_left.json` only approximately (see "Known limits"). `--save-dir DIR --tag
NAME` writes `probe_weights_NAME.npz` (fitted weights), `probe_pairs_NAME.csv` (per held-out pair: position,
HH-RLHF train index, label, every probe's score) and `probe_fits_NAME.json` (per fit: L2 normalization n,
iterations, final loss, max |gradient|; n is 32,000 pair rows for the relational probe and the surface logistic,
64,000 response rows for the single-response probe). `--check SAVED.json` fails unless the new aggregate equals
the saved one exactly.

## Where each number in the paper comes from

| Paper | Script | Result file |
|---|---|---|
| Table 1; Fig. 1 and 2; abstract and Sec. 3.1 probe values (relational and single-response accuracies, per-reply accuracy, paired gaps); surface baselines in Sec. 3.3 (length rule, norm rule, surface logistic, relational minus surface); Table 3 probe rows | `extract_hh_states.py --mode left` (GPU), `fit_probes.py --converged` | `probes_thinking_converged.json`; per pair `probe_pairs_thinking_converged.csv`; weights `probe_weights_thinking_converged.npz`; fits `probe_fits_thinking_converged.json` |
| Sec. 3.1 "Backbones", Ouro-RLTT; Fig. 2 | same, `--backbone rltt` | `probes_rltt_left_converged.json` and the matching `*_rltt_left_converged.*` files |
| Superseded: the probe values in the October 2026 draft | `fit_probes.py` (published settings) | `probes_thinking.json`, `probes_rltt_left.json` |
| Table 2; Fig. 3B–C (solid); Sec. 3.2 (64.7 [63.7, 65.7], 63.7–65.3); Sec. 4.1 fixed order 94.9, offset +1.26; test-split norm rule 52.3, evaluator minus norm 12.4 [11.0, 13.8] | `epochs_and_surface.py` (GPU), `summarize_epochs.py` | `epochs_thinking_1024.json` |
| Sec. 3.2 Ouro-RLTT (65.0 [64.0, 66.0], 64.2–65.5); Fig. 3B–C (dashed) | same, RLTT at 768 tokens | `epochs_rltt.json` |
| Fig. 3A and Sec. 4.1 (19.2, 75.8, 5.1, 0.0; order/content 1.5; correlation −0.93; 0.5% clipped) | `figure_data.py`, `paper_numbers.py` | `evaluator_pair_scores_thinking.npz` |
| Fig. 4; Sec. 4.3 (649 = 615 + 34 of 8,000; 8.1%, 7.7%) | `figure_data.py` | `first_difference_heldout.json` |
| Sec. 4.3 exact zeros for all 649 pairs | `extract_hh_states.py --mode repro` (GPU) | `repro_thinking.json`, `repro_rltt.json` |
| Sec. 4.3 0.615 on the 7,351 differing pairs | `paper_numbers.py` | version 2 result + `repro_thinking.json` |
| Sec. 4.3 version 2 audit at 768 tokens: 125 identical, 64.6% on the rest | `identical_tokens_evaluator.py rltt` | `identical_at_768_tokens_rltt.json` |
| Sec. 4.3 evaluator at 1,024 tokens: 68 identical, accuracy change without them | `identical_tokens_evaluator.py thinking` | `identical_at_1024_tokens_thinking.json` |
| Sec. 3.3 longer reply preferred on the test split, 52.9 [51.9, 53.9]; "about 12 points" | `length_vs_evaluator.py` | `length_vs_evaluator.json` |
| App. B run-to-run variation (64.5 vs 63.9, 311 pairs, 0.32 vs 1.47) | `paper_numbers.py` (asserts the saved record) | `rerun_vs_audit_rltt.json` |
| Version 2 values (0.555 / 0.565 / 0.570; RLTT − base +0.015 [+0.004, +0.025]; 56.5; 54.2; +2.3; +1.0; 63.9) | `paper_numbers.py` | version 2 records in `$RPE_SPLIT`, `$RPE_AUDIT` |
| Sec. 2 evaluator size (4.7M parameters) | `paper_numbers.py` | `evaluator_core/pairwise_evaluator.py` |
| App. B "state-only copies with identical weights" | none (record only) | `checkpoint_copies_identical.json` |
| Figs. 1–4 | `make_figures.py` | `figures/*.pdf` |
| Relational test (pair model vs decomposable model), if reported | `relational_test.py` | `relational_test_20261018/results_*.json`, `pairs_*.csv` |
| Version 1 values (95.2, 84.5, 21.75, 83.3 → 62.4 by epoch, 60–70% training accuracy, 500-pair length and norm checks, correlation −0.94 on 100 pairs) and version 2's 74% partner share | not recomputed; quoted from versions 1 and 2 of arXiv:2604.09870 | |

`length_probe_pairs.json` (length rules on the probe pairs under version 2's right truncation) is supporting
material; the paper quotes the probe-pair length rule from `probes_thinking_converged.json` (no fit involved; the
same value as in `probes_thinking.json`). `identical_at_1024_thinking.json`
is the earlier, ad hoc calculation, which excluded only the 37 pairs with exactly equal pooled states; it is
superseded by `identical_at_1024_tokens_thinking.json`, which identifies all 68 token-identical pairs (the 37 are
among them).

## Relational test (post hoc)

`relational_test_20261018/` holds a controlled test that linear probes cannot make. A linear difference probe is
decomposable (w.h_A - w.h_B), so it cannot separate relational from per-reply encoding. The test compares a pair model
P (an MLP on [h_A, h_B, h_A - h_B, h_A * h_B], antisymmetrized) with a decomposable model D (s = f(h_A) - f(h_B),
f an MLP) on the same pairs, split and features. Both share preprocessing (standardization, 512 principal
components), grid, training and selection. The protocol, including the verdict rule, was written and hashed before
any fit (`PROTOCOL.md`, `PROTOCOL.md.sha256`). It was designed after the linear results were seen.

```bash
python relational_test.py gate --out ../relational_test_20261018                  # exactness and data gates
python relational_test.py run --analysis thinking_last --out ../relational_test_20261018   # primary
python relational_test.py run --analysis thinking_all4 --out ../relational_test_20261018
python relational_test.py run --analysis rltt_last     --out ../relational_test_20261018
```

Each run takes about 10 minutes on 8 CPU threads. Outputs per analysis: `results_*.json` (selection table, final
seeds, accuracies, paired differences, verdict), `pairs_*.csv` (per held-out pair scores of D, P, capacity-matched P,
every seed and the linear references), `weights_*.pt` (preprocessing and every final model; 44 MB, not included in
this package, recreated by the commands above), `config_*.json` and `log_*.txt`. `gate.json` records the gates.

## Known limits

- **The first probe fits were not converged.** `probes_thinking.json` and `probes_rltt_left.json` came from float32
  L-BFGS capped at 200 iterations, on a Runpod pod (torch 2.8.0). Three of the four probe fits stopped at the cap,
  and refitting with those settings changes held-out decisions with the environment and the thread count (records
  in `analysis/results/probe_refit_gate_failed_20261018/` of the working tree, not in this package). The
  `*_converged.*` files replace them: refit at 1 and at 8 threads, every aggregate value is exactly equal, per-pair
  scores differ by at most 1e-10 (the smallest nonzero decision score is 1.2e-7), and no decision changes
  (`results/probe_converged_thread_check_20261018/`, including the float64 L-BFGS calibration that motivated
  Newton's method).
- `checkpoint_copies_identical.json` was produced when the state-only checkpoint copies existed; the copies were
  not kept, so it is a record, not a reproducible output.
- Per-pair evaluator scores are included for Ouro-Thinking (`evaluator_pair_scores_thinking.npz`); those for the
  Ouro-RLTT run are in the offline shards `$RPE_RUNS/A_rltt/`.
- The GPU steps were run once each; bfloat16 results depend on batching, so a rerun can move pairs whose scores are
  near zero (App. B of the paper).

## License

Apache-2.0 (as the [Hidden-State-Evaluator](https://github.com/VykosMolt/Hidden-State-Evaluator) repository).
HH-RLHF is Anthropic's dataset; this package contains only row indices into it, no text.
