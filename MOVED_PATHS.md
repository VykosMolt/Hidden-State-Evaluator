# Path map — reorganization of 2026-09-15

The tree was split by publication line. Live code under `src/`, `utilities/`,
`tools/` and `compat/` was rewritten to the new paths. Recorded run outputs
(`artifacts/`, `o1_runs/`, `o1_packages/`) and dated documents under `docs/`
were **not** rewritten — they are records of what ran, and editing them would
falsify provenance. Use this table to resolve any old path you find there.

| Old | New | Line |
|---|---|---|
| `artifacts/reports/evaluator` | `rpe/evaluator` | RPE (arXiv 2604.09870) |
| `artifacts/checkpoints/evaluator` | `rpe/checkpoints/evaluator` | RPE |
| `artifacts/reports/family_xloop_v3_20260726` | `opi/localization/…` | Paper A (ICLR) |
| `artifacts/reports/huginn_probe_v3_20260726` | `opi/localization/…` | Paper A |
| `artifacts/reports/thinking_preanswer_v3_20260726` | `opi/preanswer/…` | pre-answer |
| `artifacts/reports/thinking_preanswer_power_v5_20260727` | `opi/preanswer/…` | pre-answer |
| `artifacts/reports/horizon_power_v3_20260726` | `opi/preanswer/…` | pre-answer |
| `artifacts/reports/fixed_prefix_horizon_20260729T082827Z` | `opi/preanswer/…` | pre-answer |
| `artifacts/reports/visible_prefix_hidden_increment_20260729T075247Z` | `opi/preanswer/…` | pre-answer |
| `artifacts/reports/paper1_v2_malformed_sibling_control_20260725` | `opi/preanswer/…` | pre-answer |
| `artifacts/reports/probes` | `opi/taps/probes` | OPI §4 taps |
| `artifacts/models/branch_training_*` | `opi/taps/models/…` | OPI §4 taps |
| `artifacts/reports/tournament_v4_20260727` | `opi/branching/…` | OPI §6–7 |
| `artifacts/reports/wellformed_terminal_v4_20260727` | `opi/branching/…` | OPI §6–7 |
| `artifacts/reports/lora_s3a_pilot_20260727` | `opi/control/…` | OPI §8 |
| `artifacts/reports/depth_alloc_v3_20260726` | `opi/control/…` | OPI §8 |
| `artifacts/reports/selective_prediction_v3_20260726` | `opi/control/…` | OPI §8 |
| `artifacts/reports/paper_verification` | `opi/verification/…` | OPI |
| `artifacts/hf_cache` | `shared/hf_cache` | shared |
| `models/` | `shared/models/` | shared |
| `data/` | `shared/data/` | shared |
| `venv/` | `shared/venv/` | shared |

`~/.bashrc` aliases `ouro-venv` and `pip-freeze-report` were updated to the
new venv path.

## Line → publication

- `rpe/` — Relational Preference Encoding, arXiv 2604.09870 + July 2026 erratum.
  Superseded by OPI; kept as the audited prior result.
- `opi/localization/` — Paper A (ICLR): recurrent refinement moves
  candidate-quality readability earlier in the reused stack. OPI paper1 §4.4.
- `opi/preanswer/` — strict pre-answer success prediction, OPI paper1 §5.
  Basis for a future ICML paper; that paper is not yet written and is blocked
  on a third domain (see ~/Documents/Research/ouro/paperB_third_domain_plan.md).
- `opi/taps/` — role-specialized readouts, DualAnchor / CoreContent, §4.
- `opi/branching/` — generated-branch correctness and commitment, §6–7.
- `opi/control/` — the readout–control boundary, §8.

The curated, publishable subset of the OPI evidence lives separately in
`~/branching-looped-transformer` (public repo). This tree holds the heavy
working versions.

---

# Second round — 2026-09-15

Everything shared between research lines moved under `shared/`, and four
sibling repositories were absorbed.

| Old | New |
|---|---|
| `src/` | `shared/src/` |
| `utilities/` | `shared/utilities/` |
| `tools/` | `shared/tools/` |
| `docs/` | `shared/docs/` |
| `compat/` | `shared/compat/` |
| `requirements/` | `shared/requirements/` |
| `~/jacobian-lens` | `jacobian-lens/` (purged 2026-09-20; `VykosMolt/JLens-Ouro`, branch `jlens-ouro`) |
| `~/lifetime-rltt` | `lifetime-rltt/` |
| `~/elastic_reasoner` | `elastic_reasoner/` |
| `~/branching-looped-transformer` | dissolved — see below |

`pytest.ini` now points at `shared/utilities/tests`, with `shared` and
`shared/src` on the path (the old `src/local_agent` entry was dead and removed).

## Branching-Looped-Transformer

The published OPI repo was distributed by content on 2026-09-15. Most of it was
byte-identical to material already here. What moved:

| From | To |
|---|---|
| `papers/*.pdf`, `papers/figures_v3/` | `~/Documents/Research/ouro/Operational-Proto-Introspection/published/` |
| `results/cross_loop_early_layer_taps_20260720/` | `opi/localization/` |
| `results/proto_introspection/` | `opi/preanswer/` |
| `results/paper1_v2_overnight_20260724/` | `opi/` |
| `probes/` | `opi/taps/blt_probes/` |
| `results/paper_verification/` (30 files) | merged into `opi/verification/paper_verification/` |
| `tools/` (22 files), `docs/` (3 files) | merged into `shared/` |
| `docs/hunter_seeker_*`, `docs/local_agent` | `~/archive/hunter-seeker/blt-docs/` (purged 2026-09-20; still in the BLT GitHub repo under `docs/`) |

The remaining 468 files had byte-identical copies here already. The GitHub repo
`VykosMolt/Branching-Looped-Transformer` still exists; the local clone does not.
The pre-distribution snapshot that was kept in `~/archive/` was a clean clone at
`cb174ef`, an ancestor of the GitHub `main`; it was purged on 2026-09-20.
