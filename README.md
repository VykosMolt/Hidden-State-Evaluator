# ouro_project

Research on Ouro, a 2.6B looped transformer: what its recurrent hidden states
reveal about the quality of its own ongoing computation, and whether that
readout can be turned into better outcomes.

The tree is split by **publication line**. Everything used by more than one line
lives under `shared/`.

    opi/                 Operational Proto-Introspection  (arXiv 2607.18553)
    rpe/                 Relational Preference Encoding   (arXiv 2604.09870 + erratum)
    shared/              code, docs, models, data, venv — used across lines
    artifacts/           run records: logs, ops, the 2026-07 storage-cleanup reports
    elastic_reasoner/    absorbed repo — FABLE/Astra two-agent Huginn pilot
    jacobian-lens/       absorbed repo — J-Lens vs the raw logit lens
    lifetime-rltt/       absorbed repo — O1 / Foundation-Learner ladder

## opi/ — Operational Proto-Introspection

The umbrella programme. RPE is its predecessor: OPI reuses the evaluator
machinery, audits it, and corrects three of its published figures.

| directory | paper section | what it holds |
|---|---|---|
| `localization/` | §4.4 | Recurrent depth moves candidate-quality readability earlier in the reused stack. `cross_loop_early_layer_taps_20260720` (the original RLTT run), `family_xloop_v3_20260726` (Ouro-family replication), `huginn_probe_v3_20260726` (out-of-family). **This is Paper A, the ICLR line.** |
| `preanswer/` | §5 | Strict pre-answer success prediction on GSM8K and Horizon Logic, plus `proto_introspection/` provenance. **Basis for the planned ICML paper, which is not yet written** — blocked on a third domain (`~/Documents/Research/ouro/paperB_third_domain_plan.md`). |
| `taps/` | §4 | Role-specialised readouts. `probes/` is 113 `bg_*` runs (DualAnchor, two-tap, hidden-origin); `models/` holds the branch-training generators; `blt_probes/` came from the published repo. |
| `branching/` | §6–7 | Generated-branch correctness and commitment: `tournament_v4_20260727`, `wellformed_terminal_v4_20260727`. |
| `control/` | §8 | The readout–control boundary: `lora_s3a_pilot_20260727`, `depth_alloc_v3_20260726`, `selective_prediction_v3_20260726`. |
| `verification/` | — | `paper_verification/` — the audit and hardening checks. |
| `paper1_v2_overnight_20260724/` | §5, §8 | Cross-cutting overnight run (horizon_logic, subspace_geometry, terminal_selection). |

## rpe/ — Relational Preference Encoding

The prior paper. Its three headline figures did not survive audit; the erratum
is the correction of record. Only two things survive here:

- `checkpoints/evaluator/pairwise_epoch2.pt` — the frozen evaluator OPI audits
- `evaluator/` — the HH probe reports

Its 360 GB of extracted hidden states were never on this disk; they lived on the
external SSD.

## shared/

Used by more than one line, which is the sole criterion for living here.

| directory | contents |
|---|---|
| `src/` | `evaluator/`, `evaluator_core/`, `ouro_rltt/` |
| `utilities/` | probes, tools and the test suite (`utilities/tests`) |
| `tools/` | audit and paper-verification scripts |
| `docs/` | `evaluator/` (the `bg_*` design record), `project/` |
| `models/` | `ouro_rltt_local` (5 GB, the Paper A checkpoint), `ouro_thinking_local` (symlinks into `hf_cache`) |
| `data/` | `corecontent_v2`, `branch_training_logic_expansion_v1` |
| `hf_cache/` | Ouro-2.6B, Ouro-2.6B-Thinking, Huginn-0125, MiniCPM-2B (the §4.7 non-looped control), deberta-v3-small |
| `venv/` | Python 3.14, torch 2.12 + cu128, transformers 4.54.1 |
| `compat/`, `requirements/` | legacy path shims; pinned requirements |

## Absorbed repositories

These keep their own git history and are gitignored by this repo.

- **`jacobian-lens/`** — `origin` is `anthropics/jacobian-lens` (upstream, read-only).
  Your work is on branch `jlens-ouro`, which tracks the `jlens` remote
  (`VykosMolt/JLens-Ouro`). Push to `jlens`, never `origin`.
- **`lifetime-rltt/`** — a container, not a repo: `o1-runner`
  (`VykosMolt/Lifetime-Meta-Learning`) and `foundation-learner-b200-v0` (no
  remote). They must stay siblings. **Do not edit anything under
  `o1-runner/o1_b200/`** — a source digest binds the built RunPod image, and any
  `.py`/`.sh` change there unbinds it.
- **`elastic_reasoner/`** — `VykosMolt/elastic-reasoner`. Never edit the
  `ASTRA_*.md` root files.

## Running things

    shared/venv/bin/python -m pytest -q        # 36 tests
    source shared/venv/bin/activate            # or: ouro-venv

`pytest.ini` points at `shared/utilities/tests` with `shared` and `shared/src`
on the path.

## Conventions

- Code computes its root from `__file__`; absolute paths are not hardcoded.
- Recorded outputs under `artifacts/`, `opi/` and `rpe/` are **not** rewritten
  when paths change — they record what ran. `MOVED_PATHS.md` maps every old path
  to its new home, so any stale reference in a record can be resolved.
- The publishable subset of the OPI evidence was previously a separate repo
  (`VykosMolt/Branching-Looped-Transformer`); its contents were distributed into
  `opi/` and `~/Documents/Research` on 2026-09-15. The GitHub repo still exists.

## Related locations

- `~/Documents/Research/ouro/` — manuscripts, the erratum, Paper A's artifact inventory
- Hunter-Seeker, the retired ARC agent line, is no longer on this machine (purged
  2026-09-20): `VykosMolt/Hunter-Seeker-v1`, `VykosMolt/Hunter-Seeker-v2`
- External SSD — cold archives, and the pre-answer feature tensors
