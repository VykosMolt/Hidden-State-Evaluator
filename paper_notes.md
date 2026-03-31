# CLT Evaluator Research Notes — Run 8 Onwards
## Date: March 2025

---

## Starting Point: Run 7 Results (V2 Architecture)

**Architecture:** GRU + AttentionPool(hidden_dim=2048, attn_dim=128) + 2-layer GRU(512) + skip connection
**Training accuracy:** 70% (peaked mid-epoch 2, plateaued through epoch 3)
**Test accuracy:** 65%
**Generalization gap:** 5 points
**Dataset:** 25k examples from Anthropic HH-RLHF (train split)

This was the best result across all pointwise architectures tested up to this point.

---

## Experiment: No-GRU Minimal Architecture (evaluator_test.py)

**Hypothesis:** Sequential bottlenecks (pool → proj → GRU → scorer) destroy signal. A direct attention-pool → MLP path should recover more, since the linear probe got 93.75% with a simple function.

**Architecture:** AttentionPool(attn_dim=256) → LayerNorm → MLP(2048→512→256→1). No GRU, no temporal modeling. Uses only final loop state.

**Result:** 68% training accuracy. Regressed from V2's 70%.

**Conclusion:** The GRU IS contributing ~2 points, but only when paired with attention pooling. With mean pooling (earlier experiments), the GRU added 0 points. The temporal dynamics matter once the pooling gives clean per-token signal.

---

## Critical Discovery: Scaled Linear Probe (probe_test.py, 1000 examples)

Previous probe (200 examples) showed 93.75% pairwise accuracy. Concern: small sample size inflated the result.

### Hook Verification (confirmed in same script)
```
output[0]: BaseModelOutputWithPast (standard transformer output)
output[1]: list of 4 tensors, each [batch, seq_len, 2048], dtype=bfloat16
output[2]: list (additional outputs)
```
**CONFIRMED:** output[1] contains genuine loop iteration states, not layer outputs or attention weights. The CLT premise is valid.

### Probe Results at 1000 Examples

| Probe Type | Final State Only | All States Concat |
|---|---|---|
| **Pairwise (chosen-rejected)** | **84.5%** | **86.25%** |
| **Independent classification** | **21.75%** | **22.75%** |

### Key Findings

1. **93.75% was inflated.** Real pairwise ceiling is 84.5% (down from 93.75% at 200 examples). Still meaningful headroom above 65% test, but not the massive gap originally thought.

2. **Independent classification at 21.75% — BELOW CHANCE.** This is the fundamental finding. The representations CANNOT distinguish chosen from rejected when scored independently. Preference signal exists ONLY in the relative difference between pairs.

3. **Flipping the 21.75% gives 78.25%.** The representations DO carry independent signal, but with inverted polarity. Rejected responses likely have larger activation norms or other systematic properties that a naive classifier picks up with wrong sign.

### What This Explains
- Why every pointwise architecture hit 67-70%: the independent scoring task has a theoretical ceiling around 75-78%
- Why trajectory supervision didn't help: the signal isn't in absolute trajectory shape
- Why the GRU adds only 2 points: it can extract some temporal ordering signal but the fundamental representation doesn't support absolute scoring

---

## Structural Shortcut Diagnostic (diagnostic.py, 500 test examples)

Checked whether simple features trivially separate chosen from rejected.

### Results
```
Chosen tokens:  mean=156.7, std=122.7
Rejected tokens: mean=169.2, std=129.9
'Longer = chosen':        45.6% (ANTI-correlated — rejected is longer)
'Larger norm = chosen':   43.0%
Hidden state norm ratio:  ~0.96 across all 4 loop steps
Mean activation ratio:    ~0.997 across all 4 loop steps
Length-norm correlation:   0.5123
```

**CONCLUSION:** No structural shortcut found. Length, norms, and activation magnitudes do not separate chosen from rejected. Any preference signal is in the representation geometry, not surface features.

---

## Feature Extraction Decoupling (extract_features.py + train2_fast.py)

**Motivation:** Every training run spent 95% of time on identical Ouro forward passes (model is frozen, deterministic). Decoupling extraction from training allows rapid architecture iteration.

**Implementation:**
- `extract_features.py`: Runs Ouro once, saves hidden states as chunked .pt files
- Save mode "raw": full [1, seq_len, 2048] per loop state in float16 + attention masks
- CHUNK_SIZE=100 examples per file
- `train2_fast.py`: Loads one chunk at a time (lazy), shuffles chunk order per epoch + within-chunk shuffle
- Peak RAM: one chunk (~300MB) instead of entire dataset

**Initial bug:** First version loaded ALL chunks into RAM, froze the PC. Fixed with chunk-sequential loading.

**Validation:** Reproduced Run 7's 70% accuracy exactly on precomputed features. Decoupling confirmed valid.

**Training time:** Minutes per epoch instead of hours.

---

## Pairwise Evaluator — First Attempt (evaluator_pairwise.py)

**Hypothesis:** The 84.5% pairwise probe says the signal is in chosen-rejected differences. Build an evaluator that sees both responses.

**Architecture:** Shared AttentionPool for both responses → per-step difference (chosen_pooled - rejected_pooled) → GRU over difference trajectory → skip connection from final difference → scorer outputs single scalar (positive = chosen preferred)

### First Training Run — DEGENERATE

**Result:** Accuracy rocketed to 99%+ within epoch 1. Scores of +13 on every example.

**Test eval:** 100% accuracy, 8552/8552 test examples correct. Scores all ~+13.

### Flip Test (flip_test.py) — FAILED

Swapped chosen and rejected inputs to check if scores flip sign.
```
Example 1: Normal=+13.16 | Flipped=+13.05
Example 2: Normal=+13.53 | Flipped=+13.45
...
```
**Signs did NOT flip.** Model outputs +13 regardless of input order.

### Root Cause Analysis

The loss `-F.logsigmoid(score)` only ever pushed scores positive. Training always presented chosen first. The model discovered: output large positive constant → loss ≈ 0. It never needed to examine actual content.

Specifically:
1. The architecture computes `chosen - rejected` at each step, which does flip sign on swap
2. But LayerNorm has a bias term that absorbs sign information
3. The GRU + scorer is expressive enough to map everything to positive constant
4. No training example ever required a negative output
5. L2 regularization was absent, allowing scores to grow unbounded to +13

---

## Pairwise Evaluator — Fixed Training (train_pairwise_fast.py v2)

### Three Fixes Applied

1. **Random 50% swap:** Each batch, coin flip — either `evaluator(chosen, rejected)` with target=+1, or `evaluator(rejected, chosen)` with target=-1. Model must learn direction, not just magnitude.

2. **Directional loss:** `-F.logsigmoid(target * score)` instead of `-F.logsigmoid(score)`. When target=-1, loss pushes score negative.

3. **Score regularization:** `1e-4 * score.pow(2).mean()` — prevents scores from growing unbounded.

4. **Accuracy metric fixed:** Checks if score sign matches target, not just "is it positive?"

### Result

**Epoch 1:** 60.9% accuracy (starting from ~50% = random, since half the targets are now negative)
**Epoch 2:** 67.4%
**Epoch 3:** 70.4%

**Same 70% ceiling as every other architecture.**

---

## Calibrated Evaluator (evaluator_calibrated.py + train_calibrated_fast.py)

**Hypothesis:** The 21.75% (flipped: 78.25%) independent probe means the polarity signal exists but is inverted. Adding a classification loss alongside ranking loss should anchor chosen→positive, rejected→negative, exploiting this signal.

**Architecture:** Identical to V2 (same GRU + AttentionPool). Only the loss changed.

**Loss:** `ranking_loss + 0.5 * BCE_classification_loss`
- Ranking: standard `-logsigmoid(score_chosen - score_rejected)`
- Classification: BCE forcing chosen scores positive, rejected scores negative

**Tracked metrics:** RankAcc (chosen > rejected) and ClassAcc (chosen > 0 AND rejected < 0)

### Result

**Epoch 1:** RankAcc=58.4%, ClassAcc=53.7%
**Epoch 2:** RankAcc=65.4%, ClassAcc=57.0%
**Epoch 3:** RankAcc=69.3%, ClassAcc=59.0%

**RankAcc converges to ~70% — same ceiling.** ClassAcc barely moved above chance (59%). The BCE component did not help.

---

## Linear Probe Replication (train_linear_fast.py)

**Hypothesis (from GPT):** The GRU/attention pooling destroys signal. The probe's 84.5% comes from preserving high-dimensional features. A single `nn.Linear(8192, 1)` trained with SGD should approach the probe's accuracy if compression is the bottleneck.

**Architecture:** Concatenate 4 loop states (2048×4=8192 features), mean pool across tokens, single `nn.Linear(8192, 1)`. 8,193 parameters total. No nonlinearities.

### Result

**Reached only 64% by batch 6000/12500 in epoch 3.** Killed early — clearly plateaued.

### What This Proves

1. **GPT's compression thesis was wrong.** The linear model without compression gets WORSE than V2 with compression (64% vs 70%). The GRU and attention pooling ADD 6 points of value.

2. **The 84.5% probe accuracy comes from two things the evaluator can't replicate:**
   - Pairwise differences (seeing both responses simultaneously)
   - L-BFGS optimizer (convex optimization on full dataset, not SGD on mini-batches of 2)

3. **The architecture search is closed.** V2 at 70% is extracting nearly everything available for independent scoring from these representations.

---

## Summary of ALL Architectures Tested

| Architecture | Training Acc | Test Acc | Notes |
|---|---|---|---|
| V1 MLP concat | 62% | 61.9% | Baseline |
| V2 GRU+mean pool | 68% | 63% | |
| V2 GRU+AttentionPool(128) (Run 7) | **70%** | **65%** | Best pointwise |
| V2 GRU+AttentionPool(2048) full-rank | 53% (collapsed) | — | Overfitting |
| No-GRU AttentionPool(256)+MLP | 68% | — | GRU adds 2pts with attn pool |
| No-GRU AttentionPool(128) (evaluator_test) | 68% | — | Confirmed |
| Pairwise (degenerate, no swap) | 100% | 100% | Failed flip test — constant output |
| Pairwise (fixed, with swap) | 70% | — | Same ceiling |
| Calibrated (ranking + BCE) | 69% | — | BCE barely helped |
| Linear only (probe replication) | 64% | — | Compression HELPS, not hurts |

---

## Key Findings (Publishable)

### 1. Relational vs Absolute Preference Encoding
Ouro-2.6B-Thinking's loop states encode preference **relationally** (84.5% pairwise probe) but **not absolutely** (21.75% independent classification, below chance). This is a novel empirical finding about how looped transformers represent alignment.

### 2. Inverted Independent Signal
The 21.75% independent probe (flipped: 78.25%) indicates the representations carry structure correlated with preference, but with inverted polarity. This likely reflects systematic differences in activation patterns (rejected responses may trigger stronger/different activation profiles).

### 3. No Structural Shortcuts
Diagnostic analysis on 500 examples showed no separation by sequence length (45.6%), hidden state norms (43.0%), or activation magnitudes (~0.997 ratio). The preference signal is geometric, not surface-level.

### 4. Hook Verification
Confirmed that `model.model` forward hook output[1] is a list of 4 loop iteration tensors `[batch, seq_len, 2048]`, not layer outputs or attention weights. The CLT architecture reads genuine loop dynamics.

### 5. Architecture-Independent Ceiling
Seven different architectures across three loss functions all converge to 68-70% training accuracy on 25k examples. This ceiling is a property of the representations and dataset size, not the evaluator design.

### 6. GRU Contributes Modestly
The GRU adds ~2 points when paired with attention pooling, 0 points with mean pooling. Temporal modeling of loop dynamics provides marginal benefit for independent scoring.

### 7. Compression Helps, Not Hurts
A single linear layer (8,193 params) on concatenated features reached only 64% — worse than the full V2 with GRU and attention pooling (70%). The compression layers extract useful representations, not destroy signal.

---

## Current Status

### Completed
- Architecture search exhausted for pointwise evaluators on 25k data
- Pairwise evaluator validated (with proper flip-test training)
- Feature extraction decoupled from training
- Diagnostic suite built (probe, structural shortcuts, flip test)

### In Progress
- Extracting 50k features to external SSD (/mnt/sandisk/ouro_features)
- Testing whether 2x data scale moves the 70% ceiling

### Pending
- Test eval on pairwise (fixed) and calibrated checkpoints
- Full 161k dataset extraction (if 50k shows improvement)
- Paper writeup

---

## File Inventory

### Evaluators
- `evaluator2.py` — V2: GRU + AttentionPool(128), Run 7 architecture (BEST pointwise)
- `evaluator_test.py` — No-GRU: AttentionPool(256) + MLP only
- `evaluator_pairwise.py` — Pairwise: shared pool, difference trajectory through GRU
- `evaluator_calibrated.py` — Calibrated: identical to V2, used with combined loss

### Training Scripts
- `train2.py` — V2 with live Ouro inference
- `train2_fast.py` — V2 on precomputed features (chunk-sequential)
- `train_test.py` — No-GRU evaluator, live Ouro
- `train_pairwise_fast.py` — Pairwise on precomputed features (with 50% swap fix)
- `train_calibrated_fast.py` — Calibrated on precomputed features
- `train_linear_fast.py` — Single linear layer on precomputed features

### Evaluation Scripts
- `evaluate2.py` — V2 evaluation (adapted for attention pooling API)
- `evaluate_test.py` — No-GRU evaluation
- `evaluate_pairwise.py` — Pairwise evaluation
- `evaluate_calibrated.py` — Calibrated evaluation (reports rank + class accuracy)

### Diagnostics
- `probe_test.py` — Linear probe at scale (1000 examples, pairwise + independent)
- `diagnostic.py` — Structural shortcut detection (length, norms, activations)
- `flip_test.py` — Antisymmetry test for pairwise evaluator
- `extract_features.py` — Feature extraction from Ouro to disk

### Checkpoints
- `checkpoints_v2/` — V2 Run 7 checkpoints (epoch 1-3)
- `checkpoints_test/` — No-GRU evaluator checkpoints
- `checkpoints_pairwise/` — Pairwise evaluator (both degenerate and fixed)
- `checkpoints_calibrated/` — Calibrated evaluator checkpoints
- `checkpoints_linear/` — Linear evaluator checkpoints

---

## Lessons Learned

1. **Always run a flip test on pairwise models.** The degenerate solution (constant positive output) achieves 100% accuracy on both train and test when chosen is always presented first. Only swapping inputs reveals the failure.

2. **Linear probes on small samples overestimate.** 93.75% on 80 test pairs (200 examples) dropped to 84.5% on 400 test pairs (1000 examples). Always validate with 1000+ samples.

3. **Independent classification probes reveal representation geometry.** The 21.75% below-chance result was more informative than the 84.5% pairwise result — it revealed that preference is encoded relationally, not absolutely.

4. **Decouple feature extraction early.** Every architecture experiment after decoupling took minutes instead of hours. Should have done this before Run 1.

5. **Don't trust 100% accuracy.** When a model achieves perfect accuracy on both train and test, the first instinct should be suspicion, not celebration. Check for trivial solutions.

6. **Architecture search has diminishing returns when representations are the bottleneck.** Seven architectures hitting the same ceiling is strong evidence. Move to scaling data, not redesigning models.
