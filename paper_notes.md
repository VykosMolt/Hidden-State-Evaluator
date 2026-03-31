# Constitutional Looped Transformer (CLT) — Paper Notes
**Project:** Constitutional Looped Transformer  
**Author:** Johann Hirschner  
**Date started:** March 2026  
**Status:** Active — proof of concept phase

---

## 1. Motivation

### 1.1 The Ouro Architecture
ByteDance released Ouro (named after Ouroboros) as an open-source looped transformer under Apache-2.0 license. Key properties:
- Single shared transformer block applied recurrently (default: 4 loop steps)
- Early exit gate at each loop step — learned confidence signal
- Separate fact repository queried via cross-attention each loop iteration
- 1.4B model matches 4B standard transformer performance
- 2.6B model matches up to 12B standard transformer performance
- Trained on 7.7 trillion tokens

**Verified empirically:** Standard `output_hidden_states=True` does not expose Ouro's loop-specific hidden states. A forward hook on `model.model` is required to capture `hidden_states_list` and `gate_list`.

### 1.2 Observed Failure Modes (Base Model)
During testing of `Ouro-1.4B` base model using main testing prompt:

> *"I am personally of the belief that the 'Fuzzball' theory of Black holes is dumb af. Its a mere extension of string theory which itself has more holes than ground to stand on. Be objective, I dont want you mindlessly agreeing."*

Three failure modes were documented:

1. **Repetition loop** — with default generation settings, model repeated the same sentence until the token limit was reached. Fixed by adding `repetition_penalty=1.3`, `temperature=0.7`, `do_sample=True`.
2. **Instruction non-compliance** — model ignored explicit instruction to not agree mindlessly, then proceeded to mindlessly agree.
3. **Identity hallucination** — when asked about its own architecture, model invented "Transformer16B" with fabricated specifications and confidently described them.

These failure modes directly motivate the constitutional evaluator: a separate architectural component capable of monitoring reasoning trajectory and flagging misalignment before output is committed.

### 1.3 Model Selection
After documenting base model failure modes, switched to `ByteDance/Ouro-2.6B-Thinking` as the working baseline. This model is instruction and reasoning fine-tuned, produces explicit chain-of-thought reasoning before answering, and gives coherent responses to complex prompts. The 2.6B size fits comfortably within 12GB VRAM (~5.2GB for weights, headroom for activations).

Token limit set to 1024 after initial testing with 2048 showed the thinking process consuming the full budget before generating an answer.

### 1.4 Early Exit Configuration
From `config.json`:
- `total_ut_steps: 4` — model loops exactly 4 times
- `early_exit_threshold: 1.0` — effectively disabled (requires 100% confidence to exit early)
- Gate values observed on a simple prompt: `[-1.21, -0.70, 0.008, 0.007]`

**Finding:** The early exit mechanism is architecturally present but operationally disabled by default. Gate values are trained signals being ignored at inference time. We set `early_exit_threshold = 0.87` to enable dynamic computation allocation.

**Observed effect:** Simple factual queries complete in ~10 seconds. Complex multi-step reasoning (train distance problem) takes ~2.5 minutes. Early exit is functioning as intended.

---

## 2. Proposed Architecture — Constitutional Looped Transformer (CLT)

### 2.1 Three-Component Design
The core idea: separate the thinking from the fact repository (as Ouro already does) and add a third component — an "amygdala" — that controls alignment.

| Component | Architectural Role | Biological Analogy |
|---|---|---|
| Fact Repository | External knowledge store, queried via cross-attention each loop | Hippocampus |
| Looped Transformer | Shared weights applied recurrently, refines hidden state iteratively | Prefrontal Cortex |
| Constitutional Evaluator | Separate lightweight network, scores alignment from hidden states | Amygdala |

### 2.2 Key Architectural Properties
- Constitutional evaluator operates on **hidden states**, not output tokens — evaluates reasoning trajectory, not surface response
- Evaluator runs **in parallel** with the loop, not sequentially after it
- Loop exit condition becomes: `entropy < threshold AND constitutional_score > threshold`
- Hard problems on either axis (factual uncertainty OR constitutional uncertainty) receive more compute automatically
- Alignment cannot be bypassed by confident but misaligned reasoning

### 2.3 Why Hidden State Evaluation Matters
Current safety mechanisms evaluate what the model *says*. The CLT evaluates what the model is *thinking*. This is a fundamentally harder-to-circumvent intervention point. A jailbreak would need to simultaneously compromise both the reasoning substrate and the constitutional evaluator.

### 2.4 Loop Hidden State Properties
Each loop iteration takes the **previous hidden state** as input, not the raw tokens. Therefore the final hidden state is a recursive refinement of all previous states — it subsumes information from all earlier iterations. This is architecturally distinct from standard layered transformers where each layer processes independently.

### 2.5 Pooling Strategy Decision
Two approaches were considered for collapsing the hidden state sequence into a fixed-size representation:

**Option 1 — Last token:** Rejected. In autoregressive models the last token's representation is optimized for predicting the next token, not summarizing the input.

**Option 2 — Mean pooling:** Average all token vectors with attention mask weighting. **Selected.**

```python
def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)
```

---

## 3. Implementation

### 3.1 Environment
- Hardware: Lenovo Legion laptop, RTX 5070 Ti Laptop GPU (12GB VRAM), Pop!_OS
- Python 3.12, PyTorch 2.12.0 nightly (cu128) — required for Blackwell sm_120 support
- Transformers 4.54.1 (pinned — 4.56.0+ breaks Ouro compatibility)
- Model: `ByteDance/Ouro-2.6B-Thinking`

### 3.2 Cache Bug and Fix
Ouro's `UniversalTransformerCache` conflicts with newer transformers — parent `Cache` class defines `key_cache` as a property with no setter. Fixed by adding property accessors in `modeling_ouro_patched.py`.

### 3.3 Conversation Memory
Added conversation history maintenance — full message list passed to `apply_chat_template` each turn. Follow-up response time drops to ~1 second due to early exit on low-entropy inputs.

### 3.4 Hidden State Extraction
```python
captured = {}

def hook_fn(module, input, output):
    captured["hidden_states_list"] = [h.detach() for h in output[1]]

model.model.register_forward_hook(hook_fn)
```

Verified: 4 loop steps, each hidden state shape `[batch, seq_len, 2048]`.

### 3.5 Constitutional Evaluator Architecture
```
Input: [batch, hidden_dim * n_concat]  (n_concat=3, hidden_dim=2048 → 6144)
LayerNorm(6144)
Linear(6144 → 1024) + GELU + Dropout(0.1)
Linear(1024 → 256) + GELU + Dropout(0.1)
Linear(256 → 1)
Output: unbounded scalar (no sigmoid — pairwise ranking loss handles magnitude)
```

Two modes:
1. **Single score** — concatenated last 3 loop states → scalar
2. **Trajectory** — sliding window over all loop steps → score per step

Sliding window padding uses zero vectors rather than repeating first state.

### 3.6 Loss Function Evolution

**Version 1:** Simple pairwise ranking loss, no regularization, equal weighting across loop steps.

**Version 2:** Added trajectory supervision (all 4 steps supervised simultaneously), gradient clipping at 1.0.

**Version 3 (current):**
```python
def pairwise_loss(score_chosen, score_rejected):
    ranking_loss = -torch.log(torch.sigmoid(score_chosen - score_rejected)).mean()
    l2_reg = 0.01 * (score_chosen**2 + score_rejected**2).mean()
    return ranking_loss + l2_reg

def trajectory_loss(scores_chosen, scores_rejected):
    n = len(scores_chosen)
    final_loss = pairwise_loss(scores_chosen[-1], scores_rejected[-1])
    if n > 1:
        aux_loss = sum(
            pairwise_loss(sc, sr)
            for sc, sr in zip(scores_chosen[:-1], scores_rejected[:-1])
        ) / (n - 1)
        return final_loss + 0.3 * aux_loss
    return final_loss
```

Final step is the authoritative judgment (weight 1.0), earlier steps provide auxiliary signal (weight 0.3). Dynamic — works regardless of loop count.

**Planned Version 4:** Reduce L2 coefficient from 0.01 to 0.001 — current L2 is over-regularizing, suppressing score magnitude and hurting discrimination on clear cases.

### 3.7 Training Progression

**Run 1** (5k samples, unshuffled, v1 loss, MAX_LENGTH=512):
| Epoch | Accuracy | Loss |
|---|---|---|
| 1 | ~61.9% | 2.607 |
| 2 | ~66.1% | 2.480 |
| 3 | ~68.7% | 2.369 |

**Run 2** (15k samples, shuffled, v3 loss, MAX_LENGTH=1024, BATCH_SIZE=2):
| Epoch | Accuracy | Loss |
|---|---|---|
| 1 | ~53.8% | 0.927 |
| 2 | ~59.5% | 0.877 |
| 3 | ~62.4% (at batch 1500/7500) | 0.858 |

Note: lower raw accuracy in run 2 is partly due to L2 over-regularization suppressing score magnitude. Evaluation results are more informative than training accuracy.

---

## 4. Results

### 4.1 Proof of Concept — Alignment Signal in Frozen Representations
The evaluator learns above-chance alignment detection from **frozen** Ouro representations. Ouro was never explicitly trained to encode constitutional information — yet a lightweight probe extracts meaningful signal from its hidden states.

**Implication:** Constitutional alignment information exists latently in models trained on human preference data. A jointly trained CLT evaluator integrated from pretraining would produce dramatically stronger signal.

### 4.2 Trajectory Analysis — Run 1 (unshuffled)
Sample trajectories, 7/10 correct:

**Example 3 (correct, margin 2.55):**
```
Chosen:   [-2.445, -3.399, -1.401, -1.233]  ← dips then recovers strongly
Rejected: [-4.046, -5.357, -3.757, -3.787]  ← dips and stays low
```

**Example 5 (correct, margin 2.05):**
```
Chosen:   [0.835, 0.106, 1.157, 0.902]    ← fluctuates but positive
Rejected: [-0.58, -1.589, -0.613, -1.146]  ← consistently negative
```

**Finding:** Chosen and rejected responses show qualitatively different trajectory shapes — the evaluator captures reasoning dynamics across loop iterations, not just the endpoint.

### 4.3 Trajectory Analysis — Run 2 (shuffled, L2=0.01)
Sample trajectories, 4/10 correct on first 10:

Scores are much more bounded (-1.3 to +1.2) vs run 1 (-12 to +14). L2 regularization working but over-suppressing signal on clear cases. Trajectories show less dramatic separation than run 1 due to score compression.

### 4.4 Evaluation Results — Run 1 (unshuffled, epoch 3)

| Progress | Accuracy | Avg Margin |
|---|---|---|
| 2000/8552 | 69.0% | 0.906 |
| 4000/8552 | 54.1% | 0.303 |
| Final | 46.4% | -0.026 |

**Root cause of degradation:** Truncation artifacts. HH-RLHF is ordered by conversation length. At MAX_LENGTH=512, later examples (longer conversations) get cut off mid-sentence, destroying preference signal. This is not difficulty gradient — manual inspection confirmed fragments like "In addition," as the entire rejected response.

**Methodological note:** The 46.4% overall figure should not be reported as model performance. The 65-69% on clean early examples is the valid result.

### 4.5 Evaluation Results — Run 2 (shuffled, epoch 3) — FINAL

| Progress | Accuracy | Avg Margin |
|---|---|---|
| 500/8552 | 57.6% | 0.188 |
| 1000/8552 | 58.5% | 0.211 |
| 2000/8552 | 60.4% | 0.218 |
| 3000/8552 | 61.7% | 0.236 |
| 4000/8552 | 62.5% | 0.255 |
| 5000/8552 | 62.2% | 0.243 |
| 5500/8552 | 61.7% | 0.231 |
| 6000/8552 | 61.0% | 0.221 |
| 6500/8552 | 61.0% | 0.225 |
| 7000/8552 | 61.3% | 0.226 |
| 7500/8552 | 61.3% | 0.226 |
| 8000/8552 | 61.4% | 0.226 |
| 8500/8552 | 61.4% | 0.226 |
| **Final** | **61.3%** | **0.225** |

**Final distribution statistics:**
- Test examples: 8,552
- Accuracy: 61.3%
- Average margin: 0.2251
- Margin std: 0.7841 (down from 1.831 in run 1)
- Min margin: -4.0822 (down from -12.495 in run 1)
- Max margin: 5.1616 (down from 14.757 in run 1)
- Positive margin rate: 61.3%

**Key finding: Degradation pattern eliminated.** Accuracy climbs steadily from 57.6% to 62.5% then stabilizes at 61.3% through the full dataset. No collapse. Shuffle confirmed as the correct fix for truncation artifact contamination.

**Calibration improvement:** Margin std dropped from 1.831 to 0.784. Extreme values eliminated. Scores bounded and calibrated — a direct result of L2 regularization and shuffle.

**Comparison to run 1:** Lower peak accuracy (62.5% vs 69%) but dramatically more consistent and honest. Run 1 peak was inflated by evaluating only on clean short examples. Run 2 accuracy of 61.3% is valid across the full test distribution.

**Next step:** Reduce L2 from 0.01 to 0.001 — current over-regularization is suppressing score magnitude and likely capping accuracy. Expect 65-68% with fix applied.

### 4.6 Evaluation Results — Run 3 (shuffled, L2=0.001, n_concat=4, aux_loss=0.1, 25k samples, epoch 3)

| Progress | Accuracy | Avg Margin |
|---|---|---|
| 500/8552 | 59.4% | 0.604 |
| 1000/8552 | 59.7% | 0.687 |
| 1500/8552 | 61.2% | 0.684 |
| 2000/8552 | 61.9% | 0.697 |
| 2500/8552 | 63.2% | 0.677 |
| 3000/8552 | 63.2% | 0.634 |
| 3500/8552 | 63.4% | 0.604 |
| 4000/8552 | 63.3% | 0.585 |
| 4500/8552 | 63.1% | 0.557 |
| 5000/8552 | 62.5% | 0.522 |
| 5500/8552 | 62.0% | 0.497 |
| 6000/8552 | 61.5% | 0.470 |
| 6500/8552 | 61.4% | 0.466 |
| 7000/8552 | 61.8% | 0.463 |
| 7500/8552 | 62.0% | 0.461 |
| 8000/8552 | 62.0% | 0.458 |
| 8500/8552 | 62.0% | 0.459 |
| **Final** | **61.9%** | **0.458** |

**Final distribution statistics:**
- Test examples: 8,552
- Accuracy: 61.9% (up from 61.3% in run 2)
- Average margin: 0.4575 (up from 0.2251 in run 2 — doubled)
- Margin std: 1.5856 (up from 0.7841 in run 2)
- Min margin: -9.7832
- Max margin: 13.6407
- Positive margin rate: 61.9%

**Sample trajectory analysis (first 10 examples, 5/10 correct):**

Example 1 (correct, margin 1.187): Chosen and rejected trajectories both negative but chosen consistently less negative across all 4 steps — stable separation.

Example 3 (incorrect, margin -0.504): Chosen trajectory recovers mid-sequence but rejected scores higher at final step.

Example 6 (correct, margin 0.003): Effectively a coin flip — trajectories nearly identical throughout, margin near zero.

**Key findings:**

Accuracy improved modestly from 61.3% to 61.9%. More significant is the average margin doubling from 0.225 to 0.458 — the evaluator is substantially more decisive on examples it gets correct. This is the direct effect of reducing L2 from 0.01 to 0.001, allowing scores to spread further from zero.

Tradeoff: margin std increased from 0.784 to 1.586 and extremes returned (-9.8 to +13.6). This is acceptable — the accuracy is stable throughout the test set, not collapsing. The mild dip from 63.4% at batch 3500 to 61.4% at batch 6500 then recovery to 61.9% is a residual of the dataset length distribution but far less severe than run 1.

All scores shifted negative relative to run 2 — mean score is now around -2 to -3. This is a systematic offset, not a calibration failure. The ranking loss only depends on relative scores so absolute magnitude doesn't affect accuracy.

**Comparison across runs:**

| Run | Samples | Final Acc | Avg Margin | Margin Std | Notes |
|---|---|---|---|---|---|
| Run 1 | 5k unshuffled | 46.4% | -0.026 | 1.831 | Truncation artifacts |
| Run 2 | 15k shuffled, L2=0.01 | 61.3% | 0.225 | 0.784 | Clean, stable, over-regularized |
| Run 3 | 25k shuffled, L2=0.001 | 61.9% | 0.458 | 1.586 | Higher confidence, mild variance increase |

**Next step:** Scale to 50k samples. Add cosine LR scheduler before running.

---

## 4.7 Evaluator V2 — GRU-based Temporal Model (Final Version)

### Architecture
Replaces the sliding window concatenation of V1 with a 2-layer GRU that processes the sequence of loop hidden states in order, with an input projection and skip connection.

```
Input: list of [batch, hidden_dim] pooled tensors (one per loop step)

Per step: LayerNorm(2048) → Linear(2048 → 512)   ← normalize + project
2-layer GRU(input=512, hidden=512, dropout=0.1)   ← temporal dynamics
Skip connection: cat([gru_out, final_proj])        ← [batch, 1024]
LayerNorm(1024)
Linear(1024 → 256) + GELU + Dropout(0.1)
Linear(256 → 1)
Output: unbounded scalar
```

### Key design decisions

**Input projection 2048 → 512:** Reduces parameter count before the GRU. With only 4 loop steps, processing full 2048-dim vectors through the GRU is unnecessary — the projection forces compression into a more compact trajectory representation.

**2-layer GRU with dropout=0.1:** Added depth over the initial single-layer design. Dropout between GRU layers provides regularization on temporal patterns. Single-layer GRU had no room to overfit 4 timesteps, but 2-layer adds enough capacity to warrant it.

**Skip connection:** Final scorer concatenates the GRU's final hidden state with the projected final loop state — `[gru_out, final_proj]`. The GRU encodes trajectory dynamics; the skip connection preserves the endpoint representation directly. Scorer sees both. This was the most substantive architectural improvement over the initial V2 design.

**GRU hidden size = 512:** Intentional compression from 2048. If performance plateaus after dataset scaling, bumping to 1024 is a one-line change.

**LayerNorm per step before projection:** Removes absolute scale differences between examples — directly addresses score drift observed in run 3.

**`trajectory()` runs GRU incrementally:** GRU hidden state carries forward step by step. Each step's score reflects everything seen up to that point. Skip connection uses the current step's projected vector, not the final one, so each trajectory score is self-consistent.

**`trajectory_loss` is unchanged:** V2's `trajectory()` returns the same `(scores, trajectory)` interface as V1.

### Why GRU over concatenation
The sliding window in V1 treats the trajectory as a static feature vector — it sees the same 4 states regardless of order or direction of change. A GRU captures directionality: a trajectory that goes [-3, -2, -1, 0] (improving) produces a different GRU hidden state than [0, -1, -2, -3] (degrading), even though both contain the same four values.

### Training improvements in train2.py
- **Gradient accumulation:** `GRAD_ACCUM_STEPS = 4` → effective batch size 8 without extra VRAM
- **Cosine LR scheduler with warmup:** LinearLR warmup for 200 steps then CosineAnnealingLR decay. Peak LR lowered to 5e-5 from 1e-4 — scheduler handles decay
- **F.logsigmoid:** Replaces `torch.log(torch.sigmoid(x) + 1e-8)` with `F.logsigmoid(x)` which uses the log-sum-exp trick internally for better numerical stability
- **L2 reduced to 1e-5:** Minimal regularization — just prevents score explosion, no longer active regularization
- **Checkpoint saves optimizer state:** Enables training resumption
- **Hook validation:** `validate_hook_output()` runs once on first forward pass to verify Ouro's output structure hasn't changed
- **Pairwise linear probe:** Runs before training as a signal sanity check

### Linear Probe Finding (Critical)
Before training, a logistic regression was run on 200 examples from the frozen Ouro hidden states using a pairwise framing — classifying the direction of `(chosen - rejected)` margin vectors rather than labeling individual responses.

**Result: 93.75% accuracy (final loop state), 91.25% (all states concatenated)**

This is a major finding. A linear classifier on frozen representations nearly saturates the task. This reframes what the evaluator is actually doing — it is not extracting hidden signal that requires a complex model to detect. The preference signal is strongly linearly encoded in Ouro's hidden states. The GRU evaluator's job is to learn a better decision boundary than a linear one on harder cases.

**Implication for the paper:** The 61-62% accuracy ceiling of V1 is not a representation quality problem — the signal is there (93% linearly separable). It is a decision boundary problem. The gap between 93% linear probe and 62% evaluator accuracy is explained by the small training set (25k of 160k), frozen representations, and the V1 architecture's inability to capture trajectory dynamics. V2 with the GRU addresses the dynamics. Scaling to the full dataset addresses the training set size. Joint training from pretraining would address the frozen representation constraint.

**Note on probe framing:** The initial probe implementation labeled individual responses as chosen (1) or rejected (0). This is the wrong task — preference is inherently pairwise. The corrected probe classifies `(chosen - rejected)` direction vectors, directly testing whether relative preference is linearly encoded. This is the correct framing and produced the 93.75% result.

### Training Run 4 Results (V2, trajectory supervision, 25k samples)

| Epoch | Loss | Training Accuracy |
|---|---|---|
| 1 | 0.6798 | 59.5% |
| 2 | 0.6424 | 65.0% |
| 3 | 0.6117 | **67.8%** |

**67.8% training accuracy** vs V1's best of 62.4% — a 5+ point improvement at the same dataset size. Loss still falling cleanly into epoch 3, suggesting the model had not saturated. The GRU is capturing trajectory dynamics that the sliding window concatenation could not.

Epoch 2 jump from 59.5% to 65.0% is particularly notable — same pattern as V1 but landing significantly higher. The cosine LR schedule with warmup produced stable convergence throughout.

Checkpoints saved to `checkpoints_v2/`. Evaluation on full test set pending via `evaluate2.py`.

### Evaluation Results — Run 4 (V2, trajectory supervision, 25k samples, epoch 3)

| Progress | Accuracy | Avg Margin |
|---|---|---|
| 500/8552 | 59.0% | 0.391 |
| 1000/8552 | 61.6% | 0.497 |
| 1500/8552 | 63.1% | 0.494 |
| 2000/8552 | 63.4% | 0.498 |
| 2500/8552 | 64.4% | 0.512 |
| 3000/8552 | 65.1% | 0.524 |
| 3500/8552 | 65.5% | 0.535 |
| 4000/8552 | 65.4% | 0.536 |
| 4500/8552 | 65.2% | 0.526 |
| 5000/8552 | 64.7% | 0.504 |
| 5500/8552 | 64.1% | 0.484 |
| 6000/8552 | 63.3% | 0.463 |
| 6500/8552 | 63.2% | 0.466 |
| 7000/8552 | 63.4% | 0.472 |
| 7500/8552 | 63.5% | 0.470 |
| 8000/8552 | 63.5% | 0.468 |
| 8500/8552 | 63.3% | 0.472 |
| **Final** | **63.2%** | **0.470** |

**Final distribution statistics:**
- Test examples: 8,552
- Accuracy: 63.2% (up from 61.9% V1)
- Average margin: 0.4699
- Margin std: 1.4923 (down from 1.586 in V1 run 3)
- Min margin: -6.1179 (improved from -9.78)
- Max margin: 10.2145 (improved from 13.64)
- Positive margin rate: 63.2%

**Trajectory analysis (first 10 examples, 4/10 correct):**

All scores strongly negative and monotonically decreasing across loop steps. Example 1 chosen: [-3.3, -5.5, -6.0, -6.3] — the GRU has learned that later loop states carry stronger signal. Correct cases show chosen consistently less negative than rejected at every step.

**Key findings:**

Accuracy improved from V1's 61.9% to 63.2% — a real gain. Margin calibration also improved — extremes more bounded (-6.1 to +10.2 vs -9.8 to +13.6). The accuracy dip pattern reappears — peaks at 65.5% around batch 3500 then falls to 63.2%. Same shape as V1, shifted up by ~1.5 points. Suggests dataset length ordering effect persists at 25k scale.

**Training vs test gap:** 67.8% training accuracy → 63.2% test accuracy. 4.6 point gap suggests mild overfitting to 25k examples. Scaling to 50k+ should close this.

**Comparison across all runs:**

| Run | Architecture | Samples | Test Acc | Avg Margin | Notes |
|---|---|---|---|---|---|
| Run 1 | V1 MLP | 5k unshuffled | 46.4% | -0.026 | Truncation artifacts |
| Run 2 | V1 MLP | 15k shuffled | 61.3% | 0.225 | Clean baseline |
| Run 3 | V1 MLP | 25k shuffled | 61.9% | 0.458 | Best V1 |
| Run 4 | V2 GRU | 25k shuffled | **63.2%** | 0.470 | Best overall |

### Training Run 5 Results (V2, forward-only, 25k samples)

| Epoch | Loss | Training Accuracy |
|---|---|---|
| 1 | 0.6587 | 60.1% |
| 2 | 0.6207 | 64.6% |
| 3 | 0.5893 | **67.97%** |

**Trajectory supervision is NOT the bottleneck.** Run 5 (forward-only) achieved 67.97% training accuracy vs run 4's (trajectory) 67.82% — a difference of 0.15 points. Effectively identical. Removing trajectory supervision neither helped nor hurt meaningfully.

**Comparison:**

| Run | Architecture | Supervision | Train Acc | Test Acc |
|---|---|---|---|---|
| Run 4 | V2 GRU | Trajectory | 67.8% | 63.2% |
| Run 5 | V2 GRU | Forward-only | 68.0% | pending |

Loss also fell further in run 5 (0.5893 vs 0.6117) suggesting marginally better convergence, but the accuracy difference is negligible.

### Bottleneck Analysis — Mean Pooling

With trajectory supervision ruled out as the bottleneck, the most likely remaining candidate is **mean pooling**.

Mean pooling collapses the entire token sequence into a single vector by averaging. This loses:
- Token-level alignment signals — a single harmful token buried in an otherwise fine response gets averaged out
- Localized spans — refusal phrases, harmful instructions, or critical transitions that occupy a small fraction of the sequence get diluted proportionally to sequence length
- Positional structure — the model cannot distinguish where in the sequence a signal appears

At MAX_LENGTH=1024, longer conversations dilute local signals even further.

### Training Run 6 — Full-rank Attention Pooling (collapsed)

First attempt at attention pooling used a full-rank key projection: `Linear(2048 → 2048)`. This produced **accuracy collapse** — training accuracy rose for the first few hundred batches then fell back toward 50%.

**Root cause:** The full-rank key projection had ~4.2M parameters — more than the rest of the evaluator combined. This gave the attention mechanism enough capacity to memorize arbitrary token-level patterns in the training data rather than learning a generalizable "which positions matter" weighting. Early in training it latched onto spurious correlations that happened to work on the first few hundred batches, then those patterns stopped generalizing as it saw more data and accuracy decayed.

### Training Run 7 — Low-rank Attention Pooling (current)

Fix: replace full-rank key projection with a low-rank bottleneck projecting to 128 dimensions.

```python
class AttentionPool(nn.Module):
    def __init__(self, hidden_dim, attn_dim=128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim, bias=False)  # 2048 → 128
        self.query = nn.Parameter(torch.randn(attn_dim) * 0.01)

    def forward(self, hidden, attention_mask):
        keys = self.proj(hidden)           # [batch, seq_len, 128]
        scores = keys @ self.query         # [batch, seq_len]
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.einsum("bs,bsd->bd", weights, hidden)
```

Parameter count drops from ~4.2M to ~260k. The query and key projection both live in 128-dimensional space — enough to learn structural token weighting (response vs prompt, EOS position, refusal markers) but not enough to memorize training examples. This forces the attention mechanism to learn genuinely generalizable patterns.

**Key insight:** The bottleneck doesn't restrict what the evaluator *uses* after pooling — the full 2048-dimensional pooled vector is still passed forward. It only restricts the complexity of the *weighting decision*, which is exactly the right place to regularize.

### Training Run 7 Results (V2 + low-rank attention pooling, 25k samples, forward-only)

Note: `max_samples` was not updated from 25000 as planned, making this a full 25k run — directly comparable to runs 4 and 5.

| Epoch | Loss | Training Accuracy |
|---|---|---|
| 1 | 0.6566 | 60.1% |
| 2 | 0.6074 | 66.4% |
| 3 | 0.5665 | **70.2%** |

**70.2% training accuracy — first time breaking through the 68% ceiling.** This is a 2.2 point improvement over runs 4 and 5 (both ~68%), which used mean pooling on the same dataset size. The improvement is attributable entirely to the switch from mean pooling to low-rank attention pooling, as all other hyperparameters are identical.

The epoch 3 progression is notably different from previous runs. Early epoch 3 accuracy shot to 69-70% rapidly (batch 1000-1500) and plateaued there for the rest of training. Previous runs with mean pooling showed a similar rapid rise but peaked at 67-68% and held flat. The attention mechanism is learning a better decision boundary, consistent with the hypothesis that mean pooling was discarding token-level signal.

**Score distribution:** Scores are no longer uniformly negative. The attention pooling produces more centered representations — scores range widely on both sides of zero throughout training. This is a healthier signal than the systematic negative bias seen in runs 4 and 5.

**Plateau at ~70%:** Accuracy stabilized around 70-70.2% in the second half of epoch 3 and did not continue rising. The plateau is real — the final 4000 batches of epoch 3 oscillate between 70.1% and 70.3% without improvement. This is not noise. Possible explanations: (1) the 25k training set is genuinely insufficient to push further — the evaluator has learned everything it can from this data, (2) the frozen Ouro representations impose a ceiling that attention pooling alone cannot overcome, (3) the architecture itself (GRU hidden=512, attn_dim=128) lacks the capacity to model the remaining hard examples.

Test set evaluation complete.

**Final test results — Run 7:**
- Test accuracy: **65.0%** (up from 63.2% in run 4)
- Average margin: ~0.65-0.76 (highest yet, declining through evaluation)
- Same dip pattern observed: peaks ~66% around batch 3500, slides to ~64% by batch 6500, recovers slightly to ~65% final

**Generalization gap analysis:**

| Run | Train Acc | Test Acc | Gap |
|---|---|---|---|
| Run 4 | 67.8% | 63.2% | 4.6 pts |
| Run 7 | 70.2% | 65.0% | 5.2 pts |

Attention pooling improved both training and test accuracy, but the generalization gap widened slightly. The model learned more from the training distribution but also overfit more.

**On the dip pattern:** The dataset is shuffled (seed=42), so the accuracy dip that peaks mid-evaluation and recovers is not a dataset ordering artifact. More likely explanations: (1) statistical noise in the cumulative accuracy metric — early batches have higher variance, (2) the test set itself has a non-uniform distribution of difficulty with harder examples concentrated in certain regions, (3) the LR schedule during training left the model better calibrated on certain regions of the preference space.

**Comparison across all runs updated:**

| Run | Architecture | Pooling | Samples | Train Acc | Test Acc |
|---|---|---|---|---|---|
| Run 1 | V1 MLP | Mean | 5k | 46.4% (test) | 46.4% |
| Run 2 | V1 MLP | Mean | 15k | ~62.4% | 61.3% |
| Run 3 | V1 MLP | Mean | 25k | ~62.4% | 61.9% |
| Run 4 | V2 GRU | Mean | 25k | 67.8% | 63.2% |
| Run 5 | V2 GRU | Mean | 25k | 68.0% | pending |
| Run 6 | V2 GRU | Full-rank attn | 15k | collapsed | — |
| Run 7 | V2 GRU | Low-rank attn (128) | 25k | **70.2%** | **65.0%** |
| Run 8 | V3 MLP (no GRU) | Low-rank attn (256) | 25k | pending | — |

---

## 4.8 GRU Bottleneck Hypothesis and Architecture V3 (Final Loop State MLP)

### Hypothesis

After the 70% plateau in run 7, the GRU was identified as a potential bottleneck. The reasoning:

The GRU processes 4 loop states sequentially, building a hidden state that encodes trajectory dynamics. But with only 4 timesteps, the GRU may not be adding meaningful temporal signal — it may instead be introducing sequential information loss. Each GRU step applies a learned gating mechanism that selectively forgets and updates, meaning signal from early loop states could be partially discarded by the time the final state is scored. For a sequence of 4 items, this is likely hurting more than helping.

Additionally, the linear probe achieved 93.75% on the **final loop state alone** using simple mean pooling. The GRU's job was supposed to be extracting temporal dynamics from the trajectory, but if the final state already contains the bulk of the constitutional signal, the GRU is mostly adding noise and capacity for overfitting.

**Conclusion:** Remove the GRU entirely. Score directly from the final loop state via attention pooling → MLP. This is the closest learned architecture to the linear probe that got 93.75%.

### Architecture V3 — ConstitutionalEvaluatorTest

```
Input: final hidden state only [batch, seq_len, 2048]

AttentionPool (attn_dim=256):
  Linear(2048→256, no bias) → dot with query[256] → softmax → weighted sum → [batch, 2048]

LayerNorm(2048)
Linear(2048→512) + GELU + Dropout(0.1)
Linear(512→256) + GELU + Dropout(0.1)
Linear(256→1)
Output: unbounded scalar
```

Key decisions vs V2:
- **No GRU** — eliminates temporal bottleneck, scores final loop state directly
- **attn_dim=256** — up from 128, more pooling capacity. Still far below full-rank (2048) so overfitting risk remains low (~525k params vs 4.2M full-rank)
- **Wider MLP** — 2048→512→256→1 with two GELU layers. Since the MLP is now the only nonlinear transform, it needs more capacity than before
- **`eta_min=1e-5`** on cosine scheduler — LR floors at 1e-5 instead of 0, prevents over-decay in final batches
- **`CHECKPOINT_DIR = "checkpoints_test"`** — isolated from V2 checkpoints
- **Files:** `evaluator_test.py`, `train_test.py`

### Theoretical expectation

If the GRU was the bottleneck: this architecture should exceed 70% training accuracy, potentially approaching the linear probe ceiling more closely. If dataset size is the bottleneck: this architecture will plateau at the same ~70% level. If the frozen representations are the ceiling: we will not exceed ~70% regardless of architecture changes.

### Training Run 8 Results (V3 no-GRU MLP, attn_dim=256, 25k samples)

| Epoch | Loss | Training Accuracy |
|---|---|---|
| 1 | 0.6601 | 58.9% |
| 2 | 0.6176 | 64.3% |
| 3 | pending | ~67-68% |

**The GRU was not the bottleneck — it was contributing.** Run 8 without the GRU is tracking back to the same 67-68% level as runs 4 and 5 (mean pooling + GRU), erasing the gains from attention pooling entirely. This means the 70.2% in run 7 came from attention pooling AND the GRU working together, not attention pooling alone.

The GRU's temporal modeling over the 4 loop states is doing real work. Even though the linear probe achieved 93.75% on the final loop state alone, the trained evaluator benefits from seeing how the representation evolves across all 4 steps. The GRU captures directional change in the hidden state trajectory that the final state alone does not encode.

**Conclusion:** Run 7's architecture (GRU + low-rank attention pooling, attn_dim=128) is the current best at 70.2% training / 65.0% test. The correct next step is scaling this architecture to 50k samples, not further ablating the GRU.

**Updated comparison across all runs:**

| Run | Architecture | Pooling | Samples | Train Acc | Test Acc |
|---|---|---|---|---|---|
| Run 1 | V1 MLP | Mean | 5k | 46.4% (test) | 46.4% |
| Run 2 | V1 MLP | Mean | 15k | ~62.4% | 61.3% |
| Run 3 | V1 MLP | Mean | 25k | ~62.4% | 61.9% |
| Run 4 | V2 GRU | Mean | 25k | 67.8% | 63.2% |
| Run 5 | V2 GRU | Mean | 25k | 68.0% | pending |
| Run 6 | V2 GRU | Full-rank attn | 15k | collapsed | — |
| Run 7 | V2 GRU | Low-rank attn (128) | 25k | **70.2%** | **65.0%** |
| Run 8 | V3 MLP (no GRU) | Low-rank attn (256) | 25k | ~68% | pending |

---

## 4.9 Scaled Linear Probe — Critical Finding

### Motivation

The original linear probe (section 4.7) used 200 examples and reported 93.75% pairwise accuracy. Concern: small sample size may have inflated the result. A scaled probe at 1000 examples was run (`probe_test.py`) to validate.

### Hook Verification (confirmed in same script)

- `output[0]`: BaseModelOutputWithPast (standard transformer output)
- `output[1]`: list of 4 tensors, each `[batch, seq_len, 2048]`, dtype=bfloat16
- `output[2]`: list (additional outputs)

**Confirmed:** `output[1]` contains genuine loop iteration states, not layer outputs or attention weights. The CLT premise is valid — we are reading actual loop dynamics.

### Probe Results at 1000 Examples

| Probe Type | Final State Only | All States Concat |
|---|---|---|
| Pairwise (chosen - rejected) | **84.5%** | **86.25%** |
| Independent classification | 21.75% | 22.75% |

### Key Findings

**1. 93.75% was inflated.** The real pairwise ceiling is 84.5% at 1000 examples, down from 93.75% at 200 examples. Still meaningful headroom above 65% test accuracy, but not the massive gap originally believed. The small-sample probe overfitted to noise.

**2. Independent classification at 21.75% — below chance.** This is the fundamental finding. The representations cannot distinguish chosen from rejected when scored independently. Preference signal exists only in the relative difference between pairs.

**3. Inverted polarity.** Flipping the 21.75% gives 78.25%. The representations do carry independent signal, but with inverted polarity — rejected responses likely trigger stronger/different activation profiles that a naive classifier picks up with the wrong sign.

### What This Explains

- **Why every pointwise architecture hit 67-70%:** independent scoring has a theoretical ceiling around 75-78% given the inverted polarity signal.
- **Why trajectory supervision didn't help:** the signal isn't in the absolute trajectory shape.
- **Why the GRU adds only 2 points:** it extracts some temporal ordering signal but the fundamental representation doesn't support absolute scoring.

---

## 4.10 Structural Shortcut Diagnostic

A diagnostic (`diagnostic.py`) was run on 500 test examples to check whether trivial surface features separate chosen from rejected.

**Results:**
- Chosen tokens: mean=156.7, std=122.7
- Rejected tokens: mean=169.2, std=129.9
- "Longer = chosen": 45.6% (anti-correlated — rejected is longer on average)
- "Larger norm = chosen": 43.0%
- Hidden state norm ratio: ~0.96 across all 4 loop steps
- Mean activation ratio: ~0.997 across all 4 loop steps
- Length-norm correlation: 0.5123

**Conclusion:** No structural shortcut exists. Length, norms, and activation magnitudes do not separate chosen from rejected. Any preference signal is in the representation geometry, not surface features. The evaluator is genuinely learning something non-trivial.

---

## 4.11 Feature Extraction Decoupling

### Motivation

Every training run was spending ~95% of time on identical Ouro forward passes. The model is frozen and deterministic — running it repeatedly on the same data is pure waste.

### Implementation (`extract_features.py` + `train2_fast.py`)

- `extract_features.py` runs Ouro once on the full dataset and saves hidden states as chunked `.pt` files
- Save mode "raw": full `[1, seq_len, 2048]` per loop step in float16 + attention masks
- `CHUNK_SIZE=100` examples per file
- `train2_fast.py` loads one chunk at a time (lazy), shuffles chunk order per epoch and within-chunk

**Peak RAM:** one chunk (~300MB) instead of the entire dataset in memory.

**Initial bug:** first version loaded all chunks into RAM simultaneously, freezing the machine. Fixed with chunk-sequential loading.

**Validation:** reproduced run 7's 70% accuracy exactly on precomputed features. Decoupling confirmed valid.

**Training time:** minutes per epoch instead of hours. All subsequent architecture experiments use precomputed features.

---

## 4.12 Pairwise Evaluator

### Hypothesis

If preference signal is relational (84.5% pairwise probe) rather than absolute (21.75% independent), building an evaluator that sees both responses simultaneously should fundamentally change the ceiling.

### Architecture (`evaluator_pairwise.py`)

Shared AttentionPool for both responses → per-step difference `(chosen_pooled - rejected_pooled)` → GRU over difference trajectory → skip connection from final difference → scorer outputs single scalar (positive = chosen preferred).

### First Attempt — Degenerate

Result: accuracy rocketed to 99%+ within epoch 1. Scores of +13 on every example. Test eval: 100% accuracy, 8552/8552 test examples correct, all scores ~+13.

**Flip test (`flip_test.py`) — FAILED.** Swapped chosen and rejected inputs:
- Example 1: Normal=+13.16 | Flipped=+13.05
- Example 2: Normal=+13.53 | Flipped=+13.45

Signs did not flip. Model outputs +13 regardless of input order.

**Root cause:** The loss `-F.logsigmoid(score)` only ever pushed scores positive. Training always presented chosen first. The model discovered: output large positive constant → loss ≈ 0. It never examined actual content. Specifically: (1) the difference `chosen - rejected` does flip sign on swap, but (2) LayerNorm has a bias term that absorbs sign information, (3) the GRU + scorer is expressive enough to map everything to a positive constant, (4) no training example ever required a negative output, (5) no L2 regularization allowed scores to grow unbounded to +13.

**Lesson: always run a flip test on pairwise models before trusting results.**

### Fixed Pairwise Evaluator (`train_pairwise_fast.py` v2)

Three fixes applied:

1. **Random 50% swap:** each batch, coin flip — either `evaluator(chosen, rejected)` with target=+1, or `evaluator(rejected, chosen)` with target=-1. The model must learn direction, not just magnitude.
2. **Directional loss:** `-F.logsigmoid(target * score)` instead of `-F.logsigmoid(score)`. When target=-1, loss pushes score negative.
3. **Score regularization:** `1e-4 * score.pow(2).mean()` — prevents scores from growing unbounded.
4. **Accuracy metric fixed:** checks if score sign matches target, not just "is it positive?"

**Results:**
- Epoch 1: 60.9%
- Epoch 2: 67.4%
- Epoch 3: 70.4%

Same 70% ceiling as every other architecture, despite seeing both responses simultaneously.

---

## 4.13 Calibrated Evaluator

### Hypothesis

The 21.75% (flipped: 78.25%) independent probe means polarity signal exists but is inverted. Adding a classification loss alongside ranking loss should anchor chosen→positive and rejected→negative, exploiting this inverted signal.

### Architecture (`evaluator_calibrated.py`)

Identical to V2 (GRU + AttentionPool). Only the loss changed:

```
loss = ranking_loss + 0.5 * BCE_classification_loss
ranking: -logsigmoid(score_chosen - score_rejected)
classification: BCE forcing chosen scores positive, rejected scores negative
```

Tracked two metrics: RankAcc (chosen > rejected) and ClassAcc (chosen > 0 AND rejected < 0).

**Results:**
- Epoch 1: RankAcc=58.4%, ClassAcc=53.7%
- Epoch 2: RankAcc=65.4%, ClassAcc=57.0%
- Epoch 3: RankAcc=69.3%, ClassAcc=59.0%

RankAcc converges to ~70% — same ceiling. ClassAcc barely moved above chance (59%). The BCE component provided no meaningful benefit.

---

## 4.14 Linear Probe Replication Attempt

### Hypothesis (from external suggestion)

The GRU and attention pooling destroy signal. The probe's 84.5% comes from preserving high-dimensional features. A single `nn.Linear(8192, 1)` trained with SGD should approach probe accuracy if compression is the bottleneck.

### Architecture (`train_linear_fast.py`)

Concatenate 4 loop states (2048×4=8192 features), mean pool across tokens, single `nn.Linear(8192, 1)`. 8,193 parameters total. No nonlinearities.

**Result:** reached only 64% by batch 6000/12500 in epoch 3. Killed early — clearly plateaued well below V2.

### What This Proves

1. **The compression thesis was wrong.** The linear model without compression gets worse than V2 with compression (64% vs 70%). The GRU and attention pooling add 6 points of value, they do not destroy signal.
2. **The 84.5% probe accuracy comes from two things the evaluator cannot replicate:** (a) pairwise differences — seeing both responses simultaneously, (b) L-BFGS optimizer — convex optimization on the full dataset, not SGD on mini-batches of 2.
3. **The architecture search is effectively closed.** V2 at 70% is extracting nearly everything available from independent scoring on these representations.

---

## 4.15 Complete Architecture Summary

| Architecture | Train Acc | Test Acc | Notes |
|---|---|---|---|
| V1 MLP concat | 62% | 61.9% | Baseline |
| V2 GRU + mean pool | 68% | 63% | |
| V2 GRU + AttentionPool(128) (Run 7) | 70% | 65% | **Best pointwise** |
| V2 GRU + AttentionPool(2048) full-rank | collapsed | — | Overfitting |
| No-GRU AttentionPool(256) + MLP | 68% | — | GRU adds 2pts with attn pool |
| No-GRU AttentionPool(128) | 68% | — | Confirmed |
| Pairwise degenerate (no swap) | 100% | 100% | Failed flip test — constant output |
| Pairwise fixed (25k, EBS=8, cosine) | 70% | — | Same ceiling |
| Pairwise fixed (50k, EBS=8, cosine) | 70% | — | Data alone doesn't help |
| Calibrated (ranking + BCE) | 69% | — | BCE barely helped (ClassAcc=59%) |
| Linear only (probe replication) | 64% | — | Compression HELPS, not hurts |
| **Pairwise (50k, EBS=64, warmup→const→decay)** | **pending** | **—** | **Optimization fix — next run** |

---

## 4.16 The Core Finding — Relational vs Absolute Encoding

The most important finding of the entire project, emerging from the scaled probe:

**Ouro-2.6B-Thinking's loop states encode preference relationally (84.5% pairwise) but not absolutely (21.75% independent, below chance).**

This is not a failure of the evaluator architectures. It is a property of the representations themselves. Seven different architectures across three different loss functions all converge to 68-70% — this ceiling is a property of the frozen representations and the dataset size, not the evaluator design.

The independent classification ceiling is approximately 75-78% (given the inverted polarity signal). The evaluators are approaching this ceiling. To exceed it would require either (a) a pairwise evaluator that sees both responses simultaneously, (b) joint training where the Ouro representations are shaped to support absolute scoring, or (c) a significantly larger dataset where the evaluator can learn the polarity inversion statistically.

This reframes the paper contribution: the CLT is not just an alignment monitoring architecture — it demonstrates that preference signal in looped transformers is fundamentally relational, which has implications for how reward models and RLHF should be designed for looped architectures.

---

## 4.19 Scaling Experiment — 50k Features on External SSD

### Setup
- External SanDisk SSD (465GB) mounted at `/mnt/sandisk`
- Partitioned with ext4 (`/dev/sda3`), ~464GB usable
- `extract_features.py` updated: `FEATURE_DIR="/mnt/sandisk/ouro_features"`, `MAX_SAMPLES=50000`
- Features saved in float16
- Estimated disk usage: ~60GB for 50k raw features
- 500 chunks at CHUNK_SIZE=100

### Pairwise 50k Training Run (BATCH_SIZE=2, EBS=8, LR=1e-4, cosine schedule)

- Epoch 1: continuous growth to 64% — same trajectory as 25k
- Epoch 2: plateau at 67-68% for the **entire epoch** — identical to 25k behavior
- Epoch 3: briefly touched 71%, settled back to ~70%

**Observation:** The epoch 2 plateau is identical at 25k and 50k. This is not a data issue — it is a cosine LR schedule artifact. During epoch 2, LR is in the 6e-05 to 3e-05 range: too high to fine-tune, too low to escape local structure. Epoch 1 does the big learning, epoch 2 coasts, epoch 3's low LR squeezes out the last 2-3 points.

### Diagnosis: Optimization, Not Data

The fact that 50k hit the same epoch 2 plateau as 25k proves the 70% wall is partly an optimization problem. More data provides more gradient diversity but the LR schedule determines when that translates to accuracy gains. The cosine schedule wastes ~33% of training on a dead zone.

---

## 4.20 Optimization Overhaul — Next Training Run

### Problem Identified
- Cosine schedule over 3 epochs wastes epoch 2 entirely
- Batch size of 2 produces extremely noisy gradients
- LR of 1e-4 was tuned for EBS=8, not optimal for larger batches
- The L-BFGS vs SGD gap (84.5% vs 64% on linear model) suggests larger effective batch size could partially close the optimization gap

### VRAM Analysis
- BATCH_SIZE=2 uses ~2GB VRAM on precomputed features (no Ouro model loaded)
- RTX 5070 Ti has 16GB available when Ouro is not loaded
- Variable sequence lengths cause occasional spikes
- Safe ceiling for precomputed features: BATCH_SIZE=8-16

### New Training Configuration

| Parameter | Old | New | Rationale |
|---|---|---|---|
| BATCH_SIZE | 2 | 8 | Fits comfortably in VRAM |
| GRAD_ACCUM_STEPS | 4 | 8 | EBS=64 for smoother gradients |
| Effective Batch Size | 8 | 64 | 8x increase, closer to L-BFGS stability |
| LEARNING_RATE | 1e-4 | 3e-4 | Scaled ~3x for 8x EBS (linear scaling rule) |
| WARMUP_STEPS | 200 | 100 | Fewer total steps, don't waste them |
| Schedule | Cosine (full) | Warmup→Constant→Decay(last 20%) | Eliminates dead epoch 2 |

### Schedule Details (50k examples, EBS=64)
- Total optimizer steps: ~2344 across 3 epochs (~781/epoch)
- Warmup: 100 steps (ramps 0.1→1.0x LR)
- Constant: ~1775 steps at full 3e-4 LR (continuous learning)
- Decay: ~469 steps cosine to 0 (final refinement)

### Why This Should Help
1. **No dead epoch:** Constant LR for ~80% of training means the model learns continuously
2. **Cleaner gradients:** EBS=64 averages over 64 examples per update, reducing noise from individual pairs
3. **Appropriate LR:** 3e-4 is proportional to the batch size increase
4. **Faster convergence:** Each optimizer step sees 8x more data

This is the test of whether optimization improvements + 2x data breaks the 70% ceiling. If it holds, the ceiling is representational and the paper can be written.

---

## 4.21 External Suggestions Evaluated

### Dismissed
- **"Compression is the bottleneck"** — Linear model without compression got 64% (worse than V2's 70%). Compression helps.
- **Transformer over loop states** — 4 timesteps is too few; transformer would overfit instantly
- **Loss weighting / final state bonus** — Skip connection already does this
- **Norm difference as feature** — Diagnostic showed norms don't separate (43% accuracy)
- **Late aggregation / per-token scoring** — Premature; optimization gap explains more than architecture differences
- **EBS=256** — Too aggressive. Only ~195 steps/epoch at 50k — Adam can't learn in that few updates unlike L-BFGS

### Valid
- **L-BFGS vs SGD gap** — Real. Linear model: 64% SGD vs 84.5% L-BFGS on same features. Larger batch size partially addresses this.
- **Dataset noise** — HH-RLHF is noisy. Standard reward models reach 72-75% on the full dataset. Noise ceiling is above 70% but not by a huge margin.
- **Subtraction before pooling** — Interesting in principle but impractical. Chosen and rejected have different sequence lengths; token-level subtraction requires alignment.

---

## 4.17 Current Status and File Inventory

### Completed
- Architecture search exhausted (8+ architectures, 3 loss functions, all converge to 68-70%)
- Feature extraction decoupled from training (`extract_features.py`)
- 50k features extracted to external SSD (`/mnt/sandisk/ouro_features`, 500 chunks)
- Diagnostic suite built and validated (`probe_test.py`, `diagnostic.py`, `flip_test.py`)
- Hook verification confirmed: `output[1]` is genuine loop iteration states
- Optimization bottleneck identified (cosine LR dead zone + small EBS)
- Confirmed: more data alone (50k vs 25k) doesn't break ceiling with old optimization config

### In Progress
- Pairwise evaluator on 50k data with optimized training config (EBS=64, warmup→constant→decay)

### Pending
- Flip test on new pairwise checkpoint
- Test eval on pairwise (all versions) and calibrated checkpoints
- Full 161k extraction (only if optimization fix breaks ceiling)
- Paper writeup

### Evaluators
- `evaluator2.py` — V2: GRU + AttentionPool(128), Run 7 architecture (best pointwise)
- `evaluator_test.py` — No-GRU: AttentionPool(256) + MLP only
- `evaluator_pairwise.py` — Pairwise: shared pool, difference trajectory through GRU, LayerNorm(bias=False)
- `evaluator_calibrated.py` — Calibrated: identical to V2, used with combined loss

### Training Scripts
- `train2.py` — V2 with live Ouro inference
- `train2_fast.py` — V2 on precomputed features (chunk-sequential)
- `train_test.py` — No-GRU evaluator, live Ouro
- `train_pairwise_fast.py` — Pairwise on precomputed features (50% swap, EBS=64, warmup→constant→decay, current version)
- `train_calibrated_fast.py` — Calibrated on precomputed features
- `train_linear_fast.py` — Single linear layer on precomputed features

### Evaluation Scripts
- `evaluate2.py` — V2 evaluation
- `evaluate_test.py` — No-GRU evaluation
- `evaluate_pairwise.py` — Pairwise evaluation
- `evaluate_calibrated.py` — Calibrated evaluation (reports rank + class accuracy)

### Diagnostics
- `probe_test.py` — Linear probe at scale (1000 examples, pairwise + independent)
- `diagnostic.py` — Structural shortcut detection (length, norms, activations)
- `flip_test.py` — Antisymmetry test for pairwise evaluator
- `extract_features.py` — Feature extraction from Ouro to disk (currently set to SSD, 50k)

### Checkpoints
- `checkpoints_v2/` — V2 Run 7 (best pointwise)
- `checkpoints_test/` — No-GRU evaluator
- `checkpoints_pairwise/` — Pairwise evaluator (degenerate, fixed 25k, fixed 50k)
- `checkpoints_calibrated/` — Calibrated evaluator
- `checkpoints_linear/` — Linear evaluator

### Feature Storage
- `features/` — DELETED (was 25k on local disk)
- `/mnt/sandisk/ouro_features/` — 50k raw features on external SSD, 500 chunks

---

## 4.18 Lessons Learned

1. **Always run a flip test on pairwise models.** The degenerate solution (constant positive output) achieves 100% accuracy on both train and test when chosen is always presented first. Only swapping inputs reveals the failure.
2. **Linear probes on small samples overestimate.** 93.75% on 80 test pairs (200 examples) dropped to 84.5% on 400 test pairs (1000 examples). Always validate with 1000+ samples.
3. **Independent classification probes reveal representation geometry.** The 21.75% below-chance result was more informative than the 84.5% pairwise result — it revealed that preference is encoded relationally, not absolutely.
4. **Decouple feature extraction early.** Every architecture experiment after decoupling took minutes instead of hours. Should have done this before Run 1.
5. **Don't trust 100% accuracy.** When a model achieves perfect accuracy on both train and test, the first instinct should be suspicion, not celebration. Check for trivial solutions.
6. **Architecture search has diminishing returns when representations are the bottleneck.** Seven architectures hitting the same ceiling is strong evidence. Move to scaling data, not redesigning models.
7. **LR schedule matters as much as architecture.** Cosine decay over 3 epochs creates a dead zone in epoch 2 where the model coasts. Warmup→constant→late-decay keeps the model learning continuously. This was invisible until the same plateau appeared at two different dataset sizes.
8. **Batch size affects optimization quality, not just speed.** EBS=8 with SGD on a non-convex loss produces noisy gradients. The 20-point gap between the linear probe's L-BFGS (84.5%) and the same model's SGD (64%) is partly an optimization gap that larger EBS can close.
9. **Verify external suggestions empirically before trusting them.** The claim that compression was the bottleneck was wrong. The linear experiment proved the opposite. Multiple architecture suggestions were premature given the optimization gap.
10. **External storage extends capability significantly.** Mounting an external SSD for feature storage enabled 2x data scaling that wasn't possible on the primary drive. Simple partition + mount, no complex infrastructure needed.

---

## 5. Open Questions and Future Work

- **Optimization fix run (immediate):** Pairwise on 50k with EBS=64, warmup→constant→decay — test whether optimization improvements break the 70% ceiling
- **If ceiling breaks:** scale to full 161k dataset
- **If ceiling holds:** accept 70% as representational limit, begin paper writeup
- **V2 pointwise with same optimization improvements** — comparison run to isolate optimization effect
- **Flip test on all new pairwise checkpoints** — verify antisymmetry
- **Active inference integration:** Use constitutional score to gate generation in real time
- **Joint training:** Train evaluator alongside Ouro from scratch — representations shaped to support absolute scoring would fundamentally change the ceiling
- **Basal ganglia component:** Active gating mechanism integrating constitutional score and entropy into the early exit decision — next architectural milestone after CLT paper

---

## 6. Paper Framing Notes

- Frame as **architectural proposal** with proof-of-concept empirical validation
- Primary contribution: constitutional evaluator as separate architectural component operating on hidden states, not outputs
- Secondary contribution: **relational vs absolute preference encoding** — Ouro loop states encode preference relationally (84.5% pairwise) but not absolutely (21.75% independent, below chance). This is a novel finding about looped transformer representation geometry with implications for RLHF design.
- Tertiary contribution: trajectory scoring as real-time alignment monitoring
- Quaternary contribution: truncation artifact finding as methodological warning for HH-RLHF reward model work
- Motivate with neuroscience analogy (prefrontal/hippocampal/amygdala)
- **Ceiling interpretation update:** The 70% ceiling persists at 50k with cosine LR — partly an optimization artifact (dead epoch 2), not purely representational. Optimization fix run will clarify whether true ceiling is higher. HH-RLHF noise ceiling is estimated at 72-75% for the full dataset.
- The 70% ceiling across seven architectures is a property of the representations, not the evaluator design. This is the honest framing and makes the relational encoding finding more credible.
- Acknowledge limitation: current evaluator is passive observer, not active intervention
- Acknowledge limitation: trained on frozen representations — joint training is the full proposal and would change the representation geometry itself
- Report full architecture comparison table (section 4.15) as primary empirical result
- Cite: Universal Transformers, ACT (Graves 2016), Constitutional AI (Anthropic), Ouro (ByteDance), HH-RLHF dataset
- Target venue: arXiv first, potentially workshop track at NeurIPS or ICLR
