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

---

## 5. Open Questions and Future Work

- **L2 fix (immediate):** Retrain with L2=0.001, expect accuracy improvement
- **Progressive scaling:** 15k → 50k → 100k → 160k full dataset
- **Learning rate scheduler:** Add cosine decay for larger dataset runs
- **Active inference integration:** Use constitutional score to gate generation in real time
- **Joint training:** Train evaluator alongside Ouro from scratch
- **GRU/Transformer over loop states:** Replace sliding window with recurrent model
- **Two-head evaluator:** Separate heads for final state and trajectory dynamics
- **Larger training set:** 160k full HH-RLHF training split

---

## 6. Paper Framing Notes

- Frame as **architectural proposal** with proof-of-concept empirical validation
- Primary contribution: constitutional evaluator as separate architectural component operating on hidden states, not outputs
- Secondary contribution: trajectory scoring as real-time alignment monitoring
- Tertiary contribution: truncation artifact finding as methodological warning for HH-RLHF reward model work
- Motivate with neuroscience analogy (prefrontal/hippocampal/amygdala)
- Acknowledge limitation: current evaluator is passive observer, not active intervention
- Acknowledge limitation: trained on frozen representations — joint training is the full proposal
- Report run 2 evaluation accuracy (~62%) as primary result, note L2 fix expected to improve this
- Cite: Universal Transformers, ACT (Graves 2016), Constitutional AI (Anthropic), Ouro (ByteDance), HH-RLHF dataset
- Target venue: arXiv first, potentially workshop track at NeurIPS or ICLR
