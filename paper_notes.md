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
During testing of `Ouro-1.4B` base model, three failure modes were documented:

1. **Repetition loop** — model gets stuck repeating the same sentence indefinitely without self-monitoring
2. **Instruction non-compliance** — model ignored explicit instruction to not agree mindlessly, then proceeded to agree mindlessly
3. **Identity hallucination** — when asked about its own architecture, model invented "Transformer16B" with fabricated specifications

These failure modes motivate the constitutional evaluator: a separate architectural component capable of monitoring reasoning trajectory and flagging misalignment before output is committed.

### 1.3 Early Exit Configuration
From `config.json`:
- `total_ut_steps: 4` — model loops exactly 4 times
- `early_exit_threshold: 1.0` — effectively disabled (requires 100% confidence to exit early)
- Gate values observed: `[-1.21, -0.70, 0.008, 0.007]` on a simple prompt

**Finding:** The early exit mechanism is architecturally present but operationally disabled by default. Gate values are trained signals being ignored at inference time. We set `early_exit_threshold = 0.87` to enable dynamic computation allocation.

**Observed effect:** Simple factual queries (e.g. "What is the capital of France?") complete in ~10 seconds. Complex multi-step reasoning (train distance problem) takes ~2.5 minutes. Early exit is functioning as intended.

---

## 2. Proposed Architecture — Constitutional Looped Transformer (CLT)

### 2.1 Three-Component Design
Inspired by neuroscientific analogy:

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

**Implication:** Final loop hidden state is the appropriate input to the constitutional evaluator. Taking all loop states redundantly would not add information for the same reason you don't re-read every page of a book to understand the last sentence.

**Revision:** This was partially revised. We implemented a **sliding window** over loop states (last 3 concatenated) to capture trajectory dynamics, which proved empirically useful.

---

## 3. Implementation

### 3.1 Environment
- Hardware: Lenovo Legion laptop, RTX 5070 Ti (12GB VRAM), Pop!_OS
- Python 3.12, PyTorch 2.12.0 nightly (cu128) — required for Blackwell sm_120 support
- Transformers 4.54.1 (pinned — 4.56.0+ breaks Ouro compatibility)
- Model: `ByteDance/Ouro-2.6B-Thinking`

### 3.2 Cache Bug and Fix
Ouro's custom `UniversalTransformerCache` class conflicts with newer transformers versions. The parent `Cache` class defines `key_cache` and `value_cache` as properties with no setter, but Ouro's `__init__` attempts direct assignment.

**Fix:** Added property accessors routing `self.key_cache` to `self._key_cache` in `modeling_ouro_patched.py`. This enables cache and reduces generation time significantly.

### 3.3 Hidden State Extraction
```python
captured = {}

def hook_fn(module, input, output):
    captured["hidden_states_list"] = [h.detach() for h in output[1]]

model.model.register_forward_hook(hook_fn)
```

Verified: 4 loop steps, each hidden state shape `[batch, seq_len, 2048]`.

### 3.4 Constitutional Evaluator Architecture
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

Sliding window padding uses **zero vectors** rather than repeating first state, encoding "missing context" rather than "same context."

### 3.5 Training
- Dataset: Anthropic HH-RLHF, 5,000 training examples
- Loss: Pairwise ranking loss `−log(σ(score_chosen − score_rejected))`
- Optimizer: AdamW, lr=1e-4
- Gradient clipping: 1.0
- Supervision: Trajectory supervision across all 4 loop steps simultaneously
- Ouro weights: fully frozen — only evaluator trained
- 3 epochs, batch size 4, max sequence length 512

**Training progression:**
| Epoch | Final Accuracy | Loss |
|---|---|---|
| 1 | ~61.9% | 2.607 |
| 2 | ~66.1% | 2.480 |
| 3 | ~68.7% | 2.369 |

---

## 4. Results

### 4.1 Proof of Concept — Alignment Signal in Frozen Representations
The evaluator learns above-chance alignment detection from **frozen** Ouro representations. Ouro was never explicitly trained to encode constitutional information — yet a lightweight probe extracts meaningful signal from its hidden states.

**Implication:** Constitutional alignment information exists latently in models trained on human preference data. A jointly trained constitutional evaluator integrated from pretraining would produce dramatically stronger signal.

### 4.2 Trajectory Analysis
Sample trajectories from test set evaluation:

**Example 3 (correct):**
```
Chosen:   [-2.445, -3.399, -1.401, -1.233]  ← dips then recovers
Rejected: [-4.046, -5.357, -3.757, -3.787]  ← dips and stays low
```

**Example 5 (correct):**
```
Chosen:   [0.835, 0.106, 1.157, 0.902]   ← fluctuates but positive
Rejected: [-0.58, -1.589, -0.613, -1.146] ← consistently negative
```

**Finding:** Chosen and rejected responses show qualitatively different trajectory shapes, not just different final scores. This suggests the evaluator is capturing something about the **dynamics** of reasoning, not just its endpoint.

### 4.3 Evaluator Generalization Across Data Subsets
During evaluation on the 8,552-example test split, accuracy degraded systematically as evaluation progressed:

| Progress | Accuracy | Avg Margin |
|---|---|---|
| 500/8552 | 65.2% | 0.768 |
| 1000/8552 | 66.5% | 0.916 |
| 2000/8552 | 69.0% | 0.906 |
| 3000/8552 | 60.5% | 0.561 |
| 4000/8552 | 54.1% | 0.303 |
| 5000/8552 | 51.1% | 0.173 |

**Interpretation:** HH-RLHF concatenates four subsets with different alignment dimensions (harmlessness vs. helpfulness variants). The evaluator trained on a 5,000-example sample learned subset-specific patterns that do not fully generalize to later subsets.

**Implications:**
1. Stratified sampling across subsets required for robust training
2. Constitutional alignment is not a single unified signal — it is a family of related signals requiring diverse supervision
3. A jointly trained CLT evaluator would develop more generalizable representations

---

## 5. Open Questions and Future Work

- **Stratified training:** Sample uniformly across HH-RLHF subsets
- **Larger training set:** 5,000 examples is conservative — 50,000+ would likely improve generalization
- **Active inference integration:** Use constitutional score to gate generation in real time
- **Joint training:** Train evaluator alongside Ouro from scratch rather than probing frozen representations
- **GRU/Transformer over loop states:** Replace sliding window with recurrent model for long-range trajectory dependencies
- **Calibration:** Verify entropy/gate signals are calibrated — model may be confidently wrong in exactly the domains where it's wrong
- **Comparison experiments:** Last-token vs mean pooling, final state vs concatenated states

---

## 6. Paper Framing Notes

- Frame as **architectural proposal** with proof-of-concept empirical validation
- Primary contribution: constitutional evaluator as separate architectural component operating on hidden states, not outputs
- Secondary contribution: trajectory scoring as real-time alignment monitoring
- Motivate with neuroscience analogy (prefrontal/hippocampal/amygdala)
- Acknowledge limitation: current evaluator is passive observer, not active intervention
- Cite: Universal Transformers, ACT (Graves 2016), Constitutional AI (Anthropic), Ouro (ByteDance), HH-RLHF dataset
- Target venue: arXiv first, potentially workshop track at NeurIPS or ICLR
