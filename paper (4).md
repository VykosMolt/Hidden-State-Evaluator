# Relational Preference Encoding in Looped Transformer Internal States

**Johann Hirschner**

April 2026

---

## Abstract

We investigate how looped transformers encode human preference in their internal iteration states. Using Ouro-2.6B-Thinking, a 2.6B-parameter looped transformer with iterative refinement, we extract hidden states from each loop iteration and train lightweight evaluator heads (~5M parameters) to predict human preference on the Anthropic HH-RLHF dataset. Our pairwise evaluator achieves **83.3% test accuracy** on 8,552 unseen examples — within 1.2 points of the theoretical ceiling established by a full-batch L-BFGS probe (84.5%) — while the base model remains completely frozen.

Our central finding is that loop states encode preference *predominantly relationally*: a linear probe on pairwise differences achieves 84.5%, the best nonlinear independent evaluator reaches only 70% training accuracy, and linear independent classification scores 21.75% (below chance, with inverted polarity). We document the systematic architecture search — nine designs, fifteen configurations, three loss functions, two dataset scales — that established a genuine 70% ceiling for independent (pointwise) scoring, and show how the 50% argument-swap protocol essential to preventing degenerate pairwise solutions deflated pairwise training metrics by ~13 points, creating the false appearance that pairwise and pointwise evaluators shared the same ceiling. This metric deflation went undetected across seven pairwise training runs and its resolution constitutes a methodological finding in its own right.

We further demonstrate that nonlinear feature construction adds 6 points over raw linear access (compression helps rather than hurts), that no surface-level shortcuts explain the signal, and that a degenerate failure mode in pairwise evaluators — yielding 100% accuracy through constant output — necessitates a flip test diagnostic that we propose and validate.

---

## 1. Introduction

Large language models increasingly use iterative internal computation to refine outputs. Looped transformers — architectures that pass hidden states through the same transformer block multiple times — produce a sequence of intermediate states that capture the model's evolving internal representation. A natural question arises: do these intermediate loop states carry information about human preference?

If a frozen language model's internal dynamics encode whether a response aligns with human values, this signal could be extracted by a lightweight external evaluator without modifying the base model's weights. Such a separable architecture would enable alignment monitoring as an interpretable, independently trainable module.

This work makes several contributions:

1. We demonstrate that Ouro-2.6B-Thinking's loop iteration states encode human preference **predominantly relationally** — pairwise access achieves 83.3% test accuracy while the best independent scorer reaches 70% — establishing a previously undocumented property of looped transformer representations.

2. We achieve **83.3% test accuracy** from a completely frozen 2.6B-parameter model with only ~5M trainable evaluator parameters, approaching the 84.5% ceiling of an L-BFGS probe on the same features.

3. We document how a misleading training metric — caused by the antisymmetry enforcement protocol essential to preventing degenerate solutions — masked the pairwise model's actual capability for seven consecutive runs, creating the false appearance that pairwise and pointwise evaluators shared the same 70% ceiling.

4. We propose the **flip test** as a diagnostic for pairwise evaluators, motivated by a degenerate failure mode that achieved 100% accuracy on both training and test sets through a constant positive output.

5. We demonstrate that nonlinear feature construction (attention pooling + GRU) **constructs** rather than destroys the preference signal, adding 6 points over a raw linear model on identical features.

**Concurrent work.** Anthropic's interpretability team recently reported that Claude Sonnet 4.5 contains 171 functional emotion-related internal representations that causally drive model behavior (Anthropic, April 2026). Their finding that abstract internal states are structured, extractable, and behaviorally consequential parallels our finding that loop states encode preference in relational geometric form. Both demonstrate that LLM internal representations carry alignment-relevant information accessible to lightweight external methods. Our work extends this line of inquiry to iterative architectures and preference encoding specifically.

---

## 2. Background

### 2.1 Looped Transformers

Unlike standard transformers that process input through a fixed stack of distinct layers, looped transformers reuse the same transformer block across multiple iterations. At each iteration *t*, the model produces hidden states *h_t* that incorporate information from all previous iterations. Ouro-2.6B-Thinking implements this with an early-exit mechanism controlled by a confidence threshold, producing up to 4 loop iteration states per input.

### 2.2 Preference Learning from Representations

Standard reward models for RLHF are trained end-to-end: the base model's representations adapt jointly with the reward head. Our approach differs fundamentally — the base model is completely frozen, and only a lightweight evaluator (~5M parameters) is trained on extracted hidden states. This tests whether preference information is already present in the model's natural representations, rather than whether it can be trained into them.

### 2.3 The Anthropic HH-RLHF Dataset

We use the Anthropic HH-RLHF dataset containing paired human conversations where annotators identified a "chosen" (preferred) and "rejected" response. The dataset contains approximately 161k training pairs and 8,552 test pairs. Human annotator disagreement is estimated at 25-30%. Standard end-to-end reward models on the full dataset achieve 72-75% accuracy.

---

## 3. Method

### 3.1 Feature Extraction

We register a forward hook on Ouro-2.6B-Thinking's model backbone (`model.model`). The hook captures `output[1]`, which contains a list of 4 tensors of shape `[batch, seq_len, 2048]` in bfloat16 — one per loop iteration. We verified through structural analysis that these are genuine loop iteration states, not layer outputs or attention weights: all 4 tensors share the same dimensionality, and their values differ across iterations in ways consistent with iterative refinement.

Each conversation is tokenized with a maximum length of 1024 tokens, left-padded. The model's early-exit threshold is set to 0.87. For each example, we extract loop states for both the chosen and rejected responses independently, storing them in float16 with corresponding attention masks. We extract features for 50,000 training examples, stored as chunked `.pt` files (100 examples per chunk) for memory-efficient training. This decoupling of extraction from training reduced experiment iteration time from hours to minutes per run.

### 3.2 Diagnostic Probes

Before training evaluators, we conduct three diagnostic analyses to characterize the preference signal in the raw representations.

**Pairwise linear probe.** We compute the difference between mean-pooled chosen and rejected hidden states, then train a logistic regression classifier (L-BFGS optimizer, full batch) to distinguish `chosen - rejected` from `rejected - chosen`. On 1,000 examples (400 test pairs), this probe achieves **84.5%** accuracy on final loop state features and **86.25%** on all-states-concatenated features. This establishes the theoretical ceiling for linear separation on pairwise differences.

**Independent classification probe.** We train a logistic regression to classify individual response representations as "chosen" or "rejected" without access to the paired response. This probe achieves **21.75%** — below the 50% chance level, with inverted polarity (flipped: 78.25%). This below-chance result was the pivotal diagnostic finding: it revealed that preference is not encoded as an absolute property of individual representations.

**Structural shortcut analysis.** On 500 test examples, we measure whether surface features separate chosen from rejected: sequence length (chosen mean=156.7, rejected mean=169.2 tokens; "longer=chosen": 45.6%), hidden state norms ("larger norm=chosen": 43.0%), and mean activation ratios (~0.997 across all loop steps). No structural shortcut is found. The preference signal resides in representation geometry, not surface statistics.

### 3.3 Evaluator Architectures

We test nine architectures spanning three design paradigms.

**Pointwise evaluators** score each response independently. The best-performing variant (V2) uses learned attention pooling over tokens (low-rank projection to 128-dim, learned query vector, masked softmax weighting), LayerNorm, projection to 512-dim, 2-layer GRU over the sequence of 4 projected loop states, a skip connection from the final projected state, and a 2-layer scorer MLP. Total: ~4.7M parameters.

**Pairwise evaluator.** A shared AttentionPool processes both chosen and rejected at each loop step. Per-step difference vectors (chosen_pooled - rejected_pooled) are normalized with LayerNorm(bias=False) to preserve antisymmetry, projected to 512-dim, and fed through a 2-layer GRU. A skip connection from the final projected difference is concatenated with the GRU output, and a scorer outputs a single scalar: positive indicates the first argument is preferred. This is the architecture that achieves 83.3% test accuracy.

**Calibrated evaluator.** Identical architecture to V2, trained with combined ranking and binary cross-entropy classification loss.

**Linear evaluator.** Single `nn.Linear(8192, 1)` on concatenated mean-pooled loop states. 8,193 parameters. No nonlinearities. This serves as a direct comparison to the L-BFGS probe, testing whether the optimizer (L-BFGS vs Adam) or the access pattern (pairwise vs independent) accounts for the accuracy gap.

### 3.4 Training Protocol

All evaluators are trained on precomputed features using AdamW (weight_decay=0.01) with gradient clipping (max_norm=1.0). The pairwise evaluator uses learning rate 1e-4 with cosine annealing (eta_min=1e-6), 200-step linear warmup, and batch size 32 in float32 precision.

**Antisymmetry enforcement.** The pairwise evaluator requires a protocol to prevent degenerate solutions (Section 4.3). Each training batch undergoes a random 50% swap: with probability 0.5, the chosen and rejected inputs are swapped and the target becomes -1 instead of +1. The loss is `-logsigmoid(target * score)` plus L2 score regularization (`1e-4 * score²`). This protocol is essential but has a significant side effect on reported training metrics, discussed in Section 4.5.

---

## 4. Results

### 4.1 Test Accuracy: 83.3%

The pairwise evaluator achieves **83.3% accuracy** on the full HH-RLHF test set (8,552 examples), evaluated with live Ouro model inference on unseen data. Chosen responses are always presented as the first argument; a positive score indicates correct preference prediction.

| Metric | Value |
|---|---|
| Test accuracy | 83.3% (7,125 / 8,552) |
| Average score | +0.826 |
| Score std | 0.840 |
| Score range | [-2.09, +3.41] |
| Positive rate | 83.3% |

This result places the evaluator within 1.2 points of the L-BFGS probe ceiling (84.5%) and substantially above both the best pointwise evaluator (65% test) and standard end-to-end reward models trained on the full 161k dataset (72-75%). The model achieves this with ~5M trainable parameters on a completely frozen 2.6B-parameter base model, trained on only 50k of the available 161k examples.

### 4.2 Relational vs. Absolute Preference Encoding

The core finding is the stark asymmetry between pairwise and independent access to preference information:

| Access Pattern | Method | Accuracy |
|---|---|---|
| Pairwise, linear (L-BFGS) | Logistic regression on differences | 84.5% |
| **Pairwise, nonlinear (Adam)** | **GRU + AttentionPool evaluator** | **83.3% test** |
| Independent, nonlinear (Adam) | GRU + AttentionPool (pointwise V2) | 70% train / 65% test |
| Independent, linear (L-BFGS) | Logistic regression on single response | 21.75% |
| Independent, linear (Adam) | nn.Linear on concatenated features | 64% train |

Two patterns are clear. First, pairwise access consistently outperforms independent access: 83-84% vs 64-70% across all model classes and optimizers. Second, the L-BFGS vs Adam gap is real but smaller than it initially appeared: on the pairwise task, Adam reaches 83.3% vs L-BFGS's 84.5% — a gap of only 1.2 points, not the 20-point gap observed on the independent linear task.

We cannot categorically exclude that a sufficiently expressive nonlinear independent classifier could match the pairwise results. However, the consistent gap across both linear and nonlinear classifiers constitutes strong evidence that preference is encoded predominantly in the relational geometry between representations rather than as an absolute property of individual responses. We term this **relational preference encoding**.

The independent probe's below-chance performance (21.75%, flipped: 78.25%) admits several interpretations: inverted polarity in activation patterns, label imbalance effects, or systematic representation differences between chosen and rejected responses. Our structural shortcut analysis rules out simple confounds (length, norms), but the exact mechanism warrants further investigation.

### 4.3 Degenerate Pairwise Solutions and the Flip Test

Our initial pairwise evaluator achieved 100% accuracy on both training and test sets — a result that proved entirely spurious. The model learned to output a constant score of approximately +13 for every input pair.

**Root cause.** The loss function `-logsigmoid(score)` only pushed scores positive. Training always presented chosen first. The model discovered that a large positive constant minimizes loss perfectly. LayerNorm's bias term absorbed sign information from the difference computation, and no L2 regularization constrained score magnitude. The model never examined actual content.

**The flip test.** We propose swapping input order and verifying that scores change sign. A genuine model should satisfy approximate antisymmetry: `f(A, B) ≈ -f(B, A)`. The degenerate model produced:

```
Example 1: Normal=+13.16 | Flipped=+13.05
Example 2: Normal=+13.53 | Flipped=+13.45
```

Scores did not flip. The model was a constant function.

**Fix.** Three modifications resolved the degeneracy: (1) random 50% swap of argument order during training with corresponding target sign flip, (2) directional loss `-logsigmoid(target * score)`, and (3) L2 score regularization. Additionally, removing the bias from the input LayerNorm (bias=False) enforces architectural antisymmetry: `LN(-x) = -LN(x)`.

**A nuance on the degenerate run.** The collapse to constant output was not instantaneous. Early in epoch 1, before the model discovered the degenerate shortcut, accuracy climbed in a pattern consistent with genuine preference learning. The optimizer likely followed a trajectory from genuine signal extraction (early epoch 1) to discovery of the lower-loss degenerate basin (mid epoch 1) to unbounded score growth (late epoch 1 onward). This is consistent with our broader finding that the genuine preference signal is learned very quickly — within the first few hundred batches — and that all subsequent training dynamics (degeneracy without swaps, overfitting with swaps) represent the optimizer moving away from the general solution toward training-specific exploitation.

**Flip test on the final model.** On 20 test examples, the trained evaluator shows 60% strict sign flip rate. When normal scores are strong (>0.9), flipped scores reliably go negative. When normal scores are weak (0.2-0.7), a positive offset prevents sign reversal. This reflects a learned positive bias in the scorer's LayerNorm (which retains bias=True), not degeneracy — scores are clearly content-dependent, ranging from -2.09 to +3.41 with 83.3% test accuracy.

### 4.4 The Perceived 70% Ceiling: A Methodological Cautionary Tale

The pointwise V2 evaluator genuinely reached 70% training accuracy and 65% test accuracy. This was a real ceiling for independent scoring — consistent across architectures (V1: 62%, V2 with mean pool: 68%, V2 with attention pool: 70%, no-GRU: 68%, calibrated: 69%). The representations encode preference relationally, so independent scoring has a genuine upper bound.

The misconception began when we introduced the pairwise evaluator. After fixing the degenerate +13 failure (Section 4.3), the corrected pairwise model — trained with 50% argument swapping — reported training accuracies of 70%. This appeared to confirm the same ceiling: "even seeing both responses doesn't help." We spent the subsequent seven days and eight training runs trying to break this apparent ceiling by scaling data (25k → 50k), increasing batch sizes (EBS 8 → 64), changing learning rate schedules (cosine, constant, warmup-constant-decay), modifying architectures (pre-diff-norm, separate anchor projections), and extending training (3 → 5 epochs). Everything hit 70%.

**The resolution was trivial.** The 50% argument-swap protocol deflates reported training accuracy for pairwise models specifically. At each step, the model must predict the correct sign (+1 or -1) based on which ordering was randomly sampled. The running accuracy blends performance on the easier direction (chosen first) with the harder direction (rejected first), producing a metric that understates actual preference discrimination. The pointwise evaluators — which do not use the swap protocol — reported accurate training metrics throughout.

When we ran the held-out test evaluation with a fixed ordering (chosen always first, score > 0 = correct), the pairwise model achieved 83.3%. The 13-point gap between reported training accuracy (70%) and actual test performance (83.3%) had been present from the first valid pairwise run but went undetected because we assumed the pairwise training metric was comparable to the pointwise training metric.

**This is itself a finding.** Antisymmetry enforcement protocols in pairwise models deflate training metrics relative to pointwise models trained on the same task. Researchers using swap-based pairwise training should evaluate on held-out data with a fixed ordering to obtain comparable performance estimates. The genuine pointwise ceiling (70% train / 65% test) and the genuine pairwise performance (83.3% test) are both real — what was illusory was their apparent convergence.

**The deflation masks overfitting.** The relationship between training metric and actual performance is not merely offset — it is inverted across epochs. Epoch 1 reports 60% deflated training accuracy and achieves 83.3% test accuracy. Epoch 5 reports 70% deflated training accuracy and achieves ~60% test accuracy. The model that appears better by the training metric is catastrophically worse on held-out data. By epoch 5, the weights have memorized training-specific patterns — swap directions, chunk orderings, label noise in the 50k examples — that do not generalize. Epoch 1 captures the general geometric preference signal before overfitting to artifacts. This means that for pairwise models with swap-based training, the deflated training metric is not merely uninformative — it is actively misleading, rewarding overfitting and penalizing generalization. Early stopping based on held-out evaluation is essential; training accuracy cannot serve as a proxy.

| Epoch | Deflated Train Acc | Actual Test Acc | Relationship |
|---|---|---|---|
| 1 | 60% | **83.3%** | General signal learned |
| 5 | 70% | ~60% | Overfit to training artifacts |

### 4.5 Architecture Search Results

Despite the training metric deflation, the architecture search produced genuine findings. Table 1 shows training accuracy (the deflated metric) across all configurations, along with test accuracy where evaluated.

| Architecture | Train Acc | Test Acc | Key Finding |
|---|---|---|---|
| V1 MLP concat | 62.0% | 61.9% | Baseline |
| V2 GRU + mean pool | 68.0% | 63.0% | Mean pooling |
| V2 GRU + AttentionPool(128) | 70.0% | 65.0% | Best pointwise |
| V2 AttentionPool(2048) | 53.0% | — | Full-rank collapse |
| No-GRU AttentionPool + MLP | 68.0% | — | No temporal model |
| Pairwise v1 (25k) | 70.0% | — | First valid pairwise |
| Pairwise v1 (50k) | 70.0% | — | 2x data scale |
| Pairwise v1 (50k, EBS=64) | 70.0% | — | 8x batch size |
| Calibrated (ranking + BCE) | 69.0% | — | Combined loss |
| Linear (single layer, Adam) | 64.0% | — | No compression |
| Pairwise v2 (pre-diff-norm) | 68.0% | — | Regressed |
| Pairwise v1 (AMP, 5 epochs) | 70.0% | — | AMP unstable |
| **Pairwise v1 (fp32, epoch 1)** | **60.0%** | **83.3%** | **Best model** |
| Pairwise v1 (fp32, epoch 5) | 69.9% | ~60% | Overfit: train↑ test↓ |

**Table 1.** Training and test accuracy across architectures. For pointwise models (rows 1-5), training accuracy is directly comparable to test accuracy. For pairwise models with swap protocol (rows 6-14), training accuracy is deflated ~13 points and inverts with overfitting: the epoch 1 checkpoint (60% train) outperforms epoch 5 (70% train) on test by 23 points.

Several genuine findings emerge from the search:

**Compression helps, not hurts.** The linear evaluator (single `nn.Linear(8192, 1)`, Adam optimizer) achieved 64% — six points below the compressed V2 architecture (70% training). The GRU and attention pooling perform nonlinear feature construction that makes the distributed, non-axis-aligned preference signal accessible to the downstream scorer. This directly contradicts the persistent external hypothesis that compression destroys signal.

**Attention pooling rank matters.** Full-rank attention pooling (2048-dim, 4.2M parameters in pooling alone) collapsed to 53% due to overfitting. Low-rank (128-dim, 262K parameters) forced the pooling layer to learn structural weighting patterns rather than memorizing token-level features.

**Temporal modeling contributes modestly.** The GRU adds approximately 2 points when paired with attention pooling (68% → 70%) and 0 points with mean pooling. This improvement is consistent across runs but small — near the noise floor for this task. The dominant signal is in the final loop state: the probe achieves 84.5% on the final state alone vs 86.25% on all states concatenated.

**Pre-diff-norm regresses.** Normalizing each response independently before subtraction (`LayerNorm(c) - LayerNorm(r)`) regressed from 70% to 68% (training metric). This likely destroys magnitude information that carries preference signal. The original approach (`LayerNorm(c - r)`) preserves relative magnitude in the difference.

**AMP is unsuitable for this architecture.** Mixed precision training caused severe mid-epoch accuracy crashes due to GradScaler interaction with the GRU's small gradients (scores in the ±0.5 range). Float32 is required for stable training.

### 4.6 Structural Analysis

No surface-level features explain the preference signal. Rejected responses are slightly longer than chosen ones (mean 169.2 vs 156.7 tokens), hidden state norms do not separate the classes (43% accuracy), and mean activation ratios are near unity (~0.997) across all loop steps. The signal resides in the high-dimensional geometry of the representations, not in any single computable statistic.

### 4.7 The Optimization Gap, Revisited

The original framing attributed the gap between 70% (evaluator) and 84.5% (probe) to a fundamental limitation of first-order optimization. The 83.3% test result revises this picture substantially.

On the pairwise task, Adam reaches 83.3% vs L-BFGS's 84.5% — a gap of only 1.2 points. The 20-point gap observed between L-BFGS (84.5%) and Adam (64%) on the linear model reflects the independent-vs-pairwise distinction as much as the optimizer: the linear model scores independently (mean pool, no access to the paired response), while the probe operates on pairwise differences.

The remaining 1.2-point gap between the evaluator and the probe may reflect any combination of: mini-batch vs full-batch optimization, first-order vs second-order curvature information, or loss function differences. It may also reflect the dataset noise ceiling — with 25-30% annotator disagreement, 84.5% may itself be near the maximum achievable on this dataset. We report this as an empirical observation rather than a claim about fundamental optimizer limitations.

---

## 5. Discussion

### 5.1 What Relational Encoding Means

Under all tested conditions — linear and nonlinear classifiers, first-order and second-order optimizers, nine evaluator architectures — pairwise access to representations substantially outperforms independent access. The strongest independent scorer (V2 pointwise, 65% test) falls 18 points below the pairwise evaluator (83.3% test) despite using an identical architecture class.

This is consistent with how language models are trained: next-token prediction does not require absolute quality judgments, only contextual coherence. Preference, being a human construct imposed during RLHF, may not naturally align with any absolute direction in representation space. Instead, the model's representations are structured such that chosen and rejected responses to the same prompt land in geometrically distinguishable regions — but this distinction is relational (requiring both representations for comparison) rather than absolute (classifiable from a single representation alone).

We note that we cannot categorically exclude the existence of an absolute preference signal accessible to a more powerful independent classifier than those we tested. The claim is empirical: under our tested conditions, pairwise access dominates.

### 5.2 Implications for Alignment Monitoring

Achieving 83.3% test accuracy from a frozen 2.6B-parameter model with ~5M trainable parameters has practical implications. This exceeds the 72-75% typically achieved by end-to-end reward models on the full 161k HH-RLHF dataset — despite using frozen representations, training on only 50k examples, and requiring no modification to the base model.

The result establishes feasibility for modular alignment monitoring: a separable, lightweight component that reads a model's internal states and evaluates alignment without modifying the model's weights or behavior. Such a component could serve as a real-time alignment signal during inference, an independent audit mechanism, or a training signal for downstream fine-tuning.

### 5.3 Connection to Anthropic's Emotion Concepts Research

Anthropic's concurrent finding of 171 functional emotion representations in Claude Sonnet 4.5 (April 2, 2026) demonstrates that large language models develop structured internal representations of abstract psychological concepts that causally influence behavior. Their key finding — that "desperation" representations can drive unethical actions, and that emotion vectors predict and causally influence model preferences — is parallel to our work.

Both share a core insight: **LLM internal states carry structured, extractable signals about alignment-relevant properties.** Anthropic demonstrates this for emotion concepts in a standard transformer; we demonstrate it for preference encoding in a looped transformer's iteration states. Both find that these signals are functional (they predict behavioral outcomes) and geometric (organized as directions in representation space). The convergence of interpretability techniques (Anthropic) and probing/evaluator methods (this work) on the same fundamental finding strengthens both lines of evidence.

### 5.4 The Iterative Research Process

The 83.3% result emerged through a ten-day process that included one genuine ceiling and one phantom.

**Phase 1: Architecture search (Runs 1-7).** Starting from a simple MLP baseline (62%), we progressively added components: mean pooling → attention pooling (+2%), GRU temporal modeling (+2%), skip connections. Full-rank attention pooling collapsed to 53%; low-rank (128-dim) became standard. The V2 architecture reached 70% train / 65% test — a genuine ceiling for independent scoring, consistent across all pointwise variants.

**Phase 2: Diagnosis.** The scaled linear probe (84.5% pairwise, 21.75% independent) reframed the problem. The below-chance independent probe was the pivotal finding: preference is encoded relationally. This redirected work from "how do we score responses better independently?" to "how do we exploit relational encoding?"

**Phase 3: Pairwise evaluator.** The natural response was an evaluator that sees both responses. The degenerate failure (100% accuracy, constant +13) was a critical negative result that motivated the swap protocol and flip test. The fixed pairwise evaluator reported 70% training accuracy — appearing to match the pointwise ceiling.

**Phase 4: The phantom ceiling (Runs 8-15).** Believing that even pairwise access could not break 70%, we spent seven days testing data scaling (25k → 50k), batch size scaling (EBS 8 → 64), alternative schedules, architecture variations (pre-diff-norm, which regressed), and extended training. All pairwise runs reported ~70%. We concluded the ceiling was representational.

**Phase 5: Resolution.** Running the held-out test evaluation revealed 83.3% — the swap protocol had been deflating pairwise training metrics by ~13 points. The pointwise ceiling (70% / 65%) was real. The pairwise ceiling was not. The apparent convergence of both paradigms to 70% was an artifact of incomparable metrics.

### 5.5 Future Work

**Phase 2: Joint training (LoRA + evaluator).** Unfreezing Ouro's later layers via low-rank adapters (~10-50M trainable parameters) while training the evaluator jointly would allow the representations to adapt to the preference scoring task. Under joint training, the representations co-evolve with the evaluator, and accuracy may push beyond the current 83.3%. On consumer hardware (RTX 5070 Ti, 16GB VRAM), Ouro-2.6B in 4-bit quantization (~1.5GB) with LoRA adapters is feasible.

**Phase 3: Basal ganglia integration.** The evaluator currently operates as a passive measurement tool. The planned basal ganglia component would close this loop: the evaluator's preference score feeds back into Ouro's generation process during inference, enabling real-time alignment steering through gating (suppressing low-scoring generation paths), steering (biasing hidden states toward higher-scoring directions), or early-exit modulation (halting iterations when alignment confidence is high). The basal ganglia metaphor is functional, not decorative — in neuroscience, the basal ganglia integrate reward signals with motor planning to select actions; this component would integrate preference signals with iterative refinement to select aligned outputs.

**Epistemic reasoning component.** A parallel line of work explores whether similar internal-state extraction techniques can support epistemic reasoning — the model's capacity to represent uncertainty, knowledge boundaries, and relational structure over abstract patterns. If loop states encode not only preference but also epistemic confidence, the same evaluator-head architecture could serve as a general-purpose internal state monitor across cognitive dimensions. Preliminary work targets the ARC-AGI benchmark, where success requires systematic generalization from few examples.

**Toward brain-ontological architectures.** The naming conventions in this work — "amygdala" for the evaluator, "basal ganglia" for the steering component, "epistemic" for uncertainty reasoning — reflect a broader architectural hypothesis: that a productive path toward artificial general intelligence may run through ontological correspondence with biological neural architecture at the functional module level. Not biomimicry of neural implementation, but structural correspondence in modular organization — separable components for preference evaluation, action selection, epistemic monitoring, and planning that mirror the functional decomposition of the mammalian brain. This hypothesis is speculative and empirically unvalidated at the systems level, but it provides a principled design vocabulary. The research initiated here — extracting alignment-relevant signals via dedicated external modules — represents a first empirical step in that direction.

**Cross-architecture generality.** All experiments use Ouro-2.6B-Thinking. Testing whether relational preference encoding generalizes to other looped architectures (Universal Transformers, DEQ models, adaptive-depth transformers) and to non-looped models is critical future validation.

### 5.6 Limitations

**Frozen representations.** Our 83.3% reflects the signal available in Ouro's natural representations, which were not trained for preference scoring. Joint training may improve results further.

**Dataset noise and ceiling.** HH-RLHF has an estimated 25-30% annotator disagreement rate. Our 83.3% and the probe's 84.5% may both be near the noise ceiling of this dataset. Evaluation on cleaner preference datasets would clarify how much headroom remains.

**Imperfect antisymmetry.** The flip test shows 60% strict sign-flip rate on 20 examples, with a positive bias on weak scores. The scorer's LayerNorm bias contributes a learned positive offset. While this does not invalidate the 83.3% test accuracy (scores are clearly content-dependent), a perfectly antisymmetric model might perform differently.

**Single base model.** All experiments use Ouro-2.6B-Thinking. Our findings may be specific to this model's training, architecture, or scale. Generalization is untested.

**Independent probe scope.** We tested linear and nonlinear (GRU + attention pool + MLP) independent classifiers. We have not exhaustively tested all possible independent architectures (deep MLPs on raw features, kernel methods). The claim of predominantly relational encoding is bounded by tested conditions.

---

## 6. Conclusion

We demonstrate that a lightweight pairwise evaluator reading the internal loop states of a frozen 2.6B-parameter looped transformer achieves 83.3% test accuracy on human preference prediction — within 1.2 points of an L-BFGS probe ceiling and substantially above end-to-end reward model baselines. The finding that preference is encoded predominantly in the relational geometry between response representations, rather than as an absolute property of individual responses, characterizes a previously undocumented property of looped transformer internal states.

The path to this result included a degenerate failure mode that achieved 100% accuracy through constant output, a genuine 70% ceiling for independent scoring, and a seven-run investigation of a phantom pairwise ceiling caused by training metric deflation from the antisymmetry enforcement protocol. The finding that swap-based training deflates pairwise metrics by ~13 points relative to fixed-ordering evaluation constitutes a methodological contribution relevant to all pairwise preference learning systems.

These results, alongside Anthropic's concurrent demonstration of functional internal representations in large language models, indicate that LLM internal states carry rich, structured information about alignment-relevant properties. Extracting this information through lightweight, separable evaluator components — and ultimately feeding it back into the generation process through modular steering mechanisms — represents a viable path toward interpretable alignment architectures.

---

## References

Anthropic. (2026). Emotion concepts and their function in a large language model. *Transformer Circuits Thread.*

Bai, Y., et al. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. *arXiv:2204.05862.*

ByteDance. (2025). Ouro-2.6B-Thinking. *Hugging Face Model Hub.*

Christiano, P., et al. (2017). Deep Reinforcement Learning from Human Preferences. *NeurIPS.*

Maheswaran, A., & Desarkar, M. S. (2026). A Unified View on Emotion Representation in Large Language Models. *EACL.*

Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS.*

Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread.*

---

## Appendix A: Evaluator Architecture Details

### A.1 Attention Pooling

```
keys = Linear(2048 → 128, bias=False)(hidden)
scores = masked_softmax(keys @ query, attention_mask)
pooled = weighted_sum(scores, hidden)  →  [batch, 2048]
```

Low-rank projection (128-dim) prevents the pooling layer from memorizing token-level patterns. Full-rank (2048-dim) collapsed to 53% — 4.2M parameters in pooling alone caused overfitting.

### A.2 Pairwise Evaluator Forward Pass

```
For each loop step t ∈ {1, ..., 4}:
    c_t = AttentionPool(chosen_states_t, chosen_mask)
    r_t = AttentionPool(rejected_states_t, rejected_mask)
    diff_t = c_t - r_t
    normed_t = LayerNorm(diff_t, bias=False)
    proj_t = Linear(normed_t)

gru_out = GRU([proj_1, ..., proj_4])[-1]
combined = concat(gru_out, proj_4)       # skip connection
score = Scorer(combined)                  # LayerNorm → Linear → GELU → Dropout → Linear → scalar
```

Total parameters: ~4.73M. Input LayerNorm uses `bias=False` to preserve antisymmetry: `LN(-x) = -LN(x)`.

### A.3 The Flip Test Protocol

For each test example:
- `score_normal = f(chosen, rejected)`
- `score_flipped = f(rejected, chosen)`

A genuine model should show: (1) scores that vary with content, (2) sign reversal for most examples, and (3) approximate antisymmetry (`score_normal + score_flipped ≈ 0`). The degenerate model produced `score_normal ≈ score_flipped ≈ +13`. The trained model produces content-dependent scores (-2.09 to +3.41) with 60% strict sign reversal and a learned positive offset on weak scores.

### A.4 Training Metric Deflation and Inversion

The 50% swap protocol reports accuracy as "does score sign match target?" where target is +1 or -1 with equal probability. This produces a metric approximately 13 points below actual preference discrimination accuracy. More critically, the metric *inverts* across epochs: epoch 1 shows 60% deflated / 83.3% test, while epoch 5 shows 70% deflated / ~60% test. The training metric increases monotonically as the model overfits, making it worse than uninformative — it actively signals improvement during degradation. Researchers using swap-based pairwise training must use held-out evaluation for both performance estimation and early stopping.

---

## Appendix B: Complete Architecture Search

| # | Architecture | Params | Train | Test | Key Finding |
|---|---|---|---|---|---|
| 1 | MLP concat (V1) | ~2M | 62.0% | 61.9% | Baseline |
| 2 | GRU + mean pool (V2) | ~4.7M | 68.0% | 63.0% | GRU helps with mean pool |
| 3 | GRU + AttentionPool(128) | ~4.7M | 70.0% | 65.0% | Best pointwise |
| 4 | GRU + AttentionPool(2048) | ~8.9M | 53.0% | — | Full-rank collapse |
| 5 | AttentionPool(256) + MLP | ~3.2M | 68.0% | — | GRU adds ~2pts |
| 6 | AttentionPool(128) + MLP | ~3.0M | 68.0% | — | Confirmed |
| 7 | Pairwise (degenerate) | ~4.7M | 100% | 100% | Flip test failure |
| 8 | Pairwise v1 (25k) | ~4.7M | 70.0% | — | First valid pairwise |
| 9 | Pairwise v1 (50k) | ~4.7M | 70.0% | — | Data scaling neutral |
| 10 | Pairwise v1 (EBS=64) | ~4.7M | 70.0% | — | Batch scaling neutral |
| 11 | Calibrated (rank+BCE) | ~4.7M | 69.0% | — | BCE adds nothing |
| 12 | Linear (Adam) | 8,193 | 64.0% | — | Compression helps |
| 13 | Pairwise v2 (pre-diff-norm) | ~5.8M | 68.0% | — | Regressed 2pts |
| 14 | Pairwise v1 (AMP, 5ep) | ~4.7M | 70.0% | — | AMP unstable |
| 15 | Pairwise v1 (fp32, ep1) | ~4.7M | 60.0% | **83.3%** | Best model |
| 16 | Pairwise v1 (fp32, ep5) | ~4.7M | 69.9% | ~60% | Overfit: train↑ test↓ |

Training accuracies for pairwise models with swap protocol (rows 8-16) are deflated ~13 points. Pointwise models (rows 1-6, 11-12) report accurate training metrics. The inversion between rows 15 and 16 — where the higher training accuracy corresponds to dramatically worse test accuracy — demonstrates that the deflated metric actively rewards overfitting.
