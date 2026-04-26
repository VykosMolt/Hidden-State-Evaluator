# Ouro Loop-State Pairwise Evaluator

This repository contains the evaluator code from my work on **Ouro-2.6B-Thinking** loop states.

The project began as an alignment / preference-readout experiment. I wanted to test whether human preference could be read from the hidden states produced by Ouro's internal loop iterations, without fine-tuning the base model. The surprising result was not just that preference was readable, but that it was readable mainly **relationally**: comparing two loop-state trajectories worked far better than trying to score each response independently.

## Main result

Using frozen hidden states extracted from Ouro loop iterations, I trained lightweight evaluator heads of roughly 5M parameters.

On HH-RLHF preference pairs:

| Evaluator | Setup | Result |
|---|---|---|
| Pairwise evaluator | Compares chosen and rejected loop-state trajectories directly | **95.2% test accuracy** |
| Independent nonlinear scorer | Scores each response separately, then compares scores | ~65% test accuracy |
| Linear probe | Linear independent classification | 21.75% accuracy, below chance / inverted polarity |

The main result is therefore not simply "a preference classifier works." The stronger interpretation is that Ouro's loop-state trajectories expose a relational evaluative structure: the model's internal trajectory is much easier to judge comparatively than absolutely.

## Related papers

**Relational Preference Encoding in Looped Transformer Internal States**  
https://arxiv.org/abs/2604.09870

**Scaling Latent Reasoning via Looped Language Models**  
https://arxiv.org/abs/2510.25741

## What this repository is

This is the evaluator repository. It is focused on extracting Ouro loop-state features, training the pairwise evaluator, evaluating checkpoints, and running the linear probe experiments.

The base Ouro model is kept frozen. The trainable part is the lightweight evaluator placed on top of extracted loop-state representations.

## Repository contents

| File | Purpose |
|---|---|
| `evaluator_pairwise.py` | Pairwise evaluator architecture. |
| `train_pairwise_fast.py` | Training script for the pairwise evaluator. |
| `evaluate_pairwise.py` | Evaluation script for trained evaluator checkpoints. |
| `extract_features.py` | Feature extraction from frozen Ouro loop states. |
| `ouro_inspect.py` | Utilities for inspecting Ouro hidden states and loop behavior. |
| `probe_test.py` | Linear probe experiments. |
| `modeling_ouro_patched.py` | Local patched Ouro modeling file used for compatibility during the experiments. |
| `requirements.txt` | Python dependencies. |

## Method

At a high level, the pairwise evaluator works like this:

```text
chosen response loop states     rejected response loop states
             ↓                              ↓
      attention pooling              attention pooling
             ↓                              ↓
         pooled chosen   -   pooled rejected
                         ↓
             sequence of loop differences
                         ↓
                        GRU
                         ↓
                 preference logit
```

The important design choice is that the evaluator is given the **difference between two loop-state trajectories**, rather than being asked to assign each response an absolute score independently.

This matters because the independent / pointwise versions were much weaker, while the pairwise setup reached 95.2% test accuracy.

## Interpretation

I initially treated this as an alignment result. That is still the domain where the experiment was first tested, and HH-RLHF provides clean chosen / rejected pairs.

But the result seems broader than a narrow preference-reader story. The evaluator appears to exploit structure in the loop trajectories themselves. In other words, the interesting object is not only the final answer or the final hidden state, but the **trajectory of recurrent refinement** produced by Ouro's loop iterations.

A cautious interpretation is:

- Ouro loop states contain strong relational evaluative information.
- This information is much easier to read comparatively than absolutely.
- Lightweight external evaluators can extract that signal while the base model remains frozen.
- Looped latent trajectories may be useful objects for future reasoning, evaluation, and alignment work.

## Follow-up direction

After this experiment, I began testing whether the same pairwise evaluator signal transfers outside HH-RLHF-style preference data. Early math-domain experiments suggest that the evaluator does not create correct answers when no candidate is correct, but can help select better candidates when the candidate pool contains a correct answer.

That has led me toward the broader hypothesis that Ouro-like loop-state trajectories should be treated as trainable reasoning objects: representations that can be pooled, compared, scored, anchored, and eventually grounded in perception / action loops.

## Setup

Create a virtual environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

The exact setup may depend on your local Ouro / Transformers environment. During these experiments I used the patched Ouro modeling file included in this repository for compatibility with my local setup.

## Basic usage

### 1. Extract features

```bash
python extract_features.py
```

### 2. Train the pairwise evaluator

```bash
python train_pairwise_fast.py
```

### 3. Evaluate a checkpoint

```bash
python evaluate_pairwise.py
```

Paths and dataset locations may need to be adjusted depending on where the extracted features and checkpoints are stored.

## Notes

This repository should not be read as a general reward-model package or as a universal correctness oracle.

The result is narrower and, in my opinion, more interesting: frozen Ouro loop states appear to contain a strong relational signal, and a small evaluator can read that signal when it is allowed to compare trajectories directly.

## License

This repository is released under the Apache-2.0 license.
