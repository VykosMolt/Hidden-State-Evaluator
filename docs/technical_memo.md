# Ouro-Like Loop-State Trajectories as Trainable Reasoning Objects

**Status:** Short technical memo / research summary  
**Related repository:** `ouro-loop-evaluator`

## 1. Core idea

My current research is built around a simple hypothesis:

> Ouro-like loop-state trajectories should be treated as first-class reasoning objects.

The stronger version of this hypothesis is that the loop trajectory may not merely be where reasoning is recorded, but where reasoning is actually happening. In a looped model, the answer is not produced by a single static representation; it emerges through a sequence of internal refinements. If that is true, then the trajectory through loop states is not just a diagnostic artifact. It is the model’s latent reasoning process.

In most language-model work, the final output is the main object of training and evaluation. In chain-of-thought systems, the generated text may also become part of the reasoning process. Ouro / LoopLM suggests a different axis: repeated latent computation inside the model, where each forward pass contains several internal loop iterations.

My interest is in whether those internal loop trajectories can be directly studied and used. Not only as hidden activations to inspect after the fact, but as objects that can be:

- pooled,
- compared,
- scored,
- used for candidate selection,
- used as an anchor for learned encoders,
- and eventually grounded in perception / action loops.

The current work began with preference and alignment, but the broader question is about the structure and usefulness of looped latent reasoning itself.

## 2. Background: Ouro / LoopLM

Ouro-2.6B-Thinking is a looped language model: instead of relying only on a single feedforward pass plus generated chain-of-thought tokens, it performs multiple internal loop iterations that refine the representation before producing output.

The key reason this is interesting is that recurrence changes what “reasoning compute” can mean. Extra reasoning does not have to appear only as more generated text. It can also appear as internal iterative refinement.

This makes the loop trajectory itself a natural research object.

Related paper:

**Scaling Latent Reasoning via Looped Language Models**  
https://arxiv.org/abs/2510.25741

## 3. Result 1: relational preference encoding in Ouro loop states

I initially started this work as an alignment / preference-readout project.

The experiment was straightforward:

1. Keep Ouro-2.6B-Thinking frozen.
2. Extract hidden states from its loop iterations on HH-RLHF chosen / rejected response pairs.
3. Train lightweight evaluator heads on top of those frozen loop-state features.
4. Compare pairwise evaluation against independent / pointwise evaluation.

The surprising result was not only that preference was readable. It was that it was readable mainly **relationally**.

Using roughly 5M trainable parameters and no fine-tuning of the base Ouro model:

| Evaluator | Setup | Result |
|---|---|---|
| Pairwise evaluator | Directly compares chosen and rejected loop-state trajectories | **95.2% test accuracy** |
| Independent nonlinear scorer | Scores each response separately, then compares scores | ~65% test accuracy |
| Linear probe | Linear independent classification | 21.75% accuracy, below chance / inverted polarity |

The best checkpoint was the epoch-2 pairwise evaluator. Earlier intermediate runs reported lower epoch-1 numbers.

The interpretation I take from this is cautious but important:

> Ouro’s loop states appear to expose a strong relational evaluative structure. The model’s internal trajectory is much easier to judge comparatively than absolutely.

This does not mean the evaluator is a universal reward model, and it does not mean preference is “solved.” It means the loop-state trajectory contains structure that becomes highly readable when two trajectories are compared directly.

Related paper:

**Relational Preference Encoding in Looped Transformer Internal States**  
https://arxiv.org/abs/2604.09870

## 4. Result 2: domain-transfer probe beyond HH-RLHF

After the preference result, I began testing whether the same pairwise loop-state evaluator was only reading HH-RLHF-style preference signal, or whether it was reading a broader internal trajectory-quality signal.

The first domain-transfer tests were on math candidate selection.

The setup was:

1. Generate multiple candidate answers for a math problem.
2. Extract Ouro loop-state trajectories for each candidate.
3. Use the frozen pairwise evaluator in a tournament / round-robin selection setup.
4. Compare the evaluator-selected candidate against simple baselines such as first candidate and random candidate.
5. Check correctness using a math answer oracle.

The current conclusion is not “the evaluator solves math.”

It does not. The important observation is more specific:

> The evaluator cannot create a correct answer when the candidate pool contains none, but when at least one candidate is correct, it can help select the better candidate.

In the cleanest math transfer experiment so far, the evaluator selected correct candidates at about **2.7× the base selection rate** in the candidate-selection setup.

The effect also appears much stronger on **Ouro-2.6B-Thinking** than on base **Ouro-2.6B**. That contrast matters because it suggests the evaluator may depend on the reasoning-specialized loop geometry produced by the Thinking model, rather than merely exploiting superficial preference style.

My current interpretation is:

- the evaluator is not just an HH-RLHF alignment-style classifier;
- it is also not a direct correctness oracle;
- it appears to read something like trajectory quality, internal consistency, or refinement structure;
- that signal seems much more legible in the reasoning-trained Ouro model.

This is still early and needs broader testing, but it is one of the main reasons I now think loop-state trajectories are worth treating as trainable objects.

## 5. Current extension: grounding loop states in ARC-style reasoning

My current work extends this from text preference / candidate selection toward interactive abstract reasoning.

I am building an object-centric ARC-style agent around frozen Ouro loop states as the central reasoning substrate.

The system currently includes:

- a grid encoder that maps structured inputs into Ouro-compatible token space;
- scene and object parsing;
- object/event memory;
- affordance learning from observed interactions;
- loop-state pooling over multiple Ouro loop iterations;
- failure diagnostics for tracking, self-model, topology, mechanism, and planner failures;
- and use of the same pairwise loop-state evaluator as an anchor for keeping learned representations closer to Ouro-compatible latent space.

ARC is not the end goal. I am using it as a clean testbed because it stresses object-centric reasoning, spatial structure, causal interaction, and adaptation to unfamiliar environments.

The broader target is to understand whether looped latent computation can be grounded in perception and action.

## 6. Architecture sketch

The current direction can be summarized as:

```text
structured observation / grid
          ↓
      grid encoder
          ↓
 Ouro-compatible token sequence
          ↓
  Ouro loop states L1-L4
          ↓
 loop-state pooling / comparison
          ↓
 object, event, affordance, and action scoring
          ↓
      action selection
```

The pairwise evaluator fits into this as a reader over loop-state trajectories:

```text
candidate A loop trajectory     candidate B loop trajectory
              ↓                           ↓
        pooled loop states          pooled loop states
              ↓                           ↓
              trajectory difference sequence
                          ↓
                      GRU evaluator
                          ↓
                    preference / quality logit
```

The long-term idea is that the evaluator should not just be a post-hoc judge. It can also become a training anchor: a way to pressure learned encoders and context modules to produce representations that remain legible to the looped reasoning substrate.

## 7. Why this matters

If reasoning is iterative, then the internal iterative trajectory may be the computation we actually care about. The loop states are not just intermediate activations on the way to an answer; they may be the latent form of the reasoning process itself.

Current reasoning systems often put most of the pressure on final outputs, generated chain-of-thought, or external search. Ouro-like recurrence suggests another path:

> train, evaluate, and ground the latent refinement process itself.

This does not replace output supervision, search, or tool use. But it adds another object of supervision: the loop-state trajectory.

This is the main reason I think loop-state trajectories deserve direct supervision. If the trajectory is the thinking process, then training only the final output is an indirect way of shaping the thing we actually care about. A more direct approach is to evaluate and shape the latent refinement trajectory itself.

That opens several research directions:

- external evaluators over loop trajectories;
- candidate selection using latent trajectory comparison;
- learned encoders anchored to a frozen looped backbone;
- self/context tokens that shape recurrent refinement;
- object/event memory that interacts with latent reasoning;
- and eventually, grounded agents whose internal loop trajectories are part of their world-modeling process.

## 8. What is not being claimed

This work is early-stage.

I am not claiming:

- a state-of-the-art ARC solver;
- a universal reward model;
- that the evaluator is a direct correctness oracle;
- or that HH-RLHF preference accuracy alone proves general reasoning ability.

I am also not claiming that every loop-state transition is meaningful reasoning. Some of it may be routing, stabilization, or representational cleanup. The claim is that in Ouro-like models, a substantial part of the reasoning process may live in the trajectory of latent refinement, and that trajectory appears readable enough to be worth training and evaluating directly.

The claims are narrower:

1. Frozen Ouro loop states contain a strong relational signal on preference pairs.
2. Pairwise comparison is much more effective than independent pointwise scoring.
3. The same evaluator shows early evidence of partial transfer to math candidate selection.
4. Reasoning-trained Ouro appears to expose more readable loop-state geometry than base Ouro.
5. These results motivate treating loop-state trajectories as trainable reasoning objects.

## 9. Current research direction

The immediate next steps are:

- make the math transfer result cleaner and better controlled;
- analyze whether the evaluator is partly reading loop depth / exit behavior;
- test more domains beyond HH-RLHF and math;
- continue grounding Ouro loop states into ARC-style object and action reasoning;
- improve loop-state pooling;
- and explore evaluator anchoring for learned encoders and self/context modules.

The long-term research question is:

> Can looped latent trajectories become the central substrate for reasoning, evaluation, and grounded control?

That is the direction this repository and the surrounding project are exploring.
