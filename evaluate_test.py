import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluator_test import ConstitutionalEvaluatorTest, validate_hook_output

# --- Configuration ---
MODEL_NAME = "ByteDance/Ouro-2.6B-Thinking"
CHECKPOINT_PATH = "checkpoints_test/evaluator_test_epoch3.pt"
MAX_LENGTH = 1024
BATCH_SIZE = 2
TRAJECTORY_SAMPLES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Hook ---
captured = {}
_hook_validated = False

def hook_fn(module, input, output):
    global _hook_validated
    hidden_states = output[1]
    if not _hook_validated:
        validate_hook_output(hidden_states)
        _hook_validated = True
    captured["hidden_states_list"] = [h.detach() for h in hidden_states]

def get_all_hidden_states(model, tokens):
    captured.clear()
    with torch.no_grad():
        model(**tokens)
    hidden_states_list = captured["hidden_states_list"]
    device = tokens["input_ids"].device
    raw_list = [
        h.to(device=device, dtype=torch.float32)
        for h in hidden_states_list
    ]
    attention_mask = tokens["attention_mask"]
    return raw_list, attention_mask

def evaluate():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model.config.early_exit_threshold = 0.87
    model.model.register_forward_hook(hook_fn)

    print(f"Loading evaluator checkpoint: {CHECKPOINT_PATH}")
    evaluator = ConstitutionalEvaluatorTest().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    evaluator.load_state_dict(checkpoint["model_state_dict"])
    evaluator.eval()

    print("Loading test split...")
    ds = load_dataset("Anthropic/hh-rlhf", split="test")
    print(f"Test examples: {len(ds)}")

    correct = 0
    total = 0
    margins = []

    # --- Per-example analysis on first TRAJECTORY_SAMPLES examples ---
    print(f"\n--- Per-Example Analysis (first {TRAJECTORY_SAMPLES} examples) ---")
    for idx in range(TRAJECTORY_SAMPLES):
        chosen = ds[idx]["chosen"]
        rejected = ds[idx]["rejected"]

        tokens_chosen = tokenizer(
            chosen, return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)

        tokens_rejected = tokenizer(
            rejected, return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)

        hidden_chosen, mask_chosen = get_all_hidden_states(model, tokens_chosen)
        hidden_rejected, mask_rejected = get_all_hidden_states(model, tokens_rejected)

        with torch.no_grad():
            sc = evaluator(hidden_chosen, mask_chosen).item()
            sr = evaluator(hidden_rejected, mask_rejected).item()

        result = "✓" if sc > sr else "✗"
        print(f"[{result}] Example {idx+1}")
        print(f"  Final scores — Chosen: {sc:.4f} | Rejected: {sr:.4f} | Margin: {sc-sr:.4f}")

    # --- Batched evaluation ---
    print(f"\n--- Batched Evaluation (batch_size={BATCH_SIZE}) ---")

    for start in range(0, len(ds), BATCH_SIZE):
        batch = ds[start:start + BATCH_SIZE]
        chosen_texts = batch["chosen"]
        rejected_texts = batch["rejected"]

        # process chosen
        tokens_chosen = tokenizer(
            chosen_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)
        hidden_chosen, mask_chosen = get_all_hidden_states(model, tokens_chosen)

        with torch.no_grad():
            scores_chosen = evaluator(hidden_chosen, mask_chosen)

        del hidden_chosen
        torch.cuda.empty_cache()

        # process rejected
        tokens_rejected = tokenizer(
            rejected_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)
        hidden_rejected, mask_rejected = get_all_hidden_states(model, tokens_rejected)

        with torch.no_grad():
            scores_rejected = evaluator(hidden_rejected, mask_rejected)

        del hidden_rejected
        torch.cuda.empty_cache()

        batch_margins = (scores_chosen - scores_rejected).view(-1).tolist()
        margins.extend(batch_margins)
        correct += sum(1 for m in batch_margins if m > 0)
        total += len(batch_margins)

        del scores_chosen, scores_rejected
        torch.cuda.empty_cache()

        if total % 500 < BATCH_SIZE:
            acc = correct / total
            avg_margin = np.mean(margins)
            print(f"Progress: {total}/{len(ds)} | Acc: {acc:.4f} | Avg Margin: {avg_margin:.4f}")

    # --- Final results ---
    margins = np.array(margins)
    final_acc = correct / total

    print(f"\n--- Final Results ---")
    print(f"Test examples:       {total}")
    print(f"Accuracy:            {final_acc:.4f} ({final_acc*100:.1f}%)")
    print(f"Average margin:      {np.mean(margins):.4f}")
    print(f"Margin std:          {np.std(margins):.4f}")
    print(f"Min margin:          {np.min(margins):.4f}")
    print(f"Max margin:          {np.max(margins):.4f}")
    print(f"Positive margin rate:{np.mean(margins > 0)*100:.1f}%")

if __name__ == "__main__":
    evaluate()
