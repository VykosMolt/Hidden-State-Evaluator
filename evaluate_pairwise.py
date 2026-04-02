import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluator_pairwise import PairwiseEvaluator, validate_hook_output

# --- Configuration ---
MODEL_NAME = "ByteDance/Ouro-2.6B-Thinking"
CHECKPOINT_PATH = "checkpoints_pairwise/pairwise_epoch5.pt"
MAX_LENGTH = 1024
BATCH_SIZE = 2
DETAIL_SAMPLES = 10
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
    raw_list = [h.to(device=device, dtype=torch.float32) for h in hidden_states_list]
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
    evaluator = PairwiseEvaluator().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    evaluator.load_state_dict(checkpoint["model_state_dict"])
    evaluator.eval()

    print("Loading test split...")
    ds = load_dataset("Anthropic/hh-rlhf", split="test")
    print(f"Test examples: {len(ds)}")

    correct = 0
    total = 0
    scores_list = []

    # --- Detailed per-example analysis ---
    print(f"\n--- Per-Example Analysis (first {DETAIL_SAMPLES} examples) ---")
    for idx in range(DETAIL_SAMPLES):
        tokens_chosen = tokenizer(
            ds[idx]["chosen"], return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)
        tokens_rejected = tokenizer(
            ds[idx]["rejected"], return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)

        hidden_c, mask_c = get_all_hidden_states(model, tokens_chosen)
        hidden_r, mask_r = get_all_hidden_states(model, tokens_rejected)

        with torch.no_grad():
            score = evaluator(hidden_c, mask_c, hidden_r, mask_r).item()

        result = "✓" if score > 0 else "✗"
        print(f"[{result}] Example {idx+1} | Score: {score:.4f}")

    # --- Batched evaluation ---
    print(f"\n--- Batched Evaluation ---")

    for start in range(0, len(ds), BATCH_SIZE):
        batch = ds[start:start + BATCH_SIZE]
        chosen_texts = batch["chosen"]
        rejected_texts = batch["rejected"]

        tokens_chosen = tokenizer(
            chosen_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)
        tokens_rejected = tokenizer(
            rejected_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)

        hidden_c, mask_c = get_all_hidden_states(model, tokens_chosen)
        hidden_r, mask_r = get_all_hidden_states(model, tokens_rejected)

        with torch.no_grad():
            scores = evaluator(hidden_c, mask_c, hidden_r, mask_r)

        batch_scores = scores.view(-1).tolist()
        scores_list.extend(batch_scores)
        correct += sum(1 for s in batch_scores if s > 0)
        total += len(batch_scores)

        del hidden_c, hidden_r
        torch.cuda.empty_cache()

        if total % 500 < BATCH_SIZE:
            acc = correct / total
            avg_score = np.mean(scores_list)
            print(f"Progress: {total}/{len(ds)} | Acc: {acc:.4f} | Avg Score: {avg_score:.4f}")

    # --- Final results ---
    scores_arr = np.array(scores_list)
    final_acc = correct / total

    print(f"\n--- Final Results ---")
    print(f"Test examples:       {total}")
    print(f"Accuracy:            {final_acc:.4f} ({final_acc*100:.1f}%)")
    print(f"Average score:       {np.mean(scores_arr):.4f}")
    print(f"Score std:           {np.std(scores_arr):.4f}")
    print(f"Min score:           {np.min(scores_arr):.4f}")
    print(f"Max score:           {np.max(scores_arr):.4f}")
    print(f"Positive rate:       {np.mean(scores_arr > 0)*100:.1f}%")

if __name__ == "__main__":
    evaluate()
