import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from evaluator import ConstitutionalEvaluator, mean_pool, concat_loop_states

# --- Configuration ---
MODEL_NAME = "ByteDance/Ouro-2.6B-Thinking"
CHECKPOINT_PATH = "checkpoints/evaluator_epoch3.pt"
MAX_LENGTH = 1024
BATCH_SIZE = 2
TRAJECTORY_SAMPLES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Hook ---
captured = {}

def hook_fn(module, input, output):
    captured["hidden_states_list"] = [h.detach() for h in output[1]]

def get_all_hidden_states(model, tokens):
    captured.clear()
    with torch.no_grad():
        model(**tokens)
    hidden_states_list = captured["hidden_states_list"]
    assert len(hidden_states_list) > 0, "No hidden states captured"
    device = tokens["input_ids"].device
    return [
        mean_pool(h.to(device=device, dtype=torch.float32), tokens["attention_mask"])
        for h in hidden_states_list
    ]

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
    evaluator = ConstitutionalEvaluator().to(DEVICE)
    evaluator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    evaluator.eval()

    print("Loading test split...")
    ds = load_dataset("Anthropic/hh-rlhf", split="test")
    print(f"Test examples: {len(ds)}")

    correct = 0
    total = 0
    margins = []

    # --- Trajectory analysis on first TRAJECTORY_SAMPLES examples ---
    print(f"\n--- Trajectory Analysis (first {TRAJECTORY_SAMPLES} examples) ---")
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

        pooled_chosen = get_all_hidden_states(model, tokens_chosen)
        pooled_rejected = get_all_hidden_states(model, tokens_rejected)

        _, traj_chosen = evaluator.trajectory(pooled_chosen)
        _, traj_rejected = evaluator.trajectory(pooled_rejected)

        concat_c = concat_loop_states(pooled_chosen).to(DEVICE)
        concat_r = concat_loop_states(pooled_rejected).to(DEVICE)

        with torch.no_grad():
            sc = evaluator(concat_c).item()
            sr = evaluator(concat_r).item()

        result = "✓" if sc > sr else "✗"
        print(f"[{result}] Example {idx+1}")
        print(f"  Chosen trajectory:   {[round(t, 3) for t in traj_chosen]}")
        print(f"  Rejected trajectory: {[round(t, 3) for t in traj_rejected]}")
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
        pooled_chosen = get_all_hidden_states(model, tokens_chosen)
        concat_chosen = concat_loop_states(pooled_chosen).to(DEVICE)
        del pooled_chosen
        torch.cuda.empty_cache()
    
        # process rejected
        tokens_rejected = tokenizer(
            rejected_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)
        pooled_rejected = get_all_hidden_states(model, tokens_rejected)
        concat_rejected = concat_loop_states(pooled_rejected).to(DEVICE)
        del pooled_rejected
        torch.cuda.empty_cache()
    
        with torch.no_grad():
            scores_chosen = evaluator(concat_chosen)
            scores_rejected = evaluator(concat_rejected)
    
        batch_margins = (scores_chosen - scores_rejected).view(-1).tolist()
        margins.extend(batch_margins)
        correct += sum(1 for m in batch_margins if m > 0)
        total += len(batch_margins)
    
        del concat_chosen, concat_rejected, scores_chosen, scores_rejected
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
