import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluator2 import validate_hook_output

# --- Configuration ---
MODEL_NAME = "ByteDance/Ouro-2.6B-Thinking"
MAX_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 1000  # up from 200

# --- Hook ---
captured = {}
_hook_validated = False

def hook_fn(module, input, output):
    global _hook_validated
    hidden_states = output[1]
    if not _hook_validated:
        validate_hook_output(hidden_states)
        print(f"Hook output structure: {type(output)}, len={len(output)}")
        print(f"  output[0] type: {type(output[0])}, shape: {output[0].shape if hasattr(output[0], 'shape') else 'N/A'}")
        print(f"  output[1] type: {type(output[1])}, len={len(output[1])}")
        for i, h in enumerate(output[1]):
            print(f"    loop state {i}: {h.shape}, dtype={h.dtype}")
        if len(output) > 2:
            print(f"  output[2] type: {type(output[2])}")
        _hook_validated = True
    captured["hidden_states_list"] = [h.detach() for h in hidden_states]

def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)

def get_pooled_states(model, tokens):
    captured.clear()
    with torch.no_grad():
        model(**tokens)
    hidden_states_list = captured["hidden_states_list"]
    device = tokens["input_ids"].device
    return [
        mean_pool(h.to(device=device, dtype=torch.float32), tokens["attention_mask"])
        for h in hidden_states_list
    ]

def run_probe(features, labels, label):
    X = np.array(features)
    y = np.array(labels)

    n = len(y)
    split = int(0.8 * n)
    idx = np.random.RandomState(42).permutation(n)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    print(f"  {label}: {acc:.4f} (train={len(y_train)}, test={len(y_test)}, features={X.shape[1]})")
    return acc

def main():
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

    print("Loading dataset...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    ds = ds.shuffle(seed=42)

    print(f"\n--- Linear Probe ({N_SAMPLES} examples) ---")

    all_chosen_states = []
    all_rejected_states = []

    for idx in range(min(N_SAMPLES, len(ds))):
        item = ds[idx]

        tokens_c = tokenizer(
            item["chosen"], return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)
        tokens_r = tokenizer(
            item["rejected"], return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)

        pooled_c = get_pooled_states(model, tokens_c)
        pooled_r = get_pooled_states(model, tokens_r)

        all_chosen_states.append(pooled_c)
        all_rejected_states.append(pooled_r)

        if (idx + 1) % 100 == 0:
            print(f"  Extracted {idx + 1}/{N_SAMPLES}...")

    # --- Pairwise probe (chosen - rejected) ---
    print("\n--- Pairwise Probe (chosen - rejected direction) ---")

    for use_final_only, label in [(True, "final_state_only"), (False, "all_states_concat")]:
        features = []
        labels = []

        for chosen_states, rejected_states in zip(all_chosen_states, all_rejected_states):
            if use_final_only:
                c = chosen_states[-1].detach().cpu().numpy()
                r = rejected_states[-1].detach().cpu().numpy()
            else:
                c = torch.cat(chosen_states, dim=-1).detach().cpu().numpy()
                r = torch.cat(rejected_states, dim=-1).detach().cpu().numpy()

            diff = c - r
            for row in diff:
                features.append(row)
                labels.append(1)
            for row in -diff:
                features.append(row)
                labels.append(0)

        run_probe(features, labels, label)

    # --- Independent classification probe (for comparison) ---
    print("\n--- Independent Classification Probe ---")

    for use_final_only, label in [(True, "final_state_only"), (False, "all_states_concat")]:
        features = []
        labels = []

        for chosen_states, rejected_states in zip(all_chosen_states, all_rejected_states):
            if use_final_only:
                c = chosen_states[-1].detach().cpu().numpy()
                r = rejected_states[-1].detach().cpu().numpy()
            else:
                c = torch.cat(chosen_states, dim=-1).detach().cpu().numpy()
                r = torch.cat(rejected_states, dim=-1).detach().cpu().numpy()

            for row in c:
                features.append(row)
                labels.append(1)
            for row in r:
                features.append(row)
                labels.append(0)

        run_probe(features, labels, label)

    print("\nDone.")

if __name__ == "__main__":
    main()
