import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluator_pairwise import validate_hook_output

# --- Configuration ---
MODEL_NAME = "ByteDance/Ouro-2.6B-Thinking"
MAX_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 500

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

def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)

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

    print("Loading dataset (test split)...")
    ds = load_dataset("Anthropic/hh-rlhf", split="test")
    ds = ds.select(range(min(N_SAMPLES, len(ds))))

    # --- Collectors ---
    chosen_lengths = []
    rejected_lengths = []
    chosen_char_lengths = []
    rejected_char_lengths = []
    chosen_norms = {i: [] for i in range(4)}
    rejected_norms = {i: [] for i in range(4)}
    chosen_mean_acts = {i: [] for i in range(4)}
    rejected_mean_acts = {i: [] for i in range(4)}
    chosen_pooled_norms = []
    rejected_pooled_norms = []
    diff_norms = []
    norm_correct = 0  # does larger norm = chosen?
    length_correct = 0  # does longer = chosen?

    print(f"\n--- Analyzing {N_SAMPLES} examples ---")

    for idx in range(len(ds)):
        item = ds[idx]

        # raw text lengths
        chosen_char_lengths.append(len(item["chosen"]))
        rejected_char_lengths.append(len(item["rejected"]))

        # tokenize
        tokens_c = tokenizer(
            item["chosen"], return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)
        tokens_r = tokenizer(
            item["rejected"], return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)

        seq_len_c = tokens_c["attention_mask"].sum().item()
        seq_len_r = tokens_r["attention_mask"].sum().item()
        chosen_lengths.append(seq_len_c)
        rejected_lengths.append(seq_len_r)

        if seq_len_c > seq_len_r:
            length_correct += 1
        elif seq_len_c == seq_len_r:
            length_correct += 0.5

        # get hidden states - chosen
        captured.clear()
        with torch.no_grad():
            model(**tokens_c)
        hidden_c = captured["hidden_states_list"]

        # get hidden states - rejected
        captured.clear()
        with torch.no_grad():
            model(**tokens_r)
        hidden_r = captured["hidden_states_list"]

        n_loops = min(len(hidden_c), len(hidden_r), 4)

        for i in range(n_loops):
            hc = hidden_c[i].to(dtype=torch.float32)
            hr = hidden_r[i].to(dtype=torch.float32)

            # per-loop-step norms
            c_norm = hc.norm().item()
            r_norm = hr.norm().item()
            chosen_norms[i].append(c_norm)
            rejected_norms[i].append(r_norm)

            # mean activation magnitude
            c_mean = hc.abs().mean().item()
            r_mean = hr.abs().mean().item()
            chosen_mean_acts[i].append(c_mean)
            rejected_mean_acts[i].append(r_mean)

        # mean-pooled final state norms
        pooled_c = mean_pool(
            hidden_c[-1].to(dtype=torch.float32), tokens_c["attention_mask"]
        )
        pooled_r = mean_pool(
            hidden_r[-1].to(dtype=torch.float32), tokens_r["attention_mask"]
        )

        pc_norm = pooled_c.norm().item()
        pr_norm = pooled_r.norm().item()
        chosen_pooled_norms.append(pc_norm)
        rejected_pooled_norms.append(pr_norm)

        diff_norms.append(pc_norm - pr_norm)

        if pc_norm > pr_norm:
            norm_correct += 1
        elif pc_norm == pr_norm:
            norm_correct += 0.5

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(ds)}...")

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC REPORT ({N_SAMPLES} examples)")
    print(f"{'='*60}")

    print(f"\n--- SEQUENCE LENGTH ---")
    print(f"  Chosen  tokens: mean={np.mean(chosen_lengths):.1f}, std={np.std(chosen_lengths):.1f}")
    print(f"  Rejected tokens: mean={np.mean(rejected_lengths):.1f}, std={np.std(rejected_lengths):.1f}")
    print(f"  Chosen longer: {length_correct/len(ds)*100:.1f}%")
    print(f"  Avg diff: {np.mean(np.array(chosen_lengths) - np.array(rejected_lengths)):.1f} tokens")
    print(f"  Chosen  chars: mean={np.mean(chosen_char_lengths):.0f}")
    print(f"  Rejected chars: mean={np.mean(rejected_char_lengths):.0f}")

    print(f"\n--- HIDDEN STATE NORMS (per loop step) ---")
    for i in range(4):
        if chosen_norms[i]:
            c_mean = np.mean(chosen_norms[i])
            r_mean = np.mean(rejected_norms[i])
            print(f"  Step {i}: Chosen={c_mean:.2f}, Rejected={r_mean:.2f}, "
                  f"Ratio={c_mean/r_mean:.4f}")

    print(f"\n--- MEAN ACTIVATION MAGNITUDE (per loop step) ---")
    for i in range(4):
        if chosen_mean_acts[i]:
            c_mean = np.mean(chosen_mean_acts[i])
            r_mean = np.mean(rejected_mean_acts[i])
            print(f"  Step {i}: Chosen={c_mean:.4f}, Rejected={r_mean:.4f}, "
                  f"Ratio={c_mean/r_mean:.4f}")

    print(f"\n--- MEAN-POOLED FINAL STATE NORMS ---")
    print(f"  Chosen:  mean={np.mean(chosen_pooled_norms):.4f}, std={np.std(chosen_pooled_norms):.4f}")
    print(f"  Rejected: mean={np.mean(rejected_pooled_norms):.4f}, std={np.std(rejected_pooled_norms):.4f}")
    print(f"  Norm predicts chosen: {norm_correct/len(ds)*100:.1f}%")
    print(f"  Avg norm diff: {np.mean(diff_norms):.4f}")

    print(f"\n--- SIMPLE BASELINES ---")
    print(f"  'Longer sequence = chosen': {length_correct/len(ds)*100:.1f}%")
    print(f"  'Larger pooled norm = chosen': {norm_correct/len(ds)*100:.1f}%")

    # correlation between length diff and norm diff
    len_diffs = np.array(chosen_lengths) - np.array(rejected_lengths)
    norm_diffs = np.array(diff_norms)
    corr = np.corrcoef(len_diffs, norm_diffs)[0, 1]
    print(f"  Length-norm correlation: {corr:.4f}")

    print(f"\n--- CONCLUSION ---")
    if length_correct / len(ds) > 0.9:
        print("  ⚠️  Length alone separates 90%+ of pairs.")
        print("  The pairwise model is likely exploiting sequence length.")
    elif norm_correct / len(ds) > 0.9:
        print("  ⚠️  Pooled norm alone separates 90%+ of pairs.")
        print("  The pairwise model is likely exploiting activation magnitude.")
    elif length_correct / len(ds) > 0.7 or norm_correct / len(ds) > 0.7:
        print("  ⚠  Structural features partially separate pairs.")
        print("  The model may be combining structural + semantic signal.")
    else:
        print("  ✓ No obvious structural shortcut found.")
        print("  The model may be learning genuine preference signal.")

if __name__ == "__main__":
    main()
