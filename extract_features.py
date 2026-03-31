import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluator2 import validate_hook_output

# --- Configuration ---
MODEL_NAME = "ByteDance/Ouro-2.6B-Thinking"
MAX_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_DIR = "/mnt/sandisk/ouro_features"
CHUNK_SIZE = 100            # examples per file
MAX_SAMPLES = 50000

# What to save:
#   "raw"    — full [seq_len, hidden_dim] per loop state (large, needed for attention pooling)
#   "pooled" — mean-pooled [hidden_dim] per loop state (compact, ~800MB for 25k)
SAVE_MODE = "raw"

os.makedirs(FEATURE_DIR, exist_ok=True)

# --- Hook ---
captured = {}
_hook_validated = False

def hook_fn(module, input, output):
    global _hook_validated
    hidden_states = output[1]
    if not _hook_validated:
        validate_hook_output(hidden_states)
        print(f"Confirmed: {len(hidden_states)} loop states, shape {hidden_states[0].shape}")
        _hook_validated = True
    captured["hidden_states_list"] = [h.detach() for h in hidden_states]

def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)

def extract():
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
    if MAX_SAMPLES:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    print(f"Examples to extract: {len(ds)}")
    print(f"Save mode: {SAVE_MODE}")

    chunk = []
    chunk_idx = 0
    total_extracted = 0

    for idx in range(len(ds)):
        item = ds[idx]

        # process chosen
        tokens_c = tokenizer(
            item["chosen"], return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)

        captured.clear()
        with torch.no_grad():
            model(**tokens_c)

        if SAVE_MODE == "raw":
            chosen_states = [h.cpu().to(torch.float16) for h in captured["hidden_states_list"]]
            chosen_mask = tokens_c["attention_mask"].cpu()
        else:
            chosen_states = [
                mean_pool(h.to(dtype=torch.float32), tokens_c["attention_mask"]).cpu()
                for h in captured["hidden_states_list"]
            ]
            chosen_mask = None

        # process rejected
        tokens_r = tokenizer(
            item["rejected"], return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH
        ).to(DEVICE)

        captured.clear()
        with torch.no_grad():
            model(**tokens_r)

        if SAVE_MODE == "raw":
            rejected_states = [h.cpu().to(torch.float16) for h in captured["hidden_states_list"]]
            rejected_mask = tokens_r["attention_mask"].cpu()
        else:
            rejected_states = [
                mean_pool(h.to(dtype=torch.float32), tokens_r["attention_mask"]).cpu()
                for h in captured["hidden_states_list"]
            ]
            rejected_mask = None

        example = {
            "chosen_states": chosen_states,
            "rejected_states": rejected_states,
        }
        if SAVE_MODE == "raw":
            example["chosen_mask"] = chosen_mask
            example["rejected_mask"] = rejected_mask

        chunk.append(example)
        total_extracted += 1

        # save chunk when full
        if len(chunk) >= CHUNK_SIZE:
            chunk_path = os.path.join(FEATURE_DIR, f"chunk_{chunk_idx:04d}.pt")
            torch.save(chunk, chunk_path)
            chunk = []
            chunk_idx += 1
            print(f"  Saved chunk {chunk_idx} | Total: {total_extracted}/{len(ds)}")

    # save remaining
    if chunk:
        chunk_path = os.path.join(FEATURE_DIR, f"chunk_{chunk_idx:04d}.pt")
        torch.save(chunk, chunk_path)
        chunk_idx += 1
        print(f"  Saved chunk {chunk_idx} | Total: {total_extracted}/{len(ds)}")

    # save metadata
    meta = {
        "total_examples": total_extracted,
        "num_chunks": chunk_idx,
        "chunk_size": CHUNK_SIZE,
        "save_mode": SAVE_MODE,
        "max_length": MAX_LENGTH,
        "model_name": MODEL_NAME,
        "early_exit_threshold": 0.87,
    }
    torch.save(meta, os.path.join(FEATURE_DIR, "meta.pt"))

    print(f"\nExtraction complete.")
    print(f"  {total_extracted} examples in {chunk_idx} chunks")
    print(f"  Saved to: {FEATURE_DIR}/")

    # estimate disk usage
    total_size = sum(
        os.path.getsize(os.path.join(FEATURE_DIR, f))
        for f in os.listdir(FEATURE_DIR)
    )
    print(f"  Total disk: {total_size / 1e9:.1f} GB")

if __name__ == "__main__":
    extract()
