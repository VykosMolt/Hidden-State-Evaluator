import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluator2 import ConstitutionalEvaluatorV2, validate_hook_output

# --- Configuration ---
MODEL_NAME = "ByteDance/Ouro-2.6B-Thinking"
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4          # effective batch size = 2 * 4 = 8
EPOCHS = 3
LEARNING_RATE = 1e-4
WARMUP_STEPS = 200
MAX_LENGTH = 1024
CHECKPOINT_DIR = "checkpoints_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Dataset ---
class ConstitutionalDataset(Dataset):
    def __init__(self, split="train", max_samples=25000):
        print(f"Loading HH-RLHF dataset ({split})...")
        ds = load_dataset("Anthropic/hh-rlhf", split=split)
        ds = ds.shuffle(seed=42)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.data = ds
        print(f"Loaded {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "chosen": self.data[idx]["chosen"],
            "rejected": self.data[idx]["rejected"]
        }

# --- Hook to capture all loop hidden states ---
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
    """
    Returns raw hidden states (no pooling) and attention mask.
    Pooling is now handled inside the evaluator via AttentionPool.
    """
    captured.clear()
    with torch.no_grad():
        model(**tokens)

    hidden_states_list = captured["hidden_states_list"]
    device = tokens["input_ids"].device

    # cast to float32 on correct device, but do NOT pool
    raw_list = [
        h.to(device=device, dtype=torch.float32)
        for h in hidden_states_list
    ]
    attention_mask = tokens["attention_mask"]
    return raw_list, attention_mask

# --- Loss ---
def pairwise_loss(score_chosen, score_rejected):
    ranking_loss = -F.logsigmoid(score_chosen - score_rejected).mean()
    l2_reg = 1e-5 * (score_chosen ** 2 + score_rejected ** 2).mean()
    return ranking_loss + l2_reg

# --- Main Training ---
def train():
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

    dataset = ConstitutionalDataset(split="train", max_samples=25000)

    evaluator = ConstitutionalEvaluatorV2().to(DEVICE)
    optimizer = torch.optim.AdamW(evaluator.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM_STEPS

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_STEPS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - WARMUP_STEPS)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS]
    )

    print(f"Total steps: {total_steps} (warmup: {WARMUP_STEPS})")
    print(f"LR: {LEARNING_RATE}, effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}, attention pooling")
    print("Starting training...")

    global_step = 0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            tokens_chosen = tokenizer(
                batch["chosen"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)

            tokens_rejected = tokenizer(
                batch["rejected"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)

            hidden_chosen, mask_chosen = get_all_hidden_states(model, tokens_chosen)
            hidden_rejected, mask_rejected = get_all_hidden_states(model, tokens_rejected)

            if batch_idx == 0 and epoch == 0:
                print(f"Num loop states: {len(hidden_chosen)}, shape: {hidden_chosen[0].shape}")

            score_chosen = evaluator(hidden_chosen, mask_chosen)
            score_rejected = evaluator(hidden_rejected, mask_rejected)

            loss = pairwise_loss(score_chosen, score_rejected)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(evaluator.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * GRAD_ACCUM_STEPS

            batch_size_actual = score_chosen.shape[0]
            correct += (score_chosen > score_rejected).sum().item()
            total += batch_size_actual

            if batch_idx % 50 == 0:
                acc = correct / total if total > 0 else 0
                margin = (score_chosen - score_rejected).mean().item()
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1} | Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {total_loss / (batch_idx + 1):.4f} | "
                    f"Acc: {acc:.4f} | "
                    f"Margin: {margin:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Chosen: {score_chosen.mean().item():.4f} | "
                    f"Rejected: {score_rejected.mean().item():.4f}"
                )

        # flush remaining gradients
        if (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(evaluator.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_acc = correct / total if total > 0 else 0
        epoch_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch + 1} Summary: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f} ---")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"evaluator_v2_epoch{epoch + 1}.pt")
        torch.save({
            "model_state_dict": evaluator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "accuracy": epoch_acc,
            "loss": epoch_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    print("Training complete.")

if __name__ == "__main__":
    train()
