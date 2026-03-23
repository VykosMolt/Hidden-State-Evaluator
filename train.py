import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluator import ConstitutionalEvaluator, mean_pool, concat_loop_states

# --- Configuration ---
MODEL_NAME = "ByteDance/Ouro-2.6B-Thinking"
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_LENGTH = 1024
CHECKPOINT_DIR = "checkpoints"
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

def hook_fn(module, input, output):
    captured["hidden_states_list"] = [h.detach() for h in output[1]]

def get_all_hidden_states(model, tokens):
    captured.clear()
    with torch.no_grad():
        model(**tokens)

    hidden_states_list = captured["hidden_states_list"]
    assert len(hidden_states_list) > 0, "No hidden states captured"

    device = tokens["input_ids"].device

    pooled_list = [
        mean_pool(h.to(device=device, dtype=torch.float32), tokens["attention_mask"])
        for h in hidden_states_list
    ]
    return pooled_list

# --- Pairwise Ranking Loss ---
def pairwise_loss(score_chosen, score_rejected):
    ranking_loss = -torch.log(torch.sigmoid(score_chosen - score_rejected)).mean()
    l2_reg = 0.001 * (score_chosen**2 + score_rejected**2).mean()
    return ranking_loss + l2_reg

def trajectory_loss(scores_chosen, scores_rejected):
    n = len(scores_chosen)
    final_loss = pairwise_loss(scores_chosen[-1], scores_rejected[-1])

    if n > 1:
        weights = torch.linspace(0.5, 1.0, steps=n-1).to(scores_chosen[0].device)
        weights = weights / weights.sum()

        aux_loss = sum(
            w * pairwise_loss(sc, sr)
            for w, sc, sr in zip(weights, scores_chosen[:-1], scores_rejected[:-1])
        )
        # normalize by loop count relative to expected 4 steps
        aux_loss = aux_loss * (n / 4.0)
        return final_loss + 0.1 * aux_loss

    return final_loss
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

    evaluator = ConstitutionalEvaluator().to(DEVICE)
    optimizer = torch.optim.AdamW(evaluator.parameters(), lr=LEARNING_RATE)

    dataset = ConstitutionalDataset(split="train", max_samples=25000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

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

            pooled_chosen = get_all_hidden_states(model, tokens_chosen)
            pooled_rejected = get_all_hidden_states(model, tokens_rejected)

            # sanity print on first batch
            if batch_idx == 0 and epoch == 0:
                print(f"Num loop states: {len(pooled_chosen)}, shape: {pooled_chosen[0].shape}")

            # trajectory supervision — supervise alignment at every loop step
            scores_chosen, _ = evaluator.trajectory(pooled_chosen)
            scores_rejected, _ = evaluator.trajectory(pooled_rejected)

            loss = trajectory_loss(scores_chosen, scores_rejected)

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(evaluator.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            batch_size_actual = scores_chosen[0].shape[0]
            correct += (scores_chosen[-1] > scores_rejected[-1]).sum().item()
            total += batch_size_actual

            if batch_idx % 50 == 0:
                acc = correct / total if total > 0 else 0
                margin = (scores_chosen[-1] - scores_rejected[-1]).mean().item()
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {total_loss/(batch_idx+1):.4f} | "
                    f"Acc: {acc:.4f} | "
                    f"Margin: {margin:.4f} | "
                    f"Chosen: {scores_chosen[-1].mean().item():.4f} | "
                    f"Rejected: {scores_rejected[-1].mean().item():.4f}"
                )

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"evaluator_epoch{epoch+1}.pt")
        torch.save(evaluator.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    print("Training complete.")

if __name__ == "__main__":
    train()
