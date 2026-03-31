import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from evaluator_calibrated import CalibratedEvaluator

# --- Configuration ---
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
EPOCHS = 3
LEARNING_RATE = 1e-4
WARMUP_STEPS = 200
CLASSIFICATION_WEIGHT = 0.5    # weight of BCE classification loss vs ranking loss
CHECKPOINT_DIR = "checkpoints_calibrated"
FEATURE_DIR = "/mnt/sandisk/ouro_features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Batching ---
def make_batch(examples):
    n_loops = len(examples[0]["chosen_states"])

    def pad_and_stack(states_list, masks_list):
        max_len = max(s.shape[1] for s in states_list)
        padded = []
        padded_masks = []
        for s, m in zip(states_list, masks_list):
            pad_len = max_len - s.shape[1]
            if pad_len > 0:
                s = F.pad(s, (0, 0, pad_len, 0))
                m = F.pad(m, (pad_len, 0), value=0)
            padded.append(s)
            padded_masks.append(m)
        return torch.cat(padded, dim=0), torch.cat(padded_masks, dim=0)

    chosen_batched = []
    rejected_batched = []
    chosen_masks = [ex["chosen_mask"] for ex in examples]
    rejected_masks = [ex["rejected_mask"] for ex in examples]

    for loop_idx in range(n_loops):
        c_states = [ex["chosen_states"][loop_idx] for ex in examples]
        r_states = [ex["rejected_states"][loop_idx] for ex in examples]
        c_padded, c_mask = pad_and_stack(c_states, chosen_masks)
        r_padded, r_mask = pad_and_stack(r_states, rejected_masks)
        chosen_batched.append(c_padded)
        rejected_batched.append(r_padded)

    return {
        "chosen_states": chosen_batched,
        "chosen_mask": c_mask,
        "rejected_states": rejected_batched,
        "rejected_mask": r_mask,
    }

# --- Loss ---
def combined_loss(score_chosen, score_rejected):
    """
    Two components:
    1. Ranking: -logsigmoid(score_chosen - score_rejected)
       → preserves ordering (chosen > rejected)
    2. Classification: BCE on individual scores
       → anchors polarity (chosen → positive, rejected → negative)
       → exploits the 78% flipped independent signal
    """
    # ranking component (what we've always used)
    ranking_loss = -F.logsigmoid(score_chosen - score_rejected).mean()

    # classification component — forces chosen positive, rejected negative
    scores = torch.cat([score_chosen, score_rejected], dim=0)
    labels = torch.cat([
        torch.ones_like(score_chosen),
        torch.zeros_like(score_rejected)
    ], dim=0)
    classification_loss = F.binary_cross_entropy_with_logits(scores, labels)

    return ranking_loss + CLASSIFICATION_WEIGHT * classification_loss

# --- Main Training ---
def train():
    meta = torch.load(os.path.join(FEATURE_DIR, "meta.pt"))
    save_mode = meta["save_mode"]
    assert save_mode == "raw", "Requires raw features with attention masks"
    total_examples = meta["total_examples"]
    num_chunks = meta["num_chunks"]
    chunk_size = meta["chunk_size"]

    print(f"Features: {total_examples} examples, {num_chunks} chunks")

    evaluator = CalibratedEvaluator().to(DEVICE)
    total_params = sum(p.numel() for p in evaluator.parameters())
    print(f"Evaluator params: {total_params:,}")

    optimizer = torch.optim.AdamW(evaluator.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    batches_per_epoch = total_examples // BATCH_SIZE
    total_steps = batches_per_epoch * EPOCHS // GRAD_ACCUM_STEPS

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_STEPS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - WARMUP_STEPS)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS]
    )

    print(f"Total steps: {total_steps} (warmup: {WARMUP_STEPS})")
    print(f"LR: {LEARNING_RATE}, effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Loss: ranking + {CLASSIFICATION_WEIGHT} * classification (BCE)")
    print("Starting training...")

    global_step = 0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        rank_correct = 0       # chosen > rejected (ranking accuracy)
        class_correct = 0      # chosen > 0 and rejected < 0 (classification accuracy)
        total = 0
        batch_count = 0
        optimizer.zero_grad()

        chunk_order = np.random.permutation(num_chunks)

        for chunk_idx in chunk_order:
            chunk_path = os.path.join(FEATURE_DIR, f"chunk_{chunk_idx:04d}.pt")
            chunk = torch.load(chunk_path)
            np.random.shuffle(chunk)

            for start in range(0, len(chunk), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(chunk))
                if end - start < BATCH_SIZE:
                    continue

                batch = make_batch(chunk[start:end])

                chosen_states = [s.to(DEVICE, dtype=torch.float32) for s in batch["chosen_states"]]
                rejected_states = [s.to(DEVICE, dtype=torch.float32) for s in batch["rejected_states"]]
                chosen_mask = batch["chosen_mask"].to(DEVICE)
                rejected_mask = batch["rejected_mask"].to(DEVICE)

                score_chosen = evaluator(chosen_states, chosen_mask)
                score_rejected = evaluator(rejected_states, rejected_mask)

                if batch_count == 0 and epoch == 0:
                    print(f"Num loop states: {len(chosen_states)}, shape: {chosen_states[0].shape}")

                loss = combined_loss(score_chosen, score_rejected)
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()

                if (batch_count + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(evaluator.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                total_loss += loss.item() * GRAD_ACCUM_STEPS
                batch_size_actual = score_chosen.shape[0]

                # ranking accuracy: chosen > rejected
                rank_correct += (score_chosen > score_rejected).sum().item()
                # classification accuracy: chosen > 0 AND rejected < 0
                class_correct += ((score_chosen > 0).sum() + (score_rejected < 0).sum()).item()
                total += batch_size_actual
                batch_count += 1

                if batch_count % 50 == 0:
                    rank_acc = rank_correct / total if total > 0 else 0
                    class_acc = class_correct / (total * 2) if total > 0 else 0
                    margin = (score_chosen - score_rejected).mean().item()
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch + 1} | Batch {batch_count}/{batches_per_epoch} | "
                        f"Loss: {total_loss / batch_count:.4f} | "
                        f"RankAcc: {rank_acc:.4f} | "
                        f"ClassAcc: {class_acc:.4f} | "
                        f"Margin: {margin:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"C: {score_chosen.mean().item():.3f} | "
                        f"R: {score_rejected.mean().item():.3f}"
                    )

            del chunk

        if batch_count % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(evaluator.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_rank_acc = rank_correct / total if total > 0 else 0
        epoch_class_acc = class_correct / (total * 2) if total > 0 else 0
        epoch_loss = total_loss / batch_count
        print(f"--- Epoch {epoch + 1} Summary: Loss={epoch_loss:.4f}, "
              f"RankAcc={epoch_rank_acc:.4f}, ClassAcc={epoch_class_acc:.4f} ---")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"calibrated_epoch{epoch + 1}.pt")
        torch.save({
            "model_state_dict": evaluator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "rank_accuracy": epoch_rank_acc,
            "class_accuracy": epoch_class_acc,
            "loss": epoch_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    print("Training complete.")

if __name__ == "__main__":
    train()
