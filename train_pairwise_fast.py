import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from evaluator_pairwise import PairwiseEvaluator

# --- Configuration ---
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
EPOCHS = 5
LEARNING_RATE = 1e-4
WARMUP_STEPS = 200
CHECKPOINT_DIR = "checkpoints_pairwise"
FEATURE_DIR = "features"
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
def pairwise_preference_loss(score, target):
    directional_loss = -F.logsigmoid(target * score).mean()
    l2_reg = 1e-4 * score.pow(2).mean()
    return directional_loss + l2_reg

# --- Main Training ---
def train():
    meta = torch.load(os.path.join(FEATURE_DIR, "meta.pt"))
    save_mode = meta["save_mode"]
    assert save_mode == "raw", "Pairwise evaluator requires raw features with attention masks"
    total_examples = meta["total_examples"]
    num_chunks = meta["num_chunks"]
    chunk_size = meta["chunk_size"]

    print(f"Features: {total_examples} examples, {num_chunks} chunks")

    evaluator = PairwiseEvaluator().to(DEVICE)
    total_params = sum(p.numel() for p in evaluator.parameters())
    print(f"Evaluator params: {total_params:,}")

    optimizer = torch.optim.AdamW(evaluator.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    batches_per_epoch = total_examples // BATCH_SIZE
    total_steps = batches_per_epoch * EPOCHS // GRAD_ACCUM_STEPS

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_STEPS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - WARMUP_STEPS, eta_min=1e-6)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS]
    )

    print(f"Total steps: {total_steps} (warmup: {WARMUP_STEPS}, cosine: {total_steps - WARMUP_STEPS})")
    print(f"LR: {LEARNING_RATE}, effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Architecture: PAIRWISE v1 (original 70% architecture)")
    print("Starting training...")

    global_step = 0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        n_normal = 0
        n_flipped = 0
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

                flip = np.random.random() < 0.5

                if flip:
                    states_a, mask_a = rejected_states, rejected_mask
                    states_b, mask_b = chosen_states, chosen_mask
                    target = -1.0
                    n_flipped += 1
                else:
                    states_a, mask_a = chosen_states, chosen_mask
                    states_b, mask_b = rejected_states, rejected_mask
                    target = 1.0
                    n_normal += 1

                if batch_count == 0 and epoch == 0:
                    print(f"Num loop states: {len(chosen_states)}, shape: {chosen_states[0].shape}")

                score = evaluator(states_a, mask_a, states_b, mask_b)
                loss = pairwise_preference_loss(score, target)
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()

                if (batch_count + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(evaluator.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                total_loss += loss.item() * GRAD_ACCUM_STEPS

                batch_size_actual = score.shape[0]
                with torch.no_grad():
                    if target > 0:
                        correct += (score > 0).sum().item()
                    else:
                        correct += (score < 0).sum().item()
                total += batch_size_actual
                batch_count += 1

                if batch_count % 50 == 0:
                    acc = correct / total if total > 0 else 0
                    avg_score = score.mean().item()
                    lr = optimizer.param_groups[0]["lr"]
                    flip_pct = n_flipped / (n_normal + n_flipped) * 100
                    print(
                        f"Epoch {epoch + 1} | Batch {batch_count}/{batches_per_epoch} | "
                        f"Loss: {total_loss / batch_count:.4f} | "
                        f"Acc: {acc:.4f} | "
                        f"Score: {avg_score:+.4f} | "
                        f"Target: {target:+.0f} | "
                        f"LR: {lr:.2e} | "
                        f"Flips: {flip_pct:.0f}%"
                    )

            del chunk

        if batch_count % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(evaluator.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_acc = correct / total if total > 0 else 0
        epoch_loss = total_loss / batch_count
        flip_pct = n_flipped / (n_normal + n_flipped) * 100
        print(f"--- Epoch {epoch + 1} Summary: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, "
              f"Flips={flip_pct:.1f}% ---")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"pairwise_epoch{epoch + 1}.pt")
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
