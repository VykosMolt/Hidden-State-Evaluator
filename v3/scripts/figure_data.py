"""Small data files for the figures, each checked against the published result JSONs before it is written.

1. results/evaluator_pair_scores_thinking.npz: per-pair evaluator scores, preferred reply first ('normal') and second
   ('flipped'), for all five epochs, on the 8,552 HH-RLHF test pairs (Ouro-Thinking, 1,024 tokens), ordered by dataset
   index as in summarize_epochs.py. Asserts that epoch 2's fixed-order and order-independent accuracies match
   epochs_thinking_1024.json.
2. results/first_difference_heldout.json: for the 8,000 held-out pairs of the pair-disjoint probes, the token position
   at which the chosen and rejected texts first differ, and their full token lengths. Asserts that the number of pairs
   that are identical within the first 384 tokens equals the 649 found by the repro run (repro_thinking.json).

Run (CPU only, after `source config.sh`): python figure_data.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch

from config import RESULTS as RES, env, load_hh, tokenizer  # release copy: locations from config.py

RUNS = env("RPE_RUNS")
PROBES = env("RPE_SPLIT")
TOKENIZER = "Ouro-RLTT local checkpoint (RPE_RLTT_MODEL)"  # vocab and merges identical to Ouro-Thinking's
MAXLEN = 384

# 1. evaluator per-pair scores
sh = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(str(RUNS / "A_thinking_1024/shard_*.pt")))]
di = np.concatenate([s["dataset_index"] for s in sh])
o = np.argsort(di)
epochs = sorted(sh[0]["scores"])
normal = np.stack([np.concatenate([s["scores"][e]["normal"] for s in sh])[o] for e in epochs], 1).astype(np.float32)
flipped = np.stack([np.concatenate([s["scores"][e]["flipped"] for s in sh])[o] for e in epochs], 1).astype(np.float32)
ref = json.load(open(RES / "epochs_thinking_1024.json"))["epochs"]["2"]
n2, f2 = normal[:, epochs.index(2)], flipped[:, epochs.index(2)]
a = np.where(n2 - f2 > 0, 1.0, np.where(n2 - f2 == 0, 0.5, 0.0))
assert len(di) == 8552 and len(set(di.tolist())) == 8552
assert abs((n2 > 0).mean() - ref["fixed_order_acc"]) < 1e-9, ((n2 > 0).mean(), ref["fixed_order_acc"])
assert abs(a.mean() - ref["strict_antisym_acc"][0]) < 1e-9, (a.mean(), ref["strict_antisym_acc"][0])
assert abs(((n2 > 0) & (f2 > 0)).mean() - ref["both_orders_prefer_first"]) < 1e-9
np.savez_compressed(RES / "evaluator_pair_scores_thinking.npz", dataset_index=di[o], epochs=np.array(epochs),
                    normal=normal, flipped=flipped)
print("wrote evaluator_pair_scores_thinking.npz", normal.shape, "epoch-2 checks pass")

# 2. first difference on the held-out probe pairs
ds = load_hh("train")
idx = json.load(open(PROBES / "hh_train_indices_40000.json"))["indices"]
ev = [int(x) for x in torch.load(PROBES / "hh_thinking_40000_pair_disjoint_probe.pt", map_location="cpu",
                                 weights_only=False)["eval_positions"]]
tok = tokenizer()
rows = []
for p in ev:
    c, r = tok(ds[idx[p]]["chosen"]).input_ids, tok(ds[idx[p]]["rejected"]).input_ids
    same = c == r  # duplicate pairs: the two texts are identical, so there is no first difference
    k = None if same else next((i for i, (x, y) in enumerate(zip(c, r)) if x != y), min(len(c), len(r)))
    rows.append({"position": p, "dataset_index": idx[p], "texts_identical": same, "first_difference": k,
                 "len_chosen": len(c), "len_rejected": len(r), "identical_at_384_right": c[:MAXLEN] == r[:MAXLEN]})
n_ident = sum(x["identical_at_384_right"] for x in rows)
n_same = sum(x["texts_identical"] for x in rows)
n_beyond = sum(x["first_difference"] is not None and x["first_difference"] >= MAXLEN for x in rows)
want = json.load(open(RES / "repro_thinking.json"))["summary"]["n_identical"]
assert len(rows) == 8000 and n_ident == want == n_same + n_beyond, (len(rows), n_ident, n_same, n_beyond, want)
assert n_same == json.load(open(RES / "probes_thinking.json"))["n_eval_identical_after_left_truncation"], n_same
json.dump({"n_pairs": len(rows), "max_length": MAXLEN, "n_identical_at_384_right": n_ident, "n_texts_identical": n_same,
           "n_first_difference_at_or_beyond_384": n_beyond, "tokenizer": TOKENIZER, "rows": rows},
          open(RES / "first_difference_heldout.json", "w"))
print(f"wrote first_difference_heldout.json: {n_ident}/{len(rows)} identical at {MAXLEN} (matches repro) = "
      f"{n_beyond} that differ only after token {MAXLEN} + {n_same} with identical texts")
