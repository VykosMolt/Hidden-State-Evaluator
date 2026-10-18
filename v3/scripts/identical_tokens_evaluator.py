"""Evaluator test pairs that are token-identical at the evaluator's right truncation, and accuracy without them.

Rebuilds each response's input ids exactly as epochs_and_surface.py built them (Ouro-RLTT tokenizer, truncation=True,
right truncation, max_length 1,024 for the Ouro-Thinking run and 768 for the Ouro-RLTT run; left padding adds only pad
tokens and does not change the ids). Then:

  gate 1  the per-epoch order-independent accuracies and bootstrap intervals of epochs_<run>.json are reproduced
          exactly from the per-pair shards (same seed and draws as summarize_epochs.py);
  gate 2  the rebuilt token count of every response equals the count the run saved (attention-mask sum);
  gate 3  the number of token-identical pairs equals the published count (68 at 1,024; 125 at 768, where the
          version 2 audit's accuracy on the other pairs, ties counted as errors, must also be 0.6465 to 4 places).

It reports how the token-identical pairs relate to the pairs whose pooled states are exactly equal (the set that
summarize_epochs.py and identical_at_1024_thinking.json excluded), and the order-independent accuracy of all five
epochs with the token-identical pairs excluded, with a pair bootstrap over the remaining pairs (seed 20260710,
10,000 draws, as summarize_epochs.py).

Run (CPU): python identical_tokens_evaluator.py thinking   -> results/identical_at_1024_tokens_thinking.json
           python identical_tokens_evaluator.py rltt       -> results/identical_at_768_tokens_rltt.json
"""
import csv
import glob
import json
import sys

import numpy as np
import torch

from config import RESULTS, env, load_hh, tokenizer

RUN = {"thinking": ("A_thinking_1024", 1024, "epochs_thinking_1024.json", 68),
       "rltt": ("A_rltt", 768, "epochs_rltt.json", 125)}
backbone = sys.argv[1]
shard_dir, maxlen, saved_name, published_count = RUN[backbone]

sh = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(str(env("RPE_RUNS") / shard_dir / "shard_*.pt")))]
idx = np.concatenate([s["dataset_index"] for s in sh])
o = np.argsort(idx)
idx = idx[o]
assert (idx == np.arange(8552)).all()
pc = torch.cat([s["pooled"]["chosen"] for s in sh])[o].float()
pr = torch.cat([s["pooled"]["rejected"] for s in sh])[o].float()
state_same = (pc == pr).flatten(1).all(1).numpy()
ntok = {k: np.concatenate([s["tokens"][k] for s in sh])[o] for k in ("chosen", "rejected")}
scores = {e: (np.concatenate([s["scores"][e]["normal"] for s in sh])[o], np.concatenate([s["scores"][e]["flipped"] for s in sh])[o])
          for e in range(1, 6)}
corr = lambda m: np.where(m > 0, 1.0, np.where(m == 0, 0.5, 0.0))  # noqa: E731
C = {e: corr((n - f) / 2) for e, (n, f) in scores.items()}


def boot(mask):
    """summarize_epochs.py's bootstrap (seed 20260710, 10,000 draws of len(pairs) pairs) on the pairs in mask."""
    rng, m = np.random.default_rng(20260710), np.flatnonzero(mask)
    means = {e: np.empty(10000) for e in C}
    for i in range(10000):
        b = rng.integers(0, len(m), len(m))
        for e in C:
            means[e][i] = C[e][m][b].mean()
    return {e: [float(np.percentile(v, q)) for q in (2.5, 97.5)] for e, v in means.items()}


# gate 1: the saved per-epoch accuracies and intervals
saved = json.load(open(RESULTS / saved_name))
assert saved["n_identical_states"] == int(state_same.sum())
ci_all = boot(np.ones(8552, bool))
for e in C:
    s = saved["epochs"][str(e)]
    assert C[e].mean() == s["strict_antisym_acc"][0] and ci_all[e] == s["strict_antisym_acc"][1], (e, C[e].mean(), ci_all[e], s)
    assert C[e][~state_same].mean() == s["strict_antisym_acc_distinguishable"]
print(f"gate 1 passed: {saved_name} reproduced exactly (5 epochs, accuracy, interval, state-identical exclusion)")

# gate 2: rebuilt inputs have the saved token counts
ds, tok = load_hh("test"), tokenizer()
assert len(ds) == 8552 and tok.truncation_side == "right"
ids = {k: [tok(ds[int(i)][k], truncation=True, max_length=maxlen).input_ids for i in idx] for k in ("chosen", "rejected")}
for k in ids:
    assert (np.array([len(x) for x in ids[k]]) == ntok[k]).all(), k
print(f"gate 2 passed: rebuilt token counts equal the saved counts for all {2 * 8552} responses")

# gate 3: the published count
tok_same = np.array([a == b for a, b in zip(ids["chosen"], ids["rejected"])])
text_same = np.array([ds[int(i)]["chosen"] == ds[int(i)]["rejected"] for i in idx])
n_tok = int(tok_same.sum())
if n_tok != published_count:
    sys.exit(f"STOP: {n_tok} token-identical pairs at {maxlen} tokens, not the published {published_count}")
extra = {}
if backbone == "rltt":
    rows = {int(r["dataset_index"]): (float(r["normal_score"]), float(r["flipped_score"]))
            for r in csv.DictReader(open(env("RPE_AUDIT") / "full_hh_antisymmetry_rows.csv"))}
    a = np.array([rows[int(i)][0] - rows[int(i)][1] > 0 for i in idx], float)
    extra["v2_audit_epoch2_ties_wrong"] = {"all": float(a.mean()), "excluding_token_identical": float(a[~tok_same].mean())}
    assert round(extra["v2_audit_epoch2_ties_wrong"]["excluding_token_identical"], 4) == 0.6465
    th = RESULTS / "identical_at_1024_tokens_thinking.json"  # the Thinking run's 68, for comparison (run thinking first)
    if th.exists():
        m = np.ones(8552, bool)
        m[json.load(open(th))["token_identical_dataset_indices"]] = False
        extra["acc_excluding_thinking_1024_token_identical"] = {e: float(C[e][m].mean()) for e in C}
print(f"gate 3 passed: {n_tok} token-identical pairs at {maxlen} tokens (published {published_count})")

keep = ~tok_same
ci_keep = boot(keep)
epochs = {}
for e in C:
    n, f = scores[e]
    epochs[e] = {"acc_all": float(C[e].mean()), "ci_all": ci_all[e],
                 "acc_excluding_token_identical": float(C[e][keep].mean()), "ci_excluding_token_identical": ci_keep[e],
                 "change_points": 100 * float(C[e][keep].mean() - C[e].mean()),
                 "acc_excluding_state_identical_saved": saved["epochs"][str(e)]["strict_antisym_acc_distinguishable"],
                 "token_identical_pairs": {"exact_score_ties": int((n[tok_same] == f[tok_same]).sum()),
                                           "counted_correct": float(C[e][tok_same].sum()),
                                           "accuracy": float(C[e][tok_same].mean())}}
    print(e, {k: v for k, v in epochs[e].items() if k != "token_identical_pairs"}, epochs[e]["token_identical_pairs"])
out = {
    "backbone": backbone, "run": shard_dir, "max_length": maxlen, "truncation_side": "right",
    "tokenizer": "Ouro-RLTT local checkpoint (RPE_RLTT_MODEL)", "dataset": "Anthropic/hh-rlhf test",
    "dataset_revision": __import__("config").HH_REVISION, "n_pairs": 8552,
    "n_token_identical": n_tok, "token_identical_dataset_indices": idx[tok_same].tolist(),
    "n_token_identical_with_identical_texts": int((tok_same & text_same).sum()),
    "n_state_identical": int(state_same.sum()), "state_identical_dataset_indices": idx[state_same].tolist(),
    "n_state_identical_within_token_identical": int((state_same & tok_same).sum()),
    "n_state_identical_outside_token_identical": int((state_same & ~tok_same).sum()),
    "note": "state-identical = pooled states of chosen and rejected exactly equal (what summarize_epochs.py excluded); "
            "chosen and rejected are run as separate left-padded batches of 4, so identical inputs can meet different "
            "padding and batch companions; this is the likely (not tested) reason some token-identical pairs have "
            "slightly different bfloat16 states.",
    "bootstrap": {"seed": 20260710, "draws": 10000, "unit": "pair", "same_as": "summarize_epochs.py"},
    "epochs": epochs, **extra,
}
name = f"identical_at_{maxlen}_tokens_{backbone}.json"
json.dump(out, open(RESULTS / name, "w"), indent=1)
print("wrote", RESULTS / name)
