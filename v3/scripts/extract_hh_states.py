"""RPE v3, analysis C (GPU part): mean-pooled loop-boundary states for the 40k HH pairs of the pair-disjoint probes.

Same model loading, hook, pooling (masked mean over the 4 loop boundaries of model.model's output[1]), bf16 forward,
384-token limit and pair selection as the published probes (powered_hh_pair_disjoint_probe.py), with two modes:

  left   all 40,000 pairs, truncated from the LEFT so the final responses (where chosen and rejected differ) survive;
         pairs batched 8 at a time with chosen and rejected adjacent, length-bucketed. Saves per response the pooled
         states [4, 2048] (fp16), the token count after truncation and the full token count.
  repro  the published protocol exactly (right truncation, one pair = one batch of 2), on the held-out pairs that are
         token-identical at 384 tokens plus a fixed random sample of others. Checks that identical pairs give a
         difference of exactly 0 and that the published probe weights then score them 0 (counted wrong by `> 0`).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from config import THINKING, env, load_hh  # release copy: locations from config.py

PROBES = env("RPE_SPLIT")
RLTT = str(env("RPE_RLTT_MODEL"))
SPECS = {"thinking": THINKING, "rltt": (RLTT, None)}
MAXLEN, SHARD = 384, 2000


def masked_mean(x, mask):  # as published
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp_min(1)


def load(backbone):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RLTT, trust_remote_code=True,
                                        local_files_only=True)  # the published probes used this tokenizer for all
    tok.padding_side = "left"
    ref, rev = SPECS[backbone]
    kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    if rev:
        kw["revision"] = rev
    model = AutoModelForCausalLM.from_pretrained(ref, **kw).to("cuda").eval()
    model.config.early_exit_threshold = 1.0
    for p in model.parameters():
        p.requires_grad_(False)
    cap = []
    model.model.register_forward_hook(lambda m, i, o: cap.__setitem__(slice(None), [h.detach() for h in o[1]]))

    def pooled(texts):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to("cuda")
        cap.clear()
        with torch.inference_mode():
            model(**enc, use_cache=False)
        assert len(cap) == 4
        return torch.stack([masked_mean(h, enc["attention_mask"]) for h in cap], 1), enc["attention_mask"].sum(1)
    return tok, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(SPECS), required=True)
    ap.add_argument("--mode", choices=["left", "repro"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs-per-batch", type=int, default=8)
    a = ap.parse_args()
    ds = load_hh("train")
    idx = json.load(open(PROBES / "hh_train_indices_40000.json"))["indices"]
    probe = torch.load(PROBES / f"hh_{a.backbone}_40000_pair_disjoint_probe.pt", map_location="cpu", weights_only=False)
    ev = [int(x) for x in probe["eval_positions"]]
    tok, pooled = load(a.backbone)
    out = a.out / f"{a.backbone}_{a.mode}"
    out.mkdir(parents=True, exist_ok=True)

    if a.mode == "repro":
        tok.truncation_side = "right"
        ident = [p for p in ev if tok(ds[idx[p]]["chosen"], truncation=True, max_length=MAXLEN).input_ids ==
                 tok(ds[idx[p]]["rejected"], truncation=True, max_length=MAXLEN).input_ids]
        rest = sorted(set(ev) - set(ident))
        others = sorted(np.random.default_rng(20261003).choice(rest, 351, replace=False).tolist())
        w = probe["weight"].float()
        rows = []
        for p in ident + others:
            pl, _ = pooled([ds[idx[p]]["chosen"], ds[idx[p]]["rejected"]])  # one pair per batch, as published
            d = (pl[0] - pl[1]).flatten().half().float().cpu()  # published saved diffs in fp16
            rows.append({"position": p, "identical": p in ident, "diff_abs_max": float(d.abs().max()),
                         "score": float(d @ w), "counted_correct_published": bool(float(d @ w) > 0)})
        ri = [r for r in rows if r["identical"]]
        summary = {"backbone": a.backbone, "n_identical": len(ri), "n_others": len(others),
                   "identical_with_exactly_zero_diff": sum(r["diff_abs_max"] == 0 for r in ri),
                   "identical_counted_correct_by_published_rule": sum(r["counted_correct_published"] for r in ri),
                   "others_accuracy_published_rule": float(np.mean([r["counted_correct_published"] for r in rows if not r["identical"]]))}
        json.dump({"summary": summary, "rows": rows}, open(out / "repro.json", "w"), indent=1)
        print(json.dumps(summary, indent=2), flush=True)
        return

    tok.truncation_side = "left"
    order = sorted(range(len(idx)), key=lambda p: max(len(ds[idx[p]]["chosen"]), len(ds[idx[p]]["rejected"])))
    t0 = time.time()
    for s0 in range(0, len(order), SHARD):
        path = out / f"shard_{s0:05d}.pt"
        if path.exists():
            continue
        pos = order[s0:s0 + SHARD]
        P = torch.zeros(len(pos), 2, 4, 2048, dtype=torch.float16)
        ntok = torch.zeros(len(pos), 2, dtype=torch.int32)
        for b0 in range(0, len(pos), a.pairs_per_batch):
            batch = pos[b0:b0 + a.pairs_per_batch]
            texts = [t for p in batch for t in (ds[idx[p]]["chosen"], ds[idx[p]]["rejected"])]
            pl, n = pooled(texts)
            P[b0:b0 + len(batch)] = pl.view(len(batch), 2, 4, 2048).half().cpu()
            ntok[b0:b0 + len(batch)] = n.view(len(batch), 2).int().cpu()
        full = torch.tensor([[len(tok(ds[idx[p]][k]).input_ids) for k in ("chosen", "rejected")] for p in pos])
        torch.save({"positions": pos, "dataset_indices": [idx[p] for p in pos], "pooled": P, "tokens": ntok,
                    "tokens_untruncated": full, "backbone": a.backbone, "truncation_side": "left", "max_length": MAXLEN}, path)
        done = min(s0 + SHARD, len(order))
        el = time.time() - t0
        print(f"[C {a.backbone}] {done}/{len(order)} pairs, {el / 60:.1f} min, {done / el:.2f} pairs/s", flush=True)
    print(f"[C {a.backbone}] done", flush=True)


if __name__ == "__main__":
    main()
