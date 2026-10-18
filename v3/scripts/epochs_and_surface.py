"""RPE v3, analyses A and B: per-epoch fixed-order vs antisymmetrized accuracy of the pairwise evaluator, and
surface baselines, on the HH-RLHF test set.

One pass over the 8,552 test pairs with the settings of the July full-set audit (Ouro-RLTT, bf16, left padding,
max_length 768, early-exit threshold 1.0, batch 4, length-bucketed): loop states of each response are computed
once and scored by the epoch 1-5 checkpoints in both orders. Built-in check: epoch 2 must reproduce the audit's
strict antisymmetrized accuracy 0.6391 (per-pair scores in full_hh_antisymmetry_rows.csv).

Also saved per response: token count after truncation, character count, and the mean-pooled state of each of the
4 loops (fp16), for the surface baselines (shorter wins; smaller norm wins; directions fixed in advance from v1's
own observation, sec. 3.3) and for later probes. Writes into --out; resumable by shard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

from config import RLTT_SHARDS as SHARDS, THINKING, env, load_hh  # release copy: locations from config.py
from evaluator_core.pairwise_evaluator import PairwiseEvaluator  # noqa: E402

MODEL = env("RPE_RLTT_MODEL")
CKPT_DIR = env("RPE_CKPT_DIR")
EPOCH2_SHA = "3630c2092eca8db13239f763bc9c212f4b673866e47f811c3095efc57409ec96"
SHARD = 500


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--check-shards", action="store_true")
    ap.add_argument("--backbone", choices=["rltt", "thinking"], default="rltt")
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--early-exit", type=float, default=1.0)  # affects logits only; model.model always returns 4 loop states
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--ckpt-suffix", default=".pt")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.check_shards:
        for f, h in SHARDS.items():
            assert sha(MODEL / f) == h, f"model shard {f} differs from the audit's"
    if a.ckpt_suffix == ".pt":  # the original checkpoints (the Thinking run read tensor-identical state-only copies)
        assert sha(a.ckpt_dir / "pairwise_epoch2.pt") == EPOCH2_SHA

    from transformers import AutoModelForCausalLM, AutoTokenizer
    ds = load_hh("test")
    assert len(ds) == 8552
    n = len(ds) if a.max_pairs <= 0 else a.max_pairs
    order = sorted(range(n), key=lambda i: max(len(ds[i]["chosen"]), len(ds[i]["rejected"])))  # as the audit

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ref, kw = (str(MODEL), {}) if a.backbone == "rltt" else (THINKING[0], {"revision": THINKING[1]})
    model = AutoModelForCausalLM.from_pretrained(ref, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                 low_cpu_mem_usage=True, **kw).to(dev).eval()
    model.config.early_exit_threshold = a.early_exit
    cap: list = []
    hook = model.model.register_forward_hook(lambda m, i, o: cap.__setitem__(slice(None), [x.detach() for x in o[1]]))
    heads = {}
    for e in range(1, 6):
        h = PairwiseEvaluator().to(dev).eval()
        h.load_state_dict(torch.load(a.ckpt_dir / f"pairwise_epoch{e}{a.ckpt_suffix}", map_location=dev, weights_only=False)["model_state_dict"])
        heads[e] = h

    def encode(texts):
        t = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=a.max_length).to(dev)
        cap.clear()
        with torch.no_grad():
            model(**t, use_cache=False)
        states = [x.float() for x in cap]
        mask = t["attention_mask"].float()
        pooled = torch.stack([(s * mask[..., None]).sum(1) / mask.sum(1, keepdim=True) for s in states], 1)  # [b, 4, 2048]
        return states, mask, pooled

    t0 = time.time()
    for s0 in range(0, n, SHARD):
        path = a.out / f"shard_{s0:05d}.pt"
        if path.exists():
            continue
        idx_all = order[s0:s0 + SHARD]
        rec = {"dataset_index": [], "scores": {e: {"normal": [], "flipped": []} for e in heads},
               "tokens": {"chosen": [], "rejected": []}, "chars": {"chosen": [], "rejected": []},
               "pooled": {"chosen": [], "rejected": []}}
        for b0 in range(0, len(idx_all), a.batch_size):
            idx = idx_all[b0:b0 + a.batch_size]
            c_txt, r_txt = [ds[i]["chosen"] for i in idx], [ds[i]["rejected"] for i in idx]
            cs, cm, cp = encode(c_txt)
            rs, rm, rp = encode(r_txt)
            with torch.no_grad():
                for e, h in heads.items():
                    rec["scores"][e]["normal"] += h(cs, cm, rs, rm).view(-1).float().cpu().tolist()
                    rec["scores"][e]["flipped"] += h(rs, rm, cs, cm).view(-1).float().cpu().tolist()
            rec["dataset_index"] += idx
            for k, m, p, txt in (("chosen", cm, cp, c_txt), ("rejected", rm, rp, r_txt)):
                rec["tokens"][k] += m.sum(1).int().cpu().tolist()
                rec["chars"][k] += [len(x) for x in txt]
                rec["pooled"][k].append(p.half().cpu())
        for k in ("chosen", "rejected"):
            rec["pooled"][k] = torch.cat(rec["pooled"][k])
        torch.save(rec, path)
        done = min(s0 + SHARD, n)
        el = time.time() - t0
        print(f"[A] {done}/{n} pairs, {el / 60:.1f} min", flush=True)
    hook.remove()
    json.dump({"n_pairs": n, "settings": {"backbone": a.backbone, "max_length": a.max_length, "early_exit_threshold": a.early_exit,
                                          "dtype": "bfloat16", "padding": "left", "batch_size": a.batch_size},
               "epoch_checkpoints": {e: sha(a.ckpt_dir / f"pairwise_epoch{e}{a.ckpt_suffix}") for e in heads}},
              open(a.out / "run_info.json", "w"), indent=2)
    print("[A] done", flush=True)


if __name__ == "__main__":
    main()
