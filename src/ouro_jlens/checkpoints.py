"""Lens-free exit divergence across Ouro checkpoints (base / Thinking / RLTT).

Answers, without fitting anything: does post-training widen the gap between what an
intermediate recurrent exit says and what the finished model says? That is section 3 of
RESULTS.md measured on each checkpoint. All three share Ouro's architecture, so the base
tokenizer is used throughout to guarantee identical input ids.

  python src/ouro_jlens/checkpoints.py --out artifacts/jlens/checkpoints
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jlens.hooks import ActivationRecorder
from jlens.vis import _ranks_of

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouro_jlens.evaluate import GREEDY_STEPS, greedy, is_correct  # noqa: E402
from ouro_jlens.evidence import atomic_write_json, file_record  # noqa: E402
from ouro_jlens.evaldata import load_items  # noqa: E402
from ouro_jlens.recurrent import OURO_SNAPSHOT, PROJECT_ROOT, load_ouro  # noqa: E402

HUB = PROJECT_ROOT / "artifacts" / "hf_cache" / "hub"
_THINKING_SNAPSHOTS = sorted((HUB / "models--ByteDance--Ouro-2.6B-Thinking" / "snapshots").glob("*"))
CHECKPOINTS = {
    "base": OURO_SNAPSHOT,
    "thinking": _THINKING_SNAPSHOTS[0] if len(_THINKING_SNAPSHOTS) == 1 else None,
    "rltt": PROJECT_ROOT / "models" / "ouro_rltt_local",
}
OPERATIONS = {"addition", "subtraction", "multiplication", "division", "mod", "squared"}


@torch.no_grad()
def measure(m, items, tokenizer) -> dict:
    """Per-checkpoint exit statistics at the readout position."""
    n_ut = m.n_ut
    kl = np.zeros((len(items), n_ut), np.float32)
    js = np.zeros((len(items), n_ut), np.float32)
    ent = np.zeros((len(items), n_ut), np.float32)
    rank_final = np.zeros((len(items), n_ut), np.int32)
    same = np.zeros((len(items), n_ut), bool)
    inter_top1 = np.zeros((len(items), n_ut), bool)
    correct = np.zeros(len(items), bool)
    for i, item in enumerate(items):
        ids = torch.tensor([item.token_ids], device=m.input_device)
        with ActivationRecorder(m.layers, at=[m.exit_index(u) for u in range(n_ut)]) as rec:
            m.forward(ids)
        h = torch.stack([rec.activations[m.exit_index(u)][0, -1] for u in range(n_ut)])
        logits = m.unembed(h).float()                              # [n_ut, vocab]
        lp = F.log_softmax(logits, -1)
        p_ = lp.exp()
        kl[i] = (p_ * (lp - lp[-1])).sum(-1).cpu().numpy()
        ent[i] = (-(p_ * lp).sum(-1)).cpu().numpy()
        # KL confounds "different content" with "different sharpness"; JS is bounded and
        # symmetric, and the rank of the final answer under exit k is scale-free entirely.
        mlog = ((p_ + p_[-1]) / 2).clamp_min(1e-12).log()
        js[i] = (0.5 * (p_ * (lp - mlog)).sum(-1) + 0.5 * (p_[-1] * (lp[-1] - mlog)).sum(-1)).cpu().numpy()
        top1 = logits.argmax(-1)
        rank_final[i] = _ranks_of(logits, top1[-1].view(1))[:, 0].cpu().numpy()
        same[i] = (top1 == top1[-1]).cpu().numpy()
        toks = [t for k in item.scorable for t in item.intermediate_tokens[k]
                if k not in OPERATIONS and not item.leaked[k]]
        if toks:
            r = _ranks_of(logits, torch.tensor(toks, device=logits.device)).min(1).values
            inter_top1[i] = (r == 0).cpu().numpy()
        correct[i] = is_correct(greedy(m, ids, GREEDY_STEPS), item.target)
    out = {"n_items": len(items), "model_correct": float(correct.mean())}
    for task in ("multihop", "order-ops"):
        sel = np.array([it.task == task for it in items])
        scored = sel & np.array([any(k not in OPERATIONS and not it.leaked[k] for k in it.scorable) for it in items])
        out[task] = {
            "n": int(sel.sum()),
            "kl_exit_k_to_exit_last_mean": kl[sel].mean(0).round(3).tolist(),
            "kl_median": np.median(kl[sel], 0).round(3).tolist(),
            "exit_top1_equals_final": same[sel].mean(0).round(3).tolist(),
            "js_to_final_mean": js[sel].mean(0).round(4).tolist(),
            "entropy_mean": ent[sel].mean(0).round(3).tolist(),
            "median_rank_of_final_top1": np.median(rank_final[sel], 0).round(1).tolist(),
            "intermediate_is_exit_top1": inter_top1[scored].mean(0).round(3).tolist(),
            "n_scored": int(scored.sum()),
            "model_correct": float(correct[sel].mean()),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/jlens/checkpoints")
    p.add_argument("--only", nargs="*", default=list(CHECKPOINTS))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(OURO_SNAPSHOT))
    results = {"schema_version": 1, "status": "RUNNING", "checkpoints": {}, "missing": []}
    items = None
    for name in args.only:
        path = CHECKPOINTS[name]
        if path is None or not Path(path).exists():
            print(f"{name}: MISSING at {path}", flush=True)
            results["missing"].append(name)
            continue
        t0 = time.perf_counter()
        m = load_ouro(path)
        m.tokenizer = tokenizer  # identical input ids across checkpoints
        if items is None:
            items = load_items(tokenizer, encode=lambda s: m.encode(s)[0].tolist())
        measured = measure(m, items, tokenizer)
        measured["path"] = str(path)
        model_file = Path(path) / "model.safetensors"
        if model_file.exists():
            measured["model_file"] = file_record(model_file)
        measured["seconds"] = round(time.perf_counter() - t0, 1)
        results["checkpoints"][name] = measured
        print(f"{name}: {json.dumps(measured['order-ops'])}", flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()
    results["status"] = "COMPLETE" if not results["missing"] else "INCOMPLETE"
    atomic_write_json(out / "exit_divergence.json", results)

    print("\n== KL(exit k || final), mean over items")
    for task in ("multihop", "order-ops"):
        print(f"  {task}")
        for name, r in results["checkpoints"].items():
            print(f"    {name:9s} KL {r[task]['kl_exit_k_to_exit_last_mean']}  JS {r[task]['js_to_final_mean']}")
            print(f"    {'':9s} entropy {r[task]['entropy_mean']}  rank(final top1) {r[task]['median_rank_of_final_top1']}")
            print(f"    {'':9s} exit==final {r[task]['exit_top1_equals_final']}  intermediate-is-top1 "
                  f"{r[task]['intermediate_is_exit_top1']}  acc {r[task]['model_correct']:.2f}")
    if results["missing"]:
        raise SystemExit(f"INCOMPLETE: missing required checkpoints {results['missing']}")


if __name__ == "__main__":
    main()
