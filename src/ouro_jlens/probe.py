"""Task-specific supervised linear reference on a templated arithmetic family.

Prompts "(a + b) * c = " with a, b in 1..9 and c in 2..9; the latent
intermediate is a+b (17 classes). Split by held-out (a, b) pairs so a probe
cannot memorise operand pairs. Per location: multinomial logistic regression on
standardised residuals, C chosen on validation pairs, refit on train+val,
scored on test pairs. The Jacobian lens and logit lens are scored on the same
test prompts with the same 17-way candidate set (label score = max lens logit
over the label's single-token forms) so the comparison is like-for-like; the
full-vocab lens rank is stored too.

  python src/ouro_jlens/probe.py --lens artifacts/jlens/lens/exit3.pt --out artifacts/jlens/probe/run1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import jlens
from jlens.hooks import ActivationRecorder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouro_jlens.evaldata import single_token_ids, surface_forms  # noqa: E402
from ouro_jlens.evaluate import greedy, is_correct, stacked_jacobians  # noqa: E402
from ouro_jlens.recurrent import load_ouro  # noqa: E402

LABELS = list(range(2, 19))
C_GRID = (0.01, 0.1, 1.0)


def make_prompts() -> list[dict]:
    return [
        {"a": a, "b": b, "c": c, "prompt": f"({a} + {b}) * {c} = ", "label": a + b, "target": str((a + b) * c)}
        for a in range(1, 10) for b in range(1, 10) for c in range(2, 10)
    ]


def split_pairs(seed: int = 0) -> dict[str, set[tuple[int, int]]]:
    """Split by *unordered* operand pair. a+b is symmetric, so putting (3,5) in train and
    (5,3) in test would hand the probe the answer; both always land in the same split."""
    unordered = [(a, b) for a in range(1, 10) for b in range(a, 10)]
    rng = np.random.default_rng(seed)
    rng.shuffle(unordered)
    parts = {"test": unordered[:11], "val": unordered[11:17], "train": unordered[17:]}
    return {k: {(a, b) for x, y in v for a, b in ((x, y), (y, x))} for k, v in parts.items()}


@torch.no_grad()
def cache(m, prompts: list[dict]) -> tuple[torch.Tensor, list[bool]]:
    H = torch.empty(len(prompts), m.n_layers, m.d_model, dtype=torch.float16)
    correct = []
    for i, p in enumerate(prompts):
        ids = m.encode(p["prompt"])
        with ActivationRecorder(m.layers, at=range(m.n_layers)) as rec:
            m.forward(ids)
        H[i] = torch.stack([rec.activations[v][0, -1] for v in range(m.n_layers)]).half().cpu()
        correct.append(is_correct(greedy(m, ids, 3), p["target"]))
    return H, correct


@torch.no_grad()
def lens_candidate_ranks(m, H: torch.Tensor, labels: np.ndarray, J, label_tokens: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank of the true label among the 17 candidates scored by (a) the best of all its
    single-token forms and (b) one fixed form each, plus its full-vocab rank: each [n, 192].
    Labels differ in how many single-token forms they have (1 to 5), which favours the
    many-form labels under (a); (b) removes that advantage."""
    from jlens.vis import _ranks_of

    cand_rank = np.zeros((len(H), m.n_layers), np.int32)
    one_rank = np.zeros((len(H), m.n_layers), np.int32)
    vocab_rank = np.zeros((len(H), m.n_layers), np.int32)
    first = [toks[:1] for toks in label_tokens]  # one form per label: no form-count advantage
    for i in range(len(H)):
        h = H[i].to(m.input_device).float()
        logits = m.unembed(h if J is None else torch.einsum("vde,ve->vd", J, h)).float()
        true = LABELS.index(int(labels[i]))
        for out, toks in ((cand_rank, label_tokens), (one_rank, first)):
            scores = torch.stack([logits[:, t].max(-1).values for t in toks], dim=1)  # [192, 17]
            out[i] = (scores > scores[:, true : true + 1]).sum(1).cpu().numpy()
        vocab_rank[i] = _ranks_of(logits, torch.tensor(label_tokens[true], device=logits.device)).min(1).values.cpu().numpy()
    return cand_rank, one_rank, vocab_rank


def fit_probes(H: np.ndarray, labels: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Per-location probe rank of the true label on test prompts: [n_test, 192]."""
    tr, va, te = (split == s for s in ("train", "val", "test"))
    ranks = np.zeros((te.sum(), H.shape[1]), np.int32)
    chosen = []
    for v in range(H.shape[1]):
        X = H[:, v].astype(np.float32)
        scaler = StandardScaler().fit(X[tr])
        Xs = scaler.transform(X)
        best = max(C_GRID, key=lambda C: LogisticRegression(C=C, max_iter=2000).fit(Xs[tr], labels[tr]).score(Xs[va], labels[va]))
        scaler = StandardScaler().fit(X[tr | va])
        Xs = scaler.transform(X)
        clf = LogisticRegression(C=best, max_iter=2000).fit(Xs[tr | va], labels[tr | va])
        proba = np.zeros((te.sum(), len(LABELS)))  # labels unseen in training keep probability 0
        proba[:, [LABELS.index(c) for c in clf.classes_]] = clf.predict_proba(Xs[te])
        true_p = proba[np.arange(len(proba)), [LABELS.index(l) for l in labels[te]]]
        ranks[:, v] = (proba > true_p[:, None]).sum(1)
        chosen.append(best)
        if v % 24 == 0:
            print(f"  probe location {v}/{H.shape[1]}: C={best} test top1={np.mean(ranks[:, v] == 0):.3f}", flush=True)
    return ranks, chosen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lens", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts = make_prompts()
    parts = split_pairs()
    split = np.array([next(k for k, s in parts.items() if (q["a"], q["b"]) in s) for q in prompts])
    labels = np.array([q["label"] for q in prompts])

    te = split == "test"
    gpu_cache = out / "gpu_cache.npz"
    if gpu_cache.exists():
        c = np.load(gpu_cache)
        assert str(c["lens"]) == args.lens, f"{gpu_cache} was built from lens {c['lens']}, not {args.lens}"
        H, correct = torch.from_numpy(c["H"]), c["correct"]
        jl_cand, jl_one, jl_vocab = c["jl_cand"], c["jl_one"], c["jl_vocab"]
        ll_cand, ll_one, ll_vocab = c["ll_cand"], c["ll_one"], c["ll_vocab"]
    else:
        m = load_ouro()
        t0 = time.perf_counter()
        H, correct = cache(m, prompts)
        print(f"cached {H.shape} in {time.perf_counter()-t0:.0f}s; model correct on (a+b)*c: {np.mean(correct):.3f}", flush=True)
        label_tokens = [single_token_ids(m.tokenizer, surface_forms(str(s))) for s in LABELS]
        assert all(label_tokens)
        H_test = H[torch.from_numpy(te)]
        lens = jlens.JacobianLens.load(args.lens)
        J = stacked_jacobians(m, lens, m.exit_index(m.n_ut - 1))
        jl_cand, jl_one, jl_vocab = lens_candidate_ranks(m, H_test, labels[te], J, label_tokens)
        del J
        ll_cand, ll_one, ll_vocab = lens_candidate_ranks(m, H_test, labels[te], None, label_tokens)
        del m
        torch.cuda.empty_cache()
        np.savez(gpu_cache, H=H.numpy(), correct=np.array(correct), lens=args.lens,
                 jl_cand=jl_cand, jl_one=jl_one, jl_vocab=jl_vocab,
                 ll_cand=ll_cand, ll_one=ll_one, ll_vocab=ll_vocab)

    t0 = time.perf_counter()
    probe_rank, chosen_C = fit_probes(H.numpy(), labels, split)
    print(f"probes fitted in {time.perf_counter()-t0:.0f}s", flush=True)

    np.savez_compressed(
        out / "arrays.npz", probe_rank=probe_rank, jlens_cand_rank=jl_cand, jlens_vocab_rank=jl_vocab,
        jlens_oneform_rank=jl_one, logitlens_oneform_rank=ll_one,
        logitlens_cand_rank=ll_cand, logitlens_vocab_rank=ll_vocab, test_labels=labels[te],
        test_correct=np.array(correct)[te], chosen_C=np.array(chosen_C),
    )
    meta = {"n_prompts": len(prompts), "n_train": int((split == "train").sum()), "n_val": int((split == "val").sum()),
            "n_test": int(te.sum()), "model_accuracy_all": float(np.mean(correct)), "lens": args.lens,
            "labels": LABELS, "C_grid": C_GRID}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    for name, r in [("probe", probe_rank), ("jlens", jl_cand), ("logitlens", ll_cand),
                    ("jlens 1-form", jl_one), ("logitlens 1-form", ll_one)]:
        acc = (r == 0).mean(0)
        best = int(acc.argmax())
        print(f"{name}: best candidate-set top1 {acc[best]:.3f} at ut{best // 48} L{best % 48}; "
              f"per-loop max {[round(float(acc[u*48:(u+1)*48].max()), 3) for u in range(4)]}")


if __name__ == "__main__":
    main()
