"""Relational test (protocol: analysis/relational_test_20261018/PROTOCOL.md, hashed before any fit): a pair model P
against a decomposable model D of matched capacity, on the published probe pairs, split and features.

    python relational_test.py gate --out DIR                       # gates 1 and 2 of the protocol
    python relational_test.py run --analysis thinking_last --out DIR   # primary; also thinking_all4, rltt_last
    python relational_test.py run --analysis thinking_last --out DIR --smoke   # code check only (tiny grid)

CPU only. Needs RPE_RUNS, RPE_SPLIT and RPE_RESULTS (for probe_pairs_*_converged.csv).
"""
import argparse, copy, csv, glob, hashlib, itertools, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fit_probes as fp
from config import RESULTS, env

ANALYSES = {"thinking_last": ("thinking_left", slice(3, 4), "thinking_converged", "relational_final"),
            "thinking_all4": ("thinking_left", slice(0, 4), "thinking_converged", "relational_all4"),
            "rltt_last": ("rltt_left_cloud", slice(3, 4), "rltt_left_converged", "relational_final")}
GRID = list(itertools.product([64, 256], [1, 2], [1e-4, 1e-3, 1e-2], [0.0, 0.2]))  # width, depth, weight decay, dropout
SEL_SEEDS, FINAL_SEEDS = [0, 1, 2], [100, 101, 102, 103, 104]
PCA_DIM, LR, BATCH, MAX_EPOCHS, PATIENCE, BOOT_SEED = 512, 1e-3, 256, 40, 5, 20261018
PROTOCOL_SHA = "d81ecafd7e7e59e6396fd0ee20c4932acdb7773a6e5d76c140e4b0b4713cb6f5"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(feat, sl):
    sh = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(str(env("RPE_RUNS") / feat / "shard_*.pt")))]
    pos = np.concatenate([s["positions"] for s in sh]); o = np.argsort(pos); assert (pos[o] == np.arange(40000)).all()
    dsi = np.concatenate([s["dataset_indices"] for s in sh])[o]
    X = torch.cat([s["pooled"] for s in sh])[o][:, :, sl].flatten(2).float()  # [40000, 2, d], chosen first
    probe = torch.load(env("RPE_SPLIT") / "hh_thinking_40000_pair_disjoint_probe.pt", map_location="cpu", weights_only=False)
    tr, ev = np.array(probe["train_positions"], int), np.array(probe["eval_positions"], int)
    val = np.array([int.from_bytes(hashlib.sha256(str(int(dsi[p])).encode()).digest()[:8], "big") % 10 == 0 for p in tr])
    return X, dsi, tr, ev, tr[~val], tr[val]


def preprocess(X, fit):
    """Standardize with fit-set replies, then project on their top PCA_DIM principal components (float64, no whitening)."""
    R = X[fit].reshape(-1, X.shape[-1]).double()
    mu, sd = R.mean(0), R.std(0) + 1e-6
    Z = (R - mu) / sd
    evals, evecs = torch.linalg.eigh(Z.T @ Z / len(Z))
    V = evecs[:, -PCA_DIM:].flip(1)                                  # descending variance
    V = V * torch.sign(V[V.abs().argmax(0), torch.arange(PCA_DIM)])   # fixed sign convention
    explained = float(evals[-PCA_DIM:].sum() / evals.sum())
    Y = torch.stack([((X[:, k].double() - mu) / sd) @ V for k in (0, 1)], 1).float()
    return Y, {"mean": mu.float(), "std": sd.float(), "components": V.float()}, explained


def mlp(inp, width, depth, dropout):
    layers, d = [], inp
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]; d = width
    return nn.Sequential(*layers, nn.Linear(d, 1))


class Decomposable(nn.Module):
    def __init__(self, d, width, depth, dropout):
        super().__init__(); self.f = mlp(d, width, depth, dropout)
    def f1(self, a):
        return self.f(a)[:, 0]
    def train_score(self, a, b):
        u = self.f1(torch.cat([a, b])); return u[:len(a)] - u[len(a):]
    def score(self, a, b):  # evaluation: f on each reply set separately, so s(B, A) = -s(A, B) and s = f(A) - f(B) exactly
        return self.f1(a) - self.f1(b)


class Pair(nn.Module):
    def __init__(self, d, width, depth, dropout):
        super().__init__(); self.g = mlp(4 * d, width, depth, dropout)
    def g1(self, a, b):
        return self.g(torch.cat([a, b, a - b, a * b], 1))[:, 0]
    def train_score(self, a, b):
        u = self.g(torch.cat([torch.cat([a, b, a - b, a * b], 1), torch.cat([b, a, b - a, b * a], 1)]))[:, 0]
        return 0.5 * (u[:len(a)] - u[len(a):])
    def score(self, a, b):
        return 0.5 * (self.g1(a, b) - self.g1(b, a))


MODELS = {"D": Decomposable, "P": Pair}
n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
acc = lambda s: float(np.where(s > 0, 1.0, np.where(s == 0, 0.5, 0.0)).mean())  # noqa: E731
correct = lambda s: np.where(s > 0, 1.0, np.where(s == 0, 0.5, 0.0))  # noqa: E731


def train(kind, setting, seed, Y, fit, va, max_epochs=MAX_EPOCHS):
    width, depth, wd, dropout = setting
    torch.manual_seed(seed)
    model = MODELS[kind](Y.shape[-1], width, depth, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)
    gen = torch.Generator().manual_seed(seed)
    A, B, Av, Bv = Y[fit, 0], Y[fit, 1], Y[va, 0], Y[va, 1]
    best, best_state, best_ep, bad, t0 = -1.0, None, -1, 0, time.time()
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(A), generator=gen)
        for i in range(0, len(A), BATCH):
            j = perm[i:i + BATCH]
            loss = F.softplus(-model.train_score(A[j], B[j])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = acc(model.score(Av, Bv).numpy())
        if v > best:
            best, best_state, best_ep, bad = v, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    return model, {"val_acc": best, "best_epoch": best_ep + 1, "epochs_run": ep + 1, "seconds": round(time.time() - t0, 1)}


def held_out(model, Y, ev):
    with torch.no_grad():
        return model.score(Y[ev, 0], Y[ev, 1]).double().numpy()


def gate(out):
    """Gate 1: exact antisymmetry and decomposability of D, exact antisymmetry of P (random models, eval mode).
    Gate 2: this code's data loading and split reproduce the converged linear relational last-loop accuracy."""
    X, dsi, tr, ev, fit, va = load("thinking_left", slice(3, 4))
    res = {"n_fit": len(fit), "n_val": len(va), "n_eval": len(ev), "gate1": [], "gate2": {}}
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(500, 64, generator=g), torch.randn(500, 64, generator=g)
    c = torch.randn(500, 64, generator=g)
    ok = True
    for kind, (width, depth, dropout) in itertools.product("DP", [(64, 1, 0.0), (256, 2, 0.2)]):
        torch.manual_seed(1); m = MODELS[kind](64, width, depth, dropout).eval()
        with torch.no_grad():
            sab, sba, sbc, sac = m.score(a, b), m.score(b, a), m.score(b, c), m.score(a, c)
            r = {"model": kind, "width": width, "depth": depth, "dropout": dropout,
                 "antisymmetric_exact": bool((sab == -sba).all()),
                 "max_abs_cycle_residual": float((sab + sbc - sac).abs().max()), "max_abs_score": float(sab.abs().max())}
            if kind == "D":
                r["decomposable_exact"] = bool((sab == m.f1(a) - m.f1(b)).all())
                ok &= r["decomposable_exact"] and r["max_abs_cycle_residual"] < 1e-5
            ok &= r["antisymmetric_exact"]
        res["gate1"].append(r); log("gate1", r)  # P's cycle residual is reported (nonzero: not decomposable), not gated
    fp.CONVERGED, fp.DTYPE = True, torch.float64
    D = X[:, 0].double() - X[:, 1].double()
    w = fp.relational(D[tr])
    a2 = acc((D[ev] @ w).numpy())
    want = json.load(open(RESULTS / "probes_thinking_converged.json"))["acc"]["relational_final"]["all_ties_half"][0]
    res["gate2"] = {"linear_relational_last_loop_heldout_acc": a2, "probes_thinking_converged": want, "equal": a2 == want}
    ok &= a2 == want
    log("gate2", res["gate2"])
    res["passed"] = bool(ok)
    json.dump(res, open(out / "gate.json", "w"), indent=1)
    log("GATES", "PASSED" if ok else "FAILED")
    return ok


def mlp_params(inp, w, depth):
    return (inp + 1) * w + (depth - 1) * (w + 1) * w + w + 1


def matched_width(depth, d, target):
    """Largest P width whose parameter count does not exceed target."""
    assert mlp_params(4 * d, 7, 2) == n_params(mlp(4 * d, 7, 2, 0.0)) and mlp_params(4 * d, 7, 1) == n_params(mlp(4 * d, 7, 1, 0.0))
    return max(w for w in range(1, 4097) if mlp_params(4 * d, w, depth) <= target)


def run(analysis, out, smoke=False):
    feat, sl, tag, lin_col = ANALYSES[analysis]
    grid, sel_seeds, final_seeds, max_ep = (GRID[:2], [0], [100], 2) if smoke else (GRID, SEL_SEEDS, FINAL_SEEDS, MAX_EPOCHS)
    t_start = time.time()
    X, dsi, tr, ev, fit, va = load(feat, sl)
    Y, pre, explained = preprocess(X, fit)
    log(analysis, f"fit {len(fit)}, val {len(va)}, eval {len(ev)}; PCA {PCA_DIM} of {X.shape[-1]} dims explains {explained:.4f}")
    res = {"analysis": analysis, "protocol_sha256": PROTOCOL_SHA, "smoke": smoke, "features": feat, "dims": int(X.shape[-1]),
           "n_fit": len(fit), "n_val": len(va), "n_eval": len(ev), "pca_dim": PCA_DIM, "pca_explained_variance": explained,
           "settings": {"lr": LR, "batch": BATCH, "max_epochs": max_ep, "patience": PATIENCE, "optimizer": "AdamW",
                        "selection_seeds": sel_seeds, "final_seeds": final_seeds, "grid": "width x depth x weight_decay x dropout"},
           "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(), "selection": {}, "deviations": []}
    scores, weights, selected = {}, {"preprocessing": pre}, {}
    for kind in "DP":
        table = []
        for s in grid:
            runs = [train(kind, s, seed, Y, fit, va, max_ep)[1] for seed in sel_seeds]
            table.append({"setting": dict(zip(("width", "depth", "weight_decay", "dropout"), s)),
                          "mean_val_acc": float(np.mean([r["val_acc"] for r in runs])), "runs": runs,
                          "n_params": n_params(MODELS[kind](Y.shape[-1], s[0], s[1], s[3]))})
            log(kind, s, f"mean val {table[-1]['mean_val_acc']:.5f}", [r["best_epoch"] for r in runs])
        i = max(range(len(table)), key=lambda k: (table[k]["mean_val_acc"], -k))  # first in grid order on ties
        selected[kind] = grid[i]
        res["selection"][kind] = {"table": table, "selected": table[i]["setting"], "selected_n_params": table[i]["n_params"]}
    dsel = selected["D"]
    pw = matched_width(selected["P"][1], Y.shape[-1], res["selection"]["D"]["selected_n_params"])
    variants = {"D": selected["D"], "P": selected["P"], "P_matched": (pw, selected["P"][1], selected["P"][2], selected["P"][3])}
    res["final"] = {}
    for name, s in variants.items():
        kind = name[0]
        per, states = [], []
        for seed in final_seeds:
            m, info = train(kind, s, seed, Y, fit, va, max_ep)
            sc = held_out(m, Y, ev)
            scores[f"{name}_seed{seed}"] = sc
            per.append({**info, "seed": seed, "heldout_acc": acc(sc)}); states.append(m.state_dict())
            log(name, s, seed, per[-1])
        scores[name] = np.mean([scores[f"{name}_seed{seed}"] for seed in final_seeds], 0)
        weights[name] = states
        res["final"][name] = {"setting": dict(zip(("width", "depth", "weight_decay", "dropout"), s)),
                              "n_params": n_params(MODELS[kind](Y.shape[-1], s[0], s[1], s[3])), "seeds": per,
                              "heldout_acc_seed_mean": float(np.mean([p["heldout_acc"] for p in per])),
                              "heldout_acc_seed_sd": float(np.std([p["heldout_acc"] for p in per], ddof=1)) if len(per) > 1 else None}
    # linear references
    rows = {int(r["position"]): float(r[lin_col]) for r in csv.DictReader(open(RESULTS / f"probe_pairs_{tag}.csv"))}
    scores["linear_published_converged"] = np.array([rows[int(p)] for p in ev])
    fp.CONVERGED, fp.DTYPE = True, torch.float64
    Yd = Y.double()
    wl = fp.relational(Yd[fit, 0] - Yd[fit, 1], "linear_pca")
    scores["linear_pca"] = ((Yd[ev, 0] - Yd[ev, 1]) @ wl).numpy()
    weights["linear_pca"] = wl.float()
    res["linear_pca_fit"] = fp.FITS["linear_pca"]
    # accuracies, paired differences, verdict
    rng = np.random.default_rng(BOOT_SEED)
    Bs = [rng.integers(0, len(ev), len(ev)) for _ in range(10000)]
    ci = lambda x: [float(np.percentile([x[b].mean() for b in Bs], q)) for q in (2.5, 97.5)]  # noqa: E731
    C = {k: correct(v) for k, v in scores.items()}
    res["heldout"] = {k: {"acc": float(C[k].mean()), "ci95": ci(C[k])} for k in ("D", "P", "P_matched", "linear_published_converged", "linear_pca")}
    res["paired"] = {}
    for a_, b_ in (("P", "D"), ("P_matched", "D"), ("P", "linear_published_converged"), ("D", "linear_published_converged"),
                   ("P", "linear_pca"), ("D", "linear_pca")):
        d = C[a_] - C[b_]
        res["paired"][f"{a_} - {b_}"] = {"diff": float(d.mean()), "ci95": ci(d)}
    lo, hi = res["paired"]["P - D"]["ci95"]
    res["verdict"] = "RELATIONAL_ADVANTAGE" if lo > 0 else ("NO_RELATIONAL_ADVANTAGE_ABOVE_1PT" if hi < 0.01 else "UNRESOLVED")
    res["runtime_seconds"] = round(time.time() - t_start, 1)
    log("held-out", {k: round(v["acc"], 5) for k, v in res["heldout"].items()})
    log("P - D", res["paired"]["P - D"], "VERDICT", res["verdict"])
    # outputs
    json.dump(res, open(out / f"results_{analysis}.json", "w"), indent=1)
    cols = ["D", "P", "P_matched", "linear_published_converged", "linear_pca"] + sorted(k for k in scores if "_seed" in k)
    with open(out / f"pairs_{analysis}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["position", "hh_train_index", "label", "identical_after_left_truncation"] + cols)
        same = (X[ev, 0] == X[ev, 1]).all(1).numpy()
        for i, p in enumerate(ev):
            wr.writerow([int(p), int(dsi[p]), 1, int(same[i])] + [repr(float(scores[c][i])) for c in cols])
    torch.save(weights, out / f"weights_{analysis}.pt")
    json.dump({"analysis": analysis, "variants": {k: dict(zip(("width", "depth", "weight_decay", "dropout"), v)) for k, v in variants.items()},
               "grid": GRID, "pca_dim": PCA_DIM, "lr": LR, "batch": BATCH, "max_epochs": max_ep, "patience": PATIENCE,
               "selection_seeds": sel_seeds, "final_seeds": final_seeds, "bootstrap_seed": BOOT_SEED,
               "validation_rule": "sha256(str(hh_train_index))[:8] as big-endian int % 10 == 0"},
              open(out / f"config_{analysis}.json", "w"), indent=1)
    log("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "run"])
    ap.add_argument("--analysis", choices=list(ANALYSES))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    proto = Path(__file__).resolve().parent.parent / "relational_test_20261018" / "PROTOCOL.md"
    for p in (proto, a.out / "PROTOCOL.md"):
        if p.exists():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == PROTOCOL_SHA, f"{p} differs from the hashed protocol"
    sys.exit(0 if gate(a.out) else 1) if a.mode == "gate" else run(a.analysis, a.out, a.smoke)
