"""RPE v3, analysis C (CPU part): relational vs pointwise linear probes on identical features, with surface baselines.

Input: left-truncated pooled states from extract_hh_states.py. Split: the published pair-disjoint split (32,000 train /
8,000 held-out source pairs; positions stored in the published probe file). Probes as published:
  relational  bias-free logistic on d = h_chosen - h_rejected; loss softplus(-d.w) + 0.5|w|^2/n; L-BFGS.
  pointwise   logistic with intercept on single responses (chosen 1, rejected 0); same regularization, intercept
              unregularized; pairs ranked by the two candidates' scores ("induced pairwise").
Feature sets: final loop boundary (2,048) and all four concatenated (8,192). Raw (unnormalized), as published.
Surface baselines, fit on the training pairs only: longer-response rule (direction from train), smaller/larger-norm
rule (direction from train), and a logistic on [token-count difference, norm difference per loop].
Scoring: ties count 1/2. Reported on all held-out pairs and on distinguishable ones (not token-identical after
truncation). Paired differences by pair bootstrap (10,000 draws, seed 20260711).

Release copy of analysis/fit_probes.py. The fitting and scoring are unchanged; added:
  - paths from config.py (RPE_SPLIT) instead of an absolute path;
  - --check SAVED.json: compares every value of the new aggregate result with the saved one and fails on a difference;
  - --save-dir: writes probe_weights_<tag>.npz (fitted weights), probe_pairs_<tag>.csv (per held-out pair: position,
    HH-RLHF train index, label, and each probe's score) and probe_fits_<tag>.json (L2 normalization n, L-BFGS
    iterations, final loss and gradient norm of each fit);
  - --converged: fit every probe to convergence: float64, Newton's method with backtracking on the same objective,
    stopped at max |gradient| <= 1e-10. Model, objective, regularization (n), features, split and scoring are
    unchanged; the objective is strictly convex, so its minimizer is unique and Newton reaches it in a few dozen
    steps. (L-BFGS in float64 stalls at max |gradient| ~2e-9 on the last-loop relational fit and is still at 1e-6
    after 20,000 iterations on the last-loop single-response fit.) Without --converged, the published settings are
    used (float32 L-BFGS, 200 iterations, tolerance_grad 1e-7, tolerance_change 1e-9), under which three of the four
    probe fits stop at the iteration cap and the solutions depend on the numerical environment.
"""
import argparse, csv, glob, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

from config import env

FITS = {}  # per fit: L2 normalization n, rows, dimension, L-BFGS iterations, final loss, gradient norm (metadata only)
LB = dict(max_iter=200, tolerance_grad=1e-7, tolerance_change=1e-9)  # the published L-BFGS settings
NEWTON = dict(max_iter=100, tolerance_grad=1e-10)                      # --converged
CONVERGED, DTYPE = False, torch.float32                                 # main() sets both with --converged


def final_state(loss_fn, params):
    l = loss_fn()
    g = max(float(x.abs().max()) for x in torch.autograd.grad(l, params))
    return {"final_loss": float(l.detach()), "grad_inf_norm": g, "dtype": str(DTYPE).replace("torch.", "")}


def lbfgs(loss_fn, params):
    opt = torch.optim.LBFGS(params, lr=1, line_search_fn="strong_wolfe", **LB)
    def closure():
        opt.zero_grad(); l = loss_fn(); l.backward(); return l
    opt.step(closure)
    st = opt.state[opt._params[0]]
    return {"optimizer": "L-BFGS (strong Wolfe)", "iterations": int(st["n_iter"]), "function_evaluations": int(st["func_evals"]),
            **final_state(loss_fn, params), **LB, "stopped_at_cap": int(st["n_iter"]) >= LB["max_iter"]}


def newton(X, y, w, b, loss_fn, params):
    """Minimize mean BCE(X.w (+ b), y) + 0.5|w|^2/n, the objective of the relational (y = 1, no b) and single-response
    fits, by Newton's method with backtracking; w and b (None: no intercept) are updated in place."""
    n, d = X.shape
    f = lambda w_, b_: float(F.binary_cross_entropy_with_logits(X @ w_ + (0 if b_ is None else b_), y) + .5 * (w_ @ w_) / n)  # noqa: E731
    with torch.no_grad():
        for it in range(NEWTON["max_iter"] + 1):
            p = torch.sigmoid(X @ w + (0 if b is None else b)); r, s = p - y, p * (1 - p)
            g = X.T @ r / n + w / n
            if b is not None:
                g = torch.cat([g, r.sum().reshape(1) / n])
            if float(g.abs().max()) <= NEWTON["tolerance_grad"] or it == NEWTON["max_iter"]:
                break
            k = d + (b is not None)
            H = torch.zeros(k, k, dtype=X.dtype)
            for i in range(0, n, 4096):  # X^T diag(s) X in row blocks
                Xi = X[i:i + 4096]
                H[:d, :d] += (Xi * s[i:i + 4096, None]).T @ Xi
            if b is not None:
                H[:d, d] = H[d, :d] = X.T @ s
                H[d, d] = s.sum()
            H /= n
            H.diagonal()[:d] += 1.0 / n
            step = -torch.cholesky_solve(g[:, None], torch.linalg.cholesky(H))[:, 0]
            f0, slope, t = f(w, b), float(g @ step), 1.0
            while True:  # backtracking (Armijo); a change below float64 resolution of the loss counts as no increase
                w1 = w + t * step[:d]
                b1 = None if b is None else b + t * step[d:]
                if f(w1, b1) <= f0 + 1e-4 * t * slope + 1e-15 * abs(f0) or t < 1e-10:
                    break
                t /= 2
            w.copy_(w1)
            if b is not None:
                b.copy_(b1)
    return {"optimizer": "Newton with backtracking", "iterations": it, **final_state(loss_fn, params), **NEWTON,
            "stopped_at_cap": it >= NEWTON["max_iter"]}


def relational(Dtr, name=None):
    w = torch.zeros(Dtr.shape[1], dtype=Dtr.dtype, requires_grad=True); n = len(Dtr)
    loss = lambda: F.softplus(-(Dtr @ w)).mean() + .5 * (w @ w) / n  # noqa: E731
    info = newton(Dtr, torch.ones(n, dtype=Dtr.dtype), w, None, loss, [w]) if CONVERGED else lbfgs(loss, [w])
    if name:
        FITS[name] = {"objective": "mean softplus(-d.w) + 0.5*|w|^2/n, no intercept", "l2_n": n,
                      "l2_n_unit": "training pairs (one difference row per pair)", "dim": int(Dtr.shape[1]), **info}
    return w.detach()


def pointwise(Xtr, ytr, name=None):
    w = torch.zeros(Xtr.shape[1], dtype=Xtr.dtype, requires_grad=True); b = torch.zeros(1, dtype=Xtr.dtype, requires_grad=True); n = len(Xtr)
    loss = lambda: F.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + .5 * (w @ w) / n  # noqa: E731
    info = newton(Xtr, ytr, w, b, loss, [w, b]) if CONVERGED else lbfgs(loss, [w, b])
    if name:
        FITS[name] = {"objective": "mean BCE(x.w + b, y) + 0.5*|w|^2/n, intercept unregularized", "l2_n": n,
                      "l2_n_unit": "training responses (chosen and rejected rows of each training pair)",
                      "dim": int(Xtr.shape[1]), **info}
    return w.detach(), b.detach()


def correct(margin):  # margin > 0 means chosen ranked first; exact ties count 1/2
    m = np.asarray(margin, float)
    return np.where(m > 0, 1.0, np.where(m == 0, 0.5, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, help="saved aggregate JSON that the new result must equal exactly")
    ap.add_argument("--save-dir", type=Path, help="where to write weights, per-pair scores and fit metadata")
    ap.add_argument("--tag", help="name used in the saved file names, e.g. thinking")
    ap.add_argument("--converged", action="store_true", help="float64 and L-BFGS run to convergence (see docstring)")
    a = ap.parse_args()
    global DTYPE, CONVERGED
    if a.converged:
        CONVERGED, DTYPE = True, torch.float64
    sh = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(str(a.features / "shard_*.pt")))]
    pos = np.concatenate([s["positions"] for s in sh]); order = np.argsort(pos); assert (pos[order] == np.arange(40000)).all()
    dsi = np.concatenate([s["dataset_indices"] for s in sh])[order]               # HH-RLHF train index of each position
    P = torch.cat([s["pooled"] for s in sh])[order].float()                      # [40000, 2, 4, 2048]
    tok_full = torch.cat([s["tokens_untruncated"] for s in sh])[order].float()  # [40000, 2]
    probe = torch.load(env("RPE_SPLIT") / "hh_thinking_40000_pair_disjoint_probe.pt", map_location="cpu", weights_only=False)
    tr, ev = np.array(probe["train_positions"], int), np.array(probe["eval_positions"], int)
    same = (P[:, 0] == P[:, 1]).flatten(1).all(1).numpy()                       # identical states (identical texts)
    rng = np.random.default_rng(20260711)
    B = [rng.integers(0, len(ev), len(ev)) for _ in range(10000)]
    ci = lambda x: [float(np.percentile([x[b].mean() for b in B], q)) for q in (2.5, 97.5)]
    dist = ~same[ev]
    res = {"n_eval": len(ev), "n_eval_identical_after_left_truncation": int(same[ev].sum()), "acc": {}, "paired": {}}
    C = {}
    W, S_EV = {}, {}  # fitted weights; per held-out pair scores (chosen minus rejected unless named otherwise)

    def report(name, c):
        C[name] = c
        res["acc"][name] = {"all_ties_half": [c.mean(), ci(c)], "distinguishable": float(c[dist].mean())}

    for fname, sl in (("final", slice(3, 4)), ("all4", slice(0, 4))):
        X = lambda rows, k: P[rows, k, sl].flatten(1).to(DTYPE)                 # [len(rows), d], one reply per pair
        Dtr, Dev = X(tr, 0) - X(tr, 1), X(ev, 0) - X(ev, 1)                     # built per split to fit float64 in memory
        w = relational(Dtr, f"relational_{fname}"); del Dtr
        report(f"relational_{fname}", correct((Dev @ w).numpy()))
        W[f"relational_{fname}_w"] = w.numpy(); S_EV[f"relational_{fname}"] = (Dev @ w).numpy()
        Xs = torch.cat([X(tr, 0), X(tr, 1)])
        ys = torch.cat([torch.ones(len(tr), dtype=DTYPE), torch.zeros(len(tr), dtype=DTYPE)])
        wp, bp = pointwise(Xs, ys, f"pointwise_{fname}"); del Xs
        sc, sr = (X(ev, 0) @ wp + bp).numpy(), (X(ev, 1) @ wp + bp).numpy()
        W[f"pointwise_{fname}_w"], W[f"pointwise_{fname}_b"] = wp.numpy(), bp.numpy()
        S_EV[f"pointwise_{fname}_chosen"], S_EV[f"pointwise_{fname}_rejected"] = sc, sr
        report(f"pointwise_{fname}_induced_pairwise", correct(sc - sr))
        cand = np.concatenate([(sc > 0), (sr <= 0)]).astype(float)
        res["acc"][f"pointwise_{fname}_candidate"] = float(cand.mean())
        print(fname, {k: round(v["all_ties_half"][0], 4) for k, v in res["acc"].items() if isinstance(v, dict)}, flush=True)

    # surface baselines, directions and weights from the training pairs only
    dlen = (tok_full[:, 0] - tok_full[:, 1]).numpy()
    nrm = P.norm(dim=-1)                                                         # [40000, 2, 4]
    dn = (nrm[:, 0] - nrm[:, 1]).numpy()                                         # [40000, 4]
    for name, d in (("length_rule", dlen), ("norm_final_rule", dn[:, 3])):
        sign = 1.0 if correct(d[tr]).mean() >= 0.5 else -1.0
        report(name, correct(sign * d[ev])); res["acc"][name]["direction_from_train"] = "larger wins" if sign > 0 else "smaller wins"
        W[f"{name}_sign"] = np.float32(sign); S_EV[name] = sign * d[ev]
    S = torch.tensor(np.column_stack([dlen, dn]), dtype=torch.float32).to(DTYPE)
    mu, sd = S[tr].mean(0), S[tr].std(0) + 1e-6
    ws = relational((S[tr] - 0) / sd, "surface_logistic")                       # bias-free on differences, as the relational probe
    report("surface_logistic", correct(((S[ev] / sd) @ ws).numpy()))
    W["surface_logistic_w"], W["surface_logistic_sd"] = ws.numpy(), sd.numpy(); S_EV["surface_logistic"] = ((S[ev] / sd) @ ws).numpy()

    for a_, b_ in (("relational_final", "pointwise_final_induced_pairwise"), ("relational_all4", "pointwise_all4_induced_pairwise"),
                   ("relational_all4", "relational_final"), ("relational_final", "surface_logistic"),
                   ("relational_all4", "surface_logistic"), ("pointwise_final_induced_pairwise", "surface_logistic")):
        d = C[a_] - C[b_]
        res["paired"][f"{a_} - {b_}"] = {"all": [d.mean(), ci(d)], "distinguishable": float(d[dist].mean())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))

    if a.check:  # gate: per-pair outputs are saved only if the refit reproduces the saved aggregate exactly
        new, old = json.loads(json.dumps(res, default=float)), json.load(open(a.check))
        diffs, n_vals = [], [0]
        def walk(x, y, path):
            if isinstance(y, dict):
                if set(x) != set(y): diffs.append((path, "keys", sorted(set(x) ^ set(y))))
                for k in y:
                    if k in x: walk(x[k], y[k], f"{path}/{k}")
            elif isinstance(y, list):
                if len(x) != len(y): diffs.append((path, x, y))
                for i, (u, v) in enumerate(zip(x, y)): walk(u, v, f"{path}[{i}]")
            else:
                n_vals[0] += 1
                if x != y: diffs.append((path, x, y))
        walk(new, old, "")
        print(f"CHECK against {a.check.name}: {len(diffs)} of {n_vals[0]} values differ" + ("" if diffs else " (exact match)"))
        for d in diffs: print("  DIFF", d)
        if diffs: raise SystemExit(1)

    if a.save_dir:
        a.save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.save_dir / f"probe_weights_{a.tag}.npz", train_positions=tr, eval_positions=ev, **W)
        cols = list(S_EV)
        with open(a.save_dir / f"probe_pairs_{a.tag}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["position", "hh_train_index", "label", "identical_after_left_truncation"] + cols)
            for i, p in enumerate(ev):  # repr of the score as computed (float32 or float64): round-trips exactly
                wr.writerow([int(p), int(dsi[p]), 1, int(same[p])] + [repr(float(S_EV[c][i])) for c in cols])
        nz = lambda v: float(np.abs(v[v != 0]).min())  # noqa: E731  smallest nonzero |score| deciding a held-out pair
        margins = {k: nz(S_EV[k]) for k in ("relational_final", "relational_all4", "surface_logistic")}
        for f_ in ("final", "all4"):
            c_, r_ = S_EV[f"pointwise_{f_}_chosen"], S_EV[f"pointwise_{f_}_rejected"]
            margins[f"pointwise_{f_}_pair"], margins[f"pointwise_{f_}_reply"] = nz(c_ - r_), nz(np.concatenate([c_, r_]))
        meta = {"tag": a.tag, "settings": "converged" if a.converged else "published", "solver": NEWTON if a.converged else LB,
                "smallest_nonzero_heldout_decision_score": margins,
                "dtype": str(DTYPE).replace("torch.", ""), "features": a.features.name, "n_train_pairs": len(tr), "n_eval_pairs": len(ev),
                "fits": FITS, "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
                "columns": {
                    "position": "index into the 40,000-pair selection (hh_train_indices_40000.json)",
                    "hh_train_index": "row of the Anthropic/hh-rlhf train split",
                    "label": "always 1: the human-preferred (chosen) reply is the first member; a positive pair score is correct, 0 is a tie (1/2)",
                    "identical_after_left_truncation": "1 if the two replies' pooled states are identical (identical texts)",
                    "relational_*": "w.(h_chosen - h_rejected)",
                    "pointwise_*_chosen / _rejected": "w.h + b for each reply; the pair score is chosen minus rejected; per-reply accuracy counts chosen > 0 and rejected <= 0",
                    "length_rule / norm_final_rule": "sign_from_train * (chosen - rejected) of untruncated token count / last-loop state norm",
                    "surface_logistic": "w.(differences in token count and the four loop norms, divided by train std)"}}
        json.dump(meta, open(a.save_dir / f"probe_fits_{a.tag}.json", "w"), indent=1)
        print("wrote weights, per-pair scores and fit metadata to", a.save_dir)


if __name__ == "__main__":
    main()
