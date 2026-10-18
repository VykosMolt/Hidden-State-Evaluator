"""RPE v3, analysis A (CPU part): per-epoch metrics of the pairwise evaluator on the HH test set, from
epochs_and_surface.py shards, plus the norm baseline on the same pairs.

Per epoch: fixed-order accuracy (chosen first, score > 0; what v1 reported), swapped-order accuracy, strict
antisymmetrized accuracy (normal - flipped > 0, ties 1/2) on all pairs and on distinguishable pairs, mean
symmetric (order) component, sign-flip rate, and the share of pairs where both orders prefer the first argument.
Check: epoch 2 against the July audit's per-pair scores. Norm baseline: smaller pooled-state norm of the final loop
wins (v1's direction, fixed in advance), and the post-hoc reverse."""
import csv, glob, json, sys
import numpy as np, torch

from config import env  # release copy: location from config.py
ROWS = env("RPE_AUDIT") / "full_hh_antisymmetry_rows.csv"
src, out = sys.argv[1], sys.argv[2]
sh = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(f"{src}/shard_*.pt"))]
idx = np.concatenate([s["dataset_index"] for s in sh])
o = np.argsort(idx); idx = idx[o]
pc = torch.cat([s["pooled"]["chosen"] for s in sh])[o].float(); pr = torch.cat([s["pooled"]["rejected"] for s in sh])[o].float()
same = (pc == pr).flatten(1).all(1).numpy()
rng = np.random.default_rng(20260710); B = [rng.integers(0, len(idx), len(idx)) for _ in range(10000)]
ci = lambda x: [float(np.percentile([x[b].mean() for b in B], q)) for q in (2.5, 97.5)]
corr = lambda m: np.where(m > 0, 1.0, np.where(m == 0, 0.5, 0.0))
res = {"n_pairs": int(len(idx)), "n_identical_states": int(same.sum()), "epochs": {}}
for e in range(1, 6):
    nrm = np.concatenate([s["scores"][e]["normal"] for s in sh])[o]; flp = np.concatenate([s["scores"][e]["flipped"] for s in sh])[o]
    anti, sym = (nrm - flp) / 2, (nrm + flp) / 2
    a = corr(anti)
    res["epochs"][e] = {"fixed_order_acc": float((nrm > 0).mean()), "swapped_order_acc": float((flp < 0).mean()),
                        "strict_antisym_acc": [a.mean(), ci(a)], "strict_antisym_acc_distinguishable": float(a[~same].mean()),
                        "mean_symmetric_component": float(sym.mean()), "sign_flip_rate": float((np.sign(nrm) != np.sign(flp)).mean()),
                        "both_orders_prefer_first": float(((nrm > 0) & (flp > 0)).mean())}
    print(e, {k: (round(v, 4) if isinstance(v, float) else [round(v[0], 4), [round(x, 4) for x in v[1]]]) for k, v in res["epochs"][e].items()}, flush=True)
e2 = {int(r["dataset_index"]): (float(r["normal_score"]), float(r["flipped_score"])) for r in csv.DictReader(open(ROWS))}
n2 = np.concatenate([s["scores"][2]["normal"] for s in sh])[o]; f2 = np.concatenate([s["scores"][2]["flipped"] for s in sh])[o]
an, af = np.array([e2[i][0] for i in idx]), np.array([e2[i][1] for i in idx])
res["check_epoch2_vs_audit"] = {"pearson_normal": float(np.corrcoef(n2, an)[0, 1]), "max_abs_diff_normal": float(np.abs(n2 - an).max()),
                                "antisym_decision_agreement": float(((n2 - f2 > 0) == (an - af > 0)).mean()),
                                "audit_strict_antisym_acc": float((an - af > 0).mean())}
dn = (pc[:, 3].norm(dim=-1) - pr[:, 3].norm(dim=-1)).numpy()
s_small, s_large = corr(-dn), corr(dn)
res["norm_baseline"] = {"smaller_norm_wins_prespecified": [s_small.mean(), ci(s_small)], "larger_norm_wins_posthoc": [s_large.mean(), ci(s_large)]}
best_norm = s_small if s_small.mean() >= s_large.mean() else s_large  # the better direction, post hoc (favours the baseline)
d = corr((n2 - f2) / 2) - best_norm
res["evaluator_e2_minus_best_norm_rule_paired"] = [d.mean(), ci(d)]
json.dump(res, open(out, "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "epochs"}, indent=2, default=float))
