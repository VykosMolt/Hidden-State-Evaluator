"""Numbers in the paper that no other script prints directly: derived from per-pair records or from version 2's
saved results, each asserted against its saved record where one exists. Prints one line per number (paper wording,
value, source). CPU only; needs RPE_RUNS (A_rltt shards), RPE_AUDIT and RPE_SPLIT.

    python paper_numbers.py
"""
import csv
import glob
import json

import numpy as np
import torch

from config import RESULTS, env
from evaluator_core.pairwise_evaluator import PairwiseEvaluator

J = lambda p: json.load(open(p))  # noqa: E731
pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
lines = []


def show(what, value, source):
    lines.append((what, value, source))


# Position bias, epoch 2, Ouro-Thinking (Sec. 4.1, Fig. 3A), from the per-pair scores
z = np.load(RESULTS / "evaluator_pair_scores_thinking.npz")
e = list(z["epochs"]).index(2)
x, y = z["normal"][:, e].astype(float), z["flipped"][:, e].astype(float)
anti, sym = (x - y) / 2, (x + y) / 2
ET = J(RESULTS / "epochs_thinking_1024.json")["epochs"]["2"]
assert (x > 0).mean() == ET["fixed_order_acc"] and (y < 0).mean() == ET["swapped_order_acc"]
show("fixed order, preferred reply first (epoch 2)", pct((x > 0).mean()), "evaluator_pair_scores_thinking.npz")
show("swapped order", pct((y < 0).mean()), "evaluator_pair_scores_thinking.npz")
for k, m in (("first reply in both orders", (x > 0) & (y > 0)), ("correct in both orders", (x > 0) & (y <= 0)),
             ("wrong in both orders", (x <= 0) & (y > 0)), ("second reply in both orders", (x <= 0) & (y <= 0))):
    show(f"Fig. 3A quadrant: {k}", pct(m.mean()), "evaluator_pair_scores_thinking.npz")
show("mean order part of the score", f"{sym.mean():+.2f}", "evaluator_pair_scores_thinking.npz")
show("order part / content part (mean absolute values)", f"{np.abs(sym).mean() / np.abs(anti).mean():.2f}", "evaluator_pair_scores_thinking.npz")
show("correlation of the two orders' scores, full test split", f"{np.corrcoef(x, y)[0, 1]:.2f}", "evaluator_pair_scores_thinking.npz")
show("Fig. 3A pairs beyond +-4 (drawn at the edge)", pct(((np.abs(x) > 4) | (np.abs(y) > 4)).mean()), "evaluator_pair_scores_thinking.npz")

# Run-to-run variation (App. B): RLTT rerun (A_rltt, epoch 2) against version 2's audit, per pair
sh = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(str(env("RPE_RUNS") / "A_rltt" / "shard_*.pt")))]
idx = np.concatenate([s["dataset_index"] for s in sh])
o = np.argsort(idx)
idx = idx[o]
a = np.concatenate([s["scores"][2]["normal"] for s in sh])[o] - np.concatenate([s["scores"][2]["flipped"] for s in sh])[o]
rows = {int(r["dataset_index"]): float(r["normal_score"]) - float(r["flipped_score"])
        for r in csv.DictReader(open(env("RPE_AUDIT") / "full_hh_antisymmetry_rows.csv"))}
b = np.array([rows[int(i)] for i in idx])
ch = (a > 0) != (b > 0)
R = J(RESULTS / "rerun_vs_audit_rltt.json")
assert (a > 0).mean() == R["rerun_acc_ties_wrong"] and (b > 0).mean() == R["audit_acc_ties_wrong"]
assert ch.sum() == R["pairs_changing_sign"] and np.median(np.abs(a[ch])) == R["median_abs_score_changing"]
assert np.median(np.abs(a)) == R["median_abs_score_all"]
show("rerun, ties as errors / version 2 audit", f"{pct((a > 0).mean())} / {pct((b > 0).mean())}", "A_rltt shards + full_hh_antisymmetry_rows.csv")
show("pairs changing sign; median |score| changing / all", f"{ch.sum()}; {np.median(np.abs(a[ch])):.2f} / {np.median(np.abs(a)):.2f}", "rerun_vs_audit_rltt.json")

# Version 2's numbers quoted in the paper (Sec. 3.1 Backbones, Table 3, Fig. 1)
S = env("RPE_SPLIT")
v2 = {b_: J(S / f"hh_{b_}_40000_pair_disjoint_results.json")["eval_accuracy"] for b_ in ("base", "thinking", "rltt")}
show("v2 relational probe base / Thinking / RLTT", " / ".join(f"{v:.3f}" for v in v2.values()), "RPE_SPLIT/hh_*_40000_pair_disjoint_results.json")
c = J(S / "powered_hh_paired_bootstrap.json")["comparisons"]["rltt_minus_base"]
show("v2 RLTT minus base", f"{c['delta']:+.4f} [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}]", "RPE_SPLIT/powered_hh_paired_bootstrap.json")
pw = J(S / "hh_thinking_40000_pointwise_pair_disjoint_results.json")["variants"]["all_boundaries_concat"]
show("v2 single-response probe per reply (all loops)", pct(pw["eval_candidate_accuracy"]), "RPE_SPLIT/hh_thinking_40000_pointwise_pair_disjoint_results.json")
show("v2 relational minus single response: per reply / ranking the same pairs",
     f"{100 * (v2['thinking'] - pw['eval_candidate_accuracy']):+.1f} / {100 * (v2['thinking'] - pw['eval_induced_pairwise_accuracy']):+.1f} points", "same")
show("v2 evaluator audit, ties as errors", pct(J(env("RPE_AUDIT") / "full_hh_antisymmetry.json")["metrics"]["strict_antisym_accuracy"]), "RPE_AUDIT/full_hh_antisymmetry.json")

# Truncation (Sec. 4.3)
F = J(RESULTS / "first_difference_heldout.json")
rep = J(RESULTS / "repro_thinking.json")["summary"]
assert rep["identical_with_exactly_zero_diff"] == rep["n_identical"] == F["n_identical_at_384_right"] and rep["identical_counted_correct_by_published_rule"] == 0
n_ok = round(v2["thinking"] * 8000)
show("identical at 384 (right) / differ only after 384 / identical texts",
     f"{F['n_identical_at_384_right']} ({pct(F['n_identical_at_384_right'] / 8000)}) / {F['n_first_difference_at_or_beyond_384']} ({pct(F['n_first_difference_at_or_beyond_384'] / 8000)}) / {F['n_texts_identical']}",
     "first_difference_heldout.json, repro_thinking.json")
show("v2 relational probe on the pairs whose inputs differ", f"{n_ok}/{8000 - rep['n_identical']} = {n_ok / (8000 - rep['n_identical']):.3f}", "v2 result + repro_thinking.json")
I7 = J(RESULTS / "identical_at_768_tokens_rltt.json")
show("v2 audit at 768: identical pairs; accuracy on the rest (ties as errors)",
     f"{I7['n_token_identical']}; {pct(I7['v2_audit_epoch2_ties_wrong']['excluding_token_identical'])}", "identical_at_768_tokens_rltt.json")
I10 = J(RESULTS / "identical_at_1024_tokens_thinking.json")
show("evaluator at 1,024: identical pairs; change in epoch-2 accuracy without them",
     f"{I10['n_token_identical']}; {I10['epochs']['2']['change_points']:+.3f} points", "identical_at_1024_tokens_thinking.json")

# Length on the test split (Sec. 3.3), and the evaluator's size (Sec. 2)
L = J(RESULTS / "length_vs_evaluator.json")["shorter_tokens_wins"]
show("longer reply preferred, test split (768 tokens)", f"{pct(1 - L[0])} [{100 * (1 - L[1][1]):.1f}, {100 * (1 - L[1][0]):.1f}]", "length_vs_evaluator.json")
show("evaluator parameters", f"{sum(p.numel() for p in PairwiseEvaluator().parameters()):,}", "evaluator_core/pairwise_evaluator.py")

w = max(len(l[0]) for l in lines)
for what, value, source in lines:
    print(f"{what:<{w}}  {value:<40}  {source}")
