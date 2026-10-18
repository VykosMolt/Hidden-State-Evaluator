"""RPE v3, analysis B (CPU part): response-length rule vs the epoch-2 evaluator on the HH test set, paired.
Length rule fixed in advance from v1 sec. 3.3 (rejected responses are longer): the shorter text wins; ties 1/2.
Evaluator per-pair scores: the July full-set audit (RLTT, 768 tokens). Strict antisymmetrized decision:
normal - flipped > 0."""
import csv, json, sys
import numpy as np
from config import env, load_hh, tokenizer  # release copy: locations and the dataset revision from config.py
ROWS = env("RPE_AUDIT") / "full_hh_antisymmetry_rows.csv"
ds = load_hh("test")
tok = tokenizer()
rows = {int(r["dataset_index"]): (float(r["normal_score"]), float(r["flipped_score"])) for r in csv.DictReader(open(ROWS))}
idx = sorted(rows)
nt = lambda t: min(len(tok(t).input_ids), 768)
tc = np.array([nt(ds[i]["chosen"]) for i in idx]); tr = np.array([nt(ds[i]["rejected"]) for i in idx])
cc = np.array([len(ds[i]["chosen"]) for i in idx]); cr = np.array([len(ds[i]["rejected"]) for i in idx])
ev = np.array([1.0 if rows[i][0] - rows[i][1] > 0 else 0.0 for i in idx])
rule = lambda a, b: np.where(a < b, 1.0, np.where(a == b, 0.5, 0.0))
L_tok, L_chr = rule(tc, tr), rule(cc, cr)
rng = np.random.default_rng(20260710)
def ci(x, draws=10000):
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(draws)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
out = {"n_pairs": len(idx),
       "evaluator_epoch2_strict_antisym": [ev.mean(), ci(ev)],
       "shorter_tokens_wins": [L_tok.mean(), ci(L_tok)], "shorter_chars_wins": [L_chr.mean(), ci(L_chr)],
       "evaluator_minus_shorter_tokens_paired": [(ev - L_tok).mean(), ci(ev - L_tok)],
       "ties_tokens": int((tc == tr).sum()), "truncated_768_any": int(((tc == 768) | (tr == 768)).sum()),
       "evaluator_acc_when_lengths_equal": float(ev[tc == tr].mean()) if (tc == tr).any() else None,
       "agreement_evaluator_vs_length_rule": float(((ev == 1) == (L_tok == 1))[tc != tr].mean())}
json.dump(out, open(sys.argv[1], "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
