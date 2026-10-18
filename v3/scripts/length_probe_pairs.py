"""RPE v3, analysis B (CPU part 2): response-length rules on the 8,000 held-out pairs of the pair-disjoint 40k probes
(HH train split, 384-token truncation as the probes). Reports both directions; v1's direction (shorter wins) was
fixed in advance and failed on the test set, so the longer-wins rule is reported as the post-hoc best."""
import json, sys
import numpy as np, torch
from config import env, load_hh, tokenizer  # release copy: locations and the dataset revision from config.py
P = str(env("RPE_SPLIT"))
idx = json.load(open(f"{P}/hh_train_indices_40000.json"))["indices"]
ev = torch.load(f"{P}/hh_thinking_40000_pair_disjoint_probe.pt", map_location="cpu", weights_only=False)["eval_positions"]
ev = [int(x) for x in ev]
for b in ("rltt", "base"):  # the same split for every backbone
    assert [int(x) for x in torch.load(f"{P}/hh_{b}_40000_pair_disjoint_probe.pt", map_location="cpu", weights_only=False)["eval_positions"]] == ev
ds = load_hh("train")
tok = tokenizer()
nt = lambda t: min(len(tok(t).input_ids), 384)
tc = np.array([nt(ds[idx[p]]["chosen"]) for p in ev]); tr = np.array([nt(ds[idx[p]]["rejected"]) for p in ev])
rule = lambda a, b: np.where(a < b, 1.0, np.where(a == b, 0.5, 0.0))
rng = np.random.default_rng(20260711)
ci = lambda x: [float(np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)], q)) for q in (2.5, 97.5)]
s, l = rule(tc, tr), rule(tr, tc)
out = {"n_pairs": len(ev), "shorter_wins": [s.mean(), ci(s)], "longer_wins": [l.mean(), ci(l)],
       "ties": int((tc == tr).sum()), "both_truncated_384": int(((tc == 384) & (tr == 384)).sum()),
       "for_reference_probes": {"relational_thinking": 0.56525, "relational_rltt": 0.56975, "relational_base": 0.55525,
                                "pointwise_final_induced_pairwise_thinking": 0.567625}}
json.dump(out, open(sys.argv[1], "w"), indent=2, default=float); print(json.dumps(out, indent=2, default=float))
