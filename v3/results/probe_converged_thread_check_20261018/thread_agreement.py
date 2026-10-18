"""1-thread vs 8-thread converged fits: aggregates (exact and at paper precision), per-pair decisions, weights."""
import csv, json, sys
import numpy as np

d1, d8, tag = sys.argv[1], sys.argv[2], sys.argv[3]
a1, a8 = (json.load(open(f"{d}/probes_{tag}.json")) for d in (d1, d8))
flat = lambda j, p="": sum((flat(v, f"{p}/{k}") for k, v in j.items()), []) if isinstance(j, dict) else (  # noqa: E731
    sum((flat(v, f"{p}[{i}]") for i, v in enumerate(j)), []) if isinstance(j, list) else [(p, j)])
f1, f8 = dict(flat(a1)), dict(flat(a8))
num = [k for k in f1 if isinstance(f1[k], float)]
exact = sum(f1[k] == f8[k] for k in num)
p3 = sum(round(f1[k], 3) == round(f8[k], 3) for k in num)
print(f"{tag}: aggregates exactly equal {exact}/{len(num)}; equal at 3 decimals {p3}/{len(num)}; max |diff| {max(abs(f1[k] - f8[k]) for k in num):.3e}")
r1, r8 = (list(csv.DictReader(open(f"{d}/probe_pairs_{tag}.csv"))) for d in (d1, d8))
cols = [c for c in r1[0] if c not in ("position", "hh_train_index", "label", "identical_after_left_truncation")]
for c in cols:
    x1, x8 = np.array([float(r[c]) for r in r1]), np.array([float(r[c]) for r in r8])
    print(f"  {c:28s} max |score diff| {np.abs(x1 - x8).max():.3e}  sign changes {(np.sign(x1) != np.sign(x8)).sum()}")
w1, w8 = (np.load(f"{d}/probe_weights_{tag}.npz") for d in (d1, d8))
print("  max |weight diff|:", {k: float(np.abs(w1[k] - w8[k]).max()) for k in w1.files if "positions" not in k})
m1, m8 = (json.load(open(f"{d}/probe_fits_{tag}.json")) for d in (d1, d8))
print("  threads:", m1["torch_threads"], m8["torch_threads"], " iterations 1t/8t:", {k: (m1["fits"][k]["iterations"], m8["fits"][k]["iterations"]) for k in m1["fits"]})
