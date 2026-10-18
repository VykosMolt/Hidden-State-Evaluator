"""Calibration: iterations needed by the converged settings, last-loop fits only (Thinking features)."""
import glob, json, sys, time
import numpy as np, torch
sys.path.insert(0, sys.argv[2])
import fit_probes as fp
from config import env

fp.LB.update(fp.CONVERGED); fp.DTYPE = torch.float64
sh = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(f"{sys.argv[1]}/shard_*.pt"))]
pos = np.concatenate([s["positions"] for s in sh]); order = np.argsort(pos)
P = torch.cat([s["pooled"] for s in sh])[order].float()
tr = np.array(torch.load(env("RPE_SPLIT") / "hh_thinking_40000_pair_disjoint_probe.pt", map_location="cpu", weights_only=False)["train_positions"], int)
X = lambda rows, k: P[rows, k, 3:4].flatten(1).double()
t = time.time(); fp.relational(X(tr, 0) - X(tr, 1), "relational_final"); print("rel", time.time() - t, flush=True)
t = time.time(); fp.pointwise(torch.cat([X(tr, 0), X(tr, 1)]), torch.cat([torch.ones(len(tr)), torch.zeros(len(tr))]).double(), "pointwise_final"); print("pw", time.time() - t)
print(json.dumps(fp.FITS, indent=1))
