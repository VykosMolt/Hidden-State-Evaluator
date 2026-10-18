"""Convergence trace (float64 L-BFGS, same objective): max |grad| and loss every 250 iterations, relational last loop."""
import glob, sys, time
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, sys.argv[2])
from config import env

sh = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(f"{sys.argv[1]}/shard_*.pt"))]
pos = np.concatenate([s["positions"] for s in sh]); order = np.argsort(pos)
P = torch.cat([s["pooled"] for s in sh])[order].float()
tr = np.array(torch.load(env("RPE_SPLIT") / "hh_thinking_40000_pair_disjoint_probe.pt", map_location="cpu", weights_only=False)["train_positions"], int)
sl = slice(3, 4) if sys.argv[3] == "final" else slice(0, 4)
D = (P[tr, 0, sl].flatten(1).double() - P[tr, 1, sl].flatten(1).double()); n = len(D); del P
print("feature row norm median", float(D.norm(dim=1).median()), flush=True)
w = torch.zeros(D.shape[1], dtype=D.dtype, requires_grad=True)
loss = lambda: F.softplus(-(D @ w)).mean() + .5 * (w @ w) / n
opt = torch.optim.LBFGS([w], lr=1, max_iter=250, max_eval=500, tolerance_grad=0, tolerance_change=0, line_search_fn="strong_wolfe")
def closure():
    opt.zero_grad(); l = loss(); l.backward(); return l
t = time.time()
for k in range(int(sys.argv[4])):
    opt.step(closure)
    l = loss(); g = torch.autograd.grad(l, [w])[0]
    st = opt.state[w]
    print(f"iter {st['n_iter']} evals {st['func_evals']} loss {float(l):.15f} gmax {float(g.abs().max()):.3e} |w| {float(w.norm()):.6f} t {time.time() - t:.0f}s", flush=True)
