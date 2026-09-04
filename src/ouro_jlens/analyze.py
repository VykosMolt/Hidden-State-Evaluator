"""Summaries and figures from evaluate.py (and optionally probe.py) outputs.

  python src/ouro_jlens/analyze.py --eval artifacts/jlens/eval/run1 [--probe artifacts/jlens/probe/run1]

Metrics per location (ut, layer), over scorable non-leaked intermediates:
  hit@k        own intermediate ranked < k (0 = top)
  control@k    same for the other intermediate names of the task (position prior)
  excess@k     hit@k - control@k, per item then averaged
  cand top-1   own name ranked strictly above every other task name
Writes summary.json, summary.md and fig_*.png into the eval directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ouro_jlens.evidence import atomic_write_json, atomic_write_text

N_UT, N_LAYER = 4, 48
K = 10
OPERATIONS = {"addition", "subtraction", "multiplication", "division", "mod", "squared"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f5f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7a1f1f"])
LOOP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BOOT_SEED = 0
N_BOOT = 20000


def boundary_prefix_match(continuation: str, target: str) -> bool:
    c, t = continuation.strip().strip('"').lower(), target.strip().lower()
    if not c.startswith(t):
        return False
    rest = c[len(t):]
    return rest == "" or not rest[0].isalnum()


class Eval:
    """Rank tensors plus the item/name bookkeeping needed to score them."""

    def __init__(self, eval_dir: Path) -> None:
        self.arrays = dict(np.load(eval_dir / "arrays.npz"))
        self.items = json.loads((eval_dir / "items.json").read_text())
        # Old artifacts used an unsafe prefix match ("11" counted as target
        # "1").  Correctness is derived from the retained continuation every
        # time so analysis cannot inherit that stale boolean.
        for item in self.items:
            if "continuation" in item and "target" in item:
                item["correct"] = boundary_prefix_match(item["continuation"], item["target"])
        self.task_names = json.loads((eval_dir / "task_names.json").read_text())
        n, max_names = self.arrays["jlens_exit3_allrank"].shape[:2]
        self.own = np.zeros((n, max_names), bool)       # item's own scorable names
        self.valid = np.zeros((n, max_names), bool)     # real (unpadded) names of the item's task
        # Controls must be the same kind as the scored intermediate: operation names rank far
        # lower than numbers at these readout positions, so mixing them in flatters the metric.
        self.is_op = np.zeros((n, max_names), bool)
        self.slot_mask = self.groups()
        for i, it in enumerate(self.items):
            names = self.task_names[it["task"]]
            self.valid[i, : len(names)] = True
            self.is_op[i, : len(names)] = [nm in OPERATIONS for nm in names]
            for j in it["own_index"]:
                if j >= 0:
                    self.own[i, j] = True
    def groups(self) -> dict[str, np.ndarray]:
        """Boolean masks over (item, intermediate-slot) for each analysis group."""
        n = len(self.items)
        scorable, leaked, numeric, correct = (np.zeros((n, 3), bool) for _ in range(4))
        task = np.empty((n, 3), object)
        for i, it in enumerate(self.items):
            for j, name in enumerate(it["intermediates"]):
                scorable[i, j], leaked[i, j], task[i, j] = it["scorable"][j], it["leaked"][j], it["task"]
                numeric[i, j] = name not in OPERATIONS
                correct[i, j] = it["correct"]
        clean = scorable & ~leaked
        return {
            "multihop": clean & (task == "multihop"),
            "order-ops numeric": clean & (task == "order-ops") & numeric,
            "multihop (model correct)": clean & (task == "multihop") & correct,
            "order-ops numeric (model correct)": clean & (task == "order-ops") & numeric & correct,
            "leaked (all tasks)": scorable & leaked,
        }

    def own_slots(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(item index, name index) for every masked slot."""
        ii, jj = [], []
        for i, it in enumerate(self.items):
            for k, j in enumerate(it["own_index"]):
                if mask[i, k] and j >= 0:
                    ii.append(i)
                    jj.append(j)
        return np.array(ii, int), np.array(jj, int)

    def scores(self, allrank: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        """Per-slot metrics, each [n_slots, *loc] for a rank tensor [n, names, *loc]."""
        ii, jj = self.own_slots(mask)
        own_rank = allrank[ii, jj]                                   # [slots, *loc]
        hit = (own_rank < K).astype(float)
        ctrl, cand = np.zeros_like(hit), np.zeros_like(hit)
        ctrl_any = np.zeros(_drop_layer_axis(hit.shape))
        for s, (i, j) in enumerate(zip(ii, jj)):
            others = self.valid[i] & ~self.own[i] & (self.is_op[i] == self.is_op[i, j])
            r = (allrank[i, others] < K).astype(float)                # [n_ctrl, *loc]
            ctrl[s] = r.mean(0)
            ctrl_any[s] = _any_layer_per_name(r).mean(0)
            cand[s] = (allrank[i, others] > own_rank[s]).all(0)
        return {"hit": hit, "control": ctrl, "control_any": ctrl_any, "excess": hit - ctrl,
                "cand_top1": cand, "items": ii}


def _split_layers(a: np.ndarray) -> np.ndarray:
    """Expose the layer axis: a trailing 192 becomes (N_UT, N_LAYER); a trailing 48 is already it."""
    return a.reshape(*a.shape[:-1], N_UT, N_LAYER) if a.shape[-1] == N_UT * N_LAYER else a


def _any_layer_per_name(r: np.ndarray) -> np.ndarray:
    """[n_names, *loc] -> [n_names, *loc without the layer axis], hit at any layer of the loop."""
    return _split_layers(r).max(-1)


def _drop_layer_axis(shape: tuple[int, ...]) -> tuple[int, ...]:
    trailing = (N_UT,) if shape[-1] == N_UT * N_LAYER else ()
    return (*shape[:-1], *trailing)


def by_item(values: np.ndarray, items: np.ndarray) -> np.ndarray:
    """Average slot-level values within item -> [n_items, ...] (items as the bootstrap unit)."""
    uniq = np.unique(items)
    return np.stack([values[items == u].mean(0) for u in uniq])


def boot_ci(values: np.ndarray, stat) -> list[float]:
    """Percentile CI over items. A fresh generator per call so a CI does not depend on how
    many earlier calls drew from a shared stream."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(values)
    draws = [stat(values[rng.integers(0, n, n)]) for _ in range(N_BOOT)]
    return [round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3)]


def loc_maps(sc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-location means. Items are the unit throughout: slots within an item (e.g. the
    near-duplicate names "3"/"three"/"third") are one measurement, not three."""
    return {k: by_item(v, sc["items"]).mean(0).reshape(N_UT, N_LAYER)
            for k, v in sc.items() if k not in ("items", "control_any")}


def any_layer(sc: dict[str, np.ndarray]) -> dict:
    """Paper-style pass@k: hit at any layer within the loop. The control is scored the same
    way per control name and only then averaged; taking the max of the averaged control
    instead would understate it (max of a mean <= mean of maxes) and inflate every excess."""
    def per_loop(v):
        return by_item(_any_layer_per_name(v[:, None])[:, 0], sc["items"]).mean(0).round(3).tolist()

    return {"pass10_per_loop": per_loop(sc["hit"]),
            "control_pass10_per_loop": by_item(sc["control_any"], sc["items"]).mean(0).round(3).tolist(),
            "cand_top1_any_layer_per_loop": per_loop(sc["cand_top1"])}


def any_layer_items(sc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Item-level values underlying :func:`any_layer` and paired inference."""
    hit = _any_layer_per_name(sc["hit"][:, None])[:, 0]
    cand = _any_layer_per_name(sc["cand_top1"][:, None])[:, 0]
    return {
        "pass10": by_item(hit, sc["items"]),
        "control_pass10": by_item(sc["control_any"], sc["items"]),
        "excess_pass10": by_item(hit - sc["control_any"], sc["items"]),
        "candidate_top1": by_item(cand, sc["items"]),
    }


#: Off-diagonal cells, and the two roles loop 1 can play in them.
_OFF = ~np.eye(N_UT, dtype=bool)
_FIT1 = np.zeros((N_UT, N_UT), bool); _FIT1[0, :] = True      # J fitted at loop 1
_STATE1 = np.zeros((N_UT, N_UT), bool); _STATE1[:, 0] = True  # applied to a loop-1 state


def _contrast(sel: np.ndarray, rest: np.ndarray):
    """Statistic: mean over `sel` cells minus mean over `rest` cells, off-diagonal only."""
    a, b = _OFF & sel, _OFF & rest

    def stat(h):
        m = h.mean(0)
        return m[a].mean() - m[b].mean()

    return stat


def cross_loop_summary(ev: Eval, mask: np.ndarray) -> dict:
    """xloop_allrank [n, names, 4(fit), 4(state), 48] -> 4x4 any-layer excess hit@10.

    Reports the contemporaneous H1 contrast (cells involving loop 1 in either
    role) and the two role-separated descriptive contrasts.  ``state_only``
    and ``fit_only`` distinguish where the association sits in the matrix;
    neither identifies a causal mechanism.
    """
    sc = ev.scores(ev.arrays["xloop_allrank"], mask)
    ex_items = by_item(sc["hit"].max(-1) - sc["control_any"], sc["items"])  # [items, 4, 4]
    M = ex_items.mean(0)
    grand = M.mean()
    row, col = M.mean(1) - grand, M.mean(0) - grand
    ss_tot = float(((M - grand) ** 2).sum())
    ss_row, ss_col = float(N_UT * (row ** 2).sum()), float(N_UT * (col ** 2).sum())

    contrasts = {
        "h1_preregistered": _contrast(_FIT1 | _STATE1, ~(_FIT1 | _STATE1)),
        "h1_state_only": _contrast(_STATE1, ~_STATE1 & ~_FIT1),
        "h1_fit_only": _contrast(_FIT1, ~_STATE1 & ~_FIT1),
    }
    out = {
        "excess_hit10_fit_by_state": M.round(3).tolist(),
        "hit10_fit_by_state": by_item(sc["hit"].max(-1), sc["items"]).mean(0).round(3).tolist(),
        "control10_fit_by_state": by_item(sc["control_any"], sc["items"]).mean(0).round(3).tolist(),
        "cand_top1_fit_by_state": by_item(sc["cand_top1"].max(-1), sc["items"]).mean(0).round(3).tolist(),
        "fit_loop_effect": row.round(3).tolist(),
        "state_loop_effect": col.round(3).tolist(),
        "variance_share": {"fit_loop": round(ss_row / ss_tot, 2), "state_loop": round(ss_col / ss_tot, 2),
                           "interaction": round(1 - (ss_row + ss_col) / ss_tot, 2)},
        "n_items": int(len(ex_items)), "n_slots": int(len(sc["hit"])),
    }
    for name, stat in contrasts.items():
        out[name] = round(float(stat(ex_items)), 3)
        out[f"{name}_ci95"] = boot_ci(ex_items, stat)
    out["h1_loop1_minus_later_offdiag"] = out["h1_preregistered"]  # figure label
    out["h1_ci95"] = out["h1_preregistered_ci95"]
    return out


def fig_heatmaps(maps: dict[str, dict[str, dict[str, np.ndarray]]], out: Path) -> None:
    tasks = list(maps)
    rows = [("Jacobian lens", "hit", SEQ, 0, 1), ("logit lens", "hit", SEQ, 0, 1),
            ("Jacobian lens", "excess", DIV, -0.6, 0.6), ("logit lens", "excess", DIV, -0.6, 0.6),
            ("Jacobian lens", "cand_top1", SEQ, 0, 1), ("logit lens", "cand_top1", SEQ, 0, 1)]
    labels = {"hit": f"hit@{K}", "excess": f"excess hit@{K} over matched controls", "cand_top1": "candidate-set top-1"}
    fig, axes = plt.subplots(len(rows), len(tasks), figsize=(13, 12), constrained_layout=True)
    for col, task in enumerate(tasks):
        for r, (method, key, cmap, lo, hi) in enumerate(rows):
            ax = axes[r, col]
            im = ax.imshow(maps[task][method][key], cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
            ax.set_title(f"{task} — {method}: {labels[key]}", fontsize=9)
            ax.set_yticks(range(N_UT), [f"loop {u+1}" for u in range(N_UT)], fontsize=8)
            ax.set_xticks(range(0, N_LAYER, 8))
            if r == len(rows) - 1:
                ax.set_xlabel("physical layer")
            if r % 2 == 1:
                fig.colorbar(im, ax=axes[r - 1 : r + 1, col], shrink=0.7)
    fig.suptitle("Known-intermediate readout at the token preceding the target: recurrent loop × physical layer")
    fig.savefig(out / "fig1_readout_heatmaps.png", dpi=150)
    plt.close(fig)


def fig_lines(maps, out: Path) -> None:
    tasks = list(maps)
    fig, axes = plt.subplots(2, len(tasks), figsize=(13, 7.5), constrained_layout=True, sharex=True)
    for col, task in enumerate(tasks):
        for row, key in enumerate(("excess", "cand_top1")):
            ax = axes[row, col]
            for u in range(N_UT):
                ax.plot(maps[task]["Jacobian lens"][key][u], color=LOOP_COLORS[u], lw=2, label=f"loop {u+1} Jacobian lens")
                ax.plot(maps[task]["logit lens"][key][u], color=LOOP_COLORS[u], lw=1.2, ls="--", label=f"loop {u+1} logit lens")
            ax.axhline(0, color="#9e9d98", lw=1)
            ax.set_title(f"{task}: {'excess hit@10 over controls' if key == 'excess' else 'candidate-set top-1'}", fontsize=10)
            ax.grid(alpha=0.25)
        axes[1, col].set_xlabel("physical layer")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Solid = Jacobian lens (eventual-exit target), dashed = vanilla logit lens")
    fig.savefig(out / "fig2_lines.png", dpi=150)
    plt.close(fig)


def fig_xloop(xl: dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(2, len(xl), figsize=(5.2 * len(xl), 8.5), constrained_layout=True)
    for col, (task, res) in enumerate(xl.items()):
        for row, (key, cmap, lo, hi, title) in enumerate([
            ("excess_hit10_fit_by_state", DIV, -0.6, 0.6, "excess hit@10 over controls (any layer)"),
            ("cand_top1_fit_by_state", SEQ, 0, 1, "candidate-set top-1 (any layer)"),
        ]):
            ax = axes[row, col]
            M = np.array(res[key])
            ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi)
            for i in range(N_UT):
                for j in range(N_UT):
                    dark_cell = M[i, j] > 0.6 if cmap is SEQ else abs(M[i, j]) > 0.4
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                            color="white" if dark_cell else "#0b0b0b")
            ax.set_xticks(range(N_UT), [f"state loop {u+1}" for u in range(N_UT)], fontsize=8)
            ax.set_yticks(range(N_UT), [f"J fit loop {u+1}" for u in range(N_UT)], fontsize=8)
            ax.set_title(f"{task}: {title}" + (f"\nH1 stat {res['h1_loop1_minus_later_offdiag']:+.3f}, CI {res['h1_ci95']}" if row == 0 else ""), fontsize=9)
    fig.suptitle("Cross-loop transfer: J fitted at (loop i, L) applied to the state at (loop j, L)")
    fig.savefig(out / "fig3_cross_loop.png", dpi=150)
    plt.close(fig)


def model_exits(ev: Eval) -> dict:
    """Lens-free: what each recurrent exit itself verbalizes at the readout position.
    The logit lens at (ut, 47) is exactly the actual exit-ut logits."""
    a = ev.arrays
    exits = [u * N_LAYER + N_LAYER - 1 for u in range(N_UT)]
    res = {}
    for task in ("multihop", "order-ops numeric"):
        m = ev.slot_mask[task]
        if not m.sum():
            continue
        sc = ev.scores(a["logitlens_allrank"], m)
        ii, jj = ev.own_slots(m)
        own_rank = a["logitlens_allrank"][ii, jj][:, exits]           # [slots, 4]
        raw_items = np.asarray([i for i, item in enumerate(ev.items)
                                if item["task"] == ("order-ops" if task.startswith("order-ops") else task)])
        top1_final = a["exit_top1"][raw_items, N_UT - 1]
        res[task] = {
            "intermediate_is_exit_top1": by_item((own_rank == 0).astype(float), ii).mean(0).round(3).tolist(),
            "intermediate_cand_top1": by_item(sc["cand_top1"][:, exits], ii).mean(0).round(3).tolist(),
            "intermediate_excess10": by_item(sc["excess"][:, exits], ii).mean(0).round(3).tolist(),
            "exit_top1_equals_final_top1_all_items": [
                round(float((a["exit_top1"][raw_items, u] == top1_final).mean()), 3)
                for u in range(N_UT)
            ],
            "n_raw_items": int(len(raw_items)),
            "n_items": int(len(np.unique(ii))),
        }
    # Lens-free divergence between recurrent exits: the logit lens at (u, 47) IS the exit-u logits.
    boundary = [u * N_LAYER + N_LAYER - 1 for u in range(N_UT)]
    kl = a["logitlens_kl_to_final"][:, boundary]
    res["exit_divergence"] = {"kl_exit_k_to_exit_4_mean": kl.mean(0).round(3).tolist(),
                              "kl_exit_k_to_exit_4_median": np.median(kl, 0).round(3).tolist()}
    return res


def local_vs_eventual(ev: Eval) -> dict | None:
    a = ev.arrays
    exits = [u for u in range(N_UT - 1) if f"jlens_exit{u}_allrank" in a]
    if not exits:
        return None
    res = {}
    for u in exits:
        rows = slice(u * N_LAYER, (u + 1) * N_LAYER)
        r = {
            "kl_local_to_eventual_readout": a[f"jlens_exit{u}_kl_to_eventual_readout"][:, rows].mean(0).round(3).tolist(),
            "top1_agreement_local_vs_eventual": (a[f"jlens_exit{u}_top1"][:, rows] == a["jlens_exit3_top1"][:, rows]).mean(0).round(3).tolist(),
            "local_lens_kl_to_actual_local_exit": a[f"jlens_exit{u}_kl_to_local"][:, rows].mean(0).round(3).tolist(),
            "local_lens_kl_to_actual_final": a[f"jlens_exit{u}_kl_to_final"][:, rows].mean(0).round(3).tolist(),
            "eventual_lens_kl_to_actual_final": a["jlens_exit3_kl_to_final"][:, rows].mean(0).round(3).tolist(),
            "actual_exit_top1_equals_final_top1": round(float((a["exit_top1"][:, u] == a["exit_top1"][:, N_UT - 1]).mean()), 3),
            "top1_agreement_last8_layers": round(float((a[f"jlens_exit{u}_top1"][:, rows][:, -8:] == a["jlens_exit3_top1"][:, rows][:, -8:]).mean()), 3),
            # KL grows with the sharpness of the target exit, so read it beside the rank measures.
            "median_rank_of_own_exit_top1_last8": np.median(a[f"jlens_exit{u}_rank_of_local_top1"][:, rows][:, -8:], 0).round(0).tolist(),
        }
        for task in ("multihop", "order-ops numeric"):
            m = ev.slot_mask[task]
            for lens, key in (("local", f"jlens_exit{u}_allrank"), ("eventual", "jlens_exit3_allrank")):
                r[f"excess10_{lens}_{task}"] = loc_maps(ev.scores(a[key], m))["excess"][u].round(3).tolist()
        res[f"loop{u+1}"] = r
    return res


def fig_local_vs_eventual(lve: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for name, r in lve.items():
        u = int(name[-1]) - 1
        axes[0].plot(r["kl_local_to_eventual_readout"], color=LOOP_COLORS[u], lw=2, label=f"state at {name}")
        axes[1].plot(r["top1_agreement_local_vs_eventual"], color=LOOP_COLORS[u], lw=2, label=name)
        axes[2].plot(r["excess10_local_multihop"], color=LOOP_COLORS[u], lw=2, ls="--", label=f"{name} local-exit lens")
        axes[2].plot(r["excess10_eventual_multihop"], color=LOOP_COLORS[u], lw=2, label=f"{name} eventual-exit lens")
    axes[0].set_title("KL(local-exit readout || eventual-exit readout), nats", fontsize=10)
    axes[1].set_title("top-1 agreement: local-exit vs eventual-exit readout", fontsize=10)
    axes[2].set_title("multihop excess hit@10: local-exit (dashed) vs eventual-exit (solid)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("physical layer")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[1].set_ylim(0, 1)
    fig.suptitle("Monitor gap: what a loop-k exit monitor reads vs what the completed recurrence is disposed to say")
    fig.savefig(out / "fig4_local_vs_eventual.png", dpi=150)
    plt.close(fig)


def probe_summary(probe_dir: Path, out: Path) -> dict:
    if not (probe_dir / "meta.json").exists():
        summary = json.loads((probe_dir / "summary.json").read_text())
        if summary.get("schema_version") != 1 or "readouts" not in summary:
            raise ValueError(f"unsupported probe summary in {probe_dir}")
        return summary
    a = dict(np.load(probe_dir / "arrays.npz"))
    meta = json.loads((probe_dir / "meta.json").read_text())
    res = {"meta": meta, "chance": 1 / len(meta["labels"])}
    fig, axes = plt.subplots(1, N_UT, figsize=(16, 3.8), constrained_layout=True, sharey=True)
    for name, key, color, ls in [("supervised probe", "probe_rank", "#52514e", "-"), ("Jacobian lens", "jlens_cand_rank", "#2a78d6", "-"),
                                 ("logit lens", "logitlens_cand_rank", "#eb6834", "--")]:
        acc = (a[key] == 0).mean(0).reshape(N_UT, N_LAYER)
        res[name] = {"top1_by_location": acc.round(3).tolist(), "best": round(float(acc.max()), 3),
                     "best_location": [int(x) for x in np.unravel_index(acc.argmax(), acc.shape)],
                     "per_loop_max": acc.max(1).round(3).tolist()}
        for u in range(N_UT):
            axes[u].plot(acc[u], color=color, ls=ls, lw=2, label=name)
    for u in range(N_UT):
        axes[u].axhline(res["chance"], color="#9e9d98", lw=1, ls=":")
        axes[u].set_title(f"loop {u+1}: 17-way top-1 on held-out (a, b) pairs", fontsize=10)
        axes[u].set_xlabel("physical layer")
        axes[u].set_ylim(0, 1)
        axes[u].grid(alpha=0.25)
    axes[0].set_ylabel("top-1 accuracy (candidate set)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Intermediate a+b in '(a + b) * c = ': supervised reference vs lenses (n_test={meta['n_test']}, model acc {meta['model_accuracy_all']:.2f})")
    fig.savefig(out / "fig5_probe_vs_lens.png", dpi=150)
    plt.close(fig)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True)
    p.add_argument("--probe")
    args = p.parse_args()
    out = Path(args.eval)
    ev = Eval(out)
    a = ev.arrays
    summary: dict = {"n_items": len(ev.items), "model_correct": int(sum(i["correct"] for i in ev.items)),
                     "group_sizes": {k: int(v.sum()) for k, v in ev.slot_mask.items()}}
    maps: dict = {}
    for task, mask in ev.slot_mask.items():
        if mask.sum() == 0:
            continue
        summary[task] = {}
        for method, key in (("Jacobian lens", "jlens_exit3_allrank"), ("logit lens", "logitlens_allrank")):
            sc = ev.scores(a[key], mask)
            lm = loc_maps(sc)
            maps.setdefault(task, {})[method] = lm
            ex_items = by_item(sc["excess"], sc["items"])
            best = int(lm["excess"].argmax())
            summary[task][method] = {
                **any_layer(sc),
                "excess10_max_per_loop": lm["excess"].max(1).round(3).tolist(),
                "cand_top1_max_per_loop": lm["cand_top1"].max(1).round(3).tolist(),
                "best_excess_location_ut_layer": [best // N_LAYER, best % N_LAYER],
                "selected_location_ci95_conditional": boot_ci(ex_items, lambda v: v.mean(0)[best]),
                "n_items": int(len(ex_items)), "n_slots": int(len(sc["hit"])),
            }
    xl = {task: cross_loop_summary(ev, ev.slot_mask[task]) for task in ("multihop", "order-ops numeric") if ev.slot_mask[task].sum()}
    summary["cross_loop"] = xl
    summary["model_exits"] = model_exits(ev)
    lve = local_vs_eventual(ev)
    if lve:
        summary["local_vs_eventual"] = lve
        fig_local_vs_eventual(lve, out)
    if args.probe:
        summary["probe"] = probe_summary(Path(args.probe), out)
    primary = {t: maps[t] for t in ("multihop", "order-ops numeric") if t in maps}
    fig_heatmaps(primary, out)
    fig_lines(primary, out)
    fig_xloop(xl, out)
    atomic_write_json(out / "summary.json", summary)

    lines = [f"# Summary ({out})", "", f"items {len(ev.items)}, model correct on target {summary['model_correct']}", "",
             "| group | method | n items | pass@10 per loop (any layer) | control pass@10 | cand top-1 (any layer) | max excess@10 per loop |",
             "|---|---|---|---|---|---|---|"]
    for task in ev.slot_mask:
        for method in ("Jacobian lens", "logit lens"):
            if task in summary and method in summary[task]:
                s = summary[task][method]
                lines.append(f"| {task} | {method} | {s['n_items']} | {s['pass10_per_loop']} | {s['control_pass10_per_loop']} | {s['cand_top1_any_layer_per_loop']} | {s['excess10_max_per_loop']} |")
    lines += ["", "## Model's own exits (lens-free): intermediate at exit k, per loop 1..4",
              f"- KL(exit k || exit 4), nats: mean {summary['model_exits']['exit_divergence']['kl_exit_k_to_exit_4_mean']}, "
              f"median {summary['model_exits']['exit_divergence']['kl_exit_k_to_exit_4_median']}"]
    for task, r in summary["model_exits"].items():
        if task == "exit_divergence":
            continue
        lines.append(f"- {task} (n={r['n_items']}): intermediate is exit top-1 {r['intermediate_is_exit_top1']}; "
                     f"cand top-1 {r['intermediate_cand_top1']}; excess@10 {r['intermediate_excess10']}; "
                     f"exit-k top-1 == final top-1 over all raw task items "
                     f"{r['exit_top1_equals_final_top1_all_items']}")
    lines += ["", "## Cross-loop transfer (rows = J fit loop, cols = state loop)"]
    for task, r in xl.items():
        lines += [f"**{task}** (n_items={r['n_items']}), excess hit@10 contrasts:",
                  f"- preregistered H1 (loop 1 in either role) {r['h1_preregistered']:+.3f} CI {r['h1_preregistered_ci95']}",
                  f"- state-only (loop-1 states; descriptive role contrast) {r['h1_state_only']:+.3f} CI {r['h1_state_only_ci95']}",
                  f"- fit-only (J fitted at loop 1; descriptive horizon correlate) {r['h1_fit_only']:+.3f} CI {r['h1_fit_only_ci95']}",
                  f"- variance share: fit loop {r['variance_share']['fit_loop']}, state loop {r['variance_share']['state_loop']}, interaction {r['variance_share']['interaction']}", ""]
        for key in ("excess_hit10_fit_by_state", "cand_top1_fit_by_state"):
            lines += [f"{key}:", "| | " + " | ".join(f"state L{u+1}" for u in range(N_UT)) + " |", "|---" * (N_UT + 1) + "|"]
            lines += [f"| fit L{i+1} | " + " | ".join(f"{v:.2f}" for v in row) + " |" for i, row in enumerate(r[key])]
            lines.append("")
    if lve:
        lines += ["## Local vs eventual exit (per loop: mean over layers of KL(local||eventual), top-1 agreement)"]
        lines += ["", "| state | task | local-exit lens excess@10 | eventual-exit lens excess@10 | top-1 agree (last 8 layers) | actual exit top-1 == final |", "|---|---|---|---|---|---|"]
        for name, r in lve.items():
            for task in ("multihop", "order-ops numeric"):
                loc, evl = np.array(r[f"excess10_local_{task}"]), np.array(r[f"excess10_eventual_{task}"])
                lines.append(f"| {name} | {task} | {loc.max():.3f} (L{loc.argmax()}) | {evl.max():.3f} (L{evl.argmax()}) | "
                             f"{r['top1_agreement_last8_layers']:.3f} | {r['actual_exit_top1_equals_final_top1']} |")
    if args.probe:
        pr = summary["probe"]
        if "readouts" in pr:
            lines += ["", "## Supervised reference (576 fold-trainable prompts; cross-fitted layer)"]
            for name, values in pr["readouts"].items():
                lines.append(f"- {name}: {values['per_loop']} CI {values['ci95']}")
            lines.append(f"- baselines: {pr['baselines']}")
        else:
            lines += ["", "## Legacy supervised reference (single split; descriptive per-loop maxima)"]
            for name in ("supervised probe", "Jacobian lens", "logit lens"):
                lines.append(f"- {name}: {pr[name]['per_loop_max']} (best {pr[name]['best']} at ut{pr[name]['best_location'][0]} L{pr[name]['best_location'][1]}); chance {pr['chance']:.3f}")
    atomic_write_text(out / "summary.md", "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
