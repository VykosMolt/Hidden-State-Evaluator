#!/usr/bin/env python3
"""Figures for RPE v3. Reads only the result files in ../results (each already checked against its source; see
figure_data.py) and, for Figure 1, the version-1 and version-2 values quoted in Table 3. Writes ../figures/*.pdf.
Run from any directory (CPU only):

    python make_figures.py
"""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from config import PKG, RESULTS as RES  # release copy: reads RPE_RESULTS (default ../results), writes ../figures
from paper_style import figure_style as st  # noqa: E402

HERE = PKG / "figures"

st.apply(paper=1)
J = lambda name: json.load(open(RES / name))  # noqa: E731
PT, PR = J("probes_thinking_converged.json"), J("probes_rltt_left_converged.json")  # probes fit to convergence
ET, ER = J("epochs_thinking_1024.json")["epochs"], J("epochs_rltt.json")["epochs"]
EPOCHS = [1, 2, 3, 4, 5]
V1_EPOCHS = [83.3, 95.2, 89.5, 67.2, 62.4]  # version 1's reported fixed-order test accuracy (Table 2)


def save(fig, name):
    st.save(fig, HERE / name)
    plt.close(fig)
    print("wrote", name)


def r1(v):
    """One decimal, halves rounded up, as in the text (0.5495 -> 55.0)."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(round(float(v), 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def acc(p, k):
    v = p["acc"][k]
    return (100 * v["all_ties_half"][0], [100 * x for x in v["all_ties_half"][1]]) if isinstance(v, dict) else (100 * v, None)


# ---------------------------------------------------------------------------
# Figure 1: the three headline readouts in version 1, version 2 and here (values of Table 3).
def fig_summary():
    rows = [  # label, v1 (as reported), v2, v3
        ("Nonlinear evaluator\n(test pairs)", "95.2", 63.9, 100 * ET["2"]["strict_antisym_acc"][0]),
        ("Relational linear probe\n(held-out pairs)", "84.5", 56.5, acc(PT, "relational_final")[0]),
        ("Single-response probe\n(held-out replies)", "21.75", 54.2, acc(PT, "pointwise_final_candidate")[0]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.25))
    for i, (lab, v1s, v2, v3) in enumerate(rows):
        y, v1 = len(rows) - 1 - i, float(v1s)
        ax.annotate("", xy=(v3, y), xytext=(v1, y), arrowprops=dict(arrowstyle="-|>", color=st.RULE, lw=2.2,
                                                                     shrinkA=6, shrinkB=6))
        ax.plot(v1, y, "o", mfc=st.WHITE, mec=st.NEGATIVE, mew=1.6, ms=7.5, zorder=3)
        ax.plot(v2, y, "D", color=st.BASELINE, ms=5.5, zorder=3)
        ax.plot(v3, y, "o", color=st.P1, ms=7.5, zorder=4)
        for v, t, c, dy in ((v1, v1s, st.NEGATIVE, 0.22), (v3, r1(v3), st.P1, -0.36)):
            ax.text(v, y + dy, t, ha="center", va="bottom", fontsize=7.8, color=c)
    ax.axvline(50, color=st.BASELINE, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(50.6, len(rows) - 0.45, "chance", fontsize=7.6, color=st.MUTED, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.4)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.35)
    ax.set_xlabel("Accuracy (%)")
    ax.grid(axis="y", visible=False)
    h = [plt.Line2D([], [], ls="", marker="o", mfc=st.WHITE, mec=st.NEGATIVE, mew=1.6, ms=7, label="Version 1"),
         plt.Line2D([], [], ls="", marker="D", color=st.BASELINE, ms=5.5, label="Version 2 (erratum)"),
         plt.Line2D([], [], ls="", marker="o", color=st.P1, ms=7, label="This version")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, handletextpad=0.3, columnspacing=1.6)
    save(fig, "fig_summary.pdf")


# ---------------------------------------------------------------------------
# Figure 2: probes on the 8,000 held-out pairs, both backbones.
def fig_probes():
    rows = [("Relational, last loop", "relational_final"),
            ("Single response, last loop", "pointwise_final_induced_pairwise"),
            ("Relational, all loops", "relational_all4"),
            ("Single response, all loops", "pointwise_all4_induced_pairwise"),
            ("Length and norms (logistic)", "surface_logistic"),
            ("Larger-norm rule (last loop)", "norm_final_rule"),
            ("Longer-reply rule", "length_rule")]
    diffs = [("Relational $-$ single,\nlast loop", "relational_final - pointwise_final_induced_pairwise"),
             ("Relational $-$ single,\nall loops", "relational_all4 - pointwise_all4_induced_pairwise"),
             ("Relational (last loop)\n$-$ length and norms", "relational_final - surface_logistic")]
    fig = plt.figure(figsize=(7.1, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.95)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    series = (("Ouro-Thinking", PT, st.P1, "o", 0.16), ("Ouro-RLTT", PR, st.P2, "s", -0.16))
    for name, p, c, m, dy in series:
        for i, (_, k) in enumerate(rows):
            a, ci = acc(p, k)
            y = len(rows) - 1 - i + dy
            axA.plot(ci, [y, y], color=c, lw=1.4, solid_capstyle="butt")
            axA.plot(a, y, m, color=c, ms=4.6, label=name if i == 0 else None)
        for i, (_, k) in enumerate(diffs):
            d, (lo, hi) = p["paired"][k]["all"]
            y = len(diffs) - 1 - i + dy
            axB.plot([100 * lo, 100 * hi], [y, y], color=c, lw=1.4, solid_capstyle="butt")
            axB.plot(100 * d, y, m, color=c, ms=4.6)
    axA.axvline(50, color=st.BASELINE, lw=0.9, ls=(0, (3, 2)))
    axA.axhline(2.5, color=st.RULE, lw=0.8)
    axA.set_yticks(range(len(rows)))
    axA.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.2)
    axA.set_xlim(48, 64)
    axA.set_xlabel("Accuracy at picking the preferred reply (%)")
    axA.grid(axis="y", visible=False)
    axA.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2)
    axB.axvline(0, color=st.BASELINE, lw=0.9, ls=(0, (3, 2)))
    axB.set_yticks(range(len(diffs)))
    axB.set_yticklabels([d[0] for d in diffs][::-1], fontsize=8.2)
    axB.set_xlabel("Paired difference (points)")
    axB.grid(axis="y", visible=False)
    st.panel_label(axA, "A", x=-0.62)
    st.panel_label(axB, "B", x=-0.72)
    save(fig, "fig_probes.pdf")


# ---------------------------------------------------------------------------
# Figure 3: position bias of version 1's evaluator (test split, 8,552 pairs).
def fig_position():
    z = np.load(RES / "evaluator_pair_scores_thinking.npz")
    e = list(z["epochs"]).index(2)
    x, y = z["normal"][:, e].astype(float), z["flipped"][:, e].astype(float)  # preferred first, preferred second
    n = len(x)
    q = {"pf": ((x > 0) & (y > 0)).mean(), "ok": ((x > 0) & (y <= 0)).mean(),
         "bad": ((x <= 0) & (y > 0)).mean(), "ps": ((x <= 0) & (y <= 0)).mean()}
    assert abs(q["pf"] - ET["2"]["both_orders_prefer_first"]) < 1e-9
    fig = plt.figure(figsize=(7.1, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 0.85], wspace=0.5)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))
    lim = float(np.ceil(np.percentile(np.abs(np.r_[x, y]), 99.5)))
    cmap = LinearSegmentedColormap.from_list("p1", st.SEQ_BLUES[1:])
    axA.hexbin(np.clip(x, -lim, lim), np.clip(y, -lim, lim), gridsize=38, extent=(-lim, lim, -lim, lim),
               cmap=cmap, norm=LogNorm(), mincnt=1, linewidths=0.1, edgecolors=st.WHITE)
    axA.axhline(0, color=st.MUTED, lw=0.7)
    axA.axvline(0, color=st.MUTED, lw=0.7)
    axA.plot([-lim, lim], [-lim, lim], color=st.NEGATIVE, lw=0.9, ls=(0, (3, 2)))
    for (tx, ty, ha, va, key, lab) in ((0.97, 0.97, "right", "top", "pf", "first reply\nin both orders"),
                                       (0.97, 0.03, "right", "bottom", "ok", "correct in\nboth orders"),
                                       (0.03, 0.97, "left", "top", "bad", "wrong in\nboth orders"),
                                       (0.03, 0.03, "left", "bottom", "ps", "second reply\nin both orders")):
        axA.text(tx, ty, f"{r1(100 * q[key])}%\n{lab}", transform=axA.transAxes, ha=ha, va=va, fontsize=7.0,
                 color=st.INK, linespacing=1.05, bbox=dict(fc=st.WHITE, ec="none", alpha=0.85, pad=0.6))
    axA.set_xlim(-lim, lim)
    axA.set_ylim(-lim, lim)
    axA.set_aspect("equal")
    axA.grid(False)
    axA.set_xlabel("Score, preferred reply first")
    axA.set_ylabel("Score, preferred reply second")
    # B: accuracy by epoch
    oi = np.array([100 * ET[str(k)]["strict_antisym_acc"][0] for k in EPOCHS])
    lo = np.array([100 * ET[str(k)]["strict_antisym_acc"][1][0] for k in EPOCHS])
    hi = np.array([100 * ET[str(k)]["strict_antisym_acc"][1][1] for k in EPOCHS])
    axB.plot(EPOCHS, V1_EPOCHS, "o", ls=(0, (3, 2)), mfc=st.WHITE, mec=st.NEGATIVE, color=st.NEGATIVE, mew=1.3, ms=5,
             lw=1.0, label="Version 1 (reported)")
    axB.plot(EPOCHS, [100 * ET[str(k)]["fixed_order_acc"] for k in EPOCHS], "-", color=st.BASELINE, marker="D",
             ms=3.6, label="Preferred reply first")
    axB.fill_between(EPOCHS, lo, hi, color=st.P1_PALE, lw=0)
    axB.plot(EPOCHS, oi, "-o", color=st.P1, ms=4.2, label="Both orders")
    axB.plot(EPOCHS, [100 * ER[str(k)]["strict_antisym_acc"][0] for k in EPOCHS], ls=(0, (2, 1.5)), color=st.P2,
             lw=1.2, label="Both orders, Ouro-RLTT")
    axB.set_xticks(EPOCHS)
    axB.set_ylim(40, 100)
    axB.set_xlabel("Training epoch")
    axB.set_ylabel("Test accuracy (%)")
    axB.legend(loc="lower right", fontsize=6.9, handlelength=1.8, labelspacing=0.25, borderaxespad=0.1)
    # C: order offset by epoch
    axC.axhline(0, color=st.BASELINE, lw=0.9, ls=(0, (3, 2)))
    axC.plot(EPOCHS, [ET[str(k)]["mean_symmetric_component"] for k in EPOCHS], "-o", color=st.P1, ms=4.2,
             label="Ouro-Thinking")
    axC.plot(EPOCHS, [ER[str(k)]["mean_symmetric_component"] for k in EPOCHS], ls=(0, (2, 1.5)), color=st.P2, lw=1.2,
             label="Ouro-RLTT")
    axC.set_xticks(EPOCHS)
    axC.set_xlabel("Training epoch")
    axC.set_ylabel("Mean order part of the score")
    axC.set_ylim(-0.45, 1.45)
    axC.legend(loc="lower left", fontsize=6.9, handlelength=1.8, borderaxespad=0.1)
    for ax, lab, xo in ((axA, "A", -0.32), (axB, "B", -0.3), (axC, "C", -0.36)):
        st.panel_label(ax, lab, x=xo)
    print(f"  position: n={n}, quadrants " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in q.items()))
    save(fig, "fig_position.pdf")


# ---------------------------------------------------------------------------
# Figure 4: where the two replies of a held-out pair first differ, against the 384-token limit.
def fig_truncation():
    F = J("first_difference_heldout.json")
    k = np.array([r["first_difference"] for r in F["rows"] if r["first_difference"] is not None])
    L = F["max_length"]
    beyond, same, n = int((k >= L).sum()), F["n_texts_identical"], F["n_pairs"]
    assert beyond == F["n_first_difference_at_or_beyond_384"] and beyond + same == F["n_identical_at_384_right"]
    fig, ax = plt.subplots(figsize=(4.6, 2.35))
    w = (np.log10(k.max() + 1) - np.log10(max(1, k.min()))) / 45  # log-width of a bin; one edge exactly at the limit
    lo_n = int(np.ceil((np.log10(L) - np.log10(max(1, k.min()))) / w))
    hi_n = int(np.ceil((np.log10(k.max() + 1) - np.log10(L)) / w))
    bins = 10 ** (np.log10(L) + w * np.arange(-lo_n, hi_n + 1))
    bins[lo_n] = L  # exact, so a pair whose first difference is at token 384 falls in the first red bin
    cnt, edges, patches = ax.hist(k, bins=bins, color=st.P1, alpha=0.95)
    for p, left in zip(patches, edges[:-1]):
        if left >= L - 1e-9:
            p.set_facecolor(st.NEGATIVE)
    red = int(sum(c for c, left in zip(cnt, edges[:-1]) if left >= L - 1e-9))
    assert red == beyond, (red, beyond)  # every pair beyond the limit is drawn red, and only those
    ax.axvline(L, color=st.INK, lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 100, 384, 1000])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.text(L * 1.08, ax.get_ylim()[1] * 0.97, f"{L}-token limit", fontsize=7.6, color=st.INK, va="top")
    ax.text(L * 1.08, ax.get_ylim()[1] * 0.80,
            f"{beyond} pairs ({r1(100 * beyond / n)}%) differ only\nafter the limit and become identical\n"
            f"when truncated from the right", fontsize=7.2, color=st.NEGATIVE, va="top", linespacing=1.1)
    ax.set_xlabel("Token position where the two texts first differ")
    ax.set_ylabel("Held-out pairs")
    ax.grid(axis="x", visible=False)
    print(f"  truncation: {beyond} beyond {L} + {same} identical texts = {beyond + same} of {n}")
    save(fig, "fig_truncation.pdf")


if __name__ == "__main__":
    fig_summary()
    fig_probes()
    fig_position()
    fig_truncation()
