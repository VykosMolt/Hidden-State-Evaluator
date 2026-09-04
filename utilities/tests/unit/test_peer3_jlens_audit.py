"""PEER3 regression tests for the two jlens defects that were fixed tonight and the one
that is still open. Pure CPU, no model, no artifacts required except where marked.

Written by PEER3 during the swarm audit of docs/jlens/RESULTS.md. Each test locks in a
specific bug that was actually present in an earlier draft, so a silent revert is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from ouro_jlens import analyze
from ouro_jlens.probe import make_prompts
from ouro_jlens.probe_cv import (
    N_FOLDS,
    N_VAL_PAIRS,
    fold_assignment,
    prompt_folds,
    unordered_pairs,
)


# --------------------------------------------------------------------------- #
# 1. The any-layer control must be scored per control name, then averaged.
#    The bug: max_over_layers(mean_over_names(hit)) instead of
#             mean_over_names(max_over_layers(hit)). Jensen makes the first <= the
#    second, so every any-layer excess was inflated.
# --------------------------------------------------------------------------- #

def _jensen_gap_example() -> np.ndarray:
    """[n_names, 192] where each name hits at a different layer: mean-of-maxes is 1.0 and
    max-of-means is 1/n_names, the largest possible gap."""
    n_names = 4
    r = np.zeros((n_names, analyze.N_UT * analyze.N_LAYER))
    for k in range(n_names):
        r[k, k] = 1.0  # all inside loop 0, different layers
    return r


def test_any_layer_control_is_mean_of_maxes_not_max_of_means():
    r = _jensen_gap_example()
    per_name = analyze._any_layer_per_name(r)          # [n_names, N_UT]
    assert per_name.shape == (4, analyze.N_UT)
    correct = per_name.mean(0)[0]
    buggy = analyze._split_layers(r.mean(0)).max(-1)[0]
    assert correct == pytest.approx(1.0)
    assert buggy == pytest.approx(0.25)
    assert correct > buggy, "the Jensen bug is back: control scored as max of a mean"


def test_split_layers_dispatches_on_192_not_48():
    """The cross-loop tensor's trailing axis is 48, not 192; it must be left alone so that
    .max(-1) collapses the layer axis of [n, 4(fit), 4(state), 48]."""
    main = np.zeros((7, analyze.N_UT * analyze.N_LAYER))
    assert analyze._split_layers(main).shape == (7, analyze.N_UT, analyze.N_LAYER)
    xloop = np.zeros((7, analyze.N_UT, analyze.N_UT, analyze.N_LAYER))
    assert analyze._split_layers(xloop).shape == xloop.shape
    assert analyze._any_layer_per_name(xloop).shape == (7, analyze.N_UT, analyze.N_UT)


def test_drop_layer_axis_matches_any_layer_output_on_both_paths():
    for shape in [(11, analyze.N_UT * analyze.N_LAYER),
                  (11, analyze.N_UT, analyze.N_UT, analyze.N_LAYER)]:
        got = analyze._drop_layer_axis(shape)
        want = analyze._any_layer_per_name(np.zeros(shape)).shape
        assert got == want, f"_drop_layer_axis{shape} = {got}, but _any_layer_per_name gives {want}"


def test_boot_ci_does_not_depend_on_call_order():
    """The bug: a module-level RNG advanced between calls, so a CI depended on how many
    earlier calls had drawn from the stream."""
    v = np.linspace(0.0, 1.0, 40)[:, None]
    stat = lambda x: x.mean(0)[0]
    first = analyze.boot_ci(v, stat)
    for _ in range(3):
        analyze.boot_ci(np.random.default_rng(1).random((40, 1)), stat)
    assert analyze.boot_ci(v, stat) == first


# --------------------------------------------------------------------------- #
# 2. Probe folds must split on the UNORDERED operand pair.
#    The bug: (3, 5) in train with (5, 3) in test handed the probe the answer.
# --------------------------------------------------------------------------- #

def test_probe_folds_keep_mirror_pairs_together():
    prompts = make_prompts()
    folds = prompt_folds(prompts, seed=0)
    by_ordered: dict[tuple[int, int], set[int]] = {}
    for q, f in zip(prompts, folds):
        by_ordered.setdefault((q["a"], q["b"]), set()).add(int(f))
    for (a, b), fs in by_ordered.items():
        assert len(fs) == 1, f"ordered pair {(a, b)} spans folds {fs}"
        assert fs == by_ordered[(b, a)], f"{(a, b)} and {(b, a)} are in different folds"


def test_probe_folds_partition_all_45_pairs_evenly():
    assign = fold_assignment(seed=0)
    assert set(assign) == set(unordered_pairs()) and len(assign) == 45
    counts = np.bincount(list(assign.values()), minlength=N_FOLDS)
    assert counts.tolist() == [9] * N_FOLDS


def test_probe_folds_have_no_train_test_pair_overlap():
    prompts = make_prompts()
    folds = prompt_folds(prompts, seed=0)
    pair_of = [(min(q["a"], q["b"]), max(q["a"], q["b"])) for q in prompts]
    for f in range(N_FOLDS):
        te = {pair_of[i] for i in np.where(folds == f)[0]}
        tr = {pair_of[i] for i in np.where(folds != f)[0]}
        assert not (te & tr)


# --------------------------------------------------------------------------- #
# 3. OPEN DEFECT, documented not asserted-away: some test labels are absent from
#    their own fold's training set, so the probe cannot possibly predict them while
#    the two lenses have no such handicap. This test pins the size of the handicap
#    so that a change to the fold seed or to N_VAL_PAIRS cannot silently move it.
# --------------------------------------------------------------------------- #

def _unpredictable_mask(seed: int = 0) -> np.ndarray:
    prompts = make_prompts()
    labels = np.array([q["label"] for q in prompts])
    pair_of = [(min(q["a"], q["b"]), max(q["a"], q["b"])) for q in prompts]
    folds = prompt_folds(prompts, seed)
    rng = np.random.default_rng(seed + 1)  # same stream cv_probe uses
    bad = np.zeros(len(prompts), bool)
    for f in range(N_FOLDS):
        te = folds == f
        train_pairs = sorted({pair_of[i] for i in np.where(~te)[0]})
        val_pairs = {train_pairs[i] for i in rng.permutation(len(train_pairs))[:N_VAL_PAIRS]}
        va = np.array([p in val_pairs for p in pair_of]) & ~te
        seen = set(labels[~te & ~va | va].tolist())
        bad |= te & ~np.isin(labels, list(seen))
    return bad


def test_label_coverage_handicap_is_the_documented_size():
    """RESULTS.md section 8 states 72 of 648. If this number moves, the probe's ceiling
    moved and section 8's probe-versus-lens comparison must be re-derived."""
    bad = _unpredictable_mask(0)
    assert int(bad.sum()) == 72, f"handicap changed: {int(bad.sum())}/648"
    assert 1.0 - bad.mean() == pytest.approx(0.8889, abs=5e-4)


def test_singleton_labels_are_the_root_cause():
    """Labels 2, 3, 17 and 18 come from exactly one unordered pair each, so whichever fold
    holds that pair can never train on the label."""
    n_pairs: dict[int, int] = {}
    for a in range(1, 10):
        for b in range(a, 10):
            n_pairs[a + b] = n_pairs.get(a + b, 0) + 1
    assert sorted(l for l, n in n_pairs.items() if n == 1) == [2, 3, 17, 18]
