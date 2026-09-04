"""Cross-validated supervised probe on ``(a + b) * c = ``.

`probe.py` fits one probe on 384 prompts from 24 of the 45 unordered operand pairs and
scores it on a fixed 176-prompt test split. That is under-powered, which is why an
untrained logit lens beats it at loop 1, and it leaves the comparison on a quarter of
the family. Here the 45 unordered pairs are split into 5 folds: each prompt is a test
prompt in exactly one fold, C is chosen inside the fold on validation pairs drawn from
that fold's training pairs only, and the probe is refit on all training pairs before
scoring the held-out ones. Because a+b is symmetric, unordered pairs are the split unit
throughout, so a test pair's mirror is never in training (asserted per fold).

All 648 prompts receive a held-out score, but 72 have a label absent from their
training fold.  They are retained in the raw arrays and explicitly excluded by
``probe_report.py`` from the fair 576-prompt comparison.  Layer selection for
the reported comparison is also cross-fitted; a max over these held-out scores
is descriptive only and is not the reported estimand.

Hidden states are reused from `probe.py`'s GPU cache (`gpu_cache.npz`, key "H",
[648, 192, 2048] fp16), so the probe needs no GPU. The two lenses are re-scored on all
648 prompts with `probe.lens_candidate_ranks` (the same code path `probe.py` uses on its
176) so the three readouts are compared on identical data; that step does need the GPU
and is skipped if its cache is already present.

  # GPU step (needs the lock), writes lens_all648.npz next to --out
  python src/ouro_jlens/probe_cv.py --cache artifacts/jlens/probe/n80_v2/gpu_cache.npz \
      --lens artifacts/jlens/lens/exit3/exit3_n80.pt --out <dir> --lens-only
  # CPU step
  python src/ouro_jlens/probe_cv.py --cache artifacts/jlens/probe/n80_v2/gpu_cache.npz \
      --lens artifacts/jlens/lens/exit3/exit3_n80.pt --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouro_jlens.evidence import (  # noqa: E402
    atomic_savez,
    atomic_write_json,
    file_record,
    source_manifest,
)
from ouro_jlens.probe import C_GRID, LABELS, make_prompts  # noqa: E402

N_UT, N_LAYER = 4, 48
N_FOLDS = 5
N_VAL_PAIRS = 8  # validation pairs held out of the 36 training pairs, for choosing C


def unordered_pairs() -> list[tuple[int, int]]:
    return [(a, b) for a in range(1, 10) for b in range(a, 10)]


def fold_assignment(seed: int = 0) -> dict[tuple[int, int], int]:
    """Assign each of the 45 unordered pairs to one of 5 folds (9 pairs each)."""
    pairs = unordered_pairs()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    return {pairs[p]: int(k % N_FOLDS) for k, p in enumerate(order)}


def prompt_folds(prompts: list[dict], seed: int = 0) -> np.ndarray:
    """Fold index per prompt, keyed by the *unordered* pair so mirrors share a fold."""
    assign = fold_assignment(seed)
    return np.array([assign[(min(q["a"], q["b"]), max(q["a"], q["b"]))] for q in prompts], int)


def validation_pairs(folds: np.ndarray, pair_of: list[tuple[int, int]], seed: int = 0
                     ) -> list[list[tuple[int, int]]]:
    """Persistable validation-pair selection used for regularisation tuning."""
    rng = np.random.default_rng(seed + 1)
    selected = []
    for fold in range(N_FOLDS):
        train_pairs = sorted({pair_of[i] for i in np.where(folds != fold)[0]})
        selected.append(sorted(train_pairs[i] for i in rng.permutation(len(train_pairs))[:N_VAL_PAIRS]))
    return selected


def _scaled(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return StandardScaler().fit(X[rows]).transform(X)


def _fit_location(H: np.ndarray, location: int, labels: np.ndarray, tr: np.ndarray,
                  va: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit/tune one location.  Scaling is shared across C candidates."""
    X = H[:, location]
    Xs = _scaled(X.astype(np.float32), tr)
    best, best_acc = C_GRID[0], -1.0
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000).fit(Xs[tr], labels[tr])
        acc = clf.score(Xs[va], labels[va])
        if acc > best_acc:
            best, best_acc = C, acc
    fit_rows = tr | va
    Xs = _scaled(X.astype(np.float32), fit_rows)
    clf = LogisticRegression(C=best, max_iter=2000).fit(Xs[fit_rows], labels[fit_rows])
    return _rank_of_true(clf, Xs, labels, te), best


def _rank_of_true(clf, Xs: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Rank of the true label among the 17 candidates. Labels absent from the training
    fold keep probability 0, exactly as in probe.py."""
    proba = np.zeros((rows.sum(), len(LABELS)))
    proba[:, [LABELS.index(c) for c in clf.classes_]] = clf.predict_proba(Xs[rows])
    true_p = proba[np.arange(len(proba)), [LABELS.index(l) for l in y[rows]]]
    return (proba > true_p[:, None]).sum(1)


def cv_probe(H: np.ndarray, labels: np.ndarray, folds: np.ndarray, pair_of: list[tuple[int, int]],
             seed: int = 0, jobs: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Rank of the true label for every prompt at every location: [648, 192], plus chosen C."""
    n_loc = H.shape[1]
    ranks = np.full((len(H), n_loc), -1, np.int32)
    chosen = np.zeros((N_FOLDS, n_loc))
    val_pairs_by_fold = validation_pairs(folds, pair_of, seed)
    for f in range(N_FOLDS):
        train_pairs = sorted({pair_of[i] for i in np.where(folds != f)[0]})
        val_pairs = set(val_pairs_by_fold[f])
        te = folds == f
        va = np.array([p in val_pairs for p in pair_of]) & ~te
        tr = ~te & ~va
        # leak checks: unordered pair (hence its mirror) never spans train and test
        assert not ({pair_of[i] for i in np.where(tr | va)[0]} & {pair_of[i] for i in np.where(te)[0]})
        assert te.sum() and va.sum() and tr.sum()
        fitted = Parallel(n_jobs=jobs, prefer="processes", max_nbytes="10M")(
            delayed(_fit_location)(H, v, labels, tr, va, te) for v in range(n_loc)
        )
        for v, (location_ranks, best) in enumerate(fitted):
            ranks[te, v] = location_ranks
            chosen[f, v] = best
        print(f"fold {f}: test {int(te.sum())} val {int(va.sum())} train {int(tr.sum())} "
              f"prompts; loop-1 best top1 {(ranks[te, :N_LAYER] == 0).mean(0).max():.3f}", flush=True)
    assert (ranks >= 0).all(), "every prompt must be scored exactly once"
    return ranks, chosen


def score_lenses_all(cache_path: Path, lens_path: str, out: Path) -> dict:
    """GPU: candidate/one-form/vocab ranks for both lenses on ALL 648 prompts."""
    import torch

    import jlens
    from ouro_jlens.evaldata import single_token_ids, surface_forms
    from ouro_jlens.evaluate import stacked_jacobians
    from ouro_jlens.probe import lens_candidate_ranks
    from ouro_jlens.recurrent import load_ouro

    H = torch.from_numpy(np.load(cache_path)["H"])
    prompts = make_prompts()
    labels = np.array([q["label"] for q in prompts])
    m = load_ouro()
    label_tokens = [single_token_ids(m.tokenizer, surface_forms(str(s))) for s in LABELS]
    assert all(label_tokens)
    t0 = time.perf_counter()
    lens = jlens.JacobianLens.load(lens_path)
    J = stacked_jacobians(m, lens, m.exit_index(m.n_ut - 1))
    jl = lens_candidate_ranks(m, H, labels, J, label_tokens)
    del J, lens
    torch.cuda.empty_cache()
    ll = lens_candidate_ranks(m, H, labels, None, label_tokens)
    print(f"lens scoring on all {len(H)} prompts in {time.perf_counter()-t0:.0f}s", flush=True)
    arrays = dict(zip(("jl_cand", "jl_one", "jl_vocab"), jl)) | dict(zip(("ll_cand", "ll_one", "ll_vocab"), ll))
    atomic_savez(out, lens=np.asarray(lens_path), **arrays)
    source = source_manifest([
        Path(__file__).resolve(),
        Path(__file__).with_name("probe.py").resolve(),
        Path(__file__).with_name("evaluate.py").resolve(),
        Path(__file__).with_name("recurrent.py").resolve(),
    ])
    atomic_write_json(out.with_suffix(".provenance.json"), {
        "schema_version": 1,
        "status": "FRESH_CURRENT_SOURCE",
        "inputs": {"gpu_cache": file_record(cache_path), "lens": file_record(Path(lens_path))},
        "source_files": source["files"],
        "source_sha256": source["sha256"],
        "output": file_record(out),
    })
    return arrays


def rebuild_gpu_cache(cache_path: Path, lens_path: str) -> None:
    """Regenerate all 648 hidden states under current source and bind the model bytes."""
    import gc
    import torch

    from ouro_jlens.probe import cache as cache_hidden
    from ouro_jlens.recurrent import OURO_SNAPSHOT, load_ouro

    prompts = make_prompts()
    model = load_ouro()
    hidden, correct = cache_hidden(model, prompts)
    atomic_savez(cache_path, H=hidden.numpy(), correct=np.asarray(correct), lens=np.asarray(lens_path))
    model_files = [
        OURO_SNAPSHOT / "model.safetensors",
        OURO_SNAPSHOT / "config.json",
        OURO_SNAPSHOT / "modeling_ouro.py",
        OURO_SNAPSHOT / "tokenizer.json",
    ]
    source = source_manifest([
        Path(__file__).resolve(),
        Path(__file__).with_name("probe.py").resolve(),
        Path(__file__).with_name("recurrent.py").resolve(),
    ])
    atomic_write_json(cache_path.with_suffix(".provenance.json"), {
        "schema_version": 1,
        "status": "FRESH_CURRENT_SOURCE_AND_MODEL",
        "design": {"prompts": len(prompts), "virtual_locations": int(hidden.shape[1]),
                   "hidden_width": int(hidden.shape[2])},
        "model_files": [file_record(path) for path in model_files],
        "source_files": source["files"],
        "source_sha256": source["sha256"],
        "output": file_record(cache_path),
    })
    del model, hidden
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="artifacts/jlens/probe/n80_v2/gpu_cache.npz")
    p.add_argument("--lens", default="artifacts/jlens/lens/exit3/exit3_n80.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--lens-only", action="store_true", help="run only the GPU step and exit")
    p.add_argument("--rebuild-gpu-cache", action="store_true",
                   help="replace the hidden-state cache under current source/model before scoring")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache, lens_all = Path(args.cache), out / "lens_all648.npz"

    if args.rebuild_gpu_cache:
        rebuild_gpu_cache(cache, args.lens)
        lens_all.unlink(missing_ok=True)
        lens_all.with_suffix(".provenance.json").unlink(missing_ok=True)
    if not lens_all.exists():
        score_lenses_all(cache, args.lens, lens_all)
    if args.lens_only:
        return
    L = dict(np.load(lens_all))
    lens_provenance = lens_all.with_suffix(".provenance.json")
    lens_score_status = "FRESH_CURRENT_SOURCE" if lens_provenance.is_file() else (
        "RECOVERED_VOLATILE_ARTIFACT_UNVERIFIED_MODEL_BINDING"
    )

    prompts = make_prompts()
    labels = np.array([q["label"] for q in prompts])
    pair_of = [(min(q["a"], q["b"]), max(q["a"], q["b"])) for q in prompts]
    folds = prompt_folds(prompts, args.seed)
    c = np.load(cache)
    H, correct = c["H"], c["correct"]
    assert H.shape[0] == len(prompts) == 648

    t0 = time.perf_counter()
    ranks, chosen = cv_probe(H, labels, folds, pair_of, args.seed, jobs=args.jobs)
    print(f"cross-validated probes fitted in {time.perf_counter()-t0:.0f}s", flush=True)

    arrays_path = out / "arrays.npz"
    atomic_savez(arrays_path, probe_rank=ranks, chosen_C=chosen, folds=folds,
                 labels=labels, correct=correct, **L)
    vals = validation_pairs(folds, pair_of, args.seed)
    atomic_write_json(out / "design.json", {
        "schema_version": 1,
        "seed": args.seed,
        "fold_assignment_unit": "unordered_operand_pair",
        "fold_sizes": [int((folds == f).sum()) for f in range(N_FOLDS)],
        "validation_pairs_by_fold": [[[int(a), int(b)] for a, b in pairs] for pairs in vals],
        "inputs": {"gpu_cache": file_record(cache), "lens_scores": file_record(lens_all)},
        "lens_score_status": lens_score_status,
        "lens_score_provenance": file_record(lens_provenance) if lens_provenance.is_file() else None,
    })
    from ouro_jlens.probe_report import build_report

    report = build_report(arrays_path, seed=args.seed)
    report["lens_score_status"] = lens_score_status
    if lens_score_status != "FRESH_CURRENT_SOURCE":
        report["status"] = "POINT_ESTIMATES_REGENERATED_FROM_RECOVERED_GPU_SCORES_UNFROZEN"
    atomic_write_json(out / "summary.json", report)
    for name, values in report["readouts"].items():
        print(f"{name}: cross-fitted eligible-population per-loop {values['per_loop']}")


if __name__ == "__main__":
    main()
