"""Task-specific supervised linear reference on a templated arithmetic family.

Prompts "(a + b) * c = " with a, b in 1..9 and c in 2..9; the latent
intermediate is a+b (17 classes). Split by held-out (a, b) pairs so a probe
cannot memorise operand pairs. Per location: multinomial logistic regression on
standardised residuals, C chosen on validation pairs, refit on train+val,
scored on test pairs. The Jacobian lens and logit lens are scored on the same
test prompts with the same 17-way candidate set (label score = max lens logit
over the label's single-token forms) so the comparison is like-for-like; the
full-vocab lens rank is stored too.

  python src/ouro_jlens/probe.py --lens artifacts/jlens/lens/exit3.pt --out artifacts/jlens/probe/run1
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import jlens
from jlens.hooks import ActivationRecorder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouro_jlens.evidence import (  # noqa: E402
    aggregate_sha256,
    atomic_savez,
    atomic_write_json,
    file_record,
    sha256_file,
)
from ouro_jlens.evaldata import single_token_ids, surface_forms  # noqa: E402
from ouro_jlens.evaluate import greedy, is_correct, stacked_jacobians  # noqa: E402
from ouro_jlens.recurrent import OURO_SNAPSHOT, load_ouro, model_snapshot_files  # noqa: E402

LABELS = list(range(2, 19))
C_GRID = (0.01, 0.1, 1.0)
N_UT, N_LAYER = 4, 48
REPO_ROOT = Path(__file__).resolve().parents[2]
JLENS_LOGICAL_ROOT = "dependency/jlens"


def _logical_record(path: Path, logical_path: str) -> dict:
    record = file_record(path)
    record["path"] = logical_path
    return record


def _source_paths() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        here,
        here.with_name("evaluate.py"),
        here.with_name("evaldata.py"),
        here.with_name("recurrent.py"),
        here.with_name("fit_lens.py"),
        here.with_name("evidence.py"),
    ]


def _jlens_package_root() -> Path:
    """Return the installed package root without accepting linked source."""

    package_file = getattr(jlens, "__file__", None)
    if not package_file:
        raise ValueError("installed jlens package has no source path")
    package_path = Path(package_file).absolute()
    if not package_path.is_file() or package_path.is_symlink():
        raise ValueError("installed jlens package source is missing or linked")
    root = package_path.parent
    for candidate in (root, *root.parents):
        if candidate.is_symlink():
            raise ValueError(f"installed jlens package traverses a link: {candidate}")
    return root


def _jlens_source_paths() -> list[Path]:
    """Enumerate every regular Python source file in installed ``jlens``."""

    root = _jlens_package_root()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"installed jlens package contains a linked path: {candidate}")
    paths = []
    for candidate in sorted(root.rglob("*.py"), key=lambda value: value.relative_to(root).as_posix()):
        if candidate.is_symlink():
            raise ValueError(f"installed jlens source is linked: {candidate}")
        current = candidate.parent
        while current != root:
            if current.is_symlink():
                raise ValueError(f"installed jlens source traverses a link: {candidate}")
            current = current.parent
        if candidate.is_file():
            paths.append(candidate)
    if not paths:
        raise ValueError("installed jlens package contains no Python sources")
    return paths


def _jlens_source_records() -> list[dict]:
    root = _jlens_package_root()
    records = []
    for path in _jlens_source_paths():
        record = _logical_record(path, f"{JLENS_LOGICAL_ROOT}/{path.relative_to(root).as_posix()}")
        records.append(record)
    return records


def _source_manifest(paths: list[Path] | None = None, *, include_jlens: bool = True) -> dict:
    paths = _source_paths() if paths is None else paths
    records = []
    for path in paths:
        candidate = Path(path)
        try:
            logical = candidate.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError as exc:
            raise ValueError(f"project source is outside repository: {candidate}") from exc
        records.append(_logical_record(candidate, logical))
    if include_jlens:
        records.extend(_jlens_source_records())
    return {
        "files": records,
        "sha256": aggregate_sha256({r["path"]: r["sha256"] for r in records}),
    }


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in (
        "numpy", "torch", "scikit-learn", "threadpoolctl", "jlens", "transformers",
        "safetensors", "accelerate", "huggingface-hub",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    return versions


def _check_record(
    record: object,
    path: Path,
    logical_path: str,
    label: str,
    *,
    allow_symlink: bool = False,
) -> None:
    if not isinstance(record, dict) or record.get("path") != logical_path:
        raise ValueError(f"{label} record is missing or has the wrong logical identity")
    if not path.is_file() or (path.is_symlink() and not allow_symlink):
        raise ValueError(f"{label} is missing or linked")
    if record.get("size") != path.stat().st_size or record.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} byte identity mismatch")


def _reject_writable_symlinks(path: Path) -> None:
    lexical = path.absolute()
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink():
            raise ValueError(f"writable output path traverses a symlink: {candidate}")


def _invalidate_outputs(root: Path, names: tuple[str, ...]) -> None:
    """Remove explicitly named descendants and durably publish the deletion.

    Rebuilds are intentionally fail-closed: every derivative is absent before
    the first expensive operation starts.  The paths are fixed by the command
    contract rather than a caller-controlled glob.
    """

    for name in names:
        path = root / name
        if path.is_dir() and not path.is_symlink():
            raise ValueError(f"cannot invalidate directory artifact: {path}")
        path.unlink(missing_ok=True)
    try:
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Atomic artifact writes remain safe on filesystems that do not expose
        # fsync on directories; no stale file is kept in that case.
        pass


def _lens_lineage(lens_path: Path) -> str:
    sidecar = lens_path.with_suffix(".json")
    if lens_path.is_symlink() or not lens_path.is_file() or sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError("direct probe requires a regular lens and retained sidecar")
    try:
        preview = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("lens sidecar is unreadable") from exc
    if isinstance(preview, dict) and (
        preview.get("schema_version") is not None or preview.get("kind") is not None
    ):
        if preview.get("kind") not in {"fit", "merged"}:
            raise ValueError("current-schema lens sidecar has an invalid kind")
        from ouro_jlens.fit_lens import validate_sidecar

        validate_sidecar(lens_path, kind=str(preview["kind"]))
        return "HASH_BOUND"
    # Historical pre-custody lenses have unversioned sidecars.  Their exact
    # bytes remain usable for a clearly degraded local diagnostic, never as
    # hash-bound fitted evidence.
    return "RETAINED_PRE_CUSTODY_EXACT_BYTES_ONLY"


def _validate_direct_cache(cache_path: Path, lens_path: Path, test_count: int) -> dict:
    provenance_path = cache_path.with_suffix(".provenance.json")
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise ValueError("direct-probe cache provenance is missing or linked")
    try:
        document = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("direct-probe cache provenance is unreadable") from exc
    lineage = _lens_lineage(lens_path)
    expected_status = (
        "FRESH_CURRENT_SOURCE_AND_MODEL_HASH_BOUND_LENS"
        if lineage == "HASH_BOUND"
        else "FRESH_CURRENT_SOURCE_AND_MODEL_RETAINED_PRE_CUSTODY_LENS"
    )
    if document.get("schema_version") != 2 or document.get("status") != expected_status \
            or document.get("lens_lineage_status") != lineage:
        raise ValueError("direct-probe cache is not current provenance")
    expected_design = {
        "prompts": 648,
        "test_prompts": test_count,
        "virtual_locations": N_UT * N_LAYER,
        "hidden_width": 2048,
    }
    if document.get("design") != expected_design:
        raise ValueError("direct-probe design manifest mismatch")
    _check_record(document.get("output"), cache_path, "gpu_cache", "direct-probe cache")
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("direct-probe cache input manifest is missing")
    _check_record(inputs.get("lens"), lens_path, "lens", "direct-probe lens")
    _check_record(
        inputs.get("lens_sidecar"), lens_path.with_suffix(".json"), "lens_sidecar", "lens sidecar"
    )
    source = _source_manifest()
    if document.get("source_files") != source["files"] or document.get("source_sha256") != source["sha256"]:
        raise ValueError("direct-probe source manifest mismatch")
    if document.get("runtime_versions") != _runtime_versions():
        raise ValueError("direct-probe runtime dependency versions changed")
    model_paths = model_snapshot_files(OURO_SNAPSHOT)
    model_records = document.get("model_files")
    if not isinstance(model_records, list) or len(model_records) != len(model_paths):
        raise ValueError("direct-probe model manifest is incomplete")
    for record, path in zip(model_records, model_paths, strict=True):
        _check_record(
            record,
            path,
            f"model_snapshot/{path.relative_to(OURO_SNAPSHOT).as_posix()}",
            f"model file {path.relative_to(OURO_SNAPSHOT).as_posix()}",
            allow_symlink=True,
        )
    expected_keys = {
        "H", "correct", "lens_sha256", "jl_cand", "jl_one", "jl_vocab",
        "ll_cand", "ll_one", "ll_vocab",
    }
    with np.load(cache_path, allow_pickle=False) as values:
        if set(values.files) != expected_keys:
            raise ValueError("direct-probe cache fields are incomplete or unexpected")
        if values["H"].shape != (648, N_UT * N_LAYER, 2048) \
                or values["H"].dtype != np.float16 \
                or values["correct"].shape != (648,) \
                or values["correct"].dtype != np.bool_:
            raise ValueError("direct-probe hidden-state cache shape mismatch")
        if not np.isfinite(values["H"]).all():
            raise ValueError("direct-probe hidden-state cache contains non-finite values")
        if str(values["lens_sha256"].item()) != sha256_file(lens_path):
            raise ValueError("direct-probe cache is bound to another lens")
        for key in ("jl_cand", "jl_one", "jl_vocab", "ll_cand", "ll_one", "ll_vocab"):
            value = values[key]
            if value.shape != (test_count, N_UT * N_LAYER) \
                    or not np.issubdtype(value.dtype, np.integer) or value.min() < 0:
                raise ValueError(f"direct-probe rank array {key} is invalid")
            if key.endswith(("cand", "one")) and value.max() >= len(LABELS):
                raise ValueError(f"direct-probe candidate rank array {key} exceeds its support")
    return document


def make_prompts() -> list[dict]:
    return [
        {"a": a, "b": b, "c": c, "prompt": f"({a} + {b}) * {c} = ", "label": a + b, "target": str((a + b) * c)}
        for a in range(1, 10) for b in range(1, 10) for c in range(2, 10)
    ]


def split_pairs(seed: int = 0) -> dict[str, set[tuple[int, int]]]:
    """Split by *unordered* operand pair. a+b is symmetric, so putting (3,5) in train and
    (5,3) in test would hand the probe the answer; both always land in the same split."""
    unordered = [(a, b) for a in range(1, 10) for b in range(a, 10)]
    rng = np.random.default_rng(seed)
    rng.shuffle(unordered)
    parts = {"test": unordered[:11], "val": unordered[11:17], "train": unordered[17:]}
    return {k: {(a, b) for x, y in v for a, b in ((x, y), (y, x))} for k, v in parts.items()}


@torch.no_grad()
def cache(m, prompts: list[dict]) -> tuple[torch.Tensor, list[bool]]:
    H = torch.empty(len(prompts), m.n_layers, m.d_model, dtype=torch.float16)
    correct = []
    for i, p in enumerate(prompts):
        ids = m.encode(p["prompt"])
        with ActivationRecorder(m.layers, at=range(m.n_layers)) as rec:
            m.forward(ids)
        H[i] = torch.stack([rec.activations[v][0, -1] for v in range(m.n_layers)]).half().cpu()
        correct.append(is_correct(greedy(m, ids, 3), p["target"]))
    if not torch.isfinite(H).all():
        raise ValueError("hidden-state cache contains non-finite values")
    return H, correct


@torch.no_grad()
def lens_candidate_ranks(m, H: torch.Tensor, labels: np.ndarray, J, label_tokens: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank of the true label among the 17 candidates scored by (a) the best of all its
    single-token forms and (b) one fixed form each, plus its full-vocab rank: each [n, 192].
    Labels differ in how many single-token forms they have (1 to 5), which favours the
    many-form labels under (a); (b) removes that advantage."""
    if not torch.isfinite(H).all():
        raise ValueError("hidden-state input contains non-finite values")
    from jlens.vis import _ranks_of

    cand_rank = np.zeros((len(H), m.n_layers), np.int32)
    one_rank = np.zeros((len(H), m.n_layers), np.int32)
    vocab_rank = np.zeros((len(H), m.n_layers), np.int32)
    first = [toks[:1] for toks in label_tokens]  # one form per label: no form-count advantage
    for i in range(len(H)):
        h = H[i].to(m.input_device).float()
        logits = m.unembed(h if J is None else torch.einsum("vde,ve->vd", J, h)).float()
        true = LABELS.index(int(labels[i]))
        for out, toks in ((cand_rank, label_tokens), (one_rank, first)):
            scores = torch.stack([logits[:, t].max(-1).values for t in toks], dim=1)  # [192, 17]
            out[i] = (scores > scores[:, true : true + 1]).sum(1).cpu().numpy()
        vocab_rank[i] = _ranks_of(logits, torch.tensor(label_tokens[true], device=logits.device)).min(1).values.cpu().numpy()
    return cand_rank, one_rank, vocab_rank


def fit_probes(H: np.ndarray, labels: np.ndarray, split: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Per-location probe rank of the true label on test prompts: [n_test, 192]."""
    if not np.isfinite(H).all():
        raise ValueError("hidden-state input contains non-finite values")
    tr, va, te = (split == s for s in ("train", "val", "test"))
    ranks = np.zeros((te.sum(), H.shape[1]), np.int32)
    chosen = []
    for v in range(H.shape[1]):
        X = H[:, v].astype(np.float32)
        scaler = StandardScaler().fit(X[tr])
        Xs = scaler.transform(X)
        best = max(C_GRID, key=lambda C: LogisticRegression(C=C, max_iter=2000).fit(Xs[tr], labels[tr]).score(Xs[va], labels[va]))
        scaler = StandardScaler().fit(X[tr | va])
        Xs = scaler.transform(X)
        clf = LogisticRegression(C=best, max_iter=2000).fit(Xs[tr | va], labels[tr | va])
        proba = np.zeros((te.sum(), len(LABELS)))  # labels unseen in training keep probability 0
        proba[:, [LABELS.index(c) for c in clf.classes_]] = clf.predict_proba(Xs[te])
        true_p = proba[np.arange(len(proba)), [LABELS.index(l) for l in labels[te]]]
        ranks[:, v] = (proba > true_p[:, None]).sum(1)
        chosen.append(best)
        if v % 24 == 0:
            print(f"  probe location {v}/{H.shape[1]}: C={best} test top1={np.mean(ranks[:, v] == 0):.3f}", flush=True)
    return ranks, chosen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lens", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--rebuild-gpu-cache",
        action="store_true",
        help="invalidate and regenerate the direct-probe cache under current source/model bytes",
    )
    args = p.parse_args()
    out = Path(args.out)
    _reject_writable_symlinks(out)
    out.mkdir(parents=True, exist_ok=True)
    if not out.is_dir():
        raise ValueError(f"probe output is not a directory: {out}")

    prompts = make_prompts()
    parts = split_pairs()
    split = np.array([next(k for k, s in parts.items() if (q["a"], q["b"]) in s) for q in prompts])
    labels = np.array([q["label"] for q in prompts])

    te = split == "test"
    gpu_cache = out / "gpu_cache.npz"
    gpu_provenance = gpu_cache.with_suffix(".provenance.json")
    lens_path = Path(args.lens)
    if args.rebuild_gpu_cache:
        _invalidate_outputs(
            out,
            (
                "gpu_cache.npz",
                "gpu_cache.provenance.json",
                "arrays.npz",
                "design.json",
                "meta.json",
                "summary.json",
                "lens_all648.npz",
                "lens_all648.provenance.json",
            ),
        )
    if gpu_cache.exists() or gpu_provenance.exists():
        _validate_direct_cache(gpu_cache, lens_path, int(te.sum()))
        with np.load(gpu_cache, allow_pickle=False) as c:
            H, correct = torch.from_numpy(c["H"]), c["correct"].copy()
            jl_cand, jl_one, jl_vocab = (c[key].copy() for key in ("jl_cand", "jl_one", "jl_vocab"))
            ll_cand, ll_one, ll_vocab = (c[key].copy() for key in ("ll_cand", "ll_one", "ll_vocab"))
    else:
        lens_lineage = _lens_lineage(lens_path)
        m = load_ouro()
        t0 = time.perf_counter()
        H, correct = cache(m, prompts)
        print(f"cached {H.shape} in {time.perf_counter()-t0:.0f}s; model correct on (a+b)*c: {np.mean(correct):.3f}", flush=True)
        label_tokens = [single_token_ids(m.tokenizer, surface_forms(str(s))) for s in LABELS]
        if not all(label_tokens):
            raise ValueError("one or more arithmetic labels have no single-token surface form")
        H_test = H[torch.from_numpy(te)]
        lens = jlens.JacobianLens.load(str(lens_path))
        J = stacked_jacobians(m, lens, m.exit_index(m.n_ut - 1))
        jl_cand, jl_one, jl_vocab = lens_candidate_ranks(m, H_test, labels[te], J, label_tokens)
        del J
        ll_cand, ll_one, ll_vocab = lens_candidate_ranks(m, H_test, labels[te], None, label_tokens)
        del m
        torch.cuda.empty_cache()
        if not torch.isfinite(H).all():
            raise ValueError("direct-probe hidden-state cache contains non-finite values")
        atomic_savez(
            gpu_cache,
            H=H.numpy(),
            correct=np.asarray(correct),
            lens_sha256=np.asarray(sha256_file(lens_path)),
            jl_cand=jl_cand,
            jl_one=jl_one,
            jl_vocab=jl_vocab,
            ll_cand=ll_cand,
            ll_one=ll_one,
            ll_vocab=ll_vocab,
        )
        source = _source_manifest()
        model_paths = model_snapshot_files(OURO_SNAPSHOT)
        atomic_write_json(gpu_provenance, {
            "schema_version": 2,
            "status": (
                "FRESH_CURRENT_SOURCE_AND_MODEL_HASH_BOUND_LENS"
                if lens_lineage == "HASH_BOUND"
                else "FRESH_CURRENT_SOURCE_AND_MODEL_RETAINED_PRE_CUSTODY_LENS"
            ),
            "lens_lineage_status": lens_lineage,
            "design": {
                "prompts": len(prompts),
                "test_prompts": int(te.sum()),
                "virtual_locations": N_UT * N_LAYER,
                "hidden_width": int(H.shape[2]),
            },
            "inputs": {
                "lens": _logical_record(lens_path, "lens"),
                "lens_sidecar": _logical_record(lens_path.with_suffix(".json"), "lens_sidecar"),
            },
            "model_files": [
                _logical_record(
                    path, f"model_snapshot/{path.relative_to(OURO_SNAPSHOT).as_posix()}"
                ) for path in model_paths
            ],
            "runtime_versions": _runtime_versions(),
            "source_files": source["files"],
            "source_sha256": source["sha256"],
            "output": _logical_record(gpu_cache, "gpu_cache"),
        })
        _validate_direct_cache(gpu_cache, lens_path, int(te.sum()))

    # A CPU refit is a new derived generation even when its validated GPU
    # cache is reused.  Remove prior claims before fitting so interruption can
    # leave only missing/incomplete evidence, never a stale accepted summary.
    _invalidate_outputs(out, ("arrays.npz", "design.json", "meta.json", "summary.json"))
    t0 = time.perf_counter()
    probe_rank, chosen_C = fit_probes(H.numpy(), labels, split)
    print(f"probes fitted in {time.perf_counter()-t0:.0f}s", flush=True)

    arrays_path = out / "arrays.npz"
    atomic_savez(
        arrays_path, probe_rank=probe_rank, jlens_cand_rank=jl_cand, jlens_vocab_rank=jl_vocab,
        jlens_oneform_rank=jl_one, logitlens_oneform_rank=ll_one,
        logitlens_cand_rank=ll_cand, logitlens_vocab_rank=ll_vocab, test_labels=labels[te],
        test_correct=np.array(correct)[te], chosen_C=np.array(chosen_C),
    )
    meta = {
        "schema_version": 2,
        "status": "UNDERPOWERED_DIRECT_SPLIT_DESCRIPTIVE_ONLY",
        "n_prompts": len(prompts),
        "n_train": int((split == "train").sum()),
        "n_val": int((split == "val").sum()),
        "n_test": int(te.sum()),
        "model_accuracy_all": float(np.mean(correct)),
        "labels": LABELS,
        "C_grid": C_GRID,
        "inputs": {
            "gpu_cache": _logical_record(gpu_cache, "gpu_cache"),
            "gpu_cache_provenance": _logical_record(gpu_provenance, "gpu_cache_provenance"),
        },
        "output": _logical_record(arrays_path, "probe_arrays"),
    }
    atomic_write_json(out / "meta.json", meta)
    for name, r in [("probe", probe_rank), ("jlens", jl_cand), ("logitlens", ll_cand),
                    ("jlens 1-form", jl_one), ("logitlens 1-form", ll_one)]:
        acc = (r == 0).mean(0)
        best = int(acc.argmax())
        print(f"{name}: best candidate-set top1 {acc[best]:.3f} at ut{best // 48} L{best % 48}; "
              f"per-loop max {[round(float(acc[u*48:(u+1)*48].max()), 3) for u in range(4)]}")


if __name__ == "__main__":
    main()
