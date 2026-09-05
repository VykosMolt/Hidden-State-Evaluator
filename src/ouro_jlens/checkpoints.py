"""Lens-free exit divergence across Ouro checkpoints (base / Thinking / RLTT).

Answers, without fitting anything: does post-training widen the gap between what an
intermediate recurrent exit says and what the finished model says? That is section 3 of
RESULTS.md measured on each checkpoint. All three share Ouro's architecture, so the base
tokenizer is used throughout to guarantee identical input ids.

  python src/ouro_jlens/checkpoints.py --out artifacts/jlens/checkpoints
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import jlens

from jlens.hooks import ActivationRecorder
from jlens.vis import _ranks_of

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouro_jlens.evaluate import GREEDY_STEPS, greedy, is_correct  # noqa: E402
from ouro_jlens.evidence import (  # noqa: E402
    aggregate_sha256,
    atomic_write_json,
    file_record,
)
from ouro_jlens.evaldata import load_items  # noqa: E402
from ouro_jlens.recurrent import OURO_REVISION, OURO_SNAPSHOT, PROJECT_ROOT, load_ouro  # noqa: E402

HUB = PROJECT_ROOT / "artifacts" / "hf_cache" / "hub"
_THINKING_SNAPSHOTS = sorted((HUB / "models--ByteDance--Ouro-2.6B-Thinking" / "snapshots").glob("*"))
CHECKPOINTS = {
    "base": OURO_SNAPSHOT,
    "thinking": _THINKING_SNAPSHOTS[0] if len(_THINKING_SNAPSHOTS) == 1 else None,
    "rltt": PROJECT_ROOT / "models" / "ouro_rltt_local",
}
OPERATIONS = {"addition", "subtraction", "multiplication", "division", "mod", "squared"}
JLENS_LOGICAL_ROOT = "dependency/jlens"


def _reject_output_path(path: Path) -> None:
    """Reject a destination or existing ancestor that is a symbolic link."""

    lexical = Path(os.path.abspath(path))
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink():
            raise ValueError(f"checkpoint output path traverses a link: {candidate}")


def _jlens_package_root() -> Path:
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


def _jlens_source_records() -> list[dict]:
    root = _jlens_package_root()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"installed jlens package contains a linked path: {candidate}")
    paths = sorted(root.rglob("*.py"), key=lambda value: value.relative_to(root).as_posix())
    if not paths:
        raise ValueError("installed jlens package contains no Python sources")
    records = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"installed jlens source is missing or linked: {path}")
        current = path.parent
        while current != root:
            if current.is_symlink():
                raise ValueError(f"installed jlens source traverses a link: {path}")
            current = current.parent
        record = file_record(path)
        record["path"] = f"{JLENS_LOGICAL_ROOT}/{path.relative_to(root).as_posix()}"
        records.append(record)
    return records


def _source_manifest() -> dict:
    """Hash project code and every installed jlens Python source byte."""

    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("evaluate.py").resolve(),
        Path(__file__).with_name("evaldata.py").resolve(),
        Path(__file__).with_name("recurrent.py").resolve(),
        Path(__file__).with_name("evidence.py").resolve(),
    ]
    records = []
    for path in paths:
        try:
            logical = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise ValueError(f"checkpoint source is outside project: {path}") from exc
        record = file_record(path)
        record["path"] = logical
        records.append(record)
    records.extend(_jlens_source_records())
    return {
        "files": records,
        "sha256": aggregate_sha256({record["path"]: record["sha256"] for record in records}),
    }


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in (
        "torch", "transformers", "jlens", "numpy", "safetensors",
        "accelerate", "huggingface-hub",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    return versions


def _checkpoint_files(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"checkpoint root is missing or linked: {path}")
    required = [path / "config.json", path / "modeling_ouro.py", path / "tokenizer.json"]
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
            names = sorted(set(weight_map.values()))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"checkpoint weight index is malformed: {index_path}") from exc
        if not names or any(
            not isinstance(name, str) or Path(name).name != name for name in names
        ):
            raise ValueError(f"checkpoint weight index has unsafe or empty shard names: {index_path}")
        weights = [path / name for name in names]
        required.append(index_path)
    else:
        weights = [path / "model.safetensors"]
    if any(not candidate.is_file() for candidate in [*required, *weights]):
        raise ValueError(f"checkpoint is missing required model/tokenizer bytes: {path}")
    candidates: list[Path] = []
    for candidate in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        relative = candidate.relative_to(path)
        current = candidate.parent
        while current != path:
            if current.is_symlink():
                raise ValueError(f"checkpoint traverses a linked directory: {relative}")
            current = current.parent
        if candidate.is_file():
            candidates.append(candidate)
        elif candidate.is_symlink():
            raise ValueError(f"checkpoint contains a dangling or directory link: {relative}")
    records = []
    for candidate in candidates:
        record = file_record(candidate)
        record["path"] = f"checkpoint/{candidate.relative_to(path).as_posix()}"
        records.append(record)
    return records


@torch.no_grad()
def measure(m, items, tokenizer) -> dict:
    """Per-checkpoint exit statistics at the readout position."""
    n_ut = m.n_ut
    kl = np.zeros((len(items), n_ut), np.float32)
    js = np.zeros((len(items), n_ut), np.float32)
    ent = np.zeros((len(items), n_ut), np.float32)
    rank_final = np.zeros((len(items), n_ut), np.int32)
    same = np.zeros((len(items), n_ut), bool)
    inter_top1 = np.zeros((len(items), n_ut), bool)
    correct = np.zeros(len(items), bool)
    for i, item in enumerate(items):
        ids = torch.tensor([item.token_ids], device=m.input_device)
        with ActivationRecorder(m.layers, at=[m.exit_index(u) for u in range(n_ut)]) as rec:
            m.forward(ids)
        h = torch.stack([rec.activations[m.exit_index(u)][0, -1] for u in range(n_ut)])
        logits = m.unembed(h).float()                              # [n_ut, vocab]
        lp = F.log_softmax(logits, -1)
        p_ = lp.exp()
        kl[i] = (p_ * (lp - lp[-1])).sum(-1).cpu().numpy()
        ent[i] = (-(p_ * lp).sum(-1)).cpu().numpy()
        # KL confounds "different content" with "different sharpness"; JS is bounded and
        # symmetric, and the rank of the final answer under exit k is scale-free entirely.
        mlog = ((p_ + p_[-1]) / 2).clamp_min(1e-12).log()
        js[i] = (0.5 * (p_ * (lp - mlog)).sum(-1) + 0.5 * (p_[-1] * (lp[-1] - mlog)).sum(-1)).cpu().numpy()
        top1 = logits.argmax(-1)
        rank_final[i] = _ranks_of(logits, top1[-1].view(1))[:, 0].cpu().numpy()
        same[i] = (top1 == top1[-1]).cpu().numpy()
        toks = [t for k in item.scorable for t in item.intermediate_tokens[k]
                if k not in OPERATIONS and not item.leaked[k]]
        if toks:
            r = _ranks_of(logits, torch.tensor(toks, device=logits.device)).min(1).values
            inter_top1[i] = (r == 0).cpu().numpy()
        correct[i] = is_correct(greedy(m, ids, GREEDY_STEPS), item.target)
    out = {"n_items": len(items), "model_correct": float(correct.mean())}
    for task in ("multihop", "order-ops"):
        sel = np.array([it.task == task for it in items])
        scored = sel & np.array([any(k not in OPERATIONS and not it.leaked[k] for k in it.scorable) for it in items])
        out[task] = {
            "n": int(sel.sum()),
            "kl_exit_k_to_exit_last_mean": kl[sel].mean(0).round(3).tolist(),
            "kl_median": np.median(kl[sel], 0).round(3).tolist(),
            "exit_top1_equals_final": same[sel].mean(0).round(3).tolist(),
            "js_to_final_mean": js[sel].mean(0).round(4).tolist(),
            "entropy_mean": ent[sel].mean(0).round(3).tolist(),
            "median_rank_of_final_top1": np.median(rank_final[sel], 0).round(1).tolist(),
            "intermediate_is_exit_top1": inter_top1[scored].mean(0).round(3).tolist(),
            "n_scored": int(scored.sum()),
            "model_correct": float(correct[sel].mean()),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/jlens/checkpoints")
    p.add_argument("--only", nargs="*", choices=tuple(CHECKPOINTS), default=list(CHECKPOINTS))
    args = p.parse_args()
    if not args.only:
        p.error("--only must name at least one checkpoint")
    out = Path(args.out)
    _reject_output_path(out)
    out.mkdir(parents=True, exist_ok=True)
    _reject_output_path(out)
    output_path = out / "exit_divergence.json"
    _reject_output_path(output_path)

    # Replace any prior completed artifact with a durable non-accepting marker
    # before loading a tokenizer or beginning a measurement.  SIGKILL can leave
    # this marker (or the later RUNNING marker), and neither status is usable
    # as completed evidence.
    atomic_write_json(output_path, {
        "schema_version": 2,
        "status": "INVALIDATED_BEFORE_ATTEMPT",
        "requested_checkpoints": list(args.only),
        "interpretation_limit": (
            "heterogeneous named checkpoints are a cross-check, not an ordered or monotonic training trajectory"
        ),
    })

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(OURO_SNAPSHOT))
    source = _source_manifest()
    tokenizer_records = _checkpoint_files(OURO_SNAPSHOT)
    for record in tokenizer_records:
        record["path"] = record["path"].replace("checkpoint/", "input_tokenizer/", 1)
    runtime_versions = _runtime_versions()
    jlens_source_records = [
        record for record in source["files"]
        if str(record.get("path", "")).startswith(f"{JLENS_LOGICAL_ROOT}/")
    ]
    results = {
        "schema_version": 2,
        "status": "RUNNING",
        "interpretation_limit": (
            "heterogeneous named checkpoints are a cross-check, not an ordered or monotonic training trajectory"
        ),
        "requested_checkpoints": list(args.only),
        "checkpoints": {},
        "missing": [],
        "source_files": source["files"],
        "source_sha256": source["sha256"],
        "jlens_source": {
            "files": jlens_source_records,
            "aggregate_sha256": aggregate_sha256(
                {record["path"]: record["sha256"] for record in jlens_source_records}
            ),
        },
        "input_tokenizer": {
            "checkpoint_revision": OURO_REVISION,
            "files": tokenizer_records,
            "aggregate_sha256": aggregate_sha256(
                {record["path"]: record["sha256"] for record in tokenizer_records}
            ),
        },
        "runtime_versions": runtime_versions,
    }
    atomic_write_json(output_path, results)
    items = None
    try:
        for name in args.only:
            path = CHECKPOINTS[name]
            if path is None or not Path(path).is_dir():
                print(f"{name}: MISSING at {path}", flush=True)
                results["missing"].append(name)
                continue
            t0 = time.perf_counter()
            checkpoint_files = _checkpoint_files(Path(path))
            m = load_ouro(path)
            m.tokenizer = tokenizer  # identical input ids across checkpoints
            if items is None:
                items = load_items(tokenizer, encode=lambda s: m.encode(s)[0].tolist())
            measured = measure(m, items, tokenizer)
            measured["checkpoint_id"] = name
            measured["model_revision"] = m.model_revision
            measured["checkpoint_files"] = checkpoint_files
            measured["model_files_sha256"] = aggregate_sha256(
                {record["path"]: record["sha256"] for record in checkpoint_files}
            )
            measured["seconds"] = round(time.perf_counter() - t0, 1)
            results["checkpoints"][name] = measured
            print(f"{name}: {json.dumps(measured['order-ops'])}", flush=True)
            del m
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException as exc:
        # Preserve a machine-readable terminal failure when Python gets a
        # chance to handle the interruption.  A hard kill leaves RUNNING.
        results["status"] = "INTERRUPTED_OR_FAILED"
        results["error"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_write_json(output_path, results)
        raise
    results["status"] = (
        "REQUESTED_CHECKPOINT_MEASUREMENTS_COMPLETE_UNFROZEN"
        if not results["missing"]
        else "INCOMPLETE_MISSING_REQUESTED_CHECKPOINTS"
    )
    atomic_write_json(output_path, results)

    print("\n== KL(exit k || final), mean over items")
    for task in ("multihop", "order-ops"):
        print(f"  {task}")
        for name, r in results["checkpoints"].items():
            print(f"    {name:9s} KL {r[task]['kl_exit_k_to_exit_last_mean']}  JS {r[task]['js_to_final_mean']}")
            print(f"    {'':9s} entropy {r[task]['entropy_mean']}  rank(final top1) {r[task]['median_rank_of_final_top1']}")
            print(f"    {'':9s} exit==final {r[task]['exit_top1_equals_final']}  intermediate-is-top1 "
                  f"{r[task]['intermediate_is_exit_top1']}  acc {r[task]['model_correct']:.2f}")
    if results["missing"]:
        raise SystemExit(f"INCOMPLETE: missing required checkpoints {results['missing']}")


if __name__ == "__main__":
    main()
