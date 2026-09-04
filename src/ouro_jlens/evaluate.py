"""Known-intermediate readout across (ut, layer) for Jacobian lens vs logit lens.

One forward per item caches the residual at the readout position (the token
preceding the target) for all 192 virtual locations. For every lens (one per
exit target) we then store per location: the rank of *every* intermediate name
of the item's task (min over single-token forms) — the item's own names give the
hit rate, the other names are matched controls for the position's prior — plus
top-1 token, KL to the actual exit logits, and, for the eventual-exit lens, the
cross-loop tensor from applying J fitted at (ut_i, L) to the state at (ut_j, L).
Output: one .npz + items.json under --out.

  python src/ouro_jlens/evaluate.py --lens 3=artifacts/jlens/lens/exit3.pt \
      [--lens 2=... --lens 1=... --lens 0=...] --out artifacts/jlens/eval/run1
"""

from __future__ import annotations

import argparse
import copy
import json
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
from ouro_jlens.evaldata import Item, load_items  # noqa: E402
from ouro_jlens.evidence import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_savez,
    atomic_write_json,
    file_record,
    sha256_json,
    sha256_file,
    source_manifest,
)
from ouro_jlens.fit_lens import IntegrityError, sidecar_path, validate_sidecar  # noqa: E402
from ouro_jlens.recurrent import OURO_REVISION, load_ouro  # noqa: E402

MAX_INTER = 3
MAX_NAMES = 128
GREEDY_STEPS = 4
TASK_NAMES: dict[str, TaskNames] = {}


@torch.no_grad()
def greedy(m, ids: torch.Tensor, steps: int) -> str:
    out = ids
    for _ in range(steps):
        logits = m.hf_model(out, use_cache=False, exit_at_step=m.n_ut - 1).logits[0, -1]
        out = torch.cat([out, logits.argmax().view(1, 1)], dim=1)
    return m.tokenizer.decode(out[0, ids.shape[1]:])


def is_correct(continuation: str, target: str) -> bool:
    """Prefix match that must end at a token boundary, so "11" does not count as "1"."""
    c, t = continuation.strip().strip('"').lower(), target.strip().lower()
    if not c.startswith(t):
        return False
    rest = c[len(t):]
    return rest == "" or not rest[0].isalnum()


@torch.no_grad()
def cache_states(m, items: list[Item], position: int = -1) -> tuple[torch.Tensor, list[str]]:
    """Residuals at the readout position for all virtual locations: [n_items, 192, d]."""
    H = torch.empty(len(items), m.n_layers, m.d_model, dtype=torch.float32)
    continuations = []
    for i, item in enumerate(items):
        ids = torch.tensor([item.token_ids], device=m.input_device)
        with ActivationRecorder(m.layers, at=range(m.n_layers)) as rec:
            m.forward(ids)
        H[i] = torch.stack([rec.activations[v][0, position] for v in range(m.n_layers)]).float().cpu()
        continuations.append(greedy(m, ids, GREEDY_STEPS))
    return H, continuations


def kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) along the last dim, in nats."""
    lp, lq = F.log_softmax(logits_p.float(), -1), F.log_softmax(logits_q.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


class TaskNames:
    """All scorable intermediate names of a task with their single-token forms."""

    def __init__(self, items: list[Item], task: str) -> None:
        self.names: list[str] = []
        self.forms: dict[str, list[int]] = {}
        for it in items:
            if it.task != task:
                continue
            for name in it.scorable:
                if name not in self.forms:
                    self.names.append(name)
                    self.forms[name] = it.intermediate_tokens[name]
        flat = [t for n in self.names for t in self.forms[n]]
        self.flat_ids = torch.tensor(flat)
        self.slices = np.cumsum([0, *(len(self.forms[n]) for n in self.names)])

    def ranks(self, logits: torch.Tensor) -> np.ndarray:
        """[n_names, N] min-over-forms rank of every name of the task."""
        r = _ranks_of(logits, self.flat_ids.to(logits.device)).cpu().numpy()  # [N, n_forms]
        return np.stack([r[:, a:b].min(1) for a, b in zip(self.slices[:-1], self.slices[1:])])

    def own_index(self, item: Item) -> list[int]:
        return [self.names.index(n) if n in self.forms else -1 for n in item.intermediates[:MAX_INTER]] + [-1] * (MAX_INTER - len(item.intermediates[:MAX_INTER]))


def pad_names(rows: np.ndarray, n: int) -> np.ndarray:
    out = np.full((n, *rows.shape[1:]), -1, np.int32)
    out[: len(rows)] = rows
    return out


@torch.no_grad()
def readout_arrays(m, items, H, J, exit_logits: torch.Tensor, target_ut: int, eventual=None) -> tuple[dict, list]:
    """Per-location readout for one lens. J: [192, d, d] on GPU (identity at the target), or None
    for the vanilla logit lens. `eventual` holds the eventual-exit lens logits per item (fp16 CPU)
    so the local-vs-eventual monitor gap is measured directly."""
    n, V = len(items), m.n_layers
    res = {
        "rank": np.full((n, MAX_INTER, V), -1, np.int32),
        "allrank": np.full((n, MAX_NAMES, V), -1, np.int32),
        "top1": np.zeros((n, V), np.int64),
        "kl_to_final": np.zeros((n, V), np.float32),
        "kl_to_local": np.zeros((n, V), np.float32),
        "rank_of_final_top1": np.zeros((n, V), np.int32),
        "rank_of_local_top1": np.zeros((n, V), np.int32),
    }
    if eventual is not None:
        res["kl_to_eventual_readout"] = np.zeros((n, V), np.float32)
    kept = []
    for i, item in enumerate(items):
        h = H[i].to(m.input_device)
        transported = h if J is None else torch.einsum("vde,ve->vd", J, h)
        logits = m.unembed(transported).float()  # [192, vocab]
        final, local = exit_logits[i, m.n_ut - 1], exit_logits[i, target_ut]
        tn = TASK_NAMES[item.task]
        allrank = tn.ranks(logits)
        res["allrank"][i] = pad_names(allrank, MAX_NAMES)
        for k, j in enumerate(tn.own_index(item)):
            if j >= 0:
                res["rank"][i, k] = allrank[j]
        res["top1"][i] = logits.argmax(-1).cpu().numpy()
        res["kl_to_final"][i] = kl(logits, final).cpu().numpy()
        res["kl_to_local"][i] = kl(logits, local).cpu().numpy()
        res["rank_of_final_top1"][i] = _ranks_of(logits, final.argmax().view(1))[:, 0].cpu().numpy()
        res["rank_of_local_top1"][i] = _ranks_of(logits, local.argmax().view(1))[:, 0].cpu().numpy()
        if eventual is not None:
            res["kl_to_eventual_readout"][i] = kl(logits, eventual[i].to(logits.device)).cpu().numpy()
        kept.append(logits.half().cpu())
    return res, kept


@torch.no_grad()
def cross_loop(m, items, H, J: torch.Tensor) -> np.ndarray:
    """allrank[item, name, fit_ut, applied_ut, layer]: J at (fit_ut, L) applied to state (applied_ut, L)."""
    n, U, L = len(items), m.n_ut, m.n_physical
    Jr = J.view(U, L, m.d_model, m.d_model)
    out = np.full((n, MAX_NAMES, U, U, L), -1, np.int32)
    for i, item in enumerate(items):
        h = H[i].to(J.device).view(U, L, m.d_model)
        # per layer: J at (i, L) applied to the state at (j, L) -> [i, d, j]; stacked as [L, i, d, j]
        transported = torch.stack([Jr[:, l] @ h[:, l].T for l in range(L)])
        transported = transported.permute(1, 3, 0, 2).reshape(U * U * L, m.d_model)  # order (i, j, L)
        logits = m.unembed(transported).float()
        out[i] = pad_names(TASK_NAMES[item.task].ranks(logits), MAX_NAMES).reshape(MAX_NAMES, U, U, L)
    return out


def stacked_jacobians(m, lens: jlens.JacobianLens, target: int) -> torch.Tensor:
    """Build a complete virtual-layer map only from an exact contiguous lens."""

    if not isinstance(target, int) or target <= 0 or target >= m.n_layers:
        raise IntegrityError(f"invalid virtual target layer {target!r}")
    expected = list(range(target))
    if list(lens.source_layers) != expected:
        raise IntegrityError(
            f"lens source_layers must be exactly contiguous 0..{target - 1}; "
            f"got {lens.source_layers}"
        )
    if int(lens.d_model) != int(m.d_model):
        raise IntegrityError(
            f"lens d_model={lens.d_model} is incompatible with model d_model={m.d_model}"
        )
    J = torch.eye(m.d_model).repeat(m.n_layers, 1, 1)
    for v in lens.source_layers:
        matrix = lens.jacobians[v]
        if tuple(matrix.shape) != (m.d_model, m.d_model):
            raise IntegrityError(f"invalid Jacobian shape at virtual layer {v}: {tuple(matrix.shape)}")
        J[v] = matrix
    return J.to(m.input_device)


def _evaluation_source_manifest() -> dict[str, object]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("evaldata.py").resolve(),
        Path(__file__).with_name("evidence.py").resolve(),
        Path(__file__).with_name("fit_lens.py").resolve(),
        Path(__file__).with_name("recurrent.py").resolve(),
    ]
    return source_manifest([path for path in paths if path.is_file()])


def load_lens_metadata(path: str | Path) -> dict[str, object]:
    """Load a lens only after verifying its sidecar and output hash."""
    metadata_path = sidecar_path(path)
    try:
        preview = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read lens sidecar {metadata_path}: {exc}") from exc
    kind = preview.get("kind") if isinstance(preview, dict) else None
    if kind not in {"fit", "merged"}:
        raise IntegrityError(f"unsupported lens sidecar kind {kind!r}: {metadata_path}")
    return validate_sidecar(path, kind=kind)


def validate_lens_metadata(
    metadata: dict[str, object],
    *,
    target_ut: int,
    model: object,
    baseline: dict[str, object] | None = None,
) -> None:
    """Check target, model shape, and cross-lens input/code identity."""

    target_virtual = int(model.exit_index(target_ut))
    if metadata.get("target_ut") != target_ut or metadata.get("target_virtual") != target_virtual:
        raise IntegrityError(
            f"lens metadata target mismatch: expected ut={target_ut}, virtual={target_virtual}"
        )
    if metadata.get("source_layers") != list(range(target_virtual)):
        raise IntegrityError("lens metadata source_layers are not exact contiguous sources")
    model_record = metadata.get("model")
    if not isinstance(model_record, dict):
        raise IntegrityError("lens metadata is missing model identity")
    current_revision = str(getattr(model, "model_revision", OURO_REVISION))
    if metadata.get("model_revision") != current_revision:
        raise IntegrityError(
            f"lens model revision {metadata.get('model_revision')!r} != loaded {current_revision!r}"
        )
    expected_shape = {
        "n_physical": int(model.n_physical),
        "n_ut": int(model.n_ut),
        "n_layers": int(model.n_layers),
        "d_model": int(model.d_model),
    }
    actual_shape = {key: model_record.get(key) for key in expected_shape}
    if actual_shape != expected_shape:
        raise IntegrityError(f"lens metadata model shape mismatch: {actual_shape} != {expected_shape}")
    identity_fields = (
        "model_revision",
        "model_snapshot_sha256",
        "jlens_commit",
        "prompt_file_sha256",
        "prompt_file_count",
        "source_sha256",
        "generator_sha256",
        "max_seq_len",
        "skip_first",
    )
    if baseline is not None:
        for field in identity_fields:
            if metadata.get(field) != baseline.get(field):
                raise IntegrityError(f"lens metadata {field} mismatch between inputs")


def _evaluation_inputs(tasks: list[str]) -> list[dict[str, object]]:
    """Record the exact evaluation stimulus bytes consumed by ``load_items``."""

    from ouro_jlens.evaldata import JLENS_DATA

    records = []
    for task in tasks:
        path = JLENS_DATA / f"lens-eval-{task}.json"
        records.append(file_record(path))
    return records


def _parse_lens_specs(values: list[str]) -> list[tuple[int, Path]]:
    specs: list[tuple[int, Path]] = []
    seen: set[int] = set()
    for value in values:
        try:
            target_text, path_text = value.split("=", 1)
            target_ut = int(target_text)
        except (TypeError, ValueError) as exc:
            raise IntegrityError(f"lens must be specified as target_ut=path, got {value!r}") from exc
        if target_ut in seen:
            raise IntegrityError(f"duplicate lens target_ut={target_ut}")
        seen.add(target_ut)
        specs.append((target_ut, Path(path_text)))
    return sorted(specs, reverse=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lens", action="append", required=True, help="target_ut=path")
    p.add_argument("--tasks", nargs="+", default=["multihop", "order-ops"])
    p.add_argument(
        "--position",
        type=int,
        default=-1,
        help="readout token position (default: last prompt token)",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.is_symlink():
        raise IntegrityError(f"evaluation output root is a symlink: {out}")
    out.mkdir(parents=True, exist_ok=True)

    m = load_ouro()
    specs = _parse_lens_specs(args.lens)
    if not specs or specs[0][0] != m.n_ut - 1:
        raise IntegrityError("the eventual-exit lens (target_ut = last) is required")

    # Verify all lens metadata before spending time on model forwards. Every
    # lens must describe this exact model and the same fit inputs/code; only
    # target_ut and its resulting source range may differ.
    lens_entries: list[tuple[int, Path, jlens.JacobianLens, dict[str, object]]] = []
    baseline: dict[str, object] | None = None
    for target_ut, path in specs:
        metadata = load_lens_metadata(path)
        validate_lens_metadata(metadata, target_ut=target_ut, model=m, baseline=baseline)
        baseline = metadata if baseline is None else baseline
        lens = jlens.JacobianLens.load(str(path))
        J_check = stacked_jacobians(m, lens, m.exit_index(target_ut))
        del J_check
        lens_entries.append((target_ut, path, lens, metadata))

    items = load_items(m.tokenizer, args.tasks, encode=lambda s: m.encode(s)[0].tolist())
    TASK_NAMES.clear()
    for task in args.tasks:
        TASK_NAMES[task] = TaskNames(items, task)
        if len(TASK_NAMES[task].names) > MAX_NAMES:
            raise IntegrityError(
                f"task {task!r} has {len(TASK_NAMES[task].names)} names, exceeds MAX_NAMES={MAX_NAMES}"
            )
    H, continuations = cache_states(m, items, args.position)
    exit_logits = torch.stack(
        [
            m.unembed(H[:, m.exit_index(ut)].to(m.input_device)).float()
            for ut in range(m.n_ut)
        ],
        dim=1,
    )

    arrays: dict[str, np.ndarray] = {"exit_top1": exit_logits.argmax(-1).cpu().numpy()}
    eventual = None
    for target_ut, path, lens, _metadata in lens_entries:
        J = stacked_jacobians(m, lens, m.exit_index(target_ut))
        res, kept = readout_arrays(m, items, H, J, exit_logits, target_ut, eventual)
        arrays.update({f"jlens_exit{target_ut}_{key}": value for key, value in res.items()})
        if target_ut == m.n_ut - 1:
            eventual = kept
            arrays["xloop_allrank"] = cross_loop(m, items, H, J)
        del J
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"lens exit{target_ut}: {lens} from {path}", flush=True)
    ll, _ = readout_arrays(m, items, H, None, exit_logits, m.n_ut - 1, eventual)
    arrays.update({f"logitlens_{key}": value for key, value in ll.items()})

    # Every output is independently replace-written. The provenance document
    # records the actual byte hashes after the arrays/JSON files are complete.
    arrays_path = out / "arrays.npz"
    items_path = out / "items.json"
    names_path = out / "task_names.json"
    provenance_path = out / "provenance.json"
    atomic_savez(arrays_path, **arrays)
    item_metadata = [
        {
            "name": it.name,
            "task": it.task,
            "prompt": it.prompt,
            "target": it.target,
            "intermediates": it.intermediates[:MAX_INTER],
            "own_index": TASK_NAMES[it.task].own_index(it),
            "scorable": [
                bool(it.intermediate_tokens[k]) for k in it.intermediates[:MAX_INTER]
            ],
            "leaked": [bool(it.leaked[k]) for k in it.intermediates[:MAX_INTER]],
            "n_tokens": len(it.token_ids),
            "readout_token": m.tokenizer.decode([it.token_ids[args.position]]),
            "continuation": c,
            "correct": is_correct(c, it.target),
            "exit_top1": [
                m.tokenizer.decode([token]) for token in arrays["exit_top1"][i]
            ],
        }
        for i, (it, c) in enumerate(zip(items, continuations))
    ]
    atomic_write_json(items_path, item_metadata)
    atomic_write_json(
        names_path,
        {task: task_names.names for task, task_names in TASK_NAMES.items()},
    )
    input_records = _evaluation_inputs(args.tasks)
    lens_records = []
    for target_ut, path, _lens, metadata in lens_entries:
        lens_records.append(
            {
                "target_ut": target_ut,
                "binary": file_record(path),
                "sidecar": file_record(sidecar_path(path)),
                "identity": {
                    key: metadata.get(key)
                    for key in (
                        "target_ut",
                        "target_virtual",
                        "source_layers",
                        "model_revision",
                        "jlens_commit",
                        "prompt_file_sha256",
                        "prompt_slice_sha256",
                        "source_sha256",
                        "generator_sha256",
                    )
                },
            }
        )
    source = _evaluation_source_manifest()
    provenance: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "revision": str(getattr(m, "model_revision", OURO_REVISION)),
            "n_physical": int(m.n_physical),
            "n_ut": int(m.n_ut),
            "n_layers": int(m.n_layers),
            "d_model": int(m.d_model),
        },
        "jlens_commit": lens_entries[0][3].get("jlens_commit"),
        "config": {
            "tasks": list(args.tasks),
            "position": args.position,
            "max_intermediates": MAX_INTER,
            "max_names": MAX_NAMES,
            "greedy_steps": GREEDY_STEPS,
        },
        "inputs": {
            "evaluation_files": input_records,
            "evaluation_input_sha256": sha256_json(input_records),
        },
        "lens_inputs": lens_records,
        "source_files": source["files"],
        "source_sha256": source["sha256"],
        "outputs": {
            "arrays": file_record(arrays_path),
            "items": file_record(items_path),
            "task_names": file_record(names_path),
        },
        "item_count": len(items),
        "correct_count": sum(bool(row["correct"]) for row in item_metadata),
        "item_metadata_sha256": sha256_json(item_metadata),
    }
    atomic_write_json(provenance_path, provenance)
    print(
        f"saved {out}: {len(items)} items, "
        f"model correct on target {provenance['correct_count']}/{len(items)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
