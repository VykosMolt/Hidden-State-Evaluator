"""Mandatory validation milestones for the recurrent Jacobian-lens path.

The validator distinguishes numerical acceptance from bit-exact reproducibility.
The former is the gate used by automation; the latter is recorded explicitly
because attention kernels and backends can produce numerically equivalent
results without identical bytes. Reports are written atomically both under a
timestamped name and through a ``milestones.json`` latest copy.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

import jlens
from jlens.hooks import ActivationRecorder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ouro_jlens.evidence import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_write_json,
    file_record,
    sha256_file,
    source_manifest,
)
from ouro_jlens.recurrent import OURO_REVISION, PROJECT_ROOT, load_ouro  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "jlens" / "validation"
PROMPT = (
    "Fact: The capital of Japan is Tokyo.\n"
    "Fact: The currency used in the country shaped like a boot is"
)
LONG_PROMPT = (
    "The Jacobian of a function of several variables is the matrix of all its "
    "first-order partial derivatives. When the function maps a space to itself, "
    "the Jacobian is square and its determinant measures how the function "
    "locally scales volume. In deep networks the Jacobian of later activations "
    "with respect to earlier ones describes how a small change propagates."
)

TOL = 1e-6


def stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Return numerical and bit-exact comparison metrics for two tensors."""

    if tuple(a.shape) != tuple(b.shape):
        return {
            "equal": False,
            "close": False,
            "max_abs_diff": float("inf"),
            "rel_fro": float("inf"),
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    diff = (a.float() - b.float()).abs()
    denominator = b.float().norm().item()
    numerator = diff.norm().item()
    rel = numerator / denominator if denominator else (0.0 if numerator == 0 else float("inf"))
    return {
        "equal": bool(torch.equal(a, b)),
        "close": bool(torch.allclose(a.float(), b.float(), rtol=TOL, atol=TOL)),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "rel_fro": float(rel),
    }


def _exit_ut(m: Any) -> int:
    return int(m.n_ut - 1)


def _n_physical(m: Any) -> int:
    value = getattr(m, "n_physical", None)
    return int(value if value is not None else len(m.blocks))


def m1_noninterference(m: Any, ids: torch.Tensor) -> dict[str, Any]:
    hf = m.hf_model
    final_ut = _exit_ut(m)
    with torch.no_grad():
        ref = hf(ids, use_cache=False, exit_at_step=final_ut).logits
        with ActivationRecorder(m.layers, at=range(m.n_layers)) as rec:
            hooked = hf(ids, use_cache=False, exit_at_step=final_ut).logits
        after = hf(ids, use_cache=False, exit_at_step=final_ut).logits
    hooks_left = sum(len(b._forward_hooks) for b in m.blocks)
    hooked_stats = stats(hooked, ref)
    after_stats = stats(after, ref)
    return {
        "hooked_vs_plain": hooked_stats,
        "after_vs_plain": after_stats,
        "n_recorded": len(rec.activations),
        "hooks_left_after_exit": hooks_left,
        "pass": hooked_stats["close"]
        and after_stats["close"]
        and len(rec.activations) == m.n_layers
        and hooks_left == 0,
    }


def m2_exit_equality(m: Any, ids: torch.Tensor) -> dict[str, Any]:
    """Check every explicit exit and the model's default final exit."""

    hf = m.hf_model
    out: dict[str, Any] = {}
    with torch.no_grad():
        with ActivationRecorder(
            m.layers, at=[m.exit_index(ut) for ut in range(m.n_ut)]
        ) as rec:
            m.forward(ids)
        for ut in range(m.n_ut):
            lens_logits = m.unembed(rec.activations[m.exit_index(ut)])
            reference = hf(ids, use_cache=False, exit_at_step=ut).logits
            out[f"ut{ut}"] = stats(lens_logits, reference)
        default = hf(ids, use_cache=False).logits
        final = hf(ids, use_cache=False, exit_at_step=_exit_ut(m)).logits
        out["default_forward_vs_final"] = stats(default, final)
        # Retain the historical key for four-step Ouro reports while making
        # the new key independent of a hard-coded exit count.
        if _exit_ut(m) == 3:
            out["default_forward_vs_ut3"] = out["default_forward_vs_final"]
    out["pass"] = all(
        out[f"ut{ut}"]["close"] for ut in range(m.n_ut)
    ) and out["default_forward_vs_final"]["close"]
    return out


def m3_recurrent_identity(m: Any, ids: torch.Tensor) -> dict[str, Any]:
    """Check recurrent firing order, carries, and independent loop storage."""

    fired: list[tuple[int, int]] = []
    layer0_inputs: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def order_hook(layer: int) -> Callable[..., Any]:
        def hook(module: Any, args: Any, kwargs: dict[str, Any], output: Any) -> None:
            fired.append((kwargs["current_ut"], layer))

        return hook

    def layer0_pre(module: Any, args: Any, kwargs: dict[str, Any]) -> None:
        layer0_inputs[kwargs["current_ut"]] = args[0]

    try:
        for i, block in enumerate(m.blocks):
            handles.append(block.register_forward_hook(order_hook(i), with_kwargs=True))
        handles.append(m.blocks[0].register_forward_pre_hook(layer0_pre, with_kwargs=True))
        with torch.no_grad():
            with ActivationRecorder(m.layers, at=range(m.n_layers)) as rec:
                _, hidden_states_list, _ = m.forward(ids)
    finally:
        for handle in handles:
            handle.remove()
    # The native-block recorder needs a second forward.  Run it only after the
    # order/carry hooks above are removed, otherwise that second forward is
    # accidentally counted as part of the first-forward order assertion.
    with torch.no_grad():
        with ActivationRecorder(m.blocks, at=range(_n_physical(m))) as stock:
            m.forward(ids)

    expected_order = [
        (ut, layer)
        for ut in range(m.n_ut)
        for layer in range(_n_physical(m))
    ]
    acts = rec.activations
    distinct_ptrs = all(
        len({acts[m.index(ut, layer)].data_ptr() for ut in range(m.n_ut)}) == m.n_ut
        for layer in range(_n_physical(m))
    )
    distinct_values = all(
        not torch.equal(acts[m.index(ut, layer)], acts[m.index(ut + 1, layer)])
        for layer in range(_n_physical(m))
        for ut in range(m.n_ut - 1)
    )
    boundary = {
        f"ut{ut}": stats(
            m._final_norm(acts[m.exit_index(ut)]), hidden_states_list[ut]
        )
        for ut in range(m.n_ut)
    }
    carry = {
        f"ut{ut}->{ut + 1}": stats(
            layer0_inputs[ut + 1], m._final_norm(acts[m.exit_index(ut)])
        )
        for ut in range(m.n_ut - 1)
    }
    embed_in = stats(layer0_inputs[0], m._embed_tokens(ids))
    stock_is_last_loop = all(
        torch.equal(
            stock.activations[layer],
            acts[m.index(m.n_ut - 1, layer)],
        )
        for layer in range(_n_physical(m))
    )
    return {
        "fire_order_ok": fired == expected_order,
        "n_fired": len(fired),
        "distinct_storage": distinct_ptrs,
        "distinct_values_between_loops": distinct_values,
        "norm_of_recorded_L47_equals_native_hidden_states_list": boundary,
        "next_loop_input_equals_normed_L47": carry,
        "loop1_layer0_input_equals_embeddings": embed_in,
        "stock_recorder_returns_last_loop": stock_is_last_loop,
        "pass": fired == expected_order
        and distinct_ptrs
        and distinct_values
        and all(value["close"] for value in boundary.values())
        and all(value["close"] for value in carry.values())
        and embed_in["close"]
        and stock_is_last_loop,
    }


def m4_distinct_vjps(m: Any, ids: torch.Tensor) -> dict[str, Any]:
    """Check distinct VJPs, explicit causal locality, and gradient plumbing."""

    n_physical = _n_physical(m)
    source_layer = min(16, max(0, n_physical - 2))
    src = [m.index(ut, source_layer) for ut in range(m.n_ut)]
    target_index = m.exit_index(_exit_ut(m))
    out: dict[str, Any] = {
        "source_virtual_layers": src,
        "source_physical_layer": source_layer,
    }
    with torch.enable_grad(), ActivationRecorder(
        m.layers, at=[*src, target_index], start_graph_at=src[0]
    ) as rec:
        m.forward(ids)
        target = rec.activations[target_index]
        sources = [rec.activations[source] for source in src]
        for source in sources:
            if not source.requires_grad:
                raise RuntimeError("source detached from graph")
        for dimension in range(min(3, m.d_model)):
            cotangent = torch.zeros_like(target)
            cotangent[0, -1, dimension] = 1.0
            grads = torch.autograd.grad(
                target, sources, cotangent, retain_graph=True
            )
            first = grads[0][0]
            earlier_norm = first[:-1].float().norm().item() if first.shape[0] > 1 else 0.0
            last_norm = first[-1].float().norm().item()

            # A penultimate-position cotangent gives a real causality check:
            # the final source position must not influence an earlier target.
            if target.shape[1] > 1:
                earlier_target_cot = torch.zeros_like(target)
                earlier_target_cot[0, -2, dimension] = 1.0
                earlier_target_grads = torch.autograd.grad(
                    target, sources, earlier_target_cot, retain_graph=True
                )
                later_norm = earlier_target_grads[0][0, -1].float().norm().item()
            else:
                later_norm = 0.0
            out[f"dim{dimension}"] = {
                "shapes": [list(gradient.shape) for gradient in grads],
                "distinct_ptrs": len({gradient.data_ptr() for gradient in grads}) == len(grads),
                "norms": [gradient.float().norm().item() for gradient in grads],
                "pairwise_equal": [
                    bool(torch.equal(grads[i], grads[j]))
                    for i in range(len(grads))
                    for j in range(i + 1, len(grads))
                ],
                "rel_diff_first_two": (
                    (grads[0].float() - grads[1].float()).norm().item()
                    / grads[1].float().norm().item()
                    if len(grads) > 1 and grads[1].float().norm().item()
                    else float("inf")
                ),
                "earlier_position_gradient_norm": earlier_norm,
                "last_position_gradient_norm": last_norm,
                "earlier_position_gradient_nonzero": earlier_norm > 0.0,
                "last_position_gradient_nonzero": last_norm > 0.0,
                "penultimate_target_later_position_gradient_norm": later_norm,
                "causal_later_position_gradient_zero": later_norm <= TOL,
                "causal_position_gradients_pass": earlier_norm > 0.0
                and last_norm > 0.0
                and later_norm <= TOL,
            }
        # Gradient of target with respect to itself is the cotangent.
        self_grad = torch.autograd.grad(target, [target], cotangent, retain_graph=False)[0]
        out["self_grad_is_cotangent"] = bool(torch.equal(self_grad, cotangent))
    dimensions = [value for key, value in out.items() if key.startswith("dim")]
    out["pass"] = bool(dimensions) and all(
        value["distinct_ptrs"]
        and not any(value["pairwise_equal"])
        and all(norm > 0 for norm in value["norms"])
        and value["shapes"] == [[1, ids.shape[1], m.d_model]] * len(src)
        and value["causal_position_gradients_pass"]
        for value in dimensions
    ) and out["self_grad_is_cotangent"]
    return out


def m5_stock_consistency(
    m: Any, prompt: str, dim_batch: int = 8, max_seq_len: int = 64
) -> dict[str, Any]:
    """Stock jlens must reproduce the final recurrent-loop Jacobian."""

    stock = jlens.HFLensModel(m.hf_model, m.tokenizer, force_bos=False)
    stock.encode = m.encode
    physical_source = min(40, max(0, _n_physical(m) - 2))
    while True:
        try:
            t0 = time.perf_counter()
            J_stock, seq_len, n_valid = jlens.jacobian_for_prompt(
                stock,
                prompt,
                [physical_source],
                target_layer=_n_physical(m) - 1,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
            )
            t_stock = time.perf_counter() - t0
            t0 = time.perf_counter()
            J_rec, _, _ = jlens.jacobian_for_prompt(
                m,
                prompt,
                [m.index(_exit_ut(m), physical_source), m.index(max(0, _exit_ut(m) - 1), physical_source)],
                target_layer=m.exit_index(_exit_ut(m)),
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
            )
            t_rec = time.perf_counter() - t0
            break
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if dim_batch == 1:
                raise
            dim_batch //= 2
            print(f"  m5: out of memory, retrying at dim_batch={dim_batch}", flush=True)
    same = stats(J_rec[m.index(_exit_ut(m), physical_source)], J_stock[physical_source])
    other = stats(
        J_rec[m.index(max(0, _exit_ut(m) - 1), physical_source)],
        J_stock[physical_source],
    )
    return {
        "seq_len": seq_len,
        "n_valid": n_valid,
        "dim_batch": dim_batch,
        "stock_L40_vs_recurrent_final_L40": same,
        "stock_L40_vs_recurrent_previous_loop_L40": other,
        "stock_L40_vs_recurrent_ut3_L40": same,
        "stock_L40_vs_recurrent_ut2_L40": other,
        "stock_seconds": t_stock,
        "recurrent_seconds": t_rec,
        "pass": same["close"] and not other["close"],
    }


def _jlens_commit() -> str:
    try:
        repo = Path(jlens.__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "UNKNOWN"
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "UNKNOWN"


def runtime_provenance(m: Any) -> dict[str, Any]:
    """Capture hardware/backend and validator source identity."""

    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory": int(properties.total_memory),
                    "capability": [properties.major, properties.minor],
                }
            )
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("recurrent.py").resolve(),
        Path(__file__).with_name("evidence.py").resolve(),
    ]
    source = source_manifest([path for path in source_paths if path.is_file()])
    return {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_revision": str(getattr(m, "model_revision", OURO_REVISION)),
        "jlens_commit": _jlens_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": {
            "available": cuda_available,
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": devices,
        },
        "backend": {
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "model_shape": {
            "n_layers": int(m.n_layers),
            "d_model": int(m.d_model),
            "n_physical": _n_physical(m),
            "n_ut": int(m.n_ut),
        },
        "source_files": source["files"],
        "source_sha256": source["sha256"],
    }


def _lookup(report: dict[str, Any], dotted: str) -> Any:
    value: Any = report
    for part in dotted.split("."):
        value = value[part]
    return value


def required_bit_exact_comparisons(report: dict[str, Any]) -> list[str]:
    """Explicitly enumerate every comparison included in bit-exact rollup."""

    paths = [
        "m1_noninterference.hooked_vs_plain",
        "m1_noninterference.after_vs_plain",
        "m2_exit_equality.default_forward_vs_final",
    ]
    paths.extend(
        f"m2_exit_equality.ut{ut}" for ut in range(report["model"]["n_ut"])
    )
    m3 = report.get("m3_recurrent_identity", {})
    boundary = m3.get("norm_of_recorded_L47_equals_native_hidden_states_list", {})
    carry = m3.get("next_loop_input_equals_normed_L47", {})
    n_ut = int(report.get("model", {}).get("n_ut", 0))
    paths.extend(
        f"m3_recurrent_identity.norm_of_recorded_L47_equals_native_hidden_states_list.{key}"
        for key in (boundary or {f"ut{ut}": None for ut in range(n_ut)})
    )
    paths.extend(
        f"m3_recurrent_identity.next_loop_input_equals_normed_L47.{key}"
        for key in (carry or {f"ut{ut}->{ut + 1}": None for ut in range(max(0, n_ut - 1))})
    )
    paths.append("m3_recurrent_identity.loop1_layer0_input_equals_embeddings")
    paths.append("m3_recurrent_identity.stock_recorder_returns_last_loop")
    paths.append("m4_distinct_vjps.self_grad_is_cotangent")
    if "m5_stock_consistency" in report:
        paths.append("m5_stock_consistency.stock_L40_vs_recurrent_final_L40")
    return paths


def bit_exact_rollup(report: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    paths = required_bit_exact_comparisons(report)
    comparisons: list[dict[str, Any]] = []
    for path in paths:
        try:
            value = _lookup(report, path)
        except (KeyError, TypeError):
            comparisons.append({"path": path, "equal": False, "close": False})
            continue
        if isinstance(value, dict):
            comparisons.append(
                {
                    "path": path,
                    "equal": bool(value.get("equal", False)),
                    "close": bool(value.get("close", False)),
                }
            )
        else:
            comparisons.append({"path": path, "equal": bool(value), "close": bool(value)})
    return comparisons, all(item["equal"] for item in comparisons)


def write_report(report: dict[str, Any], out: str | Path = OUT) -> tuple[Path, Path]:
    """Atomically write a timestamped report and latest pointer/copy."""

    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    timestamped = destination / f"milestones_{stamp}.json"
    latest = destination / "milestones.json"
    latest_pointer = destination / "latest.json"
    payload = dict(report)
    payload["report_file"] = timestamped.name
    atomic_write_json(timestamped, payload)
    atomic_write_json(latest, payload)
    atomic_write_json(
        latest_pointer,
        {
            "schema_version": SCHEMA_VERSION,
            "latest_report": timestamped.name,
            "latest_sha256": sha256_file(timestamped),
        },
    )
    return timestamped, latest


def run_validation(model: Any | None = None, ids: torch.Tensor | None = None) -> dict[str, Any]:
    """Execute all milestones and return a report, preserving failures."""

    m = model or load_ouro()
    input_ids = ids if ids is not None else m.encode(PROMPT)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prompt_tokens": int(input_ids.shape[1]),
        "model": {
            "n_layers": int(m.n_layers),
            "d_model": int(m.d_model),
            "n_physical": _n_physical(m),
            "n_ut": int(m.n_ut),
        },
        "provenance": runtime_provenance(m),
    }
    milestones: list[tuple[str, Callable[..., dict[str, Any]], tuple[Any, ...]]] = [
        ("m1_noninterference", m1_noninterference, (m, input_ids)),
        ("m2_exit_equality", m2_exit_equality, (m, input_ids)),
        ("m3_recurrent_identity", m3_recurrent_identity, (m, input_ids)),
        ("m4_distinct_vjps", m4_distinct_vjps, (m, input_ids)),
        ("m5_stock_consistency", m5_stock_consistency, (m, LONG_PROMPT)),
    ]
    for name, function, function_args in milestones:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            result = function(*function_args)
        except Exception as exc:  # preserve an executable failure report
            result = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        result["seconds"] = round(time.perf_counter() - started, 3)
        result["peak_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            3,
        )
        report[name] = result
        print(
            f"{name}: {'PASS' if result.get('pass') else 'FAIL'} "
            f"({result['seconds']}s, {result['peak_gb']} GB)",
            flush=True,
        )
    comparisons, bit_exact = bit_exact_rollup(report)
    report["required_bit_exact_comparisons"] = [item["path"] for item in comparisons]
    report["bit_exact_comparisons"] = comparisons
    report["bit_exact"] = bit_exact
    report["numerical_pass"] = all(
        report[name].get("pass", False) for name, _, _ in milestones
    )
    report["pass"] = report["numerical_pass"]
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_validation()
    write_report(report, args.out)
    print(
        "ALL PASS" if report["numerical_pass"] else "SOME FAILED",
        "(bit-exact)" if report["bit_exact"] else f"(within {TOL:g}, not bit-exact)",
    )
    return 0 if report["numerical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
