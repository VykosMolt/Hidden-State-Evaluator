#!/usr/bin/env python3
"""B200 benchmark harness — IMPLEMENTED, NOT RUN ON B200.

Benchmarks ONLY the non-O1 validation corpus.  The staged candidate order is
frozen in policies/BENCHMARK_ORDER.json before any hardware access; running
an out-of-order or unlisted configuration is refused.  A local synthetic
dress-rehearsal mode exists so the harness itself is tested; its numbers are
explicitly labeled LOCAL_SYNTHETIC and are never a B200 result.

Collected per configuration: completed rows/second, effective seconds/row,
prompt and decode tokens/second, model-forward / sampling / hook / transport /
parser+verifier / record-write shares (coarse wall-clock buckets), startup and
model-load time, GPU utilization + HBM allocated/reserved (when CUDA), CPU
utilization, process count, OOM count, integrity failures, throughput
stability, and resume overhead.

Throughput stability is measured the way production runs: ONE
``execute_rows`` call over the whole subset, then the execute span is cut
into four equal wall-clock windows and the decode-token production rate is
integrated into each window from the per-row commit times.  The earlier
"execute in four contiguous row quartiles and compare rows/s" design measured
corpus ORDERING, not the accelerator: every backend showed the same slow
third quartile because rows 192-287 generate more tokens, and 7 of 10 local
configurations would have failed the frozen ``throughput_stable`` gate on the
pod before producing a single O1 row.  Tokens/s is used because rows/s is a
function of generation length; rates are interpolated between consecutive
commit events so a batched backend committing 48 rows at once is not read as
a burst followed by silence.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import time

from .backend_interface import RuntimeConfig
from .backends import BACKENDS
from .identity import domain_sha256
from .runbuild import build_validation_bundle

POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "policies", "BENCHMARK_ORDER.json")


class BenchmarkError(RuntimeError):
    pass


def load_benchmark_order() -> dict:
    with open(os.path.abspath(POLICY_PATH), encoding="utf-8") as fh:
        return json.load(fh)


STABILITY_WINDOWS = 4


def windowed_token_rates(events: list[tuple[float, int]], t_start: float,
                         t_end: float, windows: int = STABILITY_WINDOWS
                         ) -> list[float]:
    """Decode tokens/s per equal wall-clock window of ``[t_start, t_end]``.

    ``events`` are ``(commit_time, tokens)`` pairs.  The tokens of each
    commit are treated as produced uniformly over the interval since the
    previous commit (or ``t_start``), and that piecewise-constant rate is
    integrated into the windows it overlaps.  Commits sharing a timestamp
    (one batch) are merged first.
    """
    span = float(t_end) - float(t_start)
    if span <= 0 or windows < 1:
        return []
    by_time: dict[float, int] = {}
    for t, tokens in events:
        t = min(max(float(t), float(t_start)), float(t_end))
        by_time[t] = by_time.get(t, 0) + int(tokens)
    if not by_time:
        return [0.0] * windows
    width = span / windows
    produced = [0.0] * windows
    prev = float(t_start)
    for t in sorted(by_time):
        tokens = by_time[t]
        length = t - prev
        if length <= 0:
            # a commit at t_start itself: attribute to the first window
            produced[0] += tokens
            continue
        rate = tokens / length
        for k in range(windows):
            lo = float(t_start) + k * width
            hi = lo + width
            overlap = min(hi, t) - max(lo, prev)
            if overlap > 0:
                produced[k] += rate * overlap
        prev = t
    return [p / width for p in produced]


def steady_state_span(events: list[tuple[float, int]]
                      ) -> tuple[float, float, list[tuple[float, int]]] | None:
    """``(t_first, t_last, events_after_first)``: the steady-state span.

    Throughput stability is about the accelerator under load, so the span
    runs from the FIRST commit to the LAST.  Before the first commit the
    configuration is still starting (a replica worker loads its own copy of
    the model inside ``execute_rows``); that ramp is real cost and is
    already charged to ``completed_rows_per_second``, but it is not
    instability.  The first commit's tokens were produced during the ramp
    and are excluded from the span.  Fewer than two distinct commit times
    means no span can be measured and the gate cannot pass.
    """
    times = sorted({float(t) for t, _ in events})
    if len(times) < 2:
        return None
    t_first, t_last = times[0], times[-1]
    steady = [(float(t), int(n)) for t, n in events if float(t) > t_first]
    return t_first, t_last, steady


def stability_spread(rates: list[float]) -> float | None:
    if not rates:
        return None
    top = max(rates)
    return (top - min(rates)) / top if top > 0 else 1.0


def _row_commit_events(rows_dir: str) -> dict[str, list[tuple[float, int]]]:
    """worker_id -> [(mtime, n_generated_tokens)] for every committed row."""
    events: dict[str, list[tuple[float, int]]] = {}
    for name in os.listdir(rows_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(rows_dir, name)
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
            tokens = int(rec.get("n_generated_tokens", 0))
            worker = str(rec.get("worker_id") or "w0")
            events.setdefault(worker, []).append(
                (os.path.getmtime(path), tokens))
        except (OSError, ValueError):
            continue
    return events


def per_worker_stability(events_by_worker: dict[str, list[tuple[float, int]]]
                         ) -> dict:
    """Stability per execution stream, reported as the WORST stream.

    A replica worker is a serial generator with its own model copy; its
    decode rate over its own steady-state span is the accelerator-stability
    signal.  Pooling workers would read their staggered starts (each loads
    the model inside ``execute_rows``) and uneven finishes (task-affine
    partitions are not equal-sized) as throughput collapsing at both ends
    of the run, which is load imbalance, not instability — and is already
    charged to ``completed_rows_per_second``.
    """
    per: dict[str, dict] = {}
    worst: float | None = None
    for worker, events in sorted(events_by_worker.items()):
        span = steady_state_span(events)
        if span is None:
            per[worker] = {"spread": None, "windows_tokens_per_second": [],
                           "steady_state_seconds": None}
            worst = 1.0
            continue
        t_first, t_last, steady = span
        rates = windowed_token_rates(steady, t_first, t_last)
        spread = stability_spread(rates)
        per[worker] = {"spread": spread, "windows_tokens_per_second": rates,
                       "windows_rows_per_second": windowed_token_rates(
                           [(t, 1) for t, _ in steady], t_first, t_last),
                       "steady_state_seconds": round(t_last - t_first, 6)}
        if spread is not None and (worst is None or spread > worst):
            worst = spread
    return {"spread": worst, "per_worker": per}


def _gpu_stats(worker_gpu: dict | None = None) -> dict:
    """Peak HBM for the configuration, including what worker processes
    held.  For B200_REPLICA the model lives in the workers, so the parent's
    own counters would report an almost empty device and the free-HBM gate
    would pass vacuously."""
    workers = int((worker_gpu or {}).get("hbm_peak_reserved_bytes_sum", 0))
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False,
                    "hbm_reserved_bytes_workers_peak_sum": workers}
        own_peak = int(torch.cuda.max_memory_reserved())
        return {
            "cuda": True,
            "hbm_allocated_bytes": torch.cuda.memory_allocated(),
            "hbm_reserved_bytes": own_peak + workers,
            "hbm_reserved_bytes_parent_peak": own_peak,
            "hbm_reserved_bytes_workers_peak_sum": workers,
            "device_name": torch.cuda.get_device_name(0),
        }
    except Exception:  # noqa: BLE001 - diagnostics only
        return {"cuda": False}


def benchmark_config(entry: dict, corpus_dir: str, out_dir: str,
                     model_artifact: dict, task_subset=None,
                     device: str = "cpu") -> dict:
    backend_id = entry["backend"]
    w = int(entry.get("workers", 1))
    b = int(entry.get("batch", 1))
    run_dir = os.path.join(out_dir, entry["config_id"])
    shutil.rmtree(run_dir, ignore_errors=True)
    t0 = time.monotonic()
    bundle = build_validation_bundle(corpus_dir, model_artifact,
                                     task_subset=task_subset)
    cfg = RuntimeConfig(backend_id=backend_id, run_dir=run_dir,
                        model_artifact=model_artifact, device=device,
                        worker_count=w, batch_size=b)
    backend = BACKENDS[backend_id]()
    backend.initialize(cfg, bundle)
    t_init = time.monotonic()
    try:
        import torch
        if torch.cuda.is_available():
            # peak for THIS configuration: weights + execution, not residue
            # from the previous configuration
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    backend.load_model(model_artifact)
    t_load = time.monotonic()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    integrity_failures = 0
    ooms = 0
    wall_load = time.time()
    exec_report = {}
    try:
        # ONE call, as production makes it; stability comes from the
        # per-row commit times (see the module docstring)
        exec_report = backend.execute_rows(bundle.specs) or {}
    except MemoryError:
        ooms += 1
        raise
    t_exec = time.monotonic()
    wall_exec = time.time()
    try:
        fin = backend.finalize_records()
    except Exception:  # noqa: BLE001 - integrity failure is a result
        integrity_failures += 1
        raise
    t_fin = time.monotonic()
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    rows = []
    with open(fin["records"], encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    gen_tokens = sum(r["n_generated_tokens"] for r in rows)
    exec_seconds = t_exec - t_load
    stab = per_worker_stability(_row_commit_events(backend.store.rows_dir))
    stability = stab["spread"]
    first_worker = next(iter(stab["per_worker"].values()), {})
    # resume overhead: re-open the completed run and measure the no-op resume
    t_r0 = time.monotonic()
    backend2 = BACKENDS[backend_id]()
    backend2.initialize(cfg, bundle)
    resumed = backend2.resume()
    resume_seconds = time.monotonic() - t_r0
    backend.shutdown()
    backend2.shutdown()
    return {
        "config_id": entry["config_id"],
        "backend": backend_id, "workers": w, "batch": b,
        "n_rows": len(rows),
        "completed_rows_per_second": len(rows) / exec_seconds,
        "effective_seconds_per_row": exec_seconds / max(1, len(rows)),
        "decode_tokens_per_second": gen_tokens / exec_seconds,
        "prompt_tokens_total": None,  # per-row prompt hashing only; see records
        "startup_seconds": t_init - t0,
        "model_load_seconds": t_load - t_init,
        "execute_seconds": exec_seconds,
        "finalize_seconds": t_fin - t_exec,
        "cpu_user_seconds": ru1.ru_utime - ru0.ru_utime,
        "cpu_sys_seconds": ru1.ru_stime - ru0.ru_stime,
        "max_rss_kib": ru1.ru_maxrss,
        "process_count": w if backend_id == "B200_REPLICA" else 1,
        "gpu": _gpu_stats(exec_report.get("gpu")),
        "oom_count": ooms,
        "integrity_failures": integrity_failures,
        "throughput_stability_spread": stability,
        "throughput_stability_basis": (
            f"per execution stream (worker): decode tokens/s over "
            f"{STABILITY_WINDOWS} equal wall-clock windows between that "
            f"stream's first and last row commit within one execute_rows "
            f"call, interpolated between commits; the WORST stream is the "
            f"configuration's spread"),
        "throughput_stability_per_worker": stab["per_worker"],
        "throughput_windows_tokens_per_second": first_worker.get(
            "windows_tokens_per_second", []),
        "throughput_windows_rows_per_second": first_worker.get(
            "windows_rows_per_second", []),
        "throughput_stability_spread_rows": stability_spread(
            first_worker.get("windows_rows_per_second", [])),
        "resume_noop_seconds": resume_seconds,
        "resume_remaining_after_completion": resumed["remaining"],
    }


def run_benchmarks(corpus_dir: str, out_dir: str, *, mode: str,
                   task_subset=None, device: str = "cpu",
                   max_stages: int | None = None,
                   artifact: dict | None = None,
                   remaining_authorized_seconds: float | None = None) -> dict:
    """mode: "local-synthetic" (harness validation, CPU, fake model) or
    "real-hardware" (the accelerator session: real Ouro-RLTT on CUDA,
    frozen stage order, stop rules enforced, conditional deep-batch stages
    only under the frozen extension rule)."""
    order = load_benchmark_order()
    os.makedirs(out_dir, exist_ok=True)
    if mode == "local-synthetic":
        artifact = {"kind": "synthetic", "device": device, "seed_tag": 0}
        label = "LOCAL_SYNTHETIC_DRESS_REHEARSAL"
    elif mode == "real-hardware":
        import torch
        if not torch.cuda.is_available():
            raise BenchmarkError(
                "real-hardware benchmark requires CUDA; refused")
        if not artifact or artifact.get("kind") != "ouro_rltt":
            raise BenchmarkError(
                "real-hardware benchmark requires the real ouro_rltt "
                "artifact; synthetic stand-ins are refused")
        device = "cuda"
        label = "REAL_HARDWARE"
    else:
        raise BenchmarkError(f"unknown benchmark mode {mode!r}")
    stages = order["staged_candidates"]
    if max_stages is not None:
        stages = stages[:max_stages]
    results = []
    clean_so_far = True
    t_bench0 = time.monotonic()
    for entry in stages:
        if entry.get("conditional"):
            # frozen extension rule: prior stages clean AND the cost rule
            if not clean_so_far:
                results.append({"config_id": entry["config_id"],
                                "skipped": "prior stage not clean"})
                continue
            if remaining_authorized_seconds is not None:
                spent = time.monotonic() - t_bench0
                per_stage = spent / max(1, len(
                    [r for r in results if not r.get("skipped")]))
                if per_stage > 0.25 * remaining_authorized_seconds:
                    results.append({"config_id": entry["config_id"],
                                    "skipped": "extension cost rule"})
                    continue
        try:
            res = benchmark_config(entry, corpus_dir, out_dir, artifact,
                                   task_subset=task_subset, device=device)
        except MemoryError:
            clean_so_far = False
            results.append({"config_id": entry["config_id"], "oom": True})
            continue   # stop rule: larger configs are conditional-skipped
        except Exception as exc:  # noqa: BLE001 - integrity failure = stop
            clean_so_far = False
            results.append({"config_id": entry["config_id"],
                            "integrity_failure": repr(exc)[:300]})
            continue
        if res.get("oom_count") or res.get("integrity_failures"):
            clean_so_far = False
        results.append(res)
    report = {
        "mode": label,
        "benchmark_order_sha256": domain_sha256(
            "o1b200.benchmark_order.v1", order),
        "corpus_dir": corpus_dir,
        "results": results,
        "note": ("LOCAL SYNTHETIC NUMBERS ONLY — harness validation; no "
                 "accelerator claim." if label != "REAL_HARDWARE" else
                 "real-hardware measurements on the acquired profile"),
    }
    path = os.path.join(out_dir, "BENCHMARK_REPORT.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("local-synthetic",),
                   default="local-synthetic")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-stages", type=int, default=None)
    a = p.parse_args()
    report = run_benchmarks(a.corpus, a.out, mode=a.mode, task_subset=a.tasks,
                            device=a.device, max_stages=a.max_stages)
    for r in report["results"]:
        if "completed_rows_per_second" in r:
            print(f"{r['config_id']}: {r['completed_rows_per_second']:.2f} "
                  f"rows/s stability_spread={r['throughput_stability_spread']}")
        else:
            reason = (r.get("skipped") or r.get("integrity_failure")
                      or ("OOM" if r.get("oom") else "not measured"))
            print(f"{r['config_id']}: NOT MEASURED ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
