"""Precomputed batched backend for the SEALED v2.1 calibration orchestrator.

Why this exists
---------------
``orchestrate_calibration`` asks its backend for one row at a time and uses
each answer immediately (the baseline bank's boundary feeds the transport of
every structured row of the same task).  On a B300 a single sealed row costs
~23-35 s and that latency is architectural: total_ut_steps=4 over 48 layers is
~192 layer passes per token, and the measured per-row latency is FLAT from
batch 1 to batch 64.  All throughput therefore comes from batch parallelism,
and 4,608 serial rows is ~30-45 h of rented GPU.

A batching *proxy* is impossible: ``.generate()`` is synchronous.  So this
backend PRECOMPUTES.  The calibration request set is fully determined before
the first token is generated (task order, bank size, the sealed Latin square,
the swept alpha grid and the deterministic seed derivation are all sealed), so
we enumerate every ``(task, seed, direction, signed_alpha)`` the orchestrator
will ask for, generate them through the equivalence-gated batched engine in
large batches, and then serve ``.generate()`` as a map lookup.

The sealed package is NOT modified: ``run_calibration``/
``orchestrate_calibration`` already accepts ``backend=``, and this class
implements exactly the two methods it uses (``generate`` / ``decode``) with
``RealBackend`` semantics.

NO SILENT FALLBACK
------------------
A lookup miss means the enumeration disagrees with the sealed loop.  That is a
stop-the-run condition: serving it by generating serially would produce a
records file that looks correct while mixing two engines, and the mixture
would be invisible in the artifacts.  Every miss raises
``MissingPrecomputedRow`` naming the key.  There is no default result, no
lazy serial path, and no swallowed exception anywhere in this module.

Memory
------
Every precomputed row is buffered until the orchestrator has consumed it, and
the orchestrator only reaches the last task at the very end, so in practice
all 4,608 results are resident.  Per row, worst case:

    generated_token_ids   <= 448 ints   ~16 KB (list ptrs + int objects)
    _prompt_token_ids     <= 1536 ints  ~55 KB   (prompt truncation limit)
    generated_text        <= ~2 KB
    _prefill_l4_47        2048 float32   8 KB
                                        -------
                                        ~80 KB / row

4,608 x ~80 KB ~= 370 MB worst case, and roughly 100-200 MB for realistic
prompt/answer lengths.  That is a rounding error next to the model itself on
the rented host, so full buffering is accepted rather than adding a spill
file: an on-disk buffer would introduce a second copy of the scientific
payload that nothing else verifies.  (The largest term, the per-row prompt
token list, is identical for the 48 rows of one task and could be interned if
a future manifest ever makes this matter.)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import sealed_import

sealed_import.ensure_sealed_path()

from o1_analysis import (  # noqa: E402 (sealed)
    BANK_SIZE,
    N_DIRECTIONS,
    action_index,
    derive_stream_seed,
    load_jsonl,
)
from run_o1_v2_orchestrator import (  # noqa: E402 (sealed)
    HIDDEN,
    SIGNED_ACTIONS,
    load_axis_tensor,
)

__all__ = [
    "PrecomputedBackendError",
    "MissingPrecomputedRow",
    "CalibrationRequest",
    "PrecomputedBatchedBackend",
    "canonical_alpha",
    "direction_digest",
    "request_key",
    "enumerate_calibration_groups",
]


class PrecomputedBackendError(RuntimeError):
    """Enumeration, precompute or serving fault.  Always fatal to the run."""


class MissingPrecomputedRow(PrecomputedBackendError):
    """``.generate()`` was asked for a row the enumeration never planned."""


# ==========================================================================
# keying
# ==========================================================================


def canonical_alpha(signed_alpha) -> str:
    """Round-trip-exact text form of a signed alpha, with -0.0 == +0.0.

    ``%.17g`` round-trips every IEEE double, so two alphas share a key only
    if they are the same double.  The zero collapse matters because the
    orchestrator computes ``s * alpha``: the negative action of alpha 0.0
    produces -0.0, which is ``==`` +0.0 but has different bytes.
    """
    a = float(signed_alpha)
    if a != a:
        raise PrecomputedBackendError("signed_alpha is NaN")
    if a == 0.0:
        a = 0.0                       # collapses -0.0 onto +0.0
    return f"{a:.17g}"


def direction_digest(direction) -> str | None:
    """None for baseline, else sha256 of the float64 axis-row bytes.

    Keyed on CONTENT, never on array identity: the orchestrator hands out
    ``axes[d - 1]`` views and nothing guarantees the same object reaches us.
    """
    if direction is None:
        return None
    arr = np.ascontiguousarray(np.asarray(direction, dtype=np.float64))
    if arr.ndim != 1:
        raise PrecomputedBackendError(
            f"direction must be a 1-D axis row, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise PrecomputedBackendError("direction contains non-finite values")
    return hashlib.sha256(arr.tobytes()).hexdigest()


#: Above 2**53 a float64 no longer represents every integer, and
#: ``derive_stream_seed`` returns a full-width uint64.  ``float(seed)`` then
#: silently lands on a NEIGHBOURING integer, so a float spelling of a real
#: stream seed is not the same seed.  Such a spelling is rejected outright
#: rather than turned into a mystifying "never precomputed" miss.
_EXACT_FLOAT_INT = 2 ** 53


def _canonical_seed(seed) -> int:
    """Accept the int and float spellings of the same seed, reject the rest."""
    if isinstance(seed, bool):
        raise PrecomputedBackendError("seed must not be a bool")
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    if isinstance(seed, (float, np.floating)):
        f = float(seed)
        if not f.is_integer():
            raise PrecomputedBackendError(f"seed {seed!r} is not an integer")
        if abs(f) >= _EXACT_FLOAT_INT:
            raise PrecomputedBackendError(
                f"seed was passed as the float {seed!r}, which is >= 2**53 and "
                f"therefore cannot exactly represent a 64-bit stream seed; "
                f"pass the integer returned by derive_stream_seed")
        return int(f)
    raise PrecomputedBackendError(f"seed {seed!r} has non-numeric type")


def request_key(task_id: str, seed, direction, signed_alpha) -> tuple:
    """The canonical lookup key: (task_id, seed, alpha, direction digest)."""
    if not isinstance(task_id, str) or not task_id:
        raise PrecomputedBackendError("task_id must be a non-empty string")
    return (task_id, _canonical_seed(seed), canonical_alpha(signed_alpha),
            direction_digest(direction))


# ==========================================================================
# enumeration
# ==========================================================================


@dataclass(frozen=True)
class CalibrationRequest:
    """One row the sealed calibration loop will ask the backend to generate."""

    task: dict
    seed: int
    direction: Any          # np.ndarray float64 axis row, or None
    signed_alpha: float
    arm: str                # "baseline" | "structured"
    stream_index: int
    direction_id: int | None
    sign: int | None
    alpha: float

    @property
    def key(self) -> tuple:
        return request_key(self.task["task_id"], self.seed, self.direction,
                           self.signed_alpha)

    def describe(self) -> str:
        return (f"task={self.task['task_id']} arm={self.arm} "
                f"alpha={self.alpha!r} d={self.direction_id} s={self.sign} "
                f"stream={self.stream_index} seed={self.seed}")


def enumerate_calibration_groups(tasks: list[dict], axes: np.ndarray,
                                 grid: list[float],
                                 master_seed: int) -> list[list[CalibrationRequest]]:
    """Mirror ``orchestrate_calibration``'s request order, EXACTLY.

    Sealed source of truth (run_o1_v2_orchestrator.py, calibration loop and
    ``_run_baseline_bank``)::

        for g, task in enumerate(tasks):
            rank  = g % BANK_SIZE
            seeds = {i: derive_stream_seed(master, task_id, i) for i in ...}
            for i in range(BANK_SIZE):                  # baseline bank
                backend.generate(task, seeds[i], None, 0.0)
            for alpha in grid:
                for d, s in SIGNED_ACTIONS:
                    stream = (action_index(d, s) + rank) % BANK_SIZE
                    backend.generate(task, seeds[stream], axes[d-1], s * alpha)

    The resume path (``h_base is None``) re-requests ``(task, seeds[0], None,
    0.0)``, which is already the first member of the baseline bank, so it
    needs no extra entry.

    Returns GROUPS of BANK_SIZE requests.  A group is the unit of batching:
    the sealed orchestrator requires the 8 baseline streams of a task to
    produce a BITWISE-identical prefill boundary, so those 8 rows must be
    generated together in one batch, never split across two batch shapes.
    """
    if not tasks:
        raise PrecomputedBackendError("calibration task list is empty")
    if not grid:
        raise PrecomputedBackendError("alpha grid is empty")
    groups: list[list[CalibrationRequest]] = []
    for g, task in enumerate(tasks):
        task_id = task["task_id"]
        rank = g % BANK_SIZE
        seeds = {i: derive_stream_seed(master_seed, task_id, i)
                 for i in range(BANK_SIZE)}
        groups.append([
            CalibrationRequest(task=task, seed=seeds[i], direction=None,
                               signed_alpha=0.0, arm="baseline",
                               stream_index=i, direction_id=None, sign=None,
                               alpha=0.0)
            for i in range(BANK_SIZE)])
        for alpha in grid:
            group = []
            for d, s in SIGNED_ACTIONS:
                stream = (action_index(d, s) + rank) % BANK_SIZE
                group.append(CalibrationRequest(
                    task=task, seed=seeds[stream], direction=axes[d - 1],
                    signed_alpha=s * alpha, arm="structured",
                    stream_index=stream, direction_id=d, sign=s,
                    alpha=float(alpha)))
            groups.append(group)
    return groups


def _default_generate_batch():
    """Import the equivalence-gated batched engine (torch) only when needed."""
    from .engine_batched import generate_batch
    return generate_batch


# ==========================================================================
# the backend
# ==========================================================================


#: The only backend family the precomputed calibration path can run.
#: B200_REPLICA parallelises across PROCESSES at batch 1 per row; there is
#: nothing for generate_batch to batch, so it cannot serve this path.
BATCHED_CALIBRATION_BACKEND = "B200_BATCHED"


def supports_calibration(backend_id, workers, batch) -> tuple[bool, str]:
    """Can PrecomputedBatchedBackend actually run this configuration?

    THE AFFORDABILITY GATE AND THE LAUNCHER MUST BOTH ASK THIS, and they must
    ask the same function.  Asking it in only one place is how the gate comes
    to project a rate the calibration will not deliver:

      * every B200_REPLICA stage in the frozen order is batch 1.  A launcher
        that keys on ``batch <= 1`` injects nothing, the sealed RealBackend is
        built, and the calibration runs SERIALLY while the gate projected the
        8-worker replica rate -- ~3.9 h projected against ~30.7 h actual.  The
        session is killed part-way through the corpus having paid for all of
        it, and the precommit records a backend that did not produce the rows.
      * B200_BATCHED_w1_b4 is a non-conditional stage, so it can be selected,
        and batch 4 < BANK_SIZE is refused at construction -- aborting the run
        after the whole pre-calibration phase has been paid for.

    Returns (ok, reason); ``reason`` is empty when ok.
    """
    try:
        workers = int(workers or 1)
        batch = int(batch or 1)
    except (TypeError, ValueError):
        return False, f"workers/batch are not integers: {workers!r}/{batch!r}"
    if str(backend_id or "") != BATCHED_CALIBRATION_BACKEND:
        return False, (f"backend {backend_id!r} is not "
                       f"{BATCHED_CALIBRATION_BACKEND}: the precomputed "
                       f"calibration path batches rows, and only that engine "
                       f"does")
    if workers != 1:
        return False, (f"workers={workers}: the precomputed path generates in "
                       f"one process")
    if batch < BANK_SIZE:
        return False, (f"batch={batch} < BANK_SIZE {BANK_SIZE}: the sealed "
                       f"orchestrator requires all {BANK_SIZE} baseline "
                       f"streams of a task to yield a bitwise-identical "
                       f"prefill boundary, so the bank must fit in one batch")
    return True, ""


class PrecomputedBatchedBackend:
    """Backend injectable into the sealed calibration orchestrator.

    Construction enumerates and keys the whole request set (no torch, no GPU,
    no checkpoint required).  ``precompute()`` loads the sealed model and
    fills every result through ``engine_batched.generate_batch``.
    ``generate()`` then serves lookups only.
    """

    def __init__(self, tasks: list[dict], axes, grid, master_seed: int, *,
                 batch_size: int = 32,
                 checkpoint: str | None = None,
                 model=None, tokenizer=None,
                 manifest_raw: dict | None = None,
                 max_new_tokens: int | None = None,
                 generate_fn: Callable | None = None,
                 progress_cb: Callable[[dict], None] | None = None,
                 verify_baseline_boundaries: bool = True):
        self.tasks = list(tasks)
        ids = [t["task_id"] for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise PrecomputedBackendError(
                "duplicate task_id in the calibration task manifest")
        self.axes = np.asarray(axes, dtype=np.float64)
        if self.axes.shape != (N_DIRECTIONS, HIDDEN):
            raise PrecomputedBackendError(
                f"axis tensor shape {self.axes.shape} is not "
                f"({N_DIRECTIONS}, {HIDDEN})")
        if not np.isfinite(self.axes).all():
            raise PrecomputedBackendError("axis tensor is not finite")
        self.grid = [float(x) for x in grid]
        self.master_seed = int(master_seed)
        self.batch_size = int(batch_size)
        if self.batch_size < BANK_SIZE:
            raise PrecomputedBackendError(
                f"batch_size {self.batch_size} < BANK_SIZE {BANK_SIZE}: the "
                f"sealed orchestrator requires all {BANK_SIZE} baseline "
                f"streams of a task to yield a bitwise-identical prefill "
                f"boundary, so the bank must fit in one batch")
        self._checkpoint = checkpoint
        self._model = model
        self._tokenizer = tokenizer
        self._manifest_raw = manifest_raw
        self._max_new_tokens = max_new_tokens
        self._generate_fn = generate_fn
        self._progress_cb = progress_cb
        self._verify_baseline_boundaries = bool(verify_baseline_boundaries)

        self.groups = enumerate_calibration_groups(
            self.tasks, self.axes, self.grid, self.master_seed)
        self.requests: list[CalibrationRequest] = [
            req for group in self.groups for req in group]

        # Unique keys are the whole safety argument for map-lookup serving:
        # a collision would silently serve one row's tokens for another.
        self._index: dict[tuple, int] = {}
        for i, req in enumerate(self.requests):
            key = req.key
            prev = self._index.get(key)
            if prev is not None:
                raise PrecomputedBackendError(
                    "enumerated calibration requests are not unique: "
                    f"{req.describe()} collides with "
                    f"{self.requests[prev].describe()} on key {key!r}")
            self._index[key] = i
        expected = len(self.tasks) * BANK_SIZE * (1 + len(self.grid))
        if len(self.requests) != expected:
            raise PrecomputedBackendError(
                f"enumerated {len(self.requests)} requests, sealed row count "
                f"is {expected}")

        self._results: list[dict | None] = [None] * len(self.requests)
        self._served: list[int] = [0] * len(self.requests)
        self.precompute_seconds = 0.0
        self.batches_run = 0

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def __len__(self) -> int:
        return len(self.requests)

    def keys(self) -> list[tuple]:
        return list(self._index)

    def n_precomputed(self) -> int:
        return sum(1 for r in self._results if r is not None)

    def unserved_indices(self) -> list[int]:
        return [i for i, n in enumerate(self._served) if n == 0]

    def coverage_report(self) -> dict:
        return {
            "schema": "o1b200.precomputed_calibration_coverage.v1",
            "tasks": len(self.tasks),
            "bank_size": BANK_SIZE,
            "alpha_grid": list(self.grid),
            "requests_enumerated": len(self.requests),
            "requests_precomputed": self.n_precomputed(),
            "requests_served_at_least_once":
                len(self.requests) - len(self.unserved_indices()),
            "generate_calls": int(sum(self._served)),
            "batches_run": self.batches_run,
            "batch_size": self.batch_size,
            "precompute_seconds": round(self.precompute_seconds, 3),
        }

    # ------------------------------------------------------------------
    # batching plan
    # ------------------------------------------------------------------

    def batches(self) -> list[list[int]]:
        """Chunk the request indices, never splitting a BANK_SIZE group."""
        out: list[list[int]] = []
        cur: list[int] = []
        pos = 0
        for group in self.groups:
            idxs = list(range(pos, pos + len(group)))
            pos += len(group)
            if cur and len(cur) + len(idxs) > self.batch_size:
                out.append(cur)
                cur = []
            cur.extend(idxs)
        if cur:
            out.append(cur)
        return out

    # ------------------------------------------------------------------
    # precompute
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        if self._checkpoint is None:
            raise PrecomputedBackendError(
                "no model/tokenizer supplied and no checkpoint to load one "
                "from; precompute cannot run")
        if self._manifest_raw is None:
            # RealBackend runs this sealed check before it loads the model;
            # an injected backend must not be the way it gets skipped.
            raise PrecomputedBackendError(
                "loading the sealed checkpoint requires manifest_raw so that "
                "assert_runtime_versions() runs exactly as RealBackend runs "
                "it; refusing to load without the sealed runtime check")
        import pathlib

        from run_o1_v2_orchestrator import assert_runtime_versions  # sealed
        assert_runtime_versions(self._manifest_raw)
        from run_o1_v2_generation import _load_model  # sealed
        self._model, self._tokenizer = _load_model(pathlib.Path(self._checkpoint))
        return self._model, self._tokenizer

    def precompute(self) -> dict:
        """Generate every enumerated row through the batched engine."""
        gen = self._generate_fn or _default_generate_batch()
        model, tokenizer = self._ensure_model()
        kwargs: dict = {"compaction": False}
        if self._max_new_tokens is not None:
            kwargs["max_new_tokens"] = int(self._max_new_tokens)
        plan = self.batches()
        t0 = time.monotonic()
        for b, idxs in enumerate(plan):
            reqs = [self.requests[i] for i in idxs]
            results = gen(
                model, tokenizer,
                [r.task for r in reqs],
                [int(r.seed) for r in reqs],
                [None if r.direction is None
                 else np.asarray(r.direction, dtype=np.float64) for r in reqs],
                [float(r.signed_alpha) for r in reqs],
                **kwargs)
            if len(results) != len(reqs):
                raise PrecomputedBackendError(
                    f"batch {b}: engine returned {len(results)} results for "
                    f"{len(reqs)} rows")
            for i, res in zip(idxs, results):
                if self._results[i] is not None:
                    raise PrecomputedBackendError(
                        f"batch {b}: request {i} generated twice")
                self._results[i] = res
            self.batches_run += 1
            if self._progress_cb is not None:
                self._progress_cb({
                    "batches_done": self.batches_run,
                    "batches_total": len(plan),
                    "rows_done": self.n_precomputed(),
                    "rows_total": len(self.requests),
                    "elapsed_seconds": round(time.monotonic() - t0, 3),
                })
        self.precompute_seconds += time.monotonic() - t0
        missing = [i for i, r in enumerate(self._results) if r is None]
        if missing:
            raise PrecomputedBackendError(
                f"{len(missing)} enumerated rows were not generated "
                f"(first: {self.requests[missing[0]].describe()})")
        if self._verify_baseline_boundaries:
            self._check_baseline_boundaries()
        return self.coverage_report()

    def _boundary(self, result: dict) -> np.ndarray:
        b = np.asarray(result["_prefill_l4_47"])
        if b.ndim == 2 and b.shape[0] == 1:
            b = b[0]
        if b.shape != (HIDDEN,):
            raise PrecomputedBackendError(
                f"prefill boundary shape {b.shape} != ({HIDDEN},)")
        return b

    def _check_baseline_boundaries(self) -> None:
        """Fail here, not 20 minutes into the orchestrator's own O4 check.

        ``_run_baseline_bank`` refuses a task whose BANK_SIZE baseline
        boundaries are not bitwise equal.  This is the same condition, tested
        as soon as the rows exist; it weakens nothing and only moves the
        failure to where the GPU time can still be saved.
        """
        pos = 0
        for group in self.groups:
            if group[0].arm == "baseline":
                ref = self._boundary(self._results[pos])
                for off in range(1, len(group)):
                    got = self._boundary(self._results[pos + off])
                    if not np.array_equal(got, ref):
                        raise PrecomputedBackendError(
                            f"task {group[0].task['task_id']}: baseline "
                            f"prefill boundary differs between stream "
                            f"{group[0].stream_index} and stream "
                            f"{group[off].stream_index}; the sealed "
                            f"orchestrator would refuse this bank (O4)")
            pos += len(group)

    # ------------------------------------------------------------------
    # the backend interface the sealed orchestrator calls
    # ------------------------------------------------------------------

    def generate(self, task: dict, seed: int, direction, signed_alpha: float) -> dict:
        task_id = task.get("task_id") if isinstance(task, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise PrecomputedBackendError(
                f"generate() called with a task carrying no task_id: {task!r}")
        key = request_key(task_id, seed, direction, signed_alpha)
        idx = self._index.get(key)
        if idx is None:
            raise MissingPrecomputedRow(
                "the sealed orchestrator requested a row that was never "
                f"precomputed: key={key!r} "
                f"(task_id={task_id!r}, seed={seed!r}, "
                f"signed_alpha={signed_alpha!r}, "
                f"direction={'None' if direction is None else 'axis row'}). "
                "The enumeration disagrees with the sealed calibration loop; "
                "the run must stop rather than mix a serial row into a "
                "batched session.")
        req = self.requests[idx]
        want = req.task.get("task_content_sha256")
        got = task.get("task_content_sha256")
        if want is not None and got is not None and want != got:
            raise PrecomputedBackendError(
                f"task {task_id}: task_content_sha256 {got} does not match "
                f"the enumerated {want}; the precomputed row was generated "
                f"from a different prompt")
        result = self._results[idx]
        if result is None:
            raise PrecomputedBackendError(
                f"row was enumerated but never generated ({req.describe()}); "
                "precompute() must complete before the orchestrator runs")
        self._served[idx] += 1
        return result

    def decode(self, token_ids) -> str:
        """Sealed tokenizer decode, identical to ``RealBackend.decode``."""
        if self._tokenizer is None:
            raise PrecomputedBackendError(
                "decode() called before the sealed tokenizer was loaded")
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=True)

    # ------------------------------------------------------------------
    # construction from the sealed calibration inputs
    # ------------------------------------------------------------------

    @classmethod
    def from_manifest(cls, *, manifest_design_path: str,
                      task_manifest_path: str,
                      axis_tensor_path: str,
                      checkpoint: str | None = None,
                      **kwargs) -> "PrecomputedBatchedBackend":
        """Build from exactly the inputs the sealed orchestrator reads.

        ``grid`` and ``master_seed`` are taken from the same manifest keys the
        orchestrator uses, and the axis tensor goes through the sealed
        ``load_axis_tensor`` against the manifest's pinned hash, so the
        enumerated directions are byte-identical to the ones the orchestrator
        will hand back to ``generate()``.
        """
        with open(manifest_design_path) as fh:
            manifest_raw = json.load(fh)
        grid = [float(x) for x in
                manifest_raw["action_space"]["alpha_swept_in_calibration"]]
        master = int(manifest_raw["cohorts"]["master_seed"])
        axes = load_axis_tensor(
            axis_tensor_path,
            manifest_raw["artifact_hashes"]["structured_axis_tensor"],
            "structured")
        tasks = load_jsonl(task_manifest_path)
        kwargs.setdefault("manifest_raw", manifest_raw)
        return cls(tasks, axes, grid, master, checkpoint=checkpoint, **kwargs)
