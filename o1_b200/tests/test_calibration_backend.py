"""Precomputed batched calibration backend: enumeration fidelity against a
replay of the SEALED calibration loop, key canonicalisation, and the
no-silent-fallback contract."""
from __future__ import annotations

import hashlib

import numpy as np

from _h import CORPUS_DIR, FAST_SUBSET, Runner

from o1_b200.runner import calibration_backend as cb
from o1_b200.runner.calibration_backend import (
    MissingPrecomputedRow,
    PrecomputedBackendError,
    PrecomputedBatchedBackend,
)

BANK_SIZE = cb.BANK_SIZE
SIGNED_ACTIONS = cb.SIGNED_ACTIONS
action_index = cb.action_index
derive_stream_seed = cb.derive_stream_seed

MASTER = 20260824
GRID = [0.25, 0.5, 1.0]


def _tasks(n: int = 5) -> list[dict]:
    return [{"task_id": f"cal-{i:03d}",
             "task_content_sha256": f"{i:064x}",
             "generator_setting_id": "unit_v1",
             "task_prompt": f"prompt {i}",
             "options": ["True", "False", "Unknown"]}
            for i in range(n)]


def _axes() -> np.ndarray:
    rng = np.random.default_rng(11)
    a = rng.standard_normal((cb.N_DIRECTIONS, cb.HIDDEN))
    return a / np.sqrt((a * a).mean(axis=1, keepdims=True))


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def decode(self, ids, skip_special_tokens=False):
        self.calls.append((list(ids), skip_special_tokens))
        return "|".join(str(int(i)) for i in ids)


def _fake_engine(*, perturb_baseline: tuple[str, int] | None = None,
                 record: list | None = None):
    """A stand-in for engine_batched.generate_batch (no torch, no GPU).

    Prefill is sampling-free in the sealed generator, so the boundary here
    depends only on the task and the intervention, never on the seed.
    """
    seen_baseline: dict[str, int] = {}

    def gen(model, tokenizer, tasks, seeds, directions, signed_alphas,
            *, compaction=False, max_new_tokens=None):
        assert compaction is False, "compaction must never be enabled"
        assert len(tasks) == len(seeds) == len(directions) == len(signed_alphas)
        if record is not None:
            record.append(len(tasks))
        out = []
        for task, seed, direction, alpha in zip(tasks, seeds, directions,
                                                signed_alphas):
            tid = task["task_id"]
            # deterministic across processes (Python's hash() is salted)
            base = float(int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16) % 1000)
            boundary = np.full(cb.HIDDEN, base, dtype=np.float32)
            if direction is not None:
                boundary = boundary + np.asarray(
                    direction, dtype=np.float32) * np.float32(alpha)
            elif perturb_baseline is not None:
                n = seen_baseline.get(tid, 0)
                seen_baseline[tid] = n + 1
                if (tid, n) == perturb_baseline:
                    boundary = boundary + np.float32(1.0)
            out.append({
                "generated_token_ids": [int(seed) % 1000, 7, 9],
                "generated_text": f"{tid}/{int(seed)}/{float(alpha)!r}",
                "well_formed": True,
                "verifier_correct": True,
                "parsed_answer": "True",
                "finish_reason": "parseable_final_answer",
                "logits_finite": True,
                "catastrophic_repetition": False,
                "injected_rms": abs(float(alpha)),
                "realized_delta_rms": abs(float(alpha)),
                "_prefill_l4_47": boundary,
                "_prompt_token_ids": [1, 2, 3],
                # test-only provenance so a served row can be traced back
                "_probe": (tid, int(seed), float(alpha),
                           None if direction is None
                           else cb.direction_digest(direction)),
            })
        return out

    return gen


def _backend(tasks=None, grid=None, *, batch_size=32, engine=None,
             precompute=True, **kw) -> PrecomputedBatchedBackend:
    tok = FakeTokenizer()
    b = PrecomputedBatchedBackend(
        tasks if tasks is not None else _tasks(),
        _axes(), grid if grid is not None else GRID, MASTER,
        batch_size=batch_size, model=object(), tokenizer=tok,
        generate_fn=engine or _fake_engine(), **kw)
    if precompute:
        b.precompute()
    return b


def _replay_orchestrator(backend, tasks, axes, grid, master,
                         *, include_resume_path=True) -> list:
    """A faithful, independent replay of the SEALED calibration loop.

    Transcribed from run_o1_v2_orchestrator.orchestrate_calibration (the
    ``for g, task in enumerate(tasks)`` loop) and ``_run_baseline_bank``.
    Every call goes through the real ``backend.generate``; a request the
    enumeration missed therefore raises instead of being quietly absorbed.
    """
    calls = []
    for g, task in enumerate(tasks):
        rank = g % BANK_SIZE
        seeds = {i: derive_stream_seed(master, task["task_id"], i)
                 for i in range(BANK_SIZE)}
        for i in range(BANK_SIZE):
            calls.append(backend.generate(task, seeds[i], None, 0.0))
        if include_resume_path:
            # orchestrator line ~430: resume replays baseline stream 0
            calls.append(backend.generate(task, seeds[0], None, 0.0))
        for alpha in grid:
            for d, s in SIGNED_ACTIONS:
                stream = (action_index(d, s) + rank) % BANK_SIZE
                calls.append(backend.generate(
                    task, seeds[stream], axes[d - 1], s * alpha))
    return calls


def run() -> Runner:
    r = Runner("calibration_backend")
    tasks = _tasks()
    axes = _axes()
    n_expected = len(tasks) * BANK_SIZE * (1 + len(GRID))

    def enumeration_shape():
        b = _backend(precompute=False)
        assert len(b) == n_expected, (len(b), n_expected)
        assert len(b.requests) == n_expected
        assert len(set(b.keys())) == n_expected, "enumerated keys are not unique"
        arms = {}
        for req in b.requests:
            arms[req.arm] = arms.get(req.arm, 0) + 1
        assert arms == {"baseline": len(tasks) * BANK_SIZE,
                        "structured": len(tasks) * BANK_SIZE * len(GRID)}, arms
    r.check(f"enumeration is len(tasks)*{BANK_SIZE}*(1+len(grid)) requests "
            f"with unique keys", enumeration_shape)

    def enumeration_matches_sealed_replay():
        b = _backend()
        calls = _replay_orchestrator(b, tasks, axes, GRID, MASTER)
        # every sealed request was served (no MissingPrecomputedRow raised)
        assert len(calls) == n_expected + len(tasks)   # + resume replays
        # ...and every enumerated row was consumed: set equality both ways
        assert b.unserved_indices() == [], \
            f"{len(b.unserved_indices())} enumerated rows the sealed loop " \
            f"never asked for"
        rep = b.coverage_report()
        assert rep["requests_served_at_least_once"] == n_expected
        assert rep["generate_calls"] == n_expected + len(tasks)
        # each served row really is the row that was asked for
        for g, task in enumerate(tasks):
            rank = g % BANK_SIZE
            seeds = {i: derive_stream_seed(MASTER, task["task_id"], i)
                     for i in range(BANK_SIZE)}
            for alpha in GRID:
                for d, s in SIGNED_ACTIONS:
                    stream = (action_index(d, s) + rank) % BANK_SIZE
                    got = b.generate(task, seeds[stream], axes[d - 1], s * alpha)
                    assert got["_probe"] == (
                        task["task_id"], int(seeds[stream]), float(s * alpha),
                        cb.direction_digest(axes[d - 1])), got["_probe"]
    r.check("enumerated set equals a faithful replay of the sealed "
            "calibration loop (both inclusions), resume path included",
            enumeration_matches_sealed_replay)

    def unknown_keys_raise():
        b = _backend()
        seeds = {i: derive_stream_seed(MASTER, tasks[0]["task_id"], i)
                 for i in range(BANK_SIZE)}
        cases = {
            "unknown task": ({"task_id": "cal-999",
                              "task_content_sha256": "0" * 64},
                             seeds[0], None, 0.0),
            "unknown seed": (tasks[0], 123456789, None, 0.0),
            "alpha off the swept grid": (tasks[0], seeds[0], axes[0], 0.75),
            "baseline asked with an alpha": (tasks[0], seeds[0], None, 0.25),
            "direction that is not a sealed axis row":
                (tasks[0], seeds[0], np.ones(cb.HIDDEN, dtype=np.float64), 0.25),
        }
        for label, (t, s, d, a) in cases.items():
            try:
                b.generate(t, s, d, a)
            except MissingPrecomputedRow as exc:
                assert "key=" in str(exc), label
            else:
                raise AssertionError(f"{label}: no raise — silent fallback")
    r.check("HOSTILE: any un-enumerated request RAISES MissingPrecomputedRow "
            "naming the key (never a silent serial fallback or default row)",
            unknown_keys_raise)

    def wrong_task_content_raises():
        b = _backend()
        seeds = {i: derive_stream_seed(MASTER, tasks[0]["task_id"], i)
                 for i in range(BANK_SIZE)}
        forged = dict(tasks[0])
        forged["task_content_sha256"] = "f" * 64
        try:
            b.generate(forged, seeds[0], None, 0.0)
        except PrecomputedBackendError as exc:
            assert "task_content_sha256" in str(exc)
        else:
            raise AssertionError("a different prompt was served silently")
    r.check("HOSTILE: same task_id with different task_content_sha256 is "
            "refused, not served from the precomputed row",
            wrong_task_content_raises)

    def serving_before_precompute_raises():
        b = _backend(precompute=False)
        seeds = derive_stream_seed(MASTER, tasks[0]["task_id"], 0)
        try:
            b.generate(tasks[0], seeds, None, 0.0)
        except PrecomputedBackendError as exc:
            assert "precompute()" in str(exc)
        else:
            raise AssertionError("an ungenerated row was served")
    r.check("a planned but not-yet-generated row raises rather than "
            "returning a placeholder", serving_before_precompute_raises)

    def signed_zero_and_seed_forms():
        assert cb.canonical_alpha(-0.0) == cb.canonical_alpha(0.0)
        assert cb.request_key("t", 3, None, -0.0) == \
            cb.request_key("t", 3.0, None, 0.0)
        assert cb.request_key("t", np.int64(3), None, 0.0) == \
            cb.request_key("t", 3, None, 0.0)
        b = _backend(grid=[0.0, 0.5])
        seeds = {i: derive_stream_seed(MASTER, tasks[0]["task_id"], i)
                 for i in range(BANK_SIZE)}
        # s * alpha with alpha == 0.0 and s == -1 produces -0.0
        for d, s in SIGNED_ACTIONS:
            stream = (action_index(d, s) + 0) % BANK_SIZE
            b.generate(tasks[0], seeds[stream], axes[d - 1], s * 0.0)
        # int and numpy-int spellings of the same seed all hit
        s0 = seeds[0]
        for spelling in (int(s0), np.int64(s0), np.uint64(s0)):
            b.generate(tasks[0], spelling, None, -0.0)
        # small seeds: the float spelling is exact and must agree
        assert cb.request_key("t", 12345, None, 0.0) == \
            cb.request_key("t", 12345.0, None, 0.0)
    r.check("+0.0/-0.0 and int/numpy-int/exact-float seed spellings never miss",
            signed_zero_and_seed_forms)

    def wide_float_seed_is_refused_not_truncated():
        # derive_stream_seed returns a full-width uint64; float64 cannot
        # represent every such integer, so float(seed) lands on a NEIGHBOUR.
        s0 = derive_stream_seed(MASTER, tasks[0]["task_id"], 0)
        assert s0 >= 2 ** 53 and int(float(s0)) != s0, s0
        b = _backend()
        try:
            b.generate(tasks[0], float(s0), None, 0.0)
        except PrecomputedBackendError as exc:
            assert "2**53" in str(exc), str(exc)
        else:
            raise AssertionError(
                "a lossy float spelling of a 64-bit seed was accepted")
    r.check("HOSTILE: a >=2**53 seed passed as a float is refused with a "
            "precision error, never silently truncated to a neighbour",
            wide_float_seed_is_refused_not_truncated)

    def direction_keyed_on_content():
        b = _backend()
        seeds = {i: derive_stream_seed(MASTER, tasks[0]["task_id"], i)
                 for i in range(BANK_SIZE)}
        d, s = SIGNED_ACTIONS[2]
        stream = (action_index(d, s) + 0) % BANK_SIZE
        copy = np.array(axes[d - 1], dtype=np.float64, copy=True)
        assert copy is not axes[d - 1]
        got = b.generate(tasks[0], seeds[stream], copy, s * GRID[0])
        assert got["_probe"][3] == cb.direction_digest(axes[d - 1])
        # a one-ULP perturbation is a DIFFERENT direction and must miss
        moved = np.array(copy)
        moved[0] = np.nextafter(moved[0], np.inf)
        try:
            b.generate(tasks[0], seeds[stream], moved, s * GRID[0])
        except MissingPrecomputedRow:
            pass
        else:
            raise AssertionError("a perturbed axis row was served")
    r.check("direction is keyed on array CONTENT (a copy hits, a one-ULP "
            "perturbation misses), never on array identity",
            direction_keyed_on_content)

    def decode_matches_realbackend():
        tok = FakeTokenizer()
        b = PrecomputedBatchedBackend(tasks, axes, GRID, MASTER,
                                      model=object(), tokenizer=tok,
                                      generate_fn=_fake_engine())
        out = b.decode([5, 6, 7])
        assert out == "5|6|7"
        assert tok.calls == [([5, 6, 7], True)], tok.calls
        # generators may hand over any iterable of ids
        b.decode(np.array([1, 2], dtype=np.int64))
        assert tok.calls[-1] == ([1, 2], True), tok.calls[-1]
        assert b.tokenizer is tok
    r.check("decode() delegates to the sealed tokenizer with "
            "skip_special_tokens=True, exactly like RealBackend.decode",
            decode_matches_realbackend)

    def batching_plan():
        for bs in (BANK_SIZE, 16, 24, 32, 64, 1000):
            b = _backend(batch_size=bs, precompute=False)
            plan = b.batches()
            flat = [i for chunk in plan for i in chunk]
            assert flat == list(range(len(b))), bs
            assert all(len(c) <= bs for c in plan), (bs, [len(c) for c in plan])
            # no BANK_SIZE group is ever split across two batches
            assert all(len(c) % BANK_SIZE == 0 for c in plan), bs
        rec: list = []
        b = _backend(batch_size=24, engine=_fake_engine(record=rec))
        assert sum(rec) == len(b) and max(rec) <= 24, rec
        assert b.coverage_report()["batches_run"] == len(rec)
    r.check("batch_size is honoured, every request is generated exactly once, "
            "and a baseline bank is never split across batches", batching_plan)

    def small_batch_refused():
        try:
            _backend(batch_size=BANK_SIZE - 1, precompute=False)
        except PrecomputedBackendError as exc:
            assert "baseline" in str(exc)
        else:
            raise AssertionError("a bank-splitting batch size was accepted")
    r.check(f"batch_size < {BANK_SIZE} is refused (the baseline bank must "
            f"share one batch shape)", small_batch_refused)

    def baseline_boundary_disagreement_refused():
        try:
            _backend(engine=_fake_engine(perturb_baseline=("cal-002", 3)))
        except PrecomputedBackendError as exc:
            assert "cal-002" in str(exc) and "boundary" in str(exc)
        else:
            raise AssertionError(
                "baseline boundaries that disagree across streams were "
                "accepted; the sealed O4 check would fail hours later")
    r.check("HOSTILE: baseline prefill boundaries that differ across the bank "
            "are refused at precompute (same condition as sealed O4)",
            baseline_boundary_disagreement_refused)

    def duplicate_enumeration_refused():
        try:
            _backend(grid=[0.5, 0.5], precompute=False)
        except PrecomputedBackendError as exc:
            assert "not unique" in str(exc)
        else:
            raise AssertionError("a colliding enumeration was accepted")
    r.check("a duplicated alpha in the swept grid is caught by the key "
            "uniqueness assert at construction", duplicate_enumeration_refused)

    def bad_inputs_refused():
        tok = FakeTokenizer()
        for label, kw in (
            ("axis tensor of the wrong shape",
             {"axes": np.zeros((3, cb.HIDDEN))}),
            ("non-finite axis tensor",
             {"axes": np.full((cb.N_DIRECTIONS, cb.HIDDEN), np.nan)}),
        ):
            try:
                PrecomputedBatchedBackend(tasks, kw["axes"], GRID, MASTER,
                                          model=object(), tokenizer=tok)
            except PrecomputedBackendError:
                pass
            else:
                raise AssertionError(f"{label} was accepted")
        for label, args in (
            ("empty task list", ([], _axes(), GRID, MASTER)),
            ("empty alpha grid", (tasks, _axes(), [], MASTER)),
            ("duplicate task_id", ([tasks[0], dict(tasks[0])], _axes(),
                                   GRID, MASTER)),
        ):
            try:
                PrecomputedBatchedBackend(*args, model=object(), tokenizer=tok)
            except PrecomputedBackendError:
                pass
            else:
                raise AssertionError(f"{label} was accepted")
    r.check("degenerate enumeration inputs (axis shape, non-finite axes, "
            "empty tasks/grid, duplicate task_id) are refused", bad_inputs_refused)

    def checkpoint_load_requires_sealed_runtime_check():
        b = PrecomputedBatchedBackend(tasks, axes, GRID, MASTER,
                                      checkpoint="/nonexistent/ckpt",
                                      generate_fn=_fake_engine())
        try:
            b.precompute()
        except PrecomputedBackendError as exc:
            assert "assert_runtime_versions" in str(exc)
        else:
            raise AssertionError(
                "the sealed runtime-version check was skipped on the "
                "checkpoint-loading path")
    r.check("loading the sealed checkpoint without manifest_raw is refused: "
            "assert_runtime_versions() cannot be skipped via injection",
            checkpoint_load_requires_sealed_runtime_check)

    def default_engine_is_the_gated_one():
        from o1_b200.runner.engine_batched import generate_batch
        assert cb._default_generate_batch() is generate_batch
    r.check("the default generation path is engine_batched.generate_batch "
            "(the equivalence-gated engine), not the injection hook",
            default_engine_is_the_gated_one)

    # ------------------------------------------------------------------
    # end-to-end through the REAL batched engine (synthetic runtime, CPU)
    # ------------------------------------------------------------------

    def end_to_end_on_the_real_engine():
        from o1_b200.runner.engine_batched import generate_batch
        from o1_b200.runner.runbuild import build_validation_bundle
        from o1_b200.runner.synthetic_runtime import SyntheticOuroModel
        from run_o1_v2_orchestrator import _score_fields  # sealed

        artifact = {"kind": "synthetic", "device": "cpu", "seed_tag": 0}
        bundle = build_validation_bundle(CORPUS_DIR, artifact,
                                         task_subset=FAST_SUBSET[:2])
        ctasks = sorted(bundle.tasks_by_id.values(), key=lambda t: t["task_id"])
        caxes = np.asarray(bundle.axes["structured"], dtype=np.float64)
        model = SyntheticOuroModel(device="cpu")
        grid = [0.1]
        # batch_size 16 forces more than one batch over the 32 rows
        b = PrecomputedBatchedBackend(ctasks, caxes, grid, 4242,
                                      batch_size=16, model=model,
                                      tokenizer=model.tokenizer)
        b.precompute()
        assert b.coverage_report()["batches_run"] == 2

        rows = _replay_orchestrator(b, ctasks, caxes, grid, 4242)
        assert b.unserved_indices() == []
        for row, task in zip(rows, [ctasks[0]] * 9 + [ctasks[1]] * 9):
            # the sealed scorer must accept every served row unchanged
            _score_fields(task, row)
            assert b.decode(row["generated_token_ids"]) == row["generated_text"]

        # the served row must be bitwise what a SINGLE-row batch produces for
        # the same (task, seed, direction, alpha) — i.e. the precompute
        # mapping puts every result back on its own request
        for g, task in enumerate(ctasks):
            for d, sgn in (SIGNED_ACTIONS[0], SIGNED_ACTIONS[5]):
                stream = (action_index(d, sgn) + g % BANK_SIZE) % BANK_SIZE
                seed = derive_stream_seed(4242, task["task_id"], stream)
                served = b.generate(task, seed, caxes[d - 1], sgn * grid[0])
                alone = generate_batch(model, model.tokenizer, [task], [seed],
                                       [caxes[d - 1]], [sgn * grid[0]])[0]
                assert served["generated_token_ids"] == \
                    alone["generated_token_ids"], (task["task_id"], d, sgn)
                assert served["finish_reason"] == alone["finish_reason"]
                assert np.array_equal(served["_prefill_l4_47"],
                                      alone["_prefill_l4_47"])
                assert served["injected_rms"] == alone["injected_rms"]
    r.check("end to end on the real batched engine (synthetic runtime): the "
            "sealed scorer accepts every served row, decode round-trips, and "
            "each served row is bitwise the single-row result for its own "
            "request", end_to_end_on_the_real_engine)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
