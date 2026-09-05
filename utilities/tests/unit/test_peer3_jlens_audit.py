"""PEER3 regression tests for the two jlens defects that were fixed tonight and the one
that is still open. Pure CPU, no model, no artifacts required except where marked.

Written by PEER3 during the swarm audit of docs/jlens/RESULTS.md. Each test locks in a
specific bug that was actually present in an earlier draft, so a silent revert is caught.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ouro_jlens import analyze
from ouro_jlens import checkpoints
from ouro_jlens import fetch_wikitext
from ouro_jlens import probe as probe_module
from ouro_jlens import probe_cv as probe_cv_module
from ouro_jlens import recurrent
from ouro_jlens import validate
from ouro_jlens.probe import make_prompts
from ouro_jlens.probe_cv import (
    N_LAYER,
    N_FOLDS,
    N_UT,
    N_VAL_PAIRS,
    fold_assignment,
    prompt_folds,
    unordered_pairs,
)
from ouro_jlens.probe_report import build_report, cross_fitted_point


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


# --------------------------------------------------------------------------- #
# 4. Supervised layer selection must have no ancestry through the outer fold.
#    The old report chose a layer from classifiers scored on the other outer
#    folds even though those classifiers had been trained on the target fold.
# --------------------------------------------------------------------------- #

def test_supervised_layer_selection_uses_only_inner_validation_scores():
    folds = np.repeat(np.arange(N_FOLDS), 3)
    eligible = np.ones(len(folds), dtype=bool)
    ranks = np.ones((len(folds), N_UT * N_LAYER), dtype=np.int32)
    selection = np.zeros((N_FOLDS, N_UT * N_LAYER), dtype=float)
    for fold in range(N_FOLDS):
        for loop in range(N_UT):
            selection[fold, loop * N_LAYER + (fold + loop + 1) % N_LAYER] = 1.0

    _, choices_before = cross_fitted_point(
        ranks, folds, eligible, selection_accuracy=selection
    )
    # Make every outer-test prediction advertise a different layer.  Selection
    # must remain fixed because those predictions are evaluation data, not a
    # source of hyperparameter or layer choice.
    for fold in range(N_FOLDS):
        rows = folds == fold
        for loop in range(N_UT):
            ranks[rows, loop * N_LAYER + (fold + loop + 17) % N_LAYER] = 0
    _, choices_after = cross_fitted_point(
        ranks, folds, eligible, selection_accuracy=selection
    )
    assert choices_after == choices_before
    for loop, choices in enumerate(choices_after):
        assert choices == [(fold + loop + 1) % N_LAYER for fold in range(N_FOLDS)]


def test_probe_report_rejects_arrays_without_inner_selection_scores(tmp_path: Path):
    prompts = make_prompts()
    arrays = tmp_path / "arrays.npz"
    ranks = np.ones((len(prompts), N_UT * N_LAYER), dtype=np.int32)
    np.savez_compressed(
        arrays,
        labels=np.asarray([q["label"] for q in prompts]),
        folds=prompt_folds(prompts, seed=0),
        probe_rank=ranks,
        ll_cand=ranks,
        jl_cand=ranks,
    )
    with pytest.raises(ValueError, match="inner-validation layer scores"):
        build_report(arrays, seed=0, draws=1)


def test_standalone_probe_report_does_not_claim_validated_provenance(tmp_path: Path):
    prompts = make_prompts()
    arrays = tmp_path / "arrays.npz"
    ranks = np.ones((len(prompts), N_UT * N_LAYER), dtype=np.int32)
    np.savez_compressed(
        arrays,
        labels=np.asarray([q["label"] for q in prompts]),
        folds=prompt_folds(prompts, seed=0),
        probe_rank=ranks,
        ll_cand=ranks,
        jl_cand=ranks,
        selection_accuracy=np.zeros((N_FOLDS, N_UT * N_LAYER)),
        chosen_C=np.full((N_FOLDS, N_UT * N_LAYER), 0.01),
    )
    report = build_report(arrays, seed=0, draws=1)
    assert report["status"] == "DERIVED_FROM_SUPPLIED_ARRAYS_PROVENANCE_NOT_VALIDATED"
    assert report["input_provenance_status"] == "NOT_VALIDATED_BY_STANDALONE_REPORT_GENERATOR"


# --------------------------------------------------------------------------- #
# 5. GPU-score resume is fail-closed.  Existence is never provenance.
# --------------------------------------------------------------------------- #

def test_probe_cli_rejects_existing_score_without_valid_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    out = tmp_path / "out"
    out.mkdir()
    np.savez_compressed(out / "lens_all648.npz", bogus=np.asarray([1]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_cv.py",
            "--cache", str(tmp_path / "missing-cache.npz"),
            "--lens", str(tmp_path / "missing-lens.pt"),
            "--out", str(out),
            "--lens-only",
        ],
    )
    with pytest.raises(ValueError, match="provenance is missing"):
        probe_cv_module.main()


def test_direct_probe_cli_rejects_existing_cache_without_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    out = tmp_path / "out"
    out.mkdir()
    np.savez_compressed(out / "gpu_cache.npz", bogus=np.asarray([1]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe.py",
            "--lens", str(tmp_path / "missing-lens.pt"),
            "--out", str(out),
        ],
    )
    with pytest.raises(ValueError, match="cache provenance is missing"):
        probe_module.main()


def test_direct_probe_rejects_symlinked_output_root(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="traverses a symlink"):
        probe_module._reject_writable_symlinks(linked)


def test_cache_rebuild_invalidates_dependent_scores_before_gpu_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    out = tmp_path / "out"
    out.mkdir()
    score = out / "lens_all648.npz"
    provenance = out / "lens_all648.provenance.json"
    score.write_bytes(b"old score")
    provenance.write_text("{}")

    class RebuildObserved(RuntimeError):
        pass

    def observe_rebuild(cache: Path) -> None:
        assert not score.exists()
        assert not provenance.exists()
        raise RebuildObserved

    monkeypatch.setattr(probe_cv_module, "rebuild_gpu_cache", observe_rebuild)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_cv.py",
            "--cache", str(tmp_path / "cache.npz"),
            "--lens", str(tmp_path / "lens.pt"),
            "--out", str(out),
            "--rebuild-gpu-cache",
            "--lens-only",
        ],
    )
    with pytest.raises(RebuildObserved):
        probe_cv_module.main()


def test_wikitext_resume_requires_pinned_revision_and_exact_bytes(tmp_path: Path):
    out = tmp_path / "prompts.json"
    provenance = out.with_suffix(".provenance.json")
    prompts = ["a" * 600, "b" * 601]
    from ouro_jlens.evidence import atomic_write_json

    atomic_write_json(out, prompts, indent=None)
    atomic_write_json(provenance, {
        "schema_version": 1,
        "status": "FRESH_PINNED_DATASET_REVISION",
        "source": {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-103-raw-v1",
            "split": "train",
            "revision": fetch_wikitext.WIKITEXT_REVISION,
            "minimum_characters": 600,
            "requested_prompts": 2,
        },
        "datasets_version": fetch_wikitext.importlib.metadata.version("datasets"),
        "generator": fetch_wikitext._record(
            Path(fetch_wikitext.__file__).resolve(), "src/ouro_jlens/fetch_wikitext.py"
        ),
        "output": fetch_wikitext._record(out, "wikitext_prompts"),
    })
    fetch_wikitext._validate_existing(
        out,
        provenance,
        n=2,
        min_chars=600,
        revision=fetch_wikitext.WIKITEXT_REVISION,
    )
    out.write_text("[]")
    with pytest.raises(ValueError, match="byte identity mismatch"):
        fetch_wikitext._validate_existing(
            out,
            provenance,
            n=2,
            min_chars=600,
            revision=fetch_wikitext.WIKITEXT_REVISION,
        )


def test_alternate_checkpoint_identity_is_not_silently_labeled_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    revision = "a" * 40
    snapshot = tmp_path / revision

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            return (path, kwargs)

    class LoadedModel:
        def to(self, device: str):
            self.device = device
            return self

    class AutoModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            return LoadedModel()

    monkeypatch.setattr(recurrent.transformers, "AutoTokenizer", AutoTokenizer)
    monkeypatch.setattr(recurrent.transformers, "AutoModelForCausalLM", AutoModel)
    monkeypatch.setattr(
        recurrent,
        "OuroLensModel",
        lambda model, tokenizer: SimpleNamespace(model=model, tokenizer=tokenizer),
    )
    loaded = recurrent.load_ouro(snapshot, device="cpu")
    assert loaded.snapshot_path == snapshot.absolute()
    assert loaded.model_revision == revision


def test_recurrent_index_validation_survives_optimized_python():
    model = recurrent.OuroLensModel.__new__(recurrent.OuroLensModel)
    model.n_ut, model.n_physical = 4, 48
    assert model.index(3, 47) == 191
    for ut, layer in ((-1, 0), (4, 0), (0, -1), (0, 48), (True, 0)):
        with pytest.raises(ValueError):
            model.index(ut, layer)


def test_model_snapshot_identity_includes_remote_code_and_tokenizer_inputs(tmp_path: Path):
    required = {
        "config.json": b"config",
        "model.safetensors": b"weights",
        "modeling_ouro.py": b"model code",
        "tokenizer.json": b"tokenizer",
    }
    for name, data in required.items():
        (tmp_path / name).write_bytes(data)
    (tmp_path / "configuration_ouro.py").write_bytes(b"configuration code")
    (tmp_path / "tokenizer_config.json").write_bytes(b"tokenizer config")
    assert [path.relative_to(tmp_path).as_posix() for path in recurrent.model_snapshot_files(tmp_path)] == [
        "config.json",
        "configuration_ouro.py",
        "model.safetensors",
        "modeling_ouro.py",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    (tmp_path / "linked-dir").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="link"):
        recurrent.model_snapshot_files(tmp_path)


def test_checkpoint_identity_requires_and_binds_all_weight_shards(tmp_path: Path):
    for name in ("config.json", "modeling_ouro.py", "tokenizer.json"):
        (tmp_path / name).write_text(name)
    (tmp_path / "configuration_ouro.py").write_text("remote config code")
    (tmp_path / "model-00001.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002.safetensors").write_bytes(b"two")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-00001.safetensors","b":"model-00002.safetensors"}}'
    )
    records = checkpoints._checkpoint_files(tmp_path)
    assert [record["path"] for record in records] == [
        "checkpoint/config.json",
        "checkpoint/configuration_ouro.py",
        "checkpoint/model-00001.safetensors",
        "checkpoint/model-00002.safetensors",
        "checkpoint/model.safetensors.index.json",
        "checkpoint/modeling_ouro.py",
        "checkpoint/tokenizer.json",
    ]
    (tmp_path / "model-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="missing required"):
        checkpoints._checkpoint_files(tmp_path)


def test_validator_model_provenance_binds_checkpoint_bytes(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"weights-v1")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "modeling_ouro.py").write_text("model")
    (tmp_path / "tokenizer.json").write_text("tokenizer")
    model = SimpleNamespace(snapshot_path=tmp_path)
    first = validate._model_byte_provenance(model)
    assert first["status"] == "HASH_BOUND"
    assert {record["path"] for record in first["files"]} == {
        "model_snapshot/config.json",
        "model_snapshot/model.safetensors",
        "model_snapshot/modeling_ouro.py",
        "model_snapshot/tokenizer.json",
    }
    (tmp_path / "model.safetensors").write_bytes(b"weights-v2")
    second = validate._model_byte_provenance(model)
    assert second["aggregate_sha256"] != first["aggregate_sha256"]


def test_numerical_milestones_do_not_pass_with_incomplete_provenance(
    monkeypatch: pytest.MonkeyPatch
):
    model = SimpleNamespace(n_layers=192, d_model=2048, n_physical=48, n_ut=4)
    for name in (
        "m1_noninterference",
        "m2_exit_equality",
        "m3_recurrent_identity",
        "m4_distinct_vjps",
        "m5_stock_consistency",
    ):
        monkeypatch.setattr(validate, name, lambda *args: {"pass": True})
    monkeypatch.setattr(
        validate,
        "derive_milestone_passes",
        lambda report: {
            "m1_noninterference": True,
            "m2_exit_equality": True,
            "m3_recurrent_identity": True,
            "m4_distinct_vjps": True,
            "m5_stock_consistency": True,
        },
    )
    monkeypatch.setattr(validate, "bit_exact_rollup", lambda report: ([], True))
    monkeypatch.setattr(validate, "runtime_provenance", lambda model: {"status": "PROVENANCE_INCOMPLETE"})
    report = validate.run_validation(model=model, ids=torch.tensor([[1]]))
    assert report["numerical_pass"] is True
    assert report["pass"] is False
    assert report["status"] == "FAILED_OR_INCOMPLETE_VALIDATION"
