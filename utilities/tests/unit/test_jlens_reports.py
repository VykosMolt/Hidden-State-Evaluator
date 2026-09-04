from __future__ import annotations

import numpy as np
import pytest
import torch

from ouro_jlens import lens_convergence
from ouro_jlens.fit_lens import IntegrityError
from ouro_jlens.probe import make_prompts
from ouro_jlens.probe_cv import prompt_folds
from ouro_jlens.probe_report import cross_fitted_point, design
from ouro_jlens.analyze import boundary_prefix_match
from ouro_jlens.transport_report import estimate_from_squared_norms, summarize
from ouro_jlens.report import build_report


def _probe_design_arrays():
    prompts = make_prompts()
    return {
        "labels": np.asarray([q["label"] for q in prompts]),
        "folds": prompt_folds(prompts, 0),
    }


def test_boundary_prefix_match_rejects_alphanumeric_extensions():
    assert boundary_prefix_match(" Atlantic Ocean", "Atlantic")
    assert boundary_prefix_match(" day.", "day")
    assert not boundary_prefix_match("11", "1")
    assert not boundary_prefix_match("daytime", "day")


def test_probe_design_exposes_the_exact_label_coverage_population():
    d = design(_probe_design_arrays(), seed=0)
    labels, eligible = d["labels"], d["eligible"]
    assert len(labels) == 648
    assert int(eligible.sum()) == 576
    assert len(d["all_clusters"]) == 45
    assert len(d["eligible_clusters"]) == 39
    assert len(d["excluded_clusters"]) == 6
    assert np.bincount(labels).max() / len(labels) == 1 / 9
    assert np.bincount(labels[eligible]).max() / eligible.sum() == 1 / 8


def test_cross_fitted_layer_selection_never_scores_its_selection_fold():
    arrays = _probe_design_arrays()
    d = design(arrays, seed=0)
    ranks = np.ones((648, 192), dtype=np.int32)
    # Each held-out fold has a different perfect layer. Selection on the other
    # folds must not simply recover that fold's private layer.
    for fold in range(5):
        rows = d["folds"] == fold
        ranks[rows, fold] = 0
    values, choices = cross_fitted_point(ranks, d["folds"], d["eligible"])
    assert values[0] < 1.0
    assert all(0 <= layer < 48 for layer in choices[0])
    assert values.shape == (4,)


def test_transport_regression_reports_negative_moments_instead_of_hiding_them():
    n = np.asarray([8, 32, 56, 80])
    mu2 = np.full(96, 0.25)
    sigma2 = np.full(96, 0.5)
    sigma2[-1] = -0.1
    y = np.stack([mu2 + sigma2 / size for size in n])
    fitted = estimate_from_squared_norms(n, y)
    assert np.allclose(fitted["mu_squared"], mu2)
    assert np.allclose(fitted["sigma_squared"], sigma2)
    report = summarize(n, y)
    assert report["negative_sigma_squared_total"] == 1
    assert report["per_loop"][1]["valid_moment_locations"] == 47
    assert report["per_loop"][1]["negative_total_squared"] == 0
    assert report["per_loop"][1]["legacy_zero_clipped_sigma_mean"] != report["per_loop"][1]["sigma_rms_scatter_mean_valid"]


def test_report_reads_current_validator_schema(monkeypatch, tmp_path):
    main = {
        "input": {},
        "estimand": "test estimand",
        "populations": {
            name: {
                "n_raw_items": 1, "n_clean_items": 1, "n_clean_slots": 1,
                "paired_j_minus_logit": {
                    "excess_pass10_per_loop": [0, 0, 0, 0],
                    "ci95_unadjusted": [[0, 0]] * 4,
                },
            }
            for name in ("multihop", "order_ops_numeric")
        },
        "lens_free_exits": {},
    }
    monkeypatch.setattr("ouro_jlens.report.main_readout", lambda path: (main, None))
    monkeypatch.setattr("ouro_jlens.report.local_eventual", lambda path: {"rows": []})
    validation = tmp_path / "validation.json"
    validation.write_text('{"numerical_pass": true}')
    result = build_report(tmp_path, tmp_path, validation=validation, checkpoints=None)
    assert result["claims"]["instrumentation"]["status"] == "SUPPORTED_CURRENT_VALIDATION"
    required = {"status", "population", "estimand", "value", "uncertainty",
                "input_hashes", "generator_sha256"}
    assert all(required <= record.keys() for record in result["claims"].values())


def test_convergence_requires_hash_valid_disjoint_prompt_slices(monkeypatch, tmp_path):
    class Lens:
        def __init__(self, n):
            self.n_prompts = n
            self.source_layers = [0]
            self.d_model = 2
            self.jacobians = {0: torch.eye(2)}

    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    lenses = {str(first): Lens(2), str(second): Lens(3)}
    monkeypatch.setattr(lens_convergence.JacobianLens, "load", lambda path: lenses[path])
    metadata = {
        str(first): {"kind": "fit", "prompt_file_sha256": "f" * 64,
                     "prompt_slice_sha256": "a" * 64, "start": 0, "end": 2},
        str(second): {"kind": "fit", "prompt_file_sha256": "f" * 64,
                      "prompt_slice_sha256": "b" * 64, "start": 1, "end": 4},
    }
    for path in (first, second):
        path.with_suffix(".json").write_text('{"kind":"fit"}')
    monkeypatch.setattr(lens_convergence, "validate_sidecar",
                        lambda path, kind: metadata[str(path)])
    with pytest.raises(IntegrityError, match="overlap"):
        lens_convergence.build(str(first), str(second))

    metadata[str(second)].update({"start": 2, "end": 5})
    report = lens_convergence.build(str(first), str(second))
    assert report["status"] == "INCONCLUSIVE_TWO_POINT_DIAGNOSTIC"
    assert report["disjoint_prompt_evidence"]["slice_a"]["end"] == 2
    assert report["disjoint_prompt_evidence"]["slice_b"]["start"] == 2
