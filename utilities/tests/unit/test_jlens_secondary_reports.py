"""Fail-closed checks for the JLens convergence and transport diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ouro_jlens import fitsize_report, lens_convergence, transport_report
from ouro_jlens.fit_lens import IntegrityError


class _Lens:
    def __init__(self, n: int, *, values: tuple[float, ...] = (1.0, 2.0)) -> None:
        self.n_prompts = n
        self.source_layers = list(range(len(values)))
        self.d_model = 2
        self.jacobians = {
            index: torch.eye(2) * value for index, value in enumerate(values)
        }


def _meta(n: int, *, start: int = 0, revision: str = "model-a") -> dict:
    return {
        "start": start,
        "end": start + n,
        "n_prompts": n,
        "n_fitted": n,
        "d_model": 2,
        "prompt_file_sha256": "a" * 64,
        "prompt_slice_sha256": f"{start:064x}",
        "model_revision": revision,
    }


def test_two_point_convergence_rejects_binary_sidecar_count_disagreement(monkeypatch):
    entries = iter([(_Lens(8), _meta(7)), (_Lens(16), _meta(16, start=20))])
    monkeypatch.setattr(lens_convergence, "_validated_fit", lambda _path: next(entries))
    with pytest.raises(IntegrityError, match="binary count disagrees"):
        lens_convergence.build("a.pt", "b.pt")


def test_two_point_convergence_rejects_mixed_model_or_fitting_identity(monkeypatch):
    entries = iter([
        (_Lens(8), _meta(8)),
        (_Lens(16), _meta(16, start=20, revision="model-b")),
    ])
    monkeypatch.setattr(lens_convergence, "_validated_fit", lambda _path: next(entries))
    with pytest.raises(IntegrityError, match="same model, source, and fitting contract"):
        lens_convergence.build("a.pt", "b.pt")


def test_two_point_convergence_records_binaries_sidecars_and_shared_identity(tmp_path, monkeypatch):
    path_a, path_b = tmp_path / "a.pt", tmp_path / "b.pt"
    for path in (path_a, path_b):
        path.write_bytes(b"lens")
        path.with_suffix(".json").write_text("{}")
    entries = iter([(_Lens(8), _meta(8)), (_Lens(16), _meta(16, start=20))])
    monkeypatch.setattr(lens_convergence, "_validated_fit", lambda _path: next(entries))
    report = lens_convergence.build(str(path_a), str(path_b))
    assert report["status"] == "INCONCLUSIVE_TWO_POINT_DIAGNOSTIC"
    assert set(report["inputs"][0]) == {"binary", "sidecar"}
    assert len(report["shared_identity_sha256"]) == 64


def test_transport_estimator_rejects_fractional_duplicate_and_nonfinite_inputs():
    norms = np.ones((3, 2))
    with pytest.raises(ValueError, match="exact one-dimensional integer"):
        transport_report.estimate_from_squared_norms(np.array([8.0, 16.0, 32.0]), norms)
    with pytest.raises(ValueError, match="distinct positive"):
        transport_report.estimate_from_squared_norms(np.array([8, 8, 32]), norms)
    bad = norms.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite and nonnegative"):
        transport_report.estimate_from_squared_norms(np.array([8, 16, 32]), bad)


def test_transport_report_rejects_declared_count_and_identity_mismatches(tmp_path, monkeypatch):
    paths = []
    for n in (8, 16, 32):
        path = tmp_path / f"lens-{n}.pt"
        path.write_bytes(b"lens")
        path.with_suffix(".json").write_text(json.dumps({"kind": "fit"}))
        paths.append(path)

    lenses = {str(path): _Lens(n) for path, n in zip(paths, (8, 16, 32))}
    metadata = {str(path): _meta(n) for path, n in zip(paths, (8, 16, 32))}
    monkeypatch.setattr("jlens.JacobianLens.load", lambda path: lenses[path])
    monkeypatch.setattr(transport_report, "validate_sidecar", lambda path, kind: metadata[str(path)])

    metadata[str(paths[0])]["n_fitted"] = 7
    with pytest.raises(ValueError, match="declared n=8"):
        transport_report.lens_squared_norms(list(zip((8, 16, 32), paths)))

    metadata[str(paths[0])]["n_fitted"] = 8
    metadata[str(paths[-1])]["model_revision"] = "model-b"
    with pytest.raises(IntegrityError, match="identity differs"):
        transport_report.lens_squared_norms(list(zip((8, 16, 32), paths)))

    metadata[str(paths[-1])]["model_revision"] = "model-a"
    lenses[str(paths[0])].d_model = 3
    with pytest.raises(IntegrityError, match="model width disagrees"):
        transport_report.lens_squared_norms(list(zip((8, 16, 32), paths)))

    lenses[str(paths[0])].d_model = 2
    lenses[str(paths[0])].jacobians[0][0, 0] = torch.nan
    with pytest.raises(IntegrityError, match="malformed or non-finite"):
        transport_report.lens_squared_norms(list(zip((8, 16, 32), paths)))


class _FitEval:
    def __init__(self, n: int, directory) -> None:
        self.items = [{"task": "multihop"}, {"task": "order-ops"}]
        ranks = np.full((2, 2, 192), 50, dtype=np.int32)
        ranks[:, 0] = max(1, 40 - n)
        self.arrays = {"jlens_exit3_allrank": ranks}
        self.slot_mask = {
            "multihop": np.asarray([[True, False, False], [False, False, False]]),
            "order-ops numeric": np.asarray([[False, False, False], [True, False, False]]),
        }
        shared_identity = {
            "target_ut": 3,
            "target_virtual": 191,
            "source_layers": list(range(191)),
            "model_revision": "model-a",
            "jlens_commit": "j" * 40,
            "prompt_file_sha256": "a" * 64,
            "prompt_provenance_sha256": "b" * 64,
            "prompt_source_revision": "c" * 40,
            "source_sha256": "d" * 64,
            "generator_sha256": "e" * 64,
        }
        slice_digest = f"{n:064x}"
        identity = {
            **shared_identity,
            "prompt_slice": {"start": 0, "end": n, "count": n, "sha256": slice_digest},
            "prompt_slice_sha256": slice_digest,
            "n_requested": n,
            "n_fitted": n,
            "n_prompts": n,
        }
        self.provenance = {
            "config": {
                "tasks": ["multihop", "order-ops"],
                "position": -1,
                "prompt_policy": "identical",
                "max_intermediates": 3,
                "max_names": 128,
                "greedy_steps": 4,
            },
            "inputs": {"evaluation_input_sha256": "f" * 64},
            "outputs": {
                "items": {"sha256": "1" * 64},
                "task_names": {"sha256": "2" * 64},
            },
            "item_metadata_sha256": "3" * 64,
            "item_count": 2,
            "correct_count": 2,
            "model_snapshot_sha256": "4" * 64,
            "model": {
                "revision": "model-a", "n_physical": 48, "n_ut": 4,
                "n_layers": 192, "d_model": 2,
            },
            "jlens_commit": "j" * 40,
            "source_sha256": "5" * 64,
            "lens_inputs": [{"target_ut": 3, "identity": identity}],
        }
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("arrays.npz", "items.json", "task_names.json", "provenance.json"):
            (directory / name).write_bytes(f"{n}:{name}".encode())

    def own_slots(self, mask):
        return (np.asarray([0]), np.asarray([0])) if mask[0, 0] else (
            np.asarray([1]), np.asarray([0])
        )

    def scores(self, _ranks, mask):
        item = 0 if mask[0, 0] else 1
        hit = np.ones((1, 192), dtype=float)
        control = np.zeros((1, 4), dtype=float)
        return {
            "hit": hit,
            "control_any": control,
            "excess": hit,
            "cand_top1": hit,
            "items": np.asarray([item]),
        }


def _fit_evaluations(tmp_path, sizes=(8, 32, 56, 80)):
    return {
        n: _FitEval(n, tmp_path / f"fitsize_n{n}") for n in sizes
    }


def test_fitsize_report_authenticates_labels_and_common_population(tmp_path, monkeypatch):
    evaluations = _fit_evaluations(tmp_path)
    monkeypatch.setattr(
        fitsize_report.analyze,
        "Eval",
        lambda path: evaluations[int(str(path).rsplit("fitsize_n", 1)[1])],
    )
    report = fitsize_report.build_report(tmp_path)
    assert report["evidence_status"] == "CURRENT_PROVENANCE_VERIFIED_NESTED_PREFIXES"
    assert report["inputs"]["80"]["provenance"]["path"] == "fitsize_n80/provenance.json"
    assert "n80_minus_n8_mean_log10_rank_improvement" in report["populations"]["multihop"]


def test_fitsize_report_rejects_false_size_label_and_population_drift(tmp_path, monkeypatch):
    evaluations = _fit_evaluations(tmp_path)
    monkeypatch.setattr(
        fitsize_report.analyze,
        "Eval",
        lambda path: evaluations[int(str(path).rsplit("fitsize_n", 1)[1])],
    )
    evaluations[32].provenance["lens_inputs"][0]["identity"]["n_fitted"] = 31
    with pytest.raises(ValueError, match="not authenticated"):
        fitsize_report.build_report(tmp_path)
    evaluations[32].provenance["lens_inputs"][0]["identity"]["n_fitted"] = 32
    evaluations[56].provenance["correct_count"] = 1
    with pytest.raises(ValueError, match="exact same population"):
        fitsize_report.build_report(tmp_path)
