"""Adversarial CPU-only checks for JLens evidence and validator helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ouro_jlens import evidence, recurrent, validate


def _write(kind: str, path: Path) -> None:
    if kind == "bytes":
        evidence.atomic_write_bytes(path, b"payload")
    elif kind == "json":
        evidence.atomic_write_json(path, {"payload": True})
    elif kind == "npz":
        evidence.atomic_savez(path, values=np.arange(3))
    elif kind == "torch":
        evidence.atomic_torch_save({"payload": torch.arange(3)}, path)
    else:  # pragma: no cover - test-only dispatch guard
        raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["bytes", "json", "npz", "torch"])
def test_atomic_writers_reject_symlink_leaf_and_parent(
    tmp_path: Path, kind: str
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"sentinel")
    leaf = tmp_path / f"linked-{kind}"
    leaf.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        _write(kind, leaf)
    assert target.read_bytes() == b"sentinel"

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / f"linked-parent-{kind}"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    destination = linked_parent / "nested" / f"result-{kind}"
    with pytest.raises(ValueError, match="symlink"):
        _write(kind, destination)
    assert not (real_parent / "nested").exists()


@pytest.mark.parametrize("kind", ["bytes", "json", "npz", "torch"])
def test_atomic_writers_reject_swapped_temp_path_without_following_it(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / f"result-{kind}"
    sentinel = tmp_path / f"sentinel-{kind}"
    sentinel.write_bytes(b"must remain untouched")
    captured: dict[str, Path] = {}

    original_temporary_path = evidence._temporary_path

    def capture_temporary(path: Path, suffix: str = ".tmp") -> tuple[int, Path]:
        fd, temporary = original_temporary_path(path, suffix)
        captured["temporary"] = temporary
        return fd, temporary

    monkeypatch.setattr(evidence, "_temporary_path", capture_temporary)
    original_replace = evidence._replace_complete

    def swap_before_replace(fd: int, temporary: Path, target: Path) -> None:
        assert temporary == captured["temporary"]
        temporary.unlink()
        temporary.symlink_to(sentinel)
        original_replace(fd, temporary, target)

    monkeypatch.setattr(evidence, "_replace_complete", swap_before_replace)
    with pytest.raises(OSError, match="temporary evidence path"):
        _write(kind, destination)

    assert not destination.exists()
    assert sentinel.read_bytes() == b"must remain untouched"
    # Cleanup must not unlink a pathname that no longer names our inode.
    assert captured["temporary"].is_symlink()


def test_split_rejects_bool_non_int_and_out_of_range_and_roundtrips() -> None:
    model = recurrent.OuroLensModel.__new__(recurrent.OuroLensModel)
    model.n_ut = 3
    model.n_physical = 2
    model.n_layers = model.n_ut * model.n_physical

    for virtual in (True, False, 1.0, "1", -1, model.n_layers):
        with pytest.raises(ValueError, match="virtual layer"):
            model.split(virtual)
    for virtual in range(model.n_layers):
        ut, layer = model.split(virtual)
        assert model.index(ut, layer) == virtual


def test_causal_gradient_check_aggregates_every_recurrent_source() -> None:
    # The first source is causal, while the second has a forbidden dependency.
    # A first-element-only implementation would incorrectly pass this case.
    gradients = [
        torch.zeros(1, 3, 2),
        torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]]),
    ]
    norms, passed = validate._causal_later_position_gradient_check(gradients)
    assert norms == [0.0, 1.0]
    assert passed is False
