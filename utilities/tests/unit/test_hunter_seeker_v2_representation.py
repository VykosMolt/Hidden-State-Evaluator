from __future__ import annotations

import copy

import numpy as np
import torch

from hunter_seeker_v2.contracts import Observation
from hunter_seeker_v2.representation import (
    FrozenTapRepresentationBackend,
    OuroLoopRepresentationBackend,
    TapConnector,
)


def _observation() -> Observation:
    return Observation(
        frame=np.arange(20, dtype=np.uint8).reshape(4, 5) % 8,
        available_actions=(0, 1),
        task_id="tap-test",
    )


def test_frozen_named_taps_are_compacted_to_finite_spatial_representation() -> None:
    def extractor(observation, objects):
        del observation, objects
        return {
            "early": np.arange(20, dtype=np.float32).reshape(5, 4),
            "middle": np.ones((5, 4), dtype=np.float32),
            "late": np.full((5, 4), 2.0, dtype=np.float32),
        }, (2, 2)

    backend = FrozenTapRepresentationBackend(
        extractor,
        connector=TapConnector(latent_dim=12, spatial_shape=(3, 4)),
    )
    representation = backend.encode(_observation(), ())

    assert representation.global_vector.shape == (12,)
    assert representation.spatial.shape == (3, 4)
    assert set(representation.taps) == {"early", "middle", "late"}
    assert all(value.shape == (12,) for value in representation.taps.values())
    assert np.isfinite(representation.global_vector).all()
    assert np.isfinite(representation.spatial).all()


def test_frozen_wrapper_treats_arbitrary_extractor_as_transactionally_stateful() -> None:
    class CountingExtractor:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, observation, objects):
            del observation, objects
            self.calls += 1
            return {"late": np.ones((2, 2, 1), dtype=np.float32)}

    backend = FrozenTapRepresentationBackend(CountingExtractor())
    staged = copy.deepcopy(backend)
    staged.encode(_observation(), ())

    assert backend.transactionally_stateful is True
    assert not bool(getattr(backend, "transactionally_stateless", False))
    assert backend.extractor.calls == 0
    assert staged.extractor.calls == 1


def test_connector_uses_late_loop_state_for_controller_spatial_map() -> None:
    taps = {
        "early": np.asarray(
            [[[1.0], [0.0]], [[0.0], [0.0]]],
            dtype=np.float32,
        ),
        "middle": np.asarray(
            [[[0.0], [2.0]], [[0.0], [0.0]]],
            dtype=np.float32,
        ),
        "late": np.asarray(
            [[[0.0], [0.0]], [[0.0], [3.0]]],
            dtype=np.float32,
        ),
    }

    representation = TapConnector(
        latent_dim=4,
        spatial_shape=(2, 2),
    ).connect(taps)

    assert representation.spatial.tolist() == [[0.0, 0.0], [0.0, 1.0]]


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def encode_for_ouro(self, grid):
        batch = grid.shape[0]
        tokens = torch.arange(
            batch * 5 * 4,
            dtype=torch.float32,
            device=grid.device,
        ).reshape(batch, 5, 4)
        mask = torch.ones(batch, 5, device=grid.device)
        return tokens * self.scale, mask, (2, 2)


class _FakeOuroInner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.ones(()))

    def forward(self, *, inputs_embeds, attention_mask, use_cache):
        assert attention_mask.shape[:2] == inputs_embeds.shape[:2]
        assert use_cache is False
        return (
            None,
            [
                inputs_embeds,
                inputs_embeds + self.bias,
                inputs_embeds + 2.0 * self.bias,
            ],
            [],
        )


class _FakeOuro:
    def __init__(self) -> None:
        self.model = _FakeOuroInner()
        self.config = type("Config", (), {"total_ut_steps": 3})()

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()


def test_ouro_loop_backend_reads_all_loop_loci_without_training_backbone() -> None:
    encoder = _FakeEncoder()
    ouro = _FakeOuro()
    backend = OuroLoopRepresentationBackend(
        encoder,
        ouro,
        connector=TapConnector(latent_dim=10, spatial_shape=(2, 3)),
        device="cpu",
    )

    representation = backend.encode(_observation(), ())

    assert set(representation.taps) == {"early", "middle", "late"}
    assert representation.global_vector.shape == (10,)
    assert representation.spatial.shape == (2, 3)
    assert encoder.scale.requires_grad is False
    assert ouro.model.bias.requires_grad is False
    state = backend.state_dict()
    assert state["total_ut_steps"] == 3
