"""Optional frozen multi-tap and Ouro loop-representation backends.

The default compact agent uses :class:`models.GridFeatureBackend`.  This module
preserves the stronger frozen-backbone capability behind a small protocol:
extract named spatial/token taps, compress them with a deterministic connector,
and keep the backbone outside the controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .contracts import (
    ObjectState,
    Observation,
    Representation,
    RepresentationBackend,
)
from .memory import object_summary


@runtime_checkable
class TapExtractor(Protocol):
    def __call__(
        self,
        observation: Observation,
        objects: Sequence[ObjectState],
    ) -> (
        Mapping[str, np.ndarray]
        | tuple[Mapping[str, np.ndarray], tuple[int, int] | None]
    ):
        """Return named frozen taps and an optional patch-grid shape."""


def _fold(values: np.ndarray, size: int) -> np.ndarray:
    flat = np.nan_to_num(
        np.asarray(values, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    result = np.zeros(max(1, int(size)), dtype=np.float32)
    if flat.size == 0:
        return result
    for index, value in enumerate(flat):
        result[index % result.size] += float(value)
    counts = np.bincount(
        np.arange(flat.size) % result.size,
        minlength=result.size,
    )
    result /= np.maximum(counts, 1)
    return result


def _pool_spatial(
    spatial: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    array = np.nan_to_num(
        np.asarray(spatial, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if array.ndim == 3:
        array = np.linalg.norm(array, axis=-1)
    if array.ndim != 2:
        return np.zeros(shape, dtype=np.float32)
    rows = np.array_split(np.arange(array.shape[0]), shape[0])
    cols = np.array_split(np.arange(array.shape[1]), shape[1])
    result = np.zeros(shape, dtype=np.float32)
    for row_index, row_ids in enumerate(rows):
        for col_index, col_ids in enumerate(cols):
            if row_ids.size and col_ids.size:
                result[row_index, col_index] = float(
                    np.mean(array[np.ix_(row_ids, col_ids)])
                )
    scale = max(float(np.max(np.abs(result))), 1e-6)
    return result / scale


@dataclass(slots=True)
class TapConnector:
    """Tiny deterministic early/middle/late connector."""

    latent_dim: int = 32
    spatial_shape: tuple[int, int] = (8, 8)

    def connect(
        self,
        taps: Mapping[str, np.ndarray],
        *,
        patch_grid_size: tuple[int, int] | None = None,
        objects: Sequence[ObjectState] = (),
    ) -> Representation:
        if not taps:
            raise ValueError("at least one frozen tap is required")
        names = tuple(sorted(str(name) for name in taps))
        compressed: dict[str, np.ndarray] = {}
        global_parts: list[np.ndarray] = []
        spatial_by_name: dict[str, np.ndarray] = {}
        for name in names:
            array = np.asarray(taps[name], dtype=np.float32)
            if array.ndim >= 3 and array.shape[0] == 1:
                array = array[0]
            if array.ndim == 2:
                # Token sequence: preserve both CLS-like first token and global
                # distribution summaries.
                summary = np.concatenate(
                    [
                        array[0] if array.shape[0] else np.zeros(1),
                        np.mean(array, axis=0) if array.size else np.zeros(1),
                        np.std(array, axis=0) if array.size else np.zeros(1),
                    ]
                )
                if (
                    patch_grid_size is not None
                    and array.shape[0] - 1
                    == patch_grid_size[0] * patch_grid_size[1]
                ):
                    spatial_by_name[name] = array[1:].reshape(
                        patch_grid_size[0],
                        patch_grid_size[1],
                        array.shape[-1],
                    )
            elif array.ndim == 3:
                summary = np.concatenate(
                    [np.mean(array, axis=(0, 1)), np.std(array, axis=(0, 1))]
                )
                spatial_by_name[name] = array
            else:
                summary = array.reshape(-1)
            compressed[name] = _fold(summary, self.latent_dim)
            global_parts.append(compressed[name])

        object_part = _fold(object_summary(objects), self.latent_dim)
        global_vector = np.mean(
            np.stack(global_parts + [object_part]),
            axis=0,
        )
        # The spatial route is the controller-facing map, so the deepest
        # available loop state must own it.  Lexicographic iteration orders
        # ``late`` before ``middle`` and previously left the middle map here.
        # Preserve deterministic fallback behavior for custom tap names.
        spatial_name = next(
            (
                name
                for name in ("late", "middle", "early")
                if name in spatial_by_name
            ),
            max(spatial_by_name, default=None),
        )
        latest_spatial = (
            spatial_by_name[spatial_name] if spatial_name is not None else None
        )
        spatial = (
            _pool_spatial(latest_spatial, self.spatial_shape)
            if latest_spatial is not None
            else np.zeros(self.spatial_shape, dtype=np.float32)
        )
        return Representation(
            global_vector=global_vector,
            spatial=spatial,
            taps=compressed,
        )


class FrozenTapRepresentationBackend(RepresentationBackend):
    """Representation backend around any frozen named-tap extractor."""

    # "Frozen" describes parameter training, not runtime mutation.  An
    # arbitrary extractor may still maintain caches, counters, recurrent
    # state, or hooks, so failed observe transactions must isolate it with a
    # deepcopy instead of assuming encode is read-only.
    transactionally_stateful = True

    def __init__(
        self,
        extractor: TapExtractor,
        *,
        connector: TapConnector | None = None,
    ) -> None:
        if not callable(extractor):
            raise TypeError("extractor must be callable")
        self.extractor = extractor
        self.connector = connector or TapConnector()

    def encode(
        self,
        observation: Observation,
        objects: Sequence[ObjectState],
    ) -> Representation:
        result = self.extractor(observation, objects)
        if isinstance(result, tuple):
            taps, patch_grid = result
        else:
            taps, patch_grid = result, None
        return self.connector.connect(
            taps,
            patch_grid_size=patch_grid,
            objects=objects,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "latent_dim": self.connector.latent_dim,
            "spatial_shape": list(self.connector.spatial_shape),
            "extractor": type(self.extractor).__name__,
        }


class OuroLoopRepresentationBackend(RepresentationBackend):
    """Frozen GridEncoder + Ouro loop-state backend.

    ``encoder`` must expose ``encode_for_ouro(grid)``.  ``ouro`` may be the
    causal-LM wrapper or its inner model; the forward result must expose the
    historical Ouro tuple ``(base_output, loop_states, gate_states)`` or a
    ``hidden_states`` attribute.  Loading policy stays outside this class so
    callers retain control of local checkpoints, devices, and precision.
    """

    transactionally_stateless = True

    def __init__(
        self,
        encoder: Any,
        ouro: Any,
        *,
        connector: TapConnector | None = None,
        device: str | None = None,
    ) -> None:
        self.encoder = encoder
        self.ouro = ouro
        self.connector = connector or TapConnector()
        self.device = device
        for module in (encoder, ouro):
            eval_fn = getattr(module, "eval", None)
            if callable(eval_fn):
                eval_fn()
            parameters = getattr(module, "parameters", None)
            if callable(parameters):
                for parameter in parameters():
                    parameter.requires_grad = False

    @staticmethod
    def _loop_states(output: Any, fallback: Any) -> list[Any]:
        if isinstance(output, (tuple, list)) and len(output) > 1:
            candidate = output[1]
            if isinstance(candidate, (tuple, list)) and candidate:
                return list(candidate)
        candidate = getattr(output, "hidden_states", None)
        if isinstance(candidate, (tuple, list)) and candidate:
            return list(candidate)
        return [fallback]

    def encode(
        self,
        observation: Observation,
        objects: Sequence[ObjectState],
    ) -> Representation:
        import torch

        grid = torch.from_numpy(
            np.asarray(observation.frame, dtype=np.int64)
        ).unsqueeze(0)
        if self.device is not None:
            grid = grid.to(self.device)
        with torch.no_grad():
            tokens, mask, patch_grid = self.encoder.encode_for_ouro(grid)
            model = getattr(self.ouro, "model", self.ouro)
            kwargs = {
                "inputs_embeds": tokens,
                "attention_mask": mask,
                "use_cache": False,
            }
            try:
                output = model(**kwargs)
            except TypeError:
                kwargs.pop("attention_mask", None)
                output = model(**kwargs)
            loop_states = self._loop_states(output, tokens)
        indices = sorted(
            {
                0,
                max(0, (len(loop_states) - 1) // 2),
                len(loop_states) - 1,
            }
        )
        labels = ("early", "middle", "late")
        taps: dict[str, np.ndarray] = {}
        for label, index in zip(labels[-len(indices) :], indices):
            tensor = loop_states[index]
            taps[label] = tensor.detach().float().cpu().numpy()
        return self.connector.connect(
            taps,
            patch_grid_size=tuple(int(v) for v in patch_grid),
            objects=objects,
        )

    def state_dict(self) -> dict[str, Any]:
        config = getattr(self.ouro, "config", None)
        return {
            "type": type(self).__name__,
            "latent_dim": self.connector.latent_dim,
            "spatial_shape": list(self.connector.spatial_shape),
            "ouro_type": type(self.ouro).__name__,
            "encoder_type": type(self.encoder).__name__,
            "total_ut_steps": getattr(config, "total_ut_steps", None),
        }


__all__ = [
    "FrozenTapRepresentationBackend",
    "OuroLoopRepresentationBackend",
    "TapConnector",
    "TapExtractor",
]
