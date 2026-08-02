"""Relational goal hypotheses as bounded potential functions.

Many puzzle environments display their goal as an on-screen relation: a
template region the working region must come to equal, a symmetry that must
be completed, two objects that must meet, or a categorical population that
must reach a terminal count.  This module proposes such relations, tracks how
actions change each relation's mismatch potential, and exposes bounded policy
and graph-search guidance.

The epistemic contract mirrors the taps/ego components:

* a hypothesis is a *proposal*, never a claim — unverified hypotheses carry
  only exploration-grade weight;
* promotion to verified normally requires an observed completion while the
  hypothesis potential is satisfied; when an immediate stage swap hides the
  solved frame, only an unambiguous direct-overlap reach relation reconstructed
  from supported controlled motion may use the ``completion_motion`` fallback;
* every influence on policy is one named bounded score term computed from
  observed mismatch deltas (or exact successor lookups), never from an
  assumed goal;
* ``enabled=False`` is an exact behavioral no-op.

Relation templates are domain-general — cellwise and lattice-canonical
equality, horizontal/vertical mirror equality, self-symmetry, interior
uniformity, object reach, and monotonic count/coverage targets.  Nothing
references any particular environment family.  Exogenous-masked cells (see
:mod:`exogenous`) are excluded from every potential so environment clocks
cannot masquerade as goal mismatch.
"""

from __future__ import annotations

import base64
import binascii
import copy
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    Action,
    HypothesisConfig,
    ObjectState,
    Observation,
    WorldSnapshot,
    readonly_array,
)


_BBox = tuple[int, int, int, int]
_GOAL_ORIGINS = frozenset({"goal_contrast", "completion_motion"})


def _region(frame: np.ndarray, bbox: _BBox) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return np.asarray(frame)[y0 : y1 + 1, x0 : x1 + 1]


def _bbox_valid(frame: np.ndarray, bbox: _BBox) -> bool:
    arr = np.asarray(frame)
    if arr.ndim != 2:
        return False
    x0, y0, x1, y1 = bbox
    return bool(
        0 <= x0 <= x1 < arr.shape[1]
        and 0 <= y0 <= y1 < arr.shape[0]
    )


def _region_mask(mask_cells: frozenset[tuple[int, int]], bbox: _BBox) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    out = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    for y, x in mask_cells:
        if y0 <= y <= y1 and x0 <= x <= x1:
            out[y - y0, x - x0] = True
    return out


def _mismatch(
    left: np.ndarray,
    right: np.ndarray,
    excluded: np.ndarray,
) -> float:
    if left.shape != right.shape or left.shape != excluded.shape:
        return 1.0
    valid = ~excluded
    total = int(valid.sum())
    if total <= 0:
        return 0.0
    return float((left != right)[valid].sum() / total)


def _mode_pool(region: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Block-modal downsampling to ``shape``; identity when shapes match."""

    arr = np.asarray(region)
    if arr.shape == shape:
        return arr
    rows = np.array_split(np.arange(arr.shape[0]), shape[0])
    cols = np.array_split(np.arange(arr.shape[1]), shape[1])
    out = np.zeros(shape, dtype=arr.dtype)
    for r, row_ids in enumerate(rows):
        for c, col_ids in enumerate(cols):
            block = arr[np.ix_(row_ids, col_ids)].reshape(-1)
            values, counts = np.unique(block, return_counts=True)
            out[r, c] = values[np.argmax(counts)]
    return out


def _mode_pool_masked(
    region: np.ndarray,
    excluded: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Modal pooling that excludes masked source cells.

    The returned boolean grid marks pooled blocks with no observable cells.
    """

    arr = np.asarray(region)
    mask = np.asarray(excluded, dtype=bool)
    if arr.shape != mask.shape:
        raise ValueError("canonical region and exclusion mask shapes differ")
    rows = np.array_split(np.arange(arr.shape[0]), shape[0])
    cols = np.array_split(np.arange(arr.shape[1]), shape[1])
    out = np.zeros(shape, dtype=arr.dtype)
    pooled_excluded = np.ones(shape, dtype=bool)
    for r, row_ids in enumerate(rows):
        for c, col_ids in enumerate(cols):
            block = arr[np.ix_(row_ids, col_ids)].reshape(-1)
            block_mask = mask[np.ix_(row_ids, col_ids)].reshape(-1)
            visible = block[~block_mask]
            if visible.size <= 0:
                continue
            values, counts = np.unique(visible, return_counts=True)
            out[r, c] = values[np.flatnonzero(counts == counts.max())[0]]
            pooled_excluded[r, c] = False
    return out, pooled_excluded


def _pool_purity(region: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Modal pooling plus the mean within-block modal purity."""

    arr = np.asarray(region)
    rows = np.array_split(np.arange(arr.shape[0]), shape[0])
    cols = np.array_split(np.arange(arr.shape[1]), shape[1])
    out = np.zeros(shape, dtype=arr.dtype)
    purity_sum = 0.0
    for r, row_ids in enumerate(rows):
        for c, col_ids in enumerate(cols):
            block = arr[np.ix_(row_ids, col_ids)].reshape(-1)
            values, counts = np.unique(block, return_counts=True)
            best = int(np.argmax(counts))
            out[r, c] = values[best]
            purity_sum += float(counts[best] / block.size)
    return out, purity_sum / (shape[0] * shape[1])


def _natural_lattice(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_k: int = 3,
    max_k: int = 6,
    purity_floor: float = 0.75,
) -> tuple[int, int] | None:
    """Coarsest square lattice at which both regions pool purely.

    Every block must span at least two pixels in both regions, so trivially
    pure single-pixel lattices cannot qualify.
    """

    limit = min(max_k, min(a.shape + b.shape) // 2)
    best: tuple[float, int] | None = None
    for k in range(min_k, limit + 1):
        _pooled_a, purity_a = _pool_purity(a, (k, k))
        _pooled_b, purity_b = _pool_purity(b, (k, k))
        purity = 0.5 * (purity_a + purity_b)
        if purity < purity_floor:
            continue
        if best is None or purity > best[0] + 1e-9:
            best = (purity, k)
    if best is None:
        return None
    return (best[1], best[1])


def _canonical(
    region: np.ndarray,
    excluded: np.ndarray | None = None,
) -> np.ndarray:
    """Relabel values by first occurrence, invariant to palette renaming.

    Frequency ranking is ambiguous when two colors have equal counts and its
    numeric tie-break leaks the original palette.  First-occurrence coding
    instead represents the equality partition of the grid, so any bijective
    palette renaming has exactly the same canonical form.  Excluded cells do
    not participate in the coding.
    """

    arr = np.asarray(region)
    mask = (
        np.zeros(arr.shape, dtype=bool)
        if excluded is None
        else np.asarray(excluded, dtype=bool)
    )
    if arr.shape != mask.shape:
        raise ValueError("canonical region and exclusion mask shapes differ")
    out = np.full(arr.shape, -1, dtype=np.int64)
    mapping: dict[int, int] = {}
    next_rank = 0
    for index, value in enumerate(arr.reshape(-1)):
        if mask.reshape(-1)[index]:
            continue
        key = int(value)
        if key not in mapping:
            mapping[key] = next_rank
            next_rank += 1
        out.reshape(-1)[index] = mapping[key]
    return out


def _reach_potential(
    frame: np.ndarray,
    value_a: int,
    value_b: int,
    mask_cells: frozenset[tuple[int, int]],
    cell_cap: int,
) -> float:
    """Object-relational potential: 0 when value_a and value_b cells touch.

    ``value_a`` and ``value_b`` name two colours (object identities).  The
    potential is the 8-connected gap between the nearest cell of each,
    normalized by a quarter of the frame perimeter and clipped to [0, 1]:
    adjacency (gap <= 1) reads as satisfied, distant objects as violated.
    Colours occupying more than ``cell_cap`` cells are treated as
    background-like and yield an unsatisfiable 1.0 rather than a spurious
    match.  Exogenous-masked cells are excluded so clocks cannot participate.
    """

    arr = np.asarray(frame)
    if arr.ndim != 2:
        return 1.0
    a = _visible_value_cells(arr, value_a, mask_cells)
    b = _visible_value_cells(arr, value_b, mask_cells)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 1.0
    # Apply the background-like cap to evidence that can actually
    # participate in the relation.  Masked clock/background cells must not
    # make a small visible object look too large to reason about.
    if a.shape[0] > int(cell_cap) or b.shape[0] > int(cell_cap):
        return 1.0
    gap = int(np.min(np.abs(a[:, None, :] - b[None, :, :]).max(axis=2)))
    height, width = arr.shape[0], arr.shape[1]
    norm = max(1.0, (height + width) / 4.0)
    return float(np.clip((gap - 1) / norm, 0.0, 1.0))


def _visible_value_cells(
    frame: np.ndarray,
    value: int,
    mask_cells: frozenset[tuple[int, int]],
) -> np.ndarray:
    """Visible ``(row, column)`` cells carrying ``value``."""

    cells = np.argwhere(np.asarray(frame) == int(value))
    if not mask_cells or cells.shape[0] == 0:
        return cells.astype(np.int64, copy=False).reshape(-1, 2)
    return np.asarray(
        [
            row
            for row in cells
            if (int(row[0]), int(row[1])) not in mask_cells
        ],
        dtype=np.int64,
    ).reshape(-1, 2)


def _count_target_potential(
    frame: np.ndarray,
    value: int,
    initial_count: int,
    target_count: int,
    mask_cells: frozenset[tuple[int, int]],
) -> float:
    """Normalized distance from the completion-grounded target cell count."""

    arr = np.asarray(frame)
    selected = arr == int(value)
    for y, x in mask_cells:
        if 0 <= y < selected.shape[0] and 0 <= x < selected.shape[1]:
            selected[y, x] = False
    span = abs(int(initial_count) - int(target_count))
    if span <= 0:
        return 0.0
    current = int(selected.sum())
    if int(target_count) < int(initial_count):
        remaining = max(0, current - int(target_count))
    else:
        remaining = max(0, int(target_count) - current)
    return float(np.clip(remaining / float(span), 0.0, 1.0))


@dataclass(slots=True)
class Hypothesis:
    hypothesis_id: str
    # equal | equal_canonical | mirror_h | mirror_v | self_mirror_h |
    # self_mirror_v | uniform | reach | count_at_most | count_at_least
    kind: str
    region_a: _BBox
    region_b: _BBox | None = None
    lattice: tuple[int, int] | None = None
    verified: bool = False
    refuted: bool = False
    initial_potential: float = 1.0
    # Object-relational kinds are parameterized by two colours, not regions.
    value_a: int | None = None
    value_b: int | None = None
    # "invariant" (already ~satisfied), "goal_contrast" (observed
    # violated->satisfied), or "completion_motion" (the narrow direct-overlap
    # reach proof used when an immediate stage swap hides the solved frame).
    origin: str = "invariant"
    reach_cell_cap: int = 400
    initial_count: int | None = None
    target_count: int | None = None
    # (action_index, target_signature) -> [delta_sum, count]
    deltas: dict[tuple[int, str], list[float]] = field(default_factory=dict)

    def observable(
        self,
        frame: np.ndarray,
        mask_cells: frozenset[tuple[int, int]] = frozenset(),
    ) -> bool:
        """Whether at least one valid comparison remains after masking.

        A zero potential with zero observable support means "unknown", not
        "satisfied".  Keeping observability separate preserves the bounded
        numeric potential API while preventing masked relations from being
        promoted as proven goals.
        """

        arr = np.asarray(frame)
        if arr.ndim != 2:
            return False
        if self.kind == "reach":
            if self.value_a is None or self.value_b is None:
                return False
            a = _visible_value_cells(arr, self.value_a, mask_cells)
            b = _visible_value_cells(arr, self.value_b, mask_cells)
            return bool(
                0 < a.shape[0] <= int(self.reach_cell_cap)
                and 0 < b.shape[0] <= int(self.reach_cell_cap)
            )
        if self.kind in {"count_at_most", "count_at_least"}:
            visible = np.ones(arr.shape, dtype=bool)
            for y, x in mask_cells:
                if 0 <= y < visible.shape[0] and 0 <= x < visible.shape[1]:
                    visible[y, x] = False
            return bool(visible.any())
        if not _bbox_valid(arr, self.region_a):
            return False
        if self.region_b is not None and not _bbox_valid(arr, self.region_b):
            return False
        a = _region(arr, self.region_a)
        excluded_a = _region_mask(mask_cells, self.region_a)
        if self.kind == "uniform":
            return bool((~excluded_a).any())
        if self.kind in {"self_mirror_h", "self_mirror_v"}:
            axis = 1 if self.kind == "self_mirror_h" else 0
            return bool(
                (~(excluded_a | np.flip(excluded_a, axis=axis))).any()
            )
        if self.region_b is None:
            return False
        b = _region(arr, self.region_b)
        excluded_b = _region_mask(mask_cells, self.region_b)
        if self.kind == "equal_canonical":
            shape = self.lattice or (
                min(a.shape[0], b.shape[0]),
                min(a.shape[1], b.shape[1]),
            )
            _pooled_a, pooled_excluded_a = _mode_pool_masked(
                a,
                excluded_a,
                shape,
            )
            _pooled_b, pooled_excluded_b = _mode_pool_masked(
                b,
                excluded_b,
                shape,
            )
            return bool((~(pooled_excluded_a | pooled_excluded_b)).any())
        if a.shape != b.shape:
            return False
        if self.kind == "mirror_h":
            excluded_b = np.flip(excluded_b, axis=1)
        elif self.kind == "mirror_v":
            excluded_b = np.flip(excluded_b, axis=0)
        return bool((~(excluded_a | excluded_b)).any())

    def potential(
        self,
        frame: np.ndarray,
        mask_cells: frozenset[tuple[int, int]] = frozenset(),
    ) -> float:
        if self.kind == "reach":
            assert self.value_a is not None and self.value_b is not None
            return _reach_potential(
                frame,
                self.value_a,
                self.value_b,
                mask_cells,
                self.reach_cell_cap,
            )
        if self.kind in {"count_at_most", "count_at_least"}:
            assert (
                self.value_a is not None
                and self.initial_count is not None
                and self.target_count is not None
            )
            return _count_target_potential(
                frame,
                self.value_a,
                self.initial_count,
                self.target_count,
                mask_cells,
            )
        if not _bbox_valid(frame, self.region_a):
            return 1.0
        if self.region_b is not None and not _bbox_valid(frame, self.region_b):
            return 1.0
        a = _region(frame, self.region_a)
        excluded_a = _region_mask(mask_cells, self.region_a)
        if self.kind == "uniform":
            valid = ~excluded_a
            values = a[valid]
            if values.size == 0:
                return 0.0
            counts = np.bincount(values.reshape(-1).astype(np.int64) - values.min())
            return float(1.0 - counts.max() / values.size)
        if self.kind in ("self_mirror_h", "self_mirror_v"):
            axis = 1 if self.kind == "self_mirror_h" else 0
            return _mismatch(a, np.flip(a, axis=axis), excluded_a | np.flip(excluded_a, axis=axis))
        assert self.region_b is not None
        b = _region(frame, self.region_b)
        excluded_b = _region_mask(mask_cells, self.region_b)
        if self.kind == "equal_canonical":
            # Scale- and palette-invariant comparison at the structural tile
            # lattice fixed at proposal time.  Exogenous cells are removed
            # before modal pooling so they cannot alter either the tile value
            # or the palette canonicalization.
            shape = self.lattice or (
                min(a.shape[0], b.shape[0]),
                min(a.shape[1], b.shape[1]),
            )
            pooled_a, pooled_excluded_a = _mode_pool_masked(a, excluded_a, shape)
            pooled_b, pooled_excluded_b = _mode_pool_masked(b, excluded_b, shape)
            excluded = pooled_excluded_a | pooled_excluded_b
            left = _canonical(pooled_a, excluded)
            right = _canonical(pooled_b, excluded)
            return _mismatch(left, right, excluded)
        if self.kind == "mirror_h":
            b = np.flip(b, axis=1)
            excluded_b = np.flip(excluded_b, axis=1)
        elif self.kind == "mirror_v":
            b = np.flip(b, axis=0)
            excluded_b = np.flip(excluded_b, axis=0)
        return _mismatch(a, b, excluded_a | excluded_b)

    def mismatch_cells(
        self,
        frame: np.ndarray,
        mask_cells: frozenset[tuple[int, int]] = frozenset(),
    ) -> list[tuple[int, int]]:
        """Absolute (x, y) cells that currently violate the relation.

        For two-region kinds the caller cannot know which region is the
        workspace, so mismatching cells of both regions are returned.
        """

        if self.kind == "reach":
            if self.value_a is None or self.value_b is None:
                return []
            arr = np.asarray(frame)
            if arr.ndim != 2:
                return []
            a = _visible_value_cells(arr, self.value_a, mask_cells)
            b = _visible_value_cells(arr, self.value_b, mask_cells)
            if (
                a.shape[0] == 0
                or b.shape[0] == 0
                or a.shape[0] > int(self.reach_cell_cap)
                or b.shape[0] > int(self.reach_cell_cap)
            ):
                return []
            gap = int(
                np.min(np.abs(a[:, None, :] - b[None, :, :]).max(axis=2))
            )
            if gap <= 1:
                return []
            return [
                (int(col), int(row))
                for row, col in np.concatenate((a, b), axis=0)
            ]
        if self.kind in {"count_at_most", "count_at_least"}:
            if (
                self.value_a is None
                or self.initial_count is None
                or self.target_count is None
            ):
                return []
            # For consumption goals, remaining cells of the consumed value
            # are concrete click/proposal targets.  Increasing coverage has no
            # equally safe generic target without a workspace relation.
            if int(self.target_count) >= int(self.initial_count):
                return []
            arr = np.asarray(frame)
            found = [
                (int(col), int(row))
                for row, col in np.argwhere(arr == int(self.value_a))
                if (int(row), int(col)) not in mask_cells
            ]
            if len(found) <= int(self.target_count):
                return []
            return found
        if not _bbox_valid(frame, self.region_a):
            return []
        if self.region_b is not None and not _bbox_valid(frame, self.region_b):
            return []
        a = _region(frame, self.region_a)
        excluded_a = _region_mask(mask_cells, self.region_a)
        ax0, ay0 = self.region_a[0], self.region_a[1]

        def _absolute(rows: np.ndarray, x0: int, y0: int) -> list[tuple[int, int]]:
            return [(int(x0 + c), int(y0 + r)) for r, c in rows]

        if self.kind == "uniform":
            valid = ~excluded_a
            values = a[valid]
            if values.size == 0:
                return []
            counts = np.bincount(values.reshape(-1).astype(np.int64) - values.min())
            modal = int(np.argmax(counts)) + int(values.min())
            return _absolute(np.argwhere((a != modal) & valid), ax0, ay0)
        if self.kind in ("self_mirror_h", "self_mirror_v"):
            axis = 1 if self.kind == "self_mirror_h" else 0
            diff = (a != np.flip(a, axis=axis)) & ~(
                excluded_a | np.flip(excluded_a, axis=axis)
            )
            return _absolute(np.argwhere(diff), ax0, ay0)
        assert self.region_b is not None
        b = _region(frame, self.region_b)
        excluded_b = _region_mask(mask_cells, self.region_b)
        bx0, by0 = self.region_b[0], self.region_b[1]
        if self.kind == "equal_canonical":
            shape = self.lattice or (
                min(a.shape[0], b.shape[0]),
                min(a.shape[1], b.shape[1]),
            )
            pooled_a, pooled_excluded_a = _mode_pool_masked(a, excluded_a, shape)
            pooled_b, pooled_excluded_b = _mode_pool_masked(b, excluded_b, shape)
            excluded = pooled_excluded_a | pooled_excluded_b
            diff = np.argwhere(
                (_canonical(pooled_a, excluded) != _canonical(pooled_b, excluded))
                & ~excluded
            )
            points: list[tuple[int, int]] = []
            for region, region_excluded, x0, y0 in (
                (a, excluded_a, ax0, ay0),
                (b, excluded_b, bx0, by0),
            ):
                rows = np.array_split(np.arange(region.shape[0]), shape[0])
                cols = np.array_split(np.arange(region.shape[1]), shape[1])
                for r, c in diff:
                    row_ids, col_ids = rows[int(r)], cols[int(c)]
                    visible = [
                        (int(col), int(row))
                        for row in row_ids
                        for col in col_ids
                        if not region_excluded[int(row), int(col)]
                    ]
                    if visible:
                        col, row = visible[len(visible) // 2]
                        points.append((int(x0 + col), int(y0 + row)))
            return points
        flipped = b
        excluded_flipped = excluded_b
        if self.kind == "mirror_h":
            flipped = np.flip(b, axis=1)
            excluded_flipped = np.flip(excluded_b, axis=1)
        elif self.kind == "mirror_v":
            flipped = np.flip(b, axis=0)
            excluded_flipped = np.flip(excluded_b, axis=0)
        if a.shape != flipped.shape:
            return []
        diff = (a != flipped) & ~(excluded_a | excluded_flipped)
        rows = np.argwhere(diff)
        points = _absolute(rows, ax0, ay0)
        if self.kind in {"equal", "mirror_h", "mirror_v"}:
            counterpart_rows = rows
            if self.kind == "mirror_h":
                counterpart_rows = rows.copy()
                counterpart_rows[:, 1] = b.shape[1] - 1 - rows[:, 1]
            elif self.kind == "mirror_v":
                counterpart_rows = rows.copy()
                counterpart_rows[:, 0] = b.shape[0] - 1 - rows[:, 0]
            points.extend(_absolute(counterpart_rows, bx0, by0))
        return points


@dataclass(frozen=True, slots=True)
class GoalEstimate:
    hypothesis_id: str
    kind: str
    current_potential: float


def _region_boxes(
    objects: Sequence[ObjectState],
    config: HypothesisConfig,
) -> list[_BBox]:
    boxes: list[_BBox] = []
    seen: set[_BBox] = set()
    for obj in sorted(objects, key=lambda o: -int(o.area)):
        bbox = tuple(int(v) for v in obj.bbox)
        x0, y0, x1, y1 = bbox
        if (x1 - x0 + 1) * (y1 - y0 + 1) < int(config.min_region_area):
            continue
        if bbox in seen:
            continue
        seen.add(bbox)
        boxes.append(bbox)
    return boxes[:12]


def _region_relation_candidates(
    frame: np.ndarray,
    objects: Sequence[ObjectState],
    config: HypothesisConfig,
) -> Iterable[Hypothesis]:
    """Every structural geometric relation, ungated (no potential set)."""

    arr = np.asarray(frame)
    boxes = _region_boxes(objects, config)
    for i, a in enumerate(boxes):
        aw, ah = a[2] - a[0] + 1, a[3] - a[1] + 1
        if aw >= 3 and ah >= 3:
            for kind in ("self_mirror_h", "self_mirror_v", "uniform"):
                yield Hypothesis(
                    hypothesis_id=f"{kind}|{a}|None", kind=kind, region_a=a
                )
        for b in boxes[i + 1 :]:
            bw, bh = b[2] - b[0] + 1, b[3] - b[1] + 1
            if (aw, ah) == (bw, bh):
                for kind in ("equal", "mirror_h", "mirror_v"):
                    yield Hypothesis(
                        hypothesis_id=f"{kind}|{a}|{b}",
                        kind=kind,
                        region_a=a,
                        region_b=b,
                    )
                continue
            # Scaled template pairs: comparable aspect ratio within a bounded
            # scale factor, compared palette- and scale-invariantly at the
            # pair's natural tile lattice.
            ratio_w = max(aw, bw) / max(1, min(aw, bw))
            ratio_h = max(ah, bh) / max(1, min(ah, bh))
            if (
                ratio_w <= 5.0
                and ratio_h <= 5.0
                and abs(ratio_w - ratio_h) <= 0.75
                and min(aw * ah, bw * bh) >= int(config.min_region_area)
            ):
                lattice = _natural_lattice(_region(arr, a), _region(arr, b))
                if lattice is not None:
                    yield Hypothesis(
                        hypothesis_id=f"equal_canonical|{a}|{b}",
                        kind="equal_canonical",
                        region_a=a,
                        region_b=b,
                        lattice=lattice,
                    )


def _reach_relation_candidates(
    objects: Sequence[ObjectState],
    config: HypothesisConfig,
) -> Iterable[Hypothesis]:
    """Object-relational ``reach`` relations over distinct object colours."""

    values: list[int] = []
    seen: set[int] = set()
    for obj in sorted(objects, key=lambda o: -int(o.area)):
        value = int(obj.value)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    values = values[:8]
    for i, value_a in enumerate(values):
        for value_b in values[i + 1 :]:
            yield Hypothesis(
                hypothesis_id=f"reach|{value_a}|{value_b}",
                kind="reach",
                region_a=(0, 0, 0, 0),
                value_a=value_a,
                value_b=value_b,
                reach_cell_cap=int(config.reach_cell_cap),
            )


def _visible_value_counts(
    frame: np.ndarray,
    mask_cells: frozenset[tuple[int, int]],
) -> dict[int, int]:
    arr = np.asarray(frame)
    visible = np.ones(arr.shape, dtype=bool)
    for y, x in mask_cells:
        if 0 <= y < visible.shape[0] and 0 <= x < visible.shape[1]:
            visible[y, x] = False
    values, counts = np.unique(arr[visible], return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def _count_target_candidates(
    initial_frame: np.ndarray,
    completion_frame: np.ndarray,
    config: HypothesisConfig,
    mask_cells: frozenset[tuple[int, int]],
    count_trace: Sequence[Mapping[int, int]] | None = None,
) -> Iterable[Hypothesis]:
    """Completion-grounded count/coverage relations.

    A stable modal value in both frames is treated as background.  Every
    remaining value whose count changes materially yields an exact target
    count: decreases model consumption/removal and increases model coverage
    or painting.  No target is proposed without a real completion contrast.
    """

    if not config.enabled or not config.enable_count_targets:
        return
    initial_counts = _visible_value_counts(initial_frame, mask_cells)
    completion_counts = _visible_value_counts(completion_frame, mask_cells)
    if not initial_counts or not completion_counts:
        return
    initial_mode = min(
        initial_counts,
        key=lambda value: (-initial_counts[value], value),
    )
    completion_mode = min(
        completion_counts,
        key=lambda value: (-completion_counts[value], value),
    )
    stable_background = (
        {initial_mode} if initial_mode == completion_mode else set()
    )
    ranked: list[tuple[float, int, Hypothesis]] = []
    for value in sorted(set(initial_counts) | set(completion_counts)):
        if value in stable_background:
            continue
        initial_count = int(initial_counts.get(value, 0))
        target_count = int(completion_counts.get(value, 0))
        extent = max(initial_count, target_count)
        delta = abs(initial_count - target_count)
        if (
            initial_count == target_count
            or extent < int(config.count_min_cells)
            or delta / max(1, extent)
            < float(config.count_min_change_fraction)
        ):
            continue
        trace = [
            int(row.get(value, 0))
            for row in (
                count_trace
                if count_trace is not None
                else (initial_counts, completion_counts)
            )
        ]
        if target_count < initial_count:
            monotone = all(right <= left for left, right in zip(trace, trace[1:]))
            kind = "count_at_most"
        else:
            monotone = all(right >= left for left, right in zip(trace, trace[1:]))
            kind = "count_at_least"
        if not monotone:
            continue
        hypothesis = Hypothesis(
            hypothesis_id=f"{kind}|{value}|{initial_count}|{target_count}",
            kind=kind,
            region_a=(0, 0, 0, 0),
            value_a=value,
            initial_count=initial_count,
            target_count=target_count,
        )
        ranked.append((delta / max(1, extent), delta, hypothesis))
    ranked.sort(
        key=lambda row: (
            -row[0],
            -row[1],
            row[2].hypothesis_id,
        )
    )
    for _fraction, _delta, hypothesis in ranked[
        : int(config.max_count_targets)
    ]:
        yield hypothesis


def propose_hypotheses(
    frame: np.ndarray,
    objects: Sequence[ObjectState],
    config: HypothesisConfig,
    *,
    mask_cells: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[Hypothesis, ...]:
    """Invariant proposals: relations already (nearly) satisfied on ``frame``.

    This path structurally cannot surface a goal — a relation the agent must
    still achieve is violated and so fails the ``max_initial_mismatch`` gate.
    Solved-frame goals come from :func:`propose_goal_hypotheses`; an immediate
    stage swap may instead use the separately gated, direct-overlap
    :func:`propose_completion_motion_reach` proof.
    """

    if not config.enabled:
        return ()
    arr = np.asarray(frame)
    candidates: list[Hypothesis] = []
    for hypothesis in _region_relation_candidates(frame, objects, config):
        if not hypothesis.observable(arr, mask_cells):
            continue
        potential = hypothesis.potential(arr, mask_cells)
        if 0.0 < potential <= float(config.max_initial_mismatch):
            hypothesis.initial_potential = potential
            candidates.append(hypothesis)
    candidates.sort(key=lambda h: (h.initial_potential, h.hypothesis_id))
    return tuple(candidates[: int(config.max_hypotheses)])


def propose_goal_hypotheses(
    initial_frame: np.ndarray,
    completion_frame: np.ndarray,
    objects: Sequence[ObjectState],
    config: HypothesisConfig,
    *,
    mask_cells: frozenset[tuple[int, int]] = frozenset(),
    count_trace: Sequence[Mapping[int, int]] | None = None,
) -> tuple[Hypothesis, ...]:
    """Genuine goals: relations violated at level start, satisfied at completion.

    Unlike :func:`propose_hypotheses`, this admits relations regardless of
    their potential on the completion frame's structure and then keeps only
    those that were violated (potential >= ``goal_contrast_floor``) at the
    level's initial frame and satisfied (<= ``promotion_epsilon``) at
    completion.  Such relations are the ones the agent actually resolved, so
    they are returned pre-verified and tagged ``origin="goal_contrast"``.
    This is the ordinary solved-frame route for ``reach`` and the only route
    for count/coverage goals.  A separate conservative completion-motion
    fallback exists solely for immediate stage swaps that hide the solved
    frame.
    """

    if not config.enabled or not config.enable_goal_contrast:
        return ()
    completion = np.asarray(completion_frame)
    initial = np.asarray(initial_frame)
    if completion.shape != initial.shape:
        return ()
    goals: list[Hypothesis] = []
    seen: set[str] = set()
    candidates: Iterable[Hypothesis] = (
        *_region_relation_candidates(completion_frame, objects, config),
        *_reach_relation_candidates(objects, config),
        *_count_target_candidates(
            initial_frame,
            completion_frame,
            config,
            mask_cells,
            count_trace,
        ),
    )
    for hypothesis in candidates:
        if hypothesis.hypothesis_id in seen:
            continue
        if not hypothesis.observable(completion, mask_cells):
            continue
        completion_potential = hypothesis.potential(completion, mask_cells)
        if completion_potential > float(config.promotion_epsilon):
            continue
        if not hypothesis.observable(initial, mask_cells):
            continue
        initial_potential = hypothesis.potential(initial, mask_cells)
        if initial_potential < float(config.goal_contrast_floor):
            continue
        seen.add(hypothesis.hypothesis_id)
        hypothesis.initial_potential = initial_potential
        hypothesis.verified = True
        hypothesis.origin = "goal_contrast"
        goals.append(hypothesis)
    goals.sort(key=lambda h: (-h.initial_potential, h.hypothesis_id))
    return tuple(goals[: int(config.max_hypotheses)])


def propose_completion_motion_reach(
    initial_frame: np.ndarray,
    predecessor_frame: np.ndarray,
    objects: Sequence[ObjectState],
    config: HypothesisConfig,
    *,
    controlled_motion_predictions: Mapping[
        int, Sequence[float]
    ],
    mask_cells: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[Hypothesis, ...]:
    """Infer one reach goal when a stage swap hides the solved frame.

    This is intentionally much narrower than ordinary goal contrast.  A
    candidate is accepted only when a supported ego-motion prediction moves
    exactly one controlled colour *onto* exactly one target colour whose
    visible cells have stayed byte-for-byte fixed since the stage began.
    Adjacency alone is insufficient, moving targets are excluded, and any
    ambiguity causes abstention.  The observed completion remains the causal
    ground; the learned motion model only reconstructs the otherwise hidden
    old-stage result.
    """

    if (
        not config.enabled
        or not config.enable_goal_contrast
        or not config.enable_completion_motion_reach
        or not controlled_motion_predictions
    ):
        return ()
    initial = np.asarray(initial_frame)
    predecessor = np.asarray(predecessor_frame)
    if (
        initial.ndim != 2
        or predecessor.ndim != 2
        or initial.shape != predecessor.shape
    ):
        return ()

    by_value: dict[int, list[ObjectState]] = {}
    for obj in objects:
        by_value.setdefault(int(obj.value), []).append(obj)

    candidates: list[Hypothesis] = []
    for raw_track_id, raw_prediction in sorted(
        controlled_motion_predictions.items(),
        key=lambda item: int(item[0]),
    ):
        try:
            track_id = int(raw_track_id)
            if (
                isinstance(raw_prediction, (str, bytes))
                or len(raw_prediction) != 3
            ):
                continue
            dx, dy, consistency = (float(value) for value in raw_prediction)
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite((dx, dy, consistency)).all():
            continue
        if consistency < float(config.completion_motion_min_consistency):
            continue
        shift_x, shift_y = int(round(dx)), int(round(dy))
        if shift_x == 0 and shift_y == 0:
            continue
        sources = [obj for obj in objects if int(obj.track_id) == track_id]
        if len(sources) != 1:
            continue
        source = sources[0]
        source_value = int(source.value)
        # A colour split across multiple components cannot be shifted from a
        # single object prediction without inventing which cells moved.
        if len(by_value.get(source_value, ())) != 1:
            continue
        source_initial = _visible_value_cells(
            initial,
            source_value,
            mask_cells,
        )
        source_before = _visible_value_cells(
            predecessor,
            source_value,
            mask_cells,
        )
        if (
            source_initial.shape[0] == 0
            or source_before.shape[0] == 0
            or source_initial.shape[0] > int(config.reach_cell_cap)
            or source_before.shape[0] != int(source.area)
            or np.array_equal(source_initial, source_before)
        ):
            continue
        projected_source = source_before + np.asarray(
            [shift_y, shift_x],
            dtype=np.int64,
        )
        if (
            (projected_source[:, 0] < 0).any()
            or (projected_source[:, 0] >= predecessor.shape[0]).any()
            or (projected_source[:, 1] < 0).any()
            or (projected_source[:, 1] >= predecessor.shape[1]).any()
        ):
            continue

        for target_value, target_objects in sorted(by_value.items()):
            if target_value == source_value or len(target_objects) != 1:
                continue
            target = target_objects[0]
            if int(target.track_id) == track_id:
                continue
            target_initial = _visible_value_cells(
                initial,
                target_value,
                mask_cells,
            )
            target_before = _visible_value_cells(
                predecessor,
                target_value,
                mask_cells,
            )
            if (
                target_initial.shape[0] == 0
                or target_initial.shape[0] > int(config.reach_cell_cap)
                or target_before.shape[0] != int(target.area)
                or not np.array_equal(target_initial, target_before)
            ):
                continue
            before_gap = int(
                np.min(
                    np.abs(
                        source_before[:, None, :]
                        - target_before[None, :, :]
                    ).max(axis=2)
                )
            )
            projected_gap = int(
                np.min(
                    np.abs(
                        projected_source[:, None, :]
                        - target_before[None, :, :]
                    ).max(axis=2)
                )
            )
            # The hidden solved frame must be reconstructed as direct
            # contact/overlap, not merely a weak adjacency prediction.
            if projected_gap != 0 or projected_gap >= before_gap:
                continue
            hypothesis = Hypothesis(
                hypothesis_id=f"reach|{source_value}|{target_value}",
                kind="reach",
                region_a=(0, 0, 0, 0),
                value_a=source_value,
                value_b=target_value,
                reach_cell_cap=int(config.reach_cell_cap),
                verified=True,
                origin="completion_motion",
            )
            if (
                not hypothesis.observable(initial, mask_cells)
                or not hypothesis.observable(predecessor, mask_cells)
            ):
                continue
            initial_potential = hypothesis.potential(initial, mask_cells)
            predecessor_potential = hypothesis.potential(
                predecessor,
                mask_cells,
            )
            if (
                initial_potential < float(config.goal_contrast_floor)
                or predecessor_potential
                <= float(config.promotion_epsilon)
            ):
                continue
            hypothesis.initial_potential = initial_potential
            candidates.append(hypothesis)

    # More than one geometrically plausible reconstruction is not evidence
    # for choosing among them.
    by_id = {candidate.hypothesis_id: candidate for candidate in candidates}
    if len(by_id) != 1:
        return ()
    return (next(iter(by_id.values())),)


def _bounded_hypotheses(
    hypotheses: Iterable[Hypothesis],
    limit: int,
) -> list[Hypothesis]:
    """Deduplicate and retain the strongest bounded hypothesis set.

    Completion-grounded goals outrank structural invariants.  This lets a
    newly observed genuine goal displace an exploratory proposal when a
    scope is already at its configured capacity.
    """

    by_id: dict[str, Hypothesis] = {}
    for hypothesis in hypotheses:
        existing = by_id.get(hypothesis.hypothesis_id)
        if existing is None:
            by_id[hypothesis.hypothesis_id] = hypothesis
            continue
        existing_goal = existing.origin in _GOAL_ORIGINS
        candidate_goal = hypothesis.origin in _GOAL_ORIGINS
        if (
            candidate_goal,
            hypothesis.verified,
            not hypothesis.refuted,
        ) > (
            existing_goal,
            existing.verified,
            not existing.refuted,
        ):
            by_id[hypothesis.hypothesis_id] = hypothesis

    def _rank(hypothesis: Hypothesis) -> tuple[Any, ...]:
        goal = hypothesis.origin in _GOAL_ORIGINS
        potential_rank = (
            -float(hypothesis.initial_potential)
            if goal
            else float(hypothesis.initial_potential)
        )
        return (
            hypothesis.refuted,
            not goal,
            not hypothesis.verified,
            potential_rank,
            hypothesis.hypothesis_id,
        )

    return sorted(by_id.values(), key=_rank)[: max(0, int(limit))]


def _target_signature(objects: Sequence[ObjectState], action: Action) -> str:
    if not action.has_position or not objects:
        return ""
    nearest = min(
        objects,
        key=lambda obj: (obj.centroid_x - action.x) ** 2
        + (obj.centroid_y - action.y) ** 2,
    )
    return str(nearest.signature)


class RelationalHypothesisEngine:
    """Propose, track, score, and verify relational goal hypotheses."""

    def __init__(self, config: HypothesisConfig | None = None) -> None:
        self.config = config or HypothesisConfig()
        self._by_scope: dict[tuple[str, int], list[Hypothesis]] = {}
        # (scope, hypothesis_id, memory_state_id) -> potential
        self._potential_cache: dict[tuple[str, str], float] = {}
        # Per-scope first observed frame, retained for completion evidence.
        self._initial_frames: dict[tuple[str, int], np.ndarray] = {}
        # Bounded observed-frame cache lets a goal learned only at completion
        # backfill potentials over the already-built persistent state graph.
        self._state_frames: dict[str, np.ndarray] = {}
        # Current-episode value-count trajectories for monotonic count goals.
        self._count_traces: dict[
            tuple[str, int], list[dict[int, int]]
        ] = {}
        self.observed_transitions = 0
        self.promotions = 0
        self.refutations = 0
        self.goal_proposals = 0

    def __deepcopy__(
        self,
        memo: dict[int, object],
    ) -> "RelationalHypothesisEngine":
        duplicate = type(self)(self.config)
        memo[id(self)] = duplicate
        duplicate._by_scope = copy.deepcopy(self._by_scope, memo)
        duplicate._potential_cache = dict(self._potential_cache)
        # Arrays are backed by immutable bytes and are safe to share between
        # transactional staged copies.
        duplicate._initial_frames = dict(self._initial_frames)
        duplicate._state_frames = dict(self._state_frames)
        duplicate._count_traces = copy.deepcopy(self._count_traces, memo)
        duplicate.observed_transitions = int(self.observed_transitions)
        duplicate.promotions = int(self.promotions)
        duplicate.refutations = int(self.refutations)
        duplicate.goal_proposals = int(self.goal_proposals)
        return duplicate

    def begin_episode(self, task_id: str) -> None:
        """Reset transient contrasts for a genuinely new environment run."""

        task = str(task_id)
        for scope in tuple(self._initial_frames):
            if scope[0] == task:
                self._initial_frames.pop(scope, None)
        for scope in tuple(self._count_traces):
            if scope[0] == task:
                self._count_traces.pop(scope, None)

    def _remember_frame(self, state_id: str, frame: np.ndarray) -> None:
        key = str(state_id)
        if not key:
            return
        self._state_frames.pop(key, None)
        self._state_frames[key] = readonly_array(frame)
        limit = int(self.config.goal_frame_cache_limit)
        while len(self._state_frames) > limit:
            self._state_frames.pop(next(iter(self._state_frames)))

    def _scope(self, task_id: str, stage: int) -> tuple[str, int]:
        return (str(task_id), max(1, int(stage)))

    def active(self, task_id: str, stage: int) -> tuple[Hypothesis, ...]:
        if not self.config.enabled:
            return ()
        return tuple(
            h
            for h in self._by_scope.get(self._scope(task_id, stage), ())
            if not h.refuted
        )

    def ensure_proposals(
        self,
        observation: Observation,
        objects: Sequence[ObjectState],
        *,
        mask_cells: frozenset[tuple[int, int]] = frozenset(),
        memory_state_id: str = "",
    ) -> int:
        """Propose hypotheses for an unseen (task, stage); returns count added."""

        if not self.config.enabled:
            return 0
        scope = self._scope(observation.task_id, observation.stage)
        # The first frame seen for a scope is the level's initial state; keep
        # it so a later completion can be contrasted against it.
        self._initial_frames.setdefault(
            scope, readonly_array(observation.frame)
        )
        self._count_traces.setdefault(
            scope,
            [_visible_value_counts(observation.frame, mask_cells)],
        )
        self._remember_frame(
            memory_state_id or observation.state_id,
            observation.frame,
        )
        if scope in self._by_scope:
            return 0
        proposals = list(
            propose_hypotheses(
                observation.frame,
                objects,
                self.config,
                mask_cells=mask_cells,
            )
        )
        self._by_scope[scope] = proposals
        return len(proposals)

    def _admit_goals(
        self,
        scope: tuple[str, int],
        goals: Iterable[Hypothesis],
    ) -> int:
        """Merge verified completion-grounded goals into one bounded scope."""

        bucket = self._by_scope.setdefault(scope, [])
        by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in bucket}
        added = 0
        for goal in goals:
            existing = by_id.get(goal.hypothesis_id)
            if existing is None:
                bucket.append(goal)
                by_id[goal.hypothesis_id] = goal
                self.goal_proposals += 1
                added += 1
            elif existing.origin not in _GOAL_ORIGINS and not existing.refuted:
                # Promote a coincidental invariant to a proven goal while
                # retaining the more precise evidence provenance.
                existing.origin = goal.origin
                existing.verified = True
                existing.initial_potential = goal.initial_potential
                added += 1
        bucket[:] = _bounded_hypotheses(
            bucket,
            int(self.config.max_hypotheses),
        )
        return added

    def _cached_potential(
        self,
        scope: tuple[str, int],
        hypothesis: Hypothesis,
        memory_state_id: str,
        frame: np.ndarray | None,
        mask_cells: frozenset[tuple[int, int]],
    ) -> float | None:
        key = (f"{scope[0]}␟{scope[1]}␟{hypothesis.hypothesis_id}", memory_state_id)
        if key in self._potential_cache:
            return self._potential_cache[key]
        if frame is None and memory_state_id:
            frame = self._state_frames.get(memory_state_id)
        if frame is None:
            return None
        value = hypothesis.potential(frame, mask_cells)
        if memory_state_id:
            self._potential_cache[key] = value
            if len(self._potential_cache) > 20_000:
                self._potential_cache.pop(next(iter(self._potential_cache)))
        return value

    def planning_goals(
        self,
        snapshot: WorldSnapshot,
        *,
        mask_cells: frozenset[tuple[int, int]] = frozenset(),
    ) -> tuple[GoalEstimate, ...]:
        """Verified completion-grounded goals eligible for graph guidance."""

        if not self.config.enabled:
            return ()
        scope = self._scope(
            snapshot.observation.task_id,
            snapshot.observation.stage,
        )
        estimates: list[GoalEstimate] = []
        for hypothesis in self._by_scope.get(scope, ()):
            if (
                hypothesis.refuted
                or not hypothesis.verified
                or hypothesis.origin not in _GOAL_ORIGINS
                or hypothesis.kind
                not in {"reach", "count_at_most", "count_at_least"}
            ):
                continue
            current = self._cached_potential(
                scope,
                hypothesis,
                snapshot.memory_id,
                snapshot.observation.frame,
                mask_cells,
            )
            if current is None or not np.isfinite(current):
                continue
            estimates.append(
                GoalEstimate(
                    hypothesis_id=hypothesis.hypothesis_id,
                    kind=hypothesis.kind,
                    current_potential=float(np.clip(current, 0.0, 1.0)),
                )
            )
        return tuple(
            sorted(
                estimates,
                key=lambda row: (
                    row.current_potential,
                    row.kind,
                    row.hypothesis_id,
                ),
            )
        )

    def cached_goal_potential(
        self,
        *,
        task_id: str,
        stage: int,
        hypothesis_id: str,
        state_id: str,
        mask_cells: frozenset[tuple[int, int]] = frozenset(),
    ) -> float | None:
        """Return an observed state's cached potential for one verified goal."""

        scope = self._scope(task_id, stage)
        hypothesis = next(
            (
                row
                for row in self._by_scope.get(scope, ())
                if row.hypothesis_id == str(hypothesis_id)
                and row.verified
                and not row.refuted
                and row.origin in _GOAL_ORIGINS
                and row.kind in {"reach", "count_at_most", "count_at_least"}
            ),
            None,
        )
        if hypothesis is None:
            return None
        return self._cached_potential(
            scope,
            hypothesis,
            str(state_id),
            None,
            mask_cells,
        )

    def observe_transition(
        self,
        *,
        before_observation: Observation,
        after_observation: Observation,
        objects: Sequence[ObjectState],
        action: Action,
        progressed: bool,
        before_memory_id: str,
        after_memory_id: str,
        mask_cells: frozenset[tuple[int, int]] = frozenset(),
        completion_observation: Observation | None = None,
        completion_objects: Sequence[ObjectState] | None = None,
        completion_frame_available: bool = True,
        controlled_motion_predictions: Mapping[
            int, Sequence[float]
        ] | None = None,
    ) -> int:
        """Track realized potential deltas; promote/refute on completions."""

        if not self.config.enabled:
            return 0
        self.ensure_proposals(
            before_observation,
            objects,
            mask_cells=mask_cells,
            memory_state_id=before_memory_id,
        )
        scope = self._scope(before_observation.task_id, before_observation.stage)
        trace = self._count_traces.setdefault(scope, [])
        before_counts = _visible_value_counts(
            before_observation.frame,
            mask_cells,
        )
        if not trace or trace[-1] != before_counts:
            trace.append(before_counts)
        self._remember_frame(before_memory_id, before_observation.frame)
        # On a completion, mine genuine goals only when an old-stage solved
        # frame is actually available.  An immediate next-stage board must
        # never be contrasted with this scope, nor may the predecessor be
        # mislabeled as the solved state.
        if (
            progressed
            and completion_frame_available
            and self.config.enable_goal_contrast
        ):
            initial_frame = self._initial_frames.get(scope)
            if initial_frame is not None:
                goal_observation = completion_observation or before_observation
                goal_objects = (
                    tuple(completion_objects)
                    if completion_objects is not None
                    else tuple(objects)
                )
                goal_counts = _visible_value_counts(
                    goal_observation.frame,
                    mask_cells,
                )
                if not trace or trace[-1] != goal_counts:
                    trace.append(goal_counts)
                self._admit_goals(
                    scope,
                    propose_goal_hypotheses(
                        initial_frame,
                        goal_observation.frame,
                        goal_objects,
                        self.config,
                        mask_cells=mask_cells,
                        count_trace=trace,
                    ),
                )
        elif (
            progressed
            and not completion_frame_available
            and self.config.enable_goal_contrast
        ):
            initial_frame = self._initial_frames.get(scope)
            if initial_frame is not None:
                self._admit_goals(
                    scope,
                    propose_completion_motion_reach(
                        initial_frame,
                        before_observation.frame,
                        objects,
                        self.config,
                        controlled_motion_predictions=(
                            controlled_motion_predictions or {}
                        ),
                        mask_cells=mask_cells,
                    ),
                )
        hypotheses = self._by_scope.get(scope, ())
        if not hypotheses:
            return 0
        comparable_successor = bool(
            str(after_observation.task_id) == str(before_observation.task_id)
            and int(after_observation.stage) == int(before_observation.stage)
            and tuple(after_observation.frame.shape)
            == tuple(before_observation.frame.shape)
        )
        if comparable_successor:
            self._remember_frame(after_memory_id, after_observation.frame)
        key = (int(action.index), _target_signature(objects, action))
        updated = 0
        for hypothesis in hypotheses:
            if hypothesis.refuted:
                continue
            before_potential = self._cached_potential(
                scope,
                hypothesis,
                before_memory_id,
                before_observation.frame,
                mask_cells,
            )
            if before_potential is None:
                continue
            # Completion validates/refutes the relation visible immediately
            # before the boundary.  A new stage's frame must never be fed
            # through old-stage geometry merely to perform this check.
            if progressed and completion_frame_available:
                completion_frame = (
                    completion_observation.frame
                    if completion_observation is not None
                    else before_observation.frame
                )
                if hypothesis.observable(completion_frame, mask_cells):
                    completion_potential = (
                        hypothesis.potential(
                            completion_frame,
                            mask_cells,
                        )
                        if completion_observation is not None
                        else before_potential
                    )
                    if completion_potential <= float(
                        self.config.promotion_epsilon
                    ):
                        if not hypothesis.verified:
                            hypothesis.verified = True
                            self.promotions += 1
                    elif completion_potential >= float(
                        self.config.refutation_floor
                    ):
                        hypothesis.refuted = True
                        self.refutations += 1
            if not comparable_successor:
                continue
            after_potential = self._cached_potential(
                scope,
                hypothesis,
                after_memory_id,
                after_observation.frame,
                mask_cells,
            )
            if after_potential is None:
                continue
            row = hypothesis.deltas.setdefault(key, [0.0, 0.0])
            row[0] += float(after_potential - before_potential)
            row[1] += 1.0
            updated += 1
        if comparable_successor:
            after_counts = _visible_value_counts(
                after_observation.frame,
                mask_cells,
            )
            if not trace or trace[-1] != after_counts:
                trace.append(after_counts)
        if updated:
            self.observed_transitions += 1
        return updated

    def _signal(
        self,
        scope: tuple[str, int],
        action: Action,
        objects: Sequence[ObjectState],
        *,
        current_state_id: str | None,
        current_frame: np.ndarray | None,
        successor_state_id: str | None,
        mask_cells: frozenset[tuple[int, int]],
    ) -> float:
        hypotheses = [h for h in self._by_scope.get(scope, ()) if not h.refuted]
        if not hypotheses:
            return 0.0
        key = (int(action.index), _target_signature(objects, action))
        total = 0.0
        for hypothesis in hypotheses:
            scale = float(
                self.config.verified_scale
                if hypothesis.verified
                else self.config.unverified_scale
            )
            expected_reduction: float | None = None
            if successor_state_id and current_state_id:
                current = self._cached_potential(
                    scope,
                    hypothesis,
                    current_state_id,
                    current_frame,
                    mask_cells,
                )
                successor = self._cached_potential(
                    scope,
                    hypothesis,
                    successor_state_id,
                    None,
                    mask_cells,
                )
                if current is not None and successor is not None:
                    expected_reduction = current - successor
            if expected_reduction is None:
                row = hypothesis.deltas.get(key)
                if row is None or row[1] < float(self.config.min_support):
                    continue
                trust = row[1] / (row[1] + float(self.config.min_support))
                expected_reduction = -(row[0] / row[1]) * trust
            total += scale * expected_reduction
        return float(np.clip(total * float(self.config.signal_gain), -1.0, 1.0))

    def candidate_signal(
        self,
        snapshot: WorldSnapshot,
        action: Action,
        *,
        successor_memory_id: str | None = None,
        mask_cells: frozenset[tuple[int, int]] = frozenset(),
    ) -> float:
        """Bounded [-1, 1] reading: expected mismatch reduction under action."""

        if not self.config.enabled:
            return 0.0
        return self._signal(
            self._scope(snapshot.observation.task_id, snapshot.observation.stage),
            action,
            snapshot.objects,
            current_state_id=snapshot.memory_id,
            current_frame=snapshot.observation.frame,
            successor_state_id=successor_memory_id,
            mask_cells=mask_cells,
        )

    def rollout_signal(
        self,
        snapshot: WorldSnapshot,
        action: Action,
        *,
        current_state_id: str | None,
        successor_state_id: str | None,
        mask_cells: frozenset[tuple[int, int]] = frozenset(),
    ) -> float:
        """Signal for an imagined node: cached exact potentials or the prior.

        Imagined states have no frames; exact potential differences apply
        only along known graph chains whose potentials were cached when the
        states were really visited.
        """

        if not self.config.enabled:
            return 0.0
        return self._signal(
            self._scope(snapshot.observation.task_id, snapshot.observation.stage),
            action,
            snapshot.objects,
            current_state_id=current_state_id,
            current_frame=None,
            successor_state_id=successor_state_id,
            mask_cells=mask_cells,
        )

    def mismatch_points(
        self,
        snapshot: WorldSnapshot,
        *,
        limit: int = 8,
        mask_cells: frozenset[tuple[int, int]] = frozenset(),
    ) -> tuple[tuple[int, int], ...]:
        """Click-priority (x, y) cells violating active relations."""

        if not self.config.enabled or int(limit) <= 0:
            return ()
        scope = self._scope(
            snapshot.observation.task_id,
            snapshot.observation.stage,
        )
        hypotheses = sorted(
            (h for h in self._by_scope.get(scope, ()) if not h.refuted),
            key=lambda h: (not h.verified, h.initial_potential, h.hypothesis_id),
        )
        points: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        frame = np.asarray(snapshot.observation.frame)
        for hypothesis in hypotheses:
            for point in hypothesis.mismatch_cells(frame, mask_cells):
                if point not in seen:
                    seen.add(point)
                    points.append(point)
                if len(points) >= max(0, int(limit)):
                    return tuple(points)
        return tuple(points)

    def summary(self) -> dict[str, Any]:
        hypotheses = [h for rows in self._by_scope.values() for h in rows]
        return {
            "enabled": bool(self.config.enabled),
            "scopes": len(self._by_scope),
            "hypotheses": len(hypotheses),
            "verified": sum(int(h.verified) for h in hypotheses),
            "refuted": sum(int(h.refuted) for h in hypotheses),
            # Relations proven genuine goals via violated->satisfied contrast,
            # as opposed to coincidental already-satisfied invariants.
            "goals": sum(int(h.origin in _GOAL_ORIGINS) for h in hypotheses),
            "reach_goals": sum(
                int(h.origin in _GOAL_ORIGINS and h.kind == "reach")
                for h in hypotheses
            ),
            "count_goals": sum(
                int(
                    h.origin in _GOAL_ORIGINS
                    and h.kind in {"count_at_most", "count_at_least"}
                )
                for h in hypotheses
            ),
            "observed_transitions": int(self.observed_transitions),
        }

    def state_dict(self) -> dict[str, Any]:
        def _packed_frame(state_id: str, frame: np.ndarray) -> dict[str, Any]:
            arr = np.ascontiguousarray(np.asarray(frame))
            return {
                "state_id": state_id,
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "data": base64.b64encode(
                    zlib.compress(arr.tobytes(order="C"), level=6)
                ).decode("ascii"),
            }

        payload = {}
        for (task_id, stage), rows in self._by_scope.items():
            payload[f"{task_id}␟{stage}"] = [
                {
                    "hypothesis_id": h.hypothesis_id,
                    "kind": h.kind,
                    "region_a": list(h.region_a),
                    "region_b": None if h.region_b is None else list(h.region_b),
                    "lattice": None if h.lattice is None else list(h.lattice),
                    "verified": h.verified,
                    "refuted": h.refuted,
                    "initial_potential": h.initial_potential,
                    "value_a": h.value_a,
                    "value_b": h.value_b,
                    "origin": h.origin,
                    "reach_cell_cap": int(h.reach_cell_cap),
                    "initial_count": h.initial_count,
                    "target_count": h.target_count,
                    "deltas": {
                        f"{index}␟{signature}": [float(v) for v in row]
                        for (index, signature), row in h.deltas.items()
                    },
                }
                for h in rows
            ]
        return {
            "scopes": payload,
            "initial_frames": [
                {
                    "task_id": task_id,
                    "stage": int(stage),
                    "dtype": str(frame.dtype),
                    "frame": frame.tolist(),
                }
                for (task_id, stage), frame in sorted(
                    self._initial_frames.items()
                )
            ],
            "state_frames": [
                _packed_frame(state_id, frame)
                for state_id, frame in self._state_frames.items()
            ],
            "count_traces": [
                {
                    "task_id": task_id,
                    "stage": int(stage),
                    "rows": [
                        {str(value): int(count) for value, count in row.items()}
                        for row in rows
                    ],
                }
                for (task_id, stage), rows in sorted(
                    self._count_traces.items()
                )
            ],
            "potential_cache": [
                {
                    "hypothesis_key": hypothesis_key,
                    "state_id": state_id,
                    "potential": float(value),
                }
                for (hypothesis_key, state_id), value in sorted(
                    self._potential_cache.items()
                )
            ],
            "observed_transitions": int(self.observed_transitions),
            "promotions": int(self.promotions),
            "refutations": int(self.refutations),
            "goal_proposals": int(self.goal_proposals),
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        config: HypothesisConfig | None = None,
    ) -> "RelationalHypothesisEngine":
        def _mapping(value: Any, name: str) -> Mapping[str, Any]:
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping")
            return value

        def _sequence(value: Any, name: str) -> tuple[Any, ...]:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{name} must be a sequence")
            return tuple(value)

        def _string(value: Any, name: str) -> str:
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            return value

        def _boolean(value: Any, name: str) -> bool:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
            return value

        def _integer(
            value: Any,
            name: str,
            *,
            minimum: int | None = None,
        ) -> int:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise ValueError(f"{name} must be an integer")
            result = int(value)
            if minimum is not None and result < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
            return result

        def _number(
            value: Any,
            name: str,
            *,
            minimum: float | None = None,
            maximum: float | None = None,
        ) -> float:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                raise ValueError(f"{name} must be a number")
            result = float(value)
            if not np.isfinite(result):
                raise ValueError(f"{name} must be finite")
            if minimum is not None and result < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
            if maximum is not None and result > maximum:
                raise ValueError(f"{name} must be <= {maximum}")
            return result

        def _optional_integer(
            value: Any,
            name: str,
            *,
            minimum: int | None = None,
        ) -> int | None:
            if value is None:
                return None
            return _integer(value, name, minimum=minimum)

        def _fixed_integers(
            value: Any,
            name: str,
            *,
            length: int,
        ) -> tuple[int, ...]:
            rows = _sequence(value, name)
            if len(rows) != length:
                raise ValueError(f"{name} must contain {length} integers")
            return tuple(
                _integer(item, f"{name}[{index}]")
                for index, item in enumerate(rows)
            )

        def _dtype(value: Any, name: str) -> np.dtype[Any]:
            text = _string(value, name)
            try:
                result = np.dtype(text)
            except TypeError as exc:
                raise ValueError(f"{name} is not a valid dtype") from exc
            if result.hasobject or result.kind not in "biuf":
                raise ValueError(f"{name} must be a real numeric dtype")
            return result

        def _frame(
            value: Any,
            dtype: np.dtype[Any],
            name: str,
        ) -> np.ndarray:
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise ValueError(f"{name} must be an array")
            try:
                result = np.asarray(value, dtype=dtype)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} contains invalid values") from exc
            if result.ndim != 2 or result.size == 0:
                raise ValueError(f"{name} must be a non-empty 2D array")
            if not bool(np.all(np.isfinite(result))):
                raise ValueError(f"{name} must contain only finite values")
            return result

        state = _mapping(state, "checkpoint hypotheses")
        engine = cls(config)
        raw_scopes = _mapping(
            state.get("scopes", {}),
            "checkpoint hypothesis scopes",
        )
        allowed_kinds = {
            "equal",
            "equal_canonical",
            "mirror_h",
            "mirror_v",
            "self_mirror_h",
            "self_mirror_v",
            "uniform",
            "reach",
            "count_at_most",
            "count_at_least",
        }
        allowed_origins = {"invariant", "goal_contrast", "completion_motion"}
        for raw_scope_text, raw_rows in raw_scopes.items():
            scope_text = _string(
                raw_scope_text,
                "checkpoint hypothesis scope key",
            )
            if "␟" not in scope_text:
                raise ValueError("checkpoint hypothesis scope key is invalid")
            task_id, stage_text = scope_text.rsplit("␟", 1)
            try:
                stage = int(stage_text)
            except ValueError as exc:
                raise ValueError(
                    "checkpoint hypothesis scope stage is invalid"
                ) from exc
            if stage < 1 or stage_text != str(stage):
                raise ValueError(
                    "checkpoint hypothesis scope stage is invalid"
                )
            rows = _sequence(
                raw_rows,
                f"checkpoint hypothesis scope {scope_text!r}",
            )
            if len(rows) > int(engine.config.max_hypotheses):
                raise ValueError(
                    "checkpoint hypothesis scope exceeds configured capacity"
                )
            hypotheses: list[Hypothesis] = []
            hypothesis_ids: set[str] = set()
            for row_index, raw_row in enumerate(rows):
                row_name = (
                    f"checkpoint hypothesis scope {scope_text!r}"
                    f"[{row_index}]"
                )
                row = _mapping(raw_row, row_name)
                hypothesis_id = _string(
                    row.get("hypothesis_id"),
                    f"{row_name}.hypothesis_id",
                )
                if hypothesis_id in hypothesis_ids:
                    raise ValueError(
                        "checkpoint hypothesis IDs must be unique within a scope"
                    )
                hypothesis_ids.add(hypothesis_id)
                kind = _string(row.get("kind"), f"{row_name}.kind")
                if kind not in allowed_kinds:
                    raise ValueError(f"{row_name}.kind is invalid")
                region_a = _fixed_integers(
                    row.get("region_a"),
                    f"{row_name}.region_a",
                    length=4,
                )
                raw_region_b = row.get("region_b")
                region_b = (
                    None
                    if raw_region_b is None
                    else _fixed_integers(
                        raw_region_b,
                        f"{row_name}.region_b",
                        length=4,
                    )
                )
                raw_lattice = row.get("lattice")
                lattice = (
                    None
                    if raw_lattice is None
                    else _fixed_integers(
                        raw_lattice,
                        f"{row_name}.lattice",
                        length=2,
                    )
                )
                if lattice is not None and min(lattice) < 1:
                    raise ValueError(f"{row_name}.lattice must be positive")
                origin = _string(
                    row.get("origin", "invariant"),
                    f"{row_name}.origin",
                )
                if origin not in allowed_origins:
                    raise ValueError(f"{row_name}.origin is invalid")
                hypothesis = Hypothesis(
                    hypothesis_id=hypothesis_id,
                    kind=kind,
                    region_a=region_a,  # type: ignore[arg-type]
                    region_b=region_b,  # type: ignore[arg-type]
                    lattice=lattice,  # type: ignore[arg-type]
                    verified=_boolean(
                        row.get("verified", False),
                        f"{row_name}.verified",
                    ),
                    refuted=_boolean(
                        row.get("refuted", False),
                        f"{row_name}.refuted",
                    ),
                    initial_potential=_number(
                        row.get("initial_potential", 1.0),
                        f"{row_name}.initial_potential",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    value_a=_optional_integer(
                        row.get("value_a"),
                        f"{row_name}.value_a",
                    ),
                    value_b=_optional_integer(
                        row.get("value_b"),
                        f"{row_name}.value_b",
                    ),
                    origin=origin,
                    reach_cell_cap=_integer(
                        row.get("reach_cell_cap", 400),
                        f"{row_name}.reach_cell_cap",
                        minimum=1,
                    ),
                    initial_count=_optional_integer(
                        row.get("initial_count"),
                        f"{row_name}.initial_count",
                        minimum=0,
                    ),
                    target_count=_optional_integer(
                        row.get("target_count"),
                        f"{row_name}.target_count",
                        minimum=0,
                    ),
                )
                raw_deltas = _mapping(
                    row.get("deltas", {}),
                    f"{row_name}.deltas",
                )
                for raw_key_text, raw_values in raw_deltas.items():
                    key_text = _string(
                        raw_key_text,
                        f"{row_name}.delta key",
                    )
                    if "␟" not in key_text:
                        raise ValueError(f"{row_name}.delta key is invalid")
                    index_text, signature = key_text.split("␟", 1)
                    try:
                        action_index = int(index_text)
                    except ValueError as exc:
                        raise ValueError(
                            f"{row_name}.delta action index is invalid"
                        ) from exc
                    if (
                        action_index < 0
                        or index_text != str(action_index)
                    ):
                        raise ValueError(
                            f"{row_name}.delta action index is invalid"
                        )
                    values = _sequence(
                        raw_values,
                        f"{row_name}.deltas[{key_text!r}]",
                    )
                    if len(values) != 2:
                        raise ValueError(
                            f"{row_name}.delta value must contain two numbers"
                        )
                    delta_sum = _number(
                        values[0],
                        f"{row_name}.deltas[{key_text!r}][0]",
                    )
                    count = _number(
                        values[1],
                        f"{row_name}.deltas[{key_text!r}][1]",
                        minimum=0.0,
                    )
                    if not count.is_integer():
                        raise ValueError(
                            f"{row_name}.delta count must be an integer-valued number"
                        )
                    hypothesis.deltas[(action_index, signature)] = [
                        delta_sum,
                        count,
                    ]
                hypotheses.append(hypothesis)
            engine._by_scope[(task_id, stage)] = hypotheses

        raw_initial_frames = _sequence(
            state.get("initial_frames", ()),
            "checkpoint hypothesis initial_frames",
        )
        for row_index, raw_row in enumerate(raw_initial_frames):
            row_name = f"checkpoint hypothesis initial_frames[{row_index}]"
            row = _mapping(raw_row, row_name)
            task_id = _string(row.get("task_id"), f"{row_name}.task_id")
            stage = _integer(
                row.get("stage", 1),
                f"{row_name}.stage",
                minimum=1,
            )
            key = (task_id, stage)
            if key in engine._initial_frames:
                raise ValueError(
                    "checkpoint hypothesis initial frame scopes must be unique"
                )
            dtype = _dtype(
                row.get("dtype", "int64"),
                f"{row_name}.dtype",
            )
            frame = _frame(row.get("frame"), dtype, f"{row_name}.frame")
            engine._initial_frames[key] = readonly_array(frame)

        raw_state_frames = _sequence(
            state.get("state_frames", ()),
            "checkpoint hypothesis state_frames",
        )
        limit = int(engine.config.goal_frame_cache_limit)
        if len(raw_state_frames) > limit:
            raise ValueError(
                "checkpoint hypothesis state-frame cache exceeds configured capacity"
            )
        max_cached_frame_bytes = 256 * 1024 * 1024
        for row_index, raw_row in enumerate(raw_state_frames):
            row_name = f"checkpoint hypothesis state_frames[{row_index}]"
            row = _mapping(raw_row, row_name)
            state_id = _string(row.get("state_id"), f"{row_name}.state_id")
            if not state_id:
                raise ValueError(f"{row_name}.state_id must not be empty")
            if state_id in engine._state_frames:
                raise ValueError(
                    "checkpoint hypothesis state frame IDs must be unique"
                )
            try:
                dtype = _dtype(
                    row.get("dtype", "int64"),
                    f"{row_name}.dtype",
                )
                shape = _fixed_integers(
                    row.get("shape"),
                    f"{row_name}.shape",
                    length=2,
                )
                if min(shape) < 1:
                    raise ValueError(f"{row_name}.shape must be positive")
                expected_bytes = (
                    int(shape[0]) * int(shape[1]) * int(dtype.itemsize)
                )
                if expected_bytes > max_cached_frame_bytes:
                    raise ValueError(
                        f"{row_name} exceeds the cached-frame byte limit"
                    )
                data = _string(row.get("data"), f"{row_name}.data")
                packed = base64.b64decode(data, validate=True)
                decompressor = zlib.decompressobj()
                raw = decompressor.decompress(packed, expected_bytes + 1)
                if (
                    len(raw) > expected_bytes
                    or not decompressor.eof
                    or decompressor.unused_data
                    or decompressor.unconsumed_tail
                ):
                    raise ValueError(f"{row_name} has invalid compressed data")
                raw += decompressor.flush()
                if len(raw) != expected_bytes:
                    raise ValueError(f"{row_name} has invalid decompressed size")
                frame = np.frombuffer(raw, dtype=dtype).reshape(shape)
            except (binascii.Error, TypeError, ValueError, zlib.error) as exc:
                raise ValueError("invalid cached hypothesis frame") from exc
            engine._state_frames[state_id] = readonly_array(frame)

        raw_count_traces = _sequence(
            state.get("count_traces", ()),
            "checkpoint hypothesis count_traces",
        )
        for row_index, raw_row in enumerate(raw_count_traces):
            row_name = f"checkpoint hypothesis count_traces[{row_index}]"
            row = _mapping(raw_row, row_name)
            task_id = _string(row.get("task_id"), f"{row_name}.task_id")
            stage = _integer(
                row.get("stage", 1),
                f"{row_name}.stage",
                minimum=1,
            )
            key = (task_id, stage)
            if key in engine._count_traces:
                raise ValueError(
                    "checkpoint hypothesis count-trace scopes must be unique"
                )
            raw_traces = _sequence(row.get("rows", ()), f"{row_name}.rows")
            traces: list[dict[int, int]] = []
            for trace_index, raw_trace in enumerate(raw_traces):
                trace_name = f"{row_name}.rows[{trace_index}]"
                trace = _mapping(raw_trace, trace_name)
                counts: dict[int, int] = {}
                for raw_value, raw_count in trace.items():
                    value_text = _string(
                        raw_value,
                        f"{trace_name} value key",
                    )
                    try:
                        value = int(value_text)
                    except ValueError as exc:
                        raise ValueError(
                            f"{trace_name} value key is invalid"
                        ) from exc
                    if value_text != str(value):
                        raise ValueError(
                            f"{trace_name} value key is invalid"
                        )
                    counts[value] = _integer(
                        raw_count,
                        f"{trace_name}[{value_text!r}]",
                        minimum=0,
                    )
                traces.append(counts)
            engine._count_traces[key] = traces

        raw_potential_cache = _sequence(
            state.get("potential_cache", ()),
            "checkpoint hypothesis potential_cache",
        )
        for row_index, raw_row in enumerate(raw_potential_cache):
            row_name = f"checkpoint hypothesis potential_cache[{row_index}]"
            row = _mapping(raw_row, row_name)
            hypothesis_key = _string(
                row.get("hypothesis_key"),
                f"{row_name}.hypothesis_key",
            )
            state_id = _string(
                row.get("state_id"),
                f"{row_name}.state_id",
            )
            if not hypothesis_key or not state_id:
                raise ValueError(
                    "checkpoint hypothesis potential-cache keys must not be empty"
                )
            key = (hypothesis_key, state_id)
            if key in engine._potential_cache:
                raise ValueError(
                    "checkpoint hypothesis potential-cache keys must be unique"
                )
            engine._potential_cache[key] = _number(
                row.get("potential", 0.0),
                f"{row_name}.potential",
                minimum=0.0,
                maximum=1.0,
            )
        engine.observed_transitions = _integer(
            state.get("observed_transitions", 0),
            "checkpoint hypothesis observed_transitions",
            minimum=0,
        )
        engine.promotions = _integer(
            state.get("promotions", 0),
            "checkpoint hypothesis promotions",
            minimum=0,
        )
        engine.refutations = _integer(
            state.get("refutations", 0),
            "checkpoint hypothesis refutations",
            minimum=0,
        )
        engine.goal_proposals = _integer(
            state.get("goal_proposals", 0),
            "checkpoint hypothesis goal_proposals",
            minimum=0,
        )
        return engine


__all__ = [
    "GoalEstimate",
    "Hypothesis",
    "RelationalHypothesisEngine",
    "propose_completion_motion_reach",
    "propose_goal_hypotheses",
    "propose_hypotheses",
]
