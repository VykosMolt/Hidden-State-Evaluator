"""The 14 frozen metrics of contract section 9, as PURE functions over records.

Every function here takes ``flb200.episode_record.v1`` dicts (or the small
summary dicts produced by ``interference``/``poison_eval``/``remap_eval``) and
returns numbers.  No model, no I/O, no randomness.

MACRO CONVENTION (frozen, applied everywhere)
---------------------------------------------
Aggregation is always FAMILY-LEVEL FIRST: episodes are averaged within a
family, then families are averaged with equal weight.  A family contributing
ten times as many episodes as another therefore cannot move an aggregate;
``tests/hostile/test_hostile_metric_family_domination.py`` pins this.

Curve conventions
-----------------
``R_k`` is the mean verifier success over the attempts carried by interaction
index ``k`` in one episode (so ``R_5`` averages the three QUERY items and
``R_6`` the two TRANSFER items, contract section 6).  A family curve is the
episode-mean of ``R_k``; the macro curve is the family-mean of family curves.
Because every episode of ``EPISODE_STRUCTURE_V0`` shares the index grid
``0..6``, the OLS slope of the macro curve equals the mean of the per-family
OLS slopes; :func:`improvement_slope` returns the macro-curve slope and
:func:`per_family_slopes` exposes the family values, and
:func:`slope_conventions_agree` checks the identity on real inputs rather than
assuming it.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

METRICS_SCHEMA = "flb200.metrics_v0.v1"

#: contract section 9, metric 7
DEFAULT_THRESHOLD = 0.5
#: contract section 6
DEFAULT_INDICES: tuple[int, ...] = tuple(range(7))
#: contract section 9, metric 8 (related-task transfer is the index-4 attempt)
RELATED_INDEX = 4
#: A ratio whose denominator is only floating-point noise (e.g. a "gain" of
#: 1e-17 produced by averaging identical numbers) carries no information; such
#: a ratio is reported as undefined rather than as a spurious 1.0.
MIN_DENOMINATOR = 1e-9


# --------------------------------------------------------------------------
# record accessors
# --------------------------------------------------------------------------
def record_family(record: Mapping[str, Any]) -> str:
    fid = record.get("family_id")
    if fid is None:
        raise KeyError("episode record has no family_id")
    return str(fid)


def record_R(record: Mapping[str, Any]) -> dict[int, float]:
    raw = record.get("R") or {}
    return {int(k): float(v) for k, v in raw.items()}


def record_aulc(record: Mapping[str, Any],
                indices: Sequence[int] | None = None) -> float | None:
    """Area under the learning curve of ONE episode = mean over indices of R_k."""
    R = record_R(record)
    keys = [k for k in (indices if indices is not None else sorted(R)) if k in R]
    if not keys:
        return None
    return float(sum(R[k] for k in keys) / len(keys))


def group_by_family(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        out.setdefault(record_family(r), []).append(r)
    return out


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def macro_mean(records: Sequence[Mapping[str, Any]],
               value_fn: Callable[[Mapping[str, Any]], float | None]) -> float | None:
    """Family means first, then the unweighted mean over families."""
    per_family = []
    for _fid, rs in sorted(group_by_family(records).items()):
        fam = _mean(value_fn(r) for r in rs)
        if fam is not None:
            per_family.append(fam)
    return _mean(per_family)


def per_family_mean(records: Sequence[Mapping[str, Any]],
                    value_fn: Callable[[Mapping[str, Any]], float | None]
                    ) -> dict[str, float]:
    out: dict[str, float] = {}
    for fid, rs in sorted(group_by_family(records).items()):
        fam = _mean(value_fn(r) for r in rs)
        if fam is not None:
            out[fid] = fam
    return out


# --------------------------------------------------------------------------
# 1-3: AULC and deltas
# --------------------------------------------------------------------------
def macro_aulc(records: Sequence[Mapping[str, Any]],
               indices: Sequence[int] | None = None) -> float | None:
    """METRIC 1 — macro-AULC (mean over families of mean success over 0..K)."""
    return macro_mean(records, lambda r: record_aulc(r, indices))


def per_family_aulc(records: Sequence[Mapping[str, Any]],
                    indices: Sequence[int] | None = None) -> dict[str, float]:
    return per_family_mean(records, lambda r: record_aulc(r, indices))


def delta_aulc(treatment: Sequence[Mapping[str, Any]],
               baseline: Sequence[Mapping[str, Any]],
               indices: Sequence[int] | None = None) -> float | None:
    """METRICS 2/3 — macro-AULC(treatment) - macro-AULC(baseline)."""
    a = macro_aulc(treatment, indices)
    b = macro_aulc(baseline, indices)
    if a is None or b is None:
        return None
    return float(a - b)


# --------------------------------------------------------------------------
# 4-6: curve shape
# --------------------------------------------------------------------------
def per_family_curve(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for fid, rs in sorted(group_by_family(records).items()):
        curve: dict[int, float] = {}
        indices = sorted({k for r in rs for k in record_R(r)})
        for k in indices:
            vals = [record_R(r)[k] for r in rs if k in record_R(r)]
            if vals:
                curve[k] = float(sum(vals) / len(vals))
        out[fid] = curve
    return out


def macro_curve(records: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    fam = per_family_curve(records)
    indices = sorted({k for c in fam.values() for k in c})
    out: dict[int, float] = {}
    for k in indices:
        vals = [c[k] for c in fam.values() if k in c]
        if vals:
            out[k] = float(sum(vals) / len(vals))
    return out


def r_at(records: Sequence[Mapping[str, Any]], k: int) -> float | None:
    """Macro ``R_k``."""
    return macro_curve(records).get(int(k))


def r_0(records: Sequence[Mapping[str, Any]]) -> float | None:
    """METRIC 4 — macro R_0."""
    return r_at(records, 0)


def r_K(records: Sequence[Mapping[str, Any]], K: int | None = None) -> float | None:
    """METRIC 5 — macro R_K (K = the largest observed interaction index)."""
    curve = macro_curve(records)
    if not curve:
        return None
    key = max(curve) if K is None else int(K)
    return curve.get(key)


def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Ordinary least squares slope of ``ys`` on ``xs`` (numpy only)."""
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    if x.size < 2 or y.size != x.size:
        return None
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom <= 0.0:
        return None
    return float((xc * (y - y.mean())).sum() / denom)


def ols_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    """Return ``(slope, intercept)`` of the OLS fit."""
    slope = ols_slope(xs, ys)
    if slope is None:
        return None
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    return slope, float(y.mean() - slope * x.mean())


def per_family_slopes(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for fid, curve in per_family_curve(records).items():
        if len(curve) >= 2:
            s = ols_slope(sorted(curve), [curve[k] for k in sorted(curve)])
            if s is not None:
                out[fid] = s
    return out


def improvement_slope(records: Sequence[Mapping[str, Any]]) -> float | None:
    """METRIC 6 — OLS slope of the MACRO learning curve over interaction index."""
    curve = macro_curve(records)
    if len(curve) < 2:
        return None
    ks = sorted(curve)
    return ols_slope(ks, [curve[k] for k in ks])


def slope_conventions_agree(records: Sequence[Mapping[str, Any]],
                            tol: float = 1e-9) -> bool:
    """True when macro-curve slope == mean of per-family slopes (equal grids)."""
    macro = improvement_slope(records)
    fam = per_family_slopes(records)
    if macro is None or not fam:
        return False
    return abs(macro - float(np.mean(list(fam.values())))) <= tol


# --------------------------------------------------------------------------
# 7: interactions to threshold
# --------------------------------------------------------------------------
def interactions_to_threshold(records: Sequence[Mapping[str, Any]],
                              threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """METRIC 7 — first interaction index whose family success >= threshold.

    ``None`` where the family never reaches the threshold ("where
    identifiable"); the macro summary averages only the identifiable families
    and reports how many were identifiable, never silently dropping the rest.
    """
    per_family: dict[str, int | None] = {}
    for fid, curve in per_family_curve(records).items():
        hit = None
        for k in sorted(curve):
            if curve[k] >= float(threshold):
                hit = int(k)
                break
        per_family[fid] = hit
    identifiable = [v for v in per_family.values() if v is not None]
    return {
        "threshold": float(threshold),
        "per_family": per_family,
        "macro_mean": (float(np.mean(identifiable)) if identifiable else None),
        "n_identifiable": len(identifiable),
        "n_families": len(per_family),
    }


# --------------------------------------------------------------------------
# 8-9: transfer
# --------------------------------------------------------------------------
def related_task_transfer(records: Sequence[Mapping[str, Any]]) -> float | None:
    """METRIC 8 — macro R_4 (fresh instance of the same latent rule)."""
    return r_at(records, RELATED_INDEX)


def whole_family_transfer(records: Sequence[Mapping[str, Any]],
                          trained_family_ids: Iterable[str]) -> float | None:
    """METRIC 9 — macro-AULC restricted to families whose GENERATOR was unseen."""
    trained = {str(f) for f in trained_family_ids}
    unseen = [r for r in records if record_family(r) not in trained]
    return macro_aulc(unseen)


# --------------------------------------------------------------------------
# 10: context-reset persistence
# --------------------------------------------------------------------------
def _post_reset_mean(record: Mapping[str, Any], reset_from_index: int) -> float | None:
    R = record_R(record)
    keys = [k for k in sorted(R) if k >= int(reset_from_index)]
    if not keys:
        return None
    return float(sum(R[k] for k in keys) / len(keys))


def context_reset_persistence(reset_records: Sequence[Mapping[str, Any]],
                              history_records: Sequence[Mapping[str, Any]],
                              reset_from_index: int = 4) -> dict[str, Any]:
    """METRIC 10 — retained fraction of the in-context gain.

    Per family:  ``(post_reset - R_0) / (post_history - R_0)`` where ``R_0`` is
    the pre-learning baseline measured on the matched history arm and ``post_*``
    is the mean success over interaction indices >= ``reset_from_index``.  The
    ratio is ``None`` when the history arm shows no in-context gain (a
    denominator <= 0 makes "retained fraction" undefined; it is reported as
    such rather than clipped).
    """
    hist_by_family = group_by_family(history_records)
    reset_by_family = group_by_family(reset_records)
    per_family: dict[str, Any] = {}
    for fid in sorted(set(hist_by_family) | set(reset_by_family)):
        hist = hist_by_family.get(fid, [])
        rst = reset_by_family.get(fid, [])
        base = _mean(record_R(r).get(0) for r in hist)
        post_h = _mean(_post_reset_mean(r, reset_from_index) for r in hist)
        post_r = _mean(_post_reset_mean(r, reset_from_index) for r in rst)
        gain_h = None if (base is None or post_h is None) else post_h - base
        gain_r = None if (base is None or post_r is None) else post_r - base
        ratio = None
        if gain_h is not None and gain_r is not None and gain_h > MIN_DENOMINATOR:
            ratio = float(gain_r / gain_h)
        per_family[fid] = {
            "baseline_R0": base,
            "post_history": post_h,
            "post_reset": post_r,
            "gain_history": gain_h,
            "gain_reset": gain_r,
            "retained_fraction": ratio,
        }
    ratios = [v["retained_fraction"] for v in per_family.values()
              if v["retained_fraction"] is not None]
    return {
        "reset_from_index": int(reset_from_index),
        "per_family": per_family,
        "macro_retained_fraction": (float(np.mean(ratios)) if ratios else None),
        "n_families_defined": len(ratios),
        "n_families": len(per_family),
        "macro_gain_history": _mean(v["gain_history"] for v in per_family.values()),
        "macro_gain_reset": _mean(v["gain_reset"] for v in per_family.values()),
    }


# --------------------------------------------------------------------------
# 11: A -> B -> A retention / interference
# --------------------------------------------------------------------------
def retention_interference(chain_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """METRIC 11 — family-balanced retention ratio / interference cost.

    Input: the per-chain dicts produced by ``interference.run_interference_eval``
    (keys ``family_a``, ``aulc_a1``, ``aulc_a2``, ``aulc_b``,
    ``recovery_interactions``).
    """
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for c in chain_summaries:
        by_family.setdefault(str(c.get("family_a")), []).append(c)
    per_family: dict[str, Any] = {}
    for fid, cs in sorted(by_family.items()):
        a1 = _mean(c.get("aulc_a1") for c in cs)
        a2 = _mean(c.get("aulc_a2") for c in cs)
        ratio = None
        if a1 is not None and a2 is not None and a1 > MIN_DENOMINATOR:
            ratio = float(a2 / a1)
        per_family[fid] = {
            "n_chains": len(cs),
            "aulc_a1": a1,
            "aulc_a2": a2,
            "aulc_b": _mean(c.get("aulc_b") for c in cs),
            "retention_ratio": ratio,
            "interference_cost": (None if (a1 is None or a2 is None) else float(a1 - a2)),
            "recovery_interactions": _mean(
                c.get("recovery_interactions") for c in cs),
        }
    ratios = [v["retention_ratio"] for v in per_family.values()
              if v["retention_ratio"] is not None]
    costs = [v["interference_cost"] for v in per_family.values()
             if v["interference_cost"] is not None]
    return {
        "per_family": per_family,
        "macro_retention_ratio": (float(np.mean(ratios)) if ratios else None),
        "macro_interference_cost": (float(np.mean(costs)) if costs else None),
        "n_families": len(per_family),
        "n_chains": len(chain_summaries),
    }


# --------------------------------------------------------------------------
# 12-13: poison / remap robustness
# --------------------------------------------------------------------------
#: The condition ids are the DATA LAYER's frozen strings
#: (``ecology.poison.POISON_CONDITIONS``), imported when available so a
#: consumer can never drift from the producer.  Contract section 9 spells the
#: untouched condition "correct-informative" and metric 12 calls the same thing
#: "clean"; the data layer's id is ``clean`` and both spellings alias onto it.
def _producer_conditions() -> tuple[str, ...]:
    try:
        from foundation_learner.ecology.poison import POISON_CONDITIONS as _P  # type: ignore
        return tuple(str(c) for c in _P)
    except Exception:
        return ("clean", "correct-redundant", "irrelevant",
                "partially-misleading", "corrupted")


CLEAN_CONDITION = "clean"
POISON_CONDITIONS: tuple[str, ...] = _producer_conditions()
POISON_CONDITION_ALIASES: dict[str, str] = {
    "correct_informative": "clean",
    "correct-informative": "clean",
    "correct_redundant": "correct-redundant",
    "partially_misleading": "partially-misleading",
}


def canonical_poison_condition(name: str) -> str:
    key = str(name).strip()
    key = POISON_CONDITION_ALIASES.get(key, key)
    if key not in POISON_CONDITIONS:
        raise ValueError(
            f"unknown poison condition {name!r}; frozen set {POISON_CONDITIONS}")
    return key


def poison_robustness(records_by_condition: Mapping[str, Sequence[Mapping[str, Any]]]
                      ) -> dict[str, Any]:
    """METRIC 12 — macro-AULC gap of each poison condition vs the clean one."""
    aulc = {canonical_poison_condition(c): macro_aulc(rs)
            for c, rs in records_by_condition.items()}
    clean = aulc.get(CLEAN_CONDITION)
    gaps = {c: (None if (clean is None or v is None) else float(clean - v))
            for c, v in aulc.items() if c != CLEAN_CONDITION}
    return {
        "clean_condition": CLEAN_CONDITION,
        "macro_aulc": aulc,
        "gap_vs_clean": gaps,
        "corrupted_gap": gaps.get("corrupted"),
    }


def remap_robustness(canonical_records: Sequence[Mapping[str, Any]],
                     remapped_records_by_variant: Mapping[str, Sequence[Mapping[str, Any]]]
                     ) -> dict[str, Any]:
    """METRIC 13 — macro-AULC gap between canonical and remapped surfaces."""
    base = macro_aulc(canonical_records)
    per_variant = {str(k): macro_aulc(v) for k, v in remapped_records_by_variant.items()}
    gaps = {k: (None if (base is None or v is None) else float(base - v))
            for k, v in per_variant.items()}
    defined = [g for g in gaps.values() if g is not None]
    return {
        "canonical_macro_aulc": base,
        "remapped_macro_aulc": per_variant,
        "gap_vs_canonical": gaps,
        "macro_gap": (float(np.mean(defined)) if defined else None),
    }


# --------------------------------------------------------------------------
# 14: value-head ranking / calibration / regret
# --------------------------------------------------------------------------
def value_ranking_metrics(predictions: Sequence[float], targets: Sequence[float],
                          groups: Sequence[Any] | None = None) -> dict[str, Any]:
    """METRIC 14 — delegates to ``analysis.stats`` (numpy-only rank metrics)."""
    from foundation_learner.analysis.stats import value_head_metrics
    return value_head_metrics(predictions, targets, groups)


# --------------------------------------------------------------------------
# frozen registry
# --------------------------------------------------------------------------
METRICS_V0: tuple[dict[str, str], ...] = (
    {"id": "1", "name": "macro_aulc", "fn": "macro_aulc"},
    {"id": "2", "name": "delta_aulc_vs_fl1", "fn": "delta_aulc"},
    {"id": "3", "name": "delta_aulc_vs_fl2", "fn": "delta_aulc"},
    {"id": "4", "name": "r_0", "fn": "r_0"},
    {"id": "5", "name": "r_K", "fn": "r_K"},
    {"id": "6", "name": "improvement_slope", "fn": "improvement_slope"},
    {"id": "7", "name": "interactions_to_threshold", "fn": "interactions_to_threshold"},
    {"id": "8", "name": "related_task_transfer", "fn": "related_task_transfer"},
    {"id": "9", "name": "whole_family_transfer", "fn": "whole_family_transfer"},
    {"id": "10", "name": "context_reset_persistence", "fn": "context_reset_persistence"},
    {"id": "11", "name": "retention_interference", "fn": "retention_interference"},
    {"id": "12", "name": "poison_robustness", "fn": "poison_robustness"},
    {"id": "13", "name": "remap_robustness", "fn": "remap_robustness"},
    {"id": "14", "name": "value_ranking_metrics", "fn": "value_ranking_metrics"},
)


def metric_function(name: str) -> Callable[..., Any]:
    for entry in METRICS_V0:
        if entry["name"] == name:
            return globals()[entry["fn"]]
    raise KeyError(f"unknown frozen metric {name!r}")


def summarize(records: Sequence[Mapping[str, Any]],
              trained_family_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Convenience bundle of the record-only metrics (1, 4, 5, 6, 7, 8, 9)."""
    out: dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "n_episodes": len(records),
        "n_families": len(group_by_family(records)),
        "macro_aulc": macro_aulc(records),
        "per_family_aulc": per_family_aulc(records),
        "macro_curve": {str(k): v for k, v in macro_curve(records).items()},
        "r_0": r_0(records),
        "r_K": r_K(records),
        "improvement_slope": improvement_slope(records),
        "per_family_slopes": per_family_slopes(records),
        "interactions_to_threshold": interactions_to_threshold(records),
        "related_task_transfer": related_task_transfer(records),
    }
    if trained_family_ids is not None:
        out["whole_family_transfer"] = whole_family_transfer(records, trained_family_ids)
    return out


__all__ = [
    "METRICS_SCHEMA",
    "METRICS_V0",
    "DEFAULT_THRESHOLD",
    "DEFAULT_INDICES",
    "CLEAN_CONDITION",
    "POISON_CONDITIONS",
    "canonical_poison_condition",
    "record_family",
    "record_R",
    "record_aulc",
    "group_by_family",
    "macro_mean",
    "per_family_mean",
    "macro_aulc",
    "per_family_aulc",
    "delta_aulc",
    "per_family_curve",
    "macro_curve",
    "r_at",
    "r_0",
    "r_K",
    "ols_slope",
    "ols_fit",
    "per_family_slopes",
    "improvement_slope",
    "slope_conventions_agree",
    "interactions_to_threshold",
    "related_task_transfer",
    "whole_family_transfer",
    "context_reset_persistence",
    "retention_interference",
    "poison_robustness",
    "remap_robustness",
    "value_ranking_metrics",
    "metric_function",
    "summarize",
]
