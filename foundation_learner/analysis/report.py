"""Plain-markdown + JSON reporting over episode records (contract section 14).

Local, post-transfer, numpy-only, no GPU, no network.  The report states what
was measured, with family-clustered intervals, and states nulls as nulls: this
module contains no success narrative, no adjectives, and no interpretation
beyond the frozen metric definitions.  A missing quantity is printed as
``n/a`` rather than dropped.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Sequence

from foundation_learner.evaluation import _atomic_write_text, write_json_document
from . import stats as _stats

ANALYSIS_REPORT_SCHEMA = "flb200.analysis_report.v1"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(entry: Mapping[str, Any] | None, digits: int = 4) -> str:
    if not entry:
        return "n/a"
    return (f"{_fmt(entry.get('estimate'), digits)} "
            f"[{_fmt(entry.get('ci_low'), digits)}, {_fmt(entry.get('ci_high'), digits)}]")


def build_report(
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    title: str = "Foundation Learner V0 - results",
    baselines: Sequence[str] = (),
    trained_family_ids: Iterable[str] | None = None,
    n_replicates: int = _stats.DEFAULT_REPLICATES,
    seed: int = 0,
    alpha: float = _stats.DEFAULT_ALPHA,
    context_reset: Mapping[str, Any] | None = None,
    interference: Mapping[str, Any] | None = None,
    poison: Mapping[str, Any] | None = None,
    remap: Mapping[str, Any] | None = None,
    value_head: Mapping[str, Any] | None = None,
    family_holdout: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble the machine-readable report from raw episode records."""
    arm_summaries: dict[str, Any] = {}
    for label, records in arms.items():
        arm_summaries[label] = _stats.summarize_arm(
            records, label=label, n_replicates=n_replicates, seed=seed,
            alpha=alpha, trained_family_ids=trained_family_ids)

    comparisons: dict[str, Any] = {}
    for label, records in arms.items():
        for base in baselines:
            if base == label or base not in arms:
                continue
            try:
                comparisons[f"{label}_vs_{base}"] = _stats.arm_comparison(
                    records, arms[base], label_a=label, label_b=base,
                    n_replicates=n_replicates, seed=seed, alpha=alpha)
            except _stats.ClusteredSampleError as exc:
                comparisons[f"{label}_vs_{base}"] = {
                    "schema": _stats.ARM_COMPARISON_SCHEMA,
                    "label_a": label, "label_b": base,
                    "unavailable_reason": str(exc),
                }
    return {
        "schema": ANALYSIS_REPORT_SCHEMA,
        "title": title,
        "n_replicates": int(n_replicates),
        "seed": int(seed),
        "alpha": float(alpha),
        "arms": arm_summaries,
        "comparisons": comparisons,
        "context_reset": context_reset,
        "interference": interference,
        "poison": poison,
        "remap": remap,
        "value_head": value_head,
        "family_holdout": family_holdout,
        "notes": list(notes),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# {report.get('title', 'Foundation Learner V0 - results')}")
    add("")
    add(f"Clustered bootstrap: {report.get('n_replicates')} replicates, "
        f"seed {report.get('seed')}, alpha {report.get('alpha')}. "
        "Families resampled with replacement, then episodes within family; "
        "paired comparisons share one resample plan.")
    add("")

    add("## Arms")
    add("")
    add("| arm | families | episodes | macro-AULC [95% CI] | R_0 | R_K | slope |")
    add("|---|---|---|---|---|---|---|")
    for label, summary in sorted(report.get("arms", {}).items()):
        m = summary.get("metrics", {})
        add(f"| {label} | {_fmt(m.get('n_families'))} | {_fmt(m.get('n_episodes'))} | "
            f"{_ci(summary.get('macro_aulc_ci'))} | {_fmt(m.get('r_0'))} | "
            f"{_fmt(m.get('r_K'))} | {_fmt(m.get('improvement_slope'))} |")
    add("")

    add("## Learning curves (macro R_k)")
    add("")
    for label, summary in sorted(report.get("arms", {}).items()):
        curve = summary.get("learning_curve", {})
        add(f"### {label}")
        add("")
        add("| k | macro R_k [95% CI] |")
        add("|---|---|")
        for k in curve.get("indices", []):
            add(f"| {k} | {_ci(curve.get('points', {}).get(str(k)))} |")
        add("")

    comparisons = report.get("comparisons") or {}
    if comparisons:
        add("## Paired comparisons (shared resample plan)")
        add("")
        add("| comparison | delta macro-AULC [95% CI] | excludes 0 |")
        add("|---|---|---|")
        for name, comp in sorted(comparisons.items()):
            if comp.get("unavailable_reason"):
                add(f"| {name} | n/a ({comp['unavailable_reason']}) | n/a |")
            else:
                add(f"| {name} | {_ci(comp)} | {_fmt(comp.get('excludes_zero'))} |")
        add("")

    fh = report.get("family_holdout")
    if fh:
        add("## Whole-family generalization")
        add("")
        add("| scope | families | episodes | macro-AULC | R_0 | R_K |")
        add("|---|---|---|---|---|---|")
        for scope in ("unseen_instance", "unseen_family"):
            sec = fh.get(scope, {})
            add(f"| {scope} | {_fmt(sec.get('n_families'))} | "
                f"{_fmt(sec.get('n_episodes'))} | {_fmt(sec.get('macro_aulc'))} | "
                f"{_fmt(sec.get('r_0'))} | {_fmt(sec.get('r_K'))} |")
        add("")
        add("UNSEEN-INSTANCE is generalization to fresh instances of a trained "
            "generator; it is not transferable learning and is reported separately.")
        add("")

    cr = report.get("context_reset")
    if cr:
        add("## Context-reset persistence")
        add("")
        summary = cr.get("summary", cr)
        persistence = summary.get("persistence") or {}
        add(f"- reset macro-AULC: {_fmt(summary.get('reset_macro_aulc'))}")
        add(f"- history macro-AULC: {_fmt(summary.get('history_macro_aulc'))}")
        add(f"- macro retained fraction of in-context gain: "
            f"{_fmt(persistence.get('macro_retained_fraction'))} "
            f"({_fmt(persistence.get('n_families_defined'))} of "
            f"{_fmt(persistence.get('n_families'))} families defined)")
        add("")

    inter = report.get("interference")
    if inter:
        add("## A -> B -> A retention and interference")
        add("")
        summary = inter.get("summary", inter)
        add(f"- macro retention ratio: {_fmt(summary.get('macro_retention_ratio'))}")
        add(f"- macro interference cost: {_fmt(summary.get('macro_interference_cost'))}")
        add(f"- chains: {_fmt(summary.get('n_chains'))} over "
            f"{_fmt(summary.get('n_families'))} A-families")
        add("")

    poison = report.get("poison")
    if poison:
        add("## Poison robustness")
        add("")
        rob = poison.get("robustness", poison)
        add("| condition | macro-AULC | gap vs clean |")
        add("|---|---|---|")
        for cond, value in sorted((rob.get("macro_aulc") or {}).items()):
            add(f"| {cond} | {_fmt(value)} | "
                f"{_fmt((rob.get('gap_vs_clean') or {}).get(cond))} |")
        add("")

    remap = report.get("remap")
    if remap:
        add("## Surface-remap robustness")
        add("")
        rob = remap.get("robustness", remap)
        add(f"- canonical macro-AULC: {_fmt(rob.get('canonical_macro_aulc'))}")
        add("")
        add("| variant | macro-AULC | gap vs canonical |")
        add("|---|---|---|")
        for variant, value in sorted((rob.get("remapped_macro_aulc") or {}).items()):
            add(f"| {variant} | {_fmt(value)} | "
                f"{_fmt((rob.get('gap_vs_canonical') or {}).get(variant))} |")
        add("")

    vh = report.get("value_head")
    if vh:
        add("## Value head (FL4/FL6)")
        add("")
        add(f"- Spearman: {_fmt(vh.get('spearman'))}")
        add(f"- pairwise ranking accuracy: "
            f"{_fmt((vh.get('pairwise') or {}).get('accuracy'))} over "
            f"{_fmt((vh.get('pairwise') or {}).get('n_pairs'))} within-episode pairs")
        cal = vh.get("calibration") or {}
        add(f"- calibration slope / intercept: {_fmt(cal.get('slope'))} / "
            f"{_fmt(cal.get('intercept'))}")
        reg = vh.get("regret") or {}
        add(f"- top-1 regret: {_fmt(reg.get('top1_regret'))} "
            f"(random {_fmt(reg.get('top1_regret_random'))}, oracle 0)")
        add("")

    notes = report.get("notes") or []
    if notes:
        add("## Notes")
        add("")
        for note in notes:
            add(f"- {note}")
        add("")
    return "\n".join(lines) + "\n"


def write_report(out_dir: str, report: Mapping[str, Any], *,
                 stem: str = "analysis_report") -> dict[str, str]:
    """Write ``<stem>.json`` and ``<stem>.md`` atomically; returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = write_json_document(os.path.join(out_dir, f"{stem}.json"), dict(report))
    md_path = _atomic_write_text(os.path.join(out_dir, f"{stem}.md"),
                                 render_markdown(report))
    return {"json": json_path, "markdown": md_path}


def report_from_record_files(paths: Mapping[str, str], **kwargs: Any) -> dict[str, Any]:
    """Build a report from ``arm_label -> episode-records JSONL path``."""
    from foundation_learner.evaluation import read_jsonl_records

    arms = {label: read_jsonl_records(path) for label, path in paths.items()}
    return build_report(arms, **kwargs)


__all__ = [
    "ANALYSIS_REPORT_SCHEMA",
    "build_report",
    "render_markdown",
    "write_report",
    "report_from_record_files",
]
