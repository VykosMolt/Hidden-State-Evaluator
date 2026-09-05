"""Report fit-size transport norms without relabelling scatter as a norm.

For each source location this fits

    ||J_bar_n||_F^2 / d = mu_squared + sigma_squared / n

over the retained nested fit sizes.  ``sigma`` is an RMS scatter parameter,
not a directly measured mean single-prompt Jacobian norm.  Negative moments are
reported as invalid; they are never silently clipped into an estimate.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from ouro_jlens.evidence import atomic_write_json, file_record
from ouro_jlens.fit_lens import IntegrityError, identity_projection, validate_sidecar

N_LAYER = 48


def estimate_from_squared_norms(sample_sizes: np.ndarray, squared_norms: np.ndarray) -> dict:
    raw_sizes = np.asarray(sample_sizes)
    if (raw_sizes.ndim != 1 or not np.issubdtype(raw_sizes.dtype, np.integer)
            or np.issubdtype(raw_sizes.dtype, np.bool_)):
        raise ValueError("fit sizes must be an exact one-dimensional integer array")
    sample_sizes = raw_sizes.astype(np.int64, copy=False)
    squared_norms = np.asarray(squared_norms, dtype=float)
    if squared_norms.ndim != 2 or squared_norms.shape[0] != len(sample_sizes):
        raise ValueError("squared_norms must have shape [n_fit_sizes, n_sources]")
    if len(np.unique(sample_sizes)) != len(sample_sizes) or len(sample_sizes) < 3 \
            or np.any(sample_sizes <= 0):
        raise ValueError("at least three distinct positive fit sizes are required")
    if not np.isfinite(squared_norms).all() or np.any(squared_norms < 0):
        raise ValueError("squared norms must be finite and nonnegative")
    design = np.column_stack([np.ones(len(sample_sizes)), 1.0 / sample_sizes])
    mu2, sigma2 = np.linalg.lstsq(design, squared_norms, rcond=None)[0]
    fitted = design @ np.stack([mu2, sigma2])
    residual = squared_norms - fitted
    return {"mu_squared": mu2, "sigma_squared": sigma2, "residual": residual}


def _finite_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return None if not len(values) else float(values.mean())


def summarize(sample_sizes: np.ndarray, squared_norms: np.ndarray) -> dict:
    estimate = estimate_from_squared_norms(sample_sizes, squared_norms)
    mu2, sigma2 = estimate["mu_squared"], estimate["sigma_squared"]
    rows = []
    for loop in range(int(np.ceil(len(mu2) / N_LAYER))):
        sl = slice(loop * N_LAYER, min((loop + 1) * N_LAYER, len(mu2)))
        m, s = mu2[sl], sigma2[sl]
        valid_mu, valid_sigma = m >= 0, s >= 0
        valid_model = valid_mu & valid_sigma
        total = m + s
        valid_total = total >= 0
        rows.append({
            "loop": loop + 1,
            "n_source_locations": int(len(m)),
            "raw_fitted_map_norm_by_n": {
                str(int(n)): round(float(np.sqrt(squared_norms[i, sl]).mean()), 6)
                for i, n in enumerate(sample_sizes)
            },
            "mu_norm_mean_valid": _finite_mean(np.where(valid_mu, np.sqrt(np.maximum(m, 0)), np.nan)),
            "sigma_rms_scatter_mean_valid": _finite_mean(
                np.where(valid_sigma, np.sqrt(np.maximum(s, 0)), np.nan)
            ),
            "modeled_single_prompt_rms_mean_valid": _finite_mean(
                np.where(valid_model, np.sqrt(np.maximum(m + s, 0)), np.nan)
            ),
            "modeled_total_rms_mean_where_sum_nonnegative": _finite_mean(
                np.where(valid_total, np.sqrt(np.maximum(total, 0)), np.nan)
            ),
            "negative_mu_squared": int((~valid_mu).sum()),
            "negative_sigma_squared": int((~valid_sigma).sum()),
            "negative_total_squared": int((~valid_total).sum()),
            "valid_moment_locations": int(valid_model.sum()),
            # Retained only to make the historical 0.715 value auditable.  It is
            # explicitly labelled as clipped and is not an accepted estimate.
            "legacy_zero_clipped_sigma_mean": float(np.sqrt(np.maximum(s, 0)).mean()),
        })
    return {
        "sample_sizes": sample_sizes.astype(int).tolist(),
        "n_sources": int(squared_norms.shape[1]),
        "model": "squared_norm = mu_squared + sigma_squared / n",
        "caveat": (
            "sigma is modeled RMS prompt-to-prompt scatter. It is not a directly measured "
            "single-prompt Jacobian norm. Nested prefixes are correlated and provide no fixed-n replicate variance."
        ),
        "negative_mu_squared_total": int((mu2 < 0).sum()),
        "negative_sigma_squared_total": int((sigma2 < 0).sum()),
        "rmse": float(np.sqrt(np.mean(estimate["residual"] ** 2))),
        "per_loop": rows,
        "per_source": {
            # Retain the sufficient statistics consumed by the independent
            # verifier.  Without these arrays, a verifier can only compare
            # against hard-coded historical outputs rather than recomputing
            # the regression for a fresh authenticated fit.
            "squared_norms_by_n": {
                str(int(n)): squared_norms[index].tolist()
                for index, n in enumerate(sample_sizes)
            },
            "mu_squared": mu2.tolist(),
            "sigma_squared": sigma2.tolist(),
            "valid": ((mu2 >= 0) & (sigma2 >= 0)).tolist(),
        },
    }


def lens_squared_norms(specs: list[tuple[int, Path]]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    from jlens import JacobianLens

    sizes, values, records = [], [], []
    if len(specs) < 3:
        raise ValueError("at least three fit-size lenses are required")
    declared_sizes = [spec[0] for spec in specs]
    declared_paths = [Path(spec[1]) for spec in specs]
    if len(set(declared_sizes)) != len(declared_sizes):
        raise ValueError("declared fit sizes must be distinct")
    if len({str(path.absolute()) for path in declared_paths}) != len(declared_paths):
        raise ValueError("each fit size must name a distinct lens path")
    expected_sources = None
    expected_identity = None
    for declared_n, path in sorted(specs):
        if not isinstance(declared_n, int) or isinstance(declared_n, bool) or declared_n <= 0:
            raise ValueError("declared fit sizes must be exact positive integers")
        try:
            preview = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"{path}: current fit sidecar is unavailable") from exc
        kind = preview.get("kind") if isinstance(preview, dict) else None
        if kind not in {"fit", "merged"}:
            raise IntegrityError(f"{path}: current fit sidecar kind is invalid")
        metadata = validate_sidecar(path, kind=kind)
        lens = JacobianLens.load(str(path))
        sealed_count = metadata.get("end", 0) - metadata.get("start", 0)
        if (not isinstance(lens.n_prompts, int) or isinstance(lens.n_prompts, bool)
                or lens.n_prompts != declared_n or metadata.get("n_prompts") != declared_n
                or metadata.get("n_fitted") != declared_n or sealed_count != declared_n):
            raise ValueError(f"{path}: declared n={declared_n}, lens contains {lens.n_prompts}")
        if (not isinstance(lens.d_model, int) or isinstance(lens.d_model, bool)
                or lens.d_model <= 0 or lens.d_model != metadata.get("d_model")):
            raise IntegrityError(f"{path}: binary model width disagrees with its sidecar")
        current_identity = identity_projection(metadata)
        if expected_identity is None:
            expected_identity = current_identity
        elif current_identity != expected_identity:
            raise IntegrityError(f"{path}: model/source/fitting identity differs across fit sizes")
        sources = list(lens.source_layers)
        if expected_sources is None:
            expected_sources = sources
        if sources != expected_sources or sources != list(range(len(sources))):
            raise ValueError(f"{path}: source layers are not the same exact contiguous prefix")
        squared_values = []
        for source in sources:
            matrix = lens.jacobians[source]
            if tuple(matrix.shape) != (lens.d_model, lens.d_model) \
                    or not torch.isfinite(matrix).all():
                raise IntegrityError(f"{path}: Jacobian at source {source} is malformed or non-finite")
            squared_values.append(
                float(matrix.float().norm().square().item() / lens.d_model)
            )
        squared = np.asarray(squared_values)
        sizes.append(declared_n)
        values.append(squared)
        record = file_record(path)
        record["path"] = f"lens_n{declared_n}"
        sidecar_record = file_record(path.with_suffix(".json"))
        sidecar_record["path"] = f"lens_n{declared_n}_sidecar"
        records.append({"n_prompts": declared_n, "binary": record, "sidecar": sidecar_record})
        del lens
        gc.collect()
    return np.asarray(sizes), np.stack(values), records


def build_report(specs: list[tuple[int, Path]]) -> dict:
    sizes, norms, records = lens_squared_norms(specs)
    report = summarize(sizes, norms)
    report.update({
        "schema_version": 1,
        "status": "MODELED_ASSOCIATION_ONLY",
        "inputs": records,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", action="append", required=True, help="N=path; repeat for each fit size")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    specs = []
    for spec in args.lens:
        n, path = spec.split("=", 1)
        specs.append((int(n), Path(path)))
    report = build_report(specs)
    atomic_write_json(Path(args.out), report)
    print(json.dumps({k: report[k] for k in ("status", "sample_sizes", "per_loop")}, indent=2))


if __name__ == "__main__":
    main()
