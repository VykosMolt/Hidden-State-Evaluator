"""Two-sample diagnostic for average-Jacobian convergence.

Given two lenses fitted on DISJOINT prompt sets of different sizes, the expected
squared Frobenius norm of a mean of n per-prompt Jacobians is
||mu||^2 + sigma^2 / n, so two sample sizes give the converged mean norm ||mu||
and the per-prompt scatter sigma per source. Also reports cosine(J1, J2).

The disjointness claim is load-bearing.  Both lenses therefore need current,
hash-valid fit sidecars which bind them to non-overlapping slices of the same
prompt corpus.  Legacy lenses without that evidence are rejected.

  python src/ouro_jlens/lens_convergence.py lensA.pt lensB.pt --out report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jlens import JacobianLens
from ouro_jlens.evidence import atomic_write_json, file_record
from ouro_jlens.fit_lens import IntegrityError, validate_sidecar

N_LAYER = 48


def _validated_fit(path: str) -> tuple[object, dict]:
    sidecar = Path(path).with_suffix(".json")
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot establish fit identity for {path}: {exc}") from exc
    kind = raw.get("kind")
    if kind not in {"fit", "merged"}:
        raise IntegrityError(f"{sidecar}: expected a current fit/merged sidecar")
    meta = validate_sidecar(path, kind=kind)
    return JacobianLens.load(path), meta


def _require_disjoint(a: dict, b: dict) -> dict:
    if a.get("prompt_file_sha256") != b.get("prompt_file_sha256"):
        raise IntegrityError("lenses are not bound to the same prompt corpus")
    a_range = (a.get("start"), a.get("end"))
    b_range = (b.get("start"), b.get("end"))
    if not all(isinstance(value, int) and not isinstance(value, bool)
               for value in (*a_range, *b_range)):
        raise IntegrityError("sidecars do not contain exact integer prompt ranges")
    if max(a_range[0], b_range[0]) < min(a_range[1], b_range[1]):
        raise IntegrityError(f"prompt slices overlap: {a_range} and {b_range}")
    return {
        "prompt_file_sha256": a["prompt_file_sha256"],
        "slice_a": {"start": a_range[0], "end": a_range[1],
                    "sha256": a.get("prompt_slice_sha256")},
        "slice_b": {"start": b_range[0], "end": b_range[1],
                    "sha256": b.get("prompt_slice_sha256")},
    }


def _finite(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def build(path_a: str, path_b: str) -> dict:
    a, meta_a = _validated_fit(path_a)
    b, meta_b = _validated_fit(path_b)
    disjoint = _require_disjoint(meta_a, meta_b)
    n1, n2 = a.n_prompts, b.n_prompts
    if list(a.source_layers) != list(b.source_layers):
        raise ValueError("lenses have different source layers")
    if n1 == n2:
        raise ValueError("two-point moment fit requires different prompt counts")
    if a.d_model != b.d_model:
        raise ValueError("lenses have different model widths")
    d = a.d_model
    rows = []
    clipped_sigma = clipped_mu = 0
    for v in a.source_layers:
        A, B = a.jacobians[v], b.jacobians[v]
        sq1, sq2 = (A.norm() ** 2 / d).item(), (B.norm() ** 2 / d).item()
        sigma2 = (sq1 - sq2) / (1 / n1 - 1 / n2)
        mu2 = sq2 - sigma2 / n2
        # The two-point estimator is a difference of noisy norms and goes negative often.
        # Clipping is not free: count it, because a heavily clipped column is not an estimate.
        clipped_sigma += sigma2 < 0
        clipped_mu += mu2 < 0
        denominator = A.norm() * B.norm()
        cos = ((A.flatten() @ B.flatten()) / denominator).item() if denominator.item() else np.nan
        rows.append((np.sqrt(sq1), np.sqrt(sq2), np.sqrt(mu2) if mu2 >= 0 else np.nan,
                     np.sqrt(sigma2) if sigma2 >= 0 else np.nan, cos))
    rows = np.array(rows)
    n_ut = int(np.ceil(len(rows) / N_LAYER))
    padded = np.full((n_ut * N_LAYER, rows.shape[1]), np.nan)
    padded[: len(rows)] = rows
    per_loop = np.nanmean(padded.reshape(n_ut, N_LAYER, -1), axis=1)
    report = {
        "schema_version": 1,
        "status": "INCONCLUSIVE_TWO_POINT_DIAGNOSTIC",
        "caveat": "sigma is modeled RMS scatter, not a direct single-prompt norm; invalid moments are NaN, not clipped",
        "n1": n1, "n2": n2,
        "inputs": [file_record(Path(path_a)), file_record(Path(path_b))],
        "disjoint_prompt_evidence": disjoint,
        "negative_sigma_squared": clipped_sigma,
        "negative_mu_squared": clipped_mu,
        "per_loop": [],
    }
    print(f"INVALID MOMENTS: sigma^2 at {clipped_sigma}/{len(rows)} layers, mu^2 at {clipped_mu}/{len(rows)}. "
          f"Invalid moments are omitted from aggregates; treat raw ||J_n|| as primary.")
    print("loop  ||J_n1||  ||J_n2||  ||mu|| (converged)  sigma (per-prompt scatter)  cos(J_n1,J_n2)  sigma/(||mu||*sqrt(1000))")
    for u, (s1, s2, mu, sig, cos) in enumerate(per_loop):
        ratio = sig / max(mu, 1e-9) / np.sqrt(1000) if np.isfinite(sig) and np.isfinite(mu) else np.nan
        print(f"{u+1:>4}  {s1:8.3f}  {s2:8.3f}  {mu:18.3f}  {sig:26.3f}  {cos:14.3f}  {ratio:.3f}")
        report["per_loop"].append({"loop": u + 1, "norm_n1": _finite(s1),
                                   "norm_n2": _finite(s2),
                                   "mu_norm_valid_only": _finite(mu),
                                   "sigma_scatter_valid_only": _finite(sig),
                                   "cosine": _finite(cos),
                                   "relative_scatter_at_n1000": _finite(ratio)})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lens_a")
    parser.add_argument("lens_b")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = build(args.lens_a, args.lens_b)
    if args.out:
        atomic_write_json(Path(args.out), result)
    print(json.dumps(result, indent=2, allow_nan=False))
