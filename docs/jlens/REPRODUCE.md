# JLens reproduction and verification

Run commands from the repository root. None of the commands in this document
creates a cloud resource.

## Fast mechanical suite

```bash
venv/bin/python -m pytest -q \
  utilities/tests/unit/test_jlens_integrity.py \
  utilities/tests/unit/test_jlens_rental.py \
  utilities/tests/unit/test_jlens_reports.py \
  utilities/tests/unit/test_ouro_jlens.py \
  utilities/tests/unit/test_peer3_jlens_audit.py

venv/bin/python -m pytest -q utilities/tests/unit/test_peer1_index_map.py
bash -n src/ouro_jlens/*.sh
venv/bin/python -m py_compile src/ouro_jlens/*.py
```

The last pytest file is CUDA/model-backed; it is not a CPU test.

## Current-source instrumentation validation

```bash
venv/bin/python src/ouro_jlens/validate.py --out artifacts/jlens/validation
```

Acceptance requires `numerical_pass: true`, human-loop sources
`[16,64,112,160]`, and a zero exit status. Bit-exact status is reported
separately.

## Probe evidence

Regenerate the GPU cache and both lens score arrays under current code/model:

```bash
venv/bin/python src/ouro_jlens/probe_cv.py \
  --cache artifacts/jlens/probe/n80_v2/gpu_cache.npz \
  --lens artifacts/jlens/lens/exit3/exit3_n80.pt \
  --out artifacts/jlens/probe/cv_all648 \
  --rebuild-gpu-cache --lens-only
```

Fit the five pair-held-out folds on CPU. Parallelism is over independent virtual
locations and does not alter the estimator:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  venv/bin/python src/ouro_jlens/probe_cv.py \
  --cache artifacts/jlens/probe/n80_v2/gpu_cache.npz \
  --lens artifacts/jlens/lens/exit3/exit3_n80.pt \
  --out artifacts/jlens/probe/cv_all648 --jobs 8
```

Expected fair-cohort points are probe
`[0.614583,0.590278,0.336806,0.236111]`, logit lens
`[0.763889,0.500000,0.178819,0.310764]`, and final-exit average J-lens
`[0.392361,0.500000,0.147569,0.119792]`.

## Canonical reports

```bash
venv/bin/python src/ouro_jlens/fitsize_report.py

venv/bin/python src/ouro_jlens/transport_report.py \
  --lens 8=artifacts/jlens/lens/exit3/exit3_p0-8.pt \
  --lens 32=artifacts/jlens/lens/exit3/exit3_merged.pt \
  --lens 56=artifacts/jlens/lens/exit3/exit3_n56.pt \
  --lens 80=artifacts/jlens/lens/exit3/exit3_n80.pt \
  --out artifacts/jlens/final/transport.json

venv/bin/python src/ouro_jlens/report.py \
  --eval artifacts/jlens/eval/fitsize_n80 \
  --local-eval artifacts/jlens/eval/round1_exit3x32 \
  --validation artifacts/jlens/validation/milestones.json \
  --fit-size artifacts/jlens/final/fit_size.json \
  --transport artifacts/jlens/final/transport.json \
  --probe artifacts/jlens/probe/cv_all648/summary.json \
  --checkpoints artifacts/jlens/checkpoints/exit_divergence.json

venv/bin/python src/ouro_jlens/verify_artifacts.py
venv/bin/python src/ouro_jlens/manifest.py build
venv/bin/python src/ouro_jlens/manifest.py verify
```

The independent verifier must report `PASS`; the custody verifier must report
zero mismatches. Building a manifest records byte identity but does not upgrade
the `UNFROZEN`, `INCONCLUSIVE`, `NOT_RETAINED`, or `NOT_ESTABLISHED` scientific
states.

## Historical custody

```bash
(cd docs/jlens/history/2026-09-04-opus && sha256sum -c SHA256SUMS)
```

The recovered pre-repair probe cache and score arrays are preserved under
`artifacts/jlens/probe/history/recovered-2026-09-04/`; current results do not
depend on those copies.

## Rental dry run

```bash
DRY_RUN=1 bash src/ouro_jlens/stage_upload.sh --dry-run
venv/bin/python src/ouro_jlens/pod.py create --dry-run --run-id offline-check
```

These are offline configuration checks. The stage command also refuses
uncommitted JLens source, tests, or documentation and verifies the resulting
archive twice. Do not remove `--dry-run` as part of reproduction.
