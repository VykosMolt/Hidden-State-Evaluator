# Hunter Seeker V2 bounded snapshot

Snapshot ID: `hunter_seeker_v2_bounded_snapshot_20260802`

This snapshot preserves the reviewed V2 source, documentation, focused tests,
manual runners, the pinned TU93 environment variant, and the experiment
provenance helper. The machine-readable file list and SHA-256 records are in
[SNAPSHOT_MANIFEST.json](SNAPSHOT_MANIFEST.json).

## Repository provenance

- Branch: `main`
- HEAD before the bounded commit: `ea40a48f7f41fdf60866eb2067033c62dd14be0f`
- Bounded commit paths: 69 (67 payload paths plus these two provenance files)
- Snapshot aggregate SHA-256:
  `4ebc35e23331ef9e843ef0b27ed0750e180ac0cbc426006feaed52452981e179`
- The working tree contained substantial unrelated modified, deleted, and
  untracked state. It was not reset, stashed, cleaned, or staged.
- No artifact directory, checkpoint, secret, evaluator path, or unrelated
  deletion is part of this bounded set.

## Verification

Focused command:

```text
venv/bin/python -m pytest -q utilities/tests/unit/test_hunter_seeker_v2_*.py utilities/tests/integration/test_hunter_seeker_v2_integration.py
287 passed
```

The authoritative runner is
`utilities/tests/manual/run_hs_v2_tu93_fresh_retention_v1.py`. Its contract
is documented in
[FRESH_DISCOVERY_RETENTION_EXPERIMENT.md](FRESH_DISCOVERY_RETENTION_EXPERIMENT.md).
The final smoke artifact is descriptive-only; the full 32-pair design was
attempted but remained computationally blocked and therefore produced no
causal retention claim.

## Boundary

This is a bounded V2 implementation snapshot, not a claim that fresh
discovery, held-out generalization, or RLTT has been demonstrated. Existing
acquired-route evidence remains acquisition/replay evidence. The experiment
artifacts under `artifacts/reports/hunter_seeker_v2/` are intentionally kept
outside the source commit and retain their own provenance and stale/blocked
markers.
