"""Run the SEALED calibration with an injected generation backend.

The sealed package is NEVER modified.  ``runner/sealed_import.py``
byte-verifies every sealed module against pinned SHA256s before placing the
package on ``sys.path``, so an edit there fails the import outright rather
than merely failing review.  Nothing in this file changes that.

``orchestrate_calibration`` has always accepted ``backend=None`` and only
constructs its own one-row-at-a-time ``RealBackend`` when nothing is
injected.  Its CLI never exposed that parameter, and ``production_entry``
runs the orchestrator as a SUBPROCESS through that CLI -- so there was no
way to reach the seam, and the calibration always ran serially.  That is
~4,608 rows at ~23-35 s/row on a B300: roughly 30-45 h, which no authorized
session can afford.  Per-row latency is architectural (``total_ut_steps=4``
over 48 layers is ~192 layer-passes per generated token) and FLAT from batch
1 to batch 64, so batch parallelism is the only lever that exists.

This launcher is the missing CLI surface.  It lives outside the sealed
package, mirrors the sealed calibration arguments exactly, and adds
``--backend`` / ``--batch-size``.  Invoked with ``--backend
REFERENCE_SERIAL`` (or with no ``--backend`` at all) it passes
``backend=None``, and the run is the sealed default unchanged.

The injected backend is only ever a configuration that earned its own
structural-equivalence verdict against REFERENCE_SERIAL over the full
validation corpus; ``production_entry.calibration_backend_choice`` makes
that decision, and the affordability gate projects from the same one, so
the gate cannot pass on a rate the calibration will not deliver.
"""

import argparse
import json
import os

from . import sealed_import


REFERENCE_BACKEND_IDS = ("", "REFERENCE_SERIAL", "REFERENCE_SERIAL_w1_b1")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="sealed calibration with an injected backend")
    # --- mirrored verbatim from the sealed CLI's `calibration` subcommand ---
    p.add_argument("--manifest-design", required=True)
    p.add_argument("--artifact-paths", required=True)
    p.add_argument("--precommit", required=True)
    p.add_argument("--calibration-task-manifest", required=True)
    p.add_argument("--axis-package", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metadata-output", required=True)
    p.add_argument("--progress")
    p.add_argument("--boundary-cache-dir")
    # --- the seam the sealed CLI does not expose ---
    p.add_argument("--backend", default="",
                   help="config_id or backend id to inject; the reference "
                        "backend means 'sealed default, nothing injected'")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--precompute-report",
                   help="where to write the precompute coverage report; it "
                        "carries the generation wall seconds the sealed "
                        "metadata cannot see")
    p.add_argument("--backend-family", default="",
                   help="backend id (e.g. B200_BATCHED); defaults to parsing "
                        "it from --backend")
    return p


def _sealed_preflight(args) -> dict:
    """Run the orchestrator's OWN preflight, in its order, before generating.

    ``orchestrate_calibration`` verifies the precommit, resolves and verifies
    the artifact paths, and requires the axis package to verify as SEALABLE --
    but it does all of that AFTER it would receive an injected backend.  The
    precomputed backend generates all 4,608 rows up front, so without this the
    session could spend the whole generation budget against an axis package
    that the sealed check was about to reject.

    Nothing here replaces those checks: orchestrate_calibration still runs
    every one of them itself.  This only refuses to generate before they have
    passed once.  Returns the resolved artifact paths.
    """
    from run_o1_v2_orchestrator import (              # sealed
        verify_calibration_precommit, verify_artifact_paths,
        OrchestrationError, load_jsonl, sha256_file)
    from verify_axis_artifact import verify as verify_axis  # sealed

    with open(args.manifest_design) as fh:
        manifest_raw = json.load(fh)
    verify_calibration_precommit(
        args.precommit, manifest_raw, args.calibration_task_manifest)

    # Cheap sealed refusals that would otherwise be evaluated only AFTER the
    # whole generation budget has been spent.  Each is milliseconds of
    # filesystem work and each aborts the run.
    if args.metadata_output and os.path.exists(args.metadata_output):
        raise OrchestrationError(
            "O0_PREFLIGHT",
            f"refusing to overwrite existing {args.metadata_output}")
    if not load_jsonl(args.calibration_task_manifest):
        raise OrchestrationError(
            "O0_PREFLIGHT", "calibration task manifest is empty")
    # _RecordSink refuses to mix precommit bindings (O6_RESUME_BINDING).  On a
    # resumed pod whose durable records were restored but whose precommit was
    # re-minted, that refusal lands after generation.  Read-only equivalent:
    if os.path.exists(args.output):
        want = sha256_file(args.precommit)
        for row in load_jsonl(args.output):
            got = row.get("calibration_precommit_sha256")
            if got and got != want:
                raise OrchestrationError(
                    "O6_RESUME_BINDING",
                    f"existing records at {args.output} are bound to "
                    f"precommit {got}, not {want}; refusing to generate "
                    f"against a record set this run cannot extend")
            break

    paths = verify_artifact_paths(manifest_raw, args.artifact_paths)
    axis_report = verify_axis(args.axis_package)
    if axis_report.get("verdict") != "SEALABLE":
        raise OrchestrationError(
            "O0_PREFLIGHT", "axis package did not verify as SEALABLE")
    return paths


def build_backend(args):
    """The backend to inject, or None for the sealed default.

    A batch size of 1 is not a batched run, so it is deliberately treated as
    the reference: injecting a 'batched' backend that batches nothing would
    add a moving part and buy nothing.
    """
    backend_id = str(args.backend or "").strip()
    if backend_id in REFERENCE_BACKEND_IDS:
        return None
    # The SAME predicate the affordability gate applied when it decided what
    # rate to project.  Keying on batch alone here is what let a batch-1
    # B200_REPLICA selection project the replica rate and then run serially.
    from .calibration_backend import supports_calibration
    supported, why_not = supports_calibration(
        args.backend_family or backend_id, args.workers, args.batch_size)
    if not supported:
        raise ValueError(
            f"refusing to inject {backend_id!r}: {why_not}. The affordability "
            f"gate and this launcher must agree about what the calibration "
            f"runs on; reaching here means they did not.")

    paths = _sealed_preflight(args)
    from .calibration_backend import PrecomputedBatchedBackend
    backend = PrecomputedBatchedBackend.from_manifest(
        manifest_design_path=args.manifest_design,
        task_manifest_path=args.calibration_task_manifest,
        axis_tensor_path=paths["artifact_hashes.structured_axis_tensor"],
        checkpoint=args.checkpoint,
        batch_size=int(args.batch_size))
    # generate everything up front: .generate() is a lookup and RAISES on a
    # key it never precomputed, so a silent serial fallback is impossible
    report = backend.precompute()
    if args.precompute_report:
        # The sealed CALIBRATION_METADATA's elapsed_wall_seconds is measured
        # from inside orchestrate_calibration, which this path enters only
        # after every row has been generated.  _Progress is sealed; this file
        # carries the generation time it cannot see, and is shipped in the
        # result archive so the record set can be audited.
        with open(args.precompute_report, "w", encoding="utf-8") as fh:
            json.dump({"schema": "o1b300.calibration_precompute.v1",
                       "backend": args.backend,
                       "backend_family": args.backend_family,
                       "workers": args.workers,
                       "batch_size": args.batch_size,
                       **report}, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
    print(json.dumps({"precompute": report}, indent=2, sort_keys=True,
                     default=str), flush=True)
    return backend


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sealed_import.ensure_sealed_path()
    from run_o1_v2_orchestrator import orchestrate_calibration  # sealed

    backend = build_backend(args)
    out = orchestrate_calibration(
        args.manifest_design, args.artifact_paths, args.precommit,
        args.calibration_task_manifest, args.axis_package, args.checkpoint,
        args.output, args.metadata_output, args.progress,
        args.boundary_cache_dir,
        backend=backend)
    # The sealed CLI ends by printing this same document.  production_entry
    # does NOT parse the child's stdout -- its drain thread writes raw bytes
    # to o1_orchestrator.log and never decodes them -- so nothing depends on
    # this being the only JSON value on stdout (build_backend prints the
    # precompute report before it).  Kept for parity with the sealed CLI and
    # for a human reading the log.
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
