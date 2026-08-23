#!/usr/bin/env python3
"""RunPod pre-rental master check -> RUNPOD_PRE_RENTAL_READINESS.{json,md}.

Verdicts: PASS | PASS_PENDING_READONLY_CREDENTIAL_CHECK | BLOCKED.
Never PASS while any code, image, artifact, termination, budget or
zero-touch test fails.  A missing live credential caps the verdict at
PASS_PENDING_READONLY_CREDENTIAL_CHECK.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PY = sys.executable

#: Files whose content legitimately changes AFTER the image is built (they
#: record or quote the build's own outputs), so they cannot be part of the
#: identity the build records -- a digest covering its own record could never
#: be re-synced without changing itself.
_SOURCE_HASH_SUFFIXES = (".py", ".sh")
_SOURCE_HASH_SKIP_DIRS = ("__pycache__", "reports")

#: Directories whose EVERY file is baked by `COPY o1_b200` and read at
#: runtime on the pod, whatever the extension.  A .py/.sh-only identity left
#: the frozen benchmark order, the frozen backend-selection policy and the
#: validation corpus outside the hash, so the image could differ from the
#: reviewed tree in exactly the content that decides the science.
_SOURCE_HASH_DIRS = ("policies", "corpus")

#: Excluded because they RECORD or are derived from this hash, or are
#: rewritten after the push: including them would make the identity
#: unreachable (a digest covering its own record has no fixed point).
_SOURCE_HASH_SKIP_FILES = frozenset({
    "provider/runpod/CONTAINER_IMAGE_RECORD.json",
    "provider/runpod/RUNPOD_SESSION_CONFIG.json",
    "SHA256SUMS",
})


#: The FL worktree is a sibling checkout; the image stages it in at build.
_FL_SOURCE_SELF_REFERENTIAL = ("deploy/environment_lock.json",
                               "deploy/INTEGRATION.md", "SHA256SUMS")


def _fl_source_tree_sha256(src: str | None = None) -> str:
    """The FL tree digest, by the same rule scripts/build_b300_image.sh uses.

    Returns "" when the FL worktree is not present in this checkout.
    """
    import hashlib
    src = src or os.path.join(os.path.dirname(_ROOT),
                              "foundation-learner-b200-v0",
                              "foundation_learner")
    if not os.path.isdir(src):
        return ""
    names = []
    for base, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in ("reports", "__pycache__")]
        for n in files:
            if n.endswith(".pyc"):
                continue
            rel = os.path.relpath(os.path.join(base, n), src)
            if rel.replace(os.sep, "/") in _FL_SOURCE_SELF_REFERENTIAL:
                continue
            names.append(rel)
    digests = []
    for rel in sorted(names):
        h = hashlib.sha256()
        with open(os.path.join(src, rel), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digests.append(f"{h.hexdigest()}  ./{rel}\n")
    return hashlib.sha256("".join(digests).encode()).hexdigest()


def _o1_source_tree_sha256(root: str | None = None) -> str:
    """Identity of the EXECUTABLE o1_b200 source baked into the image.

    Only .py and .sh: those decide what the pod actually does.  JSON records
    and markdown are excluded because the build and the push rewrite several
    of them afterwards.  Sorted with an explicit C collation so the same tree
    digests identically on a differently-configured machine.
    """
    import hashlib
    base = os.path.join(root or _ROOT, "o1_b200")
    entries = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in _SOURCE_HASH_SKIP_DIRS)
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, base)
            if rel.replace(os.sep, "/") in _SOURCE_HASH_SKIP_FILES:
                continue
            in_runtime_dir = rel.replace(os.sep, "/").split("/")[0] in \
                _SOURCE_HASH_DIRS
            if not (name.endswith(_SOURCE_HASH_SUFFIXES) or in_runtime_dir):
                continue
            entries.append((rel, full))
    outer = hashlib.sha256()
    for rel, full in sorted(entries, key=lambda e: e[0].encode()):
        with open(full, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        outer.update(f"{digest}  {rel}\n".encode())
    return outer.hexdigest()


def _run(cmd, cwd=None, timeout=3600, live_credentials=False):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": _ROOT}
    if not live_credentials:
        # local phases are hermetic: neither the operator credential nor the
        # live-mutation authorization flag enters a local/mock test process
        # (the independent watchdog gates only on that flag)
        env.pop("RUNPOD_API_KEY", None)
        env.pop("RUNPOD_API_KEY_FILE", None)
        env.pop("RUNPOD_ALLOW_BILLABLE_MUTATIONS", None)
        env.pop("HF_TOKEN", None)
    proc = subprocess.run(cmd, cwd=cwd or _ROOT, capture_output=True,
                          text=True, timeout=timeout, env=env)
    return proc.returncode, (proc.stdout + proc.stderr)[-4000:]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.join(_ROOT, "o1_b200",
                                                     "reports"))
    p.add_argument("--skip-full-suite", action="store_true",
                   help="reuse the latest TEST_REPORT.json instead of "
                        "rerunning the multi-minute full suite")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    checks = {}

    def record(name, ok, detail=""):
        checks[name] = {"ok": bool(ok), "detail": str(detail)[-1500:]}
        print(("PASS " if ok else "FAIL ") + name)

    # 1. base-package integrity
    rc, out = _run([PY, "run_all_tests.py"],
                   cwd=os.path.join(_ROOT, "o1_packages",
                                    "O1_oracle_reachability_v2.1.0_source",
                                    "o1_v210"))
    record("base_package_suite", rc == 0, out.splitlines()[-1] if out else "")

    # 2. full B200-runner + adapter suites (or reuse the fresh report)
    if a.skip_full_suite:
        rep_path = os.path.join(_ROOT, "o1_b200", "reports", "TEST_REPORT.json")
        with open(rep_path, encoding="utf-8") as fh:
            rep = json.load(fh)
        ok = rep["total_failed"] == 0 and any(
            m["module"].startswith("test_runpod") for m in rep["modules"])
        record("b200_runner_and_adapter_suites", ok,
               f"{rep['total_passed']} checks, {rep['total_failed']} failed "
               f"(reused {rep['utc']})")
    else:
        rc, out = _run([PY, "run_all_b200_tests.py"],
                       cwd=os.path.join(_ROOT, "o1_b200", "tests"))
        record("b200_runner_and_adapter_suites", rc == 0,
               out.splitlines()[-1] if out else "")

    # 3. pinned OpenAPI schema
    try:
        from o1_b200.provider.runpod.schema_check import verify_pinned_schema
        record("openapi_schema_pin", True, verify_pinned_schema()["sha256"])
    except Exception as exc:  # noqa: BLE001
        record("openapi_schema_pin", False, exc)

    # 4. dry-run all-profile deployment rendering (B300 primary + B200
    #    explicit fallback under one authorization hash)
    try:
        from o1_b200.provider.runpod.pod_request import (
            build_pod_request, render_canonical_deployment)
        from o1_b200.provider.runpod.policy import PROFILE_PREFERENCE
        reqs = {p.key: build_pod_request(
            profile=p,
            image_digest_ref="local/o1-b300-runner@sha256:" + "0" * 64,
            datacenter_id="DRYRUN-DC") for p in PROFILE_PREFERENCE}
        rendered = render_canonical_deployment(reqs, {"dry_run": True})
        record("dry_run_deployment_render", True,
               f"profiles={rendered['profile_preference']} "
               f"{rendered['request_sha256'][:16]}")
    except Exception as exc:  # noqa: BLE001
        record("dry_run_deployment_render", False, exc)

    # 5. container image record (B300/cu130 stack)
    img_record_path = os.path.join(_ROOT, "o1_b200", "provider", "runpod",
                                   "CONTAINER_IMAGE_RECORD.json")
    try:
        with open(img_record_path, encoding="utf-8") as fh:
            img = json.load(fh)
        env = img["environment"]
        no_ptx = not any(a.startswith("compute_") for a in env["arch_list"])
        ok = (env["transformers"] == "4.54.1"
              and env["torch"] == "2.12.1+cu130"
              and env["torch_cuda_runtime"] == "13.0"
              and "sm_100" in env["arch_list"]
              and no_ptx
              and img["platform"] == "linux/amd64"
              and img["schema"] == "o1b300.container_image_record.v2")
        record("container_image_built_and_asserted", ok,
               f"torch={env['torch']} arch={env['arch_list']} no_ptx={no_ptx}")
    except Exception as exc:  # noqa: BLE001
        record("container_image_built_and_asserted", False, exc)

    # 5b. THE image actually bound is THE image that was built and reviewed.
    # registry_digest_ref had two writers and no readers: nothing compared it
    # to the session config's image_digest_ref, so a source fix that was
    # never rebuilt/re-pushed left a stale digest bound and every gate still
    # reported PASS.  That is how a NameError fixed in git reached a paid pod.
    try:
        with open(img_record_path, encoding="utf-8") as fh:
            img = json.load(fh)
        cfg_path = os.path.join(_ROOT, "o1_b200", "provider", "runpod",
                                "RUNPOD_SESSION_CONFIG.json")
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        built = str(img.get("registry_digest_ref", ""))
        bound = str(cfg.get("image_digest_ref", ""))
        ok = (bool(built) and bool(bound)
              and not built.startswith("UNRESOLVED")
              and not bound.startswith("UNRESOLVED")
              and built == bound)
        detail = ("bound digest matches the built image record"
                  if ok else f"built={built[:60]!r} bound={bound[:60]!r}")
        record("bound_image_is_the_built_image", ok, detail)
    except Exception as exc:  # noqa: BLE001
        record("bound_image_is_the_built_image", False, exc)

    # 5c. the built image was built from THIS source tree.  The build script
    # recorded a FL source hash but nothing for o1_b200 itself, so a driver
    # edit after the build was invisible to every check.
    try:
        with open(img_record_path, encoding="utf-8") as fh:
            img = json.load(fh)
        recorded = str(img.get("o1_b200_source_sha256", ""))
        live = _o1_source_tree_sha256()
        ok = bool(recorded) and recorded == live
        record("built_image_matches_o1_source", ok,
               "source tree matches the image record" if ok
               else f"recorded={recorded[:16]!r} live={live[:16]!r}")
    except Exception as exc:  # noqa: BLE001
        record("built_image_matches_o1_source", False, exc)

    # 5d. the FL half of the SAME image.  Round 2 added this check for the
    # o1 source and left the symmetric gap open on the FL side -- the half
    # that was being actively edited.  Same exclusion rule as
    # scripts/build_b300_image.sh: the two files that RECORD the digest, plus
    # SHA256SUMS, which covers one of them.
    try:
        with open(img_record_path, encoding="utf-8") as fh:
            img = json.load(fh)
        recorded = str(img.get("foundation_learner_source_sha256", ""))
        if recorded.startswith("ABSENT_BY_REQUEST"):
            record("built_image_matches_fl_source", True,
                   "O1-only image by request")
        else:
            live = _fl_source_tree_sha256()
            ok = bool(recorded) and bool(live) and recorded == live
            record("built_image_matches_fl_source", ok,
                   "FL source tree matches the image record" if ok
                   else f"recorded={recorded[:16]!r} live={live[:16]!r}")
    except Exception as exc:  # noqa: BLE001
        record("built_image_matches_fl_source", False, exc)

    # 6. artifact transfer manifest
    try:
        man_path = os.path.join(_ROOT, "o1_b200", "provider", "runpod",
                                "RUNPOD_ARTIFACT_TRANSFER_MANIFEST.json")
        with open(man_path, encoding="utf-8") as fh:
            man = json.load(fh)
        record("artifact_transfer_manifest", len(man["artifacts"]) >= 12,
               f"{len(man['artifacts'])} artifacts")
    except Exception as exc:  # noqa: BLE001
        record("artifact_transfer_manifest", False, exc)

    # 6b. transfer manifests are FRESH: they pin tree hashes of policies/,
    # runner/ and deploy/, and a repair that touches any of those without
    # regenerating them makes verify_artifacts.py fail on every pod built
    # from HEAD (a deterministic abort, paid for).  This is sha256_tree
    # over local paths: no network, no cost.
    rc, out = _run([PY, os.path.join(_ROOT, "o1_b200", "deploy",
                                     "verify_artifacts.py"),
                    "--manifest", os.path.join(_ROOT, "o1_b200", "deploy",
                                               "TRANSFER_MANIFEST.json")])
    record("transfer_manifests_fresh", rc == 0,
           out.strip().splitlines()[-1] if out.strip() else "")

    # 7. secret scan over the tree
    rc, out = _run(["grep", "-rIlE",
                    "(rpa_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY|AKIA[A-Z0-9]{16})",
                    "--exclude-dir=local_runs", "--exclude-dir=.git",
                    os.path.join(_ROOT, "o1_b200")])
    record("secret_scan_clean", rc != 0, out.strip() or "no secrets found")

    # 8. templates unusable as authorization
    try:
        from o1_b200.provider.runpod.authorization import (
            AuthorizationError, LiveMutationAuthorization, CLI_FLAG,
            ENV_FLAG, ENV_FLAG_VALUE)
        try:
            LiveMutationAuthorization.verify(
                path=os.path.join(_ROOT, "o1_b200", "provider", "runpod",
                                  "B300_RENTAL_AUTHORIZATION.template.json"),
                expected_identity={k: "x" for k in
                                   ("project", "package_zip_sha256",
                                    "provider", "budget_policy_sha256",
                                    "deployment_spec_sha256")},
                cli_args=[CLI_FLAG],
                nonce_ledger=os.path.join(a.out_dir, "nonce_probe.txt"),
                environ={ENV_FLAG: ENV_FLAG_VALUE})
            record("authorization_template_unusable", False,
                   "template validated!")
        except AuthorizationError:
            record("authorization_template_unusable", True)
    except Exception as exc:  # noqa: BLE001
        record("authorization_template_unusable", False, exc)

    # 9. read-only live check when a credential exists
    from o1_b200.provider.runpod.redaction import load_api_key
    live_verdict = "SKIPPED_NO_CREDENTIAL"
    if load_api_key():
        rc, out = _run([PY, "-m", "o1_b200.provider.runpod.preflight",
                        "--out", os.path.join(a.out_dir,
                                              "RUNPOD_READONLY_PREFLIGHT.json")],
                       live_credentials=True)
        live_verdict = "PASS" if rc == 0 else "FAIL"
        record("readonly_live_preflight", rc == 0, out.strip()[-200:])
    else:
        print("SKIP readonly_live_preflight (no credential)")

    all_local_ok = all(c["ok"] for c in checks.values())
    if not all_local_ok:
        verdict = "BLOCKED"
    elif live_verdict == "PASS":
        verdict = "PASS"
    elif live_verdict == "SKIPPED_NO_CREDENTIAL":
        verdict = "PASS_PENDING_READONLY_CREDENTIAL_CHECK"
    else:
        verdict = "BLOCKED"

    report = {
        "schema": "o1b200.runpod_pre_rental_readiness.v1",
        "verdict": verdict,
        "readonly_live_check": live_verdict,
        "checks": checks,
    }
    json_path = os.path.join(a.out_dir, "RUNPOD_PRE_RENTAL_READINESS.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    md = ["# RunPod Pre-Rental Readiness", "",
          f"**Verdict: {verdict}**",
          f"Read-only live check: {live_verdict}", "", "| check | ok |", "|---|---|"]
    for name, c in sorted(checks.items()):
        md.append(f"| {name} | {'PASS' if c['ok'] else 'FAIL'} |")
    with open(os.path.join(a.out_dir, "RUNPOD_PRE_RENTAL_READINESS.md"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"RUNPOD PRE-RENTAL READINESS: {verdict}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
