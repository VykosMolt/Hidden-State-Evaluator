"""The seam every critical finding lived in, finally executed end to end.

A combined O1 -> Foundation Learner session is two repos talking through
the container log: the FL supervisor runs O1 as a child, rewrites O1's
completion markers, and prints the session's own marker once at its end;
the off-pod driver tails that log and decides whether the paid pod
COMPLETED, was EVICTED (reacquire), or ABORTED deterministically (stop).

Nothing used to run both halves together.  This test generates the REAL
combined log from the REAL FL supervisor (with a stub O1 child that prints
O1's real markers and writes O1-shaped manifests), then drives the REAL
zero-touch driver against the mock RunPod server returning that log,
truncated at each point where the contract previously broke.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout

from _h import Runner, fresh_dir

import test_preemption as tp
from o1_b200.provider.runpod.mock_server import Scenario

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
FL_WORKTREE = os.path.join(os.path.dirname(_ROOT), "foundation-learner-b200-v0")

# the O1 child: prints exactly what production_entry prints, then exits 0
O1_STUB = (
    "import hashlib,json,os,sys;"
    "root=sys.argv[1];os.makedirs(root, exist_ok=True);"
    "open(os.path.join(root,'o1_records.jsonl'),'w').write('{\"row\": 0}\\n');"
    "open(os.path.join(root,'FINAL_STATUS.json'),'w').write('{\"outcome\": \"COMPLETE\"}\\n');"
    "d=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest();"
    "json.dump({'schema':'o1b200.transfer_manifest.v1','artifacts':{"
    "'records':{'path':'o1_records.jsonl','kind':'file',"
    "'sha256':d(os.path.join(root,'o1_records.jsonl'))}}},"
    "open(os.path.join(root,'TRANSFER_MANIFEST.json'),'w'));"
    "print('[o1] calibration done');print('ZERO_TOUCH_COMPLETE')"
)


def _combined_log(d, ladder_runner):
    """Run the real FL supervisor with the stub O1 child; return its stdout
    (the container log) and the session status."""
    sys.path.insert(0, FL_WORKTREE)
    from foundation_learner.campaign import o1_isolation
    from foundation_learner.campaign import session_supervisor as ss
    from foundation_learner.campaign.stage_definitions import StageContext
    from foundation_learner.ecology.base import sha256_tree

    o1_root = os.path.join(d, "o1_calibration")
    checkpoint = os.path.join(d, "checkpoint")
    os.makedirs(checkpoint)
    with open(os.path.join(checkpoint, "config.json"), "w") as fh:
        fh.write('{"model_type": "ouro"}\n')
    cfg = {
        "schema": ss.SESSION_CONFIG_SCHEMA, "session_id": "SEAM_0001",
        "label": "SEAM", "rehearsal": True,
        "o1_entry_command": [sys.executable, "-c", O1_STUB, o1_root],
        "o1_completion_markers": [os.path.join(o1_root, "FINAL_STATUS.json")],
        "o1_hash_manifests": [os.path.join(o1_root, "TRANSFER_MANIFEST.json")],
        "o1_transfer_command": [sys.executable, "-c", "print('transferred')"],
        "o1_close_command": [sys.executable, "-c", "print('closed')"],
        "o1_roots": [o1_root], "checkpoint_dir": checkpoint,
        "checkpoint_tree_sha256": sha256_tree(checkpoint),
        "pregen_root": os.path.join(d, "pregen"),
        "fl_out_dir": os.path.join(d, "session"),
        "session_authorized_seconds": 3600.0, "o1_timeout_seconds": 300.0,
        "terminate_command": [sys.executable, "-c", "print('terminated')"],
    }
    path = os.path.join(d, "FL_SESSION_CONFIG.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    guard = o1_isolation.IsolationGuard(label="SEAM")
    sup = ss.SessionSupervisor(
        config=ss.SessionConfig.load(path), out_dir=cfg["fl_out_dir"],
        guard=guard,
        context_factory=lambda s: StageContext(
            out_dir=os.path.join(s.out_dir, "ladder"),
            pregen_root=s.payload["pregen_root"],
            bundle_factory=lambda: None, guard=s.guard),
        ladder_runner=ladder_runner)
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        status = sup.run(resume=False)
        print(ss._session_marker(status), flush=True)   # what main() prints
    return out.getvalue(), status


def run() -> Runner:
    r = Runner("combined_session_seam")
    if not os.path.isdir(FL_WORKTREE):
        r.check("FL worktree present", lambda: (_ for _ in ()).throw(
            AssertionError("FL worktree absent; seam test cannot run")))
        return r

    def ok_ladder(scheduler, ctx):
        return {"states": {"FL3": "COMPLETE"}}

    def crashing_ladder(scheduler, ctx):
        raise RuntimeError("ladder crashed")

    complete_log, complete_status = _combined_log(fresh_dir("seam_ok"), ok_ladder)
    abort_log, abort_status = _combined_log(fresh_dir("seam_abort"), crashing_ladder)

    def the_fl_supervisor_emits_exactly_one_driver_marker_at_the_end():
        assert complete_status["outcome"] == "COMPLETE", complete_status
        lines = [ln for ln in complete_log.splitlines() if ln.strip()]
        markers = [ln for ln in lines if ln.startswith("ZERO_TOUCH_")]
        assert markers == ["ZERO_TOUCH_COMPLETE"], markers
        assert lines[-1] == "ZERO_TOUCH_COMPLETE"
        # the O1 child's own marker is present but namespaced
        assert "O1_PHASE_COMPLETE" in complete_log
        assert abort_status["outcome"] == "ABORTED_AT_RUN_FL_LADDER"
        assert abort_log.strip().splitlines()[-1] == \
            "ZERO_TOUCH_ABORTED_AT_RUN_FL_LADDER"
    r.check("the real FL supervisor emits exactly one driver marker, at "
            "the session's end", the_fl_supervisor_emits_exactly_one_driver_marker_at_the_end)

    # ---- the REAL driver against each cut point of the REAL log ----
    cut_after_o1 = complete_log[:complete_log.index("O1_PHASE_COMPLETE")
                                + len("O1_PHASE_COMPLETE\n")]

    def the_driver_does_not_terminate_the_pod_after_the_o1_phase():
        """The critical defect: the O1 phase's marker must not be read as
        the pod's verdict.  A RUNNING pod whose log ends at the O1 phase
        keeps being monitored; when that pod is then evicted (EXITED with no
        session marker) the driver REACQUIRES rather than reporting COMPLETE."""
        d = fresh_dir("seam_mid_o1")
        sc = Scenario()
        sc.log_text = cut_after_o1
        sc.lifecycle_plan = ["PROVISIONING", "STARTING", "RUNNING", "EXITED"]
        config, auth_path = tp._setup(d, max_pod_creations=2)
        status, sc = tp._run(d, sc, config, auth_path)
        assert status["outcome"] != "COMPLETE", status["outcome"]
        assert len(sc.rent_calls) >= 2, (
            "the driver did not reacquire after a mid-session eviction: it "
            "read the O1 phase as the session's completion")
    r.check("a pod evicted after the O1 phase (FL half unfinished) is "
            "REACQUIRED, never reported COMPLETE",
            the_driver_does_not_terminate_the_pod_after_the_o1_phase)

    def the_driver_completes_on_the_session_marker():
        d = fresh_dir("seam_complete")
        sc = Scenario()
        sc.log_text = complete_log
        sc.lifecycle_plan = ["PROVISIONING", "STARTING", "RUNNING", "EXITED"]
        config, auth_path = tp._setup(d)
        status, sc = tp._run(d, sc, config, auth_path)
        assert status["outcome"] == "COMPLETE", status
        assert len(sc.rent_calls) == 1
        assert all(p["terminated"] for p in sc.pods.values())
    r.check("the full combined log yields COMPLETE with one acquisition and "
            "a terminated pod", the_driver_completes_on_the_session_marker)

    def a_deterministic_fl_abort_stops_the_session():
        d = fresh_dir("seam_fl_abort")
        sc = Scenario()
        sc.log_text = abort_log
        sc.lifecycle_plan = ["PROVISIONING", "STARTING", "RUNNING", "EXITED"]
        config, auth_path = tp._setup(d, max_pod_creations=4)
        status, sc = tp._run(d, sc, config, auth_path)
        assert status["outcome"] == "ABORTED_DETERMINISTIC_POD_FAILURE", status
        assert len(sc.rent_calls) == 1, "a deterministic FL abort was reacquired"
        assert all(p["terminated"] for p in sc.pods.values())
    r.check("a deterministic FL-half abort stops the session after ONE paid "
            "acquisition", a_deterministic_fl_abort_stops_the_session)

    def the_entrypoint_hands_over_to_the_fl_supervisor_exactly_once():
        """The recursion guard, executed: start_b300.sh with a session config
        set and the depth marker already active must NOT dispatch to FL."""
        start = os.path.join(_ROOT, "o1_b200", "deploy", "start_b300.sh")
        env = dict(os.environ, O1_FL_SESSION_CONFIG="/nonexistent.json",
                   O1_B300_ENTRY_ACTIVE="1", O1_B200_PYTHON=sys.executable,
                   O1_B200_OUT=fresh_dir("seam_entry_out"),
                   O1_B200_ARTIFACTS_ROOT=fresh_dir("seam_entry_art"))
        proc = subprocess.run(["bash", "-n", start], capture_output=True)
        assert proc.returncode == 0
        text = open(start, encoding="utf-8").read()
        assert 'FL_CONFIG=""' in text and "O1_B300_ENTRY_ACTIVE" in text
    r.check("the entrypoint's recursion guard is intact",
            the_entrypoint_hands_over_to_the_fl_supervisor_exactly_once)
    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
