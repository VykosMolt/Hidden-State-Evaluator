"""HOSTILE: a failed session leaves a rented accelerator billing.

Attack: make a state fail anywhere in the machine (an O1 halt, a bad checkpoint
hash, an exploding ladder, an interrupt) and check whether the supervisor still
terminates the accelerator.  Before the repair, ``run()`` simply ``break``-ed
out of the state loop, so ``TERMINATE_ACCELERATOR`` never ran on ANY failure
path: the pod kept billing until a human noticed.

Second attack: crash mid-ladder and resume.  Before the repair, resume marked
``COMPUTE_REMAINING_AUTHORIZED_TIME`` as "already completed" without restoring
its result, so ``RUN_FL_LADDER`` refused for want of an FL allowance — a crash
silently cost the whole FL half of the session.

Contract §11 (the reserve is never consumed), §13 (the state machine), §20.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from foundation_learner.campaign import o1_isolation
from foundation_learner.campaign import session_supervisor as ss
from foundation_learner.ecology.base import sha256_tree

STUB = (
    "import hashlib,json,os,sys;"
    "root=sys.argv[1];"
    "os.makedirs(root, exist_ok=True);"
    "open(os.path.join(root,'o1_records.jsonl'),'w').write('{\"row\": 0}\\n');"
    "open(os.path.join(root,'O1_COMPLETE'),'w').write('done\\n');"
    "d=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest();"
    "json.dump({'schema':'o1b200.transfer_manifest.v1','artifacts':{"
    "'records':{'path':'o1_records.jsonl','kind':'file',"
    "'sha256':d(os.path.join(root,'o1_records.jsonl'))}}},"
    "open(os.path.join(root,'TRANSFER_MANIFEST.json'),'w'))"
)


def fixtures(tmp_path, marker_path, **overrides):
    o1_root = tmp_path / "o1_calibration"
    o1_root.mkdir(exist_ok=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text('{"model_type": "ouro"}\n',
                                            encoding="utf-8")
    terminate = [sys.executable, "-c",
                 "import sys; open(sys.argv[1], 'a').write('TERMINATED\\n')",
                 str(marker_path)]
    payload = {
        "schema": ss.SESSION_CONFIG_SCHEMA,
        "session_id": "HOSTILE_TERMINATE",
        "label": "HOSTILE",
        "rehearsal": True,
        "o1_entry_command": [sys.executable, "-c", STUB, str(o1_root)],
        "o1_completion_markers": [str(o1_root / "O1_COMPLETE")],
        "o1_hash_manifests": [str(o1_root / "TRANSFER_MANIFEST.json")],
        "o1_transfer_command": [sys.executable, "-c", "print('t')"],
        "o1_close_command": [sys.executable, "-c", "print('c')"],
        "o1_roots": [str(o1_root)],
        "checkpoint_dir": str(checkpoint),
        "checkpoint_tree_sha256": sha256_tree(str(checkpoint)),
        "pregen_root": str(tmp_path / "pregen"),
        "fl_out_dir": str(tmp_path / "session"),
        "session_authorized_seconds": 3600.0,
        "o1_timeout_seconds": 300.0,
        "fl_transfer_command": [sys.executable, "-c", "print('ft')"],
        "terminate_command": terminate,
    }
    payload.update(overrides)
    path = tmp_path / "session_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")
    return str(path)


def make_supervisor(tmp_path, config_path, *, ladder_runner=None,
                    context_factory=None):
    guard = o1_isolation.IsolationGuard(label="HOSTILE_TERMINATE")
    from foundation_learner.campaign.stage_definitions import StageContext

    def default_context(sup):
        return StageContext(out_dir=os.path.join(sup.out_dir, "ladder"),
                            pregen_root=sup.payload["pregen_root"],
                            bundle_factory=lambda: None, guard=sup.guard)

    def default_ladder(scheduler, ctx):
        os.makedirs(scheduler.out_dir, exist_ok=True)
        with open(os.path.join(scheduler.out_dir, "stub.json"), "w",
                  encoding="utf-8") as fh:
            fh.write('{"stub": true}\n')
        return {"states": {"BENCH": "COMPLETE"}}

    return ss.SessionSupervisor(
        config=ss.SessionConfig.load(config_path, guard=guard),
        out_dir=str(tmp_path / "session"), guard=guard,
        ladder_runner=ladder_runner or default_ladder,
        context_factory=context_factory or default_context)


def terminated(marker_path) -> int:
    if not os.path.exists(marker_path):
        return 0
    return len([l for l in open(marker_path, encoding="utf-8") if l.strip()])


# ---------------- termination on every failure path ----------------

@pytest.mark.parametrize("break_it,expected_state", [
    ("o1_halt", "O1_HALT_OR_COMPLETE"),
    ("bad_checkpoint", "RELOAD_PRISTINE_OURO"),
    ("ladder_explodes", "RUN_FL_LADDER"),
])
def test_every_failure_path_still_terminates_the_accelerator(
        tmp_path, break_it, expected_state):
    marker = tmp_path / "TERMINATED.log"
    overrides = {}
    if break_it == "o1_halt":
        overrides["o1_entry_command"] = [sys.executable, "-c",
                                         "raise SystemExit(3)"]
    if break_it == "bad_checkpoint":
        overrides["checkpoint_tree_sha256"] = "b" * 64
    path = fixtures(tmp_path, marker, **overrides)

    def exploding_ladder(scheduler, ctx):
        raise RuntimeError("the ladder exploded mid-run")

    sup = make_supervisor(
        tmp_path, path,
        ladder_runner=exploding_ladder if break_it == "ladder_explodes" else None)
    status = sup.run(resume=False)

    assert status["outcome"] == f"ABORTED_AT_{expected_state}", status
    assert terminated(marker) == 1, (
        "REFUSED: the session aborted without terminating the accelerator; a "
        "rented pod keeps billing until it is terminated")
    assert status["close_out"]["terminate"]["status"] == "COMPLETED"
    events = [(r["event"], r["state"]) for r in sup.read_journal()]
    assert ("EMERGENCY_STATE_COMPLETED", "TERMINATE_ACCELERATOR") in events


def test_an_interrupt_inside_a_state_still_terminates(tmp_path):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)

    def interrupting_ladder(scheduler, ctx):
        raise KeyboardInterrupt("operator interrupt")

    sup = make_supervisor(tmp_path, path, ladder_runner=interrupting_ladder)
    status = sup.run(resume=False)
    assert status["outcome"] == "ABORTED_AT_RUN_FL_LADDER"
    assert terminated(marker) == 1


def test_a_successful_session_terminates_exactly_once(tmp_path):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    sup = make_supervisor(tmp_path, path)
    status = sup.run(resume=False)
    assert status["outcome"] == "COMPLETE"
    assert terminated(marker) == 1
    assert status["close_out"].get("note")


def test_a_failed_transfer_does_not_prevent_termination(tmp_path):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker,
                    fl_transfer_command=[sys.executable, "-c",
                                         "raise SystemExit(9)"])

    def exploding_ladder(scheduler, ctx):
        os.makedirs(scheduler.out_dir, exist_ok=True)
        with open(os.path.join(scheduler.out_dir, "partial.json"), "w") as fh:
            fh.write("{}\n")
        raise RuntimeError("boom")

    sup = make_supervisor(tmp_path, path, ladder_runner=exploding_ladder)
    status = sup.run(resume=False)
    assert status["close_out"]["transfer"]["status"] == "FAILED"
    assert status["close_out"]["terminate"]["status"] == "COMPLETED"
    assert terminated(marker) == 1


# ---------------- mid-ladder crash -> resume ----------------

def test_a_mid_ladder_crash_resumes_and_completes_the_remaining_states(tmp_path):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    calls: list[float] = []

    def crashing_ladder(scheduler, ctx):
        raise RuntimeError("simulated mid-ladder crash")

    first = make_supervisor(tmp_path, path, ladder_runner=crashing_ladder)
    status = first.run(resume=False)
    assert status["outcome"] == "ABORTED_AT_RUN_FL_LADDER"
    assert "COMPUTE_REMAINING_AUTHORIZED_TIME" in status["states_completed"]
    assert terminated(marker) == 1              # the crash still terminated

    def resuming_ladder(scheduler, ctx):
        calls.append(scheduler.available_foundation_learner_seconds)
        os.makedirs(scheduler.out_dir, exist_ok=True)
        with open(os.path.join(scheduler.out_dir, "resumed.json"), "w") as fh:
            fh.write('{"resumed": true}\n')
        return {"states": {"BENCH": "COMPLETE"}}

    second = make_supervisor(tmp_path, path, ladder_runner=resuming_ladder)
    status = second.run(resume=True)

    assert status["outcome"] == "COMPLETE", status["failure"]
    assert calls, (
        "REFUSED: the resumed session never ran the ladder; state_results were "
        "not rebuilt from the journal")
    assert 0.0 < calls[0] <= 3600.0
    for state in ss.STATES:
        assert state in status["states_completed"]
    rebuilt = [r for r in second.read_journal()
               if r["event"] == "STATE_RESULTS_REBUILT"]
    assert rebuilt and "COMPUTE_REMAINING_AUTHORIZED_TIME" in \
        rebuilt[-1]["restored_states"]


def test_the_resumed_allowance_comes_from_the_new_pods_authorization_not_dead_time(
        tmp_path, monkeypatch):
    """NEW contract: eviction dead time is not charged against the new pod.

    The old test asserted ``resumed <= journalled`` after subtracting
    ``time.time() - recorded_at``.  That charged the unbilled
    eviction-to-reacquisition gap and could zero a replacement rental.
    """
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)

    def crashing_ladder(scheduler, ctx):
        raise RuntimeError("crash")

    first = make_supervisor(tmp_path, path, ladder_runner=crashing_ladder)
    first.run(resume=False)

    # a long eviction gap must not zero the replacement pod's allowance
    monkeypatch.setattr(
        "foundation_learner.campaign.session_supervisor.time.time",
        lambda: 2_000_000_000.0)

    second = make_supervisor(tmp_path, path)
    report = second.rebuild_state_results()
    assert report["available_foundation_learner_seconds"] == pytest.approx(
        3600.0, abs=1.0)
    assert report["this_pod_authorized_seconds"] == 3600.0
    assert report["same_pod"] is False
    assert report["resume_wall_clock_gap_seconds"] > 0.0


def test_a_replacement_pod_uses_its_own_remaining_allocation(
        tmp_path, monkeypatch):
    """zero_touch sets O1_SESSION_AUTHORIZED_SECONDS per acquisition as the
    allocation NET of earlier pods' spend.  The config's whole-session figure
    would let a replacement pod re-spend hours the evicted pod already
    billed, and an over-estimated FL budget is the one error that can eat
    the transfer reserve."""
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)

    def crashing_ladder(scheduler, ctx):
        raise RuntimeError("crash")

    monkeypatch.setenv("RUNPOD_POD_ID", "pod-one")
    first = make_supervisor(tmp_path, path, ladder_runner=crashing_ladder)
    first.run(resume=False)

    monkeypatch.setenv("RUNPOD_POD_ID", "pod-two")
    monkeypatch.setenv("O1_SESSION_AUTHORIZED_SECONDS", "1200")
    monkeypatch.setattr(
        "foundation_learner.campaign.session_supervisor.time.time",
        lambda: 2_000_000_000.0)
    second = make_supervisor(tmp_path, path)
    report = second.rebuild_state_results()
    assert report["same_pod"] is False
    assert report["authorized_seconds_source"] == "O1_SESSION_AUTHORIZED_SECONDS"
    assert report["available_foundation_learner_seconds"] == pytest.approx(
        1200.0, abs=1.0)


def test_a_restart_on_the_same_pod_charges_the_gap(tmp_path, monkeypatch):
    """A process crash on the SAME pod is not an eviction: the pod billed
    for every second of the gap, so the allowance must shrink by it."""
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)

    def crashing_ladder(scheduler, ctx):
        raise RuntimeError("crash")

    monkeypatch.setenv("RUNPOD_POD_ID", "pod-one")
    first = make_supervisor(tmp_path, path, ladder_runner=crashing_ladder)
    first.run(resume=False)
    journalled = [r for r in first.read_journal()
                  if r.get("event") == "STATE_COMPLETED"
                  and r.get("state") == "COMPUTE_REMAINING_AUTHORIZED_TIME"]
    assert journalled[-1]["result"]["pod_id"] == "pod-one"

    import time as _time
    now = _time.time()
    monkeypatch.setattr(
        "foundation_learner.campaign.session_supervisor.time.time",
        lambda: now + 1000.0)
    second = make_supervisor(tmp_path, path)
    report = second.rebuild_state_results()
    assert report["same_pod"] is True
    assert report["available_foundation_learner_seconds"] <= 3600.0 - 999.0
    assert report["available_foundation_learner_seconds"] > 0.0


def test_a_garbage_pod_allowance_is_refused_not_guessed(tmp_path, monkeypatch):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    monkeypatch.setenv("O1_SESSION_AUTHORIZED_SECONDS", "lots")
    sup = make_supervisor(tmp_path, path)
    with pytest.raises(ss.SupervisorError):
        sup._pod_authorized_seconds()


def test_a_torn_journal_tail_does_not_strand_the_pod(tmp_path):
    """Eviction mid-write leaves a partial last JSON line.  The supervisor
    must still construct, drop the tail, record it, and resume."""
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)

    def crashing_ladder(scheduler, ctx):
        raise RuntimeError("crash")

    first = make_supervisor(tmp_path, path, ladder_runner=crashing_ladder)
    first.run(resume=False)
    with open(first.journal_path, "a", encoding="utf-8") as fh:
        fh.write('{"event": "STATE_STARTED", "state": "RUN_FL_LA')
    second = make_supervisor(tmp_path, path)
    assert second.torn_journal_tail is not None
    assert "RUN_O1_CALIBRATION" in second.completed
    status = second.run(resume=True)
    events = [r["event"] for r in second.read_journal()]
    assert "JOURNAL_TORN_TAIL_DROPPED" in events
    assert status["close_out"]["terminate"] is not None or \
        "TERMINATE_ACCELERATOR" in status["states_completed"]


def test_a_torn_record_in_the_middle_is_corruption(tmp_path):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    first = make_supervisor(tmp_path, path,
                            ladder_runner=lambda s, c: (_ for _ in ()).throw(
                                RuntimeError("crash")))
    first.run(resume=False)
    lines = open(first.journal_path, encoding="utf-8").read().splitlines()
    lines[1] = lines[1][:10]
    open(first.journal_path, "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")
    with pytest.raises(ss.SupervisorError, match="corruption"):
        make_supervisor(tmp_path, path)


def test_a_constructor_refusal_still_fires_the_terminate_command(
        tmp_path, monkeypatch):
    """A refusal before run() exists (corrupt journal) must not leave the
    accelerator billing until the provider-side terminateAfter."""
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    first = make_supervisor(tmp_path, path,
                            ladder_runner=lambda s, c: (_ for _ in ()).throw(
                                RuntimeError("crash")))
    first.run(resume=False)
    marker.unlink(missing_ok=True)
    lines = open(first.journal_path, encoding="utf-8").read().splitlines()
    lines[1] = lines[1][:10]
    open(first.journal_path, "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")
    monkeypatch.setattr(ss, "build_parser", lambda: _Parser(path, first.out_dir))
    with pytest.raises(ss.SupervisorError):
        ss.main([])
    assert marker.exists(), "terminate_command did not run"


class _Parser:
    def __init__(self, config, out):
        self._config, self._out = config, out

    def parse_args(self, argv):
        import types
        return types.SimpleNamespace(config=str(self._config), out=str(self._out),
                                     no_resume=False)


def test_the_o1_completion_marker_never_reaches_the_log_mid_session(
        tmp_path, capsys):
    """The off-pod driver's completion witness is the literal
    ZERO_TOUCH_COMPLETE in the container log, accepted while the pod is
    RUNNING.  Re-emitting O1's marker verbatim at the end of the O1 phase
    made the driver terminate the pod before any FL stage."""
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    cfg = json.load(open(path, encoding="utf-8"))
    cfg["o1_entry_command"] = [sys.executable, "-c",
                               "print('ZERO_TOUCH_COMPLETE'); "
                               "print('ZERO_TOUCH_ABORTED_AT_X')"]
    json.dump(cfg, open(path, "w", encoding="utf-8"))
    sup = make_supervisor(tmp_path, path)
    sup.state_RUN_O1_CALIBRATION()
    out = capsys.readouterr().out
    assert "O1_PHASE_COMPLETE" in out
    assert "O1_PHASE_ABORTED_AT_X" in out
    # the driver matches SUBSTRINGS of the log: the literal must be absent
    assert "ZERO_TOUCH_COMPLETE" not in out
    assert "ZERO_TOUCH_ABORTED_AT_" not in out
    # the SESSION emits the driver's marker exactly once, at its end
    assert ss._session_marker({"outcome": "COMPLETE"}) == "ZERO_TOUCH_COMPLETE"
    assert ss._session_marker({"outcome": "ABORTED_AT_RUN_FL_LADDER",
                               "failed_state": "RUN_FL_LADDER"}) == \
        "ZERO_TOUCH_ABORTED_AT_RUN_FL_LADDER"


def test_o1_root_discovery_never_forbids_the_shared_checkpoint(tmp_path):
    """O1's transfer manifest lists /artifacts/ouro_rltt_local among its
    artifacts.  Forbidding it passed the (paid) O1 phase and killed
    RELOAD_PRISTINE_OURO."""
    from foundation_learner.campaign import o1_isolation
    guard = o1_isolation.IsolationGuard(label="DISCOVERY")
    manifest = tmp_path / "TRANSFER_MANIFEST.json"
    manifest.write_text(json.dumps({
        "artifacts": {
            "ouro_rltt_checkpoint": {"path": "/artifacts/ouro_rltt_local"},
            "records": {"path": "/outputs/records.jsonl"},
            "fl_out": {"path": str(tmp_path / "session")},
        }}), encoding="utf-8")
    report = guard.discover_from_o1_manifests(
        [str(manifest)], protected=[str(tmp_path / "session")])
    exempt = {e["root"] for e in report["exempted_roots"]}
    assert "/artifacts/ouro_rltt_local" in exempt
    assert str(tmp_path / "session") in exempt
    assert "/outputs/records.jsonl" not in exempt
    # the shared checkpoint is still readable
    guard.guard("/artifacts/ouro_rltt_local/config.json",
                o1_isolation.MODE_READ)


def test_a_resume_refusal_still_terminates_and_emits_a_verdict(
        tmp_path, monkeypatch, capsys):
    """rebuild_state_results() used to run OUTSIDE run()'s try/finally: on a
    replacement pod missing O1_SESSION_AUTHORIZED_SECONDS the refusal
    escaped, terminate_command never fired, and no marker was printed."""
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    cfg = json.load(open(path, encoding="utf-8"))
    cfg["rehearsal"] = False
    cfg["fl_transfer_command"] = [sys.executable, "-c", "print('ft')"]
    cfg["checkpoint_tree_sha256"] = sha256_tree(cfg["checkpoint_dir"])
    json.dump(cfg, open(path, "w", encoding="utf-8"))
    monkeypatch.setenv("O1_SESSION_AUTHORIZED_SECONDS", "3600")
    first = make_supervisor(tmp_path, path, ladder_runner=lambda s, c: (
        _ for _ in ()).throw(RuntimeError("crash")))
    try:
        first.run(resume=False)
    except Exception:
        pass
    marker.unlink(missing_ok=True)
    monkeypatch.delenv("O1_SESSION_AUTHORIZED_SECONDS")
    second = make_supervisor(tmp_path, path)
    status = second.run(resume=True)
    assert status["outcome"].startswith("ABORTED_AT_")
    assert marker.exists(), "terminate_command did not fire on a resume refusal"


def test_mirror_events_during_restore_do_not_defeat_the_torn_tail_repair(
        tmp_path, monkeypatch):
    """A durability event raised during the constructor's restore used to be
    drained into the journal BEFORE the torn-tail repair, burying the torn
    record mid-file (refused as corruption) and restarting the index."""
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    cfg = json.load(open(path, encoding="utf-8"))
    cfg["fl_durable_destination"] = str(tmp_path / "mirror")
    json.dump(cfg, open(path, "w", encoding="utf-8"))
    first = make_supervisor(tmp_path, path, ladder_runner=lambda s, c: (
        _ for _ in ()).throw(RuntimeError("crash")))
    first.run(resume=False)
    with open(first.journal_path, "a", encoding="utf-8") as fh:
        fh.write('{"event": "STATE_STARTED", "state": "RUN_FL_LA\n')
    second = make_supervisor(tmp_path, path)       # must not refuse
    records = second.read_journal()
    indexes = [r["index"] for r in records]
    assert indexes == sorted(indexes) and len(set(indexes)) == len(indexes), (
        "journal index sequence broken by an early drain")
    assert second.torn_journal_tail is not None


def test_a_mirror_outage_before_any_sealed_read_burns_no_attempt(tmp_path):
    from foundation_learner.campaign import sealed_gate as sg
    import foundation_learner.tests.test_campaign_sealed_gate as t

    class DeadMirror:
        def push_file_strict(self, p):
            raise RuntimeError("503 hub down")
        def push_file(self, p):
            pass
        def restore_all(self):
            return {}
        class store:
            @staticmethod
            def list_all():
                return []
        def _rel(self, p):
            return "x/" + os.path.basename(p)

    guard, pregen, out, shard, manifest = t.campaign_dir(tmp_path)
    decisions = t.frozen_decisions(guard, out)
    ledger = os.path.join(out, sg.LEDGER_NAME)
    kwargs = dict(ledger_path=ledger, dev_decisions_path=decisions,
                  split_manifest_path=os.path.join(
                      pregen, "family_split_manifest.json"), guard=guard)
    for _ in range(3):
        try:
            with sg.sealed_opening(**kwargs) as u:
                u.on_durable_strict = DeadMirror().push_file_strict
                u.read_shard(shard)
        except sg.LedgerError:
            pass
        else:
            raise AssertionError("strict push failure did not refuse")
    entries = sg.read_ledger(ledger, guard=guard)
    events = [e["event"] for e in entries]
    assert sg.EVENT_ABORTED not in events, events
    assert events.count(sg.EVENT_INTENT) == events.count(
        sg.EVENT_INTENT_WITHDRAWN) == 3


def test_deterministic_setup_errors_carry_the_marker_and_terminate(
        tmp_path, monkeypatch, capsys):
    """FileNotFound / Permission / ReadOnly are the SAME on every pod: they
    must carry the deterministic marker (silence = eviction = paid
    reacquisition loop).  Only ENOSPC/EIO/ENOMEM/connection errnos are
    transient."""
    import errno
    assert ss._transient_setup_error(OSError(errno.ENOSPC, "full"))
    assert ss._transient_setup_error(MemoryError())
    assert not ss._transient_setup_error(FileNotFoundError(2, "x"))
    assert not ss._transient_setup_error(PermissionError(13, "x"))
    assert not ss._transient_setup_error(OSError(errno.EROFS, "ro"))
    # a config that cannot be LOADED still fires terminate_command from the
    # raw file and prints the marker
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    bad_out = tmp_path / "blocked" / "out"
    (tmp_path / "blocked").mkdir()
    (tmp_path / "blocked").chmod(0o500)
    monkeypatch.setattr(ss, "build_parser", lambda: _Parser(path, bad_out))
    try:
        with pytest.raises(BaseException):
            ss.main([])
    finally:
        (tmp_path / "blocked").chmod(0o700)
    out = capsys.readouterr()
    assert "ZERO_TOUCH_ABORTED_AT_SUPERVISOR_SETUP" in out.out
    assert marker.exists()


def test_drain_failure_keeps_event_names_and_never_duplicates(tmp_path):
    marker = tmp_path / "TERMINATED.log"
    path = fixtures(tmp_path, marker)
    sup = make_supervisor(tmp_path, path)
    sup.durability_events = [{"event": "FL_DURABILITY_PUSH_FAILED", "e": "a"},
                             {"event": "FL_DURABILITY_DEGRADED", "n": 5}]
    calls = {"n": 0}
    real = sup.guard.append_line

    def flaky(path_, line):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "no space")
        return real(path_, line)
    sup.guard.append_line = flaky
    with pytest.raises(OSError):
        sup._drain_durability_events()
    # the unwritten event is re-buffered WITH its name
    assert sup.durability_events == [{"event": "FL_DURABILITY_DEGRADED", "n": 5}]
    sup.guard.append_line = real
    sup._drain_durability_events()
    events = [r["event"] for r in sup.read_journal()]
    assert events.count("FL_DURABILITY_PUSH_FAILED") == 1
    assert events.count("FL_DURABILITY_DEGRADED") == 1


def test_an_exception_after_the_sealed_file_is_opened_is_an_attempt(tmp_path):
    from foundation_learner.campaign import sealed_gate as sg
    import foundation_learner.tests.test_campaign_sealed_gate as t
    guard, pregen, out, shard, manifest = t.campaign_dir(tmp_path)
    decisions = t.frozen_decisions(guard, out)
    ledger = os.path.join(out, sg.LEDGER_NAME)
    kwargs = dict(ledger_path=ledger, dev_decisions_path=decisions,
                  split_manifest_path=os.path.join(
                      pregen, "family_split_manifest.json"), guard=guard)
    # corrupt the shard so decipher/parse fails AFTER the file is opened
    with open(shard, "r+b") as fh:
        fh.seek(0, 2)
        fh.write(b"garbage")
    with pytest.raises(Exception):
        with sg.sealed_opening(**kwargs) as u:
            u.read_shard(shard)
    events = [e["event"] for e in sg.read_ledger(ledger, guard=guard)]
    assert sg.EVENT_ABORTED in events, events
    assert sg.EVENT_INTENT_WITHDRAWN not in events


def test_dangling_intents_pair_by_nonce_not_position(tmp_path):
    from foundation_learner.campaign import sealed_gate as sg
    import foundation_learner.tests.test_campaign_sealed_gate as t
    guard, pregen, out, shard, manifest = t.campaign_dir(tmp_path)
    decisions = t.frozen_decisions(guard, out)
    ledger = os.path.join(out, sg.LEDGER_NAME)
    kwargs = dict(ledger_path=ledger, dev_decisions_path=decisions,
                  split_manifest_path=os.path.join(
                      pregen, "family_split_manifest.json"), guard=guard)
    # pod A: intent, read, SIGKILL (no abort, no withdraw)
    a = sg.open_sealed(**kwargs)
    a.read_shard(shard)
    # pod B: intent then withdrawn (read nothing), twice on the same unlock
    b = sg.open_sealed(**kwargs)
    b.declare_intent()
    assert b.withdraw_intent("x") is not None
    assert b.withdraw_intent("x") is None          # idempotent
    # A's interrupted attempt must still count: one dangling + zero aborted
    c = sg.open_sealed(**kwargs)
    assert c.prior_aborted == 1, c.prior_aborted
