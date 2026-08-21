"""FL off-pod durability under preemption: mirror/restore semantics."""
from __future__ import annotations

import json
import os

import pytest

from foundation_learner.campaign.durability import (
    DurabilityError, FlDurableMirror, _LocalStore, parse_hf_destination,
)


def _write(path, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def test_journal_and_checkpoint_roundtrip_across_pods(tmp_path):
    dest = str(tmp_path / "durable")
    pod1 = str(tmp_path / "pod1_session")
    m1 = FlDurableMirror(dest, pod1)
    journal = os.path.join(pod1, "session_journal.jsonl")
    _write(journal, b'{"event":"STATE_COMPLETED","state":"START_SESSION"}\n')
    m1.push_file(journal)
    payload = os.path.join(pod1, "ladder", "arm_FL2", "ckpt_step120.pt")
    manifest = os.path.join(pod1, "ladder", "arm_FL2", "ckpt_step120.json")
    _write(payload, os.urandom(2048))
    _write(manifest, json.dumps({"tag": "step120",
                                 "arm_config_hash": "abc"}).encode())
    m1.push_checkpoint(payload, manifest)

    # EVICTION: pod1's disk is gone; pod2 restores before reading anything
    pod2 = str(tmp_path / "pod2_session")
    m2 = FlDurableMirror(dest, pod2)
    rep = m2.restore_all()
    assert rep["restored"] == 3 and rep["corrupt_quarantined"] == 0
    got_journal = os.path.join(pod2, "session_journal.jsonl")
    assert os.path.exists(got_journal)
    assert b"STATE_COMPLETED" in open(got_journal, "rb").read()
    assert open(os.path.join(pod2, "ladder", "arm_FL2",
                             "ckpt_step120.pt"), "rb").read() == \
        open(payload, "rb").read()


def test_restore_never_overwrites_local(tmp_path):
    dest = str(tmp_path / "durable")
    pod = str(tmp_path / "session")
    m = FlDurableMirror(dest, pod)
    journal = os.path.join(pod, "session_journal.jsonl")
    _write(journal, b"local-line-1\n")
    m.push_file(journal)
    _write(journal, b"local-line-1\nlocal-line-2\n")   # local advanced
    rep = FlDurableMirror(dest, pod).restore_all()
    assert rep["skipped_existing"] == 1
    assert open(journal, "rb").read() == b"local-line-1\nlocal-line-2\n"


def test_manifest_last_ordering(tmp_path):
    dest = str(tmp_path / "durable")
    pod = str(tmp_path / "session")
    pushed = []

    class RecordingStore(_LocalStore):
        def push(self, local, rel):
            pushed.append(os.path.basename(rel))
            super().push(local, rel)

    m = FlDurableMirror(dest, pod)
    m.store = RecordingStore(dest)
    payload = os.path.join(pod, "ckpt.pt")
    manifest = os.path.join(pod, "ckpt.json")
    _write(payload, b"payload")
    _write(manifest, b"{}")
    m.push_checkpoint(payload, manifest)
    assert pushed == ["ckpt.pt", "ckpt.json"], \
        "manifest must be pushed LAST so a durable manifest implies a " \
        "complete durable checkpoint"


def test_corrupt_mirror_is_quarantined_not_restored(tmp_path):
    dest = str(tmp_path / "durable")
    pod1 = str(tmp_path / "pod1")
    m1 = FlDurableMirror(dest, pod1)
    path = os.path.join(pod1, "receipt.json")
    _write(path, b'{"ok": true}')
    m1.push_file(path)

    class CorruptingStore(_LocalStore):
        def fetch(self, rel, local):
            raise DurabilityError("simulated corruption")

    pod2 = str(tmp_path / "pod2")
    m2 = FlDurableMirror(dest, pod2)
    m2.store = CorruptingStore(dest)
    rep = m2.restore_all()
    assert rep["corrupt_quarantined"] == 1 and rep["restored"] == 0
    assert not os.path.exists(os.path.join(pod2, "receipt.json"))


def test_outside_run_root_refused(tmp_path):
    m = FlDurableMirror(str(tmp_path / "durable"), str(tmp_path / "session"))
    outside = tmp_path / "elsewhere.txt"
    outside.write_bytes(b"x")
    with pytest.raises(DurabilityError):
        m.push_checkpoint(str(outside), str(outside))


def test_push_failure_is_loud_but_not_fatal(tmp_path):
    events = []
    pod = str(tmp_path / "session")
    m = FlDurableMirror(str(tmp_path / "durable"), pod,
                        on_event=lambda e, **kw: events.append(e))

    class FailingStore(_LocalStore):
        def push(self, local, rel):
            raise DurabilityError("simulated outage")

    m.store = FailingStore(str(tmp_path / "durable"))
    path = os.path.join(pod, "session_journal.jsonl")
    _write(path, b"line\n")
    m.push_file(path)          # must not raise
    assert "FL_DURABILITY_PUSH_FAILED" in events


def test_hf_destination_honours_the_documented_sub_path():
    """F4a: hf://ns/repo/fl must not collapse to repo ns/repo."""
    assert parse_hf_destination("hf://ns/repo/fl") == ("ns/repo", "fl")
    assert parse_hf_destination("hf://ns/repo/fl/session") == (
        "ns/repo", "fl/session")
    assert parse_hf_destination("hf://ns/repo") == ("ns/repo", "")


def test_durable_keys_are_scoped_by_session_id_so_two_sessions_do_not_collide(
        tmp_path):
    """F4b: two sessions sharing a dest must not restore each other's journal."""
    dest = str(tmp_path / "durable")
    pod_a = str(tmp_path / "pod_a")
    pod_b = str(tmp_path / "pod_b")
    m_a = FlDurableMirror(dest, pod_a, session_id="SESSION_A")
    journal_a = os.path.join(pod_a, "FL_SESSION_JOURNAL.jsonl")
    _write(journal_a, b'{"session_id":"SESSION_A"}\n')
    m_a.push_file(journal_a)

    m_b = FlDurableMirror(dest, pod_b, session_id="SESSION_B")
    report = m_b.restore_all()
    assert report["restored"] == 0
    assert not os.path.exists(os.path.join(pod_b, "FL_SESSION_JOURNAL.jsonl"))


def test_hf_subpath_and_session_id_appear_in_the_remote_key(tmp_path):
    dest = str(tmp_path / "unused-local")
    pod = str(tmp_path / "session")
    # local dest has no hf sub-path; construct via the hf parser + _rel
    m = FlDurableMirror(dest, pod, session_id="SID1")
    journal = os.path.join(pod, "FL_SESSION_JOURNAL.jsonl")
    _write(journal, b"x\n")
    assert m._rel(journal) == "fl_durable/SID1/FL_SESSION_JOURNAL.jsonl"
    m_hf = FlDurableMirror("hf://ns/repo/fl", pod, session_id="SID1")
    assert m_hf._rel(journal) == "fl/fl_durable/SID1/FL_SESSION_JOURNAL.jsonl"
    assert m_hf.store.repo_id == "ns/repo"
    assert m_hf.store.list_prefix == "fl/fl_durable/SID1"


def test_restore_refuses_a_path_escape(tmp_path):
    """A remote key like fl_durable/../../x must not write outside run_root."""
    dest = str(tmp_path / "durable")
    pod = str(tmp_path / "session")
    outside = tmp_path / "escaped.txt"

    class EscapingStore(_LocalStore):
        def list_all(self):
            return ["fl_durable/../../escaped.txt"]

        def fetch(self, rel, local):
            os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
            with open(local, "wb") as fh:
                fh.write(b"escaped")

    m = FlDurableMirror(dest, pod)
    m.store = EscapingStore(dest)
    with pytest.raises(DurabilityError, match="path escape"):
        m.restore_all()
    assert not outside.exists()
