"""Transient-vs-deterministic classification for the FL pre-entry steps.

This module ships on the money path and had NO test. It was written to stop
one Hugging Face hiccup from bricking the deployment, and its first version
inverted itself: the status regex was ``\\b(4\\d{2}|5\\d{2})\\b``, which matches
the LINE NUMBER in a traceback frame (``line 409, in hf_raise_for_status``)
because that appears before the exception line. A genuine 401 therefore read
as 409 -> "transient" -> retried, then reported transient -> the driver
reacquires -> a bad token becomes several paid pods. A genuine 5xx could read
as 404 -> "deterministic" -> a permanent, deployment-wide refusal marker.

The fixtures below use REAL huggingface_hub traceback text, because synthetic
one-line messages are exactly what let that through.
"""
from __future__ import annotations

import subprocess

import pytest

from foundation_learner.campaign import transient as tr

_TRACEBACK = '''Traceback (most recent call last):
  File "/opt/venv/lib/python3.14/site-packages/huggingface_hub/utils/_http.py", line 409, in hf_raise_for_status
    response.raise_for_status()
  File "/opt/venv/lib/python3.14/site-packages/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
huggingface_hub.errors.HfHubHTTPError: {status} for url: https://huggingface.co/api/datasets/Vykos/o1-b200-staging
'''


@pytest.mark.parametrize("status_line,expected,deterministic", [
    ("401 Client Error: Unauthorized", 401, True),
    ("403 Client Error: Forbidden", 403, True),
    ("404 Client Error: Not Found", 404, True),
    ("429 Client Error: Too Many Requests", 429, False),
    ("500 Server Error: Internal Server Error", 500, False),
    ("503 Server Error: Service Unavailable", 503, False),
])
def test_status_is_read_from_the_status_line_not_a_traceback_line_number(
        status_line, expected, deterministic):
    text = _TRACEBACK.format(status=status_line)
    assert "line 409" in text, "fixture must contain a decoy line number"
    assert tr.http_status(text) == expected
    assert (tr.http_status(text) in tr.DETERMINISTIC_STATUSES) is deterministic


def test_a_connection_reset_carries_no_status_and_is_therefore_transient():
    text = "ConnectionResetError: [Errno 104] Connection reset by peer"
    assert tr.http_status(text) is None
    assert tr.http_status(text) not in tr.DETERMINISTIC_STATUSES


def test_the_last_status_wins_when_a_helper_printed_several():
    text = (_TRACEBACK.format(status="503 Server Error: Service Unavailable")
            + _TRACEBACK.format(status="401 Client Error: Unauthorized"))
    assert tr.http_status(text) == 401


class _Proc:
    def __init__(self, returncode, stderr=""):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = stderr


def _fake_run(results):
    calls = {"n": 0}

    def run(*a, **k):
        calls["n"] += 1
        item = results[min(calls["n"] - 1, len(results) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item
    return run, calls


class _Det(RuntimeError):
    pass


def test_a_deterministic_failure_refuses_immediately_without_retrying(
        monkeypatch):
    """401 repeats on every pod: retrying wastes paid time and, worse,
    ends in a transient verdict that makes the driver reacquire."""
    run, calls = _fake_run([_Proc(1, _TRACEBACK.format(
        status="401 Client Error: Unauthorized"))])
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(_Det):
        tr.run_helper_with_retry(["scope"], 60.0, deterministic_error=_Det,
                                 label="scope", sleep=lambda s: None)
    assert calls["n"] == 1, "a deterministic failure must not be retried"


def test_a_transient_failure_retries_then_reports_transient(monkeypatch):
    run, calls = _fake_run([_Proc(1, _TRACEBACK.format(
        status="503 Server Error: Service Unavailable"))])
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(tr.TransientStepError):
        tr.run_helper_with_retry(["snapshot"], 60.0, deterministic_error=_Det,
                                 label="fetch", sleep=lambda s: None)
    assert calls["n"] == tr.ATTEMPTS


def test_a_transient_failure_that_then_succeeds_returns_the_success(
        monkeypatch):
    ok = _Proc(0)
    run, calls = _fake_run([_Proc(1, _TRACEBACK.format(
        status="503 Server Error: Service Unavailable")), ok])
    monkeypatch.setattr(subprocess, "run", run)
    got = tr.run_helper_with_retry(["snapshot"], 60.0,
                                   deterministic_error=_Det, label="fetch",
                                   sleep=lambda s: None)
    assert got is ok and calls["n"] == 2


def test_a_stalled_download_is_transient_not_a_deployment_defect(monkeypatch):
    """TimeoutExpired used to escape uncaught: it bricked the deployment
    AFTER burning the whole download window on paid time."""
    run, _ = _fake_run([subprocess.TimeoutExpired(cmd="hf", timeout=1.0)])
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(tr.TransientStepError):
        tr.run_helper_with_retry(["snapshot"], 60.0, deterministic_error=_Det,
                                 label="fetch", sleep=lambda s: None)


def test_the_aggregate_deadline_bounds_all_attempts(monkeypatch):
    """One wall-clock deadline across every attempt: four full timeouts in a
    reset-at-90% pattern would otherwise burn four download windows."""
    clock = {"t": 0.0}
    monkeypatch.setattr(tr.time, "monotonic", lambda: clock["t"])
    run, calls = _fake_run([_Proc(1, "ConnectionResetError: reset by peer")])
    monkeypatch.setattr(subprocess, "run", run)

    def advance(_seconds):
        clock["t"] += 40.0
    with pytest.raises(tr.TransientStepError, match="deadline"):
        tr.run_helper_with_retry(["snapshot"], 60.0, deterministic_error=_Det,
                                 label="fetch", sleep=advance)
    assert calls["n"] < tr.ATTEMPTS, "the deadline must cut the attempts short"
