"""An eviction must never be reported as a deterministic defect.

The first version of this handler turned the container SIGTERM into a plain
RuntimeError.  That reached ``_session_marker`` as
``ZERO_TOUCH_ABORTED_AT_<state>``, which the off-pod driver reads as
``DeterministicPodFailure``: it refuses to reacquire AND writes a durable
refusal marker keyed to the deployment.  So a routine spot eviction at hour
four killed the session, wasted the spend, produced no results, and blocked
every future run until a human deleted a file -- strictly worse than the
default SIGTERM disposition it replaced.

These tests pin the two properties that failure violated.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from foundation_learner.campaign.session_supervisor import (
    TransientO1Failure, _session_marker)
from foundation_learner.eviction import (
    EvictedBySignal, direct_children, install_eviction_handler)


def _marker_for(exc):
    transient = isinstance(exc, (TransientO1Failure, EvictedBySignal))
    return _session_marker({"transient": transient,
                            "outcome": "ABORTED_AT_RUN_FL_LADDER",
                            "failed_state": "RUN_FL_LADDER"})


def test_an_eviction_is_reported_transient_so_the_driver_reacquires():
    marker = _marker_for(EvictedBySignal("SIGTERM (signal 15)"))
    assert marker.startswith("REFUSED (transient)"), marker
    assert "ZERO_TOUCH_ABORTED_AT_" not in marker


def test_a_genuine_defect_is_still_reported_deterministic():
    """The transient path must not become a blanket amnesty."""
    marker = _marker_for(RuntimeError("a real bug in the ladder"))
    assert marker == "ZERO_TOUCH_ABORTED_AT_RUN_FL_LADDER", marker


def test_the_scheduler_does_not_record_an_eviction_as_a_failed_stage():
    """STAGE_FAILED is TERMINAL: a replacement pod would SKIP the stage.

    Leaving STAGE_STARTED with no terminal event is what marks it for re-run,
    so the eviction must propagate out of _run_stage untouched.
    """
    import inspect

    from foundation_learner.campaign import scheduler as sched

    src = inspect.getsource(sched.Scheduler.run_stage)
    assert "except EvictedBySignal" in src, (
        "run_stage must let an eviction propagate; catching it under the "
        "generic BaseException handler journals a TERMINAL STAGE_FAILED and "
        "the replacement pod then skips a paid-for ladder stage")
    assert src.index("except EvictedBySignal") < src.index(
        "except BaseException"), (
        "the eviction clause must precede the generic handler or it never "
        "runs")


def test_sigterm_is_forwarded_to_children_then_raised():
    kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert kid.pid in direct_children(os.getpid())
        install_eviction_handler(grace_seconds=3.0)
        with pytest.raises(EvictedBySignal):
            os.kill(os.getpid(), signal.SIGTERM)
        # the child got the signal too: on a real pod this is the only way
        # the O1 child's own eviction flush ever fires, because the container
        # SIGTERM is delivered to PID 1 (this supervisor) alone
        assert kid.poll() is not None or kid.wait(timeout=5) is not None
    finally:
        if kid.poll() is None:
            kid.kill()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
