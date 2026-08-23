"""Preemptible-eviction signal plumbing for the pod entry.

Deliberately OUTSIDE ``campaign/``: this is platform plumbing, not campaign
logic.  It reads ``/proc`` for process introspection and never touches a
campaign data path, so routing it through the O1 isolation guard -- which
exists to keep campaign file access away from O1's roots -- would be
meaningless.  Keeping it here preserves the campaign package's invariant
that every ``open()`` in it is guarded.
"""
from __future__ import annotations

import os
import signal
import time


class EvictedBySignal(RuntimeError):
    """SIGTERM reached the pod: a preemptible eviction, not a defect."""


def direct_children(pid: int) -> list[int]:
    """PIDs whose parent is ``pid``, read from /proc.  Best effort."""
    kids: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return kids
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                fields = fh.read().rsplit(b")", 1)[-1].split()
            if int(fields[1]) == pid:
                kids.append(int(entry))
        except (OSError, ValueError, IndexError):
            continue
    return kids


def install_eviction_handler(grace_seconds: float = 5.0) -> None:
    """Turn the eviction SIGTERM into an ordinary exception, and pass it on.

    In a COMBINED session PID 1 is the FL supervisor, not the O1 child, and a
    container SIGTERM is delivered to PID 1 only.  So the O1 child's own
    eviction flush could never fire, and the supervisor died on the default
    handler -- running no ``finally``, flushing nothing, and losing up to
    RECORDS_SYNC_EVERY_ROWS committed rows per eviction.  Forward the signal
    to the children first, give them a moment to flush, then raise so the
    supervisor's protected close-out runs.
    """
    def _on_sigterm(signum, _frame):
        try:
            signal.signal(signum, signal.SIG_IGN)      # never re-enter
        except (ValueError, OSError):
            pass
        for kid in direct_children(os.getpid()):
            try:
                os.kill(kid, signal.SIGTERM)
            except OSError:
                pass
        # Reap as we wait.  A SIGTERM'd child becomes a ZOMBIE with its
        # ppid intact -- nothing reaps it, because the parent is sitting in
        # this handler -- so polling direct_children() alone always burned
        # the full grace period out of a possibly-30-s eviction window.
        deadline = time.time() + grace_seconds
        while time.time() < deadline:
            try:
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
            except (ChildProcessError, OSError):
                pass
            if not direct_children(os.getpid()):
                break
            time.sleep(0.2)
        raise EvictedBySignal(
            f"SIGTERM (signal {signum}): the pod is being evicted")

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        pass        # not the main thread, or unsupported platform
