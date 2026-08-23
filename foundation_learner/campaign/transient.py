"""Transient-vs-deterministic discrimination for the FL pre-entry steps.

The FL pre-entry modules (``check_hf_scope``, ``fetch_pregen``) were forks of
the O1 ones with this machinery removed.  Without it, ONE Hugging Face 503,
connection reset or rate-limit during the 480 MB pregen download made
``fl_b200_entry.sh`` stamp ``ZERO_TOUCH_ABORTED_AT_FL_PRE_ENTRY_*``, which the
off-pod driver records as a PERMANENT, deployment-wide refusal -- so a network
hiccup, on a step that runs on EVERY pod and EVERY reacquisition immediately
after the image pull and the 5 GB checkpoint fetch were paid for, locked the
deployment until a human deleted a file.

401/403/404 repeat on every pod and stay deterministic.  Everything else --
5xx, resets, timeouts, rate limits -- is a hub or network condition worth a
retry, and if it persists, one a fresh pod may not see.

Deliberately a local copy of the O1 rule rather than an import: the campaign
package must not depend on the O1 package (contract 13 isolation).
"""
from __future__ import annotations

import re
import subprocess
import sys
import time

__all__ = ["ATTEMPTS", "DETERMINISTIC_STATUSES", "TransientStepError",
           "http_status", "run_helper_with_retry"]

#: HTTP statuses that mean "this will fail identically on a fresh pod".
DETERMINISTIC_STATUSES = (401, 403, 404)

#: Attempts per step, with exponential backoff capped at 30 s.
ATTEMPTS = 4

_STATUS_RE = re.compile(r"\b(4\d{2}|5\d{2})\b")


class TransientStepError(RuntimeError):
    """A pre-entry step failed for a reason a replacement pod may not see."""


def http_status(text: str) -> int | None:
    m = _STATUS_RE.search(text or "")
    return int(m.group(1)) if m else None


def run_helper_with_retry(args: list[str], timeout: float, *,
                          deterministic_error, label: str,
                          redact=lambda s: s, helper_error=lambda s: s,
                          attempts: int = ATTEMPTS, sleep=time.sleep,
                          child_env=None):
    """Run the hf_transfer helper, retrying only transient failures.

    ``deterministic_error`` is the exception TYPE raised when the failure
    would repeat on any pod.  A transient failure raises TransientStepError,
    which the caller reports as ``REFUSED (transient)`` -- the token the pod
    entry and the driver both read as "reacquire", never as a defect.

    One wall-clock deadline spans ALL attempts: four full timeouts in a
    reset-at-90% pattern would otherwise burn four fetch windows of paid time.
    """
    last = ""
    proc = None
    deadline = time.monotonic() + timeout
    for attempt in range(1, attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TransientStepError(
                f"{last or label}: aggregate deadline ({timeout:.0f}s) "
                f"exhausted after {attempt - 1} attempts")
        try:
            proc = subprocess.run(
                [sys.executable, "-m",
                 "foundation_learner.campaign.hf_transfer", *args],
                capture_output=True, text=True, timeout=remaining,
                env=child_env)
        except subprocess.TimeoutExpired:
            # a stalled download is NOT a deployment defect; it was
            # previously uncaught and bricked the deployment after burning
            # the whole timeout on paid time
            last = f"{label}: timed out after {remaining:.0f}s"
            if attempt < attempts:
                sleep(min(30.0, 2.0 ** attempt))
                continue
            raise TransientStepError(f"{last} (after {attempts} attempts)")
        if proc.returncode == 0:
            return proc
        combined = (proc.stdout or "") + (proc.stderr or "")
        last = redact(f"{label}: {helper_error(combined)}")
        if http_status(combined) in DETERMINISTIC_STATUSES:
            raise deterministic_error(last)
        if attempt < attempts:
            sleep(min(30.0, 2.0 ** attempt))
    raise TransientStepError(f"{last} (after {attempts} attempts)")
