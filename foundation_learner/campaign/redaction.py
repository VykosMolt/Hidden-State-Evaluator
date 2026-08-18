"""Secret redaction for FL text that leaves the process.

Deliberately self-contained: the isolation contract forbids importing any O1
module, so the equivalent O1 helper cannot be reused here.

FL error text reaches container stdout and the supervisor journal, and the
journal is mirrored to the FL results repository.  Nothing observed today
echoes a token into an exception message, so this is a missing defence
rather than a demonstrated leak — which is exactly when it is cheap to add.
"""
from __future__ import annotations

import os
import re

__all__ = ["redact", "register_secret", "SECRET_ENV_NAMES"]

#: Environment variables whose VALUES must never appear in emitted text.
SECRET_ENV_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "RUNPOD_API_KEY")

#: Token shapes, redacted whether or not the value is in this environment —
#: a credential echoed by a remote service was never in os.environ here.
_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\brpa_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
)

_REGISTERED: set[str] = set()


def register_secret(value: str | None) -> None:
    """Add a literal value to the redaction set (never logged itself)."""
    if value and len(value) >= 8:
        _REGISTERED.add(value)


def redact(text: str) -> str:
    if not text:
        return text
    for name in SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value and len(value) >= 8:
            text = text.replace(value, "[REDACTED]")
    for value in _REGISTERED:
        text = text.replace(value, "[REDACTED]")
    for pattern in _PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text
