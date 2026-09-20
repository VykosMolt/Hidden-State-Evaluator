"""Pytest session hygiene for the project test suite."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SKIP_DIRS = {".git", ".venv", "venv", "artifacts", "data", "models"}

# Keep the working tree clean when running the project suite. This affects only
# the current Python process and child imports made after this conftest loads.
sys.dont_write_bytecode = True


def _clear_repo_pycache() -> None:
    for root, dirs, _files in PROJECT_ROOT.walk(top_down=True):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        if root.name == "__pycache__":
            shutil.rmtree(root, ignore_errors=True)
            dirs[:] = []


def pytest_sessionstart(session) -> None:  # noqa: ANN001
    _clear_repo_pycache()


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001
    _clear_repo_pycache()
