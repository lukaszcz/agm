"""Helpers for process-oriented tests."""

from __future__ import annotations

import time
from pathlib import Path


def wait_for_path(path: Path, *, timeout: float = 30.0) -> None:
    """Block until *path* exists, or raise ``AssertionError`` after *timeout*."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")
