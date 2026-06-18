"""Streaming helpers for agent command output."""

from __future__ import annotations

import sys
from collections.abc import Callable

StreamCallback = Callable[[str], None]


def write_stdout(chunk: str) -> None:
    if not chunk:
        return
    sys.stdout.write(chunk)
    sys.stdout.flush()


def write_stderr(chunk: str) -> None:
    if not chunk:
        return
    sys.stderr.write(chunk)
    sys.stderr.flush()
