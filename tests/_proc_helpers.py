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


def wait_for_signal_ignored_script(pid: int | str, signum: int) -> str:
    """Shell snippet: block until *pid* has *signum* set in its `/proc` `SigIgn` mask.

    *pid* is a literal pid or a shell expression (e.g. ``'"$TARGET"'``) that
    evaluates to one. For a test child that signals *pid* right after
    starting: a fixed-delay child would race whatever installs the ignore in
    *pid* (e.g. a ``run_foreground_ignoring_signals`` call, which only starts
    ignoring once its own child is spawned); polling the real kernel-reported
    disposition instead makes the handshake exact.
    """
    bit = 1 << (signum - 1)
    return (
        f"t={pid}\n"
        "while :; do\n"
        '  m=$(sed -n "s/^SigIgn:[[:space:]]*//p" /proc/$t/status)\n'
        f"  [ $(( 0x$m & {bit} )) -ne 0 ] && break\n"
        "done\n"
    )
