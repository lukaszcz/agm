"""Helpers for exercising interrupt/cleanup behavior deterministically.

These helpers replace fixed ``time.sleep`` races with readiness-file
synchronization: a child process announces it is running by creating a file,
and the test waits for that file before delivering a signal.  The parent
reaches its blocking wait (``process.wait`` or the stream-drain loop) long
before the freshly spawned child finishes interpreter startup and creates the
file, so a signal sent after the file appears is guaranteed to land while the
parent is blocked inside the interrupt-protected region.

This keeps the interrupt tests reliable even when several test suites run
concurrently (e.g. across git worktrees) and the machine is heavily
oversubscribed — a situation in which a fixed sleep fires before the parent has
reached the protected region.
"""

from __future__ import annotations

import signal
import sys
import threading
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


def ready_then_sleep_command(ready_file: Path, *, seconds: int = 30) -> list[str]:
    """Return a child command that creates *ready_file*, then sleeps.

    Used as the subprocess in interrupt tests: once *ready_file* exists the
    parent is known to be blocked waiting on the child.
    """

    return [
        sys.executable,
        "-c",
        "import sys, time; open(sys.argv[1], 'w').close(); time.sleep(int(sys.argv[2]))",
        str(ready_file),
        str(seconds),
    ]


def interrupt_self_when_ready(ready_file: Path) -> threading.Thread:
    """Start a daemon thread that sends ``SIGINT`` to the main thread.

    The signal is delivered only after *ready_file* appears, guaranteeing the
    main thread is already blocked inside ``run_subprocess``' protected region.

    ``pthread_kill`` targets the main thread specifically rather than the
    process: a process-directed signal (``os.kill``) may be handled by any
    thread with ``SIGINT`` unblocked, which under load can leave the main
    thread's blocking ``wait``/``get`` uninterrupted so no ``KeyboardInterrupt``
    is ever raised.
    """

    main_thread_id = threading.main_thread().ident
    assert main_thread_id is not None

    def _run() -> None:
        wait_for_path(ready_file)
        signal.pthread_kill(main_thread_id, signal.SIGINT)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
