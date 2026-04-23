"""Focused tests for subprocess helpers."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from agm.core.process import run_capture


def test_run_capture_streams_stdout_and_stderr_before_process_exit(tmp_path: Path) -> None:
    script = tmp_path / "stream.py"
    script.write_text(
        "import sys, time\n"
        'sys.stdout.write("out-1")\n'
        "sys.stdout.flush()\n"
        "time.sleep(0.5)\n"
        'sys.stderr.write("err-1")\n'
        "sys.stderr.flush()\n"
        "time.sleep(0.5)\n"
        'sys.stdout.write("out-2\\n")\n'
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )

    events: list[tuple[str, str, float]] = []
    started = time.monotonic()

    returncode, stdout, stderr = run_capture(
        [sys.executable, str(script)],
        stdout_callback=lambda chunk: events.append(("stdout", chunk, time.monotonic() - started)),
        stderr_callback=lambda chunk: events.append(("stderr", chunk, time.monotonic() - started)),
    )

    assert returncode == 0
    assert stdout == "out-1out-2\n"
    assert stderr == "err-1"
    assert events
    assert ("stdout", "out-1", events[0][2]) == events[0]
    assert events[0][2] < 0.4
    assert any(
        stream == "stderr" and chunk == "err-1" and elapsed < 0.9
        for stream, chunk, elapsed in events
    )
