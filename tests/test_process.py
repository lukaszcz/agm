"""Focused tests for subprocess helpers."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest

from agm.core.process import run_capture, run_foreground


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


def test_run_foreground_preserves_controlling_terminal_for_interactive_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_kwargs: dict[str, Any] = {}

    class FakeProcess:
        stdout = None
        stderr = None
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

    def fake_popen(cmd: list[str], **kwargs: Any) -> FakeProcess:
        popen_kwargs.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    assert run_foreground(["git", "fetch"]) == 0

    assert popen_kwargs.get("stdin") is None
    assert popen_kwargs.get("start_new_session") is not True
    assert "process_group" not in popen_kwargs


def test_run_foreground_can_isolate_process_group_for_tree_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_kwargs: dict[str, Any] = {}

    class FakeProcess:
        stdout = None
        stderr = None
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

    def fake_popen(cmd: list[str], **kwargs: Any) -> FakeProcess:
        popen_kwargs.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    assert run_foreground(["runner"], isolate_process_group=True) == 0

    assert popen_kwargs["start_new_session"] is True
