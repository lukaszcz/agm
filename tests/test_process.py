"""Focused tests for subprocess helpers."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from agm.core.process import ProcessCaptureResult, run_capture, run_capture_result, run_foreground


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


# ---------------------------------------------------------------------------
# ProcessCaptureResult + run_capture_result
# ---------------------------------------------------------------------------


class TestProcessCaptureResultDataclass:
    """ProcessCaptureResult is a frozen dataclass with the expected fields."""

    def test_is_frozen(self) -> None:
        result = ProcessCaptureResult(
            returncode=0, stdout="", stderr="", elapsed=0.1, timed_out=False, spawn_error=None
        )
        with pytest.raises((AttributeError, TypeError)):
            setattr(result, "returncode", 1)

    def test_all_fields_accessible(self) -> None:
        result = ProcessCaptureResult(
            returncode=42,
            stdout="out",
            stderr="err",
            elapsed=1.5,
            timed_out=True,
            spawn_error="No such file",
        )
        assert result.returncode == 42
        assert result.stdout == "out"
        assert result.stderr == "err"
        assert result.elapsed == 1.5
        assert result.timed_out is True
        assert result.spawn_error == "No such file"

    def test_spawn_error_none_by_default(self) -> None:
        result = ProcessCaptureResult(
            returncode=0, stdout="", stderr="", elapsed=0.0, timed_out=False, spawn_error=None
        )
        assert result.spawn_error is None


class TestRunCaptureResultZeroExit:
    """Zero-exit, no output: returncode=0, empty streams, elapsed>0, not timed_out."""

    def test_zero_exit_empty_output(self) -> None:
        result = run_capture_result([sys.executable, "-c", "pass"])
        assert result.returncode == 0
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.elapsed > 0
        assert result.timed_out is False
        assert result.spawn_error is None

    def test_elapsed_is_positive(self) -> None:
        result = run_capture_result([sys.executable, "-c", "pass"])
        assert result.elapsed > 0


class TestRunCaptureResultSeparateStreams:
    """stdout and stderr are captured separately."""

    def test_separate_stdout_and_stderr(self) -> None:
        result = run_capture_result(
            [
                sys.executable,
                "-c",
                "import sys; print('hello'); print('world', file=sys.stderr)",
            ]
        )
        assert result.returncode == 0
        assert result.stdout == "hello\n"
        assert result.stderr == "world\n"

    def test_stdout_only(self) -> None:
        result = run_capture_result([sys.executable, "-c", "print('only-out')"])
        assert result.stdout == "only-out\n"
        assert result.stderr == ""

    def test_stderr_only(self) -> None:
        result = run_capture_result(
            [sys.executable, "-c", "import sys; print('only-err', file=sys.stderr)"]
        )
        assert result.stdout == ""
        assert result.stderr == "only-err\n"


class TestRunCaptureResultNonzeroExit:
    """Nonzero exit is represented in returncode; no exception raised."""

    def test_nonzero_returncode(self) -> None:
        result = run_capture_result([sys.executable, "-c", "raise SystemExit(7)"])
        assert result.returncode == 7
        assert result.timed_out is False
        assert result.spawn_error is None

    def test_nonzero_does_not_raise(self) -> None:
        # Must not raise SystemExit or any other exception
        result = run_capture_result([sys.executable, "-c", "raise SystemExit(99)"])
        assert result.returncode == 99


class TestRunCaptureResultStdin:
    """stdin_text is forwarded to the process."""

    def test_stdin_text_is_passed(self) -> None:
        result = run_capture_result(
            [sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
            stdin_text="hello from stdin",
        )
        assert result.returncode == 0
        assert "hello from stdin" in result.stdout


class TestRunCaptureResultSpawnError:
    """Nonexistent binary yields spawn_error set, returncode=None, no raise."""

    def test_nonexistent_binary_sets_spawn_error(self) -> None:
        result = run_capture_result(["/nonexistent/binary/that/does/not/exist"])
        assert result.spawn_error is not None
        assert len(result.spawn_error) > 0

    def test_nonexistent_binary_returncode_is_none(self) -> None:
        result = run_capture_result(["/nonexistent/binary/that/does/not/exist"])
        assert result.returncode is None

    def test_nonexistent_binary_does_not_raise(self) -> None:
        # Must not raise FileNotFoundError or anything else
        result = run_capture_result(["/nonexistent/binary/that/does/not/exist"])
        assert result.spawn_error is not None

    def test_spawn_error_streams_empty(self) -> None:
        result = run_capture_result(["/nonexistent/binary/that/does/not/exist"])
        assert result.stdout == ""
        assert result.stderr == ""


class TestRunCaptureResultIdleTimeout:
    """Idle timeout fires: timed_out=True, returncode reflects kill, no SystemExit raised."""

    def test_idle_timeout_sets_timed_out(self) -> None:
        result = run_capture_result(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            idle_timeout=0.2,
            isolate_process_group=True,
        )
        assert result.timed_out is True

    def test_idle_timeout_does_not_raise_system_exit(self) -> None:
        # Must return a result, never raise SystemExit
        result = run_capture_result(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            idle_timeout=0.2,
            isolate_process_group=True,
        )
        assert result.timed_out is True

    def test_idle_timeout_returncode_is_set(self) -> None:
        result = run_capture_result(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            idle_timeout=0.2,
            isolate_process_group=True,
        )
        # returncode is the kill exit code (nonzero), not None
        assert result.returncode is not None
        assert result.returncode != 0

    def test_idle_timeout_elapsed_is_positive(self) -> None:
        result = run_capture_result(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            idle_timeout=0.3,
            isolate_process_group=True,
        )
        assert result.elapsed > 0

    def test_process_with_output_survives_idle_timeout(self) -> None:
        """A process that keeps producing output within the timeout window completes."""
        script = (
            "import time, sys\n"
            "for i in range(3):\n"
            "    print(f'chunk {i}')\n"
            "    sys.stdout.flush()\n"
            "    time.sleep(0.1)\n"
        )
        result = run_capture_result(
            [sys.executable, "-c", script],
            idle_timeout=1.0,
            isolate_process_group=True,
        )
        assert result.returncode == 0
        assert result.timed_out is False
        assert "chunk 0" in result.stdout


class TestRunCaptureResultCwdEnv:
    """cwd and env are forwarded to the process."""

    def test_cwd_is_set(self, tmp_path: Path) -> None:
        result = run_capture_result(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=tmp_path,
        )
        assert result.returncode == 0
        assert str(tmp_path) in result.stdout

    def test_env_is_forwarded(self) -> None:
        custom_env = os.environ.copy()
        custom_env["_AGM_RCR_TEST"] = "magic42"
        result = run_capture_result(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('_AGM_RCR_TEST', ''))",
            ],
            env=custom_env,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "magic42"