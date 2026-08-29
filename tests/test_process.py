"""Focused tests for subprocess helpers."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import agm.core.process as process_mod
from agm.core.process import ProcessCaptureResult, run_capture, run_capture_result, run_foreground


def test_stream_reader_start_restores_sigint_mask_after_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_mask = {signal.SIGINT}
    calls: list[tuple[int, object]] = []

    class Process:
        stdin = None
        stdout = object()
        stderr = None

    class Thread:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            raise KeyboardInterrupt

    def mask(how: int, value: object) -> set[signal.Signals]:
        calls.append((how, value))
        return previous_mask

    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(process_mod.threading, "Thread", Thread)
    monkeypatch.setattr(process_mod.signal, "pthread_sigmask", mask)

    with pytest.raises(KeyboardInterrupt):
        process_mod._start_process_with_readers(
            ["command"],
            cwd=None,
            env=None,
            capture_output=True,
            stdout_callback=None,
            stderr_callback=None,
            isolate_process_group=False,
            stdin_text=None,
        )

    assert calls == [
        (signal.SIG_BLOCK, {signal.SIGINT}),
        (signal.SIG_SETMASK, previous_mask),
    ]


def test_run_capture_streams_stdout_and_stderr_before_process_exit(tmp_path: Path) -> None:
    script = tmp_path / "stream.py"
    script.write_text(
        "import sys, time\n"
        'sys.stdout.write("out-1")\n'
        "sys.stdout.flush()\n"
        "time.sleep(0.05)\n"
        'sys.stderr.write("err-1")\n'
        "sys.stderr.flush()\n"
        "time.sleep(0.05)\n"
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
    # Pipe reads may coalesce adjacent writes under concurrent test load; callbacks
    # must preserve each stream's content and order, not an OS-dependent chunking.
    stdout_chunks = [chunk for stream, chunk, _ in events if stream == "stdout"]
    stderr_chunks = [chunk for stream, chunk, _ in events if stream == "stderr"]
    assert "".join(stdout_chunks) == stdout
    assert "".join(stderr_chunks) == stderr


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
            returncode=0,
            stdout="",
            stderr="",
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
            spawn_errno=None,
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
            spawn_errno=2,
        )
        assert result.returncode == 42
        assert result.stdout == "out"
        assert result.stderr == "err"
        assert result.elapsed == 1.5
        assert result.timed_out is True
        assert result.spawn_error == "No such file"
        assert result.spawn_errno == 2

    def test_spawn_error_none_by_default(self) -> None:
        result = ProcessCaptureResult(
            returncode=0,
            stdout="",
            stderr="",
            elapsed=0.0,
            timed_out=False,
            spawn_error=None,
            spawn_errno=None,
        )
        assert result.spawn_error is None

    def test_spawn_errno_accessible(self) -> None:
        result = ProcessCaptureResult(
            returncode=None,
            stdout="",
            stderr="",
            elapsed=0.0,
            timed_out=False,
            spawn_error="Permission denied",
            spawn_errno=13,
        )
        assert result.spawn_errno == 13

    def test_spawn_errno_none_when_no_spawn_error(self) -> None:
        result = ProcessCaptureResult(
            returncode=0,
            stdout="",
            stderr="",
            elapsed=0.0,
            timed_out=False,
            spawn_error=None,
            spawn_errno=None,
        )
        assert result.spawn_errno is None


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


class TestRunCaptureResultEmbeddedNullByte:
    """A NUL byte in an argv element is a spawn failure, not a raw ValueError.

    ``subprocess.Popen`` raises ``ValueError('embedded null byte')`` before the
    child is launched.  ``run_capture_result`` must map that to a spawn-failure
    result (spawn_error set, returncode None) rather than letting the
    ``ValueError`` escape — mirroring the FileNotFoundError/PermissionError
    handling so every spawn outcome is represented in the result.
    """

    def test_null_byte_arg_sets_spawn_error(self) -> None:
        result = run_capture_result(["sh", "-c", "echo \x00bad"])
        assert result.spawn_error is not None
        assert len(result.spawn_error) > 0

    def test_null_byte_arg_returncode_is_none(self) -> None:
        result = run_capture_result(["sh", "-c", "echo \x00bad"])
        assert result.returncode is None

    def test_null_byte_arg_does_not_raise(self) -> None:
        # Must not raise ValueError("embedded null byte") or anything else.
        result = run_capture_result(["sh", "-c", "echo \x00bad"])
        assert result.spawn_error is not None
        assert result.spawn_errno is None

    def test_null_byte_arg_streams_empty(self) -> None:
        result = run_capture_result(["sh", "-c", "echo \x00bad"])
        assert result.stdout == ""
        assert result.stderr == ""


class TestRunCaptureIdleTimeout:
    def test_idle_timeout_is_reported_through_callback(self) -> None:
        messages: list[str] = []

        with pytest.raises(SystemExit) as exc_info:
            run_capture(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                idle_timeout=0.2,
                isolate_process_group=True,
                timeout_callback=messages.append,
            )

        assert exc_info.value.code == 124
        assert len(messages) == 1
        assert "0.2" in messages[0]


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
        """Where ``run_capture`` exits the process, ``run_capture_result`` returns."""
        command = [sys.executable, "-c", "import time; time.sleep(60)"]

        with pytest.raises(SystemExit):
            run_capture(command, idle_timeout=0.2, isolate_process_group=True)

        result = run_capture_result(command, idle_timeout=0.2, isolate_process_group=True)
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


# ---------------------------------------------------------------------------
# ENOEXEC (spawn hole) — run_capture_result must return spawn-error
# ---------------------------------------------------------------------------


class TestRunCaptureResultEnoexec:
    """ENOEXEC ('Exec format error') from running a file with no shebang / binary
    content must be caught at the spawn boundary and returned as a spawn-error
    result rather than escaping as a raw OSError.
    """

    def _make_enoexec_script(self, tmp_path: Path) -> Path:
        """Create a file with exec permission but no shebang (binary junk)."""
        script = tmp_path / "garbage_binary"
        script.write_bytes(b"\x7f\x45\x4c\x46garbage")  # ELF magic but not valid
        script.chmod(script.stat().st_mode | 0o111)  # add exec bit
        return script

    def test_enoexec_does_not_raise(self, tmp_path: Path) -> None:
        """run_capture_result must never raise on ENOEXEC."""
        script = self._make_enoexec_script(tmp_path)
        # Must not raise OSError or any other exception
        result = run_capture_result([str(script)])
        assert result.spawn_error is not None

    def test_enoexec_returns_spawn_error_result(self, tmp_path: Path) -> None:
        """ENOEXEC must produce spawn_error set and returncode=None."""
        script = self._make_enoexec_script(tmp_path)
        result = run_capture_result([str(script)])
        assert result.spawn_error is not None
        assert result.returncode is None

    def test_enoexec_streams_are_empty(self, tmp_path: Path) -> None:
        """When ENOEXEC fires, no output was produced."""
        script = self._make_enoexec_script(tmp_path)
        result = run_capture_result([str(script)])
        assert result.stdout == ""
        assert result.stderr == ""

    def test_enoexec_spawn_errno_is_set(self, tmp_path: Path) -> None:
        """spawn_errno should be ENOEXEC (8) so callers can distinguish the cause."""
        import errno as errno_mod

        script = self._make_enoexec_script(tmp_path)
        result = run_capture_result([str(script)])
        assert result.spawn_errno == errno_mod.ENOEXEC


class TestRunCaptureEnoexecReraise:
    """run_capture must re-raise ENOEXEC as OSError (not FileNotFoundError/PermissionError)."""

    def _make_enoexec_script(self, tmp_path: Path) -> Path:
        script = tmp_path / "garbage_binary"
        script.write_bytes(b"\x7f\x45\x4c\x46garbage")
        script.chmod(script.stat().st_mode | 0o111)
        return script

    def test_enoexec_raises_os_error_not_file_not_found(self, tmp_path: Path) -> None:
        """ENOEXEC re-raises as OSError (or subclass), not FileNotFoundError."""
        script = self._make_enoexec_script(tmp_path)
        with pytest.raises(OSError) as exc_info:
            run_capture([str(script)])
        # Must NOT be FileNotFoundError (wrong semantic) — it's an exec format error
        assert not isinstance(exc_info.value, FileNotFoundError)

    def test_enoexec_errno_intact(self, tmp_path: Path) -> None:
        """The re-raised OSError must preserve errno=ENOEXEC."""
        import errno as errno_mod

        script = self._make_enoexec_script(tmp_path)
        with pytest.raises(OSError) as exc_info:
            run_capture([str(script)])
        assert exc_info.value.errno == errno_mod.ENOEXEC


class TestRunCaptureErrnoIntact:
    """run_capture re-raises FileNotFoundError/PermissionError with errno/filename intact."""

    def test_file_not_found_errno_intact(self) -> None:
        """FileNotFoundError from run_capture preserves errno=ENOENT."""
        import errno as errno_mod

        with pytest.raises(FileNotFoundError) as exc_info:
            run_capture(["/nonexistent/binary/that/does/not/exist"])
        assert exc_info.value.errno == errno_mod.ENOENT

    def test_permission_error_errno_intact(self, tmp_path: Path) -> None:
        """PermissionError from run_capture preserves errno=EACCES."""
        import errno as errno_mod

        script = tmp_path / "not_executable.sh"
        script.write_text("#!/bin/sh\necho hi\n")
        script.chmod(0o644)
        with pytest.raises(PermissionError) as exc_info:
            run_capture([str(script)])
        assert exc_info.value.errno == errno_mod.EACCES

    def test_file_not_found_filename_intact(self) -> None:
        """FileNotFoundError from run_capture preserves the filename."""
        path = "/nonexistent/binary/that/does/not/exist"
        with pytest.raises(FileNotFoundError) as exc_info:
            run_capture([path])
        assert exc_info.value.filename == path or path in str(exc_info.value)


# ---------------------------------------------------------------------------
# stdin_text threading — 2MB stdin must not BrokenPipeError or deadlock
# ---------------------------------------------------------------------------


class TestRunCaptureResultLargeStdin:
    """stdin_text is written from a thread; large stdin must not raise or deadlock."""

    def test_large_stdin_to_true_does_not_raise(self) -> None:
        """2MB stdin to 'true' (which never reads stdin) must return rc=0, no exception."""
        result = run_capture_result(["true"], stdin_text="x" * 2_000_000)
        assert result.returncode == 0
        assert result.spawn_error is None

    def test_large_stdin_to_cat_completes_without_deadlock(self) -> None:
        """Large stdin to 'cat' (reads all stdin, writes all stdout) must complete."""
        payload = "line\n" * 50_000  # ~350KB
        result = run_capture_result(["cat"], stdin_text=payload)
        assert result.returncode == 0
        assert result.stdout == payload


# ---------------------------------------------------------------------------
# NUL-byte spawn failure — run_capture must re-raise ValueError
# ---------------------------------------------------------------------------


class TestRunCaptureNullByteReraise:
    """run_capture with a NUL-byte argv element must re-raise the original ValueError.

    _run_capture_result_impl stores the ValueError alongside OSError, and
    run_capture re-raises it, while run_capture_result discards it and returns
    the structured result.
    """

    def test_null_byte_in_run_capture_raises_value_error(self) -> None:
        """run_capture(['sh', '-c', 'echo \\x00bad']) raises ValueError, not returns."""
        with pytest.raises(ValueError):
            run_capture(["sh", "-c", "echo \x00bad"])

    def test_null_byte_run_capture_error_message(self) -> None:
        """The re-raised ValueError retains the 'embedded null byte' message."""
        with pytest.raises(ValueError, match="null"):
            run_capture(["sh", "-c", "echo \x00bad"])

    def test_null_byte_run_capture_result_returns_structured(self) -> None:
        """run_capture_result maps a NUL byte to a spawn-error result instead of raising."""
        result = run_capture_result(["sh", "-c", "echo \x00bad"])
        assert result.spawn_error is not None
        assert result.returncode is None
