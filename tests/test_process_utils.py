"""Comprehensive tests for agm.core.process utilities."""

from __future__ import annotations

import io
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, cast

import pytest

from agm.core import dry_run
from agm.core.process import (
    _kill_process_group,
    _read_pipe_chunks,
    _run_cleanup_command,
    _terminate_process,
    _write_stream,
    exit_with_output,
    require_capture,
    require_success,
    run_capture,
    run_foreground,
    run_subprocess,
)

# ---------------------------------------------------------------------------
# _write_stream
# ---------------------------------------------------------------------------


class TestWriteStream:
    def test_writes_non_empty_data(self) -> None:
        buf = io.StringIO()
        _write_stream(buf, "hello")
        assert buf.getvalue() == "hello"

    def test_does_nothing_for_empty_string(self) -> None:
        buf = io.StringIO()
        _write_stream(buf, "")
        assert buf.getvalue() == ""

    def test_flushes_after_write(self) -> None:
        flushed: list[bool] = []

        class TrackingStream(io.StringIO):
            def flush(self) -> None:
                flushed.append(True)
                super().flush()

        buf = TrackingStream()
        _write_stream(buf, "data")
        assert flushed, "flush() should have been called"

    def test_does_not_flush_for_empty_data(self) -> None:
        flushed: list[bool] = []

        class TrackingStream(io.StringIO):
            def flush(self) -> None:
                flushed.append(True)
                super().flush()

        buf = TrackingStream()
        _write_stream(buf, "")
        assert not flushed, "flush() should NOT be called for empty data"


# ---------------------------------------------------------------------------
# exit_with_output
# ---------------------------------------------------------------------------


class TestExitWithOutput:
    def test_raises_system_exit_with_given_returncode(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            exit_with_output(42)
        assert exc_info.value.code == 42

    def test_writes_stdout_to_sys_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            exit_with_output(0, stdout="out-text")
        captured = capsys.readouterr()
        assert captured.out == "out-text"

    def test_writes_stderr_to_sys_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            exit_with_output(1, stderr="err-text")
        captured = capsys.readouterr()
        assert captured.err == "err-text"

    def test_writes_both_stdout_and_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            exit_with_output(3, stdout="out", stderr="err")
        captured = capsys.readouterr()
        assert captured.out == "out"
        assert captured.err == "err"
        assert exc_info.value.code == 3

    def test_no_output_when_both_are_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            exit_with_output(1)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_exit_code_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            exit_with_output(0)
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# _terminate_process
# ---------------------------------------------------------------------------


class TestTerminateProcess:
    def test_terminates_running_process(self) -> None:
        proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert proc.poll() is None
            _terminate_process(proc)
            assert proc.poll() is not None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_is_noop_for_already_exited_process(self) -> None:
        proc = subprocess.Popen(
            ["true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        assert proc.poll() is not None
        # Should not raise
        _terminate_process(proc)

    def test_handles_process_that_exits_quickly_after_sigterm(self) -> None:
        # A process that responds quickly to SIGTERM
        sigterm_script = (
            "import signal, time;"
            " signal.signal(signal.SIGTERM, lambda *a: exit(0));"
            " time.sleep(30)"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", sigterm_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert proc.poll() is None
            _terminate_process(proc)
            assert proc.poll() is not None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_kills_process_that_ignores_sigterm(self) -> None:
        # Simulate a process that ignores SIGTERM by having wait always time out.
        class FakeProcess:
            _killed = False

            def poll(self) -> int | None:
                return -9 if self._killed else None

            def terminate(self) -> None:
                pass

            def wait(self, timeout: float | None = None) -> int:
                if timeout is not None:
                    raise subprocess.TimeoutExpired([], timeout)
                return -9

            def kill(self) -> None:
                self._killed = True

        fake = FakeProcess()
        _terminate_process(cast(subprocess.Popen[bytes], fake))
        assert fake._killed

    def test_handles_process_lookup_error_on_terminate(self) -> None:
        class FakeProcess:
            def poll(self) -> int | None:
                return None

            def terminate(self) -> None:
                raise ProcessLookupError

        _terminate_process(cast(subprocess.Popen[bytes], FakeProcess()))

    def test_handles_process_lookup_error_on_kill(self) -> None:
        class FakeProcess:
            def poll(self) -> int | None:
                return None

            def terminate(self) -> None:
                pass

            def wait(self, timeout: float | None = None) -> int:
                if timeout is not None:
                    raise subprocess.TimeoutExpired([], timeout)
                return -9

            def kill(self) -> None:
                raise ProcessLookupError

        _terminate_process(cast(subprocess.Popen[bytes], FakeProcess()))


# ---------------------------------------------------------------------------
# _kill_process_group
# ---------------------------------------------------------------------------


class TestKillProcessGroup:
    def test_kills_process_group_of_running_process(self) -> None:
        proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            assert proc.poll() is None
            _kill_process_group(proc)
            assert proc.poll() is not None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_is_noop_for_already_exited_process(self) -> None:
        proc = subprocess.Popen(
            ["true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        assert proc.poll() is not None
        # Should not raise
        _kill_process_group(proc)

    def test_handles_process_lookup_error_on_killpg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProcess:
            pid = 99999

            def poll(self) -> int | None:
                return None

        def fake_killpg(pgid: int, sig: signal.Signals) -> None:
            raise ProcessLookupError

        monkeypatch.setattr(os, "killpg", fake_killpg)
        _kill_process_group(cast(subprocess.Popen[bytes], FakeProcess()))

    def test_sends_sigkill_when_process_survives_sigterm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        signals_sent: list[signal.Signals] = []

        class FakeProcess:
            pid = 99999

            def poll(self) -> int | None:
                return None

            def wait(self, timeout: float | None = None) -> int:
                if timeout is not None and signal.SIGTERM in signals_sent:
                    raise subprocess.TimeoutExpired([], timeout)
                return -9

        def fake_killpg(pgid: int, sig: signal.Signals) -> None:
            signals_sent.append(sig)

        monkeypatch.setattr(os, "killpg", fake_killpg)
        _kill_process_group(cast(subprocess.Popen[bytes], FakeProcess()))

        assert signal.SIGTERM in signals_sent
        assert signal.SIGKILL in signals_sent

    def test_handles_process_lookup_error_on_sigkill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_count = 0

        class FakeProcess:
            pid = 99999

            def poll(self) -> int | None:
                return None

            def wait(self, timeout: float | None = None) -> int:
                if timeout is not None:
                    raise subprocess.TimeoutExpired([], timeout)
                return -9

        def fake_killpg(pgid: int, sig: signal.Signals) -> None:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise ProcessLookupError

        monkeypatch.setattr(os, "killpg", fake_killpg)
        _kill_process_group(cast(subprocess.Popen[bytes], FakeProcess()))


# ---------------------------------------------------------------------------
# _run_cleanup_command
# ---------------------------------------------------------------------------


class TestRunCleanupCommand:
    def test_none_cmd_is_noop(self) -> None:
        # Should complete without error
        _run_cleanup_command(None)

    def test_runs_real_command_silently(self, tmp_path: Path) -> None:
        sentinel = tmp_path / "ran"
        _run_cleanup_command(
            [sys.executable, "-c", f"open('{sentinel}', 'w').close()"],
            cwd=tmp_path,
        )
        assert sentinel.exists()

    def test_does_not_raise_on_failing_command(self, tmp_path: Path) -> None:
        # Cleanup must not propagate failures
        _run_cleanup_command(["false"], cwd=tmp_path)

    def test_passes_env_to_subprocess(self, tmp_path: Path) -> None:
        sentinel = tmp_path / "env_ok"
        custom_env = os.environ.copy()
        custom_env["_AGM_TEST_VAR"] = "yes"
        script = (
            f"import os; open('{sentinel}', 'w').write(os.environ.get('_AGM_TEST_VAR', ''))"
        )
        _run_cleanup_command(
            [sys.executable, "-c", script],
            env=custom_env,
        )
        assert sentinel.read_text() == "yes"

    def test_uses_os_environ_when_env_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_env: dict[str, Any] = {}

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
            captured_env.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        _run_cleanup_command(["echo", "hi"])
        assert captured_env.get("env") is os.environ


# ---------------------------------------------------------------------------
# _read_pipe_chunks
# ---------------------------------------------------------------------------


class TestReadPipeChunks:
    def test_reads_data_from_pipe_into_queue(self) -> None:
        read_fd, write_fd = os.pipe()
        read_stream = os.fdopen(read_fd, "rb")
        output_queue: queue.Queue[tuple[str, bytes | None]] = queue.Queue()

        thread = threading.Thread(
            target=_read_pipe_chunks,
            kwargs={"stream": read_stream, "name": "stdout", "output_queue": output_queue},
            daemon=True,
        )
        thread.start()

        with os.fdopen(write_fd, "wb") as w:
            w.write(b"hello world")

        thread.join(timeout=2)

        items: list[tuple[str, bytes | None]] = []
        while not output_queue.empty():
            items.append(output_queue.get_nowait())

        data = b"".join(chunk for _, chunk in items if chunk is not None)
        assert data == b"hello world"

    def test_puts_none_sentinel_after_eof(self) -> None:
        read_fd, write_fd = os.pipe()
        read_stream = os.fdopen(read_fd, "rb")
        output_queue: queue.Queue[tuple[str, bytes | None]] = queue.Queue()

        thread = threading.Thread(
            target=_read_pipe_chunks,
            kwargs={"stream": read_stream, "name": "stderr", "output_queue": output_queue},
            daemon=True,
        )
        thread.start()
        os.close(write_fd)
        thread.join(timeout=2)

        items: list[tuple[str, bytes | None]] = []
        while not output_queue.empty():
            items.append(output_queue.get_nowait())

        assert items[-1] == ("stderr", None), "sentinel None must be the last item"

    def test_name_is_preserved_in_queue_entries(self) -> None:
        read_fd, write_fd = os.pipe()
        read_stream = os.fdopen(read_fd, "rb")
        output_queue: queue.Queue[tuple[str, bytes | None]] = queue.Queue()

        thread = threading.Thread(
            target=_read_pipe_chunks,
            kwargs={"stream": read_stream, "name": "my_pipe", "output_queue": output_queue},
            daemon=True,
        )
        thread.start()

        with os.fdopen(write_fd, "wb") as w:
            w.write(b"chunk")

        thread.join(timeout=2)

        items: list[tuple[str, bytes | None]] = []
        while not output_queue.empty():
            items.append(output_queue.get_nowait())

        assert all(name == "my_pipe" for name, _ in items)


# ---------------------------------------------------------------------------
# run_subprocess
# ---------------------------------------------------------------------------


class TestRunSubprocess:
    def test_capture_output_collects_stdout_and_stderr(self) -> None:
        result = run_subprocess(
            [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
            capture_output=True,
        )
        assert result.returncode == 0
        assert result.stdout == "out\n"
        assert result.stderr == "err\n"

    def test_returns_non_zero_returncode(self) -> None:
        result = run_subprocess(
            [sys.executable, "-c", "raise SystemExit(7)"], capture_output=True
        )
        assert result.returncode == 7

    def test_stdout_callback_is_called_with_chunks(self) -> None:
        chunks: list[str] = []
        run_subprocess(
            [sys.executable, "-c", "print('line1'); print('line2')"],
            stdout_callback=chunks.append,
        )
        combined = "".join(chunks)
        assert "line1" in combined
        assert "line2" in combined

    def test_stderr_callback_is_called_with_chunks(self) -> None:
        chunks: list[str] = []
        run_subprocess(
            [sys.executable, "-c", "import sys; print('err', file=sys.stderr)"],
            stderr_callback=chunks.append,
        )
        combined = "".join(chunks)
        assert "err" in combined

    def test_isolate_process_group_sets_start_new_session(
        self, monkeypatch: pytest.MonkeyPatch
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

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        run_subprocess(["echo"], isolate_process_group=True)
        assert popen_kwargs["start_new_session"] is True

    def test_no_process_group_isolation_by_default(
        self, monkeypatch: pytest.MonkeyPatch
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

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        run_subprocess(["echo"])
        assert popen_kwargs.get("start_new_session") is not True

    def test_interrupt_cleanup_cmd_is_run_on_keyboard_interrupt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that interrupt_cleanup_cmd is invoked when a BaseException is raised.

        We monkeypatch _run_cleanup_command at the module level to track calls without
        executing a real subprocess, and send SIGINT to the main process via a background
        thread to trigger the BaseException path inside run_subprocess.
        """
        import agm.core.process as process_module

        cleanup_calls: list[tuple[list[str] | None, Path | None, dict[str, str] | None]] = []

        def tracking_cleanup(
            cmd: list[str] | None,
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> None:
            cleanup_calls.append((cmd, cwd, env))

        monkeypatch.setattr(process_module, "_run_cleanup_command", tracking_cleanup)

        cleanup_cmd = ["echo", "cleanup"]

        def send_interrupt() -> None:
            import time

            time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGINT)

        interrupter = threading.Thread(target=send_interrupt, daemon=True)
        interrupter.start()

        with pytest.raises(KeyboardInterrupt):
            run_subprocess(["sleep", "30"], interrupt_cleanup_cmd=cleanup_cmd)

        interrupter.join(timeout=2)

        assert cleanup_calls, "cleanup command must be invoked on KeyboardInterrupt"
        called_cmd, _, _ = cleanup_calls[0]
        assert called_cmd == cleanup_cmd

    def test_passes_custom_env_to_process(self) -> None:
        custom_env = os.environ.copy()
        custom_env["_AGM_SUBPROCESS_TEST"] = "42"
        result = run_subprocess(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('_AGM_SUBPROCESS_TEST', ''))",
            ],
            capture_output=True,
            env=custom_env,
        )
        assert result.stdout.strip() == "42"

    def test_passes_cwd_to_process(self, tmp_path: Path) -> None:
        result = run_subprocess(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=tmp_path,
            capture_output=True,
        )
        assert result.returncode == 0
        assert str(tmp_path) in result.stdout


# ---------------------------------------------------------------------------
# run_foreground
# ---------------------------------------------------------------------------


class TestRunForeground:
    def test_returns_zero_for_successful_command(self) -> None:
        rc = run_foreground([sys.executable, "-c", "pass"])
        assert rc == 0

    def test_returns_non_zero_for_failing_command(self) -> None:
        rc = run_foreground([sys.executable, "-c", "raise SystemExit(5)"])
        assert rc == 5

    def test_passes_cwd(self, tmp_path: Path) -> None:
        # Use a real check: write a file from the subprocess using relative path
        rc = run_foreground(
            [sys.executable, "-c", "open('marker', 'w').close()"],
            cwd=tmp_path,
        )
        assert rc == 0
        assert (tmp_path / "marker").exists()


# ---------------------------------------------------------------------------
# run_capture
# ---------------------------------------------------------------------------


class TestRunCapture:
    def test_captures_stdout_and_stderr(self) -> None:
        rc, stdout, stderr = run_capture(
            [
                sys.executable,
                "-c",
                "import sys; print('hello'); print('world', file=sys.stderr)",
            ]
        )
        assert rc == 0
        assert stdout == "hello\n"
        assert stderr == "world\n"

    def test_returns_empty_strings_when_no_output(self) -> None:
        rc, stdout, stderr = run_capture([sys.executable, "-c", "pass"])
        assert rc == 0
        assert stdout == ""
        assert stderr == ""

    def test_returns_non_zero_returncode(self) -> None:
        rc, _, _ = run_capture([sys.executable, "-c", "raise SystemExit(3)"])
        assert rc == 3

    def test_stdout_callback_receives_output(self) -> None:
        received: list[str] = []
        run_capture(
            [sys.executable, "-c", "print('cb_test')"],
            stdout_callback=received.append,
        )
        assert "cb_test" in "".join(received)

    def test_stderr_callback_receives_output(self) -> None:
        received: list[str] = []
        run_capture(
            [sys.executable, "-c", "import sys; print('cb_err', file=sys.stderr)"],
            stderr_callback=received.append,
        )
        assert "cb_err" in "".join(received)

    def test_passes_cwd(self, tmp_path: Path) -> None:
        rc, stdout, _ = run_capture(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=tmp_path,
        )
        assert rc == 0
        assert str(tmp_path) in stdout

    def test_isolate_process_group_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        run_capture(["echo"], isolate_process_group=True)
        assert popen_kwargs.get("start_new_session") is True


# ---------------------------------------------------------------------------
# require_success
# ---------------------------------------------------------------------------


class TestRequireSuccess:
    def test_succeeds_silently_when_command_returns_zero(self) -> None:
        require_success([sys.executable, "-c", "pass"])

    def test_raises_system_exit_when_command_fails(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            require_success([sys.executable, "-c", "raise SystemExit(11)"])
        assert exc_info.value.code == 11

    def test_dry_run_prints_command_and_skips_execution(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(dry_run, "_ENABLED", True)
        try:
            require_success(["git", "fetch"])
        finally:
            monkeypatch.setattr(dry_run, "_ENABLED", False)
        captured = capsys.readouterr()
        assert "git fetch" in captured.out
        assert "dry-run" in captured.out

    def test_dry_run_does_not_execute_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import agm.core.process as process_module

        executed: list[bool] = []

        def fake_run_foreground(cmd: list[str], **kwargs: Any) -> int:
            executed.append(True)
            return 0

        monkeypatch.setattr(dry_run, "_ENABLED", True)
        monkeypatch.setattr(process_module, "run_foreground", fake_run_foreground)
        try:
            require_success(["some", "command"])
        finally:
            monkeypatch.setattr(dry_run, "_ENABLED", False)

        assert not executed

    def test_dry_run_with_cwd_shown_in_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(dry_run, "_ENABLED", True)
        try:
            require_success(["make", "build"], cwd=tmp_path)
        finally:
            monkeypatch.setattr(dry_run, "_ENABLED", False)
        captured = capsys.readouterr()
        assert str(tmp_path) in captured.out
        assert "make build" in captured.out

    def test_passes_env_to_run_foreground(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import agm.core.process as process_module

        captured_kwargs: dict[str, Any] = {}

        def fake_run_foreground(cmd: list[str], **kwargs: Any) -> int:
            captured_kwargs.update(kwargs)
            return 0

        monkeypatch.setattr(process_module, "run_foreground", fake_run_foreground)
        custom_env = {"KEY": "val"}
        require_success(["cmd"], env=custom_env)
        assert captured_kwargs.get("env") == custom_env


# ---------------------------------------------------------------------------
# require_capture
# ---------------------------------------------------------------------------


class TestRequireCapture:
    def test_returns_stdout_on_success(self) -> None:
        result = require_capture([sys.executable, "-c", "print('captured')"])
        assert result == "captured\n"

    def test_raises_system_exit_on_non_zero_returncode(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            require_capture(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('out'); print('err', file=sys.stderr); sys.exit(4)",
                ]
            )
        assert exc_info.value.code == 4

    def test_forwards_stdout_on_failure(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            require_capture(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('fail-out'); sys.exit(1)",
                ]
            )
        captured = capsys.readouterr()
        assert "fail-out" in captured.out

    def test_forwards_stderr_on_failure(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            require_capture(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('fail-err', file=sys.stderr); sys.exit(2)",
                ]
            )
        captured = capsys.readouterr()
        assert "fail-err" in captured.err

    def test_passes_cwd(self, tmp_path: Path) -> None:
        result = require_capture(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=tmp_path,
        )
        assert str(tmp_path) in result

    def test_passes_env(self) -> None:
        custom_env = os.environ.copy()
        custom_env["_AGM_REQ_CAP_TEST"] = "yes"
        result = require_capture(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('_AGM_REQ_CAP_TEST', ''))",
            ],
            env=custom_env,
        )
        assert result.strip() == "yes"
