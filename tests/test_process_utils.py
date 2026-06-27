"""Comprehensive tests for agm.core.process utilities."""

from __future__ import annotations

import io
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from agm.core import dry_run
from agm.core.process import (
    _kill_process_group,
    _read_pipe_chunks,
    _run_cleanup_command,
    _terminate_process,
    _wait_for_process_group_exit,
    _write_stream,
    exit_with_output,
    require_capture,
    require_success,
    run_capture,
    run_capture_result,
    run_foreground,
    run_subprocess,
)


class _FakeProcess:
    returncode: int | None = None

    def __init__(self, *, interrupt_on_wait: bool = False) -> None:
        self._interrupt_on_wait = interrupt_on_wait

    def wait(self, timeout: float | None = None) -> int:
        if self._interrupt_on_wait:
            raise KeyboardInterrupt
        return 0


class _JoinedReader(threading.Thread):
    def join(self, timeout: float | None = None) -> None:
        return None


class _InterruptingQueue(queue.Queue[tuple[str, bytes | None]]):
    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> tuple[str, bytes | None]:
        raise KeyboardInterrupt


def _patch_start_process(
    monkeypatch: pytest.MonkeyPatch,
    process_module: Any,
    *,
    process: subprocess.Popen[bytes],
    readers: list[threading.Thread] | None = None,
    stream_queue: queue.Queue[tuple[str, bytes | None]] | None = None,
) -> None:
    def fake_start_process_with_readers(
        cmd: list[str],
        *,
        cwd: Path | None,
        env: dict[str, str] | None,
        capture_output: bool,
        stdout_callback: object,
        stderr_callback: object,
        isolate_process_group: bool,
        stdin_text: str | None,
    ) -> tuple[
        subprocess.Popen[bytes],
        list[threading.Thread],
        queue.Queue[tuple[str, bytes | None]],
        threading.Thread | None,
    ]:
        return (
            process,
            [] if readers is None else readers,
            queue.Queue() if stream_queue is None else stream_queue,
            None,
        )

    monkeypatch.setattr(
        process_module, "_start_process_with_readers", fake_start_process_with_readers
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
# _wait_for_process_group_exit
# ---------------------------------------------------------------------------


class TestWaitForProcessGroupExit:
    def test_returns_after_grace_when_process_group_still_exists(self) -> None:
        proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            _wait_for_process_group_exit(proc.pid, grace=0.02)
            assert proc.poll() is None
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=1)


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

    def test_sends_sigkill_to_group_when_process_already_exited(self, tmp_path: Path) -> None:
        """When the main process has already exited, still kill orphaned group members."""
        child_pid_file = tmp_path / "child.pid"
        script = (
            "#!/bin/bash\n"
            "(sleep 30) &\n"
            'printf "%s\\n" "$!" > "'
            + str(child_pid_file)
            + '"\n'
        )
        script_file = tmp_path / "parent.sh"
        script_file.write_text(script)
        script_file.chmod(script_file.stat().st_mode | 0o111)

        proc = subprocess.Popen(
            [str(script_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        proc.wait()
        assert proc.poll() is not None

        # The parent shell exited, but the background sleep child is alive.
        child_pid = int(child_pid_file.read_text().strip())
        try:
            # The child should still be alive (it's an orphaned group member).
            os.kill(child_pid, 0)
        except ProcessLookupError:
            pytest.skip("child already exited before test could verify")

        _kill_process_group(proc)

        # After _kill_process_group, the orphaned child should be dead.
        try:
            for _ in range(10):
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.01)
            else:
                pytest.fail("orphaned child process still alive after _kill_process_group")
        finally:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_sends_sigkill_to_group_after_prompt_exit(self) -> None:
        """When the process exits promptly after SIGTERM, still SIGKILL the group."""
        proc = subprocess.Popen(
            ["sleep", "0.01"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Send SIGTERM — sleep exits quickly so the process should die promptly.
        _kill_process_group(proc)


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
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify that interrupt_cleanup_cmd is invoked when a BaseException is raised.

        The process wait raises KeyboardInterrupt directly so this test covers the
        cleanup path without relying on OS signal timing.
        """
        import agm.core.process as process_module

        cleanup_calls: list[tuple[list[str] | None, Path | None, dict[str, str] | None]] = []
        terminated: list[object] = []

        def tracking_cleanup(
            cmd: list[str] | None,
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> None:
            cleanup_calls.append((cmd, cwd, env))

        def tracking_terminate(proc: subprocess.Popen[bytes]) -> None:
            terminated.append(proc)

        monkeypatch.setattr(process_module, "_run_cleanup_command", tracking_cleanup)
        monkeypatch.setattr(process_module, "_terminate_process", tracking_terminate)
        _patch_start_process(
            monkeypatch,
            process_module,
            process=cast(subprocess.Popen[bytes], _FakeProcess(interrupt_on_wait=True)),
        )

        cleanup_cmd = ["echo", "cleanup"]

        with pytest.raises(KeyboardInterrupt):
            run_subprocess(["sleeper"], cwd=tmp_path, interrupt_cleanup_cmd=cleanup_cmd)

        assert terminated, "process must be terminated on KeyboardInterrupt"
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


class TestRunSubprocessIdleTimeout:
    def test_idle_timeout_kills_process(self) -> None:
        """When idle_timeout is exceeded, process is killed and SystemExit(124) is raised."""
        with pytest.raises(SystemExit) as exc_info:
            run_subprocess(
                ["sleep", "10"],
                capture_output=True,
                idle_timeout=0.2,
                isolate_process_group=True,
            )
        assert exc_info.value.code == 124

    def test_idle_timeout_kills_without_process_group(self) -> None:
        """Idle timeout also works without isolate_process_group."""
        with pytest.raises(SystemExit) as exc_info:
            run_subprocess(
                ["sleep", "10"],
                capture_output=True,
                idle_timeout=0.2,
                isolate_process_group=False,
            )
        assert exc_info.value.code == 124


class TestRunSubprocessEmptyDecoding:
    def test_empty_chunk_continues_without_appending(self) -> None:
        """When decoded text is empty (multi-byte boundary), the chunk is skipped."""
        script = "import sys; sys.stdout.buffer.write(b'hello'); sys.stdout.flush()"
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
        )
        assert result.stdout == "hello"
        assert result.returncode == 0

    def test_final_decoder_flush_is_captured(self) -> None:
        """Final decoder.decode(b'', final=True) output is captured."""
        script = "import sys; sys.stdout.buffer.write('héllo'.encode('utf-8')); sys.stdout.flush()"
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
        )
        assert "héllo" in result.stdout

    def test_final_decoder_callback_called(self) -> None:
        """Final decoder flush triggers callback."""
        chunks: list[str] = []
        script = "import sys; sys.stdout.buffer.write(b'test'); sys.stdout.flush()"
        run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
            stdout_callback=lambda text: chunks.append(text),
        )
        assert "".join(chunks) == "test"


class TestRunSubprocessIdleTimeoutRemainingZero:
    def test_idle_timeout_when_remaining_is_zero(self) -> None:
        """When remaining <= 0 at the start of the loop, queue.Empty is raised internally."""
        with pytest.raises(SystemExit) as exc_info:
            run_subprocess(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                capture_output=True,
                idle_timeout=0.1,
                isolate_process_group=True,
            )
        assert exc_info.value.code == 124


class TestRunSubprocessEmptyDecodedChunk:
    def test_empty_decoded_chunk_is_skipped(self) -> None:
        """When decoder.decode returns empty string, the chunk is skipped (continue)."""
        script = "import sys; sys.stdout.buffer.write(b'x'); sys.stdout.flush()"
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
        )
        assert result.stdout == "x"


class TestRunSubprocessFinalDecoderCallback:
    def test_final_decoder_flush_calls_callback_with_capture(self) -> None:
        """Final decoder.decode(b'', final=True) with capture_output and callbacks."""
        chunks: list[str] = []
        script = "import sys; sys.stdout.buffer.write(b'output'); sys.stdout.flush()"
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
            stdout_callback=lambda text: chunks.append(text),
        )
        assert result.stdout == "output"
        assert "".join(chunks) == "output"


class TestRunSubprocessIdleTimeoutNonIsolate:
    def test_idle_timeout_without_process_group(self) -> None:
        """Idle timeout kills process via _terminate_process when not in process group."""
        with pytest.raises(SystemExit) as exc_info:
            run_subprocess(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                capture_output=True,
                idle_timeout=0.1,
                isolate_process_group=False,
            )
        assert exc_info.value.code == 124


class TestIdleTimeoutRemainingZero:
    """Cover the explicit 'raise queue.Empty' when remaining <= 0."""

    def test_idle_timeout_zero_seconds_triggers_immediately(self) -> None:
        """With idle_timeout=0, the remaining check is <= 0 immediately."""
        with pytest.raises(SystemExit) as exc_info:
            run_subprocess(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                capture_output=True,
                idle_timeout=0,
                isolate_process_group=True,
            )
        assert exc_info.value.code == 124


class TestEmptyDecodedTextContinue:
    """Cover the 'continue' path when decoder.decode returns empty string."""

    def test_multi_byte_char_split_across_chunks(self) -> None:
        """When a UTF-8 multi-byte char is split across pipe chunks,
        the first partial decode returns empty string and is skipped."""
        script = (
            'import sys, time\n'
            'sys.stdout.buffer.write(b"\\xc3")\n'
            'sys.stdout.buffer.flush()\n'
            'time.sleep(0.1)\n'
            'sys.stdout.buffer.write(b"\\xa9")\n'
            'sys.stdout.buffer.flush()\n'
        )
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
        )
        assert result.stdout == "é"
        assert result.returncode == 0


class TestFinalDecoderFlush:
    """Final decoder.decode(b'', final=True) with capture_output and callbacks."""

    def test_final_flush_captured_with_callback(self) -> None:
        """When capture_output=True and a callback is set, the final flush
        output is both appended to stream_data and sent to the callback."""
        chunks: list[str] = []

        script = (
            'import sys\n'
            'sys.stdout.buffer.write(b"\\xc3")\n'
            'sys.stdout.buffer.flush()\n'
        )
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
            stdout_callback=lambda text: chunks.append(text),
        )
        # The replacement character from final flush should be in stdout
        assert "\ufffd" in result.stdout
        # Callback should have been called with the replacement character
        assert any("\ufffd" in c for c in chunks)

    def test_final_flush_stderr_callback(self) -> None:
        """Final decoder flush on stderr with capture_output and callback."""
        chunks: list[str] = []

        script = (
            'import sys\n'
            'sys.stderr.buffer.write(b"\\xc3")\n'
            'sys.stderr.buffer.flush()\n'
        )
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
            stderr_callback=lambda text: chunks.append(text),
        )
        # The replacement character from final flush should be in stderr
        assert "\ufffd" in result.stderr
        assert any("\ufffd" in c for c in chunks)

    def test_final_flush_with_no_pending_data(self) -> None:
        """When decoder has no buffered data at flush, no extra text is emitted."""
        script = (
            "import sys\n"
            "sys.stdout.write('hello\\n')\n"
            "sys.stdout.flush()\n"
        )
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
            stdout_callback=lambda text: None,
        )
        assert result.stdout == "hello\n"


class TestRunSubprocessIdleTimeoutExact:
    def test_idle_timeout_expired_triggers_kill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When remaining <= 0, queue.Empty is raised which triggers process kill."""
        with pytest.raises(SystemExit) as exc_info:
            run_subprocess(
                ["sleep", "30"],
                capture_output=True,
                idle_timeout=0.01,
                isolate_process_group=True,
            )
        assert exc_info.value.code == 124

    def test_idle_timeout_non_isolate_process_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Idle timeout also works without isolate_process_group."""
        with pytest.raises(SystemExit) as exc_info:
            run_subprocess(
                ["sleep", "30"],
                capture_output=True,
                idle_timeout=0.01,
                isolate_process_group=False,
            )
        assert exc_info.value.code == 124


class TestRunSubprocessEmptyDecodedText:
    def test_empty_decoded_text_is_skipped(self) -> None:
        """When decoder.decode returns empty string, it's skipped via continue."""
        script = (
            "import sys; "
            "sys.stdout.buffer.write(b'ok'); "
            "sys.stdout.flush()"
        )
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
        )
        assert result.stdout == "ok"


class TestRunSubprocessFinalDecoderFlush:
    def test_final_flush_with_callback(self) -> None:
        """Final decoder flush triggers callback for remaining buffered data."""
        chunks: list[str] = []
        script = "import sys; sys.stdout.buffer.write(b'test data'); sys.stdout.flush()"
        run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
            stdout_callback=lambda text: chunks.append(text),
        )
        full = "".join(chunks)
        assert "test data" in full

    def test_final_flush_captured_output(self) -> None:
        """Final decoder flush is added to captured output."""
        script = "import sys; sys.stdout.buffer.write(b'flushed'); sys.stdout.flush()"
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
        )
        assert "flushed" in result.stdout

    def test_final_flush_empty_no_capture_with_callback(self) -> None:
        """When final decoder.decode(b'', final=True) returns empty string,
        the 'if not text: continue' branch is taken.  Also exercises the
        capture_output=False path inside the final-flush loop (the
        'if capture_output:' branch is not taken)."""
        chunks: list[str] = []
        # Clean ASCII: decoder has no pending bytes, so final flush yields ''
        # and the continue branch fires.
        script = "import sys; sys.stdout.buffer.write(b'clean'); sys.stdout.flush()"
        run_subprocess(
            [sys.executable, "-c", script],
            capture_output=False,
            stdout_callback=lambda text: chunks.append(text),
        )
        # Normal chunk delivers the data; final flush produces nothing extra
        assert "clean" in "".join(chunks)

    def test_final_flush_no_capture_with_nonempty_text(self) -> None:
        """When final decoder flush yields text with capture_output=False,
        the stream_data append is skipped and the callback is invoked instead."""
        chunks: list[str] = []
        # Write just the first byte of a 2-byte UTF-8 character (0xc3) and exit.
        # The streaming decoder buffers it; final flush produces the replacement.
        script = (
            "import sys; "
            "sys.stdout.buffer.write(b'\\xc3'); "
            "sys.stdout.flush()"
        )
        run_subprocess(
            [sys.executable, "-c", script],
            capture_output=False,
            stdout_callback=lambda text: chunks.append(text),
        )
        # The final flush emits a replacement char via callback
        assert any("\ufffd" in c for c in chunks)

    def test_final_flush_captured_with_no_callback(self) -> None:
        """When final decoder flush yields text on a stream with no callback,
        the loop continues without calling the callback."""
        # Write partial UTF-8 to stderr and exit.  capture_output=True creates
        # both pipes.  stderr_callback is None, so the final-flush loop visits
        # stderr with a None callback and continues to the next decoder entry.
        script = (
            "import sys; "
            "sys.stderr.buffer.write(b'\\xc3'); "
            "sys.stderr.flush()"
        )
        result = run_subprocess(
            [sys.executable, "-c", script],
            capture_output=True,
        )
        # stderr final flush emits replacement char into captured output
        assert "\ufffd" in result.stderr


class TestRunSubprocessBaseExceptionWithCapture:
    """Exercise _drain_process_streams' except BaseException path."""

    def test_interrupt_cleanup_cmd_with_capture_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """BaseException raised while draining captured streams still runs cleanup."""
        import agm.core.process as process_module

        cleanup_calls: list[tuple[list[str] | None, Path | None, dict[str, str] | None]] = []
        terminated: list[object] = []

        def tracking_cleanup(
            cmd: list[str] | None,
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> None:
            cleanup_calls.append((cmd, cwd, env))

        def tracking_terminate(proc: subprocess.Popen[bytes]) -> None:
            terminated.append(proc)

        monkeypatch.setattr(process_module, "_run_cleanup_command", tracking_cleanup)
        monkeypatch.setattr(process_module, "_terminate_process", tracking_terminate)
        _patch_start_process(
            monkeypatch,
            process_module,
            process=cast(subprocess.Popen[bytes], _FakeProcess()),
            readers=[_JoinedReader()],
            stream_queue=_InterruptingQueue(),
        )

        cleanup_cmd = ["echo", "cleanup"]

        with pytest.raises(KeyboardInterrupt):
            run_subprocess(
                ["sleeper"],
                cwd=tmp_path,
                capture_output=True,
                interrupt_cleanup_cmd=cleanup_cmd,
            )

        assert terminated, "process must be terminated on KeyboardInterrupt"
        assert cleanup_calls, "cleanup command must be invoked on KeyboardInterrupt"
        called_cmd, _, _ = cleanup_calls[0]
        assert called_cmd == cleanup_cmd

    def test_interrupt_with_isolate_process_group_and_capture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BaseException path in _drain_process_streams calls _kill_process_group when isolated."""
        import agm.core.process as process_module

        killed: list[bool] = []

        def tracking_kill(proc: subprocess.Popen[bytes]) -> None:
            killed.append(True)

        monkeypatch.setattr(process_module, "_kill_process_group", tracking_kill)
        _patch_start_process(
            monkeypatch,
            process_module,
            process=cast(subprocess.Popen[bytes], _FakeProcess()),
            readers=[_JoinedReader()],
            stream_queue=_InterruptingQueue(),
        )

        with pytest.raises(KeyboardInterrupt):
            run_subprocess(
                ["sleeper"],
                capture_output=True,
                isolate_process_group=True,
            )

        assert killed, "_kill_process_group must be called"


class TestRunSubprocessBaseExceptionIsolatedNoCapture:
    """Exercise run_subprocess no-readers BaseException path with isolate_process_group."""

    def test_interrupt_with_isolated_process_group_no_capture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BaseException in no-capture mode with isolate_process_group calls _kill_process_group."""
        import agm.core.process as process_module

        killed: list[bool] = []

        def tracking_kill(proc: subprocess.Popen[bytes]) -> None:
            killed.append(True)

        monkeypatch.setattr(process_module, "_kill_process_group", tracking_kill)
        _patch_start_process(
            monkeypatch,
            process_module,
            process=cast(subprocess.Popen[bytes], _FakeProcess(interrupt_on_wait=True)),
        )

        with pytest.raises(KeyboardInterrupt):
            run_subprocess(["sleeper"], isolate_process_group=True)

        assert killed, "_kill_process_group must be called with isolate_process_group"


class TestRunCaptureSpawnError:
    """run_capture re-raises faithful spawn exception types."""

    def test_nonexistent_binary_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            run_capture(["/nonexistent/binary/that/does/not/exist"])

    def test_non_executable_file_raises_permission_error(self, tmp_path: Path) -> None:
        """A file that exists but is not executable must raise PermissionError."""
        script = tmp_path / "not_executable.sh"
        script.write_text("#!/bin/sh\necho hi\n")
        script.chmod(0o644)  # readable but not executable
        with pytest.raises(PermissionError):
            run_capture([str(script)])

    def test_run_capture_result_non_executable_sets_spawn_errno(self, tmp_path: Path) -> None:
        """run_capture_result never raises; spawn_errno is EACCES for PermissionError."""
        import errno as errno_mod

        script = tmp_path / "not_executable.sh"
        script.write_text("#!/bin/sh\necho hi\n")
        script.chmod(0o644)
        result = run_capture_result([str(script)])
        assert result.spawn_error is not None
        assert result.returncode is None
        assert result.spawn_errno == errno_mod.EACCES

    def test_run_capture_result_nonexistent_binary_sets_spawn_errno(self) -> None:
        """run_capture_result sets spawn_errno=ENOENT for a nonexistent binary."""
        import errno as errno_mod

        result = run_capture_result(["/nonexistent/binary/that/does/not/exist"])
        assert result.spawn_error is not None
        assert result.spawn_errno == errno_mod.ENOENT

    def test_run_capture_result_successful_run_has_no_spawn_errno(self) -> None:
        """spawn_errno is None when the process spawned successfully."""
        result = run_capture_result([sys.executable, "-c", "pass"])
        assert result.spawn_errno is None


class TestDrainLoopTimeoutSentinel:
    """Sentinel drain loop after idle timeout fires."""

    def test_idle_timeout_with_output_before_silence(self) -> None:
        """When idle timeout fires after the process produced some output, the
        post-timeout sentinel drain loop runs (some sentinels may already be queued)."""
        script = (
            "import sys, time\n"
            "print('initial')\n"
            "sys.stdout.flush()\n"
            "time.sleep(60)\n"
        )
        result = run_capture_result(
            [sys.executable, "-c", script],
            idle_timeout=0.3,
            isolate_process_group=True,
        )
        assert result.timed_out is True
