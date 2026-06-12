"""Process execution helpers."""

from __future__ import annotations

import codecs
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import IO, TextIO

from agm.core import dry_run


def _write_stream(stream: TextIO, data: str) -> None:
    if data:
        stream.write(data)
        stream.flush()


def exit_with_output(returncode: int, stdout: str = "", stderr: str = "") -> None:
    """Forward captured output and exit with *returncode*."""

    _write_stream(sys.stdout, stdout)
    _write_stream(sys.stderr, stderr)
    raise SystemExit(returncode)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    try:
        process.terminate()
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            return
        process.wait()


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    if process.poll() is not None:
        # The main process already exited, but other members of its process
        # group may still be alive.  Give them a brief moment to tear down
        # after the SIGTERM we just sent, then SIGKILL any stragglers.
        _wait_for_process_group_exit(process.pid, grace=0.2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return

    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return

    # The main process exited promptly.  Give remaining group members a
    # brief moment to exit as well, then SIGKILL any stragglers.
    _wait_for_process_group_exit(process.pid, grace=0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _wait_for_process_group_exit(pgid: int, *, grace: float) -> None:
    """Poll until the process group no longer exists or *grace* seconds elapse."""
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return  # group is gone — nothing left to clean up
        time.sleep(0.01)


def _run_cleanup_command(
    cmd: list[str] | None,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    if cmd is None:
        return

    subprocess.run(
        cmd,
        cwd=cwd,
        env=os.environ if env is None else env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _read_pipe_chunks(
    stream: IO[bytes],
    *,
    name: str,
    output_queue: queue.Queue[tuple[str, bytes | None]],
) -> None:
    try:
        while True:
            chunk = os.read(stream.fileno(), 4096)
            if not chunk:
                return
            output_queue.put((name, chunk))
    finally:
        stream.close()
        output_queue.put((name, None))


def _drain_process_streams(
    process: subprocess.Popen[bytes],
    readers: list[threading.Thread],
    stream_queue: queue.Queue[tuple[str, bytes | None]],
    *,
    capture_output: bool,
    stdout_callback: Callable[[str], None] | None,
    stderr_callback: Callable[[str], None] | None,
    idle_timeout: float | None,
    isolate_process_group: bool,
    interrupt_cleanup_cmd: list[str] | None,
    cwd: Path | None,
    env: dict[str, str] | None,
) -> tuple[str, str, bool]:
    """Drain stdout/stderr reader threads and return ``(stdout, stderr, timed_out)``.

    When *idle_timeout* fires the process is killed and ``timed_out=True`` is returned.
    Any other ``BaseException`` kills the process, runs the cleanup command, and re-raises.
    """
    stream_data: dict[str, list[str]] = {"stdout": [], "stderr": []}
    callbacks: dict[str, Callable[[str], None] | None] = {
        "stdout": stdout_callback,
        "stderr": stderr_callback,
    }
    timed_out = False

    try:
        decoders = {
            "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
            "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
        }
        active_readers = len(readers)
        last_chunk_time = time.monotonic()
        while active_readers > 0:
            try:
                if idle_timeout is not None:
                    remaining = idle_timeout - (time.monotonic() - last_chunk_time)
                    if remaining <= 0:
                        raise queue.Empty
                    stream_name, chunk = stream_queue.get(timeout=remaining)
                else:
                    stream_name, chunk = stream_queue.get()
            except queue.Empty:
                # Idle timeout: no output received within the deadline.
                if isolate_process_group:
                    _kill_process_group(process)
                else:
                    _terminate_process(process)
                _run_cleanup_command(interrupt_cleanup_cmd, cwd=cwd, env=env)
                timed_out = True
                break
            if chunk is None:
                active_readers -= 1
                continue

            last_chunk_time = time.monotonic()
            text = decoders[stream_name].decode(chunk)
            if not text:
                continue
            if capture_output:
                stream_data[stream_name].append(text)
            callback = callbacks[stream_name]
            if callback is not None:
                callback(text)

        process.wait()

        for stream_name, decoder in decoders.items():
            text = decoder.decode(b"", final=True)
            if not text:
                continue
            if capture_output:
                stream_data[stream_name].append(text)
            callback = callbacks[stream_name]
            if callback is not None:
                callback(text)

        stdout = "".join(stream_data["stdout"])
        stderr = "".join(stream_data["stderr"])
    except BaseException:
        if isolate_process_group:
            _kill_process_group(process)
        else:
            _terminate_process(process)
        _run_cleanup_command(interrupt_cleanup_cmd, cwd=cwd, env=env)
        raise
    finally:
        for reader in readers:
            reader.join()

    return stdout, stderr, timed_out


def _start_process_with_readers(
    cmd: list[str],
    *,
    cwd: Path | None,
    env: dict[str, str] | None,
    capture_output: bool,
    stdout_callback: Callable[[str], None] | None,
    stderr_callback: Callable[[str], None] | None,
    isolate_process_group: bool,
    stdin_text: str | None,
) -> tuple[
    subprocess.Popen[bytes],
    list[threading.Thread],
    queue.Queue[tuple[str, bytes | None]],
    threading.Thread | None,
]:
    """Spawn the process and start pipe-reader threads.

    Return ``(process, readers, queue, stdin_writer)`` where ``readers`` contains
    only the stdout/stderr pipe-reader threads (each posts to *queue*).
    ``stdin_writer`` is a separate thread that writes *stdin_text* to the process
    stdin pipe — it does NOT post to *queue* and must be joined separately after
    draining.  It is ``None`` when *stdin_text* is ``None``."""
    need_stdout_pipe = capture_output or stdout_callback is not None
    need_stderr_pipe = capture_output or stderr_callback is not None

    stdin_pipe = subprocess.PIPE if stdin_text is not None else None

    process: subprocess.Popen[bytes] = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=os.environ if env is None else env,
        stdout=subprocess.PIPE if need_stdout_pipe else None,
        stderr=subprocess.PIPE if need_stderr_pipe else None,
        stdin=stdin_pipe,
        text=False,
        start_new_session=isolate_process_group,
    )

    stdin_writer: threading.Thread | None = None
    if stdin_text is not None and process.stdin is not None:
        stdin_pipe_ref = process.stdin

        def _write_stdin(data: bytes, pipe: IO[bytes]) -> None:
            try:
                with pipe:
                    pipe.write(data)
            except BrokenPipeError:
                # Child exited before reading all stdin — normal outcome, not an error.
                pass

        stdin_writer = threading.Thread(
            target=_write_stdin,
            args=(stdin_text.encode(), stdin_pipe_ref),
            daemon=True,
        )
        stdin_writer.start()

    readers: list[threading.Thread] = []
    stream_queue: queue.Queue[tuple[str, bytes | None]] = queue.Queue()

    if process.stdout is not None:
        reader = threading.Thread(
            target=partial(
                _read_pipe_chunks,
                process.stdout,
                name="stdout",
                output_queue=stream_queue,
            ),
            daemon=True,
        )
        reader.start()
        readers.append(reader)

    if process.stderr is not None:
        reader = threading.Thread(
            target=partial(
                _read_pipe_chunks,
                process.stderr,
                name="stderr",
                output_queue=stream_queue,
            ),
            daemon=True,
        )
        reader.start()
        readers.append(reader)

    return process, readers, stream_queue, stdin_writer


def run_subprocess(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
    interrupt_cleanup_cmd: list[str] | None = None,
    stdout_callback: Callable[[str], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
    isolate_process_group: bool = False,
    idle_timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command and clean it up on interrupt.

    When *idle_timeout* is set (in seconds), the process is killed via
    ``_kill_process_group`` if no output chunk is received for that
    duration.  Requires *isolate_process_group=True* so the entire
    process tree can be cleaned up.
    """

    process, readers, stream_queue, _stdin_writer = _start_process_with_readers(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=capture_output,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
        isolate_process_group=isolate_process_group,
        stdin_text=None,
    )

    if readers:
        stdout, stderr, timed_out = _drain_process_streams(
            process,
            readers,
            stream_queue,
            capture_output=capture_output,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
            idle_timeout=idle_timeout,
            isolate_process_group=isolate_process_group,
            interrupt_cleanup_cmd=interrupt_cleanup_cmd,
            cwd=cwd,
            env=env,
        )
        if timed_out:
            print(
                f"Idle timeout ({idle_timeout}s) exceeded, "
                "process terminated.",
                file=sys.stderr,
            )
            raise SystemExit(124)
        return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
    else:
        try:
            process.wait()
        except BaseException:
            if isolate_process_group:
                _kill_process_group(process)
            else:
                _terminate_process(process)
            _run_cleanup_command(interrupt_cleanup_cmd, cwd=cwd, env=env)
            raise
        return subprocess.CompletedProcess(cmd, process.returncode, None, None)


def run_foreground(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    interrupt_cleanup_cmd: list[str] | None = None,
    isolate_process_group: bool = False,
    idle_timeout: float | None = None,
) -> int:
    """Run a command inheriting stdio."""

    result = run_subprocess(
        cmd,
        cwd=cwd,
        env=env,
        interrupt_cleanup_cmd=interrupt_cleanup_cmd,
        isolate_process_group=isolate_process_group,
        idle_timeout=idle_timeout,
    )
    return result.returncode


def run_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    interrupt_cleanup_cmd: list[str] | None = None,
    stdout_callback: Callable[[str], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
    isolate_process_group: bool = False,
    idle_timeout: float | None = None,
) -> tuple[int, str, str]:
    """Run a command and capture stdout/stderr.

    This is a compatibility adapter over :func:`_run_capture_result_impl`.  On idle-timeout
    it prints a diagnostic to stderr and raises ``SystemExit(124)`` — matching the
    original behaviour.  Spawn errors re-raise the *original* exception object so callers
    get faithful exception types with all attributes intact:
    - ``OSError`` subclasses (``FileNotFoundError``, ``PermissionError``,
      ``OSError(ENOEXEC)``, etc.) are re-raised with ``errno``/``filename`` intact.
    - ``ValueError`` (e.g. ``'embedded null byte'`` from ``subprocess.Popen``) is
      re-raised as-is, restoring the original exception semantics.
    ``run_capture_result`` continues to discard spawn exceptions and return a
    structured :class:`ProcessCaptureResult` instead.
    """
    result, spawn_exc = _run_capture_result_impl(
        cmd,
        cwd=cwd,
        env=env,
        idle_timeout=idle_timeout,
        isolate_process_group=isolate_process_group,
        interrupt_cleanup_cmd=interrupt_cleanup_cmd,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
    )
    if spawn_exc is not None:
        raise spawn_exc
    if result.timed_out:
        print(
            f"Idle timeout ({idle_timeout}s) exceeded, "
            "process terminated.",
            file=sys.stderr,
        )
        raise SystemExit(124)
    rc = result.returncode if result.returncode is not None else 1
    return rc, result.stdout, result.stderr


@dataclass(frozen=True, slots=True)
class ProcessCaptureResult:
    """Structured result of a captured subprocess run.

    Semantics:
    - ``spawn_error`` is set (non-``None``) if and only if the process could not be
      started (any ``OSError`` at the spawn boundary — ``FileNotFoundError``,
      ``PermissionError``, ``OSError(ENOEXEC)``, etc.); in that case ``returncode``
      is ``None`` and both streams are empty.
    - ``spawn_errno`` mirrors the OS ``errno`` of the spawn exception (e.g. ``ENOENT``,
      ``EACCES``, ``ENOEXEC``) so callers can identify the cause.  It is ``None`` when
      there was no spawn error, or when the error was a ``ValueError`` (no OS errno).
    - ``timed_out`` is ``True`` when the idle timeout fired; ``returncode`` then
      reflects the kill exit code (nonzero).
    - In all normal-completion cases ``spawn_error`` is ``None``, ``timed_out`` is
      ``False``, and ``returncode`` is the process exit code.
    """

    returncode: int | None
    stdout: str
    stderr: str
    elapsed: float
    timed_out: bool
    spawn_error: str | None
    spawn_errno: int | None


def _run_capture_result_impl(
    cmd: list[str],
    *,
    idle_timeout: float | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
    isolate_process_group: bool = False,
    interrupt_cleanup_cmd: list[str] | None = None,
    stdout_callback: Callable[[str], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
) -> tuple[ProcessCaptureResult, OSError | ValueError | None]:
    """Internal implementation of ``run_capture_result``.

    Returns ``(result, original_spawn_exc)`` so that ``run_capture`` can
    re-raise the *original* exception intact:
    - For ``OSError`` (``FileNotFoundError``, ``PermissionError``, ``OSError(ENOEXEC)``
      etc.) the original object is returned so callers get faithful exception types
      with ``errno``/``filename`` intact.
    - For ``ValueError`` (e.g. ``'embedded null byte'`` from ``subprocess.Popen``
      before the child is launched) the original object is also returned so
      ``run_capture`` can re-raise it, restoring the original exception semantics.
    ``run_capture_result`` discards the exception object and returns only the result.
    """
    start = time.monotonic()

    try:
        process, readers, stream_queue, stdin_writer = _start_process_with_readers(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
            isolate_process_group=isolate_process_group,
            stdin_text=stdin_text,
        )
    except OSError as exc:
        # Catches FileNotFoundError (ENOENT), PermissionError (EACCES),
        # and all other OS-level spawn failures including ENOEXEC, ENOTDIR, etc.
        elapsed = time.monotonic() - start
        return (
            ProcessCaptureResult(
                returncode=None,
                stdout="",
                stderr="",
                elapsed=elapsed,
                timed_out=False,
                spawn_error=str(exc),
                spawn_errno=exc.errno,
            ),
            exc,
        )
    except ValueError as exc:
        # ``subprocess.Popen`` raises a plain ``ValueError`` (no ``errno``)
        # before the child is launched for malformed arguments — most notably
        # ``ValueError('embedded null byte')`` when an argv element contains a
        # NUL.  Map it to the same spawn-failure result as the OS-level spawn
        # errors; ``spawn_errno`` is ``None`` since there is no OS error number.
        # The original exception object is returned so ``run_capture`` can
        # re-raise it, preserving original exception semantics for all callers.
        elapsed = time.monotonic() - start
        return (
            ProcessCaptureResult(
                returncode=None,
                stdout="",
                stderr="",
                elapsed=elapsed,
                timed_out=False,
                spawn_error=str(exc),
                spawn_errno=None,
            ),
            exc,
        )

    stdout, stderr, timed_out = _drain_process_streams(
        process,
        readers,
        stream_queue,
        capture_output=True,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
        idle_timeout=idle_timeout,
        isolate_process_group=isolate_process_group,
        interrupt_cleanup_cmd=interrupt_cleanup_cmd,
        cwd=cwd,
        env=env,
    )

    # The stdin writer thread is a daemon, but join it now that the process has
    # exited so it never lingers beyond this call.
    if stdin_writer is not None:
        stdin_writer.join()

    elapsed = time.monotonic() - start
    return (
        ProcessCaptureResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed=elapsed,
            timed_out=timed_out,
            spawn_error=None,
            spawn_errno=None,
        ),
        None,
    )


def run_capture_result(
    cmd: list[str],
    *,
    idle_timeout: float | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
    isolate_process_group: bool = False,
    interrupt_cleanup_cmd: list[str] | None = None,
    stdout_callback: Callable[[str], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
) -> ProcessCaptureResult:
    """Run *cmd* and return a :class:`ProcessCaptureResult`.

    Unlike :func:`run_capture` this function **never** prints to stderr and
    **never** raises :exc:`SystemExit`.  All outcomes — spawn failure,
    nonzero exit, and idle-timeout — are represented in the returned
    :class:`ProcessCaptureResult`.

    Parameters match :func:`run_capture` where applicable:
    *idle_timeout* (seconds), *cwd*, *env*, *stdin_text*, *isolate_process_group*,
    *interrupt_cleanup_cmd*, *stdout_callback*, *stderr_callback*.
    """
    result, _ = _run_capture_result_impl(
        cmd,
        idle_timeout=idle_timeout,
        cwd=cwd,
        env=env,
        stdin_text=stdin_text,
        isolate_process_group=isolate_process_group,
        interrupt_cleanup_cmd=interrupt_cleanup_cmd,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
    )
    return result


def require_success(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command in the foreground and exit if it fails."""

    if dry_run.enabled():
        dry_run.print_command(cmd, cwd=cwd)
        return
    returncode = run_foreground(cmd, cwd=cwd, env=env)
    if returncode != 0:
        raise SystemExit(returncode)


def require_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run a command, return stdout, and exit if it fails."""

    returncode, stdout, stderr = run_capture(cmd, cwd=cwd, env=env)
    if returncode != 0:
        exit_with_output(returncode, stdout, stderr)
    return stdout
