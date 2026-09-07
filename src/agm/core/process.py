"""Process execution helpers."""

from __future__ import annotations

import codecs
import contextlib
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import FrameType
from typing import IO, NoReturn, TextIO

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


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate *process*, escalating to SIGKILL if it does not exit promptly."""
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


def kill_process_group(process: subprocess.Popen[bytes], *, pgid: int | None = None) -> None:
    """Tear down *process* and every other member of its process group.

    *pgid* names the group when the caller isolated the child under an id
    other than its pid; it defaults to ``process.pid``.
    """
    group = process.pid if pgid is None else pgid
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        return

    if process.poll() is not None:
        # The main process already exited, but other members of its process
        # group may still be alive.  Give them a brief moment to tear down
        # after the SIGTERM we just sent, then SIGKILL any stragglers.
        _wait_for_process_group_exit(group, grace=0.2)
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return

    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return

    # The main process exited promptly.  Give remaining group members a
    # brief moment to exit as well, then SIGKILL any stragglers.
    _wait_for_process_group_exit(group, grace=0.2)
    try:
        os.killpg(group, signal.SIGKILL)
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


@contextlib.contextmanager
def terminating_signals_raise_interrupt() -> Iterator[None]:
    """Deliver SIGTERM and SIGHUP as ``KeyboardInterrupt`` inside the block.

    Under the default disposition a termination signal tears the interpreter
    down without unwinding, so cleanup that lives in ``except`` handlers never
    runs.  That is exactly how a nested AGM dies: the outer process cleans up
    by signalling the process group of the runner it spawned, and the inner
    ``agm run`` in that group owns a transient systemd scope that only its
    cleanup command removes.  Raising instead routes the signal through the
    same teardown as Ctrl-C, so the scope goes away with its children rather
    than outliving them both.
    """

    def raise_interrupt(_signum: int, _frame: FrameType | None) -> NoReturn:
        raise KeyboardInterrupt

    previous = {
        number: signal.signal(number, raise_interrupt) for number in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _run_cleanup_command(
    cmd: list[str] | None,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    if cmd is None:
        return

    # core.env imports this module, so it cannot import resolve_env here without a cycle
    subprocess.run(
        cmd,
        cwd=cwd,
        env=os.environ if env is None else env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _stop_process(
    process: subprocess.Popen[bytes],
    *,
    isolate_process_group: bool,
    interrupt_cleanup_cmd: list[str] | None,
    cwd: Path | None,
    env: dict[str, str] | None,
) -> None:
    try:
        _run_cleanup_command(interrupt_cleanup_cmd, cwd=cwd, env=env)
    finally:
        if isolate_process_group:
            kill_process_group(process)
        else:
            terminate_process(process)


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
    The enclosing process lifetime cleans up exceptions and joins the readers.
    """
    stream_data: dict[str, list[str]] = {"stdout": [], "stderr": []}
    callbacks: dict[str, Callable[[str], None] | None] = {
        "stdout": stdout_callback,
        "stderr": stderr_callback,
    }
    timed_out = False

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
            _stop_process(
                process,
                isolate_process_group=isolate_process_group,
                interrupt_cleanup_cmd=interrupt_cleanup_cmd,
                cwd=cwd,
                env=env,
            )
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

    return stdout, stderr, timed_out


@contextlib.contextmanager
def _defer_interrupts() -> Iterator[None]:
    """Deliver Python termination handlers after child ownership is established.

    Caught handlers reset on exec, so children retain their ordinary signal
    behavior. Blocking signals across Popen would also block them in the child.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    pending: list[tuple[int, FrameType | None]] = []
    handlers: dict[int, Callable[[int, FrameType | None], object]] = {
        number: handler
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        if callable(handler := signal.getsignal(number))
    }

    def defer(number: int, frame: FrameType | None) -> None:
        pending.append((number, frame))

    for number in handlers:
        signal.signal(number, defer)
    try:
        yield
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
        for pending_number, frame in pending:
            handlers[pending_number](pending_number, frame)


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
    interrupt_cleanup_cmd: list[str] | None = None,
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

    process: subprocess.Popen[bytes] | None = None
    stdin_writer: threading.Thread | None = None
    readers: list[threading.Thread] = []
    try:
        # core.env imports this module, so it cannot import resolve_env here without a cycle
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=os.environ if env is None else env,
            stdout=subprocess.PIPE if need_stdout_pipe else None,
            stderr=subprocess.PIPE if need_stderr_pipe else None,
            stdin=stdin_pipe,
            text=False,
            start_new_session=isolate_process_group,
        )

        stream_queue: queue.Queue[tuple[str, bytes | None]] = queue.Queue()

        # Block SIGINT while the worker threads are created so they inherit a
        # blocked signal mask.  A process-directed SIGINT (e.g. Ctrl-C) may be
        # delivered to any thread that has it unblocked; if it lands on a reader
        # thread, the main thread stays blocked in wait()/get() and the interrupt
        # is effectively lost.  By blocking it in the workers, the kernel must
        # deliver it to the main thread, which restores its own mask below.
        previous_sigmask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
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
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_sigmask)

        return process, readers, stream_queue, stdin_writer
    except BaseException:
        if process is not None:
            _stop_process(
                process,
                isolate_process_group=isolate_process_group,
                interrupt_cleanup_cmd=interrupt_cleanup_cmd,
                cwd=cwd,
                env=env,
            )
            for reader in readers:
                reader.join()
            if stdin_writer is not None and stdin_writer.ident is not None:
                stdin_writer.join()
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        raise


@contextlib.contextmanager
def _running_process(
    cmd: list[str],
    *,
    cwd: Path | None,
    env: dict[str, str] | None,
    capture_output: bool,
    stdout_callback: Callable[[str], None] | None,
    stderr_callback: Callable[[str], None] | None,
    isolate_process_group: bool,
    stdin_text: str | None,
    interrupt_cleanup_cmd: list[str] | None,
) -> Iterator[
    tuple[subprocess.Popen[bytes], list[threading.Thread], queue.Queue[tuple[str, bytes | None]]]
]:
    process: subprocess.Popen[bytes] | None = None
    readers: list[threading.Thread] = []
    stdin_writer: threading.Thread | None = None
    try:
        with _defer_interrupts():
            process, readers, stream_queue, stdin_writer = _start_process_with_readers(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=capture_output,
                stdout_callback=stdout_callback,
                stderr_callback=stderr_callback,
                isolate_process_group=isolate_process_group,
                stdin_text=stdin_text,
                interrupt_cleanup_cmd=interrupt_cleanup_cmd,
            )
        yield process, readers, stream_queue
    except BaseException:
        if process is not None:
            _stop_process(
                process,
                isolate_process_group=isolate_process_group,
                interrupt_cleanup_cmd=interrupt_cleanup_cmd,
                cwd=cwd,
                env=env,
            )
        raise
    finally:
        for reader in readers:
            reader.join()
        if stdin_writer is not None:
            stdin_writer.join()


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
    ``kill_process_group`` if no output chunk is received for that
    duration.  Requires *isolate_process_group=True* so the entire
    process tree can be cleaned up.
    """

    # A registered cleanup command releases something this process owns but
    # does not contain, so a termination signal must not be allowed to skip it.
    guard: AbstractContextManager[None] = (
        contextlib.nullcontext()
        if interrupt_cleanup_cmd is None
        else terminating_signals_raise_interrupt()
    )
    with (
        guard,
        _running_process(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
            isolate_process_group=isolate_process_group,
            stdin_text=None,
            interrupt_cleanup_cmd=interrupt_cleanup_cmd,
        ) as (process, readers, stream_queue),
    ):
        timed_out = False
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
            completed = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
        else:
            process.wait()
            completed = subprocess.CompletedProcess(cmd, process.returncode, None, None)
    if timed_out:
        print(f"Idle timeout ({idle_timeout}s) exceeded, process terminated.", file=sys.stderr)
        raise SystemExit(124)
    return completed


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
    timeout_callback: Callable[[str], None] | None = None,
    isolate_process_group: bool = False,
    idle_timeout: float | None = None,
    stdin_text: str | None = None,
) -> tuple[int, str, str]:
    """Run a command, optionally supplying *stdin_text*, and capture stdout/stderr.

    This is a compatibility adapter over :func:`_run_capture_result_impl`. On idle-timeout
    it reports a diagnostic through *timeout_callback* (or stderr when omitted) and raises
    ``SystemExit(124)`` — matching the original behaviour. Spawn errors re-raise the
    *original* exception object so callers
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
        stdin_text=stdin_text,
    )
    if spawn_exc is not None:
        raise spawn_exc
    if result.timed_out:
        message = f"Idle timeout ({idle_timeout}s) exceeded, process terminated.\n"
        if timeout_callback is None:
            print(message, end="", file=sys.stderr)
        else:
            timeout_callback(message)
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

    with contextlib.ExitStack() as stack:
        try:
            process, readers, stream_queue = stack.enter_context(
                _running_process(
                    cmd,
                    cwd=cwd,
                    env=env,
                    capture_output=True,
                    stdout_callback=stdout_callback,
                    stderr_callback=stderr_callback,
                    isolate_process_group=isolate_process_group,
                    stdin_text=stdin_text,
                    interrupt_cleanup_cmd=interrupt_cleanup_cmd,
                )
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
