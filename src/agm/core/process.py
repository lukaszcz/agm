"""Process execution helpers."""

from __future__ import annotations

import codecs
import os
import queue
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
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


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()


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


def run_subprocess(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
    interrupt_cleanup_cmd: list[str] | None = None,
    stdout_callback: Callable[[str], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command in its own process group and clean it up on interrupt."""

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=os.environ if env is None else env,
        stdout=subprocess.PIPE if capture_output or stdout_callback is not None else None,
        stderr=subprocess.PIPE if capture_output or stderr_callback is not None else None,
        text=False,
        start_new_session=True,
    )

    readers: list[threading.Thread] = []
    stream_queue: queue.Queue[tuple[str, bytes | None]] = queue.Queue()
    stream_data: dict[str, list[str]] = {"stdout": [], "stderr": []}
    callbacks = {"stdout": stdout_callback, "stderr": stderr_callback}

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

    try:
        if readers:
            decoders = {
                "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
                "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
            }
            active_readers = len(readers)
            while active_readers > 0:
                stream_name, chunk = stream_queue.get()
                if chunk is None:
                    active_readers -= 1
                    continue

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
        else:
            process.wait()
            stdout = None
            stderr = None
    except BaseException:
        _kill_process_group(process)
        _run_cleanup_command(interrupt_cleanup_cmd, cwd=cwd, env=env)
        raise
    finally:
        for reader in readers:
            reader.join()

    return subprocess.CompletedProcess(
        cmd,
        process.returncode,
        stdout,
        stderr,
    )


def run_foreground(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    interrupt_cleanup_cmd: list[str] | None = None,
) -> int:
    """Run a command inheriting stdio."""

    result = run_subprocess(
        cmd,
        cwd=cwd,
        env=env,
        interrupt_cleanup_cmd=interrupt_cleanup_cmd,
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
) -> tuple[int, str, str]:
    """Run a command and capture stdout/stderr."""

    result = run_subprocess(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        interrupt_cleanup_cmd=interrupt_cleanup_cmd,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
    )
    return result.returncode, result.stdout or "", result.stderr or ""


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
