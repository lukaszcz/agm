"""A preforking launcher for the staged ``agm`` CLI used by the e2e tests.

Every e2e invocation of the checkout CLI pays a fresh interpreter start and
the import of several hundred modules before the command under test runs any
AGM code at all — around 70 ms for a command that never touches AgL and around
340 ms for one that does, against roughly 620 invocations per suite run.

This module amortizes that. A *zygote* process per xdist worker imports the CLI
and the AgL stack once, then forks a child for each invocation. The child is an
ordinary OS process: it gets the invocation's own argv, environment, working
directory and standard file descriptors, and its exit status is the real one.
Two things it cannot re-prove: that a *cold* interpreter can import AGM in the
order a given command needs, and that a command survives a hash seed of its
own, since every child inherits the zygote's. :envvar:`AGM_TEST_NO_ZYGOTE`
turns the whole mechanism off and returns every invocation to a plain
``subprocess.run``, which is how the suite is run to settle either question.

The zygote never runs a command itself, so a child inherits only import-time
state — the same state a freshly started process would build for itself.
"""

from __future__ import annotations

import array
import importlib
import locale
import os
import pickle
import socket
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# An invocation as it crosses the socket: argv, environment, working directory.
type Invocation = tuple[list[str], dict[str, str], str]

# Set to "1" to run every e2e invocation as its own cold process.
DISABLE_ENV = "AGM_TEST_NO_ZYGOTE"

# Standard descriptors, in the order the client hands them over.
_STREAM_COUNT = 3

_LENGTH = struct.Struct("!I")

# How long a client waits for a newly spawned zygote to start listening.
_STARTUP_TIMEOUT = 60.0

# How long the zygote waits for work before checking that it still has a worker
# to serve. A killed worker leaves no one to shut it down, so it notices itself.
_IDLE_CHECK = 5.0


def _reaped_cpu_seconds() -> float:
    """Return the CPU this process has been credited with for children it reaped."""
    times = os.times()
    return times.children_user + times.children_system


def _charge(seconds: float) -> None:
    """Report CPU the zygote burned on a test's behalf, or take it back when negative.

    Imported on use rather than at module scope: the server half of this module
    runs as ``__main__`` with only AGM's source on its path, and every module it
    holds is inherited by every child it forks.
    """
    from tests._durations import charge_external_cpu_seconds

    charge_external_cpu_seconds(seconds)


def _encoding() -> str:
    """Return the encoding ``subprocess.run(text=True)`` would use."""
    return locale.getencoding()


def _decode(data: bytes) -> str:
    """Decode captured output exactly as ``text=True`` does, newlines included."""
    return data.decode(_encoding()).replace("\r\n", "\n").replace("\r", "\n")


def _send(conn: socket.socket, payload: bytes, fds: Sequence[int] = ()) -> None:
    """Send one length-prefixed frame, optionally passing *fds* alongside it."""
    frame = _LENGTH.pack(len(payload)) + payload
    if not fds:
        conn.sendall(frame)
        return
    rights = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
    sent = conn.sendmsg([frame], rights)
    if sent < len(frame):
        conn.sendall(frame[sent:])


def _recv_exactly(conn: socket.socket, size: int) -> bytes:
    """Read exactly *size* bytes, or return short on a closed connection."""
    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv(conn: socket.socket) -> bytes:
    """Read one length-prefixed frame, or ``b""`` when the peer hung up."""
    header = _recv_exactly(conn, _LENGTH.size)
    if len(header) < _LENGTH.size:
        return b""
    return _recv_exactly(conn, _LENGTH.unpack(header)[0])


def _recv_with_fds(conn: socket.socket) -> tuple[bytes, list[int]]:
    """Read one frame together with the descriptors passed alongside it."""
    space = socket.CMSG_SPACE(_STREAM_COUNT * array.array("i").itemsize)
    header, ancillary, _flags, _addr = conn.recvmsg(_LENGTH.size, space)
    fds: list[int] = []
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            received = array.array("i")
            received.frombytes(data[: len(data) - len(data) % received.itemsize])
            fds.extend(received)
    if len(header) < _LENGTH.size:
        for fd in fds:
            os.close(fd)
        return b"", []
    return _recv_exactly(conn, _LENGTH.unpack(header)[0]), fds


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


# What a cold ``agm`` run imports on its way to a command. Pre-importing the
# AgL stack is the point of the exercise; the CLI modules come along because
# they are what reaches it.
_PRELOADED = (
    "agm.cli",
    "agm.cli_dispatch",
    "agm.cli_support.program_options",
    "agm.commands.exec_program",
    "agm.agl.pipeline",
    "agm.agl.runtime",
)


def _preload() -> None:
    """Import everything a command could need, so a forked child imports none of it."""
    for name in _PRELOADED:
        importlib.import_module(name)


def _run_command(request: Invocation) -> object:
    """Become the invocation described by *request* and run the CLI.

    Returns what the staged entry point would hand to ``sys.exit``.
    """
    argv, env, cwd = request
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(env)
    sys.argv = list(argv)
    from agm.cli import main

    return main()


def _exit_status(code: object) -> int:
    """Return what the interpreter would exit with for a ``SystemExit`` carrying *code*.

    A missing code is success and an integer is itself; anything else is a
    message the interpreter prints before failing.
    """
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def _child(request: Invocation, fds: Sequence[int], holding: Sequence[int]) -> None:
    """Take over the standard descriptors, run the command, and exit. Never returns."""
    code = 1
    try:
        for target, fd in enumerate(fds):
            os.dup2(fd, target)
        for fd in (*fds, *holding):
            if fd >= _STREAM_COUNT:
                os.close(fd)
        code = _exit_status(_run_command(request))
    except SystemExit as exit_request:
        code = _exit_status(exit_request.code)
    except BaseException:
        # A cold run reports an escaping exception on stderr and fails; so does this.
        traceback.print_exc()
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except ValueError:
                pass
        os._exit(code)


def _serve(socket_path: str) -> None:
    """Serve invocations until the worker that started this zygote is gone."""
    _preload()
    worker = os.getppid()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(socket_path)
        server.listen(64)
        server.settimeout(_IDLE_CHECK)
        while True:
            try:
                conn, _peer = server.accept()
            except TimeoutError:
                if os.getppid() != worker:
                    return
                continue
            with conn:
                request_bytes, fds = _recv_with_fds(conn)
                if not request_bytes:
                    for fd in fds:
                        os.close(fd)
                    continue
                request: Invocation = pickle.loads(request_bytes)
                # Nothing of this process may reach the child half-written.
                sys.stdout.flush()
                sys.stderr.flush()
                pid = os.fork()
                if pid == 0:
                    _child(request, fds, (conn.fileno(), server.fileno()))
                for fd in fds:
                    os.close(fd)
                # The worker cannot see this child in its own ``os.times``,
                # so its cost travels back with its status.
                _waited, status, usage = os.wait4(pid, 0)
                cost = usage.ru_utime + usage.ru_stime
                _send(conn, pickle.dumps((status, cost)))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Zygote:
    """A running zygote process and the socket its clients invoke it through."""

    def __init__(self, socket_path: Path, process: subprocess.Popen[bytes]) -> None:
        self._socket_path = socket_path
        self._process = process

    def run(
        self,
        argv: Sequence[str],
        env: dict[str, str],
        cwd: str,
        stdin_text: str | None,
    ) -> subprocess.CompletedProcess[str]:
        """Run *argv* in a forked child and collect it like ``subprocess.run`` would.

        Output goes to temporary files rather than pipes, so neither side can
        block on a full pipe buffer and no reader thread is needed.
        """
        with (
            tempfile.TemporaryFile() as stdin_file,
            tempfile.TemporaryFile() as stdout_file,
            tempfile.TemporaryFile() as stderr_file,
        ):
            if stdin_text is not None:
                stdin_file.write(stdin_text.encode(_encoding()))
                stdin_file.seek(0)
            request: Invocation = (list(argv), env, cwd)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.connect(str(self._socket_path))
                _send(
                    conn,
                    pickle.dumps(request),
                    (stdin_file.fileno(), stdout_file.fileno(), stderr_file.fileno()),
                )
                status_bytes = _recv(conn)
            if not status_bytes:
                raise RuntimeError("the agm zygote closed the connection before reporting")
            status, cost = pickle.loads(status_bytes)
            _charge(cost)
            stdout_file.seek(0)
            stderr_file.seek(0)
            return subprocess.CompletedProcess(
                list(argv),
                os.waitstatus_to_exitcode(status),
                _decode(stdout_file.read()),
                _decode(stderr_file.read()),
            )

    def close(self) -> None:
        """Stop the zygote, un-charge its subtree, and remove its socket.

        Reaping the zygote credits this process with the CPU of every child the
        zygote reaped -- which each invocation was already charged for as it
        finished. Left alone, that arrives as one enormous extra cost against
        whichever test happens to be the last, so it is taken back here.
        """
        before = _reaped_cpu_seconds()
        self._process.terminate()
        self._process.wait()
        _charge(before - _reaped_cpu_seconds())
        self._socket_path.unlink(missing_ok=True)


def start(socket_path: Path, source_root: Path) -> Zygote:
    """Spawn a zygote serving *socket_path* and wait until it accepts connections.

    *source_root* is prepended to the child's import path, which is what makes
    the zygote serve this checkout — the same selection the staged ``agm``
    entry point makes for itself.
    """
    process = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).resolve()), str(socket_path)],
        env={**os.environ, "PYTHONPATH": str(source_root)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    zygote = Zygote(socket_path, process)
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while True:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.connect(str(socket_path))
            return zygote
        except (FileNotFoundError, ConnectionRefusedError):
            if process.poll() is not None or time.monotonic() > deadline:
                zygote.close()
                raise RuntimeError("the agm zygote did not start") from None
            time.sleep(0.01)


if __name__ == "__main__":
    # -I ignores PYTHONPATH, so the checkout's source root is placed by hand.
    sys.path.insert(0, os.environ["PYTHONPATH"])
    _serve(sys.argv[1])
