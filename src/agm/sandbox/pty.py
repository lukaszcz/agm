"""Run a command with a controlling pseudo-terminal."""

from __future__ import annotations

import errno
import fcntl
import os
import select
import signal
import sys
import termios
import tty
from collections.abc import Sequence
from types import FrameType


def copy_terminal_size(source_fd: int, target_fd: int) -> None:
    """Copy terminal dimensions when *source_fd* provides them."""

    try:
        dimensions = fcntl.ioctl(source_fd, termios.TIOCGWINSZ, bytes(8))
        fcntl.ioctl(target_fd, termios.TIOCSWINSZ, dimensions)
    except OSError:
        pass


def write_all(fd: int, data: bytes) -> None:
    """Write all *data* to *fd*."""

    remaining = data
    while remaining:
        written = os.write(fd, remaining)
        if written == 0:
            return
        remaining = remaining[written:]


def _relay(master_fd: int) -> None:
    sources = [master_fd]
    if os.isatty(0):
        sources.append(0)
    while master_fd in sources:
        readable, _writable, _exceptional = select.select(sources, [], [])
        for source_fd in readable:
            try:
                data = os.read(source_fd, 65_536)
            except OSError as error:
                if source_fd == master_fd and error.errno == errno.EIO:
                    sources.remove(master_fd)
                    continue
                raise
            if not data:
                sources.remove(source_fd)
                continue
            write_all(1 if source_fd == master_fd else master_fd, data)


def _wait_exit_code(pid: int) -> int:
    _waited_pid, status = os.waitpid(pid, 0)
    exit_code = os.waitstatus_to_exitcode(status)
    return exit_code if exit_code >= 0 else 128 - exit_code


def run(command: Sequence[str]) -> int:
    """Run *command* on a new controlling pseudo-terminal."""

    pid, master_fd = os.forkpty()
    if pid == 0:
        os.execvp(command[0], list(command))

    copy_terminal_size(0, master_fd)
    previous_winch = signal.getsignal(signal.SIGWINCH)
    previous_termination_handlers = {
        number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)
    }
    terminal_attributes: list[int | list[bytes | int]] | None = None

    def resize(_number: int, _frame: FrameType | None) -> None:
        copy_terminal_size(0, master_fd)

    def forward(number: int, _frame: FrameType | None) -> None:
        os.killpg(pid, number)

    signal.signal(signal.SIGWINCH, resize)
    for number in previous_termination_handlers:
        signal.signal(number, forward)
    if os.isatty(0):
        terminal_attributes = termios.tcgetattr(0)
        tty.setraw(0)
    try:
        _relay(master_fd)
    finally:
        if terminal_attributes is not None:
            termios.tcsetattr(0, termios.TCSAFLUSH, terminal_attributes)
        signal.signal(signal.SIGWINCH, previous_winch)
        for number, handler in previous_termination_handlers.items():
            signal.signal(number, handler)
        os.close(master_fd)
    return _wait_exit_code(pid)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the relay CLI."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        raise SystemExit("command is required")
    raise SystemExit(run(arguments))


if __name__ == "__main__":
    main()
