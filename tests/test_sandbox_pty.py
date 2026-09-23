"""Tests for the controlling-terminal relay used by ``agm run --pty``."""

from __future__ import annotations

import errno
import os
import runpy
import signal
import subprocess
import sys

import pytest

from agm.sandbox import pty as sandbox_pty


def test_relay_gives_child_a_controlling_terminal() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agm.sandbox.pty",
            "--",
            sys.executable,
            "-c",
            "import os; fd = os.open('/dev/tty', os.O_RDWR); os.close(fd); print('tty-ok')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "tty-ok" in result.stdout


def test_relay_returns_child_exit_status() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agm.sandbox.pty",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 7


def test_main_requires_and_runs_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(sandbox_pty, "run", lambda command: captured.append(command) or 9)

    with pytest.raises(SystemExit, match="9"):
        sandbox_pty.main(["--", "echo", "hello"])

    assert captured == [["echo", "hello"]]


@pytest.mark.parametrize("argv", [[], ["--"]])
def test_main_rejects_missing_command(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="command is required"):
        sandbox_pty.main(argv)


def test_copy_terminal_size_ignores_non_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_ioctl(_fd: int, _operation: int, _data: bytes | int = 0) -> bytes:
        raise OSError

    monkeypatch.setattr(sandbox_pty.fcntl, "ioctl", fail_ioctl)

    sandbox_pty.copy_terminal_size(0, 1)


def test_copy_terminal_size_copies_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, bytes | int]] = []

    def fake_ioctl(fd: int, operation: int, data: bytes | int = 0) -> bytes:
        calls.append((fd, operation, data))
        return b"dimensions"

    monkeypatch.setattr(sandbox_pty.fcntl, "ioctl", fake_ioctl)

    sandbox_pty.copy_terminal_size(4, 8)

    assert calls == [
        (4, sandbox_pty.termios.TIOCGWINSZ, bytes(8)),
        (8, sandbox_pty.termios.TIOCSWINSZ, b"dimensions"),
    ]


def test_write_all_handles_partial_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[bytes] = []

    def fake_write(_fd: int, data: memoryview) -> int:
        writes.append(bytes(data))
        return min(2, len(data))

    monkeypatch.setattr(os, "write", fake_write)

    sandbox_pty.write_all(1, b"hello")

    assert writes == [b"hello", b"llo", b"o"]


def test_write_all_stops_if_write_makes_no_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "write", lambda _fd, _data: 0)

    sandbox_pty.write_all(1, b"hello")


def test_relay_copies_terminal_input_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    reads = {8: [b"output", b""], 0: [b"input", b""]}
    writes: list[tuple[int, bytes]] = []
    monkeypatch.setattr(os, "isatty", lambda fd: fd == 0)
    monkeypatch.setattr(
        sandbox_pty.select,
        "select",
        lambda sources, _writes, _errors: (list(sources), [], []),
    )
    monkeypatch.setattr(os, "read", lambda fd, _size: reads[fd].pop(0))
    monkeypatch.setattr(
        sandbox_pty,
        "write_all",
        lambda fd, data: writes.append((fd, data)),
    )

    sandbox_pty._relay(8)

    assert writes == [(1, b"output"), (8, b"input")]


def test_relay_stops_when_child_terminal_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    selections = 0
    monkeypatch.setattr(os, "isatty", lambda fd: fd == 0)

    def select_once(
        _sources: list[int], _writes: list[int], _errors: list[int]
    ) -> tuple[list[int], list[int], list[int]]:
        nonlocal selections
        selections += 1
        if selections > 1:
            raise AssertionError("relay waited for input after the child exited")
        return [8], [], []

    monkeypatch.setattr(sandbox_pty.select, "select", select_once)
    monkeypatch.setattr(os, "read", lambda _fd, _size: b"")

    sandbox_pty._relay(8)


def test_relay_treats_master_eio_as_child_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "isatty", lambda _fd: False)
    monkeypatch.setattr(
        sandbox_pty.select,
        "select",
        lambda sources, _writes, _errors: (list(sources), [], []),
    )

    def fail_read(_fd: int, _size: int) -> bytes:
        raise OSError(errno.EIO, "terminal closed")

    monkeypatch.setattr(os, "read", fail_read)

    sandbox_pty._relay(8)


def test_relay_raises_unexpected_read_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "isatty", lambda _fd: False)
    monkeypatch.setattr(
        sandbox_pty.select,
        "select",
        lambda sources, _writes, _errors: (list(sources), [], []),
    )

    def fail_read(_fd: int, _size: int) -> bytes:
        raise OSError(errno.EBADF, "bad descriptor")

    monkeypatch.setattr(os, "read", fail_read)

    with pytest.raises(OSError, match="bad descriptor"):
        sandbox_pty._relay(8)


@pytest.mark.parametrize(("status", "expected"), [(7 << 8, 7), (signal.SIGTERM, 143)])
def test_wait_exit_code_maps_process_status(
    monkeypatch: pytest.MonkeyPatch, status: int, expected: int
) -> None:
    monkeypatch.setattr(os, "waitpid", lambda _pid, _flags: (4, status))

    assert sandbox_pty._wait_exit_code(4) == expected


def test_run_manages_interactive_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    installed_handlers: dict[int, list[object]] = {}
    copied_sizes: list[tuple[int, int]] = []
    forwarded: list[tuple[int, int]] = []
    restored: list[tuple[int, int, object]] = []
    attributes: list[int | list[bytes | int]] = [0, 0, 0, 0, 0, 0, []]

    monkeypatch.setattr(os, "forkpty", lambda: (12, 8))
    monkeypatch.setattr(os, "isatty", lambda fd: fd == 0)
    monkeypatch.setattr(os, "killpg", lambda pid, number: forwarded.append((pid, number)))
    monkeypatch.setattr(os, "close", lambda _fd: None)
    monkeypatch.setattr(sandbox_pty, "copy_terminal_size", lambda a, b: copied_sizes.append((a, b)))
    monkeypatch.setattr(sandbox_pty, "_relay", lambda _fd: None)
    monkeypatch.setattr(sandbox_pty, "_wait_exit_code", lambda _pid: 5)
    monkeypatch.setattr(signal, "getsignal", lambda number: f"old-{number}")
    monkeypatch.setattr(
        signal,
        "signal",
        lambda number, handler: installed_handlers.setdefault(number, []).append(handler),
    )
    monkeypatch.setattr(sandbox_pty.termios, "tcgetattr", lambda _fd: attributes)
    monkeypatch.setattr(
        sandbox_pty.termios,
        "tcsetattr",
        lambda fd, when, value: restored.append((fd, when, value)),
    )
    monkeypatch.setattr(sandbox_pty.tty, "setraw", lambda _fd: None)

    assert sandbox_pty.run(["echo"]) == 5

    winch_handler = installed_handlers[signal.SIGWINCH][0]
    term_handler = installed_handlers[signal.SIGTERM][0]
    assert callable(winch_handler)
    assert callable(term_handler)
    winch_handler(signal.SIGWINCH, None)
    term_handler(signal.SIGTERM, None)
    assert copied_sizes == [(0, 8), (0, 8)]
    assert forwarded == [(12, signal.SIGTERM)]
    assert restored == [(0, sandbox_pty.termios.TCSAFLUSH, attributes)]


def test_run_skips_terminal_mode_without_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "forkpty", lambda: (12, 8))
    monkeypatch.setattr(os, "isatty", lambda _fd: False)
    monkeypatch.setattr(os, "close", lambda _fd: None)
    monkeypatch.setattr(sandbox_pty, "copy_terminal_size", lambda _a, _b: None)
    monkeypatch.setattr(sandbox_pty, "_relay", lambda _fd: None)
    monkeypatch.setattr(sandbox_pty, "_wait_exit_code", lambda _pid: 0)

    assert sandbox_pty.run(["echo"]) == 0


def test_run_executes_child_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    class Executed(Exception):
        pass

    monkeypatch.setattr(os, "forkpty", lambda: (0, 8))

    def fake_execvp(_executable: str, _arguments: list[str]) -> None:
        raise Executed

    monkeypatch.setattr(os, "execvp", fake_execvp)

    with pytest.raises(Executed):
        sandbox_pty.run(["echo"])


def test_module_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pty"])

    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit, match="command is required"):
            runpy.run_module("agm.sandbox.pty", run_name="__main__")
