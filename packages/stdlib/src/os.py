"""Host process, filesystem-adjacent, and environment lookups for ``std/os``."""

from __future__ import annotations

import os
import pwd
import shutil
import socket
import sys
import tempfile
from collections.abc import Callable, MutableMapping
from typing import NoReturn, Protocol

from agl import AglException, nominals, option_none, option_some, runtime

from agm.agl.runtime.host_fs import run_fs_action
from agm.agl.runtime.host_text import scalar_host_text
from agm.core.process import run_foreground, run_foreground_ignoring_signals

EncodingError = nominals.std.errors.EncodingError
FsError = nominals.std.fs.FsError
LaunchError = nominals.std.os.LaunchError

# Editors tried, in order, when neither `VISUAL` nor `EDITOR` is set.
_EDITOR_FALLBACKS = ("micro", "vi")

_MIN_EXIT_CODE = 0
_MAX_EXIT_CODE = 255
# `sh`'s exit status for "command not found" (POSIX): the launch target itself
# was never found, not merely a command it ran that failed.
_SHELL_COMMAND_NOT_FOUND = 127
# The ``runtime.state`` key of the directory before the first ``chdir``,
# restored when the host session ends.
_START_DIR_KEY = "std/os/start-dir"


class _EnvironLike(Protocol):
    vars: MutableMapping[str, str]


def exit(code: int = 0) -> None:
    """End the AgL host process with portable status *code*."""
    if not _MIN_EXIT_CODE <= code <= _MAX_EXIT_CODE:
        raise ValueError(f"exit code must be in {_MIN_EXIT_CODE}..{_MAX_EXIT_CODE}")
    raise SystemExit(code)


def cwd() -> str:
    """Return the current working directory; raise `FsError` if it no longer exists."""
    return scalar_host_text(run_fs_action(FsError, ".", "cwd", os.getcwd), EncodingError)


def pid() -> int:
    """Return this process identifier."""
    return os.getpid()


def hostname() -> str:
    """Return this host's name."""
    return scalar_host_text(socket.gethostname(), EncodingError)


def temp_dir() -> str:
    """Return the host's temporary-file directory."""
    return scalar_host_text(tempfile.gettempdir(), EncodingError)


def _restore_start_directory(directory: str | None) -> None:
    """Change back to *directory*; skip it if unknown or no longer enterable."""
    if directory is None:
        return
    try:
        os.chdir(directory)
    except OSError:
        pass


def _cwd_or_none() -> str | None:
    """Return the working directory, or `None` if it no longer exists."""
    try:
        return scalar_host_text(os.getcwd(), EncodingError)
    except FileNotFoundError:
        return None


def chdir(directory: str, vars_: MutableMapping[str, str]) -> None:
    """Change the working directory, recording ``OLDPWD``/``PWD`` in *vars_*.

    A failed `os.chdir` raises `FsError` and leaves the directory unchanged.
    A before or after directory that is not valid Unicode raises
    `EncodingError`; the latter case restores the prior directory first, so
    the directory is unchanged either way. The starting directory is
    registered once, on a session's first `chdir`, to be restored when the
    host session ends. A working directory that no longer exists can be left:
    `OLDPWD` then takes the previous `PWD`, and nothing is restored.
    """
    before = _cwd_or_none()
    run_fs_action(FsError, directory, "chdir", lambda: os.chdir(directory))
    try:
        after = scalar_host_text(os.getcwd(), EncodingError)
    except AglException:
        _restore_start_directory(before)
        raise
    runtime.state(_START_DIR_KEY, lambda: before, close=_restore_start_directory)
    vars_["OLDPWD"] = before if before is not None else vars_.get("PWD", "")
    vars_["PWD"] = after


def is_interactive() -> bool:
    """Whether standard input and output are both connected to a terminal."""
    return os.isatty(0) and os.isatty(1)


def _resolve_executable(name: str, env: _EnvironLike) -> str | None:
    """Look up *name* on *env*'s ``PATH``, the host default when absent."""
    return shutil.which(name, path=env.vars.get("PATH", os.defpath))


def which(name: str, env: _EnvironLike) -> object:
    """Look up *name* on *env*'s ``PATH``, the host default when absent."""
    found = _resolve_executable(name, env)
    return option_some(found) if found is not None else option_none()


def platform() -> str:
    """Return the host platform identifier."""
    return sys.platform


def cpu_count() -> int:
    """Return the usable CPU count, at least 1."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def _raise_launch_error(command: str, exit_code: int | None) -> NoReturn:
    """Raise `LaunchError` for *command*: not found (`exit_code=None`) or a bad exit."""
    if exit_code is None:
        message = f"could not find {command!r} to launch"
        code = option_none()
    else:
        message = f"{command!r} exited with status {exit_code}"
        code = option_some(exit_code)
    raise AglException(LaunchError(message=message, command=command, **{"exit-code": code}))


def _resolve_editor(env: _EnvironLike) -> str:
    """Return the editor command to run, raising `LaunchError` if none is found."""
    for variable in ("VISUAL", "EDITOR"):
        value = env.vars.get(variable, "")
        if value:
            return value
    for name in _EDITOR_FALLBACKS:
        if _resolve_executable(name, env) is not None:
            return name
    _raise_launch_error(_EDITOR_FALLBACKS[-1], None)


def _spawn_or_launch_error(command: str, run: Callable[[], int]) -> int:
    """Run *run*, mapping a `FileNotFoundError` spawning *command* to `LaunchError`.

    Covers both a `PATH` that never had *command* on it and the race between
    an earlier lookup finding it and the spawn that follows.
    """
    try:
        return run()
    except FileNotFoundError:
        _raise_launch_error(command, None)


def edit(file: str, env: _EnvironLike) -> None:
    """Edit *file*, blocking until the editor exits; see `os::edit`."""
    editor = _resolve_editor(env)
    cmd = ["sh", "-c", f'{editor} "$@"', editor, os.path.abspath(file)]
    returncode = _spawn_or_launch_error(
        "sh", lambda: run_foreground_ignoring_signals(cmd, env=dict(env.vars))
    )
    if returncode == _SHELL_COMMAND_NOT_FOUND:
        _raise_launch_error(editor, None)
    if returncode != 0:
        _raise_launch_error(editor, returncode)


def open(target: str, env: _EnvironLike) -> None:
    """Open *target* with the desktop default application; see `os::open`."""
    opener_name = "open" if sys.platform == "darwin" else "xdg-open"
    opener = _resolve_executable(opener_name, env)
    if opener is None:
        _raise_launch_error(opener_name, None)
    returncode = _spawn_or_launch_error(
        opener_name, lambda: run_foreground([opener, target], env=dict(env.vars))
    )
    if returncode != 0:
        _raise_launch_error(opener_name, returncode)


def user() -> object:
    """Return the real user's passwd name, or `None` when it has none."""
    try:
        name = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return option_none()
    return option_some(scalar_host_text(name, EncodingError))


__all__ = [
    "chdir",
    "cpu_count",
    "cwd",
    "edit",
    "exit",
    "hostname",
    "is_interactive",
    "open",
    "pid",
    "platform",
    "temp_dir",
    "user",
    "which",
]
