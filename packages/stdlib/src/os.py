"""Host process, filesystem-adjacent, and environment lookups for ``std/os``."""

from __future__ import annotations

import os
import pwd
import shutil
import socket
import sys
import tempfile
from collections.abc import MutableMapping
from typing import Protocol

from agl import AglException, nominals, option_none, option_some, runtime

from agm.agl.runtime.host_fs import run_fs_action
from agm.agl.runtime.host_text import scalar_host_text

EncodingError = nominals.std.errors.EncodingError
FsError = nominals.std.fs.FsError

_MIN_EXIT_CODE = 0
_MAX_EXIT_CODE = 255
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
    """Return the current working directory."""
    return scalar_host_text(os.getcwd(), EncodingError)


def pid() -> int:
    """Return this process identifier."""
    return os.getpid()


def hostname() -> str:
    """Return this host's name."""
    return scalar_host_text(socket.gethostname(), EncodingError)


def temp_dir() -> str:
    """Return the host's temporary-file directory."""
    return scalar_host_text(tempfile.gettempdir(), EncodingError)


def _restore_start_directory(directory: str) -> None:
    """Change back to *directory*; skip it if it no longer exists."""
    try:
        os.chdir(directory)
    except FileNotFoundError:
        pass


def chdir(directory: str, vars_: MutableMapping[str, str]) -> None:
    """Change the working directory, recording ``OLDPWD``/``PWD`` in *vars_*.

    A failed `os.chdir` raises `FsError` and leaves the directory unchanged.
    A before or after directory that is not valid Unicode raises
    `EncodingError`; the latter case restores the prior directory first, so
    the directory is unchanged either way. The starting directory is
    registered once, on a session's first `chdir`, to be restored when the
    host session ends.
    """
    before = scalar_host_text(os.getcwd(), EncodingError)
    run_fs_action(FsError, directory, "chdir", lambda: os.chdir(directory))
    try:
        after = scalar_host_text(os.getcwd(), EncodingError)
    except AglException:
        os.chdir(before)
        raise
    runtime.state(_START_DIR_KEY, lambda: before, close=_restore_start_directory)
    vars_["OLDPWD"] = before
    vars_["PWD"] = after


def is_interactive() -> bool:
    """Whether standard input and output are both connected to a terminal."""
    return os.isatty(0) and os.isatty(1)


def which(name: str, env: _EnvironLike) -> object:
    """Look up *name* on *env*'s ``PATH``, the host default when absent."""
    found = shutil.which(name, path=env.vars.get("PATH", os.defpath))
    return option_some(found) if found is not None else option_none()


def platform() -> str:
    """Return the host platform identifier."""
    return sys.platform


def cpu_count() -> int:
    """Return the usable CPU count, at least 1."""
    return os.process_cpu_count() or 1


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
    "exit",
    "hostname",
    "is_interactive",
    "pid",
    "platform",
    "temp_dir",
    "user",
    "which",
]
