"""Filesystem externs for ``std/fs``."""

from __future__ import annotations

import functools
import glob as glob_module
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from agl import array, nominals, runtime

from agm.agl.runtime.host_fs import raise_fs_error, run_fs_action
from agm.core import fs
from agm.core.cleanup import run_cleanup_steps
from agm.util.unicode import LoneSurrogateError, require_scalar_text, visible_text

FsError = nominals.std.fs.FsError

# Name prefix of every temporary path this module creates.
_TEMP_PREFIX = "agm-"
# The ``runtime.state`` key of the host session's temporary paths, removed when it ends.
_TEMP_PATHS_KEY = "std/fs/temp-paths"


def _ensure_parent_directory(path: Path) -> None:
    if not path.parent.exists():
        fs.mkdir(path.parent, parents=True, exist_ok=True)


def _scalar_entries(path: str, operation: str, names: Iterable[str]) -> list[str]:
    """Return *names* as a list, raising ``FsError`` naming the first undecodable entry.

    Skipping an undecodable entry would silently drop it, so the whole call fails.
    """
    entries: list[str] = []
    for name in names:
        try:
            require_scalar_text(name)
        except LoneSurrogateError:
            raise_fs_error(
                FsError,
                path,
                operation,
                message=f"{fs.fs_error_message(operation, path)} "
                f"Entry '{visible_text(name)}' is not valid Unicode.",
            )
        entries.append(name)
    return entries


def read(path: str) -> str:
    """Read UTF-8 text from *path*."""
    return run_fs_action(FsError, path, "read", lambda: fs.read_text(Path(path)))


def write(path: str, content: str) -> None:
    """Write UTF-8 *content* to *path*, creating missing parent directories."""

    def do() -> None:
        destination = Path(path)
        _ensure_parent_directory(destination)
        fs.write_text(destination, content)

    run_fs_action(FsError, path, "write", do)


def append(path: str, content: str) -> None:
    """Append UTF-8 *content*, creating *path* and missing parent directories."""

    def do() -> None:
        destination = Path(path)
        _ensure_parent_directory(destination)
        fs.append_text(destination, content)

    run_fs_action(FsError, path, "append", do)


def exists(path: str) -> bool:
    """Return whether *path* exists."""
    return run_fs_action(FsError, path, "exists", lambda: fs.exists(Path(path)))


def is_file(path: str) -> bool:
    """Return whether *path* is a regular file."""
    return run_fs_action(FsError, path, "is-file", lambda: fs.is_file(Path(path)))


def is_dir(path: str) -> bool:
    """Return whether *path* is a directory."""
    return run_fs_action(FsError, path, "is-dir", lambda: fs.is_dir(Path(path)))


def list(path: str) -> object:
    """Return immediate child paths of *path*."""

    def do() -> object:
        children = (str(child) for child in fs.iterdir(Path(path)))
        return array(_scalar_entries(path, "list", children))

    return run_fs_action(FsError, path, "list", do)


def mkdir(path: str) -> None:
    """Create *path* and any missing parent directories."""
    run_fs_action(FsError, path, "mkdir", lambda: fs.mkdir(Path(path), parents=True, exist_ok=True))


def _remove_entry(target: Path, *, missing_ok: bool = False) -> None:
    """Remove the file, symbolic link, or directory tree at *target*."""
    if target.is_symlink() or not target.is_dir():
        fs.unlink(target, missing_ok=missing_ok)
    else:
        fs.rmtree(target)


def remove(path: str) -> None:
    """Remove a file, symbolic link, or directory tree at *path*."""
    run_fs_action(FsError, path, "remove", lambda: _remove_entry(Path(path)))


def _prepare_destination(source: str, destination: str, operation: str) -> Path:
    """Return *destination* as a path after creating its parents, only if *source* exists.

    A failure to create the parents is reported against *destination*.
    """
    fs.lstat(Path(source))
    target = Path(destination)
    run_fs_action(FsError, destination, operation, lambda: _ensure_parent_directory(target))
    return target


def copy(source: str, destination: str) -> None:
    """Copy a file from *source* to *destination*, creating its missing parent directories."""

    def do() -> None:
        target = _prepare_destination(source, destination, "copy")
        fs.copy_file(Path(source), target)

    run_fs_action(FsError, source, "copy", do)


def move(source: str, destination: str) -> None:
    """Move a file or directory to *destination*, creating its missing parent directories."""

    def do() -> None:
        target = _prepare_destination(source, destination, "move")
        fs.move(Path(source), target)

    run_fs_action(FsError, source, "move", do)


def glob(pattern: str) -> object:
    """Return paths matching a shell-style *pattern*."""
    return run_fs_action(
        FsError,
        pattern,
        "glob",
        lambda: array(_scalar_entries(pattern, "glob", glob_module.glob(pattern, recursive=True))),
    )


def _checked_temp_root(operation: str) -> str:
    """Return the host's temporary directory, raising ``FsError`` if it is not valid Unicode."""
    directory = tempfile.gettempdir()
    return run_fs_action(
        FsError, visible_text(directory), operation, lambda: require_scalar_text(directory)
    )


def _remove_temp_path(path: Path) -> None:
    """Remove *path*; skip it if already gone, and leave it if it cannot be removed."""
    try:
        _remove_entry(path, missing_ok=True)
    except OSError:
        pass


def _remove_temp_paths(paths: list[Path]) -> None:
    """Remove every temporary path, newest first, on a best-effort basis."""
    run_cleanup_steps([functools.partial(_remove_temp_path, path) for path in reversed(paths)])


def _no_temp_paths() -> list[Path]:
    return []


def _session_temp_path(operation: str, create: Callable[[], Path]) -> str:
    """Create a temporary path with *create* and register it for removal at session end."""
    directory = _checked_temp_root(operation)
    created = run_fs_action(FsError, directory, operation, create)
    runtime.state(
        _TEMP_PATHS_KEY, _no_temp_paths, close=_remove_temp_paths, keep_in_debug=True
    ).append(created)
    return str(created)


def _make_temp_file(suffix: str) -> Path:
    handle, name = tempfile.mkstemp(prefix=_TEMP_PREFIX, suffix=suffix)
    os.close(handle)
    return Path(name)


def temp_file(suffix: str = "") -> str:
    """Create a new empty temporary file named with *suffix*; removed at session end."""
    return _session_temp_path("temp-file", lambda: _make_temp_file(suffix))


def temp_dir() -> str:
    """Create a new temporary directory; removed with its contents at session end."""
    return _session_temp_path("temp-dir", lambda: Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX)))


__all__ = [
    "append",
    "copy",
    "exists",
    "glob",
    "is_dir",
    "is_file",
    "list",
    "mkdir",
    "move",
    "read",
    "remove",
    "temp_dir",
    "temp_file",
    "write",
]
