"""Filesystem externs for ``std/fs``."""

from __future__ import annotations

import glob as glob_module
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, TypeVar

from agl import AglException, array, nominals

from agm.core import fs

Option = nominals.std.option.Option
FsError = nominals.std.fs.FsError

T = TypeVar("T")


def _raise_fs_error(path: str, operation: str) -> NoReturn:
    raise AglException(
        FsError(
            message=f"Could not {operation} {path}.",
            path=path,
            operation=operation,
        )
    )


def _run(path: str, operation: str, action: Callable[[], T]) -> T:
    if "\x00" in path:
        _raise_fs_error(path, operation)
    try:
        return action()
    except (OSError, UnicodeDecodeError, ValueError):
        _raise_fs_error(path, operation)


def read(path: str) -> str:
    """Read UTF-8 text from *path*."""
    return _run(path, "read", lambda: fs.read_text(Path(path)))


def read_option(path: str) -> object:
    """Read UTF-8 text from *path*, returning ``Option::None`` on failure."""
    try:
        return Option.Some(value=read(path))
    except AglException:
        return getattr(Option, "None")()


def write(path: str, content: str) -> None:
    """Write UTF-8 *content* to *path*."""
    _run(path, "write", lambda: fs.write_text(Path(path), content))


def append(path: str, content: str) -> None:
    """Append UTF-8 *content* to *path*."""
    _run(path, "append", lambda: fs.append_text(Path(path), content))


def exists(path: str) -> bool:
    """Return whether *path* exists."""
    return _run(path, "exists", lambda: fs.exists(Path(path)))


def is_file(path: str) -> bool:
    """Return whether *path* is a regular file."""
    return _run(path, "is-file", lambda: fs.is_file(Path(path)))


def is_dir(path: str) -> bool:
    """Return whether *path* is a directory."""
    return _run(path, "is-dir", lambda: fs.is_dir(Path(path)))


def list(path: str) -> object:
    """Return immediate child paths of *path*."""
    return _run(path, "list", lambda: array([str(child) for child in fs.iterdir(Path(path))]))


def mkdir(path: str) -> None:
    """Create *path* and any missing parent directories."""
    _run(path, "mkdir", lambda: fs.mkdir(Path(path), parents=True, exist_ok=True))


def remove(path: str) -> None:
    """Remove a file, symbolic link, or directory tree at *path*."""
    target = Path(path)

    def drop() -> None:
        if target.is_symlink() or not target.is_dir():
            fs.unlink(target)
        else:
            fs.rmtree(target)

    _run(path, "remove", drop)


def copy(source: str, destination: str) -> None:
    """Copy a file from *source* to *destination*."""
    _run(source, "copy", lambda: fs.copy_file(Path(source), Path(destination)))


def move(source: str, destination: str) -> None:
    """Move a file or directory from *source* to *destination*."""
    _run(source, "move", lambda: fs.move(Path(source), Path(destination)))


def glob(pattern: str) -> object:
    """Return paths matching a shell-style *pattern*."""
    return _run(pattern, "glob", lambda: array(glob_module.glob(pattern, recursive=True)))


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
    "write",
]
