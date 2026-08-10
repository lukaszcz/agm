"""Filesystem externs for ``std/fs``."""

from pathlib import Path

from agl import array
from agm.core import fs


def read(path: str) -> str:
    """Read UTF-8 text from *path*."""
    return fs.read_text(Path(path))


def write(path: str, content: str) -> None:
    """Write UTF-8 *content* to *path*."""
    fs.write_text(Path(path), content)


def append(path: str, content: str) -> None:
    """Append UTF-8 *content* to *path*."""
    fs.append_text(Path(path), content)


def exists(path: str) -> bool:
    """Return whether *path* exists."""
    return fs.exists(Path(path))


def list(path: str) -> list[str]:
    """Return immediate child paths of *path*."""
    return array([str(child) for child in fs.iterdir(Path(path))])


__all__ = ["append", "exists", "list", "read", "write"]
