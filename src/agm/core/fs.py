"""Filesystem helpers that respect dry-run mode."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from agm.core import dry_run
from agm.core.path import display_path


def exists(path: Path) -> bool:
    """Return whether *path* exists."""

    return path.exists()


def is_file(path: Path) -> bool:
    """Return whether *path* is a file."""

    return path.is_file()


def is_dir(path: Path) -> bool:
    """Return whether *path* is a directory."""

    return path.is_dir()


def read_text(path: Path, *, encoding: str = "utf-8") -> str:
    """Read text from *path*."""

    return path.read_text(encoding=encoding)


def read_text_arg(path: Path, *, encoding: str = "utf-8") -> str:
    """Read text from a user-supplied *path* argument.

    On failure, print a friendly ``Error: ...`` message to stderr (using the
    repo's display-path convention) and raise ``SystemExit(1)``.
    """

    try:
        return path.read_text(encoding=encoding)
    except OSError as exc:
        print(f"Error: cannot read {display_path(path)}: {exc.strerror}", file=sys.stderr)
        raise SystemExit(1) from exc


def stat(path: Path) -> os.stat_result:
    """Return stat information for *path*."""

    return path.stat()


def iterdir(path: Path) -> list[Path]:
    """Return the immediate children of *path*."""

    return list(path.iterdir())


def rglob(path: Path, pattern: str) -> list[Path]:
    """Return recursive glob matches under *path*."""

    return list(path.rglob(pattern))


def is_empty_dir(path: Path) -> bool:
    """Return whether *path* is an empty directory."""

    return not any(iterdir(path))


def access(path: Path, mode: int) -> bool:
    """Return whether *path* is accessible with *mode*."""

    return os.access(path, mode)


def mkdir(path: Path, *, parents: bool = False, exist_ok: bool = False) -> None:
    """Create a directory unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("mkdir", str(path))
        return
    path.mkdir(parents=parents, exist_ok=exist_ok)


def write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("write-file", str(path))
        return
    path.write_text(content, encoding=encoding)


def chmod(path: Path, mode: int) -> None:
    """Change file mode unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("chmod", f"{oct(mode)} {path}")
        return
    path.chmod(mode)


def append_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Append text unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("append-file", str(path))
        return
    with path.open("a", encoding=encoding) as handle:
        handle.write(content)


def rmtree(path: Path) -> None:
    """Remove a directory tree unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("remove-tree", str(path))
        return
    shutil.rmtree(path)


def rmdir(path: Path) -> None:
    """Remove an empty directory unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("rmdir", str(path))
        return
    path.rmdir()


def unlink(path: Path, *, missing_ok: bool = False) -> None:
    """Remove a file unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("unlink", str(path))
        return
    path.unlink(missing_ok=missing_ok)
