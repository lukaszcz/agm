"""Filesystem helpers that respect dry-run mode."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from stat import S_IMODE
from uuid import uuid4

from agm.core import dry_run
from agm.core.path import display_path

# (modification time, size, inode) -- see :func:`identity_stamp`.
IdentityStamp = tuple[int, int, int]


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


def read_text_arg_or_none(path: Path, *, encoding: str = "utf-8") -> str | None:
    """Read text from a user-supplied *path* argument, reporting failure.

    On failure, print a friendly ``Error: ...`` message to stderr (using the
    repo's display-path convention) and return ``None`` instead of raising —
    the seam for a command that loops over many inputs and must keep checking
    the rest after one fails. ``read_text_arg`` is the raising wrapper over
    this for callers with a single input to read.
    """

    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        print(
            f"Error: cannot read {display_path(path)}: file is not valid UTF-8 text",
            file=sys.stderr,
        )
        return None
    except OSError as exc:
        detail = exc.strerror if exc.strerror is not None else str(exc)
        print(f"Error: cannot read {display_path(path)}: {detail}", file=sys.stderr)
        return None


def read_text_arg(path: Path, *, encoding: str = "utf-8") -> str:
    """Read text from a user-supplied *path* argument.

    On failure, print a friendly ``Error: ...`` message to stderr (using the
    repo's display-path convention) and raise ``SystemExit(1)``.
    """

    text = read_text_arg_or_none(path, encoding=encoding)
    if text is None:
        raise SystemExit(1)
    return text


def stat(path: Path) -> os.stat_result:
    """Return stat information for *path*."""

    return path.stat()


def identity_stamp(path: Path) -> IdentityStamp:
    """Return a stamp that changes whenever *path*'s contents could have.

    Callers that cache something derived from a file compare this stamp to
    decide whether the cached artifact still describes the file. Size alone
    misses a same-length rewrite and modification time alone misses a file
    swapped in under a preserved timestamp, so the stamp carries both plus the
    inode, which changes when a file is replaced rather than written through.
    """

    info = stat(path)
    return (info.st_mtime_ns, info.st_size, info.st_ino)


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
        dry_run.print_operation("mkdir", display_path(path))
        return
    path.mkdir(parents=parents, exist_ok=exist_ok)


def write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("write-file", display_path(path))
        return
    path.write_text(content, encoding=encoding)


def write_text_atomic(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text through a temporary sibling that replaces *path* in one step.

    Concurrent readers therefore observe either the previous file or the
    complete new content, never a partially written file. Existing permissions
    are retained, and the temporary file remains private while it is populated.
    """

    if dry_run.enabled():
        write_text(path, content)
        return
    destination_mode = S_IMODE(path.stat().st_mode) if path.exists() else None
    temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary_path.open("x", encoding=encoding) as temporary:
            default_mode = S_IMODE(os.fstat(temporary.fileno()).st_mode)
            os.fchmod(temporary.fileno(), 0o600)
            temporary.write(content)
        temporary_path.chmod(default_mode if destination_mode is None else destination_mode)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def chmod(path: Path, mode: int) -> None:
    """Change file mode unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("chmod", f"{oct(mode)} {display_path(path)}")
        return
    path.chmod(mode)


def append_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Append text unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("append-file", display_path(path))
        return
    with path.open("a", encoding=encoding) as handle:
        handle.write(content)


def copy_file(source: Path, destination: Path) -> None:
    """Copy one file unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("copy-file", f"{display_path(source)} {display_path(destination)}")
        return
    shutil.copy2(source, destination)


def move(source: Path, destination: Path) -> None:
    """Move a file or directory unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("move", f"{display_path(source)} {display_path(destination)}")
        return
    shutil.move(source, destination)


def copy_tree(source: Path, destination: Path, *, dirs_exist_ok: bool = False) -> None:
    """Copy a tree, preserving source links and refusing linked roots or destinations.

    Descendant symbolic links are copied as links rather than dereferenced. A linked
    source root does not designate a tree, and linked destination paths or ancestors
    could redirect a merge outside its requested destination, so both are rejected.
    """

    if dry_run.enabled():
        dry_run.print_operation("copy-tree", f"{display_path(source)} {display_path(destination)}")
        return
    if source.is_symlink():
        raise ValueError(f"cannot copy symbolic-link tree root {source}")
    destination_path = Path.cwd() / destination
    for path in (*destination_path.parents, destination_path, *destination.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"cannot copy tree into symbolic-link destination path {path}")
    shutil.copytree(
        source,
        destination,
        copy_function=shutil.copy2,
        dirs_exist_ok=dirs_exist_ok,
        symlinks=True,
    )


def rmtree(path: Path) -> None:
    """Remove a directory tree unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("remove-tree", display_path(path))
        return
    shutil.rmtree(path)


def rmdir(path: Path) -> None:
    """Remove an empty directory unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("rmdir", display_path(path))
        return
    path.rmdir()


def unlink(path: Path, *, missing_ok: bool = False) -> None:
    """Remove a file unless dry-run is enabled."""

    if dry_run.enabled():
        dry_run.print_operation("unlink", display_path(path))
        return
    path.unlink(missing_ok=missing_ok)
