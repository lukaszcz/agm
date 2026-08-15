"""Path helpers shared across command areas."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def is_portable_relative_path(value: str) -> bool:
    """Whether *value* is a safe, canonical, portable POSIX-relative path.

    Rejects empty values, values naming no path component at all (``.`` and its
    spellings), backslashes, NUL bytes, absolute POSIX paths, Windows
    drive-relative or rooted paths, ``.``/``..`` components, and any value that
    does not already equal its own canonical POSIX rendering. This is the
    shared core of every "safe relative path" check in the codebase; callers
    that need additional constraints (line breaks, path depth, reserved names,
    an anchor escape check, ...) apply them on top of this predicate.
    """

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return not (
        not value
        or "\\" in value
        or "\x00" in value
        or posix.is_absolute()
        or windows.drive
        or windows.root
        or not posix.parts
        or any(part in {".", ".."} for part in posix.parts)
        or posix.as_posix() != value
    )


def path_from_cli(value: str, *, cwd: Path) -> Path:
    """Resolve a CLI path value relative to *cwd*."""

    path = Path(value)
    if path.is_absolute():
        return path
    return cwd / path


def display_path(path: Path, *, cwd: Path | None = None) -> str:
    """Return a user-facing string for *path*.

    If *path* is under *cwd*, return a relative path; otherwise return the
    absolute path unchanged.  When *cwd* is ``None``, ``Path.cwd()`` is used.
    """

    base = cwd if cwd is not None else Path.cwd()
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
