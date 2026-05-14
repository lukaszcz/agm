"""Path helpers shared across command areas."""

from __future__ import annotations

from pathlib import Path


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
