"""Path helpers shared across command areas."""

from __future__ import annotations

from pathlib import Path


def path_from_cli(value: str, *, cwd: Path) -> Path:
    """Resolve a CLI path value relative to *cwd*."""

    path = Path(value)
    if path.is_absolute():
        return path
    return cwd / path
