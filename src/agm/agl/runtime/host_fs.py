"""Host filesystem-action execution shared by the ``fs``, ``os``, and ``http`` companions.

Each companion synthesizes its own ``std/fs::FsError`` class from the running
program's type table, so this helper takes it as an argument rather than
importing it, mirroring the ``EncodingErrorFactory`` pattern in
:mod:`agm.agl.runtime.host_text`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn, Protocol, TypeVar

from agm.agl.runtime.boundary import AglException
from agm.core.fs import fs_error_message

__all__ = ["FsErrorFactory", "raise_fs_error", "run_fs_action"]

T = TypeVar("T")


class FsErrorFactory(Protocol):
    """A synthesized ``std/fs::FsError`` class, as a companion sees it."""

    def __call__(self, *, message: str, path: str, operation: str) -> object: ...


def raise_fs_error(
    fs_error: FsErrorFactory, path: str, operation: str, *, message: str | None = None
) -> NoReturn:
    """Raise *fs_error* for a failed *operation* on *path*."""
    raise AglException(
        fs_error(
            message=message if message is not None else fs_error_message(operation, path),
            path=path,
            operation=operation,
        )
    )


def run_fs_action(
    fs_error: FsErrorFactory, path: str, operation: str, action: Callable[[], T]
) -> T:
    """Run *action*, raising *fs_error* for a NUL byte in *path* or a failed host call.

    Catches ``OSError``, ``UnicodeDecodeError``, and ``ValueError`` -- what
    Python's filesystem and process calls raise for a bad path.
    """
    if "\x00" in path:
        raise_fs_error(fs_error, path, operation)
    try:
        return action()
    except (OSError, UnicodeDecodeError, ValueError):
        raise_fs_error(fs_error, path, operation)
