"""Cleanup helpers that retain exceptions already in flight."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager


@contextmanager
def preserve_primary_error(cleanup: Callable[[], None], *, label: str) -> Generator[None]:
    """Run *cleanup*, without allowing it to replace an active exception.

    A cleanup failure remains observable on a clean exit.  If execution is
    already failing, attach that failure to the active exception instead.
    """
    try:
        yield
    except BaseException as primary_error:
        try:
            cleanup()
        except BaseException as cleanup_error:
            primary_error.add_note(f"{label} also failed: {cleanup_error}")
        raise
    else:
        cleanup()
