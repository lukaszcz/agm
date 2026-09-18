"""Cleanup helpers that retain exceptions already in flight."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import cast


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


def run_cleanup_steps(steps: Sequence[Callable[[], None]]) -> None:
    """Run every *step*, even after an earlier one raises.

    Catches ``BaseException`` so one step's ``KeyboardInterrupt`` never skips
    the rest. The first failure propagates once every step has run, with each
    later failure attached to it as a note.
    """
    primary: BaseException | None = None
    for step in steps:
        try:
            step()
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note(f"cleanup step also failed: {exc}")
    if primary is not None:
        raise primary


def notes_of(exc: BaseException) -> tuple[str, ...]:
    """Return every note attached to *exc* (e.g. by :func:`preserve_primary_error`)."""
    return tuple(cast(list[str], getattr(exc, "__notes__", ())))
