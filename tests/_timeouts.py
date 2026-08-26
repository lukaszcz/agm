"""Small test deadlines for regressions whose failure mode is nontermination."""

from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType


class DeadlineExceeded(BaseException):
    """Raised when a guarded block outlives its deadline.

    Deliberately a :class:`BaseException` rather than an ``AssertionError``:
    guarded blocks run production code whose broad ``except Exception``
    handlers adapt unexpected failures into diagnostics. Such a handler would
    absorb the deadline and let the test continue against corrupted state,
    turning a nontermination bug into an unrelated assertion failure further
    down the test.
    """


@contextmanager
def fail_if_slow(message: str, *, seconds: float = 2.0) -> Iterator[None]:
    """Fail the current main-thread test if its guarded block does not terminate."""

    def fail(_signum: int, _frame: FrameType | None) -> None:
        raise DeadlineExceeded(message)

    previous = signal.signal(signal.SIGALRM, fail)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
