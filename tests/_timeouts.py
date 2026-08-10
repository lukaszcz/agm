"""Small test deadlines for regressions whose failure mode is nontermination."""

from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType


@contextmanager
def fail_if_slow(message: str, *, seconds: float = 2.0) -> Iterator[None]:
    """Fail the current main-thread test if its guarded block does not terminate."""

    def fail(_signum: int, _frame: FrameType | None) -> None:
        raise AssertionError(message)

    previous = signal.signal(signal.SIGALRM, fail)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
