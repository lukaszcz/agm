"""Overlap-safe raising of Python's process-global recursion limit."""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_lock = threading.Lock()
_holders = 0
_original_limit = 0


@contextmanager
def raised_recursion_limit(target: int) -> Iterator[None]:
    """Keep the recursion limit at least *target* while the block runs.

    The limit is global while callers may overlap (nested or on other
    threads), so the original limit is restored only when the last holder
    leaves, and it is never lowered while any holder remains.
    """
    global _holders, _original_limit
    with _lock:
        if _holders == 0:
            _original_limit = sys.getrecursionlimit()
        _holders += 1
        sys.setrecursionlimit(max(sys.getrecursionlimit(), target))
    try:
        yield
    finally:
        with _lock:
            _holders -= 1
            if _holders == 0:
                sys.setrecursionlimit(_original_limit)
