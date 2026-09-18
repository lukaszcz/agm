"""Tests for the shared, overlap-safe recursion-limit raise."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

from agm.util.recursion import raised_recursion_limit


@pytest.fixture
def base_limit() -> Iterator[int]:
    """Pin a known limit and restore the caller's afterwards."""
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        yield 1000
    finally:
        sys.setrecursionlimit(previous)


def test_raises_for_the_duration_and_restores(base_limit: int) -> None:
    with raised_recursion_limit(5000):
        assert sys.getrecursionlimit() == 5000
    assert sys.getrecursionlimit() == base_limit


def test_never_lowers_an_already_higher_limit(base_limit: int) -> None:
    with raised_recursion_limit(base_limit - 1):
        assert sys.getrecursionlimit() == base_limit
    assert sys.getrecursionlimit() == base_limit


def test_overlapping_raises_restore_the_original_limit_whatever_the_exit_order(
    base_limit: int,
) -> None:
    # Two concurrent runs (e.g. interpreters on two threads) overlap without
    # nesting: the first to enter is the first to leave.
    first = raised_recursion_limit(5000)
    second = raised_recursion_limit(3000)
    first.__enter__()
    second.__enter__()
    assert sys.getrecursionlimit() == 5000
    first.__exit__(None, None, None)
    assert sys.getrecursionlimit() == 5000
    second.__exit__(None, None, None)
    assert sys.getrecursionlimit() == base_limit


def test_a_later_higher_overlapping_raise_takes_effect(base_limit: int) -> None:
    with raised_recursion_limit(3000):
        with raised_recursion_limit(5000):
            assert sys.getrecursionlimit() == 5000
        assert sys.getrecursionlimit() == 5000
    assert sys.getrecursionlimit() == base_limit


def test_restores_after_an_exception(base_limit: int) -> None:
    with pytest.raises(ValueError), raised_recursion_limit(5000):
        raise ValueError
    assert sys.getrecursionlimit() == base_limit
