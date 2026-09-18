"""Tests for agm.core.cleanup: exception-preserving cleanup helpers."""

from __future__ import annotations

import pytest

from agm.core.cleanup import preserve_primary_error, run_cleanup_steps


def test_preserve_primary_error_runs_cleanup_on_a_clean_exit() -> None:
    ran = []
    with preserve_primary_error(lambda: ran.append(True), label="cleanup"):
        pass
    assert ran == [True]


def test_preserve_primary_error_keeps_the_active_exception_and_still_cleans_up() -> None:
    ran = []
    with pytest.raises(ValueError, match="primary"):
        with preserve_primary_error(lambda: ran.append(True), label="cleanup"):
            raise ValueError("primary")
    assert ran == [True]


def test_preserve_primary_error_attaches_a_failing_cleanup_as_a_note() -> None:
    def _fail() -> None:
        raise RuntimeError("cleanup failed")

    with pytest.raises(ValueError, match="primary") as excinfo:
        with preserve_primary_error(_fail, label="my cleanup"):
            raise ValueError("primary")

    assert any("my cleanup also failed: cleanup failed" in note for note in excinfo.value.__notes__)


def test_run_cleanup_steps_runs_every_step_in_order() -> None:
    order: list[str] = []
    run_cleanup_steps([lambda: order.append("first"), lambda: order.append("second")])
    assert order == ["first", "second"]


def test_run_cleanup_steps_with_no_steps_does_nothing() -> None:
    run_cleanup_steps([])


def test_run_cleanup_steps_runs_every_step_then_reraises_the_first_failure() -> None:
    ran: list[str] = []

    def _fail(message: str) -> None:
        ran.append(message)
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="first"):
        run_cleanup_steps([lambda: _fail("first"), lambda: ran.append("second")])

    assert ran == ["first", "second"]


def test_run_cleanup_steps_attaches_each_later_failure_as_a_note_on_the_first() -> None:
    def _fail(message: str) -> None:
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="first") as excinfo:
        run_cleanup_steps([lambda: _fail("first"), lambda: _fail("second")])

    assert any("second" in note for note in excinfo.value.__notes__)


def test_run_cleanup_steps_catches_base_exception_so_one_step_never_skips_the_rest() -> None:
    """A ``KeyboardInterrupt`` in one step must not prevent later steps from running."""
    ran: list[str] = []

    def _interrupt() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_cleanup_steps([_interrupt, lambda: ran.append("second")])

    assert ran == ["second"]
