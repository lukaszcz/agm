"""The per-test CPU budget that guards suite performance.

The accounting itself is exercised by every run of the suite — these tests pin
the parts that only fire on an overrun or under ``-n auto``, which a green run
never reaches.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from tests import _durations

#: CPU seconds a measurement has to move before it means anything. ``os.times``
#: is quantised to clock ticks (10ms on Linux) and the counters it returns are
#: whole-process totals, so a difference taken late in a long run loses the last
#: digits to floating-point as well. Several ticks of headroom clears both.
_MARGIN = 0.05

#: Wall-clock ceiling on the repeat loops below. A working counter passes it on
#: the first iteration or two; the ceiling only matters if the counter never
#: moves, and then it turns a hang into a failed assertion.
_DEADLINE = 10.0


def _repeat_until_measurable(measure: Callable[[], float], work: Callable[[], None]) -> float:
    """Run ``work`` until ``measure`` reports more than ``_MARGIN``, and report it.

    Asserting that a fixed amount of work registers on the CPU counter would be
    asserting how fast this machine runs that work: the same loop measures one
    tick or five depending on interpreter warm-up and on whether coverage
    tracing is on. Repeating until the counter has actually moved tests the
    accounting instead, and costs only as much work as that takes.
    """
    deadline = time.monotonic() + _DEADLINE
    measured = measure()
    while measured <= _MARGIN and time.monotonic() < deadline:
        work()
        measured = measure()
    return measured


def _own_cpu_seconds() -> float:
    """CPU seconds burned by this process alone, charging nothing to children."""
    times = os.times()
    return times.user + times.system


class _Reporter:
    """Stands in for pytest's terminal reporter, recording what was written."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_sep(self, _char: str, title: str) -> None:
        self.lines.append(title)

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def _session(reporter: _Reporter | None) -> SimpleNamespace:
    plugin_manager = SimpleNamespace(get_plugin=lambda _name: reporter)
    return SimpleNamespace(
        config=SimpleNamespace(pluginmanager=plugin_manager), exitstatus=pytest.ExitCode.OK
    )


class TestCpuSeconds:
    def test_counts_work_done_in_this_process(self) -> None:
        before = _durations.cpu_seconds()

        charged = _repeat_until_measurable(
            lambda: _durations.cpu_seconds() - before, lambda: sum(range(1_000_000))
        )

        assert charged > _MARGIN

    def test_counts_work_done_by_a_reaped_child(self) -> None:
        """Most of the suite's cost is subprocesses, so they have to be charged."""

        def burn_in_a_child() -> None:
            subprocess.run(
                [sys.executable, "-c", "sum(range(2_000_000))"], check=True, capture_output=True
            )

        before_total = _durations.cpu_seconds()
        before_own = _own_cpu_seconds()

        # Net of what this process spent spawning them, so the number can only
        # come from the children themselves.
        charged_to_children = _repeat_until_measurable(
            lambda: (_durations.cpu_seconds() - before_total) - (_own_cpu_seconds() - before_own),
            burn_in_a_child,
        )

        assert charged_to_children > _MARGIN


class TestBudgetEnforcement:
    def test_a_run_within_budget_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_durations.BUDGET_ENV, "1.0")
        monkeypatch.delenv(_durations.REPORT_TOP_ENV, raising=False)
        reporter = _Reporter()
        session = _session(reporter)

        _durations._enforce(session, {"fast": 0.25, "also_fast": 0.9})

        assert session.exitstatus == pytest.ExitCode.OK
        assert reporter.lines == []

    def test_an_overrun_fails_the_session_and_names_every_offender(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_durations.BUDGET_ENV, "1.0")
        reporter = _Reporter()
        session = _session(reporter)

        _durations._enforce(session, {"slow": 4.0, "fine": 0.5, "slower": 9.0})

        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
        reported = "\n".join(reporter.lines)
        assert "slower" in reported
        assert "slow" in reported
        assert "fine" not in reported

    def test_the_report_is_ordered_worst_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_durations.BUDGET_ENV, "0.1")
        reporter = _Reporter()

        _durations._enforce(_session(reporter), {"middle": 2.0, "worst": 5.0, "least": 1.0})

        offenders = [line.split()[-1] for line in reporter.lines[1:-1]]
        assert offenders == ["worst", "middle", "least"]

    def test_no_budget_reports_nothing_and_fails_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(_durations.BUDGET_ENV, raising=False)
        monkeypatch.delenv(_durations.REPORT_TOP_ENV, raising=False)
        reporter = _Reporter()
        session = _session(reporter)

        _durations._enforce(session, {"slow": 30.0})

        assert session.exitstatus == pytest.ExitCode.OK
        assert reporter.lines == []

    @pytest.mark.parametrize("raw", ["", "not-a-number"], ids=["empty", "garbage"])
    def test_an_unusable_budget_is_no_budget(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        """``just test-budget`` passes an empty value when no budget is asked for."""
        monkeypatch.setenv(_durations.BUDGET_ENV, raw)
        session = _session(_Reporter())

        _durations._enforce(session, {"slow": 30.0})

        assert session.exitstatus == pytest.ExitCode.OK

    def test_top_report_is_capped_and_does_not_fail_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_durations.REPORT_TOP_ENV, "2")
        monkeypatch.delenv(_durations.BUDGET_ENV, raising=False)
        reporter = _Reporter()
        session = _session(reporter)

        _durations._enforce(session, {"a": 3.0, "b": 2.0, "c": 1.0})

        assert session.exitstatus == pytest.ExitCode.OK
        assert [line.split()[-1] for line in reporter.lines[1:]] == ["a", "b"]

    def test_a_measurement_free_run_reports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_durations.BUDGET_ENV, "0.0")
        reporter = _Reporter()
        session = _session(reporter)

        _durations._enforce(session, {})

        assert session.exitstatus == pytest.ExitCode.OK
        assert reporter.lines == []

    def test_it_survives_a_run_with_no_terminal_reporter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``-p no:terminal`` must still fail the run, just silently."""
        monkeypatch.setenv(_durations.BUDGET_ENV, "1.0")
        monkeypatch.setenv(_durations.REPORT_TOP_ENV, "5")
        session = _session(None)

        _durations._enforce(session, {"slow": 4.0})

        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


class TestDistributedRuns:
    def test_a_worker_ships_its_measurements_instead_of_judging_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_durations, "_costs", {"one": 5.0})
        monkeypatch.setenv(_durations.BUDGET_ENV, "0.1")
        workeroutput: dict[str, object] = {}
        session = _session(_Reporter())
        session.config.workeroutput = workeroutput

        _durations.pytest_sessionfinish(session)

        assert workeroutput[_durations._WORKEROUTPUT_KEY] == {"one": 5.0}
        assert session.exitstatus == pytest.ExitCode.OK

    def test_the_controller_judges_what_every_worker_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_durations, "_costs", {})
        monkeypatch.setenv(_durations.BUDGET_ENV, "1.0")
        reporter = _Reporter()
        session = _session(reporter)

        for costs in ({"from_gw0": 4.0}, {"from_gw1": 0.5}):
            node = SimpleNamespace(workeroutput={_durations._WORKEROUTPUT_KEY: costs})
            _durations.pytest_testnodedown(node, None)
        _durations.pytest_sessionfinish(session)

        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
        reported = "\n".join(reporter.lines)
        assert "from_gw0" in reported
        assert "from_gw1" not in reported

    def test_a_worker_that_reported_nothing_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker that crashed before sessionfinish has no ``workeroutput``."""
        monkeypatch.setattr(_durations, "_costs", {})

        _durations.pytest_testnodedown(SimpleNamespace(), None)
        _durations.pytest_testnodedown(SimpleNamespace(workeroutput={}), None)

        assert _durations._costs == {}


def test_the_cost_hook_charges_the_test_it_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook is live for every run, not only when a budget is set."""
    monkeypatch.setattr(_durations, "_costs", {})
    item = SimpleNamespace(nodeid="tests/example.py::test_thing")

    wrapper = _durations.pytest_runtest_protocol(item, None)
    next(wrapper)
    sum(range(200_000))
    with pytest.raises(StopIteration):
        next(wrapper)

    assert _durations._costs["tests/example.py::test_thing"] >= 0.0
