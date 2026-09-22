"""A per-test cost budget that does not depend on how busy the machine is.

Wall-clock duration is useless as a suite-wide quality gate here.  ``just
test`` runs under ``-n auto``, which on this project's hardware means one
worker per core: every test's wall time then includes the time it spent
descheduled while its siblings ran, so the same test can report 0.3s alone
and 3s in a full run.  A wall-clock threshold would fail or pass according
to the load average, which is exactly the flakiness the testing guidelines
forbid.

What a test actually costs is CPU time, and ``os.times`` reports it for the
worker process *and* for the subprocesses it reaped -- which matters here,
because a large part of the suite drives ``git``, fake runner scripts and
real ``agm`` invocations.  That number is stable whether the test runs alone
or alongside twenty others, so it can be enforced.  A process the worker did
not reap itself charges its cost here explicitly, so how an invocation was
launched never changes what it is measured to cost.

Set ``AGM_TEST_MAX_CPU_SECONDS`` to a budget and the session fails, listing
every test that overran it.  Leave it unset (the default) and the accounting
still happens but costs nothing observable; ``AGM_TEST_REPORT_TOP`` prints
the most expensive tests, which is how the budget gets calibrated.

The numbers are configuration-sensitive in one direction only: coverage
tracing multiplies them.  A budget is therefore meaningful only against a
stated configuration -- see the ``test-budget`` recipe in the justfile.
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

BUDGET_ENV = "AGM_TEST_MAX_CPU_SECONDS"
REPORT_TOP_ENV = "AGM_TEST_REPORT_TOP"

_WORKEROUTPUT_KEY = "agm_test_cpu_seconds"

# nodeid -> CPU seconds, for the tests this process ran.
_costs: dict[str, float] = {}

# CPU seconds burned on this worker's behalf by processes it did not reap
# itself, and so cannot see in ``os.times``.
_external: float = 0.0


def charge_external_cpu_seconds(seconds: float) -> None:
    """Charge the running test with CPU burned by a process another parent reaped.

    The preforking launcher in :mod:`tests._agm_zygote` has the zygote reap the
    ``agm`` children, which takes their cost out of this worker's ``os.times``.
    Reporting it here keeps a test's measured cost the same whichever way its
    invocations were launched.
    """
    global _external
    _external += seconds


def cpu_seconds() -> float:
    """Total CPU seconds burned by this process and on its behalf."""
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system + _external


def _float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Generator[None, None, None]:
    """Charge one test with the CPU its whole setup/call/teardown consumed."""
    before = cpu_seconds()
    yield
    _costs[item.nodeid] = cpu_seconds() - before


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Hand a worker's measurements to the controller, or enforce the budget."""
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:
        # An xdist worker: ship the numbers up and let the controller judge.
        workeroutput[_WORKEROUTPUT_KEY] = _costs
        return
    _enforce(session, _costs)


def pytest_testnodedown(node: object, error: object) -> None:
    """Collect one finished xdist worker's measurements on the controller."""
    output = getattr(node, "workeroutput", None) or {}
    _costs.update(output.get(_WORKEROUTPUT_KEY, {}))


def _enforce(session: pytest.Session, costs: dict[str, float]) -> None:
    """Report the most expensive tests and fail the session if any overran."""
    if not costs:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    ranked = sorted(costs.items(), key=lambda item: item[1], reverse=True)

    top = _float_env(REPORT_TOP_ENV)
    if top is not None and reporter is not None:
        reporter.write_sep("=", "most expensive tests (CPU seconds)")
        for nodeid, cost in ranked[: int(top)]:
            reporter.write_line(f"{cost:8.2f}s  {nodeid}")

    budget = _float_env(BUDGET_ENV)
    if budget is None:
        return
    overruns = [(nodeid, cost) for nodeid, cost in ranked if cost > budget]
    if not overruns:
        return
    if reporter is not None:
        reporter.write_sep("=", f"tests over the {budget}s CPU budget")
        for nodeid, cost in overruns:
            reporter.write_line(f"{cost:8.2f}s  {nodeid}")
        reporter.write_line(f"{len(overruns)} test(s) exceeded {BUDGET_ENV}={budget}")
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
