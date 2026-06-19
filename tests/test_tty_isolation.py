"""Regression guard for the controlling-terminal isolation in ``conftest``.

The suite used to intermittently hang with ``zsh: suspended (tty input)`` when an
external process spawned by a test (an interactive shell wrapper, the fake
``tmux`` script, a runner) read the controlling terminal from a background
process group, raising ``SIGTTIN``.  ``conftest.pytest_configure`` detaches each
test-running process from its controlling terminal so ``/dev/tty`` is no longer
reachable and ``SIGTTIN`` can never fire.

These tests fail (on a real terminal) if that detachment is removed.
"""

from __future__ import annotations

import os

import pytest

from tests import conftest


@pytest.mark.skipif(
    not conftest.CONTROLLING_TTY_DETACHED,
    reason="terminal detachment was not attempted (process was a session leader)",
)
def test_no_controlling_terminal() -> None:
    """Once detached, opening ``/dev/tty`` must fail — no terminal to grab."""
    with pytest.raises(OSError):
        fd = os.open("/dev/tty", os.O_RDWR)
        os.close(fd)


@pytest.mark.skipif(
    not conftest.CONTROLLING_TTY_DETACHED,
    reason="terminal detachment was not attempted (process was a session leader)",
)
def test_subprocess_cannot_reach_controlling_terminal() -> None:
    """A spawned child inherits the terminal-less session, so it cannot SIGTTIN.

    This mirrors what every external process the suite launches sees: ``/dev/tty``
    is unopenable, so a background read can never suspend the test job.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os\n"
            "try:\n"
            "    fd = os.open('/dev/tty', os.O_RDWR); os.close(fd); print('OPENED')\n"
            "except OSError:\n"
            "    print('NO_TTY')\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "NO_TTY"
