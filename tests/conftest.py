"""Shared test fixtures."""

from __future__ import annotations

import os
import shutil
import signal
from collections.abc import Generator
from pathlib import Path

import pytest

from agm.core import dry_run

# Set to ``True`` once :func:`pytest_configure` has successfully detached the
# current (test-running) process from its controlling terminal.  Consulted by
# the ``tests/test_tty_isolation.py`` regression guard.
CONTROLLING_TTY_DETACHED = False


def _detach_from_controlling_terminal() -> None:
    """Put this process in a new session so it has no controlling terminal.

    Tests spawn many external processes — interactive shells through the
    workspace-shell wrapper (``bash -i`` / ``zsh -i``), the fake ``tmux`` shell
    script, runner/selector scripts, ``git``.  Each of these opens ``/dev/tty``
    on startup.  When the suite runs in a real terminal and one of those
    processes reads the controlling terminal while *not* in the foreground
    process group, the kernel raises ``SIGTTIN`` and suspends the whole
    ``just check`` job — the intermittent ``zsh: suspended (tty input)`` hang.

    ``os.setsid`` drops the controlling terminal entirely, so ``/dev/tty`` opens
    fail with ``ENXIO`` (exactly as under CI / an agent harness) and ``SIGTTIN``
    can never fire.  This makes the suite behave identically regardless of how
    it is launched and removes a whole class of concurrency-sensitive hangs.

    ``setsid`` fails for a process-group leader; the project always runs tests
    via ``uv run`` (so pytest is a child, never the leader), but we degrade
    gracefully if that ever changes.
    """
    global CONTROLLING_TTY_DETACHED
    try:
        os.setsid()
    except OSError:
        # Already a session/group leader — leave the disposition unchanged.
        return
    CONTROLLING_TTY_DETACHED = True


def pytest_configure(config: pytest.Config) -> None:
    """Detach test-running processes from the controlling terminal.

    Runs once per process.  Under ``-n auto`` only the xdist *workers* execute
    tests, so the controller stays attached (it owns terminal reporting); each
    worker detaches.  Without xdist the single process runs tests and detaches.
    """
    is_xdist_worker = hasattr(config, "workerinput")
    xdist_active = getattr(config.option, "dist", "no") != "no"
    if is_xdist_worker or not xdist_active:
        _detach_from_controlling_terminal()


@pytest.fixture()
def default_sigint() -> Generator[None, None, None]:
    """Ensure SIGINT uses Python's default handler for the duration of a test.

    When the test suite is launched as a background process (e.g. by a CI runner
    or agent harness, or with a trailing ``&``), the shell sets SIGINT to
    ``SIG_IGN`` and child processes — pytest-xdist workers and any subprocess
    they spawn — inherit it.  Interrupt tests must restore the default
    disposition so SIGINT is actually delivered as ``KeyboardInterrupt`` and so
    spawned ``agm`` processes reset to the default handler on ``exec`` instead of
    silently ignoring the signal.
    """

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@pytest.fixture(autouse=True)
def reset_dry_run_state() -> Generator[None, None, None]:
    dry_run.set_enabled(False)
    yield
    dry_run.set_enabled(False)


@pytest.fixture(autouse=True)
def clear_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROJ_DIR", raising=False)
    monkeypatch.delenv("REPO_DIR", raising=False)


@pytest.fixture(autouse=True)
def clear_workspace_shell_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip agm workspace-shell control variables inherited from the host.

    When the suite is run from inside an agm workspace shell, the process
    inherits ``AGM_REAL_SHELL``/``AGM_WORKSPACE_SHELL`` and friends.  Any test
    that spawns the shell wrapper with ``{**os.environ}`` would then pick up the
    host's real shell instead of the one it set up, so the wrapper would launch
    e.g. zsh in place of the requested bash.  Remove them so tests behave the
    same whether or not the suite itself runs inside a workspace shell.
    """

    for name in list(os.environ):
        if name.startswith("AGM_"):
            monkeypatch.delenv(name, raising=False)


_REPO_STDLIB_ROOT = Path(__file__).resolve().parent.parent / "stdlib"


@pytest.fixture(autouse=True)
def pin_agm_stdlib_to_repo(
    clear_workspace_shell_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the AgL stdlib to the in-repo tree so tests never read installed files.

    ``clear_workspace_shell_env`` strips every ``AGM_*`` variable first; this
    runs afterwards and points ``AGM_STDLIB`` at ``<repo>/stdlib``.  Every
    ``agm exec``/``agm repl`` invocation — in-process or as a subprocess that
    inherits the environment — then resolves the standard library from the
    current source tree, independent of whatever ``~/.agm`` happens to contain.
    """
    monkeypatch.setenv("AGM_STDLIB", str(_REPO_STDLIB_ROOT))


@pytest.fixture()
def env(tmp_path: Path) -> dict[str, str]:
    """Environment dict with git identity and isolated HOME."""
    e = os.environ.copy()
    e["GIT_AUTHOR_NAME"] = "Test"
    e["GIT_AUTHOR_EMAIL"] = "test@test.com"
    e["GIT_COMMITTER_NAME"] = "Test"
    e["GIT_COMMITTER_EMAIL"] = "test@test.com"
    e["GIT_CONFIG_NOSYSTEM"] = "1"
    e["SHELL"] = shutil.which("bash") or "/bin/sh"
    e.pop("PROJ_DIR", None)
    e.pop("REPO_DIR", None)
    e.pop("TMUX", None)
    e.pop("TMUX_PANE", None)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    e["HOME"] = str(fake_home)
    return e
