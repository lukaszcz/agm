"""Shared test fixtures."""

from __future__ import annotations

import os
import shutil
import signal
from collections.abc import Generator
from pathlib import Path

import pytest

from agm.core import dry_run


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
