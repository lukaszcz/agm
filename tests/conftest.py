"""Shared test fixtures."""

from __future__ import annotations

import os
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest

from agm.core import dry_run


@pytest.fixture(autouse=True)
def reset_dry_run_state() -> Generator[None, None, None]:
    dry_run.set_enabled(False)
    yield
    dry_run.set_enabled(False)


@pytest.fixture(autouse=True)
def clear_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROJ_DIR", raising=False)
    monkeypatch.delenv("REPO_DIR", raising=False)


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
