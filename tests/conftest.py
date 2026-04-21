"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def env(tmp_path: Path) -> dict[str, str]:
    """Environment dict with git identity and isolated HOME."""
    e = os.environ.copy()
    e["GIT_AUTHOR_NAME"] = "Test"
    e["GIT_AUTHOR_EMAIL"] = "test@test.com"
    e["GIT_COMMITTER_NAME"] = "Test"
    e["GIT_COMMITTER_EMAIL"] = "test@test.com"
    e["GIT_CONFIG_NOSYSTEM"] = "1"
    e.pop("PROJ_DIR", None)
    e.pop("REPO_DIR", None)
    e.pop("TMUX", None)
    e.pop("TMUX_PANE", None)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    e["HOME"] = str(fake_home)
    return e
