"""Tests for agm.commands.config.update."""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.commands.config.update as config_update
from agm.cli_support.args import ConfigUpdateArgs


class TestConfigUpdateRun:
    """Tests for the config update run() entrypoint."""

    def test_calls_update_all_and_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(config_update, "require_current_project_dir", lambda: project_dir)

        update_calls: list[Path] = []
        monkeypatch.setattr(
            config_update,
            "update_all_project_dependency_configs",
            lambda pd, env=None: update_calls.append(pd),
        )

        commit_calls: list[tuple[Path, str]] = []
        monkeypatch.setattr(
            config_update,
            "commit_config_dir_changes",
            lambda pd, msg, env=None: commit_calls.append((pd, msg)),
        )

        config_update.run(ConfigUpdateArgs())

        assert update_calls == [project_dir]
        assert commit_calls == [(project_dir, "chore: update config")]
