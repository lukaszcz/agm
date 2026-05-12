"""Tests for agm.commands.config.update."""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.commands.config.update as config_update
from agm.commands.args import ConfigUpdateArgs


class TestCommitGeneratedConfigs:
    """Tests for _commit_generated_configs."""

    def test_does_nothing_when_config_git_root_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(config_update.git_helpers, "exact_repo_root", lambda *_a, **_kw: None)
        called: list[str] = []
        monkeypatch.setattr(config_update, "require_success", lambda *_a, **_kw: called.append("s"))

        # _config_git_root returns None when config_dir doesn't exist
        config_update._commit_generated_configs(project_dir, env={})

        assert called == []

    def test_does_nothing_when_no_config_toml_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)

        monkeypatch.setattr(
            config_update.git_helpers,
            "exact_repo_root",
            lambda p, env: config_dir,
        )
        monkeypatch.setattr(config_update, "rglob", lambda p, pattern: [])

        called: list[str] = []
        monkeypatch.setattr(config_update, "require_success", lambda *_a, **_kw: called.append("s"))

        config_update._commit_generated_configs(project_dir, env={})

        assert called == []

    def test_stages_and_commits_when_staged_changes_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        config_toml = config_dir / "config.toml"
        config_toml.write_text("[deps]\n", encoding="utf-8")

        monkeypatch.setattr(
            config_update.git_helpers, "exact_repo_root", lambda p, env: config_dir
        )
        monkeypatch.setattr(config_update, "rglob", lambda p, pattern: [config_toml])
        monkeypatch.setattr(
            config_update.git_helpers, "has_staged_changes", lambda *_a, **_kw: True
        )

        commands_run: list[list[str]] = []
        monkeypatch.setattr(
            config_update, "require_success", lambda cmd, env=None: commands_run.append(cmd)
        )

        config_update._commit_generated_configs(project_dir, env={})

        # Should have two require_success calls: git add + git commit
        assert len(commands_run) == 2
        assert "add" in commands_run[0]
        assert "commit" in commands_run[1]

    def test_skips_commit_when_no_staged_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        config_toml = config_dir / "config.toml"
        config_toml.write_text("", encoding="utf-8")

        monkeypatch.setattr(
            config_update.git_helpers, "exact_repo_root", lambda p, env: config_dir
        )
        monkeypatch.setattr(config_update, "rglob", lambda p, pattern: [config_toml])
        monkeypatch.setattr(
            config_update.git_helpers, "has_staged_changes", lambda *_a, **_kw: False
        )

        commands_run: list[list[str]] = []
        monkeypatch.setattr(
            config_update, "require_success", lambda cmd, env=None: commands_run.append(cmd)
        )

        config_update._commit_generated_configs(project_dir, env={})

        # Only git add, no commit
        assert len(commands_run) == 1
        assert "add" in commands_run[0]


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

        commit_calls: list[Path] = []
        monkeypatch.setattr(
            config_update,
            "_commit_generated_configs",
            lambda pd, env: commit_calls.append(pd),
        )

        config_update.run(ConfigUpdateArgs())

        assert update_calls == [project_dir]
        assert commit_calls == [project_dir]
