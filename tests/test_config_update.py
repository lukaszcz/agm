"""Tests for agm.commands.config.update."""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.commands.config.update as config_update
from agm.commands.args import ConfigUpdateArgs


class TestConfigGitRoot:
    """Tests for _config_git_root."""

    def test_returns_none_when_config_dir_does_not_exist(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nonexistent"
        env: dict[str, str] = {}

        result = config_update._config_git_root(config_dir, env=env)

        assert result is None

    def test_returns_none_when_git_command_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        monkeypatch.setattr(
            config_update, "run_capture", lambda cmd, env=None: (1, "", "not a git repo")
        )

        result = config_update._config_git_root(config_dir, env={})

        assert result is None

    def test_returns_none_when_git_root_differs_from_config_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_dir = tmp_path / "config" / "subdir"
        config_dir.mkdir(parents=True)
        parent = tmp_path / "config"

        def fake_run_capture(
            cmd: list[str], env: dict[str, str] | None = None
        ) -> tuple[int, str, str]:
            return 0, str(parent) + "\n", ""

        monkeypatch.setattr(config_update, "run_capture", fake_run_capture)

        result = config_update._config_git_root(config_dir, env={})

        assert result is None

    def test_returns_path_when_config_dir_is_git_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        def fake_run_capture(
            cmd: list[str], env: dict[str, str] | None = None
        ) -> tuple[int, str, str]:
            return 0, str(config_dir) + "\n", ""

        monkeypatch.setattr(config_update, "run_capture", fake_run_capture)

        result = config_update._config_git_root(config_dir, env={})

        assert result == config_dir


class TestHasStagedChanges:
    """Tests for _has_staged_changes."""

    def test_returns_false_when_no_staged_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run_capture(
            cmd: list[str], env: dict[str, str] | None = None
        ) -> tuple[int, str, str]:
            return 0, "", ""

        monkeypatch.setattr(config_update, "run_capture", fake_run_capture)

        result = config_update._has_staged_changes(tmp_path, [], env={})

        assert result is False

    def test_returns_true_when_staged_changes_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run_capture(
            cmd: list[str], env: dict[str, str] | None = None
        ) -> tuple[int, str, str]:
            return 1, "", ""

        monkeypatch.setattr(config_update, "run_capture", fake_run_capture)

        result = config_update._has_staged_changes(tmp_path, [], env={})

        assert result is True

    def test_exits_on_unexpected_returncode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run_capture(
            cmd: list[str], env: dict[str, str] | None = None
        ) -> tuple[int, str, str]:
            return 128, "", "fatal"

        def fake_exit_with_output(returncode: int, stdout: str, stderr: str) -> None:
            raise SystemExit(returncode)

        monkeypatch.setattr(config_update, "run_capture", fake_run_capture)
        monkeypatch.setattr(config_update, "exit_with_output", fake_exit_with_output)

        with pytest.raises(SystemExit):
            config_update._has_staged_changes(tmp_path, [], env={})


class TestCommitGeneratedConfigs:
    """Tests for _commit_generated_configs."""

    def test_does_nothing_when_config_git_root_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(config_update, "run_capture", lambda *_a, **_kw: (0, "", ""))
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
            config_update,
            "_config_git_root",
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
            config_update, "_config_git_root", lambda p, env: config_dir
        )
        monkeypatch.setattr(config_update, "rglob", lambda p, pattern: [config_toml])
        monkeypatch.setattr(config_update, "_has_staged_changes", lambda *_a, **_kw: True)

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
            config_update, "_config_git_root", lambda p, env: config_dir
        )
        monkeypatch.setattr(config_update, "rglob", lambda p, pattern: [config_toml])
        monkeypatch.setattr(config_update, "_has_staged_changes", lambda *_a, **_kw: False)

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