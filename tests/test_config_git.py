"""Tests for agm.project.config_git."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agm.project.config_git as config_git
import agm.vcs.git as git_helpers
from agm.project.config_git import _add_paths, commit_config_dir_changes


class TestAddPaths:
    """Tests for _add_paths."""

    def test_stages_changes_with_git_add(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git_root = tmp_path / "config"
        git_root.mkdir()

        captured_cmds: list[list[str]] = []
        monkeypatch.setattr(
            config_git, "run_capture", lambda cmd, **kw: (captured_cmds.append(cmd) or (0, "", ""))
        )

        _add_paths(git_root, [git_root / "feature"], env={})

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]
        assert "add" in cmd
        assert "-A" in cmd

    def test_ignores_pathspec_did_not_match_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git_root = tmp_path / "config"
        git_root.mkdir()

        def fake_run_capture(cmd: list[str], **kw: Any) -> tuple[int, str, str]:
            if "add" in cmd:
                return 1, "", "pathspec 'feature' did not match any files"
            return 0, "", ""

        monkeypatch.setattr(config_git, "run_capture", fake_run_capture)

        # Should not raise
        _add_paths(git_root, [git_root / "feature"], env={})

    def test_reraises_unexpected_git_add_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git_root = tmp_path / "config"
        git_root.mkdir()

        run_capture_calls: list[list[str]] = []

        def fake_run_capture(cmd: list[str], **kw: Any) -> tuple[int, str, str]:
            run_capture_calls.append(cmd)
            if "add" in cmd:
                return 1, "", "some unexpected error"
            return 0, "", ""

        monkeypatch.setattr(config_git, "run_capture", fake_run_capture)

        with pytest.raises(SystemExit) as exc_info:
            _add_paths(git_root, [git_root / "feature"], env={})

        # The failure is surfaced without re-running the failing git add.
        assert exc_info.value.code == 1
        assert len(run_capture_calls) == 1


class TestCommitConfigDirChanges:
    """Tests for commit_config_dir_changes."""

    def test_does_nothing_when_config_dir_is_not_a_git_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)

        monkeypatch.setattr(git_helpers, "exact_repo_root", lambda path, env=None: None)

        commands_run: list[list[str]] = []
        monkeypatch.setattr(
            config_git, "require_success", lambda cmd, env=None: commands_run.append(cmd)
        )

        commit_config_dir_changes(project_dir, "chore: test", env={})

        assert commands_run == []

    def test_does_nothing_when_config_dir_does_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        commands_run: list[list[str]] = []
        monkeypatch.setattr(
            config_git, "require_success", lambda cmd, env=None: commands_run.append(cmd)
        )

        commit_config_dir_changes(project_dir, "chore: test", env={})

        assert commands_run == []

    def test_adds_all_and_commits_with_add_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        branch_config = config_dir / "feature"
        branch_config.mkdir()

        monkeypatch.setattr(
            git_helpers, "exact_repo_root", lambda path, env=None: config_dir
        )
        monkeypatch.setattr(
            config_git, "_add_paths", lambda root, paths, env=None: None
        )
        monkeypatch.setattr(
            git_helpers, "has_staged_changes", lambda repo_dir, paths, env=None: True
        )

        commands_run: list[tuple[list[str], dict[str, str] | None]] = []
        monkeypatch.setattr(
            config_git,
            "require_success",
            lambda cmd, env=None: commands_run.append((cmd, env)),
        )

        env = {"HOME": "/tmp"}
        commit_config_dir_changes(
            project_dir, "chore: add config for feature",
            add_paths=[branch_config], env=env,
        )

        # Should have one require_success call for the commit
        assert len(commands_run) == 1
        commit_cmd = commands_run[0]
        assert "commit" in commit_cmd[0]
        assert "chore: add config for feature" in commit_cmd[0]
        assert commit_cmd[1] == env

    def test_adds_tracked_changes_and_config_toml_when_no_add_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        config_toml = config_dir / "config.toml"
        config_toml.write_text("[deps]", encoding="utf-8")

        monkeypatch.setattr(
            git_helpers, "exact_repo_root", lambda path, env=None: config_dir
        )
        monkeypatch.setattr(config_git, "rglob", lambda p, pattern: [config_toml])
        monkeypatch.setattr(
            git_helpers, "has_staged_changes", lambda repo_dir, paths, env=None: True
        )

        commands_run: list[list[str]] = []
        monkeypatch.setattr(
            config_git, "require_success", lambda cmd, env=None: commands_run.append(cmd)
        )

        commit_config_dir_changes(project_dir, "chore: update config", env={})

        # git add -u, git add config.toml, git commit = 3 calls
        assert len(commands_run) == 3
        assert "add" in commands_run[0]
        assert "-u" in commands_run[0]
        assert "add" in commands_run[1]
        assert "config.toml" in " ".join(commands_run[1])
        assert "commit" in commands_run[2]

    def test_adds_tracked_changes_without_config_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)

        monkeypatch.setattr(
            git_helpers, "exact_repo_root", lambda path, env=None: config_dir
        )
        monkeypatch.setattr(config_git, "rglob", lambda p, pattern: [])
        monkeypatch.setattr(
            git_helpers, "has_staged_changes", lambda repo_dir, paths, env=None: True
        )

        commands_run: list[list[str]] = []
        monkeypatch.setattr(
            config_git, "require_success", lambda cmd, env=None: commands_run.append(cmd)
        )

        commit_config_dir_changes(project_dir, "chore: update config", env={})

        # git add -u, git commit = 2 calls (no config.toml found)
        assert len(commands_run) == 2
        assert "add" in commands_run[0]
        assert "-u" in commands_run[0]
        assert "commit" in commands_run[1]

    def test_skips_commit_when_no_staged_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        branch_config = config_dir / "feature"
        branch_config.mkdir()

        monkeypatch.setattr(
            git_helpers, "exact_repo_root", lambda path, env=None: config_dir
        )
        monkeypatch.setattr(
            config_git, "_add_paths", lambda root, paths, env=None: None
        )
        monkeypatch.setattr(
            git_helpers, "has_staged_changes", lambda repo_dir, paths, env=None: False
        )

        commands_run: list[list[str]] = []
        monkeypatch.setattr(
            config_git, "require_success", lambda cmd, env=None: commands_run.append(cmd)
        )

        commit_config_dir_changes(
            project_dir, "chore: test", add_paths=[branch_config], env={},
        )

        # No commit, _add_paths was called but no require_success calls
        assert commands_run == []

    def test_does_nothing_with_empty_add_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)

        monkeypatch.setattr(
            git_helpers, "exact_repo_root", lambda path, env=None: config_dir
        )

        commands_run: list[list[str]] = []
        monkeypatch.setattr(
            config_git, "require_success", lambda cmd, env=None: commands_run.append(cmd)
        )

        commit_config_dir_changes(project_dir, "chore: test", add_paths=[], env={})

        assert commands_run == []

    def test_uses_default_env_when_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        branch_config = config_dir / "feature"
        branch_config.mkdir()

        monkeypatch.setattr(
            git_helpers, "exact_repo_root", lambda path, env=None: config_dir
        )
        monkeypatch.setattr(
            config_git, "_add_paths", lambda root, paths, env=None: None
        )
        monkeypatch.setattr(
            git_helpers, "has_staged_changes", lambda repo_dir, paths, env=None: True
        )

        commands_run: list[tuple[list[str], dict[str, str] | None]] = []
        monkeypatch.setattr(
            config_git,
            "require_success",
            lambda cmd, env=None: commands_run.append((cmd, env)),
        )

        commit_config_dir_changes(
            project_dir, "chore: test", add_paths=[branch_config],
        )

        assert commands_run[0][1] is None

    def test_git_add_uses_correct_config_repo_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify git -C uses the config git root."""

        project_dir = tmp_path / "proj"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)

        monkeypatch.setattr(
            git_helpers, "exact_repo_root", lambda path, env=None: config_dir
        )
        monkeypatch.setattr(config_git, "rglob", lambda p, pattern: [])
        monkeypatch.setattr(
            git_helpers, "has_staged_changes", lambda repo_dir, paths, env=None: True
        )

        commands_run: list[list[str]] = []
        monkeypatch.setattr(
            config_git, "require_success", lambda cmd, env=None: commands_run.append(cmd)
        )

        commit_config_dir_changes(project_dir, "chore: test", env={})

        for cmd in commands_run:
            assert "-C" in cmd
            idx = cmd.index("-C")
            assert cmd[idx + 1] == str(config_dir)
