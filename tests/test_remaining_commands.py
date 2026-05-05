"""Tests for fetch, config.update, dep.new, project.setup, run pure functions, and worktree."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

import agm.commands.config.update as config_update
import agm.commands.dep.new as dep_new
import agm.commands.fetch as fetch_cmd
import agm.project.setup as project_setup
import agm.project.worktree as worktree_mod
from agm.commands.args import ConfigUpdateArgs, DepNewArgs
from agm.commands.run import (
    _normalize_systemd_limit,
    _systemd_run_prefix,
    _systemd_scope_name,
    normalize_run_command,
)
from agm.vcs.git import WorktreeInfo

# ===========================================================================
# agm.commands.fetch
# ===========================================================================


class TestFetchRepo:
    """Tests for the _fetch_repo helper."""

    def test_prints_dot_for_project_dir_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        fetched: list[Path] = []
        synced: list[Path] = []

        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(
            fetch_cmd, "sync_remote_tracking_branches", lambda p: synced.append(p)
        )

        fetch_cmd._fetch_repo(project_dir, project_dir)

        captured = capsys.readouterr()
        assert "Fetching ." in captured.out
        assert fetched == [project_dir]
        assert synced == [project_dir]

    def test_prints_relative_path_for_repo_subdir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd._fetch_repo(project_dir, repo_dir)

        captured = capsys.readouterr()
        assert "Fetching repo" in captured.out

    def test_prints_absolute_path_when_outside_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd._fetch_repo(project_dir, other_dir)

        captured = capsys.readouterr()
        assert str(other_dir) in captured.out

    def test_calls_fetch_prune_all_and_sync(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        fetched: list[Path] = []
        synced: list[Path] = []

        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(
            fetch_cmd, "sync_remote_tracking_branches", lambda p: synced.append(p)
        )

        fetch_cmd._fetch_repo(project_dir, repo_dir)

        assert fetched == [repo_dir]
        assert synced == [repo_dir]


class TestFetchRun:
    """Tests for the fetch run() entrypoint."""

    def test_exits_when_repo_dir_does_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(fetch_cmd, "is_dir", lambda p: False)
        monkeypatch.setattr(fetch_cmd, "is_git_repo", lambda p: False)

        with pytest.raises(SystemExit):
            fetch_cmd.run(object())

    def test_exits_when_repo_dir_is_not_git_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(fetch_cmd, "is_dir", lambda p: True)
        monkeypatch.setattr(fetch_cmd, "is_git_repo", lambda p: False)

        with pytest.raises(SystemExit):
            fetch_cmd.run(object())

    def test_fetches_main_repo_only_when_no_deps_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(fetch_cmd, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(fetch_cmd, "is_dir", lambda p: p == repo_dir)
        monkeypatch.setattr(fetch_cmd, "is_git_repo", lambda p: True)
        monkeypatch.setattr(fetch_cmd, "iterdir", lambda p: [])

        fetched: list[Path] = []
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd.run(object())

        assert repo_dir in fetched

    def test_fetches_deps_when_deps_dir_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        deps_dir = project_dir / "deps"
        dep1_dir = deps_dir / "libfoo"
        dep1_repo = dep1_dir / "main"

        def fake_is_dir(p: Path) -> bool:
            return p in {repo_dir, deps_dir, dep1_dir}

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(fetch_cmd, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(fetch_cmd, "project_deps_dir", lambda pd: deps_dir)
        monkeypatch.setattr(fetch_cmd, "is_dir", fake_is_dir)
        monkeypatch.setattr(fetch_cmd, "is_git_repo", lambda p: True)
        monkeypatch.setattr(fetch_cmd, "iterdir", lambda p: [dep1_dir])
        monkeypatch.setattr(fetch_cmd, "find_first_git_repo", lambda p: dep1_repo)

        fetched: list[Path] = []
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd.run(object())

        assert repo_dir in fetched
        assert dep1_repo in fetched

    def test_skips_non_directory_entries_in_deps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        deps_dir = project_dir / "deps"
        not_a_dir = deps_dir / "somefile"

        def fake_is_dir(p: Path) -> bool:
            return p in {repo_dir, deps_dir}

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(fetch_cmd, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(fetch_cmd, "project_deps_dir", lambda pd: deps_dir)
        monkeypatch.setattr(fetch_cmd, "is_dir", fake_is_dir)
        monkeypatch.setattr(fetch_cmd, "is_git_repo", lambda p: True)
        monkeypatch.setattr(fetch_cmd, "iterdir", lambda p: [not_a_dir])

        fetched: list[Path] = []
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd.run(object())

        # Only main repo was fetched; dep entry was skipped (not a dir)
        assert fetched == [repo_dir]


# ===========================================================================
# agm.commands.config.update
# ===========================================================================


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


# ===========================================================================
# agm.commands.dep.new
# ===========================================================================


class TestDepNewRun:
    """Tests for dep new run()."""

    def test_exits_when_dep_already_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        deps_dir = project_dir / "deps"
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir(parents=True)

        monkeypatch.setattr(dep_new, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new, "exists", lambda p: p == dep_dir)

        with pytest.raises(SystemExit):
            dep_new.run(DepNewArgs(branch=None, repo_url="https://github.com/org/mylib"))

    def test_clones_with_provided_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"

        monkeypatch.setattr(dep_new, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new, "exists", lambda p: False)
        monkeypatch.setattr(dep_new, "mkdir", lambda p, parents=False, exist_ok=False: None)

        require_success_calls: list[list[str]] = []
        monkeypatch.setattr(
            dep_new, "require_success", lambda cmd: require_success_calls.append(cmd)
        )

        update_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            dep_new,
            "update_dependency_config",
            lambda *, project_dir, dep_name, dep_branch, config_branch: update_calls.append(
                {"dep_name": dep_name, "dep_branch": dep_branch, "config_branch": config_branch}
            ),
        )
        monkeypatch.setattr(dep_new, "current_config_branch", lambda pd: None)

        dep_new.run(DepNewArgs(branch="main", repo_url="https://github.com/org/mylib"))

        assert len(require_success_calls) == 1
        clone_cmd = require_success_calls[0]
        assert "clone" in clone_cmd
        assert "--branch" in clone_cmd
        assert "main" in clone_cmd
        assert len(update_calls) == 1
        assert update_calls[0]["dep_branch"] == "main"

    def test_resolves_default_branch_when_none_provided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"

        monkeypatch.setattr(dep_new, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new, "exists", lambda p: False)
        monkeypatch.setattr(dep_new, "mkdir", lambda p, parents=False, exist_ok=False: None)
        monkeypatch.setattr(dep_new, "default_branch_from_remote", lambda url: "develop")
        monkeypatch.setattr(dep_new, "require_success", lambda cmd: None)
        monkeypatch.setattr(
            dep_new,
            "update_dependency_config",
            lambda *, project_dir, dep_name, dep_branch, config_branch: None,
        )
        monkeypatch.setattr(dep_new, "current_config_branch", lambda pd: None)

        # No branch provided — should call default_branch_from_remote
        dep_new.run(DepNewArgs(branch=None, repo_url="https://github.com/org/mylib"))

    def test_cleans_up_dep_dir_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"

        monkeypatch.setattr(dep_new, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new, "exists", lambda p: False)
        monkeypatch.setattr(dep_new, "mkdir", lambda p, parents=False, exist_ok=False: None)
        monkeypatch.setattr(dep_new, "default_branch_from_remote", lambda url: "main")

        def fail_require_success(cmd: list[str]) -> None:
            raise SystemExit(1)

        monkeypatch.setattr(dep_new, "require_success", fail_require_success)

        rmdir_calls: list[Path] = []
        monkeypatch.setattr(dep_new, "rmdir", lambda p: rmdir_calls.append(p))

        with pytest.raises(SystemExit):
            dep_new.run(DepNewArgs(branch=None, repo_url="https://github.com/org/mylib"))

        assert len(rmdir_calls) == 1

    def test_clones_to_correct_target_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        deps_dir = project_dir / "deps"

        monkeypatch.setattr(dep_new, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new, "exists", lambda p: False)

        mkdir_calls: list[Path] = []
        monkeypatch.setattr(
            dep_new, "mkdir", lambda p, parents=False, exist_ok=False: mkdir_calls.append(p)
        )

        require_success_calls: list[list[str]] = []
        monkeypatch.setattr(
            dep_new, "require_success", lambda cmd: require_success_calls.append(cmd)
        )
        monkeypatch.setattr(
            dep_new,
            "update_dependency_config",
            lambda *, project_dir, dep_name, dep_branch, config_branch: None,
        )
        monkeypatch.setattr(dep_new, "current_config_branch", lambda pd: None)

        dep_new.run(DepNewArgs(branch="feat", repo_url="https://github.com/org/mylib"))

        # mkdir should have been called for the dep parent directory
        assert any(p == deps_dir / "mylib" for p in mkdir_calls)
        # Clone target is deps/mylib/feat
        clone_cmd = require_success_calls[0]
        assert str(deps_dir / "mylib" / "feat") in clone_cmd


# ===========================================================================
# agm.project.setup – run_setup
# ===========================================================================


class TestRunSetup:
    """Tests for project.setup.run_setup."""

    def _make_project(self, tmp_path: Path) -> tuple[Path, Path]:
        """Return (project_dir, repo_dir) with minimal workspace layout."""
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)
        (project_dir / "config").mkdir()
        return project_dir, repo_dir

    def test_prints_message_when_no_setup_scripts_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir, repo_dir = self._make_project(tmp_path)

        monkeypatch.setattr(
            project_setup, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(project_setup, "current_checkout", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(
            project_setup.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            project_setup,
            "load_worktree_env",
            lambda pd, branch, checkout_dir, env=None: dict(os.environ),
        )

        project_setup.run_setup(cwd=project_dir)

        captured = capsys.readouterr()
        assert "No setup scripts found" in captured.out

    def test_runs_executable_setup_sh_in_config_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir, repo_dir = self._make_project(tmp_path)
        config_dir = project_dir / "config"
        setup_script = config_dir / "setup.sh"
        setup_script.write_text("#!/bin/sh\n", encoding="utf-8")
        setup_script.chmod(setup_script.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(
            project_setup, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(project_setup, "current_checkout", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(
            project_setup.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            project_setup,
            "load_worktree_env",
            lambda pd, branch, checkout_dir, env=None: dict(os.environ),
        )

        run_calls: list[list[str]] = []
        monkeypatch.setattr(
            project_setup, "require_success", lambda cmd, cwd=None, env=None: run_calls.append(cmd)
        )

        project_setup.run_setup(cwd=project_dir)

        assert len(run_calls) == 1
        assert run_calls[0] == ["bash", str(setup_script)]
        captured = capsys.readouterr()
        assert "Running setup for" in captured.out
        assert "Setup complete for" in captured.out

    def test_skips_non_executable_setup_sh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir, repo_dir = self._make_project(tmp_path)
        config_dir = project_dir / "config"
        setup_script = config_dir / "setup.sh"
        # Write the file but do NOT make it executable
        setup_script.write_text("#!/bin/sh\n", encoding="utf-8")
        # Remove executable bit explicitly
        setup_script.chmod(0o644)

        monkeypatch.setattr(
            project_setup, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(project_setup, "current_checkout", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(
            project_setup.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            project_setup,
            "load_worktree_env",
            lambda pd, branch, checkout_dir, env=None: dict(os.environ),
        )

        project_setup.run_setup(cwd=project_dir)

        captured = capsys.readouterr()
        assert "No setup scripts found" in captured.out

    def test_runs_all_found_setup_scripts_in_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, repo_dir = self._make_project(tmp_path)
        config_dir = project_dir / "config"

        # Create two setup scripts: one in config_dir, one in checkout_dir
        config_script = config_dir / "setup.sh"
        config_script.write_text("#!/bin/sh\n", encoding="utf-8")
        config_script.chmod(config_script.stat().st_mode | stat.S_IEXEC)

        checkout_script = repo_dir / ".setup.sh"
        checkout_script.write_text("#!/bin/sh\n", encoding="utf-8")
        checkout_script.chmod(checkout_script.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(
            project_setup, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(project_setup, "current_checkout", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(
            project_setup.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            project_setup,
            "load_worktree_env",
            lambda pd, branch, checkout_dir, env=None: dict(os.environ),
        )

        run_calls: list[list[str]] = []
        monkeypatch.setattr(
            project_setup, "require_success", lambda cmd, cwd=None, env=None: run_calls.append(cmd)
        )

        project_setup.run_setup(cwd=project_dir)

        assert len(run_calls) == 2

    def test_dry_run_prints_operation_instead_of_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir, repo_dir = self._make_project(tmp_path)
        config_dir = project_dir / "config"
        setup_script = config_dir / "setup.sh"
        setup_script.write_text("#!/bin/sh\n", encoding="utf-8")
        setup_script.chmod(setup_script.stat().st_mode | stat.S_IEXEC)

        monkeypatch.setattr(
            project_setup, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(project_setup, "current_checkout", lambda pd, cwd=None, env=None: None)
        monkeypatch.setattr(
            project_setup.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            project_setup,
            "load_worktree_env",
            lambda pd, branch, checkout_dir, env=None: dict(os.environ),
        )
        monkeypatch.setattr(project_setup.dry_run, "enabled", lambda: True)

        dry_run_calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            project_setup.dry_run,
            "print_operation",
            lambda name, detail: dry_run_calls.append((name, detail)),
        )

        run_calls: list[list[str]] = []
        monkeypatch.setattr(
            project_setup, "require_success", lambda cmd, cwd=None, env=None: run_calls.append(cmd)
        )

        project_setup.run_setup(cwd=project_dir)

        assert len(dry_run_calls) == 1
        assert dry_run_calls[0][0] == "run-setup"
        # require_success is still called in dry_run mode (dry_run just prints first)
        assert len(run_calls) == 1


# ===========================================================================
# agm.commands.run – pure functions
# ===========================================================================


class TestNormalizeRunCommand:
    """Tests for normalize_run_command."""

    def test_strips_leading_double_dash(self) -> None:
        assert normalize_run_command(["--", "echo", "hi"]) == ["echo", "hi"]

    def test_leaves_command_unchanged_without_leading_double_dash(self) -> None:
        assert normalize_run_command(["echo", "hi"]) == ["echo", "hi"]

    def test_empty_list_returns_empty(self) -> None:
        assert normalize_run_command([]) == []

    def test_only_double_dash_returns_empty(self) -> None:
        assert normalize_run_command(["--"]) == []

    def test_double_dash_not_at_start_is_left_alone(self) -> None:
        assert normalize_run_command(["echo", "--", "arg"]) == ["echo", "--", "arg"]

    def test_multiple_double_dashes_strips_only_first(self) -> None:
        assert normalize_run_command(["--", "--", "echo"]) == ["--", "echo"]


class TestNormalizeSystemdLimit:
    """Tests for _normalize_systemd_limit."""

    def test_unlimited_returns_infinity(self) -> None:
        assert _normalize_systemd_limit("unlimited") == "infinity"

    def test_unlimited_case_insensitive(self) -> None:
        assert _normalize_systemd_limit("UNLIMITED") == "infinity"
        assert _normalize_systemd_limit("Unlimited") == "infinity"

    def test_unlimited_with_whitespace(self) -> None:
        assert _normalize_systemd_limit("  unlimited  ") == "infinity"

    def test_other_values_unchanged(self) -> None:
        assert _normalize_systemd_limit("20G") == "20G"
        assert _normalize_systemd_limit("0") == "0"
        assert _normalize_systemd_limit("infinity") == "infinity"


class TestSystemdRunPrefix:
    """Tests for _systemd_run_prefix."""

    def test_includes_base_flags(self) -> None:
        prefix = _systemd_run_prefix(memory_limit=None, swap_limit=None)
        assert "systemd-run" in prefix
        assert "--user" in prefix
        assert "--scope" in prefix
        assert "-q" in prefix
        assert "-p" in prefix
        assert "Delegate=yes" in prefix

    def test_adds_memory_max_when_memory_limit_set(self) -> None:
        prefix = _systemd_run_prefix(memory_limit="10G", swap_limit=None)
        assert "MemoryMax=10G" in prefix

    def test_adds_swap_max_when_swap_limit_set(self) -> None:
        prefix = _systemd_run_prefix(memory_limit=None, swap_limit="2G")
        assert "MemorySwapMax=2G" in prefix

    def test_normalizes_unlimited_to_infinity(self) -> None:
        prefix = _systemd_run_prefix(memory_limit="unlimited", swap_limit=None)
        assert "MemoryMax=infinity" in prefix

    def test_omits_memory_max_when_none(self) -> None:
        prefix = _systemd_run_prefix(memory_limit=None, swap_limit="1G")
        assert not any("MemoryMax" in item for item in prefix)

    def test_omits_swap_max_when_none(self) -> None:
        prefix = _systemd_run_prefix(memory_limit="10G", swap_limit=None)
        assert not any("MemorySwapMax" in item for item in prefix)

    def test_includes_both_limits(self) -> None:
        prefix = _systemd_run_prefix(memory_limit="8G", swap_limit="4G")
        assert "MemoryMax=8G" in prefix
        assert "MemorySwapMax=4G" in prefix


class TestSystemdScopeName:
    """Tests for _systemd_scope_name."""

    def test_returns_agm_run_prefix(self) -> None:
        name = _systemd_scope_name()
        assert name.startswith("agm-run-")

    def test_ends_with_scope_suffix(self) -> None:
        name = _systemd_scope_name()
        assert name.endswith(".scope")

    def test_names_are_unique(self) -> None:
        names = {_systemd_scope_name() for _ in range(20)}
        assert len(names) == 20

    def test_hex_part_is_32_chars(self) -> None:
        name = _systemd_scope_name()
        # format: agm-run-{32 hex chars}.scope
        hex_part = name.removeprefix("agm-run-").removesuffix(".scope")
        assert len(hex_part) == 32
        assert all(c in "0123456789abcdef" for c in hex_part)


# ===========================================================================
# agm.project.worktree
# ===========================================================================


class TestSyncRemoteTrackingBranches:
    """Tests for sync_remote_tracking_branches."""

    def test_creates_tracking_branch_for_unmerged_remote(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "default_remote_branch_ref",
            lambda p, env=None: "origin/main",
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "remote_unmerged_branches",
            lambda p, base_ref, env=None: ["origin/feature"],
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers, "local_branch_exists", lambda p, b, env=None: False
        )

        created: list[tuple[str, str]] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "create_tracking_branch",
            lambda p, local, remote, env=None: created.append((local, remote)),
        )

        worktree_mod.sync_remote_tracking_branches(repo_dir)

        assert created == [("feature", "origin/feature")]

    def test_skips_branch_when_local_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "default_remote_branch_ref",
            lambda p, env=None: "origin/main",
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "remote_unmerged_branches",
            lambda p, base_ref, env=None: ["origin/feature"],
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers, "local_branch_exists", lambda p, b, env=None: True
        )

        created: list[tuple[str, str]] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "create_tracking_branch",
            lambda p, local, remote, env=None: created.append((local, remote)),
        )

        worktree_mod.sync_remote_tracking_branches(repo_dir)

        assert created == []

    def test_skips_origin_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "default_remote_branch_ref",
            lambda p, env=None: "origin/main",
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "remote_unmerged_branches",
            lambda p, base_ref, env=None: ["origin/HEAD", "origin/feature"],
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers, "local_branch_exists", lambda p, b, env=None: False
        )

        created: list[tuple[str, str]] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "create_tracking_branch",
            lambda p, local, remote, env=None: created.append((local, remote)),
        )

        worktree_mod.sync_remote_tracking_branches(repo_dir)

        # origin/HEAD is skipped; only feature is created
        assert created == [("feature", "origin/feature")]

    def test_handles_multiple_unmerged_branches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "default_remote_branch_ref",
            lambda p, env=None: "origin/main",
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "remote_unmerged_branches",
            lambda p, base_ref, env=None: ["origin/feat-a", "origin/feat-b"],
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers, "local_branch_exists", lambda p, b, env=None: False
        )

        created: list[tuple[str, str]] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "create_tracking_branch",
            lambda p, local, remote, env=None: created.append((local, remote)),
        )

        worktree_mod.sync_remote_tracking_branches(repo_dir)

        assert ("feat-a", "origin/feat-a") in created
        assert ("feat-b", "origin/feat-b") in created


class TestBranchSync:
    """Tests for branch_sync."""

    def test_fetches_prune_and_syncs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setattr(worktree_mod.git_helpers, "checkout_root", lambda cwd=None: repo_dir)

        fetched: list[Path] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "fetch_prune_origin",
            lambda p, env=None: fetched.append(p),
        )

        synced: list[Path] = []
        monkeypatch.setattr(
            worktree_mod,
            "sync_remote_tracking_branches",
            lambda p, env=None: synced.append(p),
        )

        worktree_mod.branch_sync(cwd=tmp_path)

        assert fetched == [repo_dir]
        assert synced == [repo_dir]


class TestEnsureWorktree:
    """Tests for ensure_worktree."""

    def _setup_mocks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        project_dir: Path,
        repo_dir: Path,
        *,
        repo_branch: str = "main",
        existing_worktrees: list[WorktreeInfo] | None = None,
    ) -> list[dict[str, object]]:
        """Patch common dependencies; return list to accumulate worktree_add calls."""
        if existing_worktrees is None:
            existing_worktrees = []
        monkeypatch.setattr(
            worktree_mod.git_helpers, "checkout_root", lambda cwd=None: repo_dir
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers, "current_branch", lambda p, env=None: repo_branch
        )
        monkeypatch.setattr(
            worktree_mod, "current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(worktree_mod.git_helpers, "fetch", lambda p, env=None: None)
        monkeypatch.setattr(
            worktree_mod.git_helpers, "worktree_list", lambda p, env=None: existing_worktrees
        )
        monkeypatch.setattr(
            worktree_mod,
            "ensure_dependency_configs_for_branch",
            lambda *, project_dir, branch: None,
        )
        monkeypatch.setattr(
            worktree_mod,
            "copy_config",
            lambda *, project_dir=None, target, branch=None, cwd=None: None,
        )

        add_calls: list[dict[str, object]] = []

        def fake_worktree_add(
            repo: Path,
            path: Path,
            branch: str,
            *,
            create: bool = False,
            env: dict[str, str] | None = None,
        ) -> None:
            add_calls.append({"repo": repo, "path": path, "branch": branch, "create": create})

        monkeypatch.setattr(worktree_mod.git_helpers, "worktree_add", fake_worktree_add)
        return add_calls

    def test_creates_new_branch_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        project_dir.mkdir()
        repo_dir.mkdir()

        add_calls = self._setup_mocks(monkeypatch, project_dir, repo_dir)

        result = worktree_mod.ensure_worktree(
            new_branch="feat",
            worktrees_dir=None,
            branch=None,
            cwd=project_dir,
        )

        assert len(add_calls) == 1
        assert add_calls[0]["branch"] == "feat"
        assert add_calls[0]["create"] is True
        assert result.name == "feat"

    def test_checks_out_existing_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        project_dir.mkdir()
        repo_dir.mkdir()

        add_calls = self._setup_mocks(monkeypatch, project_dir, repo_dir)

        result = worktree_mod.ensure_worktree(
            new_branch=None,
            worktrees_dir=None,
            branch="existing",
            cwd=project_dir,
        )

        assert len(add_calls) == 1
        assert add_calls[0]["branch"] == "existing"
        assert add_calls[0]["create"] is False
        assert result.name == "existing"

    def test_exits_without_branch_or_new_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        project_dir.mkdir()
        repo_dir.mkdir()

        self._setup_mocks(monkeypatch, project_dir, repo_dir)

        with pytest.raises(SystemExit):
            worktree_mod.ensure_worktree(
                new_branch=None,
                worktrees_dir=None,
                branch=None,
                cwd=project_dir,
            )

    def test_returns_existing_worktree_when_existing_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktrees_dir = project_dir / ".agm" / "worktrees"
        worktree_path = worktrees_dir / "feat"
        project_dir.mkdir()
        repo_dir.mkdir()
        worktree_path.mkdir(parents=True)

        existing = [WorktreeInfo(path=worktree_path, branch="feat")]
        add_calls = self._setup_mocks(
            monkeypatch, project_dir, repo_dir, existing_worktrees=existing
        )

        result = worktree_mod.ensure_worktree(
            new_branch=None,
            worktrees_dir=None,
            branch="feat",
            existing_ok=True,
            cwd=project_dir,
        )

        assert add_calls == []
        assert result == worktree_path

    def test_exits_when_worktree_exists_and_not_existing_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktrees_dir = project_dir / ".agm" / "worktrees"
        worktree_path = worktrees_dir / "feat"
        project_dir.mkdir()
        repo_dir.mkdir()
        worktree_path.mkdir(parents=True)

        existing = [WorktreeInfo(path=worktree_path, branch="feat")]
        self._setup_mocks(monkeypatch, project_dir, repo_dir, existing_worktrees=existing)

        with pytest.raises(SystemExit):
            worktree_mod.ensure_worktree(
                new_branch=None,
                worktrees_dir=None,
                branch="feat",
                existing_ok=False,
                cwd=project_dir,
            )

    def test_reuse_existing_branch_switches_to_checkout_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        project_dir.mkdir()
        repo_dir.mkdir()

        add_calls = self._setup_mocks(monkeypatch, project_dir, repo_dir)

        # Simulate that the branch exists locally
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "local_branch_exists",
            lambda p, b, env=None: True,
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "remote_branch_exists",
            lambda p, b, env=None: False,
        )

        worktree_mod.ensure_worktree(
            new_branch="feat",
            worktrees_dir=None,
            branch=None,
            reuse_existing_branch=True,
            cwd=project_dir,
        )

        assert len(add_calls) == 1
        # create should be False because branch already exists
        assert add_calls[0]["create"] is False


class TestRemoveWorktree:
    """Tests for remove_worktree (thin wrapper)."""

    def test_delegates_to_remove_worktree_from_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setattr(worktree_mod.git_helpers, "checkout_root", lambda cwd=None: repo_dir)

        from_repo_calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            worktree_mod,
            "remove_worktree_from_repo",
            lambda *, repo_dir, force, branch, env=None: from_repo_calls.append(
                {"repo_dir": repo_dir, "force": force, "branch": branch}
            ),
        )

        worktree_mod.remove_worktree(force=True, branch="feat", cwd=tmp_path)

        assert from_repo_calls == [{"repo_dir": repo_dir, "force": True, "branch": "feat"}]


class TestRemoveWorktreeFromRepo:
    """Tests for remove_worktree_from_repo."""

    def test_removes_worktree_and_deletes_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktree_path = project_dir / ".agm" / "worktrees" / "feat"
        project_dir.mkdir()
        repo_dir.mkdir()
        worktree_path.mkdir(parents=True)

        monkeypatch.setattr(worktree_mod, "current_project_dir", lambda cwd=None: project_dir)
        monkeypatch.setattr(
            worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "worktree_list",
            lambda p, env=None: [WorktreeInfo(path=worktree_path, branch="feat")],
        )

        removed: list[Path] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "worktree_remove",
            lambda p, path, force=False, env=None: removed.append(path),
        )

        deleted: list[str] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "branch_delete",
            lambda p, b, env=None: deleted.append(b),
        )

        worktree_mod.remove_worktree_from_repo(
            repo_dir=repo_dir, force=False, branch="feat"
        )

        assert removed == [worktree_path]
        assert deleted == ["feat"]

    def test_exits_when_worktree_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        project_dir.mkdir()
        repo_dir.mkdir()

        monkeypatch.setattr(worktree_mod, "current_project_dir", lambda cwd=None: project_dir)
        monkeypatch.setattr(
            worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers, "worktree_list", lambda p, env=None: []
        )

        require_calls: list[list[str]] = []
        monkeypatch.setattr(
            worktree_mod,
            "require_success",
            lambda cmd, env=None: require_calls.append(cmd),
        )

        with pytest.raises(SystemExit):
            worktree_mod.remove_worktree_from_repo(
                repo_dir=repo_dir, force=False, branch="nonexistent"
            )

    def test_exits_when_branch_is_main_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        project_dir.mkdir()
        repo_dir.mkdir()

        monkeypatch.setattr(worktree_mod, "current_project_dir", lambda cwd=None: project_dir)
        # current branch is "main" — trying to remove "main" should exit
        monkeypatch.setattr(
            worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main"
        )

        with pytest.raises(SystemExit):
            worktree_mod.remove_worktree_from_repo(
                repo_dir=repo_dir, force=False, branch="main"
            )

    def test_passes_force_flag_to_worktree_remove(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktree_path = project_dir / ".agm" / "worktrees" / "feat"
        project_dir.mkdir()
        repo_dir.mkdir()
        worktree_path.mkdir(parents=True)

        monkeypatch.setattr(worktree_mod, "current_project_dir", lambda cwd=None: project_dir)
        monkeypatch.setattr(
            worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "worktree_list",
            lambda p, env=None: [WorktreeInfo(path=worktree_path, branch="feat")],
        )

        force_values: list[bool] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "worktree_remove",
            lambda p, path, force=False, env=None: force_values.append(force),
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers, "branch_delete", lambda p, b, env=None: None
        )

        worktree_mod.remove_worktree_from_repo(
            repo_dir=repo_dir, force=True, branch="feat"
        )

        assert force_values == [True]
