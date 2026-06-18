"""Tests for simple run() entry-point wrappers in commands subpackages."""

from __future__ import annotations

import stat
import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest

import agm.commands.config.copy as config_copy_cmd
import agm.commands.config.env as config_env_cmd
import agm.commands.dep.new as dep_new_cmd
import agm.commands.tmux.close as tmux_close_cmd
import agm.commands.tmux.layout as tmux_layout_cmd
import agm.commands.tmux.open as tmux_open_cmd
import agm.commands.workspace.setup as setup_cmd
import agm.commands.worktree.new as worktree_new_cmd
import agm.commands.worktree.remove as worktree_remove_cmd
import agm.tmux.layout as tmux_layout
from agm.commands.args import (
    ConfigCopyArgs,
    ConfigEnvArgs,
    DepNewArgs,
    TmuxCloseArgs,
    TmuxLayoutArgs,
    TmuxOpenArgs,
    WorktreeNewArgs,
    WorktreeRemoveArgs,
)
from agm.core import dry_run as dry_run_module

# ---------------------------------------------------------------------------
# agm.commands.tmux.close
# ---------------------------------------------------------------------------


class TestTmuxCloseRun:
    def test_closes_named_session_in_dry_run(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dry_run_module.set_enabled(True)

        tmux_close_cmd.run(TmuxCloseArgs(session_name="mysession"))

        out = capsys.readouterr().out
        assert "tmux kill-session -t mysession" in out
        assert "Closed session mysession" in out


# ---------------------------------------------------------------------------
# agm.commands.tmux.open
# ---------------------------------------------------------------------------


class TestTmuxOpenRun:
    def test_opens_detached_session_in_dry_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dry_run_module.set_enabled(True)
        monkeypatch.chdir(tmp_path)

        tmux_open_cmd.run(TmuxOpenArgs(detach=True, pane_count="4", session_name="s"))

        out = capsys.readouterr().out
        assert "tmux new-session -dP" in out
        assert "-s s" in out
        assert out.count("tmux split-window") == 3
        assert "Detached tmux session s created" in out


# ---------------------------------------------------------------------------
# agm.commands.tmux.layout
# ---------------------------------------------------------------------------


class TestTmuxLayoutRun:
    def test_applies_layout_to_resolved_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commands: list[list[str]] = []

        def fake_require_capture(cmd: list[str]) -> str:
            if "#{window_width}" in cmd:
                return "200\n"
            if "#{window_height}" in cmd:
                return "50\n"
            return "@1\n"

        monkeypatch.setattr(tmux_layout, "require_capture", fake_require_capture)
        monkeypatch.setattr(tmux_layout, "require_success", lambda cmd: commands.append(cmd))

        tmux_layout_cmd.run(TmuxLayoutArgs(pane_count="4", window_id="@1"))

        assert len(commands) == 1
        assert commands[0][:4] == ["tmux", "select-layout", "-t", "@1"]
        assert "200x50" in commands[0][-1]


# ---------------------------------------------------------------------------
# agm.commands.worktree.new
# ---------------------------------------------------------------------------


class TestWorktreeNewRun:
    def test_creates_git_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, env=env, check=True)
        (repo / "README.md").write_text("main\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=repo, env=env, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, env=env, check=True)
        monkeypatch.chdir(repo)

        worktrees_dir = tmp_path / "external-worktrees"
        worktree_new_cmd.run(
            WorktreeNewArgs(branch="feature", worktrees_dir=str(worktrees_dir))
        )

        assert (worktrees_dir / "feature" / "README.md").is_file()

    def test_commits_config_when_project_dir_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_worktree_path = tmp_path / "worktrees" / "feature"
        fake_project_dir = tmp_path / "project"
        monkeypatch.setattr(
            worktree_new_cmd,
            "ensure_worktree",
            lambda **kw: fake_worktree_path,
        )
        monkeypatch.setattr(
            worktree_new_cmd,
            "discover_current_project_dir",
            lambda *a, **kw: fake_project_dir,
        )
        commit_calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            worktree_new_cmd,
            "commit_config_dir_changes",
            lambda *a, **kw: commit_calls.append({"args": a, "kwargs": kw}),
        )
        monkeypatch.setattr(
            worktree_new_cmd,
            "project_config_dir",
            lambda pd: pd / "config",
        )
        worktree_new_cmd.run(WorktreeNewArgs(branch="feature", worktrees_dir=None))
        assert len(commit_calls) == 1

    def test_skips_config_commit_outside_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            worktree_new_cmd,
            "ensure_worktree",
            lambda **kw: tmp_path / "standalone-worktree",
        )
        monkeypatch.setattr(
            worktree_new_cmd, "discover_current_project_dir", lambda *a, **kw: None
        )

        def fail_commit(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("commit should not run outside an AGM project")

        monkeypatch.setattr(worktree_new_cmd, "commit_config_dir_changes", fail_commit)

        worktree_new_cmd.run(WorktreeNewArgs(branch="feature", worktrees_dir=None))


# ---------------------------------------------------------------------------
# agm.commands.worktree.remove
# ---------------------------------------------------------------------------


class TestWorktreeRemoveRun:
    def test_removes_git_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        worktree = tmp_path / "worktrees" / "feature"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, env=env, check=True)
        (repo / "README.md").write_text("main\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=repo, env=env, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, env=env, check=True)
        subprocess.run(
            ["git", "worktree", "add", "-b", "feature", str(worktree)],
            cwd=repo,
            env=env,
            check=True,
        )
        monkeypatch.chdir(repo)

        worktree_remove_cmd.run(WorktreeRemoveArgs(force=True, branch="feature"))

        assert not worktree.exists()
        branches = subprocess.run(
            ["git", "branch", "--list", "feature"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert branches.stdout == ""


# ---------------------------------------------------------------------------
# agm.commands.workspace.setup
# ---------------------------------------------------------------------------


class TestSetupRun:
    def test_runs_project_setup_script(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        project = tmp_path / "project"
        repo = project / "repo"
        config = project / "config"
        repo.mkdir(parents=True)
        config.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, env=env, check=True)
        (repo / "README.md").write_text("main\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=repo, env=env, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, env=env, check=True)
        marker = tmp_path / "setup-ran"
        setup_script = config / "setup.sh"
        setup_script.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
        setup_script.chmod(setup_script.stat().st_mode | stat.S_IEXEC)
        monkeypatch.chdir(project)

        setup_cmd.run()

        assert marker.is_file()


# ---------------------------------------------------------------------------
# agm.commands.config.copy
# ---------------------------------------------------------------------------


class TestConfigCopyRun:
    def test_copies_config_to_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        project = tmp_path / "project"
        repo = project / "repo"
        target = tmp_path / "target"
        repo.mkdir(parents=True)
        (project / "config").mkdir()
        target.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=repo, env=env, check=True)
        (project / "config" / ".env").write_text("CONFIG_KEY=value\n", encoding="utf-8")
        monkeypatch.chdir(project)

        config_copy_cmd.run(ConfigCopyArgs(config_command=None, dirname=str(target)))

        assert (target / ".env").read_text(encoding="utf-8") == "CONFIG_KEY=value\n"


# ---------------------------------------------------------------------------
# agm.commands.config.env
# ---------------------------------------------------------------------------


class TestConfigEnvRun:
    def test_run_prints_shell_statements(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        before = {"PATH": "/usr/bin", "OLD_VAR": "x"}
        after = {"PATH": "/usr/local/bin", "NEW_VAR": "y"}
        monkeypatch.setattr(config_env_cmd.os, "environ", dict(before))
        monkeypatch.setattr(
            config_env_cmd,
            "load_current_workspace_env",
            lambda env: after,
        )
        config_env_cmd.run(ConfigEnvArgs())
        out = capsys.readouterr().out
        assert "unset OLD_VAR" in out
        assert "export NEW_VAR=y" in out

    def test_run_empty_delta_produces_no_output(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env = {"PATH": "/bin"}
        monkeypatch.setattr(config_env_cmd.os, "environ", dict(env))
        monkeypatch.setattr(
            config_env_cmd,
            "load_current_workspace_env",
            lambda env: dict(env),
        )
        config_env_cmd.run(ConfigEnvArgs())
        out = capsys.readouterr().out
        assert out == ""


# ---------------------------------------------------------------------------
# agm.commands.dep.new — dry_run branch (cleanup skipped)
# ---------------------------------------------------------------------------


class TestDepNewDryRun:
    @pytest.fixture(autouse=True)
    def reset_dry_run(self) -> Generator[None, None, None]:
        original = dry_run_module.enabled()
        yield
        dry_run_module.set_enabled(original)

    def test_dry_run_propagates_system_exit_without_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dry_run_module.set_enabled(True)
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(dep_new_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new_cmd, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new_cmd, "exists", lambda p: False)
        monkeypatch.setattr(dep_new_cmd, "mkdir", lambda p, **kw: None)
        monkeypatch.setattr(
            dep_new_cmd.git_helpers, "default_branch_from_remote", lambda url: "main"
        )

        def fail_require_success(cmd: list[str]) -> None:
            raise SystemExit(1)

        monkeypatch.setattr(dep_new_cmd, "require_success", fail_require_success)

        rmdir_calls: list[Path] = []
        monkeypatch.setattr(dep_new_cmd, "rmdir", lambda p: rmdir_calls.append(p))

        with pytest.raises(SystemExit):
            dep_new_cmd.run(DepNewArgs(branch=None, repo_url="https://github.com/org/mylib"))

        # In dry_run mode, cleanup (rmdir) should NOT be called
        assert rmdir_calls == []

    def test_dep_new_run_with_branch_and_dep_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cover resolved_branch = args.branch with branch provided."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(dep_new_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new_cmd, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new_cmd, "exists", lambda p: False)
        monkeypatch.setattr(dep_new_cmd, "mkdir", lambda p, **kw: None)

        require_success_calls: list[list[str]] = []
        monkeypatch.setattr(
            dep_new_cmd, "require_success", lambda cmd: require_success_calls.append(cmd)
        )
        monkeypatch.setattr(dep_new_cmd, "update_dependency_config", lambda **kw: None)
        monkeypatch.setattr(dep_new_cmd, "current_config_branch", lambda pd: None)

        dep_new_cmd.run(DepNewArgs(branch="develop", repo_url="https://github.com/org/mylib"))

        assert len(require_success_calls) == 1
        assert "develop" in require_success_calls[0]

    def test_cleanup_handles_os_error_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError during rmdir cleanup is silenced."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(dep_new_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_new_cmd, "derive_dep_name", lambda url: "mylib")
        monkeypatch.setattr(dep_new_cmd, "exists", lambda p: False)
        monkeypatch.setattr(dep_new_cmd, "mkdir", lambda p, **kw: None)
        monkeypatch.setattr(
            dep_new_cmd.git_helpers, "default_branch_from_remote", lambda url: "main"
        )

        def fail_require_success(cmd: list[str]) -> None:
            raise SystemExit(1)

        monkeypatch.setattr(dep_new_cmd, "require_success", fail_require_success)

        def raise_os_error(p: Path) -> None:
            raise OSError("permission denied")

        monkeypatch.setattr(dep_new_cmd, "rmdir", raise_os_error)

        # Should still raise SystemExit, just silences the OSError
        with pytest.raises(SystemExit):
            dep_new_cmd.run(DepNewArgs(branch=None, repo_url="https://github.com/org/mylib"))
