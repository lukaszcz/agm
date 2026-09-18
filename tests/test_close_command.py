"""Tests for agm.commands.workspace.close."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import agm.commands.workspace.close as close_module
from agm.cli_support.args import CloseArgs
from agm.commands.workspace.close import close_workspace
from tests._git_helpers import add_linked_worktree, git_output, git_run, init_repo


def _make_git_close_project(
    tmp_path: Path,
    env: dict[str, str],
    *,
    branch: str = "feature",
    unmerged: bool = False,
    dirty: bool = False,
) -> tuple[Path, Path, Path]:
    project_dir = tmp_path / "proj"
    repo_dir = project_dir / "repo"
    worktree_dir = project_dir / "worktrees" / branch
    (project_dir / "config").mkdir(parents=True)
    init_repo(repo_dir, env)
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(worktree_dir)],
        cwd=repo_dir,
        env=env,
        check=True,
    )
    if unmerged:
        (worktree_dir / "feature.txt").write_text("feature\n", encoding="utf-8")
        subprocess.run(["git", "add", "feature.txt"], cwd=worktree_dir, env=env, check=True)
        subprocess.run(
            ["git", "commit", "-m", "feature"],
            cwd=worktree_dir,
            env=env,
            check=True,
        )
    if dirty:
        (worktree_dir / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    return project_dir, repo_dir, worktree_dir


def _branch_exists(repo_dir: Path, branch: str, env: dict[str, str]) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=repo_dir,
        env=env,
        check=False,
    )
    return result.returncode == 0


def _install_fake_tmux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, path: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "tmux.log"
    tmux = bin_dir / "tmux"
    tmux.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log_path}\nexit 0\n",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{path}")
    return log_path


# ===========================================================================
# close_workspace workspace config removal
# ===========================================================================


class TestCloseSessionRemovesWorkspaceConfig:
    @pytest.mark.parametrize("config_kind", ["missing", "file", "directory"])
    def test_close_updates_versioned_config_without_touching_other_workspaces(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
        config_kind: str,
    ) -> None:
        project, _, worktree = _make_git_close_project(tmp_path, env)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])
        config_repo = init_repo(project / "config", env)
        target = config_repo / "feature"
        if config_kind == "file":
            target.write_text("settings")
        elif config_kind == "directory":
            target.mkdir()
            (target / "config.toml").write_text("[run]\ntimeout = 5\n")
        sibling = config_repo / "other"
        sibling.mkdir()
        (sibling / "config.toml").write_text("[run]\ntimeout = 7\n")
        subprocess.run(["git", "add", "."], cwd=config_repo, env=env, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "workspace configs"], cwd=config_repo, env=env, check=True
        )
        previous_head = git_output(config_repo, ["rev-parse", "HEAD"], env)

        close_workspace(branch="feature", cwd=project)

        assert not target.exists()
        assert not worktree.exists()
        assert (sibling / "config.toml").read_text() == "[run]\ntimeout = 7\n"
        assert git_output(config_repo, ["status", "--porcelain"], env) == ""
        head = git_output(config_repo, ["rev-parse", "HEAD"], env)
        assert (head == previous_head) is (config_kind == "missing")


class TestCloseSession:
    def test_removes_worktree_branch_and_closes_session(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir, repo_dir, worktree_dir = _make_git_close_project(tmp_path, env)
        tmux_log = _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])

        close_workspace(branch="feature", cwd=project_dir)

        assert not worktree_dir.exists()
        assert not _branch_exists(repo_dir, "feature", env)
        assert "kill-session -t =proj/feature" in tmux_log.read_text(encoding="utf-8")

    def test_closes_sanitized_tmux_session_for_dotted_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        branch = "rocq-9.2"
        project_dir, repo_dir, worktree_dir = _make_git_close_project(tmp_path, env, branch=branch)
        tmux_log = _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])

        close_workspace(branch=branch, cwd=project_dir)

        assert not worktree_dir.exists()
        assert not _branch_exists(repo_dir, branch, env)
        assert "kill-session -t =proj/rocq-9_2" in tmux_log.read_text(encoding="utf-8")

    @pytest.mark.parametrize("branch", ["main", "missing"])
    def test_invalid_close_preserves_the_main_checkout(
        self, tmp_path: Path, env: dict[str, str], branch: str
    ) -> None:
        project, repo, worktree = _make_git_close_project(tmp_path, env)
        with pytest.raises(SystemExit) as raised:
            close_workspace(branch=branch, cwd=project)
        assert raised.value.code == 1
        assert (repo / "README.md").read_text() == "# test\n"
        assert worktree.is_dir()
        assert _branch_exists(repo, "main", env)
        assert _branch_exists(repo, "feature", env)

    def test_refuses_git_worktree_outside_the_workspaces(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project, repo, _ = _make_git_close_project(tmp_path, env)
        outside = add_linked_worktree(repo, tmp_path / "elsewhere", env, branch="outside")

        with pytest.raises(SystemExit) as raised:
            close_workspace(branch="outside", cwd=project)

        assert raised.value.code == 1
        assert outside.is_dir()
        assert _branch_exists(repo, "outside", env)

    def test_closes_workspace_whose_head_is_detached(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project, repo, worktree = _make_git_close_project(tmp_path, env)
        _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])
        git_run(worktree, ["checkout", "-q", "--detach"], env)

        close_workspace(branch="feature", cwd=project)

        assert not worktree.exists()
        assert not _branch_exists(repo, "feature", env)

    def test_keeps_workspace_whose_branch_no_longer_exists(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project, repo, worktree = _make_git_close_project(tmp_path, env)
        git_run(worktree, ["checkout", "-q", "-b", "other"], env)
        git_run(repo, ["branch", "-q", "-D", "feature"], env)

        with pytest.raises(SystemExit) as raised:
            close_workspace(branch="feature", cwd=project)

        assert raised.value.code == 1
        assert worktree.is_dir()

    def test_exits_without_removing_worktree_when_branch_not_deletable(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir, repo_dir, worktree_dir = _make_git_close_project(tmp_path, env, unmerged=True)

        with pytest.raises(SystemExit) as exc_info:
            close_workspace(branch="feature", cwd=project_dir)

        assert exc_info.value.code == 1
        assert worktree_dir.exists()
        assert _branch_exists(repo_dir, "feature", env)
        assert "not fully merged" in capsys.readouterr().err

    def test_force_delete_removes_unmerged_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir, repo_dir, worktree_dir = _make_git_close_project(tmp_path, env, unmerged=True)
        _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])

        close_workspace(branch="feature", force_delete=True, cwd=project_dir)

        assert not worktree_dir.exists()
        assert not _branch_exists(repo_dir, "feature", env)

    def test_force_removes_dirty_worktree_and_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir, repo_dir, worktree_dir = _make_git_close_project(tmp_path, env, dirty=True)
        _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])

        close_workspace(branch="feature", force=True, force_delete=False, cwd=project_dir)

        assert not worktree_dir.exists()
        assert not _branch_exists(repo_dir, "feature", env)

    def test_keep_branch_removes_worktree_without_deleting_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir, repo_dir, worktree_dir = _make_git_close_project(tmp_path, env, unmerged=True)
        _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])

        close_workspace(branch="feature", keep_branch=True, cwd=project_dir)

        assert not worktree_dir.exists()
        assert _branch_exists(repo_dir, "feature", env)

    def test_keep_workspace_keeps_worktree_branch_and_workspace_config(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir, repo_dir, worktree_dir = _make_git_close_project(tmp_path, env, unmerged=True)
        tmux_log = _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])
        workspace_config = project_dir / "config" / "feature"
        workspace_config.mkdir()

        close_workspace(branch="feature", keep_workspace=True, cwd=project_dir)

        assert worktree_dir.exists()
        assert workspace_config.exists()
        assert _branch_exists(repo_dir, "feature", env)
        assert "kill-session -t =proj/feature" in tmux_log.read_text(encoding="utf-8")


# ===========================================================================
# run (entry point)
# ===========================================================================


class TestCloseRun:
    @pytest.mark.parametrize(
        ("force", "force_delete", "keep_workspace", "kept"),
        [(True, False, False, False), (False, True, False, False), (False, False, True, True)],
        ids=["force", "force-delete", "keep-workspace"],
    )
    def test_unmerged_branch_policy_preserves_or_removes_workspace(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
        force: bool,
        force_delete: bool,
        keep_workspace: bool,
        kept: bool,
    ) -> None:
        project, repo, worktree = _make_git_close_project(tmp_path, env, unmerged=True)
        tmux_log = _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])
        config = project / "config" / "feature"
        config.mkdir()
        (config / "config.toml").write_text("[run]\ntimeout = 5\n", encoding="utf-8")
        monkeypatch.chdir(project)

        close_module.run(
            CloseArgs(
                branch="feature",
                force=force,
                force_delete=force_delete,
                keep_branch=False,
                keep_workspace=keep_workspace,
            )
        )

        assert worktree.exists() is kept
        assert config.exists() is kept
        assert _branch_exists(repo, "feature", env) is kept
        assert "kill-session -t =proj/feature" in tmux_log.read_text(encoding="utf-8")
        if kept:
            assert (worktree / "feature.txt").read_text(encoding="utf-8") == "feature\n"
            assert (config / "config.toml").read_text(encoding="utf-8") == "[run]\ntimeout = 5\n"

    def test_rejected_close_preserves_unmerged_work_and_config(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project, repo, worktree = _make_git_close_project(tmp_path, env, unmerged=True)
        tmux_log = _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])
        config = project / "config" / "feature"
        config.mkdir()
        monkeypatch.chdir(project)

        with pytest.raises(SystemExit) as raised:
            close_module.run(
                CloseArgs(
                    branch="feature",
                    force=False,
                    force_delete=False,
                    keep_branch=False,
                    keep_workspace=False,
                )
            )

        assert raised.value.code == 1
        assert (worktree / "feature.txt").read_text(encoding="utf-8") == "feature\n"
        assert config.is_dir()
        assert _branch_exists(repo, "feature", env)
        assert not tmux_log.exists()
