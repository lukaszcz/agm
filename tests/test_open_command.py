"""Tests for agm.commands.workspace.open."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import agm.commands.workspace.open as open_module
from agm.commands.args import OpenArgs
from agm.commands.workspace.open import (
    checkout_workspace,
    create_workspace,
    open_or_create_workspace,
    open_workspace,
    queue_setup_and_focus_workspace_session,
    validate_pane_count,
)
from agm.core import dry_run
from agm.tmux.session import create_tmux_session


def _make_git_project(tmp_path: Path, env: dict[str, str]) -> Path:
    project = tmp_path / "proj"
    repo = project / "repo"
    (project / "config").mkdir(parents=True)
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, env=env, check=True)
    (repo / "README.md").write_text("main\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, env=env, check=True)
    return project

# ===========================================================================
# validate_pane_count
# ===========================================================================


class TestValidatePaneCount:
    def test_valid_count_does_not_raise(self) -> None:
        validate_pane_count("4")  # should not raise

    def test_none_does_not_raise(self) -> None:
        validate_pane_count(None)

    def test_invalid_count_exits(self) -> None:
        with pytest.raises(SystemExit):
            validate_pane_count("bad")

    def test_zero_exits(self) -> None:
        with pytest.raises(SystemExit):
            validate_pane_count("0")

    def test_re_raises_when_exit_code_is_not_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cover line 35: raise when exc.code != 1."""
        # Patch validate_tmux_pane_count to raise SystemExit(2) — code != 1
        monkeypatch.setattr(
            open_module,
            "validate_tmux_pane_count",
            lambda cmd, pane_count: (_ for _ in ()).throw(SystemExit(2)),
        )
        with pytest.raises(SystemExit) as exc_info:
            validate_pane_count("whatever")
        assert exc_info.value.code == 2


# ===========================================================================
# queue_setup_and_focus_workspace_session
# ===========================================================================


class TestQueueSetupAndFocusSession:
    def test_detached_returns_without_focusing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dry_run.set_enabled(True)

        queue_setup_and_focus_workspace_session(
            detached=True,
            pane_count=None,
            session_name="s",
            repo_path=tmp_path,
            env={},
        )

        out = capsys.readouterr().out
        assert "tmux new-session -dP" in out
        assert "tmux send-keys -t s:0.0 'agm workspace setup' C-m" in out
        assert "tmux attach-session" not in out
        assert "tmux switch-client" not in out

    def test_not_detached_raises_system_exit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dry_run.set_enabled(True)

        with pytest.raises(SystemExit) as exc_info:
            queue_setup_and_focus_workspace_session(
                detached=False,
                pane_count=None,
                session_name="s",
                repo_path=tmp_path,
                env={},
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "tmux new-session -dP" in out
        assert "tmux send-keys -t s:0.0 'agm workspace setup' C-m" in out
        assert "tmux attach-session -t s" in out

    def test_raises_assertion_when_session_name_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            open_module, "create_tmux_session", lambda **kw: None
        )
        with pytest.raises(AssertionError):
            queue_setup_and_focus_workspace_session(
                detached=True,
                pane_count=None,
                session_name="s",
                repo_path=tmp_path,
                env={},
            )


# ===========================================================================
# open_workspace
# ===========================================================================


class TestOpenSession:
    def _base_setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Path, Path]:
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        repo_dir.mkdir(parents=True)
        monkeypatch.setattr(
            open_module, "require_current_project_dir", lambda cwd=None: proj_dir
        )
        monkeypatch.setattr(open_module, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(
            open_module.git_helpers, "current_branch", lambda rd, **kw: "main"
        )
        monkeypatch.setattr(
            open_module, "load_workspace_env", lambda pd, branch, workspace_dir: {}
        )
        monkeypatch.setattr(open_module, "create_tmux_session", lambda **kw: None)
        return proj_dir, repo_dir

    def test_opens_main_repo_session_when_branch_is_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        proj_dir, repo_dir = self._base_setup(tmp_path, monkeypatch)
        monkeypatch.setattr(open_module, "create_tmux_session", create_tmux_session)

        open_workspace(detached=True, pane_count=None, branch=None, cwd=tmp_path)

        out = capsys.readouterr().out
        assert "tmux new-session -dP" in out
        assert f"-c {repo_dir}" in out
        assert f"Detached tmux session {proj_dir.name} created" in out

    def test_exits_when_worktree_not_at_expected_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir, repo_dir = self._base_setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            open_module,
            "branch_worktree_path",
            lambda pd, branch, repo_branch: tmp_path / "worktrees" / branch,
        )
        monkeypatch.setattr(
            open_module, "has_expected_worktree", lambda pd, branch, **kw: False
        )
        with pytest.raises(SystemExit) as exc_info:
            open_workspace(detached=True, pane_count=None, branch="feature", cwd=tmp_path)
        assert exc_info.value.code == 1

    def test_opens_branch_session_when_worktree_exists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        proj_dir, repo_dir = self._base_setup(tmp_path, monkeypatch)
        monkeypatch.setattr(open_module, "create_tmux_session", create_tmux_session)
        feat_path = tmp_path / "worktrees" / "feature"
        feat_path.mkdir(parents=True)
        monkeypatch.setattr(
            open_module,
            "branch_worktree_path",
            lambda pd, branch, repo_branch: feat_path,
        )
        monkeypatch.setattr(
            open_module, "has_expected_worktree", lambda pd, branch, **kw: True
        )
        monkeypatch.setattr(
            open_module, "branch_session_name", lambda pd, branch: f"{pd.name}/{branch}"
        )
        monkeypatch.setattr(
            open_module, "ensure_dependency_configs_for_branch", lambda **kw: None
        )

        open_workspace(detached=True, pane_count=None, branch="feature", cwd=tmp_path)

        out = capsys.readouterr().out
        assert "tmux new-session -dP" in out
        assert f"-c {feat_path}" in out
        assert "Detached tmux session proj/feature created" in out

    def test_commits_config_with_worktree_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        proj_dir, repo_dir = self._base_setup(tmp_path, monkeypatch)
        config_dir = proj_dir / "config"
        config_dir.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=config_dir, env=env, check=True)
        feature_config = config_dir / "feature"
        feature_config.mkdir()
        (feature_config / "config.toml").write_text("[settings]\n", encoding="utf-8")
        feat_path = tmp_path / "worktrees" / "feature"
        feat_path.mkdir(parents=True)
        monkeypatch.setattr(
            open_module,
            "branch_worktree_path",
            lambda pd, branch, repo_branch: feat_path,
        )
        monkeypatch.setattr(
            open_module, "has_expected_worktree", lambda pd, branch, **kw: True
        )
        monkeypatch.setattr(
            open_module, "branch_session_name", lambda pd, branch: f"{pd.name}/{branch}"
        )
        monkeypatch.setattr(
            open_module, "ensure_dependency_configs_for_branch", lambda **kw: None
        )
        worktree_env = dict(env)
        worktree_env["GIT_AUTHOR_NAME"] = "Worktree Env"
        worktree_env["GIT_COMMITTER_NAME"] = "Worktree Env"
        monkeypatch.setattr(
            open_module,
            "load_workspace_env",
            lambda pd, branch, workspace_dir: worktree_env,
        )

        open_workspace(detached=True, pane_count=None, branch="feature", cwd=tmp_path)

        result = subprocess.run(
            ["git", "log", "-1", "--format=%an"],
            cwd=config_dir,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "Worktree Env"


# ===========================================================================
# create_workspace
# ===========================================================================


class TestNewSession:
    def test_plans_new_worktree_and_setup_queue(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)

        create_workspace(
            detached=True, pane_count=None, parent=None, branch="feature", cwd=project
        )

        out = capsys.readouterr().out
        assert "dry-run: agm mkdir" in out
        assert "git -C" in out
        assert "worktree add -b feature" in out
        assert "tmux send-keys" in out

    def test_plans_new_worktree_from_parent_start_point(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)

        create_workspace(
            detached=True, pane_count=None, parent="shallow-parent",
            branch="feature", cwd=project,
        )

        out = capsys.readouterr().out
        assert "worktree add -b feature" in out
        assert "shallow-parent" in out


# ===========================================================================
# checkout_workspace
# ===========================================================================


class TestCheckoutSession:
    def test_plans_checkout_of_existing_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)

        checkout_workspace(
            detached=True, pane_count=None, parent=None, branch="feature", cwd=project
        )

        out = capsys.readouterr().out
        assert "worktree add" in out
        assert "-b feature" not in out
        assert "feature" in out
        assert "tmux send-keys" in out


# ===========================================================================
# open_or_create_workspace
# ===========================================================================


class TestSmartOpenSession:
    def test_opens_main_session_when_main_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)

        open_or_create_workspace(
            detached=True, pane_count=None, parent=None, branch="main", cwd=project
        )

        out = capsys.readouterr().out
        assert "tmux new-session -dP" in out
        assert f"-c {project / 'repo'}" in out

    def test_opens_existing_worktree_session(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)
        subprocess.run(
            ["git", "worktree", "add", "-b", "feature", str(project / "worktrees" / "feature")],
            cwd=project / "repo",
            env=env,
            check=True,
        )

        open_or_create_workspace(
            detached=True, pane_count=None, parent=None, branch="feature", cwd=project
        )

        out = capsys.readouterr().out
        assert "tmux new-session -dP" in out
        assert f"-c {project / 'worktrees' / 'feature'}" in out

    def test_checks_out_existing_remote_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)
        subprocess.run(["git", "branch", "remote-feat"], cwd=project / "repo", env=env, check=True)

        open_or_create_workspace(
            detached=True, pane_count=None, parent=None, branch="remote-feat", cwd=project
        )

        out = capsys.readouterr().out
        assert "worktree add" in out
        assert "-b remote-feat" not in out
        assert "remote-feat" in out

    def test_creates_create_workspace_when_branch_doesnt_exist(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        project = _make_git_project(tmp_path, env)

        open_or_create_workspace(
            detached=True, pane_count=None, parent=None, branch="new-branch", cwd=project
        )

        out = capsys.readouterr().out
        assert "worktree add -b new-branch" in out


# ===========================================================================
# run (entry point)
# ===========================================================================


class TestOpenRun:
    def test_run_opens_main_branch_in_dry_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        dry_run.set_enabled(True)
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        repo_dir.mkdir(parents=True)
        monkeypatch.setattr(
            open_module, "require_current_project_dir", lambda cwd=None: proj_dir
        )
        monkeypatch.setattr(open_module, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(open_module.git_helpers, "current_branch", lambda rd: "main")
        monkeypatch.setattr(
            open_module,
            "is_main_workspace_branch",
            lambda pd, branch, repo_branch: True,
        )
        monkeypatch.setattr(open_module, "load_workspace_env", lambda pd, branch, workspace_dir: {})

        open_module.run(
            OpenArgs(detached=True, pane_count=None, parent=None, branch="main")
        )

        out = capsys.readouterr().out
        assert "tmux new-session -dP" in out
        assert f"-c {repo_dir}" in out
        assert "Detached tmux session proj created" in out
