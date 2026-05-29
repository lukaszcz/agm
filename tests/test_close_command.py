"""Tests for agm.commands.close."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import agm.commands.close as close_module
from agm.commands.args import CloseArgs
from agm.commands.close import close_session


def _make_git_close_project(
    tmp_path: Path,
    env: dict[str, str],
    *,
    unmerged: bool = False,
    dirty: bool = False,
) -> tuple[Path, Path, Path]:
    project_dir = tmp_path / "proj"
    repo_dir = project_dir / "repo"
    worktree_dir = project_dir / "worktrees" / "feature"
    (project_dir / "config").mkdir(parents=True)
    repo_dir.mkdir()

    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    (repo_dir / "README.md").write_text("main\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_dir, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_dir, env=env, check=True)
    subprocess.run(
        ["git", "worktree", "add", "-b", "feature", str(worktree_dir)],
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


def _install_fake_tmux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, path: str
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "tmux.log"
    tmux = bin_dir / "tmux"
    tmux.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {log_path}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{path}")
    return log_path

# ===========================================================================
# close_session branch config removal
# ===========================================================================


class TestCloseSessionRemovesBranchConfig:
    def _setup_close_dependencies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, branch: str
    ) -> tuple[Path, Path]:
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        (proj_dir / "config").mkdir(parents=True)
        repo_dir.mkdir()

        monkeypatch.setattr(
            close_module, "require_current_project_dir", lambda cwd=None: proj_dir
        )
        monkeypatch.setattr(close_module, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(close_module.git_helpers, "current_branch", lambda repo: "main")
        monkeypatch.setattr(
            close_module,
            "is_main_checkout_branch",
            lambda pd, close_branch, repo_branch: False,
        )
        monkeypatch.setattr(
            close_module.git_helpers, "branch_can_delete", lambda repo, b, **kw: True
        )
        monkeypatch.setattr(close_module, "remove_worktree", lambda **kw: None)
        monkeypatch.setattr(
            close_module,
            "load_worktree_env",
            lambda pd, config_branch, checkout_dir: {"HOME": str(tmp_path / "home")},
        )
        monkeypatch.setattr(
            close_module, "branch_session_name", lambda pd, close_branch: f"proj/{close_branch}"
        )
        monkeypatch.setattr(close_module, "close_tmux_session", lambda **kw: None)
        return proj_dir, proj_dir / "config" / branch

    def test_missing_branch_config_is_left_uncommitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, branch_config = self._setup_close_dependencies(
            tmp_path, monkeypatch, branch="feature"
        )

        close_session(branch="feature", cwd=tmp_path)

        assert not branch_config.exists()

    def test_removes_branch_dir_and_commits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir, branch_config = self._setup_close_dependencies(
            tmp_path, monkeypatch, branch="feature"
        )
        branch_config.mkdir()

        commit_calls: list[tuple[Path, str, list[Path]]] = []
        monkeypatch.setattr(
            close_module,
            "commit_config_dir_changes",
            lambda pd, msg, **kw: commit_calls.append((pd, msg, kw.get("add_paths", []))),
        )

        close_session(branch="feature", cwd=tmp_path)

        assert not branch_config.exists()
        assert len(commit_calls) == 1
        assert commit_calls[0][0] == proj_dir
        assert "remove config for feature" in commit_calls[0][1]
        assert commit_calls[0][2] == [branch_config]

    def test_removes_branch_config_file_and_commits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, branch_config = self._setup_close_dependencies(
            tmp_path, monkeypatch, branch="feature"
        )
        branch_config.write_text("data", encoding="utf-8")

        commit_calls: list[str] = []
        monkeypatch.setattr(
            close_module,
            "commit_config_dir_changes",
            lambda pd, msg, **kw: commit_calls.append(msg),
        )

        close_session(branch="feature", cwd=tmp_path)

        assert not branch_config.exists()
        assert len(commit_calls) == 1

    def test_commit_message_names_removed_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, branch_config = self._setup_close_dependencies(
            tmp_path, monkeypatch, branch="my-feature"
        )
        branch_config.mkdir()

        commit_calls: list[str] = []
        monkeypatch.setattr(
            close_module,
            "commit_config_dir_changes",
            lambda pd, msg, **kw: commit_calls.append(msg),
        )

        close_session(branch="my-feature", cwd=tmp_path)

        assert not branch_config.exists()
        assert len(commit_calls) == 1
        assert "my-feature" in commit_calls[0]


# ===========================================================================
# close_session
# ===========================================================================


class TestCloseSession:
    def _setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, is_main: bool = False
    ) -> None:
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        repo_dir.mkdir(parents=True)

        monkeypatch.setattr(
            close_module, "require_current_project_dir", lambda cwd=None: proj_dir
        )
        monkeypatch.setattr(close_module, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(
            close_module.git_helpers, "current_branch", lambda repo, **kw: "main"
        )
        monkeypatch.setattr(
            close_module, "is_main_checkout_branch", lambda pd, branch, repo_branch: is_main
        )
        monkeypatch.setattr(
            close_module, "load_worktree_env", lambda pd, branch, checkout_dir: {}
        )
        monkeypatch.setattr(
            close_module.git_helpers, "branch_can_delete", lambda repo, b, **kw: True
        )
        monkeypatch.setattr(close_module, "remove_worktree", lambda **kw: None)
        monkeypatch.setattr(
            close_module, "_remove_branch_config", lambda *, proj_dir, branch, env: None
        )
        monkeypatch.setattr(
            close_module, "branch_session_name", lambda pd, branch: f"{pd.name}/{branch}"
        )
        monkeypatch.setattr(close_module, "close_tmux_session", lambda **kw: None)

    def test_removes_worktree_branch_and_closes_session(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir, repo_dir, worktree_dir = _make_git_close_project(tmp_path, env)
        tmux_log = _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])

        close_session(branch="feature", cwd=project_dir)

        assert not worktree_dir.exists()
        assert not _branch_exists(repo_dir, "feature", env)
        assert "kill-session -t proj/feature" in tmux_log.read_text(encoding="utf-8")

    def test_exits_when_branch_is_main_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch, is_main=True)
        with pytest.raises(SystemExit) as exc_info:
            close_session(branch="main", cwd=tmp_path)
        assert exc_info.value.code == 1

    def test_exits_without_removing_worktree_when_branch_not_deletable(
        self,
        tmp_path: Path,
        env: dict[str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir, repo_dir, worktree_dir = _make_git_close_project(
            tmp_path, env, unmerged=True
        )

        with pytest.raises(SystemExit) as exc_info:
            close_session(branch="feature", cwd=project_dir)

        assert exc_info.value.code == 1
        assert worktree_dir.exists()
        assert _branch_exists(repo_dir, "feature", env)
        assert "not fully merged" in capsys.readouterr().err

    def test_exits_with_not_merged_message_when_branch_not_fully_merged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            close_module.git_helpers, "branch_can_delete", lambda repo, b, **kw: False
        )
        monkeypatch.setattr(
            close_module.git_helpers, "local_branch_exists", lambda repo, b, **kw: True
        )
        with pytest.raises(SystemExit):
            close_session(branch="feature", cwd=tmp_path)
        captured = capsys.readouterr()
        assert "not fully merged" in captured.err
        assert "-D" in captured.err

    def test_exits_with_not_found_message_when_branch_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            close_module.git_helpers, "branch_can_delete", lambda repo, b, **kw: False
        )
        monkeypatch.setattr(
            close_module.git_helpers, "local_branch_exists", lambda repo, b, **kw: False
        )
        with pytest.raises(SystemExit):
            close_session(branch="feature", cwd=tmp_path)
        assert "does not exist" in capsys.readouterr().err

    def test_force_delete_removes_unmerged_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir, repo_dir, worktree_dir = _make_git_close_project(
            tmp_path, env, unmerged=True
        )
        _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])

        close_session(branch="feature", force_delete=True, cwd=project_dir)

        assert not worktree_dir.exists()
        assert not _branch_exists(repo_dir, "feature", env)

    def test_force_removes_dirty_worktree_and_branch(
        self,
        tmp_path: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir, repo_dir, worktree_dir = _make_git_close_project(
            tmp_path, env, dirty=True
        )
        _install_fake_tmux(tmp_path, monkeypatch, path=env["PATH"])

        close_session(branch="feature", force=True, force_delete=False, cwd=project_dir)

        assert not worktree_dir.exists()
        assert not _branch_exists(repo_dir, "feature", env)


# ===========================================================================
# run (entry point)
# ===========================================================================


class TestCloseRun:
    def _setup_force_sensitive_close(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        repo_dir.mkdir(parents=True)
        (proj_dir / "config").mkdir()

        monkeypatch.setattr(
            close_module, "require_current_project_dir", lambda cwd=None: proj_dir
        )
        monkeypatch.setattr(close_module, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(close_module.git_helpers, "current_branch", lambda repo: "main")
        monkeypatch.setattr(
            close_module,
            "is_main_checkout_branch",
            lambda pd, branch, repo_branch: False,
        )
        monkeypatch.setattr(
            close_module.git_helpers,
            "branch_can_delete",
            lambda repo, branch, **kw: bool(kw.get("force")),
        )
        monkeypatch.setattr(
            close_module.git_helpers, "local_branch_exists", lambda repo, branch: True
        )
        monkeypatch.setattr(close_module, "remove_worktree", lambda **kw: None)
        monkeypatch.setattr(
            close_module, "load_worktree_env", lambda pd, branch, checkout_dir: {}
        )
        monkeypatch.setattr(
            close_module, "branch_session_name", lambda pd, branch: f"proj/{branch}"
        )
        monkeypatch.setattr(close_module, "close_tmux_session", lambda **kw: None)
        return repo_dir

    def test_run_without_force_rejects_unmerged_branch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._setup_force_sensitive_close(tmp_path, monkeypatch)

        with pytest.raises(SystemExit):
            close_module.run(CloseArgs(branch="feature", force=False, force_delete=False))

        assert "not fully merged" in capsys.readouterr().err

    def test_run_force_delete_allows_unmerged_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup_force_sensitive_close(tmp_path, monkeypatch)

        close_module.run(CloseArgs(branch="feature", force=False, force_delete=True))

    def test_run_force_allows_unmerged_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup_force_sensitive_close(tmp_path, monkeypatch)

        close_module.run(CloseArgs(branch="feature", force=True, force_delete=False))
