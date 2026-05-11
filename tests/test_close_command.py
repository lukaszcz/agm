"""Tests for agm.commands.close."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agm.commands.close as close_module
from agm.commands.args import CloseArgs
from agm.commands.close import (
    _containing_git_root,
    _has_staged_changes,
    _remove_branch_config,
    close_session,
)

# ===========================================================================
# _containing_git_root
# ===========================================================================


class TestContainingGitRoot:
    def test_returns_none_for_nonexistent_path(self, tmp_path: Path) -> None:
        result = _containing_git_root(tmp_path / "nonexistent", env={})
        assert result is None

    def test_returns_none_when_git_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            close_module, "run_capture", lambda cmd, **kw: (1, "", "not a git repo")
        )
        result = _containing_git_root(tmp_path, env={})
        assert result is None

    def test_returns_path_when_git_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_root = tmp_path / "repo"
        monkeypatch.setattr(
            close_module,
            "run_capture",
            lambda cmd, **kw: (0, str(repo_root) + "\n", ""),
        )
        result = _containing_git_root(tmp_path, env={})
        assert result == repo_root

    def test_strips_whitespace_from_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_root = tmp_path / "repo"
        monkeypatch.setattr(
            close_module,
            "run_capture",
            lambda cmd, **kw: (0, f"  {repo_root}  \n", ""),
        )
        result = _containing_git_root(tmp_path, env={})
        assert result == repo_root


# ===========================================================================
# _has_staged_changes
# ===========================================================================


class TestHasStagedChanges:
    def test_returns_false_when_returncode_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            close_module, "run_capture", lambda cmd, **kw: (0, "", "")
        )
        assert _has_staged_changes(tmp_path, tmp_path / "file", env={}) is False

    def test_returns_true_when_returncode_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            close_module, "run_capture", lambda cmd, **kw: (1, "", "")
        )
        assert _has_staged_changes(tmp_path, tmp_path / "file", env={}) is True

    def test_exits_on_unexpected_returncode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            close_module, "run_capture", lambda cmd, **kw: (2, "stdout", "stderr")
        )

        def _raise_exit(rc: int, stdout: str = "", stderr: str = "") -> None:
            raise SystemExit(rc)

        monkeypatch.setattr(close_module, "exit_with_output", _raise_exit)
        with pytest.raises(SystemExit):
            _has_staged_changes(tmp_path, tmp_path / "file", env={})


# ===========================================================================
# _remove_branch_config
# ===========================================================================


class TestRemoveBranchConfig:
    def test_does_nothing_when_branch_config_dir_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        # config dir exists but no branch subdir
        config_dir = proj_dir / "config"
        config_dir.mkdir()
        monkeypatch.setattr(
            close_module, "project_config_dir", lambda pd: config_dir
        )
        # Should not raise
        _remove_branch_config(proj_dir=proj_dir, branch="feature", env={})

    def test_removes_branch_dir_and_commits_when_tracked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        branch_config = config_dir / "feature"
        branch_config.mkdir(parents=True)

        # git root must be a parent of config_dir so relative_to() works
        git_root = tmp_path / "proj"

        monkeypatch.setattr(close_module, "project_config_dir", lambda pd: config_dir)
        monkeypatch.setattr(
            close_module, "_containing_git_root", lambda path, env: git_root
        )

        rmtree_calls: list[Path] = []
        monkeypatch.setattr(close_module.fs, "rmtree", lambda p: rmtree_calls.append(p))

        run_capture_calls: list[list[str]] = []

        def fake_run_capture(cmd: list[str], **kw: Any) -> tuple[int, str, str]:
            run_capture_calls.append(cmd)
            return 0, "", ""

        monkeypatch.setattr(close_module, "run_capture", fake_run_capture)
        monkeypatch.setattr(
            close_module, "_has_staged_changes", lambda repo_dir, path, env: True
        )
        require_success_calls: list[list[str]] = []
        monkeypatch.setattr(
            close_module, "require_success", lambda cmd, env=None: require_success_calls.append(cmd)
        )

        _remove_branch_config(proj_dir=proj_dir, branch="feature", env={})

        assert rmtree_calls == [branch_config]
        # git add was called
        assert any("add" in " ".join(c) for c in run_capture_calls)
        # git commit was called
        assert any("commit" in " ".join(c) for c in require_success_calls)

    def test_skips_commit_when_no_staged_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        branch_config = config_dir / "feature"
        branch_config.mkdir(parents=True)

        monkeypatch.setattr(close_module, "project_config_dir", lambda pd: config_dir)
        monkeypatch.setattr(
            close_module, "_containing_git_root", lambda path, env: tmp_path / "proj"
        )
        monkeypatch.setattr(close_module.fs, "rmtree", lambda p: None)
        monkeypatch.setattr(
            close_module, "run_capture", lambda cmd, **kw: (0, "", "")
        )
        monkeypatch.setattr(
            close_module, "_has_staged_changes", lambda repo_dir, path, env: False
        )
        require_success_calls: list[list[str]] = []
        monkeypatch.setattr(
            close_module, "require_success", lambda cmd, env=None: require_success_calls.append(cmd)
        )

        _remove_branch_config(proj_dir=proj_dir, branch="feature", env={})

        assert not any("commit" in " ".join(c) for c in require_success_calls)

    def test_skips_commit_when_git_add_path_not_tracked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        branch_config = config_dir / "feature"
        branch_config.mkdir(parents=True)

        monkeypatch.setattr(close_module, "project_config_dir", lambda pd: config_dir)
        monkeypatch.setattr(
            close_module, "_containing_git_root", lambda path, env: tmp_path / "proj"
        )
        monkeypatch.setattr(close_module.fs, "rmtree", lambda p: None)

        def fake_run_capture(cmd: list[str], **kw: Any) -> tuple[int, str, str]:
            if "add" in cmd:
                return 1, "", "did not match any files"
            return 0, "", ""

        monkeypatch.setattr(close_module, "run_capture", fake_run_capture)
        require_success_calls: list[list[str]] = []
        monkeypatch.setattr(
            close_module, "require_success", lambda cmd, env=None: require_success_calls.append(cmd)
        )

        _remove_branch_config(proj_dir=proj_dir, branch="feature", env={})

        assert not any("commit" in " ".join(c) for c in require_success_calls)

    def test_exits_when_git_add_fails_with_unexpected_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        branch_config = config_dir / "feature"
        branch_config.mkdir(parents=True)

        monkeypatch.setattr(close_module, "project_config_dir", lambda pd: config_dir)
        monkeypatch.setattr(
            close_module, "_containing_git_root", lambda path, env: tmp_path / "proj"
        )
        monkeypatch.setattr(close_module.fs, "rmtree", lambda p: None)

        def fake_run_capture(cmd: list[str], **kw: Any) -> tuple[int, str, str]:
            if "add" in cmd:
                return 1, "", "some other error"
            return 0, "", ""

        monkeypatch.setattr(close_module, "run_capture", fake_run_capture)

        def _raise_exit(rc: int, stdout: str = "", stderr: str = "") -> None:
            raise SystemExit(rc)

        monkeypatch.setattr(close_module, "exit_with_output", _raise_exit)
        with pytest.raises(SystemExit):
            _remove_branch_config(proj_dir=proj_dir, branch="feature", env={})

    def test_removes_file_not_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        config_dir.mkdir(parents=True)
        branch_config_file = config_dir / "feature"
        branch_config_file.write_text("data")

        monkeypatch.setattr(close_module, "project_config_dir", lambda pd: config_dir)
        monkeypatch.setattr(close_module, "_containing_git_root", lambda path, env: None)

        unlink_calls: list[Path] = []
        monkeypatch.setattr(close_module.fs, "unlink", lambda p: unlink_calls.append(p))

        _remove_branch_config(proj_dir=proj_dir, branch="feature", env={})

        assert unlink_calls == [branch_config_file]

    def test_no_commit_when_no_git_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        branch_config = config_dir / "feature"
        branch_config.mkdir(parents=True)

        monkeypatch.setattr(close_module, "project_config_dir", lambda pd: config_dir)
        monkeypatch.setattr(close_module, "_containing_git_root", lambda path, env: None)
        monkeypatch.setattr(close_module.fs, "rmtree", lambda p: None)

        require_success_calls: list[list[str]] = []
        monkeypatch.setattr(
            close_module, "require_success", lambda cmd, env=None: require_success_calls.append(cmd)
        )

        _remove_branch_config(proj_dir=proj_dir, branch="feature", env={})

        assert not any("commit" in " ".join(c) for c in require_success_calls)


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

    def test_calls_remove_worktree_and_close_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        remove_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            close_module,
            "remove_worktree",
            lambda **kw: remove_calls.append(kw),
        )
        close_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            close_module,
            "close_tmux_session",
            lambda **kw: close_calls.append(kw),
        )
        close_session(branch="feature", cwd=tmp_path)
        assert len(remove_calls) == 1
        assert remove_calls[0]["branch"] == "feature"
        assert remove_calls[0]["force"] is False
        assert remove_calls[0]["force_delete"] is False
        assert len(close_calls) == 1

    def test_exits_when_branch_is_main_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch, is_main=True)
        with pytest.raises(SystemExit) as exc_info:
            close_session(branch="main", cwd=tmp_path)
        assert exc_info.value.code == 1

    def test_exits_without_removing_worktree_when_branch_not_deletable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            close_module.git_helpers, "branch_can_delete", lambda repo, b, **kw: False
        )
        monkeypatch.setattr(
            close_module.git_helpers, "local_branch_exists", lambda repo, b, **kw: False
        )
        remove_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            close_module, "remove_worktree", lambda **kw: remove_calls.append(kw)
        )
        with pytest.raises(SystemExit) as exc_info:
            close_session(branch="feature", cwd=tmp_path)
        assert exc_info.value.code == 1
        # Worktree should NOT have been removed
        assert remove_calls == []

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

    def test_passes_force_delete_to_remove_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        remove_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            close_module,
            "remove_worktree",
            lambda **kw: remove_calls.append(kw),
        )
        close_session(branch="feature", force_delete=True, cwd=tmp_path)
        assert remove_calls[0]["force"] is False
        assert remove_calls[0]["force_delete"] is True

    def test_skips_pre_check_when_force_delete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        # branch_can_delete with force=True just checks existence; it returns True
        # even for unmerged branches
        monkeypatch.setattr(
            close_module.git_helpers, "branch_can_delete", lambda repo, b, **kw: True
        )
        remove_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            close_module,
            "remove_worktree",
            lambda **kw: remove_calls.append(kw),
        )
        close_session(branch="feature", force_delete=True, cwd=tmp_path)
        assert len(remove_calls) == 1

    def test_force_passes_force_and_force_delete_to_remove_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        remove_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            close_module,
            "remove_worktree",
            lambda **kw: remove_calls.append(kw),
        )
        close_session(branch="feature", force=True, cwd=tmp_path)
        assert remove_calls[0]["force"] is True
        assert remove_calls[0]["force_delete"] is True

    def test_force_implies_force_delete_even_when_force_delete_is_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        remove_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            close_module,
            "remove_worktree",
            lambda **kw: remove_calls.append(kw),
        )
        close_session(branch="feature", force=True, force_delete=False, cwd=tmp_path)
        assert remove_calls[0]["force"] is True
        assert remove_calls[0]["force_delete"] is True


# ===========================================================================
# run (entry point)
# ===========================================================================


class TestCloseRun:
    def test_run_delegates_to_close_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            close_module,
            "close_session",
            lambda *, branch, force=False, force_delete=False, cwd=None: calls.append(
                {"branch": branch, "force": force, "force_delete": force_delete}
            ),
        )
        close_module.run(CloseArgs(branch="feature", force=False, force_delete=False))
        assert calls == [{"branch": "feature", "force": False, "force_delete": False}]

    def test_run_passes_force_delete_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            close_module,
            "close_session",
            lambda *, branch, force=False, force_delete=False, cwd=None: calls.append(
                {"branch": branch, "force": force, "force_delete": force_delete}
            ),
        )
        close_module.run(CloseArgs(branch="feature", force=False, force_delete=True))
        assert calls == [{"branch": "feature", "force": False, "force_delete": True}]

    def test_run_passes_force_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            close_module,
            "close_session",
            lambda *, branch, force=False, force_delete=False, cwd=None: calls.append(
                {"branch": branch, "force": force, "force_delete": force_delete}
            ),
        )
        close_module.run(CloseArgs(branch="feature", force=True, force_delete=False))
        assert calls == [{"branch": "feature", "force": True, "force_delete": False}]