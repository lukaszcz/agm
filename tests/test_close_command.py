"""Tests for agm.commands.close."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agm.commands.close as close_module
from agm.commands.args import CloseArgs
from agm.commands.close import _remove_branch_config, close_session

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
        commit_calls: list[tuple[Path, str]] = []
        monkeypatch.setattr(
            close_module,
            "commit_config_dir_changes",
            lambda pd, msg, **kw: commit_calls.append((pd, msg)),
        )
        # Should not raise and should not call commit (no branch dir to remove)
        _remove_branch_config(proj_dir=proj_dir, branch="feature", env={})
        assert commit_calls == []

    def test_removes_branch_dir_and_commits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        branch_config = config_dir / "feature"
        branch_config.mkdir(parents=True)

        monkeypatch.setattr(close_module, "project_config_dir", lambda pd: config_dir)

        rmtree_calls: list[Path] = []
        monkeypatch.setattr(close_module.fs, "rmtree", lambda p: rmtree_calls.append(p))

        commit_calls: list[tuple[Path, str, list[Path]]] = []
        monkeypatch.setattr(
            close_module,
            "commit_config_dir_changes",
            lambda pd, msg, **kw: commit_calls.append((pd, msg, kw.get("add_paths", []))),
        )

        env = {"HOME": "/tmp"}
        _remove_branch_config(proj_dir=proj_dir, branch="feature", env=env)

        assert rmtree_calls == [branch_config]
        assert len(commit_calls) == 1
        assert commit_calls[0][0] == proj_dir
        assert "remove config for feature" in commit_calls[0][1]
        assert commit_calls[0][2] == [branch_config]

    def test_removes_file_not_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        config_dir.mkdir(parents=True)
        branch_config_file = config_dir / "feature"
        branch_config_file.write_text("data")

        monkeypatch.setattr(close_module, "project_config_dir", lambda pd: config_dir)

        unlink_calls: list[Path] = []
        monkeypatch.setattr(close_module.fs, "unlink", lambda p: unlink_calls.append(p))

        commit_calls: list[tuple[Path, str]] = []
        monkeypatch.setattr(
            close_module,
            "commit_config_dir_changes",
            lambda pd, msg, **kw: commit_calls.append((pd, msg)),
        )

        _remove_branch_config(proj_dir=proj_dir, branch="feature", env={})

        assert unlink_calls == [branch_config_file]
        assert len(commit_calls) == 1

    def test_commits_with_branch_name_in_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        branch_config = config_dir / "my-feature"
        branch_config.mkdir(parents=True)

        monkeypatch.setattr(close_module, "project_config_dir", lambda pd: config_dir)
        monkeypatch.setattr(close_module.fs, "rmtree", lambda p: None)

        commit_calls: list[str] = []
        monkeypatch.setattr(
            close_module,
            "commit_config_dir_changes",
            lambda pd, msg, **kw: commit_calls.append(msg),
        )

        _remove_branch_config(proj_dir=proj_dir, branch="my-feature", env={})

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
