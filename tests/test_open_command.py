"""Tests for agm.commands.open."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agm.commands.open as open_module
from agm.commands.args import OpenArgs
from agm.commands.open import (
    branch_exists,
    branch_path,
    checkout_session,
    expected_branch_path,
    has_expected_worktree,
    new_session,
    open_session,
    queue_setup_and_focus_session,
    resolve_parent_checkout_dir,
    smart_open_session,
    validate_pane_count,
)
from agm.vcs.git import WorktreeInfo

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
# branch_path
# ===========================================================================


class TestBranchPath:
    def _make_project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        repo_dir.mkdir(parents=True)
        monkeypatch.setattr(open_module, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(
            open_module.git_helpers, "current_branch", lambda rd, **kw: "main"
        )
        monkeypatch.setattr(
            open_module,
            "branch_worktree_path",
            lambda pd, branch, repo_branch: pd / ".agm" / "worktrees" / branch,
        )
        return proj_dir

    def test_returns_path_for_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = self._make_project(tmp_path, monkeypatch)
        result = branch_path(proj_dir, "feature")
        assert result == proj_dir / ".agm" / "worktrees" / "feature"


# ===========================================================================
# expected_branch_path
# ===========================================================================


class TestExpectedBranchPath:
    def test_returns_resolved_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        worktree_path = tmp_path / "worktrees" / "feature"
        monkeypatch.setattr(
            open_module, "branch_path", lambda pd, branch: worktree_path
        )
        result = expected_branch_path(proj_dir, "feature")
        assert result == worktree_path.resolve(strict=False)


# ===========================================================================
# resolve_parent_checkout_dir
# ===========================================================================


class TestResolveParentCheckoutDir:
    def test_none_parent_returns_repo_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        repo_dir.mkdir(parents=True)
        monkeypatch.setattr(open_module, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(
            open_module.git_helpers, "current_branch", lambda rd, **kw: "main"
        )
        result = resolve_parent_checkout_dir(proj_dir, None, env={})
        assert result == repo_dir

    def test_parent_same_as_repo_branch_returns_repo_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        repo_dir.mkdir(parents=True)
        monkeypatch.setattr(open_module, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(
            open_module.git_helpers, "current_branch", lambda rd, **kw: "main"
        )
        result = resolve_parent_checkout_dir(proj_dir, "main", env={})
        assert result == repo_dir

    def test_different_parent_returns_branch_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        repo_dir.mkdir(parents=True)
        feat_path = tmp_path / "worktrees" / "develop"
        monkeypatch.setattr(open_module, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(
            open_module.git_helpers, "current_branch", lambda rd, **kw: "main"
        )
        monkeypatch.setattr(
            open_module, "branch_path", lambda pd, branch: feat_path
        )
        result = resolve_parent_checkout_dir(proj_dir, "develop", env={})
        assert result == feat_path


# ===========================================================================
# has_expected_worktree
# ===========================================================================


class TestHasExpectedWorktree:
    def _setup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        expected: Path,
        worktrees: list[WorktreeInfo],
    ) -> Path:
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        repo_dir.mkdir(parents=True)
        monkeypatch.setattr(open_module, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(
            open_module, "expected_branch_path", lambda pd, branch: expected
        )
        monkeypatch.setattr(
            open_module.git_helpers, "worktree_list", lambda rd, env=None: worktrees
        )
        return proj_dir

    def test_returns_true_when_worktree_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = tmp_path / "worktrees" / "feature"
        expected.mkdir(parents=True)
        proj_dir = self._setup(
            tmp_path,
            monkeypatch,
            expected=expected,
            worktrees=[WorktreeInfo(path=expected, branch="feature")],
        )
        assert has_expected_worktree(proj_dir, "feature") is True

    def test_returns_false_when_no_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = tmp_path / "worktrees" / "feature"
        other = tmp_path / "worktrees" / "other"
        other.mkdir(parents=True)
        proj_dir = self._setup(
            tmp_path,
            monkeypatch,
            expected=expected,
            worktrees=[WorktreeInfo(path=other, branch="other")],
        )
        assert has_expected_worktree(proj_dir, "feature") is False

    def test_returns_false_when_branch_differs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = tmp_path / "worktrees" / "feature"
        expected.mkdir(parents=True)
        proj_dir = self._setup(
            tmp_path,
            monkeypatch,
            expected=expected,
            worktrees=[WorktreeInfo(path=expected, branch="other-branch")],
        )
        assert has_expected_worktree(proj_dir, "feature") is False


# ===========================================================================
# branch_exists
# ===========================================================================


class TestBranchExists:
    def test_returns_true_when_local_branch_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            open_module.git_helpers, "local_branch_exists", lambda rd, branch, env=None: True
        )
        monkeypatch.setattr(
            open_module.git_helpers, "remote_branch_exists", lambda rd, branch, env=None: False
        )
        assert branch_exists(tmp_path, "feature") is True

    def test_returns_true_when_remote_branch_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            open_module.git_helpers, "local_branch_exists", lambda rd, branch, env=None: False
        )
        monkeypatch.setattr(
            open_module.git_helpers, "remote_branch_exists", lambda rd, branch, env=None: True
        )
        assert branch_exists(tmp_path, "feature") is True

    def test_returns_false_when_neither_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            open_module.git_helpers, "local_branch_exists", lambda rd, branch, env=None: False
        )
        monkeypatch.setattr(
            open_module.git_helpers, "remote_branch_exists", lambda rd, branch, env=None: False
        )
        assert branch_exists(tmp_path, "feature") is False


# ===========================================================================
# queue_setup_and_focus_session
# ===========================================================================


class TestQueueSetupAndFocusSession:
    def _mock_all(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
        calls: dict[str, list[Any]] = {"create": [], "queue": [], "focus": []}
        monkeypatch.setattr(
            open_module,
            "create_tmux_session",
            lambda **kw: (calls["create"].append(kw), "created-session")[1],
        )
        monkeypatch.setattr(
            open_module,
            "queue_command_in_session",
            lambda **kw: calls["queue"].append(kw),
        )
        monkeypatch.setattr(
            open_module,
            "focus_tmux_session",
            lambda **kw: (calls["focus"].append(kw), 0)[1],
        )
        return calls

    def test_detached_returns_without_focusing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._mock_all(monkeypatch)
        queue_setup_and_focus_session(
            detached=True,
            pane_count=None,
            session_name="s",
            repo_path=tmp_path,
            env={},
        )
        assert len(calls["create"]) == 1
        assert len(calls["queue"]) == 1
        assert len(calls["focus"]) == 0

    def test_not_detached_raises_system_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mock_all(monkeypatch)
        with pytest.raises(SystemExit) as exc_info:
            queue_setup_and_focus_session(
                detached=False,
                pane_count=None,
                session_name="s",
                repo_path=tmp_path,
                env={},
            )
        assert exc_info.value.code == 0

    def test_raises_assertion_when_session_name_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            open_module, "create_tmux_session", lambda **kw: None
        )
        with pytest.raises(AssertionError):
            queue_setup_and_focus_session(
                detached=True,
                pane_count=None,
                session_name="s",
                repo_path=tmp_path,
                env={},
            )


# ===========================================================================
# open_session
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
            open_module, "load_worktree_env", lambda pd, branch, checkout_dir: {}
        )
        monkeypatch.setattr(open_module, "create_tmux_session", lambda **kw: None)
        return proj_dir, repo_dir

    def test_opens_main_repo_session_when_branch_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir, repo_dir = self._base_setup(tmp_path, monkeypatch)
        create_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            open_module,
            "create_tmux_session",
            lambda **kw: create_calls.append(kw),
        )
        open_session(detached=True, pane_count=None, branch=None, cwd=tmp_path)
        assert len(create_calls) == 1
        assert create_calls[0]["session_name"] == proj_dir.name

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
            open_session(detached=True, pane_count=None, branch="feature", cwd=tmp_path)
        assert exc_info.value.code == 1

    def test_opens_branch_session_when_worktree_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj_dir, repo_dir = self._base_setup(tmp_path, monkeypatch)
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
        create_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            open_module,
            "create_tmux_session",
            lambda **kw: create_calls.append(kw),
        )
        open_session(detached=True, pane_count=None, branch="feature", cwd=tmp_path)
        assert len(create_calls) == 1
        assert "feature" in create_calls[0]["session_name"]


# ===========================================================================
# new_session
# ===========================================================================


class TestNewSession:
    def _setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        repo_dir.mkdir(parents=True)
        worktrees_path = tmp_path / "worktrees" / "feature"

        monkeypatch.setattr(
            open_module, "require_current_project_dir", lambda cwd=None: proj_dir
        )
        monkeypatch.setattr(
            open_module, "branch_path", lambda pd, branch: worktrees_path
        )
        monkeypatch.setattr(open_module, "mkdir", lambda p, **kw: None)
        monkeypatch.setattr(
            open_module, "ensure_dependency_configs_for_branch", lambda **kw: None
        )
        monkeypatch.setattr(
            open_module,
            "load_worktree_env",
            lambda pd, branch, checkout_dir: {"TMUX": ""},
        )
        monkeypatch.setattr(
            open_module,
            "resolve_parent_checkout_dir",
            lambda pd, parent, env: repo_dir,
        )
        monkeypatch.setattr(open_module, "ensure_worktree", lambda **kw: None)
        monkeypatch.setattr(
            open_module, "branch_session_name", lambda pd, branch: f"{pd.name}/{branch}"
        )
        monkeypatch.setattr(
            open_module, "queue_setup_and_focus_session", lambda **kw: None
        )
        monkeypatch.setattr(
            open_module, "resolve_parent_config_branch", lambda pd, parent: parent
        )
        return proj_dir

    def test_calls_ensure_worktree_with_new_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        worktree_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            open_module,
            "ensure_worktree",
            lambda **kw: worktree_calls.append(kw),
        )
        new_session(
            detached=True, pane_count=None, parent=None, branch="feature", cwd=tmp_path
        )
        assert len(worktree_calls) == 1
        assert worktree_calls[0]["new_branch"] == "feature"
        assert worktree_calls[0]["existing_ok"] is False

    def test_calls_queue_setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        setup_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            open_module,
            "queue_setup_and_focus_session",
            lambda **kw: setup_calls.append(kw),
        )
        new_session(
            detached=True, pane_count=None, parent=None, branch="feature", cwd=tmp_path
        )
        assert len(setup_calls) == 1
        assert setup_calls[0]["detached"] is True


# ===========================================================================
# checkout_session
# ===========================================================================


class TestCheckoutSession:
    def _setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
        proj_dir = tmp_path / "proj"
        repo_dir = proj_dir / "repo"
        repo_dir.mkdir(parents=True)
        worktrees_path = tmp_path / "worktrees" / "feature"

        monkeypatch.setattr(
            open_module, "require_current_project_dir", lambda cwd=None: proj_dir
        )
        monkeypatch.setattr(
            open_module, "branch_path", lambda pd, branch: worktrees_path
        )
        monkeypatch.setattr(open_module, "mkdir", lambda p, **kw: None)
        monkeypatch.setattr(
            open_module, "ensure_dependency_configs_for_branch", lambda **kw: None
        )
        monkeypatch.setattr(
            open_module, "load_worktree_env", lambda pd, branch, checkout_dir: {}
        )
        monkeypatch.setattr(
            open_module, "resolve_parent_checkout_dir", lambda pd, parent, env: repo_dir
        )
        monkeypatch.setattr(open_module, "ensure_worktree", lambda **kw: None)
        monkeypatch.setattr(
            open_module, "branch_session_name", lambda pd, branch: f"{pd.name}/{branch}"
        )
        monkeypatch.setattr(
            open_module, "queue_setup_and_focus_session", lambda **kw: None
        )
        monkeypatch.setattr(
            open_module, "resolve_parent_config_branch", lambda pd, parent: parent
        )
        return proj_dir

    def test_calls_ensure_worktree_with_existing_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._setup(tmp_path, monkeypatch)
        worktree_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            open_module, "ensure_worktree", lambda **kw: worktree_calls.append(kw)
        )
        checkout_session(
            detached=True, pane_count=None, parent=None, branch="feature", cwd=tmp_path
        )
        assert len(worktree_calls) == 1
        assert worktree_calls[0]["branch"] == "feature"
        assert worktree_calls[0]["existing_ok"] is True
        assert worktree_calls[0]["new_branch"] is None


# ===========================================================================
# smart_open_session
# ===========================================================================


class TestSmartOpenSession:
    def _base_setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
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
        monkeypatch.setattr(open_module.git_helpers, "fetch", lambda rd: None)
        return proj_dir

    def test_opens_main_session_when_main_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._base_setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            open_module,
            "is_main_checkout_branch",
            lambda pd, branch, repo_branch: True,
        )
        open_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            open_module, "open_session", lambda **kw: open_calls.append(kw)
        )
        smart_open_session(
            detached=True, pane_count=None, parent=None, branch="main", cwd=tmp_path
        )
        assert len(open_calls) == 1
        assert open_calls[0]["branch"] is None

    def test_opens_existing_worktree_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._base_setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            open_module, "is_main_checkout_branch", lambda pd, branch, repo_branch: False
        )
        monkeypatch.setattr(
            open_module, "has_expected_worktree", lambda pd, branch, **kw: True
        )
        open_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            open_module, "open_session", lambda **kw: open_calls.append(kw)
        )
        smart_open_session(
            detached=True, pane_count=None, parent=None, branch="feature", cwd=tmp_path
        )
        assert len(open_calls) == 1
        assert open_calls[0]["branch"] == "feature"

    def test_checks_out_existing_remote_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._base_setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            open_module, "is_main_checkout_branch", lambda pd, branch, repo_branch: False
        )
        monkeypatch.setattr(
            open_module, "has_expected_worktree", lambda pd, branch, **kw: False
        )
        monkeypatch.setattr(
            open_module, "branch_exists", lambda rd, branch, **kw: True
        )
        checkout_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            open_module, "checkout_session", lambda **kw: checkout_calls.append(kw)
        )
        smart_open_session(
            detached=True, pane_count=None, parent=None, branch="remote-feat", cwd=tmp_path
        )
        assert len(checkout_calls) == 1
        assert checkout_calls[0]["branch"] == "remote-feat"

    def test_creates_new_session_when_branch_doesnt_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._base_setup(tmp_path, monkeypatch)
        monkeypatch.setattr(
            open_module, "is_main_checkout_branch", lambda pd, branch, repo_branch: False
        )
        monkeypatch.setattr(
            open_module, "has_expected_worktree", lambda pd, branch, **kw: False
        )
        monkeypatch.setattr(
            open_module, "branch_exists", lambda rd, branch, **kw: False
        )
        new_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            open_module, "new_session", lambda **kw: new_calls.append(kw)
        )
        smart_open_session(
            detached=True, pane_count=None, parent=None, branch="new-branch", cwd=tmp_path
        )
        assert len(new_calls) == 1
        assert new_calls[0]["branch"] == "new-branch"


# ===========================================================================
# run (entry point)
# ===========================================================================


class TestOpenRun:
    def test_run_delegates_to_smart_open_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            open_module,
            "smart_open_session",
            lambda **kw: calls.append(kw),
        )
        open_module.run(
            OpenArgs(detached=True, pane_count=None, parent=None, branch="feature")
        )
        assert len(calls) == 1
        assert calls[0]["branch"] == "feature"
        assert calls[0]["detached"] is True
