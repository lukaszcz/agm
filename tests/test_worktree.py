"""Tests for agm.project.worktree."""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.project.worktree as worktree_mod
from agm.vcs.git import WorktreeInfo


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
    """Tests for remove_worktree."""

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

        deleted: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "branch_delete",
            lambda p, b, force=False, env=None: deleted.append((b, force)),
        )

        worktree_mod.remove_worktree(
            repo_dir=repo_dir, force=False, branch="feat"
        )

        assert removed == [worktree_path]
        assert deleted == [("feat", False)]

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
            worktree_mod.remove_worktree(
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
            worktree_mod.remove_worktree(
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
        deleted: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "branch_delete",
            lambda p, b, force=False, env=None: deleted.append((b, force)),
        )

        worktree_mod.remove_worktree(
            repo_dir=repo_dir, force=True, branch="feat"
        )

        assert force_values == [True]
        assert deleted == [("feat", False)]

    def test_passes_force_delete_to_branch_delete(
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
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "worktree_remove",
            lambda p, path, force=False, env=None: None,
        )

        deleted: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "branch_delete",
            lambda p, b, force=False, env=None: deleted.append((b, force)),
        )

        worktree_mod.remove_worktree(
            repo_dir=repo_dir, force=False, branch="feat", force_delete=True
        )

        assert deleted == [("feat", True)]
