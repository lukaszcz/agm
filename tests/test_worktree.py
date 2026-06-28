"""Tests for agm.project.worktree."""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.project.worktree as worktree_mod
from agm.project.worktree import (
    branch_exists,
    ensure_worktree,
    has_expected_worktree,
)
from agm.vcs.git import WorktreeInfo


class TestProjectWorktreeHelpers:
    def test_has_expected_worktree_returns_true_for_matching_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = tmp_path / "project" / "worktrees" / "feature"
        expected.mkdir(parents=True)
        monkeypatch.setattr(worktree_mod, "project_repo_dir", lambda project: project / "repo")
        monkeypatch.setattr(
            worktree_mod,
            "expected_branch_worktree_path",
            lambda project, branch: expected,
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "worktree_list",
            lambda repo, env=None: [WorktreeInfo(path=expected, branch="feature")],
        )

        assert has_expected_worktree(tmp_path / "project", "feature")

    def test_has_expected_worktree_returns_false_without_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = tmp_path / "project" / "worktrees" / "feature"
        expected.mkdir(parents=True)
        monkeypatch.setattr(worktree_mod, "project_repo_dir", lambda project: project / "repo")
        monkeypatch.setattr(
            worktree_mod,
            "expected_branch_worktree_path",
            lambda project, branch: expected,
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "worktree_list",
            lambda repo, env=None: [WorktreeInfo(path=expected, branch="other")],
        )

        assert not has_expected_worktree(tmp_path / "project", "feature")

    def test_branch_exists_checks_local_and_remote(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            worktree_mod.git_helpers, "local_branch_exists", lambda repo, branch, env=None: False
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers, "remote_branch_exists", lambda repo, branch, env=None: True
        )

        assert branch_exists(tmp_path, "feature")




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

    def test_skips_origin_head(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_fetches_prune_and_syncs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        monkeypatch.setattr(
            worktree_mod.git_helpers, "checkout_root", lambda cwd=None, env=None: repo_dir
        )

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
            worktree_mod.git_helpers, "checkout_root", lambda cwd=None, env=None: repo_dir
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers, "current_branch", lambda p, env=None: repo_branch
        )
        monkeypatch.setattr(
            worktree_mod, "discover_current_project_dir", lambda cwd=None, env=None: project_dir
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
            start_point: str | None = None,
            env: dict[str, str] | None = None,
        ) -> None:
            add_calls.append(
                {"repo": repo, "path": path, "branch": branch,
                 "create": create, "start_point": start_point}
            )

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
        worktrees_dir = project_dir / "worktrees"
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
        worktrees_dir = project_dir / "worktrees"
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

    def test_reuse_existing_branch_keeps_create_when_branch_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When reuse_existing_branch=True but the branch does not exist
        locally or remotely, create_branch stays True."""
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        project_dir.mkdir()
        repo_dir.mkdir()

        add_calls = self._setup_mocks(monkeypatch, project_dir, repo_dir)

        # Branch does not exist anywhere
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "local_branch_exists",
            lambda p, b, env=None: False,
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "remote_branch_exists",
            lambda p, b, env=None: False,
        )

        worktree_mod.ensure_worktree(
            new_branch="brand-new",
            worktrees_dir=None,
            branch=None,
            reuse_existing_branch=True,
            cwd=project_dir,
        )

        assert len(add_calls) == 1
        # Branch didn't exist so create_branch was NOT flipped to False
        assert add_calls[0]["create"] is True

    def test_skips_non_matching_worktrees_in_existing_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When existing_worktrees contains entries that don't match the target,
        the for-loop continues to the next entry (no match, skip)."""
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        other_path = project_dir / "worktrees" / "other"
        project_dir.mkdir()
        repo_dir.mkdir()
        other_path.mkdir(parents=True)

        # Provide an existing worktree that does NOT match the target branch+path
        non_matching = WorktreeInfo(path=other_path, branch="other")
        add_calls = self._setup_mocks(
            monkeypatch, project_dir, repo_dir, existing_worktrees=[non_matching]
        )

        worktree_mod.ensure_worktree(
            new_branch="brand-new",
            worktrees_dir=None,
            branch=None,
            cwd=project_dir,
        )

        assert len(add_calls) == 1
        assert add_calls[0]["create"] is True
        assert add_calls[0]["branch"] == "brand-new"

    def test_passes_start_point_to_worktree_add(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        project_dir.mkdir()
        repo_dir.mkdir()

        add_calls = self._setup_mocks(monkeypatch, project_dir, repo_dir)

        ensure_worktree(
            new_branch="feat",
            worktrees_dir=None,
            branch=None,
            start_point="parent-branch",
            cwd=project_dir,
        )

        assert len(add_calls) == 1
        assert add_calls[0]["start_point"] == "parent-branch"
        assert add_calls[0]["create"] is True

    def test_no_start_point_defaults_to_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        project_dir.mkdir()
        repo_dir.mkdir()

        add_calls = self._setup_mocks(monkeypatch, project_dir, repo_dir)

        ensure_worktree(
            new_branch="feat",
            worktrees_dir=None,
            branch=None,
            cwd=project_dir,
        )

        assert len(add_calls) == 1
        assert add_calls[0]["start_point"] is None

    def test_plain_git_repo_skips_agm_project_steps(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        monkeypatch.setattr(
            worktree_mod.git_helpers, "checkout_root", lambda cwd=None, env=None: repo_dir
        )
        monkeypatch.setattr(
            worktree_mod, "discover_current_project_dir", lambda cwd=None, env=None: None
        )
        monkeypatch.setattr(worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(worktree_mod.git_helpers, "fetch", lambda p, env=None: None)
        monkeypatch.setattr(worktree_mod.git_helpers, "worktree_list", lambda p, env=None: [])
        monkeypatch.setattr(
            worktree_mod,
            "exit_if_main_workspace_branch",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected check")),
        )
        monkeypatch.setattr(
            worktree_mod,
            "ensure_dependency_configs_for_branch",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected deps")),
        )
        monkeypatch.setattr(
            worktree_mod,
            "copy_config",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected config")),
        )

        add_calls: list[Path] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "worktree_add",
            lambda repo, path, branch, create=False, start_point=None, env=None: (
                add_calls.append(path)
            ),
        )

        result = worktree_mod.ensure_worktree(
            new_branch="feat",
            worktrees_dir=None,
            branch=None,
            cwd=repo_dir,
        )

        assert result == repo_dir / "worktrees" / "feat"
        assert add_calls == [repo_dir / "worktrees" / "feat"]
        assert "warning: no AGM project found" in capsys.readouterr().err


class TestRemoveWorktree:
    """Tests for remove_worktree."""

    def test_removes_worktree_and_deletes_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktree_path = project_dir / "worktrees" / "feat"
        project_dir.mkdir()
        repo_dir.mkdir()
        worktree_path.mkdir(parents=True)

        monkeypatch.setattr(
            worktree_mod, "discover_current_project_dir", lambda cwd=None, env=None: project_dir
        )
        monkeypatch.setattr(worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main")
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

        worktree_mod.remove_worktree(repo_dir=repo_dir, force=False, branch="feat")

        assert removed == [worktree_path]
        assert deleted == [("feat", False)]

    def test_exits_when_worktree_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        project_dir.mkdir()
        repo_dir.mkdir()

        monkeypatch.setattr(
            worktree_mod, "discover_current_project_dir", lambda cwd=None, env=None: project_dir
        )
        monkeypatch.setattr(worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(worktree_mod.git_helpers, "worktree_list", lambda p, env=None: [])

        require_calls: list[list[str]] = []
        monkeypatch.setattr(
            worktree_mod,
            "require_success",
            lambda cmd, env=None: require_calls.append(cmd),
        )

        with pytest.raises(SystemExit):
            worktree_mod.remove_worktree(repo_dir=repo_dir, force=False, branch="nonexistent")

    def test_exits_when_worktree_list_has_entries_but_no_branch_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worktree loop iterates over entries but finds no match;
        worktree_path stays None and the function exits."""
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        other_path = project_dir / "worktrees" / "other"
        project_dir.mkdir()
        repo_dir.mkdir()
        other_path.mkdir(parents=True)

        monkeypatch.setattr(
            worktree_mod, "discover_current_project_dir", lambda cwd=None, env=None: project_dir
        )
        monkeypatch.setattr(worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main")
        # List has one entry, but it's for "other", not "missing"
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "worktree_list",
            lambda p, env=None: [WorktreeInfo(path=other_path, branch="other")],
        )
        monkeypatch.setattr(
            worktree_mod,
            "require_success",
            lambda cmd, env=None: None,
        )

        with pytest.raises(SystemExit):
            worktree_mod.remove_worktree(repo_dir=repo_dir, force=False, branch="missing")

    def test_exits_when_branch_is_main_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        project_dir.mkdir()
        repo_dir.mkdir()

        monkeypatch.setattr(
            worktree_mod, "discover_current_project_dir", lambda cwd=None, env=None: project_dir
        )
        # current branch is "main" — trying to remove "main" should exit
        monkeypatch.setattr(worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main")

        with pytest.raises(SystemExit):
            worktree_mod.remove_worktree(repo_dir=repo_dir, force=False, branch="main")

    def test_passes_force_flag_to_worktree_remove(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktree_path = project_dir / "worktrees" / "feat"
        project_dir.mkdir()
        repo_dir.mkdir()
        worktree_path.mkdir(parents=True)

        monkeypatch.setattr(
            worktree_mod, "discover_current_project_dir", lambda cwd=None, env=None: project_dir
        )
        monkeypatch.setattr(worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main")
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

        worktree_mod.remove_worktree(repo_dir=repo_dir, force=True, branch="feat")

        assert force_values == [True]
        assert deleted == [("feat", False)]

    def test_passes_force_delete_to_branch_delete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktree_path = project_dir / "worktrees" / "feat"
        project_dir.mkdir()
        repo_dir.mkdir()
        worktree_path.mkdir(parents=True)

        monkeypatch.setattr(
            worktree_mod, "discover_current_project_dir", lambda cwd=None, env=None: project_dir
        )
        monkeypatch.setattr(worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main")
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

    def test_plain_git_repo_skips_agm_main_branch_check(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        repo_dir = tmp_path / "repo"
        worktree_path = tmp_path / "worktrees" / "main"
        repo_dir.mkdir()
        worktree_path.mkdir(parents=True)
        monkeypatch.setattr(
            worktree_mod, "discover_current_project_dir", lambda cwd=None, env=None: None
        )
        monkeypatch.setattr(worktree_mod.git_helpers, "current_branch", lambda p, env=None: "main")
        monkeypatch.setattr(
            worktree_mod,
            "exit_if_main_workspace_branch",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected check")),
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "worktree_list",
            lambda p, env=None: [WorktreeInfo(path=worktree_path, branch="main")],
        )

        removed: list[Path] = []
        deleted: list[str] = []
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "worktree_remove",
            lambda p, path, force=False, env=None: removed.append(path),
        )
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "branch_delete",
            lambda p, b, force=False, env=None: deleted.append(b),
        )

        worktree_mod.remove_worktree(repo_dir=repo_dir, force=False, branch="main")

        assert removed == [worktree_path]
        assert deleted == ["main"]
        assert "warning: no AGM project found" in capsys.readouterr().err


class TestEnsureWorktreeRelativePath:
    def test_relative_worktrees_path_resolved_against_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When worktrees_dir is relative, it's resolved against cwd."""
        import agm.project.worktree as worktree_module

        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)
        (project / "worktrees").mkdir()

        relative_dir = "custom_worktrees"
        monkeypatch.setattr(
            worktree_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: repo,
        )
        monkeypatch.setattr(
            worktree_module.git_helpers,
            "current_branch",
            lambda p, env=None: "main",
        )
        monkeypatch.setattr(worktree_module.git_helpers, "fetch", lambda p, env=None: None)
        monkeypatch.setattr(
            worktree_module.git_helpers,
            "worktree_list",
            lambda p, env=None: [],
        )
        monkeypatch.setattr(
            worktree_module.git_helpers,
            "worktree_add",
            lambda p, dirname, branch, create=False, start_point=None, env=None: None,
        )
        monkeypatch.setattr(
            worktree_module,
            "ensure_dependency_configs_for_branch",
            lambda project_dir, branch: None,
        )
        monkeypatch.setattr(
            worktree_module,
            "copy_config",
            lambda project_dir=None, target=None, branch=None, cwd=None: None,
        )
        monkeypatch.setattr(
            worktree_module, "discover_current_project_dir", lambda cwd=None, env=None: project
        )
        monkeypatch.setattr(
            worktree_module, "exit_if_main_workspace_branch", lambda pd, b, repo_branch=None: None
        )

        # Pass a relative worktrees_dir
        result = ensure_worktree(
            new_branch=None,
            worktrees_dir=relative_dir,
            branch="feat",
            cwd=repo,
        )
        # The result should be absolute (resolved against cwd=repo)
        assert result.is_absolute()
        assert result == repo / relative_dir / "feat"
