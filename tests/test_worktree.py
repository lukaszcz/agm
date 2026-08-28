"""Behavior tests for agm.project.worktree over real git repositories.

Every test builds real repositories in a temporary directory and asserts the
resulting git state — worktrees, branches, upstreams, checked-out content and
copied config — rather than which helper happened to be called.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

import agm.project.worktree as worktree_mod
from agm.project.worktree import (
    branch_exists,
    branch_sync,
    ensure_worktree,
    has_expected_worktree,
    remove_worktree,
    sync_remote_tracking_branches,
)
from tests._git_helpers import clone_with_fork_remote, init_repo


def _git(*args: str, cwd: Path, env: dict[str, str]) -> str:
    """Run a git command in *cwd* and return its stdout."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _bare_origin(
    parent: Path, env: dict[str, str], *, name: str = "origin", branches: Sequence[str] = ()
) -> Path:
    """Create a bare origin named *name* whose default branch is ``main``.

    Each entry in *branches* becomes a published branch carrying its own commit,
    so it is not merged into ``main``.
    """
    source = init_repo(parent / f"{name}-source", env)
    for branch in branches:
        _git("checkout", "-b", branch, "-q", cwd=source, env=env)
        (source / f"{branch}.txt").write_text(f"{branch}\n", encoding="utf-8")
        _git("add", ".", cwd=source, env=env)
        _git("commit", "-m", f"add {branch}", "-q", cwd=source, env=env)
        _git("checkout", "main", "-q", cwd=source, env=env)
    origin = parent / f"{name}.git"
    subprocess.run(["git", "clone", "--bare", "-q", str(source), str(origin)], env=env, check=True)
    return origin


def _make_project(tmp_path: Path, env: dict[str, str], *, branches: Sequence[str] = ()) -> Path:
    """Create a split-layout AGM project whose ``repo/`` is a real checkout."""
    origin = _bare_origin(tmp_path, env, branches=branches)
    project = tmp_path / "proj"
    for name in ("worktrees", "deps", "config"):
        (project / name).mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(project / "repo")], env=env, check=True)
    return project


def _plain_repo(tmp_path: Path, env: dict[str, str]) -> Path:
    """Create a standalone checkout that belongs to no AGM project."""
    origin = _bare_origin(tmp_path, env, name="plain-origin")
    repo = tmp_path / "plain"
    subprocess.run(["git", "clone", "-q", str(origin), str(repo)], env=env, check=True)
    return repo


def _local_branches(repo: Path, env: dict[str, str]) -> list[str]:
    """Return the repository's local branch names, sorted."""
    output = _git("for-each-ref", "--format=%(refname:short)", "refs/heads", cwd=repo, env=env)
    return sorted(line for line in output.splitlines() if line)


def _upstream(repo: Path, branch: str, env: dict[str, str]) -> str:
    """Return the upstream ref configured for *branch*."""
    return _git("rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}", cwd=repo, env=env).strip()


def _head_branch(checkout: Path, env: dict[str, str]) -> str:
    """Return the branch checked out in *checkout*."""
    return _git("rev-parse", "--abbrev-ref", "HEAD", cwd=checkout, env=env).strip()


def _worktree_paths(repo: Path, env: dict[str, str]) -> list[Path]:
    """Return the paths git currently lists as worktrees of *repo*."""
    output = _git("worktree", "list", "--porcelain", cwd=repo, env=env)
    return [
        Path(line.removeprefix("worktree ")).resolve()
        for line in output.splitlines()
        if line.startswith("worktree ")
    ]


class TestSyncRemoteTrackingBranches:
    """Local tracking branches are created for unmerged remote branches."""

    def test_creates_tracking_branches_for_unmerged_remote_branches(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env, branches=["feat-a", "feat-b"])
        repo = project / "repo"

        sync_remote_tracking_branches(repo, env=env)

        assert _local_branches(repo, env) == ["feat-a", "feat-b", "main"]
        assert _upstream(repo, "feat-a", env) == "origin/feat-a"
        assert _upstream(repo, "feat-b", env) == "origin/feat-b"

    def test_ignores_remote_branches_already_merged_into_the_default_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        repo = project / "repo"
        # A published branch pointing at main's own commit is already merged.
        _git("push", "-q", "origin", "main:refs/heads/released", cwd=repo, env=env)
        _git("fetch", "-q", "origin", cwd=repo, env=env)

        sync_remote_tracking_branches(repo, env=env)

        assert _local_branches(repo, env) == ["main"]

    def test_leaves_an_existing_local_branch_untouched(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env, branches=["feat-a"])
        repo = project / "repo"
        # A local branch of the same name that points somewhere else entirely.
        _git("branch", "feat-a", "main", cwd=repo, env=env)
        before = _git("rev-parse", "feat-a", cwd=repo, env=env).strip()

        sync_remote_tracking_branches(repo, env=env)

        assert _git("rev-parse", "feat-a", cwd=repo, env=env).strip() == before

    def test_skips_the_remote_head_pointer(
        self, tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``origin/HEAD`` names the default branch, not a branch to track.

        Git's own ref listing never spells the pointer that way, so the listing
        is stubbed at the git boundary; the assertion is still on the branches
        the repository ends up with.
        """
        project = _make_project(tmp_path, env, branches=["feat-a"])
        repo = project / "repo"
        monkeypatch.setattr(
            worktree_mod.git_helpers,
            "remote_unmerged_branches",
            lambda repo_dir, *, base_ref, env=None: ["origin/HEAD", "origin/feat-a"],
        )

        sync_remote_tracking_branches(repo, env=env)

        assert _local_branches(repo, env) == ["feat-a", "main"]


class TestBranchSync:
    """``branch_sync`` fetches first, then creates the missing tracking branches."""

    def test_fetches_before_creating_tracking_branches(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        repo = project / "repo"
        # Publish a branch after the project clone, so only a fetch reveals it.
        publisher = tmp_path / "publisher"
        subprocess.run(
            ["git", "clone", "-q", str(tmp_path / "origin.git"), str(publisher)],
            env=env,
            check=True,
        )
        _git("checkout", "-b", "late", "-q", cwd=publisher, env=env)
        (publisher / "late.txt").write_text("late\n", encoding="utf-8")
        _git("add", ".", cwd=publisher, env=env)
        _git("commit", "-m", "late", "-q", cwd=publisher, env=env)
        _git("push", "-q", "origin", "late", cwd=publisher, env=env)

        branch_sync(cwd=repo, env=env)

        assert _local_branches(repo, env) == ["late", "main"]

    def test_finds_the_checkout_from_the_project_root(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env, branches=["feat-a"])

        branch_sync(cwd=project, env=env)

        assert _local_branches(project / "repo", env) == ["feat-a", "main"]


class TestHasExpectedWorktree:
    """A branch counts as checked out only at the path the project expects."""

    def test_true_when_the_branch_sits_at_the_expected_path(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        _git(
            "worktree",
            "add",
            "-b",
            "feat",
            str(project / "worktrees" / "feat"),
            "-q",
            cwd=project / "repo",
            env=env,
        )

        assert has_expected_worktree(project, "feat", env=env)

    def test_false_when_the_branch_sits_somewhere_else(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        _git(
            "worktree",
            "add",
            "-b",
            "feat",
            str(tmp_path / "elsewhere"),
            "-q",
            cwd=project / "repo",
            env=env,
        )

        assert not has_expected_worktree(project, "feat", env=env)

    def test_false_when_the_branch_is_not_checked_out_at_all(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)

        assert not has_expected_worktree(project, "feat", env=env)


class TestBranchExists:
    """A branch exists when it is local, or carried by exactly one remote."""

    def test_finds_a_local_branch(self, tmp_path: Path, env: dict[str, str]) -> None:
        project = _make_project(tmp_path, env)
        _git("branch", "local-only", cwd=project / "repo", env=env)

        assert branch_exists(project / "repo", "local-only", env=env)

    def test_finds_a_branch_carried_only_by_a_remote(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env, branches=["feat-a"])

        assert "feat-a" not in _local_branches(project / "repo", env)
        assert branch_exists(project / "repo", "feat-a", env=env)

    def test_rejects_an_unknown_branch(self, tmp_path: Path, env: dict[str, str]) -> None:
        project = _make_project(tmp_path, env)

        assert not branch_exists(project / "repo", "ghost", env=env)


class TestEnsureWorktree:
    """``ensure_worktree`` leaves a usable checkout at the project's worktree path."""

    def test_creates_a_new_branch_and_its_worktree(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)

        result = ensure_worktree(
            new_branch="feat",
            worktrees_dir=None,
            branch=None,
            cwd=project / "repo",
            env=env,
        )

        assert result.resolve() == (project / "worktrees" / "feat").resolve()
        assert _head_branch(result, env) == "feat"
        assert "feat" in _local_branches(project / "repo", env)
        assert result.resolve() in _worktree_paths(project / "repo", env)

    def test_checks_out_an_existing_local_branch(self, tmp_path: Path, env: dict[str, str]) -> None:
        project = _make_project(tmp_path, env, branches=["feat-a"])
        repo = project / "repo"
        _git("branch", "--track", "feat-a", "origin/feat-a", cwd=repo, env=env)

        result = ensure_worktree(
            new_branch=None,
            worktrees_dir=None,
            branch="feat-a",
            cwd=repo,
            env=env,
        )

        assert _head_branch(result, env) == "feat-a"
        # The existing branch was checked out, not recreated off main.
        assert (result / "feat-a.txt").is_file()

    def test_checks_out_a_branch_carried_only_by_the_remote(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env, branches=["feat-a"])
        repo = project / "repo"

        result = ensure_worktree(
            new_branch=None,
            worktrees_dir=None,
            branch="feat-a",
            cwd=repo,
            env=env,
        )

        assert (result / "feat-a.txt").is_file()
        assert _upstream(repo, "feat-a", env) == "origin/feat-a"

    def test_exits_when_the_branch_exists_nowhere(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)

        with pytest.raises(SystemExit):
            ensure_worktree(
                new_branch=None,
                worktrees_dir=None,
                branch="ghost",
                cwd=project / "repo",
                env=env,
            )

        assert not (project / "worktrees" / "ghost").exists()

    def test_exits_without_a_branch_name(self, tmp_path: Path, env: dict[str, str]) -> None:
        project = _make_project(tmp_path, env)

        with pytest.raises(SystemExit):
            ensure_worktree(
                new_branch=None,
                worktrees_dir=None,
                branch=None,
                cwd=project / "repo",
                env=env,
            )

        assert list((project / "worktrees").iterdir()) == []

    def test_exits_when_the_branch_is_the_main_workspace(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)

        with pytest.raises(SystemExit):
            ensure_worktree(
                new_branch=None,
                worktrees_dir=None,
                branch="main",
                cwd=project / "repo",
                env=env,
            )

        assert list((project / "worktrees").iterdir()) == []

    def test_returns_the_existing_worktree_when_allowed(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        first = ensure_worktree(
            new_branch="feat", worktrees_dir=None, branch=None, cwd=project / "repo", env=env
        )

        second = ensure_worktree(
            new_branch=None,
            worktrees_dir=None,
            branch="feat",
            existing_ok=True,
            cwd=project / "repo",
            env=env,
        )

        assert second.resolve() == first.resolve()
        assert len(_worktree_paths(project / "repo", env)) == 2

    def test_exits_when_the_worktree_already_exists(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        ensure_worktree(
            new_branch="feat", worktrees_dir=None, branch=None, cwd=project / "repo", env=env
        )

        with pytest.raises(SystemExit):
            ensure_worktree(
                new_branch=None,
                worktrees_dir=None,
                branch="feat",
                existing_ok=False,
                cwd=project / "repo",
                env=env,
            )

    def test_ignores_worktrees_belonging_to_other_branches(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        ensure_worktree(
            new_branch="other", worktrees_dir=None, branch=None, cwd=project / "repo", env=env
        )

        result = ensure_worktree(
            new_branch="feat", worktrees_dir=None, branch=None, cwd=project / "repo", env=env
        )

        assert _head_branch(result, env) == "feat"
        assert len(_worktree_paths(project / "repo", env)) == 3

    def test_reuse_existing_branch_checks_out_the_branch_that_is_there(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env, branches=["feat-a"])
        repo = project / "repo"
        _git("branch", "--track", "feat-a", "origin/feat-a", cwd=repo, env=env)

        result = ensure_worktree(
            new_branch="feat-a",
            worktrees_dir=None,
            branch=None,
            reuse_existing_branch=True,
            cwd=repo,
            env=env,
        )

        # The branch's own commit is present: it was reused, not recreated.
        assert (result / "feat-a.txt").is_file()

    def test_reuse_existing_branch_still_creates_a_missing_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        repo = project / "repo"

        result = ensure_worktree(
            new_branch="brand-new",
            worktrees_dir=None,
            branch=None,
            reuse_existing_branch=True,
            cwd=repo,
            env=env,
        )

        assert _head_branch(result, env) == "brand-new"
        assert "brand-new" in _local_branches(repo, env)

    def test_reuse_existing_branch_accepts_a_worktree_that_is_already_there(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        repo = project / "repo"
        first = ensure_worktree(
            new_branch="feat", worktrees_dir=None, branch=None, cwd=repo, env=env
        )

        second = ensure_worktree(
            new_branch="feat",
            worktrees_dir=None,
            branch=None,
            reuse_existing_branch=True,
            cwd=repo,
            env=env,
        )

        assert second.resolve() == first.resolve()

    def test_starts_a_new_branch_at_the_requested_start_point(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env, branches=["feat-a"])
        repo = project / "repo"

        result = ensure_worktree(
            new_branch="derived",
            worktrees_dir=None,
            branch=None,
            start_point="feat-a",
            cwd=repo,
            env=env,
        )

        assert _head_branch(result, env) == "derived"
        assert (result / "feat-a.txt").is_file()

    def test_a_new_branch_without_a_start_point_begins_at_the_checked_out_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env, branches=["feat-a"])
        repo = project / "repo"

        result = ensure_worktree(
            new_branch="derived", worktrees_dir=None, branch=None, cwd=repo, env=env
        )

        assert not (result / "feat-a.txt").exists()
        assert (
            _git("rev-parse", "derived", cwd=repo, env=env).strip()
            == _git("rev-parse", "main", cwd=repo, env=env).strip()
        )

    def test_uses_an_explicit_absolute_worktrees_directory(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        custom = tmp_path / "custom"

        result = ensure_worktree(
            new_branch="feat",
            worktrees_dir=str(custom),
            branch=None,
            cwd=project / "repo",
            env=env,
        )

        assert result.resolve() == (custom / "feat").resolve()
        assert _head_branch(result, env) == "feat"

    def test_resolves_a_relative_worktrees_directory_against_the_current_directory(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        repo = project / "repo"

        result = ensure_worktree(
            new_branch="feat",
            worktrees_dir="custom_worktrees",
            branch=None,
            cwd=repo,
            env=env,
        )

        assert result.is_absolute()
        assert result.resolve() == (repo / "custom_worktrees" / "feat").resolve()
        assert _head_branch(result, env) == "feat"

    def test_copies_the_project_config_into_the_new_worktree(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        (project / "config" / ".env").write_text("SHARED=1\n", encoding="utf-8")

        result = ensure_worktree(
            new_branch="feat", worktrees_dir=None, branch=None, cwd=project / "repo", env=env
        )

        assert (result / ".env").read_text(encoding="utf-8") == "SHARED=1\n"

    def test_works_outside_an_agm_project(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = _plain_repo(tmp_path, env)

        result = ensure_worktree(
            new_branch="feat", worktrees_dir=None, branch=None, cwd=repo, env=env
        )

        # No project, so the worktrees directory defaults next to the checkout.
        assert result.resolve() == (repo / "worktrees" / "feat").resolve()
        assert _head_branch(result, env) == "feat"


class TestEnsureWorktreeRemoteBranches:
    """Checking out branches that exist only on a remote, including forks."""

    def _repo(
        self, tmp_path: Path, env: dict[str, str], *, on_origin: bool = False
    ) -> tuple[Path, Path]:
        repo = clone_with_fork_remote(
            tmp_path, env, branch="branch-x", on_origin=on_origin, on_fork=True
        )
        return repo, tmp_path / "worktrees"

    def test_checks_out_branch_carried_only_by_a_non_origin_remote(
        self, tmp_path: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo, worktrees = self._repo(tmp_path, env)

        result = ensure_worktree(
            new_branch=None,
            worktrees_dir=str(worktrees),
            branch="branch-x",
            existing_ok=True,
            cwd=repo,
            env=env,
        )
        capsys.readouterr()

        assert result == worktrees / "branch-x"
        # The workspace holds the fork's commit, not a fresh branch off main.
        assert (result / "fork.txt").exists()
        assert _upstream(repo, "branch-x", env) == "fork/branch-x"

    def test_exits_when_the_branch_is_carried_by_several_remotes(
        self, tmp_path: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo, worktrees = self._repo(tmp_path, env, on_origin=True)

        with pytest.raises(SystemExit) as exc_info:
            ensure_worktree(
                new_branch=None,
                worktrees_dir=str(worktrees),
                branch="branch-x",
                existing_ok=True,
                cwd=repo,
                env=env,
            )

        assert exc_info.value.code == 1
        assert not (worktrees / "branch-x").exists()
        err = capsys.readouterr().err
        assert "fork" in err
        assert "origin" in err


class TestRemoveWorktree:
    """``remove_worktree`` takes the checkout away and, by default, the branch too."""

    def _project_with_worktree(
        self, tmp_path: Path, env: dict[str, str], branch: str = "feat"
    ) -> tuple[Path, Path]:
        project = _make_project(tmp_path, env)
        worktree = ensure_worktree(
            new_branch=branch, worktrees_dir=None, branch=None, cwd=project / "repo", env=env
        )
        return project, worktree

    def test_removes_the_worktree_and_deletes_its_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project, worktree = self._project_with_worktree(tmp_path, env)
        repo = project / "repo"

        remove_worktree(repo_dir=repo, force=False, branch="feat", env=env)

        assert not worktree.exists()
        assert _worktree_paths(repo, env) == [repo.resolve()]
        assert _local_branches(repo, env) == ["main"]

    def test_can_keep_the_branch(self, tmp_path: Path, env: dict[str, str]) -> None:
        project, worktree = self._project_with_worktree(tmp_path, env)
        repo = project / "repo"

        remove_worktree(repo_dir=repo, force=False, branch="feat", delete_branch=False, env=env)

        assert not worktree.exists()
        assert _local_branches(repo, env) == ["feat", "main"]

    def test_refuses_to_remove_a_dirty_worktree_without_force(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project, worktree = self._project_with_worktree(tmp_path, env)
        (worktree / "scratch.txt").write_text("uncommitted\n", encoding="utf-8")

        with pytest.raises(SystemExit):
            remove_worktree(repo_dir=project / "repo", force=False, branch="feat", env=env)

        assert worktree.is_dir()

    def test_force_removes_a_dirty_worktree(self, tmp_path: Path, env: dict[str, str]) -> None:
        project, worktree = self._project_with_worktree(tmp_path, env)
        (worktree / "scratch.txt").write_text("uncommitted\n", encoding="utf-8")

        remove_worktree(repo_dir=project / "repo", force=True, branch="feat", env=env)

        assert not worktree.exists()
        assert _local_branches(project / "repo", env) == ["main"]

    def test_refuses_to_delete_an_unmerged_branch_without_force_delete(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project, worktree = self._project_with_worktree(tmp_path, env)
        (worktree / "work.txt").write_text("work\n", encoding="utf-8")
        _git("add", ".", cwd=worktree, env=env)
        _git("commit", "-m", "work", "-q", cwd=worktree, env=env)

        with pytest.raises(SystemExit):
            remove_worktree(repo_dir=project / "repo", force=False, branch="feat", env=env)

        assert "feat" in _local_branches(project / "repo", env)

    def test_force_delete_removes_an_unmerged_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project, worktree = self._project_with_worktree(tmp_path, env)
        (worktree / "work.txt").write_text("work\n", encoding="utf-8")
        _git("add", ".", cwd=worktree, env=env)
        _git("commit", "-m", "work", "-q", cwd=worktree, env=env)

        remove_worktree(
            repo_dir=project / "repo", force=False, branch="feat", force_delete=True, env=env
        )

        assert not worktree.exists()
        assert _local_branches(project / "repo", env) == ["main"]

    def test_exits_when_no_worktree_exists_for_the_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project = _make_project(tmp_path, env)
        repo = project / "repo"
        _git("branch", "feat", cwd=repo, env=env)

        with pytest.raises(SystemExit):
            remove_worktree(repo_dir=repo, force=False, branch="feat", env=env)

        assert "feat" in _local_branches(repo, env)

    def test_exits_when_other_worktrees_exist_but_none_match(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project, worktree = self._project_with_worktree(tmp_path, env, branch="other")
        repo = project / "repo"

        with pytest.raises(SystemExit):
            remove_worktree(repo_dir=repo, force=False, branch="missing", env=env)

        assert worktree.is_dir()

    def test_exits_when_the_branch_is_the_main_workspace(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        project, worktree = self._project_with_worktree(tmp_path, env)

        with pytest.raises(SystemExit):
            remove_worktree(repo_dir=project / "repo", force=False, branch="main", env=env)

        assert worktree.is_dir()

    def test_works_outside_an_agm_project(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = _plain_repo(tmp_path, env)
        worktree = ensure_worktree(
            new_branch="feat", worktrees_dir=None, branch=None, cwd=repo, env=env
        )

        remove_worktree(repo_dir=repo, force=False, branch="feat", env=env)

        assert not worktree.exists()
        assert _local_branches(repo, env) == ["main"]
