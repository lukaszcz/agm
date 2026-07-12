"""Tests for agm.commands.sync.fetch."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import agm.commands.sync.fetch as fetch_cmd


def _init_git_repo(path: Path) -> None:
    """Initialize a minimal git repository at *path* (no commits needed)."""
    path.mkdir(parents=True, exist_ok=True)
    # Isolate from any ambient global/system git config (templates, hooks).
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True, env=env)


class TestFetchRepo:
    """Tests for the _fetch_repo helper."""

    def test_prints_dot_for_project_dir_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        fetched: list[Path] = []
        synced: list[Path] = []

        monkeypatch.setattr(fetch_cmd, "worktree_prune", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(
            fetch_cmd, "sync_remote_tracking_branches", lambda p: synced.append(p)
        )

        monkeypatch.chdir(project_dir)
        fetch_cmd._fetch_repo(project_dir, project_dir)

        captured = capsys.readouterr()
        assert "Fetching ." in captured.out
        assert fetched == [project_dir]
        assert synced == [project_dir]

    def test_prints_relative_path_for_repo_subdir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        monkeypatch.setattr(fetch_cmd, "worktree_prune", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        monkeypatch.chdir(tmp_path)
        fetch_cmd._fetch_repo(project_dir, repo_dir)

        captured = capsys.readouterr()
        assert "Fetching proj/repo" in captured.out

    def test_prints_absolute_path_when_outside_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        monkeypatch.setattr(fetch_cmd, "worktree_prune", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd._fetch_repo(project_dir, other_dir)

        captured = capsys.readouterr()
        assert str(other_dir) in captured.out

    def test_calls_fetch_prune_all_and_sync(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        fetched: list[Path] = []
        synced: list[Path] = []

        monkeypatch.setattr(fetch_cmd, "worktree_prune", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(
            fetch_cmd, "sync_remote_tracking_branches", lambda p: synced.append(p)
        )

        fetch_cmd._fetch_repo(project_dir, repo_dir)

        assert fetched == [repo_dir]
        assert synced == [repo_dir]

    def test_prunes_stale_worktrees(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_fetch_repo prunes stale worktree registrations so later steps never
        operate on a worktree directory git already considers gone."""
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        pruned: list[Path] = []

        monkeypatch.setattr(fetch_cmd, "worktree_prune", lambda p: pruned.append(p))
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd._fetch_repo(project_dir, repo_dir)

        assert pruned == [repo_dir]


class TestFetchProjectRepos:
    """Discovery tests exercise the real filesystem and real git; only network ops are mocked."""

    def test_returns_main_repo_only_without_deps_dir(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        _init_git_repo(repo_dir)
        # No deps/ directory — only the main repo should be returned

        assert fetch_cmd.project_git_repos(project_dir) == [repo_dir]

    def test_returns_main_repo_and_dependency_repos(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        dep1_repo = project_dir / "deps" / "libfoo" / "main"
        _init_git_repo(repo_dir)
        _init_git_repo(dep1_repo)
        # deps/libfoo/main is a real git repo; discovery finds it via find_first_git_repo

        assert fetch_cmd.project_git_repos(project_dir) == [repo_dir, dep1_repo]


class TestFetchProjectReposRunner:
    def test_fetches_each_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        dep_repo = project_dir / "deps" / "mylib" / "main"
        fetched: list[Path] = []

        monkeypatch.setattr(fetch_cmd, "_fetch_repo", lambda pd, repo: fetched.append(repo))

        fetch_cmd.fetch_project_repos(project_dir, [repo_dir, dep_repo])

        assert fetched == [repo_dir, dep_repo]


class TestFetchRun:
    """Tests for the fetch run() entrypoint.

    Discovery (is_dir, is_git_repo, iterdir, find_first_git_repo, project_repo_dir,
    project_deps_dir) runs against a real on-disk git tree.  Only the network-touching
    git operations (fetch_prune_all, sync_remote_tracking_branches, worktree_prune) are
    mocked because they require a configured remote.
    """

    def test_exits_when_repo_dir_does_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        # No repo/ subdir: project_repo_dir falls back to project_dir, which is not a git repo

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)

        with pytest.raises(SystemExit):
            fetch_cmd.run(object())

    def test_exits_when_repo_dir_is_not_git_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)
        # repo/ directory exists but was never git-initialised

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)

        with pytest.raises(SystemExit):
            fetch_cmd.run(object())

    def test_fetches_main_repo_only_when_no_deps_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        _init_git_repo(repo_dir)
        # No deps/ directory — only the main repo should be fetched

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        # Network-touching ops mocked; fs/git discovery uses the real tree
        fetched: list[Path] = []
        monkeypatch.setattr(fetch_cmd, "worktree_prune", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd.run(object())

        assert fetched == [repo_dir]

    def test_fetches_deps_when_deps_dir_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        dep1_repo = project_dir / "deps" / "libfoo" / "main"
        _init_git_repo(repo_dir)
        _init_git_repo(dep1_repo)

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        fetched: list[Path] = []
        monkeypatch.setattr(fetch_cmd, "worktree_prune", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd.run(object())

        assert repo_dir in fetched
        assert dep1_repo in fetched

    def test_skips_non_directory_entries_in_deps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        deps_dir = project_dir / "deps"
        _init_git_repo(repo_dir)
        deps_dir.mkdir()
        (deps_dir / "somefile").write_text("not a dir\n", encoding="utf-8")

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        fetched: list[Path] = []
        monkeypatch.setattr(fetch_cmd, "worktree_prune", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd.run(object())

        # Only the main repo was fetched; somefile is not a directory and is skipped
        assert fetched == [repo_dir]
