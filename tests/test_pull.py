"""Tests for agm.commands.sync.pull."""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.commands.sync.pull as pull_cmd
from agm.vcs.git import WorktreeInfo


class TestMergeWorktree:
    def test_merges_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        merged: list[Path] = []
        monkeypatch.setattr(pull_cmd.git_helpers, "merge", lambda p: merged.append(p))

        pull_cmd._merge_worktree(project_dir, repo_dir)

        assert merged == [repo_dir]


class TestPullRun:
    """Tests for the pull run() entrypoint."""

    def test_fetches_before_merging_all_repo_and_dependency_worktrees(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktree_dir = project_dir / "worktrees" / "feat"
        dep_repo = project_dir / "deps" / "mylib" / "main"
        dep_worktree = project_dir / "deps" / "mylib" / "feat"

        events: list[tuple[str, Path | tuple[Path, ...]]] = []

        monkeypatch.setattr(pull_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(
            pull_cmd.fetch_command,
            "project_git_repos",
            lambda p: [repo_dir, dep_repo],
        )

        def fake_fetch(project: Path, repos: list[Path]) -> None:
            events.append(("fetch", tuple(repos)))

        def fake_worktree_list(repo_path: Path) -> list[WorktreeInfo]:
            if repo_path == repo_dir:
                return [
                    WorktreeInfo(path=repo_dir, branch="main"),
                    WorktreeInfo(path=worktree_dir, branch="feat"),
                ]
            if repo_path == dep_repo:
                return [
                    WorktreeInfo(path=dep_repo, branch="main"),
                    WorktreeInfo(path=dep_worktree, branch="feat"),
                ]
            raise AssertionError(f"unexpected repo path: {repo_path}")

        monkeypatch.setattr(pull_cmd.fetch_command, "fetch_project_repos", fake_fetch)
        monkeypatch.setattr(pull_cmd.git_helpers, "worktree_list", fake_worktree_list)
        monkeypatch.setattr(
            pull_cmd.git_helpers,
            "merge",
            lambda p: events.append(("merge", p)),
        )

        pull_cmd.run(object())

        assert events == [
            ("fetch", (repo_dir, dep_repo)),
            ("merge", repo_dir),
            ("merge", worktree_dir),
            ("merge", dep_repo),
            ("merge", dep_worktree),
        ]

    def test_prints_relative_merge_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktree_dir = project_dir / "worktrees" / "feat"

        monkeypatch.setattr(pull_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(pull_cmd.fetch_command, "project_git_repos", lambda p: [repo_dir])
        monkeypatch.setattr(pull_cmd.fetch_command, "fetch_project_repos", lambda p, r: None)
        monkeypatch.setattr(
            pull_cmd.git_helpers,
            "worktree_list",
            lambda p: [
                WorktreeInfo(path=repo_dir, branch="main"),
                WorktreeInfo(path=worktree_dir, branch="feat"),
            ],
        )
        monkeypatch.setattr(pull_cmd.git_helpers, "merge", lambda p: None)
        monkeypatch.chdir(tmp_path)

        pull_cmd.run(object())

        captured = capsys.readouterr()
        assert "Merging proj/repo" in captured.out
        assert "Merging proj/worktrees/feat" in captured.out


class TestPullRunEdgeCases:
    """Edge cases and failure scenarios for the pull run() entrypoint."""

    def test_merge_failure_propagates_and_stops_remaining_merges(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A merge failure raises SystemExit and stops merging remaining worktrees.

        We stub merge to raise SystemExit — the same exception require_success raises on
        a non-zero git exit.  Constructing a real merge conflict requires a git remote with
        conflicting branches, which is complex and fragile to set up deterministically.
        """
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktree_dir = project_dir / "worktrees" / "feat"

        monkeypatch.setattr(pull_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(pull_cmd.fetch_command, "project_git_repos", lambda p: [repo_dir])
        monkeypatch.setattr(pull_cmd.fetch_command, "fetch_project_repos", lambda p, r: None)
        monkeypatch.setattr(
            pull_cmd.git_helpers,
            "worktree_list",
            lambda p: [
                WorktreeInfo(path=repo_dir, branch="main"),
                WorktreeInfo(path=worktree_dir, branch="feat"),
            ],
        )

        merged: list[Path] = []

        def failing_merge(p: Path) -> None:
            merged.append(p)
            raise SystemExit(1)

        monkeypatch.setattr(pull_cmd.git_helpers, "merge", failing_merge)

        with pytest.raises(SystemExit):
            pull_cmd.run(object())

        # Exception stopped execution after the first attempted merge
        assert merged == [repo_dir]

    def test_only_main_worktree_completes_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A repo with only the main worktree (no linked extras) pulls without error."""
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"

        monkeypatch.setattr(pull_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(pull_cmd.fetch_command, "project_git_repos", lambda p: [repo_dir])
        monkeypatch.setattr(pull_cmd.fetch_command, "fetch_project_repos", lambda p, r: None)
        monkeypatch.setattr(
            pull_cmd.git_helpers,
            "worktree_list",
            lambda p: [WorktreeInfo(path=repo_dir, branch="main")],
        )

        merged: list[Path] = []
        monkeypatch.setattr(pull_cmd.git_helpers, "merge", lambda p: merged.append(p))

        pull_cmd.run(object())

        assert merged == [repo_dir]

    def test_missing_repo_surfaces_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the project has no git repo, pull propagates the SystemExit from discovery."""
        # No repo/ subdir and the project dir itself is not a git repo, so the REAL
        # project_git_repos discovery guard raises SystemExit — pull must not swallow it.
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(pull_cmd, "require_current_project_dir", lambda: project_dir)

        with pytest.raises(SystemExit):
            pull_cmd.run(object())
