"""Tests for agm.commands.pull."""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.commands.pull as pull_cmd
from agm.vcs.git import WorktreeInfo


class TestDisplayPath:
    def test_returns_dot_for_project_dir(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"

        assert pull_cmd._display_path(project_dir, project_dir) == "."

    def test_returns_absolute_path_for_external_path(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        external = tmp_path / "external"

        assert pull_cmd._display_path(project_dir, external) == str(external)


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

        pull_cmd.run(object())

        captured = capsys.readouterr()
        assert "Merging repo" in captured.out
        assert "Merging worktrees/feat" in captured.out
