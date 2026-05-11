"""Tests for agm.commands.fetch."""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.commands.fetch as fetch_cmd


class TestFetchRepo:
    """Tests for the _fetch_repo helper."""

    def test_prints_dot_for_project_dir_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        fetched: list[Path] = []
        synced: list[Path] = []

        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(
            fetch_cmd, "sync_remote_tracking_branches", lambda p: synced.append(p)
        )

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

        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: None)
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd._fetch_repo(project_dir, repo_dir)

        captured = capsys.readouterr()
        assert "Fetching repo" in captured.out

    def test_prints_absolute_path_when_outside_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()

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

        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(
            fetch_cmd, "sync_remote_tracking_branches", lambda p: synced.append(p)
        )

        fetch_cmd._fetch_repo(project_dir, repo_dir)

        assert fetched == [repo_dir]
        assert synced == [repo_dir]


class TestFetchRun:
    """Tests for the fetch run() entrypoint."""

    def test_exits_when_repo_dir_does_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(fetch_cmd, "is_dir", lambda p: False)
        monkeypatch.setattr(fetch_cmd, "is_git_repo", lambda p: False)

        with pytest.raises(SystemExit):
            fetch_cmd.run(object())

    def test_exits_when_repo_dir_is_not_git_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(fetch_cmd, "is_dir", lambda p: True)
        monkeypatch.setattr(fetch_cmd, "is_git_repo", lambda p: False)

        with pytest.raises(SystemExit):
            fetch_cmd.run(object())

    def test_fetches_main_repo_only_when_no_deps_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(fetch_cmd, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(fetch_cmd, "is_dir", lambda p: p == repo_dir)
        monkeypatch.setattr(fetch_cmd, "is_git_repo", lambda p: True)
        monkeypatch.setattr(fetch_cmd, "iterdir", lambda p: [])

        fetched: list[Path] = []
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd.run(object())

        assert repo_dir in fetched

    def test_fetches_deps_when_deps_dir_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        deps_dir = project_dir / "deps"
        dep1_dir = deps_dir / "libfoo"
        dep1_repo = dep1_dir / "main"

        def fake_is_dir(p: Path) -> bool:
            return p in {repo_dir, deps_dir, dep1_dir}

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(fetch_cmd, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(fetch_cmd, "project_deps_dir", lambda pd: deps_dir)
        monkeypatch.setattr(fetch_cmd, "is_dir", fake_is_dir)
        monkeypatch.setattr(fetch_cmd, "is_git_repo", lambda p: True)
        monkeypatch.setattr(fetch_cmd, "iterdir", lambda p: [dep1_dir])
        monkeypatch.setattr(fetch_cmd, "find_first_git_repo", lambda p: dep1_repo)

        fetched: list[Path] = []
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
        not_a_dir = deps_dir / "somefile"

        def fake_is_dir(p: Path) -> bool:
            return p in {repo_dir, deps_dir}

        monkeypatch.setattr(fetch_cmd, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(fetch_cmd, "project_repo_dir", lambda pd: repo_dir)
        monkeypatch.setattr(fetch_cmd, "project_deps_dir", lambda pd: deps_dir)
        monkeypatch.setattr(fetch_cmd, "is_dir", fake_is_dir)
        monkeypatch.setattr(fetch_cmd, "is_git_repo", lambda p: True)
        monkeypatch.setattr(fetch_cmd, "iterdir", lambda p: [not_a_dir])

        fetched: list[Path] = []
        monkeypatch.setattr(fetch_cmd, "fetch_prune_all", lambda p: fetched.append(p))
        monkeypatch.setattr(fetch_cmd, "sync_remote_tracking_branches", lambda p: None)

        fetch_cmd.run(object())

        # Only main repo was fetched; dep entry was skipped (not a dir)
        assert fetched == [repo_dir]