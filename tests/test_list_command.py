"""Tests for agm.commands.list."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from typer.main import get_command

import agm.cli as cli
import agm.commands.list as list_cmd
from agm.vcs.git import WorktreeInfo


def _invoke(runner: CliRunner, argv: list[str]) -> Any:
    return runner.invoke(get_command(cli.app), argv, prog_name="agm")


class TestListWorktrees:
    """Tests for list_worktrees."""

    def test_lists_main_repo_at_top(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        monkeypatch.setattr(
            list_cmd, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            list_cmd, "project_repo_dir", lambda pd: repo_dir
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "worktree_list", lambda p, env=None: []
        )
        monkeypatch.setattr(
            list_cmd, "current_checkout",
            lambda pd, cwd=None, env=None: None,
        )

        list_cmd.list_worktrees()

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 1
        assert "main" in lines[0]
        assert str(repo_dir) not in lines[0]

    def test_verbose_shows_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        monkeypatch.setattr(
            list_cmd, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            list_cmd, "project_repo_dir", lambda pd: repo_dir
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "worktree_list", lambda p, env=None: []
        )
        monkeypatch.setattr(
            list_cmd, "current_checkout",
            lambda pd, cwd=None, env=None: None,
        )

        list_cmd.list_worktrees(verbose=True)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 1
        assert "main" in lines[0]
        assert str(repo_dir) in lines[0]

    def test_lists_main_repo_and_branch_worktrees(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktrees_dir = project_dir / ".agm" / "worktrees"
        feat_path = worktrees_dir / "feat"
        fix_path = worktrees_dir / "fix"
        repo_dir.mkdir(parents=True)
        feat_path.mkdir(parents=True)
        fix_path.mkdir(parents=True)

        monkeypatch.setattr(
            list_cmd, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            list_cmd, "project_repo_dir", lambda pd: repo_dir
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            list_cmd.git_helpers,
            "worktree_list",
            lambda p, env=None: [
                WorktreeInfo(path=repo_dir, branch="main"),
                WorktreeInfo(path=feat_path, branch="feat"),
                WorktreeInfo(path=fix_path, branch="fix"),
            ],
        )
        monkeypatch.setattr(
            list_cmd, "current_checkout",
            lambda pd, cwd=None, env=None: None,
        )

        list_cmd.list_worktrees()

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 3
        assert "main" in lines[0]
        assert "feat" in lines[1]
        assert "fix" in lines[2]
        # Default (non-verbose) output should NOT contain directory paths
        assert str(repo_dir) not in lines[0]
        assert str(feat_path) not in lines[1]
        assert str(fix_path) not in lines[2]

    def test_verbose_lists_main_repo_and_branch_worktrees_with_dirs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktrees_dir = project_dir / ".agm" / "worktrees"
        feat_path = worktrees_dir / "feat"
        fix_path = worktrees_dir / "fix"
        repo_dir.mkdir(parents=True)
        feat_path.mkdir(parents=True)
        fix_path.mkdir(parents=True)

        monkeypatch.setattr(
            list_cmd, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            list_cmd, "project_repo_dir", lambda pd: repo_dir
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            list_cmd.git_helpers,
            "worktree_list",
            lambda p, env=None: [
                WorktreeInfo(path=repo_dir, branch="main"),
                WorktreeInfo(path=feat_path, branch="feat"),
                WorktreeInfo(path=fix_path, branch="fix"),
            ],
        )
        monkeypatch.setattr(
            list_cmd, "current_checkout",
            lambda pd, cwd=None, env=None: None,
        )

        list_cmd.list_worktrees(verbose=True)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 3
        assert "main" in lines[0]
        assert "feat" in lines[1]
        assert "fix" in lines[2]
        assert str(repo_dir) in lines[0]
        assert str(feat_path) in lines[1]
        assert str(fix_path) in lines[2]

    def test_marks_current_worktree_with_star(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        worktrees_dir = project_dir / ".agm" / "worktrees"
        feat_path = worktrees_dir / "feat"
        repo_dir.mkdir(parents=True)
        feat_path.mkdir(parents=True)

        from agm.project.layout import CurrentCheckout

        monkeypatch.setattr(
            list_cmd, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            list_cmd, "project_repo_dir", lambda pd: repo_dir
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            list_cmd.git_helpers,
            "worktree_list",
            lambda p, env=None: [
                WorktreeInfo(path=repo_dir, branch="main"),
                WorktreeInfo(path=feat_path, branch="feat"),
            ],
        )
        monkeypatch.setattr(
            list_cmd, "current_checkout",
            lambda pd, cwd=None, env=None: CurrentCheckout(
                checkout_dir=feat_path, branch="feat", is_main=False,
            ),
        )

        list_cmd.list_worktrees()

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 2
        # First line (main) should NOT have a star
        assert lines[0].startswith(" ")
        assert "main" in lines[0]
        # Second line (feat) SHOULD have a star
        assert lines[1].startswith("*")
        assert "feat" in lines[1]

    def test_marks_main_repo_as_current_when_on_main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        from agm.project.layout import CurrentCheckout

        monkeypatch.setattr(
            list_cmd, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            list_cmd, "project_repo_dir", lambda pd: repo_dir
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            list_cmd.git_helpers,
            "worktree_list",
            lambda p, env=None: [
                WorktreeInfo(path=repo_dir, branch="main"),
            ],
        )
        monkeypatch.setattr(
            list_cmd, "current_checkout",
            lambda pd, cwd=None, env=None: CurrentCheckout(
                checkout_dir=repo_dir, branch=None, is_main=True,
            ),
        )

        list_cmd.list_worktrees()

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 1
        assert lines[0].startswith("*")
        assert "main" in lines[0]

    def test_no_star_when_no_current_checkout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        monkeypatch.setattr(
            list_cmd, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            list_cmd, "project_repo_dir", lambda pd: repo_dir
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            list_cmd.git_helpers,
            "worktree_list",
            lambda p, env=None: [
                WorktreeInfo(path=repo_dir, branch="main"),
            ],
        )
        monkeypatch.setattr(
            list_cmd, "current_checkout",
            lambda pd, cwd=None, env=None: None,
        )

        list_cmd.list_worktrees()

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 1
        assert lines[0].startswith(" ")
        assert "main" in lines[0]

    def test_main_repo_always_shown_even_without_worktrees(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        monkeypatch.setattr(
            list_cmd, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            list_cmd, "project_repo_dir", lambda pd: repo_dir
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "current_branch", lambda p, env=None: "develop"
        )
        # No worktrees from git (including main repo not in the list)
        monkeypatch.setattr(
            list_cmd.git_helpers,
            "worktree_list",
            lambda p, env=None: [],
        )
        monkeypatch.setattr(
            list_cmd, "current_checkout",
            lambda pd, cwd=None, env=None: None,
        )

        list_cmd.list_worktrees()

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 1
        assert "develop" in lines[0]
        assert str(repo_dir) not in lines[0]

    def test_verbose_main_repo_always_shown_even_without_worktrees(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        monkeypatch.setattr(
            list_cmd, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            list_cmd, "project_repo_dir", lambda pd: repo_dir
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "current_branch", lambda p, env=None: "develop"
        )
        # No worktrees from git (including main repo not in the list)
        monkeypatch.setattr(
            list_cmd.git_helpers,
            "worktree_list",
            lambda p, env=None: [],
        )
        monkeypatch.setattr(
            list_cmd, "current_checkout",
            lambda pd, cwd=None, env=None: None,
        )

        list_cmd.list_worktrees(verbose=True)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 1
        assert "develop" in lines[0]
        assert str(repo_dir) in lines[0]

    def test_embedded_layout_uses_project_dir_as_repo(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Embedded layout: project_dir itself is the repo."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(
            list_cmd, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            list_cmd, "project_repo_dir", lambda pd: pd
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "worktree_list", lambda p, env=None: []
        )
        monkeypatch.setattr(
            list_cmd, "current_checkout",
            lambda pd, cwd=None, env=None: None,
        )

        list_cmd.list_worktrees(verbose=True)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line]
        assert len(lines) == 1
        assert "main" in lines[0]
        assert str(project_dir) in lines[0]


class TestRun:
    """Tests for the run() entrypoint."""

    def test_delegates_to_list_worktrees(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        project_dir = tmp_path / "proj"
        repo_dir = project_dir / "repo"
        repo_dir.mkdir(parents=True)

        monkeypatch.setattr(
            list_cmd, "require_current_project_dir", lambda cwd=None: project_dir
        )
        monkeypatch.setattr(
            list_cmd, "project_repo_dir", lambda pd: repo_dir
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "current_branch", lambda p, env=None: "main"
        )
        monkeypatch.setattr(
            list_cmd.git_helpers, "worktree_list", lambda p, env=None: []
        )
        monkeypatch.setattr(
            list_cmd, "current_checkout",
            lambda pd, cwd=None, env=None: None,
        )

        list_cmd.run()

        captured = capsys.readouterr()
        assert captured.out  # produces output


class TestIsCurrent:
    """Tests for _is_current helper."""

    def test_returns_false_when_current_dir_is_none(self, tmp_path: Path) -> None:
        assert list_cmd._is_current(tmp_path, None) is False

    def test_returns_true_when_paths_match(self, tmp_path: Path) -> None:
        assert list_cmd._is_current(tmp_path, tmp_path) is True

    def test_returns_false_when_paths_differ(self, tmp_path: Path) -> None:
        other = tmp_path / "other"
        other.mkdir()
        assert list_cmd._is_current(tmp_path, other) is False


class TestBranchSortKey:
    """Tests for _branch_sort_key helper."""

    def test_returns_branch_name(self, tmp_path: Path) -> None:
        wt = WorktreeInfo(path=tmp_path, branch="feat")
        assert list_cmd._branch_sort_key(wt) == "feat"

    def test_returns_empty_string_for_detached(self, tmp_path: Path) -> None:
        wt = WorktreeInfo(path=tmp_path, branch=None)
        assert list_cmd._branch_sort_key(wt) == ""


class TestListCommandViaCli:
    """Cover lines 1032-1034: list_cmd function body."""

    def test_list_cmd_via_cli(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = CliRunner()
        calls: list[object] = []

        def record(*, verbose: bool = False) -> None:
            calls.append(True)

        monkeypatch.setattr(cli.list_command, "run", record)
        result = _invoke(runner, ["list"])
        assert result.exit_code == 0
        assert len(calls) == 1

