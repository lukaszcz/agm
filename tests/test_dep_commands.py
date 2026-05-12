"""Comprehensive tests for agm.commands.dep.common, .remove, and .switch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agm.commands.dep.common as dep_common
import agm.commands.dep.remove as dep_remove
import agm.commands.dep.switch as dep_switch
from agm.commands.args import DepRemoveArgs, DepSwitchArgs
from agm.commands.dep.common import (
    derive_dep_name,
    main_dep_repo,
)
from agm.commands.dep.remove import (
    _linked_worktrees,
    _parse_target,
    _remove_dep_worktree_by_path,
    _worktree_at_path,
    _worktree_for_branch,
)
from agm.commands.dep.switch import _checkout_name, _existing_checkout_name
from agm.vcs.git import WorktreeInfo, default_branch_from_remote, default_branch_from_repo

# ---------------------------------------------------------------------------
# agm.commands.dep.common – derive_dep_name
# ---------------------------------------------------------------------------


class TestDeriveDependencyName:
    def test_plain_repo_name(self) -> None:
        assert derive_dep_name("mylib") == "mylib"

    def test_https_url(self) -> None:
        assert derive_dep_name("https://github.com/org/mylib") == "mylib"

    def test_https_url_with_git_suffix(self) -> None:
        assert derive_dep_name("https://github.com/org/mylib.git") == "mylib"

    def test_ssh_url_with_git_suffix(self) -> None:
        assert derive_dep_name("git@github.com:org/mylib.git") == "mylib"

    def test_ssh_url_without_git_suffix(self) -> None:
        assert derive_dep_name("git@github.com:org/mylib") == "mylib"

    def test_trailing_slash_stripped(self) -> None:
        assert derive_dep_name("https://github.com/org/mylib/") == "mylib"

    def test_multiple_trailing_slashes_stripped(self) -> None:
        assert derive_dep_name("https://github.com/org/mylib//") == "mylib"

    def test_git_suffix_removed_only_once(self) -> None:
        # .git.git should become .git, NOT strip twice
        assert derive_dep_name("https://github.com/org/mylib.git.git") == "mylib.git"

    def test_name_with_hyphens(self) -> None:
        assert derive_dep_name("https://github.com/org/my-lib.git") == "my-lib"

    def test_name_with_underscores(self) -> None:
        assert derive_dep_name("https://github.com/org/my_lib.git") == "my_lib"

    def test_empty_url_exits(self) -> None:
        with pytest.raises(SystemExit):
            derive_dep_name("")

    def test_root_slash_url_exits(self) -> None:
        with pytest.raises(SystemExit):
            derive_dep_name("/")

    def test_url_resolving_to_dot_exits(self) -> None:
        # A URL whose path component reduces to '.'
        with pytest.raises(SystemExit):
            derive_dep_name(".")

    def test_git_suffix_on_plain_name(self) -> None:
        assert derive_dep_name("mylib.git") == "mylib"

    def test_local_path(self) -> None:
        assert derive_dep_name("/home/user/repos/myproject") == "myproject"


# ---------------------------------------------------------------------------
# agm.commands.dep.common – default_branch_from_remote
# ---------------------------------------------------------------------------


class TestDefaultBranchFromRemote:
    def test_parses_ref_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run_capture(cmd: list[str], **_kwargs: Any) -> tuple[int, str, str]:
            output = "ref: refs/heads/main\tHEAD\nabc123\tHEAD\n"
            return 0, output, ""

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        assert default_branch_from_remote("https://github.com/org/repo") == "main"

    def test_parses_non_main_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run_capture(cmd: list[str], **_kwargs: Any) -> tuple[int, str, str]:
            output = "ref: refs/heads/develop\tHEAD\n"
            return 0, output, ""

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        assert default_branch_from_remote("https://github.com/org/repo") == "develop"

    def test_exits_on_nonzero_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run_capture(cmd: list[str], **_kwargs: Any) -> tuple[int, str, str]:
            return 1, "", "fatal: repository not found"

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        with pytest.raises(SystemExit):
            default_branch_from_remote("https://github.com/org/repo")

    def test_exits_when_no_ref_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run_capture(cmd: list[str], **_kwargs: Any) -> tuple[int, str, str]:
            output = "abc123\tHEAD\n"
            return 0, output, ""

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        with pytest.raises(SystemExit):
            default_branch_from_remote("https://github.com/org/repo")

    def test_exits_when_ref_line_has_no_parts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run_capture(cmd: list[str], **_kwargs: Any) -> tuple[int, str, str]:
            # "ref:" with no following token
            output = "ref:\n"
            return 0, output, ""

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        with pytest.raises(SystemExit):
            default_branch_from_remote("https://github.com/org/repo")


# ---------------------------------------------------------------------------
# agm.commands.dep.common – default_branch_from_repo
# ---------------------------------------------------------------------------


class TestDefaultBranchFromRepo:
    def test_returns_branch_without_origin_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run_capture(cmd: list[str], **_kwargs: Any) -> tuple[int, str, str]:
            return 0, "origin/main\n", ""

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        assert default_branch_from_repo(Path("/some/repo")) == "main"

    def test_returns_branch_without_prefix_already_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run_capture(cmd: list[str], **_kwargs: Any) -> tuple[int, str, str]:
            return 0, "main\n", ""

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        assert default_branch_from_repo(Path("/some/repo")) == "main"

    def test_exits_on_nonzero_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run_capture(cmd: list[str], **_kwargs: Any) -> tuple[int, str, str]:
            return 1, "", ""

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        with pytest.raises(SystemExit):
            default_branch_from_repo(Path("/some/repo"))

    def test_exits_on_empty_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run_capture(cmd: list[str], **_kwargs: Any) -> tuple[int, str, str]:
            return 0, "   \n", ""

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        with pytest.raises(SystemExit):
            default_branch_from_repo(Path("/some/repo"))


# ---------------------------------------------------------------------------
# agm.commands.dep.common – main_dep_repo
# ---------------------------------------------------------------------------


class TestMainDepRepo:
    def test_returns_first_git_repo_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_dir = tmp_path / "mydep"
        repo_dir = dep_dir / "repo"
        repo_dir.mkdir(parents=True)
        (repo_dir / ".git").mkdir()

        monkeypatch.setattr(dep_common.git_helpers, "is_git_repo", lambda p: p == repo_dir)
        assert main_dep_repo(dep_dir) == repo_dir

    def test_returns_first_sorted_git_repo_among_many(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_dir = tmp_path / "mydep"
        repo_a = dep_dir / "alpha"
        repo_b = dep_dir / "beta"
        repo_a.mkdir(parents=True)
        (repo_a / ".git").mkdir()
        repo_b.mkdir(parents=True)
        (repo_b / ".git").mkdir()

        monkeypatch.setattr(
            dep_common.git_helpers, "is_git_repo", lambda p: p in {repo_a, repo_b}
        )
        # sorted order: alpha < beta → should return alpha
        assert main_dep_repo(dep_dir) == repo_a

    def test_exits_when_no_git_repo_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_dir = tmp_path / "mydep"
        subdir = dep_dir / "notrepo"
        subdir.mkdir(parents=True)

        monkeypatch.setattr(dep_common.git_helpers, "is_git_repo", lambda p: False)
        with pytest.raises(SystemExit):
            main_dep_repo(dep_dir)

    def test_exits_when_dep_dir_is_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_dir = tmp_path / "mydep"
        dep_dir.mkdir()

        monkeypatch.setattr(dep_common.git_helpers, "is_git_repo", lambda p: False)
        with pytest.raises(SystemExit):
            main_dep_repo(dep_dir)


# ---------------------------------------------------------------------------
# agm.commands.dep.remove – _parse_target
# ---------------------------------------------------------------------------


class TestParseTarget:
    def test_dep_slash_branch(self) -> None:
        dep, ref = _parse_target("mylib/feature", remove_all=False)
        assert dep == "mylib"
        assert ref == "feature"

    def test_dep_slash_branch_with_nested_slash(self) -> None:
        dep, ref = _parse_target("mylib/feat/x", remove_all=False)
        assert dep == "mylib"
        assert ref == "feat/x"

    def test_remove_all_returns_dep_with_none_ref(self) -> None:
        dep, ref = _parse_target("mylib", remove_all=True)
        assert dep == "mylib"
        assert ref is None

    def test_missing_branch_without_all_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_target("mylib", remove_all=False)

    def test_dep_slash_no_ref_exits(self) -> None:
        # "mylib/" has sep but no ref
        with pytest.raises(SystemExit):
            _parse_target("mylib/", remove_all=False)

    def test_remove_all_with_slash_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_target("mylib/branch", remove_all=True)

    def test_empty_dep_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_target("/branch", remove_all=False)

    def test_empty_string_exits(self) -> None:
        with pytest.raises(SystemExit):
            _parse_target("", remove_all=False)


# ---------------------------------------------------------------------------
# agm.commands.dep.remove – _worktree_at_path
# ---------------------------------------------------------------------------


class TestWorktreeAtPath:
    def test_finds_worktree_by_exact_path(self, tmp_path: Path) -> None:
        p = tmp_path / "wt1"
        p.mkdir()
        wt = WorktreeInfo(path=p, branch="main")
        result = _worktree_at_path([wt], p)
        assert result is wt

    def test_returns_none_when_no_match(self, tmp_path: Path) -> None:
        p = tmp_path / "wt1"
        p.mkdir()
        other = tmp_path / "wt2"
        other.mkdir()
        wt = WorktreeInfo(path=p, branch="main")
        assert _worktree_at_path([wt], other) is None

    def test_returns_none_for_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "wt1"
        p.mkdir()
        assert _worktree_at_path([], p) is None

    def test_finds_worktree_among_multiple(self, tmp_path: Path) -> None:
        p1 = tmp_path / "wt1"
        p1.mkdir()
        p2 = tmp_path / "wt2"
        p2.mkdir()
        wt1 = WorktreeInfo(path=p1, branch="feat-a")
        wt2 = WorktreeInfo(path=p2, branch="feat-b")
        result = _worktree_at_path([wt1, wt2], p2)
        assert result is wt2

    def test_returns_first_match_when_duplicates(self, tmp_path: Path) -> None:
        p = tmp_path / "wt1"
        p.mkdir()
        wt_a = WorktreeInfo(path=p, branch="a")
        wt_b = WorktreeInfo(path=p, branch="b")
        result = _worktree_at_path([wt_a, wt_b], p)
        assert result is wt_a


# ---------------------------------------------------------------------------
# agm.commands.dep.remove – _worktree_for_branch
# ---------------------------------------------------------------------------


class TestWorktreeForBranch:
    def test_finds_worktree_by_branch(self, tmp_path: Path) -> None:
        wt = WorktreeInfo(path=tmp_path / "wt", branch="feature")
        result = _worktree_for_branch([wt], "feature")
        assert result is wt

    def test_returns_none_when_no_branch_match(self, tmp_path: Path) -> None:
        wt = WorktreeInfo(path=tmp_path / "wt", branch="main")
        assert _worktree_for_branch([wt], "feature") is None

    def test_returns_none_for_empty_list(self) -> None:
        assert _worktree_for_branch([], "main") is None

    def test_returns_none_for_detached_worktree(self, tmp_path: Path) -> None:
        wt = WorktreeInfo(path=tmp_path / "wt", branch=None)
        assert _worktree_for_branch([wt], "main") is None

    def test_finds_correct_one_among_multiple(self, tmp_path: Path) -> None:
        wt1 = WorktreeInfo(path=tmp_path / "wt1", branch="alpha")
        wt2 = WorktreeInfo(path=tmp_path / "wt2", branch="beta")
        result = _worktree_for_branch([wt1, wt2], "beta")
        assert result is wt2


# ---------------------------------------------------------------------------
# agm.commands.dep.remove – _linked_worktrees
# ---------------------------------------------------------------------------


class TestLinkedWorktrees:
    def test_excludes_main_repo_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        linked = tmp_path / "linked-wt"
        linked.mkdir()

        main_wt = WorktreeInfo(path=repo_path, branch="main")
        linked_wt = WorktreeInfo(path=linked, branch="feature")

        monkeypatch.setattr(dep_remove.git_helpers, "worktree_list", lambda p: [main_wt, linked_wt])
        result = _linked_worktrees(repo_path=repo_path)
        assert result == [linked_wt]

    def test_returns_empty_when_only_main_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        main_wt = WorktreeInfo(path=repo_path, branch="main")

        monkeypatch.setattr(dep_remove.git_helpers, "worktree_list", lambda p: [main_wt])
        result = _linked_worktrees(repo_path=repo_path)
        assert result == []

    def test_returns_all_non_repo_worktrees(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        wt1 = tmp_path / "wt1"
        wt1.mkdir()
        wt2 = tmp_path / "wt2"
        wt2.mkdir()

        main_wt = WorktreeInfo(path=repo_path, branch="main")
        linked1 = WorktreeInfo(path=wt1, branch="feat-a")
        linked2 = WorktreeInfo(path=wt2, branch="feat-b")

        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_list", lambda p: [main_wt, linked1, linked2]
        )
        result = _linked_worktrees(repo_path=repo_path)
        assert result == [linked1, linked2]


# ---------------------------------------------------------------------------
# agm.commands.dep.remove – _remove_dep_worktree_by_path
# ---------------------------------------------------------------------------


class TestRemoveDepWorktreeByPath:
    def test_calls_worktree_remove_and_branch_delete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        wt_path = tmp_path / "wt"
        wt_path.mkdir()
        worktree = WorktreeInfo(path=wt_path, branch="feature")

        removed: list[Path] = []
        deleted: list[str] = []

        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_remove", lambda r, p: removed.append(p)
        )
        monkeypatch.setattr(
            dep_remove.git_helpers, "branch_delete", lambda r, b: deleted.append(b)
        )

        _remove_dep_worktree_by_path(repo_path=repo_path, worktree=worktree)

        assert removed == [wt_path]
        assert deleted == ["feature"]

    def test_skips_branch_delete_for_detached_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        wt_path = tmp_path / "wt"
        wt_path.mkdir()
        worktree = WorktreeInfo(path=wt_path, branch=None)

        removed: list[Path] = []
        deleted: list[str] = []

        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_remove", lambda r, p: removed.append(p)
        )
        monkeypatch.setattr(
            dep_remove.git_helpers, "branch_delete", lambda r, b: deleted.append(b)
        )

        _remove_dep_worktree_by_path(repo_path=repo_path, worktree=worktree)

        assert removed == [wt_path]
        assert deleted == []


# ---------------------------------------------------------------------------
# agm.commands.dep.remove – run
# ---------------------------------------------------------------------------


class TestDepRemoveRun:
    def _setup_dep_dir(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Create a minimal project + dep directory layout."""
        project_dir = tmp_path / "project"
        deps_dir = project_dir / "deps"
        dep_dir = deps_dir / "mylib"
        repo_path = dep_dir / "repo"
        repo_path.mkdir(parents=True)
        return project_dir, dep_dir, repo_path

    def test_remove_single_worktree_by_branch_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep_dir(tmp_path)

        wt_path = dep_dir / "feature"
        wt_path.mkdir()
        linked_wt = WorktreeInfo(path=wt_path, branch="feature")

        monkeypatch.setattr(dep_remove, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_remove, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_remove, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_list", lambda p: [
                WorktreeInfo(path=repo_path, branch="main"),
                linked_wt,
            ]
        )

        removed: list[Path] = []
        deleted: list[str] = []
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_remove", lambda r, p: removed.append(p)
        )
        monkeypatch.setattr(
            dep_remove.git_helpers, "branch_delete", lambda r, b: deleted.append(b)
        )

        args = DepRemoveArgs(all=False, target="mylib/feature")
        dep_remove.run(args)

        assert removed == [wt_path]
        assert deleted == ["feature"]

    def test_remove_all_removes_linked_worktrees_and_dep_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep_dir(tmp_path)

        wt_path = dep_dir / "feat-a"
        wt_path.mkdir()
        linked_wt = WorktreeInfo(path=wt_path, branch="feat-a")

        monkeypatch.setattr(dep_remove, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_remove, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_remove, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_list", lambda p: [
                WorktreeInfo(path=repo_path, branch="main"),
                linked_wt,
            ]
        )

        removed: list[Path] = []
        deleted: list[str] = []
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_remove", lambda r, p: removed.append(p)
        )
        monkeypatch.setattr(
            dep_remove.git_helpers, "branch_delete", lambda r, b: deleted.append(b)
        )

        rmtree_calls: list[Path] = []
        monkeypatch.setattr(dep_remove, "rmtree", lambda p: rmtree_calls.append(p))

        args = DepRemoveArgs(all=True, target="mylib")
        dep_remove.run(args)

        assert removed == [wt_path]
        assert deleted == ["feat-a"]
        assert rmtree_calls == [dep_dir]

    def test_remove_all_exits_on_detached_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep_dir(tmp_path)

        wt_path = dep_dir / "detached"
        wt_path.mkdir()
        detached_wt = WorktreeInfo(path=wt_path, branch=None)

        monkeypatch.setattr(dep_remove, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_remove, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_remove, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_list", lambda p: [
                WorktreeInfo(path=repo_path, branch="main"),
                detached_wt,
            ]
        )

        args = DepRemoveArgs(all=True, target="mylib")
        with pytest.raises(SystemExit):
            dep_remove.run(args)

    def test_exits_when_dep_dir_does_not_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        monkeypatch.setattr(dep_remove, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_remove, "project_deps_dir", lambda pd: project_dir / "deps")

        args = DepRemoveArgs(all=False, target="nonexistent/feature")
        with pytest.raises(SystemExit):
            dep_remove.run(args)

    def test_remove_repo_ref_exits_when_linked_worktrees_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep_dir(tmp_path)

        wt_path = dep_dir / "feature"
        wt_path.mkdir()
        linked_wt = WorktreeInfo(path=wt_path, branch="feature")

        monkeypatch.setattr(dep_remove, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_remove, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_remove, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_list", lambda p: [
                WorktreeInfo(path=repo_path, branch="main"),
                linked_wt,
            ]
        )

        args = DepRemoveArgs(all=False, target="mylib/repo")
        with pytest.raises(SystemExit):
            dep_remove.run(args)

    def test_remove_repo_ref_removes_dep_dir_when_no_linked_worktrees(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep_dir(tmp_path)

        monkeypatch.setattr(dep_remove, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_remove, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_remove, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_list", lambda p: [
                WorktreeInfo(path=repo_path, branch="main"),
            ]
        )

        rmtree_calls: list[Path] = []
        monkeypatch.setattr(dep_remove, "rmtree", lambda p: rmtree_calls.append(p))

        args = DepRemoveArgs(all=False, target="mylib/repo")
        dep_remove.run(args)

        assert rmtree_calls == [dep_dir]

    def test_exits_when_worktree_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep_dir(tmp_path)

        monkeypatch.setattr(dep_remove, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_remove, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_remove, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_list", lambda p: [
                WorktreeInfo(path=repo_path, branch="main"),
            ]
        )

        args = DepRemoveArgs(all=False, target="mylib/nonexistent")
        with pytest.raises(SystemExit):
            dep_remove.run(args)

    def test_remove_worktree_matched_by_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep_dir(tmp_path)

        # The worktree path matches dep_dir / "feat-a" exactly
        wt_path = dep_dir / "feat-a"
        wt_path.mkdir()
        linked_wt = WorktreeInfo(path=wt_path, branch="different-branch-name")

        monkeypatch.setattr(dep_remove, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_remove, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_remove, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_list", lambda p: [
                WorktreeInfo(path=repo_path, branch="main"),
                linked_wt,
            ]
        )

        removed: list[Path] = []
        deleted: list[str] = []
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_remove", lambda r, p: removed.append(p)
        )
        monkeypatch.setattr(
            dep_remove.git_helpers, "branch_delete", lambda r, b: deleted.append(b)
        )

        # Target is "feat-a" which matches by path
        args = DepRemoveArgs(all=False, target="mylib/feat-a")
        dep_remove.run(args)

        assert removed == [wt_path]
        assert deleted == ["different-branch-name"]


# ---------------------------------------------------------------------------
# agm.commands.dep.switch – _checkout_name
# ---------------------------------------------------------------------------


class TestCheckoutName:
    def test_returns_relative_posix_for_child(self, tmp_path: Path) -> None:
        dep_dir = tmp_path / "deps" / "mylib"
        dep_dir.mkdir(parents=True)
        child = dep_dir / "feature"
        child.mkdir()
        result = _checkout_name(dep_dir, child)
        assert result == "feature"

    def test_returns_nested_relative_posix(self, tmp_path: Path) -> None:
        dep_dir = tmp_path / "deps" / "mylib"
        dep_dir.mkdir(parents=True)
        nested = dep_dir / "group" / "feature"
        nested.mkdir(parents=True)
        result = _checkout_name(dep_dir, nested)
        assert result == "group/feature"

    def test_returns_none_for_dep_dir_itself(self, tmp_path: Path) -> None:
        dep_dir = tmp_path / "deps" / "mylib"
        dep_dir.mkdir(parents=True)
        assert _checkout_name(dep_dir, dep_dir) is None

    def test_returns_none_for_path_outside_dep_dir(self, tmp_path: Path) -> None:
        dep_dir = tmp_path / "deps" / "mylib"
        dep_dir.mkdir(parents=True)
        outside = tmp_path / "other"
        outside.mkdir()
        assert _checkout_name(dep_dir, outside) is None

    def test_returns_none_for_sibling_dep_dir(self, tmp_path: Path) -> None:
        dep_dir = tmp_path / "deps" / "mylib"
        dep_dir.mkdir(parents=True)
        sibling = tmp_path / "deps" / "otherlib"
        sibling.mkdir()
        assert _checkout_name(dep_dir, sibling) is None

    def test_single_component_name(self, tmp_path: Path) -> None:
        dep_dir = tmp_path / "deps" / "mylib"
        dep_dir.mkdir(parents=True)
        child = dep_dir / "main"
        child.mkdir()
        assert _checkout_name(dep_dir, child) == "main"


# ---------------------------------------------------------------------------
# agm.commands.dep.switch – _existing_checkout_name
# ---------------------------------------------------------------------------


class TestExistingCheckoutName:
    def test_returns_checkout_name_matched_by_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_dir = tmp_path / "deps" / "mylib"
        dep_dir.mkdir(parents=True)
        repo_path = dep_dir / "repo"
        repo_path.mkdir()
        wt_path = dep_dir / "feature"
        wt_path.mkdir()

        worktrees = [
            WorktreeInfo(path=repo_path, branch="main"),
            WorktreeInfo(path=wt_path, branch="feature"),
        ]
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)

        result = _existing_checkout_name(dep_dir=dep_dir, repo_path=repo_path, target="feature")
        assert result == "feature"

    def test_returns_checkout_name_matched_by_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_dir = tmp_path / "deps" / "mylib"
        dep_dir.mkdir(parents=True)
        repo_path = dep_dir / "repo"
        repo_path.mkdir()
        # The worktree path is different from what target resolves to
        wt_path = dep_dir / "checkout-dir"
        wt_path.mkdir()

        worktrees = [
            WorktreeInfo(path=repo_path, branch="main"),
            WorktreeInfo(path=wt_path, branch="feature"),
        ]
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)

        # target "feature" → target_path = dep_dir/"feature" (doesn't exist); branch matches
        result = _existing_checkout_name(dep_dir=dep_dir, repo_path=repo_path, target="feature")
        assert result == "checkout-dir"

    def test_returns_none_when_no_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_dir = tmp_path / "deps" / "mylib"
        dep_dir.mkdir(parents=True)
        repo_path = dep_dir / "repo"
        repo_path.mkdir()

        worktrees = [WorktreeInfo(path=repo_path, branch="main")]
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)

        result = _existing_checkout_name(dep_dir=dep_dir, repo_path=repo_path, target="nonexistent")
        assert result is None

    def test_path_match_takes_priority_over_branch_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_dir = tmp_path / "deps" / "mylib"
        dep_dir.mkdir(parents=True)
        repo_path = dep_dir / "repo"
        repo_path.mkdir()
        # wt1 has the branch name "feature" (branch match)
        wt1_path = dep_dir / "branch-checkout"
        wt1_path.mkdir()
        # wt2 is at the path dep_dir/"feature" (path match)
        wt2_path = dep_dir / "feature"
        wt2_path.mkdir()

        worktrees = [
            WorktreeInfo(path=repo_path, branch="main"),
            WorktreeInfo(path=wt1_path, branch="feature"),
            WorktreeInfo(path=wt2_path, branch="other-branch"),
        ]
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)

        # Path match (wt2_path == dep_dir/"feature") should be returned directly
        result = _existing_checkout_name(dep_dir=dep_dir, repo_path=repo_path, target="feature")
        assert result == "feature"

    def test_ignores_worktrees_outside_dep_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dep_dir = tmp_path / "deps" / "mylib"
        dep_dir.mkdir(parents=True)
        repo_path = dep_dir / "repo"
        repo_path.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        worktrees = [
            WorktreeInfo(path=repo_path, branch="main"),
            WorktreeInfo(path=outside, branch="feature"),
        ]
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)

        result = _existing_checkout_name(dep_dir=dep_dir, repo_path=repo_path, target="feature")
        assert result is None


# ---------------------------------------------------------------------------
# agm.commands.dep.switch – run
# ---------------------------------------------------------------------------


class TestDepSwitchRun:
    def _setup_dep(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        project_dir = tmp_path / "project"
        deps_dir = project_dir / "deps"
        dep_dir = deps_dir / "mylib"
        repo_path = dep_dir / "repo"
        repo_path.mkdir(parents=True)
        return project_dir, dep_dir, repo_path

    def test_switches_to_existing_checkout_updates_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep(tmp_path)

        existing_wt = dep_dir / "feature"
        existing_wt.mkdir()
        worktrees = [
            WorktreeInfo(path=repo_path, branch="main"),
            WorktreeInfo(path=existing_wt, branch="feature"),
        ]

        monkeypatch.setattr(dep_switch, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_switch, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_switch, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)
        monkeypatch.setattr(dep_switch, "current_config_branch", lambda pd: "main")

        config_updates: list[dict[str, str]] = []

        def fake_update_config(
            *, project_dir: Path, dep_name: str, dep_branch: str, config_branch: str
        ) -> None:
            config_updates.append(
                {"dep_name": dep_name, "dep_branch": dep_branch, "config_branch": config_branch}
            )

        monkeypatch.setattr(dep_switch, "update_dependency_config", fake_update_config)

        args = DepSwitchArgs(dep="mylib", branch="feature", create_branch=False)
        dep_switch.run(args)

        assert len(config_updates) == 1
        assert config_updates[0]["dep_name"] == "mylib"
        assert config_updates[0]["dep_branch"] == "feature"

    def test_exits_when_dep_dir_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        monkeypatch.setattr(dep_switch, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_switch, "project_deps_dir", lambda pd: project_dir / "deps")

        args = DepSwitchArgs(dep="nonexistent", branch="feature", create_branch=False)
        with pytest.raises(SystemExit):
            dep_switch.run(args)

    def test_exits_when_target_dir_already_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep(tmp_path)

        # No matching checkout in worktrees
        worktrees = [WorktreeInfo(path=repo_path, branch="main")]
        monkeypatch.setattr(dep_switch, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_switch, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_switch, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)

        # Create the target directory so "exists" returns True
        target = dep_dir / "new-branch"
        target.mkdir()

        args = DepSwitchArgs(dep="mylib", branch="new-branch", create_branch=False)
        with pytest.raises(SystemExit):
            dep_switch.run(args)

    def test_adds_worktree_for_existing_remote_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep(tmp_path)

        worktrees = [WorktreeInfo(path=repo_path, branch="main")]
        monkeypatch.setattr(dep_switch, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_switch, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_switch, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)
        monkeypatch.setattr(dep_switch, "current_config_branch", lambda pd: "main")

        fetched: list[Path] = []
        monkeypatch.setattr(dep_switch.git_helpers, "fetch", lambda p: fetched.append(p))

        worktree_add_calls: list[dict[str, object]] = []

        def fake_worktree_add(
            repo: Path, path: Path, branch: str, **kwargs: object
        ) -> None:
            worktree_add_calls.append({"path": path, "branch": branch, **kwargs})

        monkeypatch.setattr(dep_switch.git_helpers, "worktree_add", fake_worktree_add)

        config_updates: list[dict[str, str]] = []

        def fake_update_config(
            *, project_dir: Path, dep_name: str, dep_branch: str, config_branch: str
        ) -> None:
            config_updates.append({"dep_branch": dep_branch})

        monkeypatch.setattr(dep_switch, "update_dependency_config", fake_update_config)

        args = DepSwitchArgs(dep="mylib", branch="feature", create_branch=False)
        dep_switch.run(args)

        assert fetched == [repo_path]
        assert len(worktree_add_calls) == 1
        assert worktree_add_calls[0]["branch"] == "feature"
        assert worktree_add_calls[0]["path"] == dep_dir / "feature"
        assert config_updates[0]["dep_branch"] == "feature"

    def test_creates_new_branch_from_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep(tmp_path)

        worktrees = [WorktreeInfo(path=repo_path, branch="main")]
        monkeypatch.setattr(dep_switch, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_switch, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_switch, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)
        monkeypatch.setattr(dep_switch, "current_config_branch", lambda pd: "main")
        monkeypatch.setattr(dep_switch.git_helpers, "default_branch_from_repo", lambda p: "main")

        fetched: list[Path] = []
        monkeypatch.setattr(dep_switch.git_helpers, "fetch", lambda p: fetched.append(p))

        worktree_add_calls: list[dict[str, object]] = []

        def fake_worktree_add(
            repo: Path, path: Path, branch: str, **kwargs: object
        ) -> None:
            worktree_add_calls.append({"path": path, "branch": branch, **kwargs})

        monkeypatch.setattr(dep_switch.git_helpers, "worktree_add", fake_worktree_add)
        monkeypatch.setattr(dep_switch, "update_dependency_config", lambda **_kw: None)

        args = DepSwitchArgs(dep="mylib", branch="new-feat", create_branch=True)
        dep_switch.run(args)

        assert fetched == [repo_path]
        assert len(worktree_add_calls) == 1
        call = worktree_add_calls[0]
        assert call["branch"] == "new-feat"
        assert call.get("create") is True
        assert call.get("start_point") == "main"