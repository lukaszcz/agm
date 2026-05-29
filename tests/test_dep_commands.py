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
        monkeypatch.setattr(
            "agm.vcs.git.ls_remote_head",
            lambda repo_url, env=None: "ref: refs/heads/main\tHEAD\nabc123\tHEAD\n",
        )
        assert default_branch_from_remote("https://github.com/org/repo") == "main"

    def test_parses_non_main_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.ls_remote_head",
            lambda repo_url, env=None: "ref: refs/heads/develop\tHEAD\n",
        )
        assert default_branch_from_remote("https://github.com/org/repo") == "develop"

    def test_exits_on_nonzero_returncode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_ls_remote_head(repo_url: str, env: dict[str, str] | None = None) -> str:
            del repo_url, env
            raise SystemExit(1)

        monkeypatch.setattr("agm.vcs.git.ls_remote_head", fake_ls_remote_head)
        with pytest.raises(SystemExit):
            default_branch_from_remote("https://github.com/org/repo")

    def test_exits_when_no_ref_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.ls_remote_head",
            lambda repo_url, env=None: "abc123\tHEAD\n",
        )
        with pytest.raises(SystemExit):
            default_branch_from_remote("https://github.com/org/repo")

    def test_exits_when_ref_line_has_no_parts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.ls_remote_head",
            lambda repo_url, env=None: "ref:\n",
        )
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

        wt_path = dep_dir / "checkout-dir"
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

    def test_remove_single_worktree_with_nested_ref(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep_dir(tmp_path)

        wt_path = dep_dir / "feat" / "x"
        wt_path.mkdir(parents=True)
        linked_wt = WorktreeInfo(path=wt_path, branch="feat/x")

        monkeypatch.setattr(dep_remove, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_remove, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_remove, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(
            dep_remove.git_helpers,
            "worktree_list",
            lambda p: [
                WorktreeInfo(path=repo_path, branch="main"),
                linked_wt,
            ],
        )

        removed: list[Path] = []
        deleted: list[str] = []
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_remove", lambda r, p: removed.append(p)
        )
        monkeypatch.setattr(
            dep_remove.git_helpers, "branch_delete", lambda r, b: deleted.append(b)
        )

        dep_remove.run(DepRemoveArgs(all=False, target="mylib/feat/x"))

        assert removed == [wt_path]
        assert deleted == ["feat/x"]

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

        args = DepRemoveArgs(all=True, target="mylib")
        dep_remove.run(args)

        assert removed == [wt_path]
        assert deleted == ["feat-a"]
        assert not dep_dir.exists()

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

    @pytest.mark.parametrize(
        ("target", "remove_all", "message"),
        [
            ("mylib", False, "expected DEP/BRANCH"),
            ("mylib/", False, "expected DEP/BRANCH"),
            ("mylib/branch", True, "--all expects DEP"),
            ("/branch", False, "invalid dependency target"),
            ("", False, "invalid dependency target"),
        ],
    )
    def test_invalid_targets_exit_before_project_lookup(
        self,
        target: str,
        remove_all: bool,
        message: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fail_project_lookup() -> Path:
            raise AssertionError("invalid targets should fail before project lookup")

        monkeypatch.setattr(dep_remove, "require_current_project_dir", fail_project_lookup)

        with pytest.raises(SystemExit):
            dep_remove.run(DepRemoveArgs(all=remove_all, target=target))

        assert message in capsys.readouterr().err

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

        args = DepRemoveArgs(all=False, target="mylib/repo")
        dep_remove.run(args)

        assert not dep_dir.exists()

    def test_exits_when_worktree_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep_dir(tmp_path)
        other_path = dep_dir / "other-checkout"
        other_path.mkdir()

        monkeypatch.setattr(dep_remove, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_remove, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_remove, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_list", lambda p: [
                WorktreeInfo(path=repo_path, branch="main"),
                WorktreeInfo(path=other_path, branch="other"),
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

    def test_remove_detached_worktree_matched_by_path_skips_branch_delete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep_dir(tmp_path)

        wt_path = dep_dir / "detached"
        wt_path.mkdir()
        linked_wt = WorktreeInfo(path=wt_path, branch=None)

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
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_remove", lambda r, p: removed.append(p)
        )

        def fail_branch_delete(repo: Path, branch: str) -> None:
            raise AssertionError(f"detached worktree should not delete branch {branch}")

        monkeypatch.setattr(dep_remove.git_helpers, "branch_delete", fail_branch_delete)

        dep_remove.run(DepRemoveArgs(all=False, target="mylib/detached"))

        assert removed == [wt_path]

    def test_remove_worktree_matched_by_later_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep_dir(tmp_path)

        first_path = dep_dir / "first"
        second_path = dep_dir / "second"
        first_path.mkdir()
        second_path.mkdir()
        first_wt = WorktreeInfo(path=first_path, branch="first")
        second_wt = WorktreeInfo(path=second_path, branch="actual-second-branch")

        monkeypatch.setattr(dep_remove, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_remove, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_remove, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(
            dep_remove.git_helpers,
            "worktree_list",
            lambda p: [
                WorktreeInfo(path=repo_path, branch="main"),
                first_wt,
                second_wt,
            ],
        )

        removed: list[Path] = []
        deleted: list[str] = []
        monkeypatch.setattr(
            dep_remove.git_helpers, "worktree_remove", lambda r, p: removed.append(p)
        )
        monkeypatch.setattr(
            dep_remove.git_helpers, "branch_delete", lambda r, b: deleted.append(b)
        )

        dep_remove.run(DepRemoveArgs(all=False, target="mylib/second"))

        assert removed == [second_path]
        assert deleted == ["actual-second-branch"]


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

    def test_switches_to_nested_checkout_matched_by_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep(tmp_path)

        existing_wt = dep_dir / "group" / "feature"
        existing_wt.mkdir(parents=True)
        worktrees = [
            WorktreeInfo(path=repo_path, branch="main"),
            WorktreeInfo(path=existing_wt, branch="feature"),
        ]

        monkeypatch.setattr(dep_switch, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_switch, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_switch, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)
        monkeypatch.setattr(dep_switch, "current_config_branch", lambda pd: "main")

        config_updates: list[str] = []
        monkeypatch.setattr(
            dep_switch,
            "update_dependency_config",
            lambda **kw: config_updates.append(kw["dep_branch"]),
        )

        dep_switch.run(DepSwitchArgs(dep="mylib", branch="feature", create_branch=False))

        assert config_updates == ["group/feature"]

    def test_existing_checkout_path_takes_priority_over_branch_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep(tmp_path)

        branch_match = dep_dir / "branch-checkout"
        path_match = dep_dir / "feature"
        branch_match.mkdir()
        path_match.mkdir()
        worktrees = [
            WorktreeInfo(path=repo_path, branch="main"),
            WorktreeInfo(path=branch_match, branch="feature"),
            WorktreeInfo(path=path_match, branch="other-branch"),
        ]

        monkeypatch.setattr(dep_switch, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_switch, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_switch, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)
        monkeypatch.setattr(dep_switch, "current_config_branch", lambda pd: "main")

        config_updates: list[str] = []
        monkeypatch.setattr(
            dep_switch,
            "update_dependency_config",
            lambda **kw: config_updates.append(kw["dep_branch"]),
        )

        dep_switch.run(DepSwitchArgs(dep="mylib", branch="feature", create_branch=False))

        assert config_updates == ["feature"]

    def test_ignores_non_checkout_worktrees_before_creating_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir, dep_dir, repo_path = self._setup_dep(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        worktrees = [
            WorktreeInfo(path=repo_path, branch="main"),
            WorktreeInfo(path=dep_dir, branch="feature"),
            WorktreeInfo(path=outside, branch="feature"),
        ]

        monkeypatch.setattr(dep_switch, "require_current_project_dir", lambda: project_dir)
        monkeypatch.setattr(dep_switch, "project_deps_dir", lambda pd: project_dir / "deps")
        monkeypatch.setattr(dep_switch, "main_dep_repo", lambda d: repo_path)
        monkeypatch.setattr(dep_switch.git_helpers, "worktree_list", lambda p: worktrees)
        monkeypatch.setattr(dep_switch, "current_config_branch", lambda pd: "main")
        monkeypatch.setattr(dep_switch.git_helpers, "fetch", lambda p: None)
        monkeypatch.setattr(dep_switch, "update_dependency_config", lambda **kw: None)

        added: list[Path] = []

        def fake_worktree_add(repo: Path, path: Path, branch: str, **kwargs: object) -> None:
            added.append(path)

        monkeypatch.setattr(dep_switch.git_helpers, "worktree_add", fake_worktree_add)

        dep_switch.run(DepSwitchArgs(dep="mylib", branch="feature", create_branch=False))

        assert added == [dep_dir / "feature"]

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
