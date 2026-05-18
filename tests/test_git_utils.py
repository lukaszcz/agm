"""Comprehensive tests for agm.vcs.git."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.vcs.git import (
    WorktreeInfo,
    _branch_upstream,
    _git_args,
    _is_ancestor,
    branch_can_delete,
    branch_delete,
    checkout_root,
    containing_root,
    create_tracking_branch,
    current_branch,
    default_remote_branch_ref,
    exact_repo_root,
    fetch,
    fetch_output,
    fetch_prune_all,
    fetch_prune_origin,
    find_first_git_repo,
    git_common_dir,
    has_staged_changes,
    is_git_repo,
    local_branch_exists,
    local_branches,
    ls_remote_head,
    merge,
    remote_branch_exists,
    remote_unmerged_branches,
    repo_name_from_url,
    symbolic_ref,
    worktree_add,
    worktree_list,
    worktree_remove,
)

# ---------------------------------------------------------------------------
# _git_args — pure function, no mocking needed
# ---------------------------------------------------------------------------


class TestGitArgs:
    def test_no_repo_dir_returns_plain_git(self) -> None:
        assert _git_args() == ["git"]

    def test_none_returns_plain_git(self) -> None:
        assert _git_args(None) == ["git"]

    def test_repo_dir_returns_git_with_C_flag(self, tmp_path: Path) -> None:
        result = _git_args(tmp_path)
        assert result == ["git", "-C", str(tmp_path)]

    def test_repo_dir_string_representation(self, tmp_path: Path) -> None:
        repo = tmp_path / "my-repo"
        result = _git_args(repo)
        assert result[2] == str(repo)


class TestGenericGitProbeHelpers:
    def test_containing_root_returns_none_for_missing_path(self, tmp_path: Path) -> None:
        assert containing_root(tmp_path / "missing") is None

    def test_containing_root_returns_none_when_git_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.run_capture", lambda *_a, **_kw: (1, "", ""))

        assert containing_root(tmp_path) is None

    def test_containing_root_returns_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.run_capture", lambda *_a, **_kw: (0, f"{tmp_path}\n", "")
        )

        assert containing_root(tmp_path) == tmp_path

    def test_exact_repo_root_requires_exact_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent = tmp_path / "repo"
        child = parent / "sub"
        child.mkdir(parents=True)
        monkeypatch.setattr("agm.vcs.git.containing_root", lambda *_a, **_kw: parent)

        assert exact_repo_root(child) is None
        assert exact_repo_root(parent) == parent

    def test_exact_repo_root_returns_none_without_containing_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.containing_root", lambda *_a, **_kw: None)

        assert exact_repo_root(tmp_path) is None

    def test_has_staged_changes_handles_git_status_codes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.run_capture", lambda *_a, **_kw: (1, "", ""))

        assert has_staged_changes(tmp_path, [Path("config.toml")])

    def test_has_staged_changes_exits_on_unexpected_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.run_capture", lambda *_a, **_kw: (128, "", "fatal"))

        with pytest.raises(SystemExit):
            has_staged_changes(tmp_path, [Path("config.toml")])

    def test_repo_name_from_url_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            repo_name_from_url("/")


# ---------------------------------------------------------------------------
# is_git_repo
# ---------------------------------------------------------------------------


class TestIsGitRepo:
    def test_returns_true_when_run_capture_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (0, "true\n", ""),
        )
        assert is_git_repo(tmp_path) is True

    def test_returns_false_when_run_capture_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (128, "", "fatal: not a git repository"),
        )
        assert is_git_repo(tmp_path) is False

    def test_passes_correct_git_command(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_run_capture(cmd: list[str], **kwargs: object) -> tuple[int, str, str]:
            captured.append(cmd)
            return (0, "true\n", "")

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        is_git_repo(tmp_path)
        assert captured[0] == ["git", "-C", str(tmp_path), "rev-parse", "--is-inside-work-tree"]


# ---------------------------------------------------------------------------
# checkout_root
# ---------------------------------------------------------------------------


class TestCheckoutRoot:
    def test_returns_toplevel_when_cwd_is_git_repo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_root = tmp_path / "myrepo"
        repo_root.mkdir()

        def fake_run_capture(cmd: list[str], **kwargs: object) -> tuple[int, str, str]:
            if "--is-inside-work-tree" in cmd:
                return (0, "true\n", "")
            return (0, "", "")

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: str(repo_root) + "\n",
        )
        result = checkout_root(cwd=tmp_path)
        assert result == repo_root

    def test_falls_back_to_repo_subdir_when_cwd_is_not_git(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo_subdir = tmp_path / "repo"
        repo_subdir.mkdir()
        repo_root = tmp_path / "actual-root"
        repo_root.mkdir()

        call_count = 0

        def fake_run_capture(cmd: list[str], **kwargs: object) -> tuple[int, str, str]:
            nonlocal call_count
            call_count += 1
            # First call: cwd itself → not a git repo
            # Second call: repo/ subdir → is a git repo
            if str(tmp_path) in cmd and str(repo_subdir) not in cmd:
                return (128, "", "not a git repo")
            return (0, "true\n", "")

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: str(repo_root) + "\n",
        )
        result = checkout_root(cwd=tmp_path)
        assert result == repo_root

    def test_exits_when_neither_cwd_nor_repo_subdir_is_git(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # No repo/ subdir exists; both is_git_repo checks fail
        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (128, "", "not a git repo"),
        )
        with pytest.raises(SystemExit) as exc_info:
            checkout_root(cwd=tmp_path)
        assert exc_info.value.code == 1

    def test_exits_when_cwd_not_git_and_repo_subdir_not_git(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Create repo/ subdir so the dir exists but it's not a git repo
        repo_subdir = tmp_path / "repo"
        repo_subdir.mkdir()

        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (128, "", "not a git repo"),
        )
        with pytest.raises(SystemExit) as exc_info:
            checkout_root(cwd=tmp_path)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# git_common_dir
# ---------------------------------------------------------------------------


class TestGitCommonDir:
    def test_returns_path_from_require_capture(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        common = tmp_path / ".git"
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: str(common) + "\n",
        )
        result = git_common_dir(cwd=tmp_path)
        assert result == common

    def test_passes_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_require_capture(cmd: list[str], **kwargs: object) -> str:
            captured.append(cmd)
            return str(tmp_path / ".git") + "\n"

        monkeypatch.setattr("agm.vcs.git.require_capture", fake_require_capture)
        git_common_dir(cwd=tmp_path)
        assert "--git-common-dir" in captured[0]
        assert "--path-format=absolute" in captured[0]


# ---------------------------------------------------------------------------
# fetch / fetch_prune_all / fetch_prune_origin
# ---------------------------------------------------------------------------


class TestFetch:
    def test_fetch_calls_require_success_with_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        fetch(tmp_path)
        assert captured[0] == ["git", "-C", str(tmp_path), "fetch"]

    def test_fetch_passes_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        received_env: list[dict[str, str] | None] = []

        def fake_require_success(cmd: list[str], **kwargs: object) -> None:
            received_env.append(kwargs.get("env"))  # type: ignore[arg-type]

        monkeypatch.setattr("agm.vcs.git.require_success", fake_require_success)
        custom_env = {"MY_VAR": "value"}
        fetch(tmp_path, env=custom_env)
        assert received_env[0] == custom_env

    def test_fetch_prune_all_passes_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        fetch_prune_all(tmp_path)
        assert captured[0] == ["git", "-C", str(tmp_path), "fetch", "--all", "--prune"]

    def test_fetch_prune_origin_passes_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        fetch_prune_origin(tmp_path)
        assert captured[0] == ["git", "-C", str(tmp_path), "fetch", "--prune", "origin"]


class TestMerge:
    def test_merge_calls_require_success_with_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        merge(tmp_path)
        assert captured[0] == ["git", "-C", str(tmp_path), "merge"]

    def test_merge_passes_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        received_env: list[dict[str, str] | None] = []

        def fake_require_success(cmd: list[str], **kwargs: object) -> None:
            del cmd
            env = kwargs.get("env")
            assert env is None or isinstance(env, dict)
            received_env.append(env)

        monkeypatch.setattr("agm.vcs.git.require_success", fake_require_success)
        custom_env = {"TOKEN": "abc"}
        merge(tmp_path, env=custom_env)
        assert received_env[0] == custom_env


# ---------------------------------------------------------------------------
# current_branch
# ---------------------------------------------------------------------------


class TestCurrentBranch:
    def test_returns_stripped_branch_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "main\n",
        )
        assert current_branch(tmp_path) == "main"

    def test_passes_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_require_capture(cmd: list[str], **kwargs: object) -> str:
            captured.append(cmd)
            return "feature\n"

        monkeypatch.setattr("agm.vcs.git.require_capture", fake_require_capture)
        current_branch(tmp_path)
        assert captured[0] == [
            "git",
            "-C",
            str(tmp_path),
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ]

    def test_passes_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        received_env: list[dict[str, str] | None] = []

        def fake_require_capture(cmd: list[str], **kwargs: object) -> str:
            received_env.append(kwargs.get("env"))  # type: ignore[arg-type]
            return "main\n"

        monkeypatch.setattr("agm.vcs.git.require_capture", fake_require_capture)
        custom_env = {"TOKEN": "abc"}
        current_branch(tmp_path, env=custom_env)
        assert received_env[0] == custom_env


# ---------------------------------------------------------------------------
# local_branches
# ---------------------------------------------------------------------------


class TestLocalBranches:
    def test_returns_sorted_non_empty_lines(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "main\nfeature\ndevelop\n",
        )
        result = local_branches(tmp_path)
        assert result == ["develop", "feature", "main"]

    def test_filters_empty_lines(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "\nmain\n\nfeature\n",
        )
        result = local_branches(tmp_path)
        assert result == ["feature", "main"]

    def test_returns_empty_list_for_no_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "",
        )
        result = local_branches(tmp_path)
        assert result == []

    def test_passes_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_require_capture(cmd: list[str], **kwargs: object) -> str:
            captured.append(cmd)
            return ""

        monkeypatch.setattr("agm.vcs.git.require_capture", fake_require_capture)
        local_branches(tmp_path)
        assert captured[0] == [
            "git",
            "-C",
            str(tmp_path),
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads",
        ]


# ---------------------------------------------------------------------------
# worktree_add
# ---------------------------------------------------------------------------


class TestWorktreeAdd:
    def test_add_existing_branch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        worktree_add(tmp_path, tmp_path / "wt", "main")
        assert captured[0] == [
            "git",
            "-C",
            str(tmp_path),
            "worktree",
            "add",
            str(tmp_path / "wt"),
            "main",
        ]

    def test_add_with_create_no_start_point(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        worktree_add(tmp_path, tmp_path / "wt", "new-branch", create=True)
        assert captured[0] == [
            "git",
            "-C",
            str(tmp_path),
            "worktree",
            "add",
            "-b",
            "new-branch",
            str(tmp_path / "wt"),
        ]

    def test_add_with_create_and_start_point(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        worktree_add(
            tmp_path,
            tmp_path / "wt",
            "new-branch",
            create=True,
            start_point="origin/main",
        )
        assert captured[0] == [
            "git",
            "-C",
            str(tmp_path),
            "worktree",
            "add",
            "-b",
            "new-branch",
            str(tmp_path / "wt"),
            "origin/main",
        ]


# ---------------------------------------------------------------------------
# worktree_remove
# ---------------------------------------------------------------------------


class TestWorktreeRemove:
    def test_remove_without_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        worktree_remove(tmp_path, tmp_path / "wt")
        assert captured[0] == [
            "git",
            "-C",
            str(tmp_path),
            "worktree",
            "remove",
            str(tmp_path / "wt"),
        ]

    def test_remove_with_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        worktree_remove(tmp_path, tmp_path / "wt", force=True)
        assert "--force" in captured[0]
        assert captured[0] == [
            "git",
            "-C",
            str(tmp_path),
            "worktree",
            "remove",
            "--force",
            str(tmp_path / "wt"),
        ]


# ---------------------------------------------------------------------------
# worktree_list — porcelain parsing
# ---------------------------------------------------------------------------


class TestWorktreeList:
    def test_parses_single_worktree_with_branch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        porcelain = "worktree /repo\nbranch refs/heads/main\n\n"
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: porcelain,
        )
        result = worktree_list(tmp_path)
        assert len(result) == 1
        assert result[0].path == Path("/repo")
        assert result[0].branch == "main"

    def test_parses_multiple_worktrees(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        porcelain = (
            "worktree /repo\nbranch refs/heads/main\n\n"
            "worktree /repo-wt/feature\nbranch refs/heads/feature\n\n"
        )
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: porcelain,
        )
        result = worktree_list(tmp_path)
        assert len(result) == 2
        assert result[0] == WorktreeInfo(path=Path("/repo"), branch="main")
        assert result[1] == WorktreeInfo(path=Path("/repo-wt/feature"), branch="feature")

    def test_parses_worktree_without_branch_detached_head(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        porcelain = "worktree /repo\nHEAD abc123\ndetached\n\n"
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: porcelain,
        )
        result = worktree_list(tmp_path)
        assert len(result) == 1
        assert result[0].path == Path("/repo")
        assert result[0].branch is None

    def test_parses_last_worktree_without_trailing_blank_line(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # No trailing blank line after last entry
        porcelain = "worktree /repo\nbranch refs/heads/main\n"
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: porcelain,
        )
        result = worktree_list(tmp_path)
        assert len(result) == 1
        assert result[0].branch == "main"

    def test_strips_refs_heads_prefix_from_branch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        porcelain = "worktree /repo\nbranch refs/heads/my-feature\n\n"
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: porcelain,
        )
        result = worktree_list(tmp_path)
        assert result[0].branch == "my-feature"

    def test_returns_empty_list_for_empty_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "",
        )
        result = worktree_list(tmp_path)
        assert result == []

    def test_passes_porcelain_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_require_capture(cmd: list[str], **kwargs: object) -> str:
            captured.append(cmd)
            return ""

        monkeypatch.setattr("agm.vcs.git.require_capture", fake_require_capture)
        worktree_list(tmp_path)
        assert "--porcelain" in captured[0]
        assert "worktree" in captured[0]
        assert "list" in captured[0]


# ---------------------------------------------------------------------------
# branch_delete
# ---------------------------------------------------------------------------


class TestBranchDelete:
    def test_calls_require_success_with_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        branch_delete(tmp_path, "old-branch")
        assert captured[0] == ["git", "-C", str(tmp_path), "branch", "-d", "old-branch"]

    def test_uses_D_flag_when_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        branch_delete(tmp_path, "old-branch", force=True)
        assert captured[0] == ["git", "-C", str(tmp_path), "branch", "-D", "old-branch"]


class TestBranchUpstream:
    def test_returns_upstream_name_when_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (0, "origin/main\n", ""),
        )
        result = _branch_upstream(tmp_path, "feature")
        assert result == "origin/main"

    def test_returns_none_when_no_upstream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (128, "", "fatal: no upstream"),
        )
        result = _branch_upstream(tmp_path, "feature")
        assert result is None

    def test_returns_none_when_empty_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (0, "\n", ""),
        )
        result = _branch_upstream(tmp_path, "feature")
        assert result is None


class TestIsAncestor:
    def test_returns_true_when_ancestor(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.run_foreground",
            lambda cmd, **kwargs: 0,
        )
        assert _is_ancestor(tmp_path, "main", "feature") is True

    def test_returns_false_when_not_ancestor(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.run_foreground",
            lambda cmd, **kwargs: 1,
        )
        assert _is_ancestor(tmp_path, "feature", "main") is False

    def test_passes_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_run_foreground(cmd: list[str], **kwargs: object) -> int:
            captured.append(cmd)
            return 0

        monkeypatch.setattr("agm.vcs.git.run_foreground", fake_run_foreground)
        _is_ancestor(tmp_path, "a", "b")
        assert captured[0] == [
            "git",
            "-C",
            str(tmp_path),
            "merge-base",
            "--is-ancestor",
            "a",
            "b",
        ]


class TestBranchCanDelete:
    def test_returns_false_when_branch_not_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.local_branch_exists", lambda repo, b, env=None: False
        )
        result = branch_can_delete(tmp_path, "missing")
        assert result is False

    def test_returns_true_when_force_and_branch_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.local_branch_exists", lambda repo, b, env=None: True
        )
        result = branch_can_delete(tmp_path, "feature", force=True)
        assert result is True

    def test_returns_true_when_merged_into_upstream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.local_branch_exists", lambda repo, b, env=None: True
        )
        monkeypatch.setattr(
            "agm.vcs.git._branch_upstream", lambda repo, b, env=None: "origin/main"
        )
        monkeypatch.setattr(
            "agm.vcs.git._is_ancestor", lambda repo, a, d, env=None: True
        )
        result = branch_can_delete(tmp_path, "feature")
        assert result is True

    def test_returns_true_when_merged_into_head_no_upstream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.local_branch_exists", lambda repo, b, env=None: True
        )
        monkeypatch.setattr(
            "agm.vcs.git._branch_upstream", lambda repo, b, env=None: None
        )
        is_ancestor_calls: list[tuple[str, str]] = []

        def fake_is_ancestor(
            repo: Path, ancestor: str, descendant: str, *, env: object = None
        ) -> bool:
            is_ancestor_calls.append((ancestor, descendant))
            return True

        monkeypatch.setattr("agm.vcs.git._is_ancestor", fake_is_ancestor)
        result = branch_can_delete(tmp_path, "feature")
        assert result is True
        # Should check against HEAD when no upstream
        assert is_ancestor_calls == [("feature", "HEAD")]

    def test_returns_false_when_not_merged_into_upstream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.local_branch_exists", lambda repo, b, env=None: True
        )
        monkeypatch.setattr(
            "agm.vcs.git._branch_upstream",
            lambda repo, b, env=None: "origin/main",
        )
        monkeypatch.setattr(
            "agm.vcs.git._is_ancestor", lambda repo, a, d, env=None: False
        )
        result = branch_can_delete(tmp_path, "feature")
        assert result is False

    def test_uses_upstream_over_head_when_upstream_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.local_branch_exists", lambda repo, b, env=None: True
        )
        monkeypatch.setattr(
            "agm.vcs.git._branch_upstream",
            lambda repo, b, env=None: "origin/develop",
        )
        is_ancestor_calls: list[tuple[str, str]] = []

        def fake_is_ancestor(
            repo: Path, ancestor: str, descendant: str, *, env: object = None
        ) -> bool:
            is_ancestor_calls.append((ancestor, descendant))
            return True

        monkeypatch.setattr("agm.vcs.git._is_ancestor", fake_is_ancestor)
        branch_can_delete(tmp_path, "feature")
        assert is_ancestor_calls == [("feature", "origin/develop")]


# ---------------------------------------------------------------------------
# local_branch_exists / remote_branch_exists
# ---------------------------------------------------------------------------


class TestLocalBranchExists:
    def test_returns_true_when_branch_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.run_foreground", lambda cmd, **kwargs: 0)
        assert local_branch_exists(tmp_path, "main") is True

    def test_returns_false_when_branch_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.run_foreground", lambda cmd, **kwargs: 1)
        assert local_branch_exists(tmp_path, "missing") is False

    def test_passes_correct_ref_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_run_foreground(cmd: list[str], **kwargs: object) -> int:
            captured.append(cmd)
            return 0

        monkeypatch.setattr("agm.vcs.git.run_foreground", fake_run_foreground)
        local_branch_exists(tmp_path, "feature")
        assert "refs/heads/feature" in captured[0]
        assert "--verify" in captured[0]
        assert "--quiet" in captured[0]


class TestRemoteBranchExists:
    def test_returns_true_when_remote_branch_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.run_foreground", lambda cmd, **kwargs: 0)
        assert remote_branch_exists(tmp_path, "main") is True

    def test_returns_false_when_remote_branch_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.run_foreground", lambda cmd, **kwargs: 1)
        assert remote_branch_exists(tmp_path, "missing") is False

    def test_passes_correct_remote_ref_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_run_foreground(cmd: list[str], **kwargs: object) -> int:
            captured.append(cmd)
            return 0

        monkeypatch.setattr("agm.vcs.git.run_foreground", fake_run_foreground)
        remote_branch_exists(tmp_path, "feature")
        assert "refs/remotes/origin/feature" in captured[0]


# ---------------------------------------------------------------------------
# default_remote_branch_ref
# ---------------------------------------------------------------------------


class TestDefaultRemoteBranchRef:
    def test_returns_default_ref_when_symbolic_ref_non_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "origin/main\n",
        )
        result = default_remote_branch_ref(tmp_path)
        assert result == "origin/main"

    def test_exits_when_symbolic_ref_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "\n",
        )
        with pytest.raises(SystemExit) as exc_info:
            default_remote_branch_ref(tmp_path)
        assert exc_info.value.code == 1

    def test_exits_when_symbolic_ref_returns_whitespace_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "   \n",
        )
        with pytest.raises(SystemExit) as exc_info:
            default_remote_branch_ref(tmp_path)
        assert exc_info.value.code == 1

    def test_queries_origin_head_ref(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_require_capture(cmd: list[str], **kwargs: object) -> str:
            captured.append(cmd)
            return "origin/main\n"

        monkeypatch.setattr("agm.vcs.git.require_capture", fake_require_capture)
        default_remote_branch_ref(tmp_path)
        assert "refs/remotes/origin/HEAD" in captured[0]


# ---------------------------------------------------------------------------
# remote_unmerged_branches
# ---------------------------------------------------------------------------


class TestRemoteUnmergedBranches:
    def test_returns_non_empty_lines(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "origin/feature-a\norigin/feature-b\n",
        )
        result = remote_unmerged_branches(tmp_path, base_ref="origin/main")
        assert result == ["origin/feature-a", "origin/feature-b"]

    def test_filters_empty_lines(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "\norigin/branch\n\n",
        )
        result = remote_unmerged_branches(tmp_path, base_ref="origin/main")
        assert result == ["origin/branch"]

    def test_returns_empty_list_for_no_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "",
        )
        result = remote_unmerged_branches(tmp_path, base_ref="origin/main")
        assert result == []

    def test_passes_base_ref_as_no_merged_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_require_capture(cmd: list[str], **kwargs: object) -> str:
            captured.append(cmd)
            return ""

        monkeypatch.setattr("agm.vcs.git.require_capture", fake_require_capture)
        remote_unmerged_branches(tmp_path, base_ref="origin/develop")
        assert "--no-merged=origin/develop" in captured[0]
        assert "refs/remotes/origin" in captured[0]


# ---------------------------------------------------------------------------
# create_tracking_branch
# ---------------------------------------------------------------------------


class TestCreateTrackingBranch:
    def test_calls_require_success_with_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        create_tracking_branch(tmp_path, "my-branch", "origin/my-branch")
        assert captured[0] == [
            "git",
            "-C",
            str(tmp_path),
            "branch",
            "--track",
            "my-branch",
            "origin/my-branch",
        ]


# ---------------------------------------------------------------------------
# symbolic_ref
# ---------------------------------------------------------------------------


class TestSymbolicRef:
    def test_returns_stripped_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "origin/main\n",
        )
        result = symbolic_ref(tmp_path, "refs/remotes/origin/HEAD")
        assert result == "origin/main"

    def test_passes_ref_in_command(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def fake_require_capture(cmd: list[str], **kwargs: object) -> str:
            captured.append(cmd)
            return "origin/main\n"

        monkeypatch.setattr("agm.vcs.git.require_capture", fake_require_capture)
        symbolic_ref(tmp_path, "refs/remotes/origin/HEAD")
        assert "refs/remotes/origin/HEAD" in captured[0]
        assert "--quiet" in captured[0]
        assert "--short" in captured[0]
        assert "symbolic-ref" in captured[0]


# ---------------------------------------------------------------------------
# ls_remote_head
# ---------------------------------------------------------------------------


class TestLsRemoteHead:
    def test_returns_output_from_require_capture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = "ref: refs/heads/main\tHEAD\nabc123\tHEAD\n"
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: expected,
        )
        result = ls_remote_head("https://example.com/repo.git")
        assert result == expected

    def test_passes_correct_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[list[str]] = []

        def fake_require_capture(cmd: list[str], **kwargs: object) -> str:
            captured.append(cmd)
            return ""

        monkeypatch.setattr("agm.vcs.git.require_capture", fake_require_capture)
        ls_remote_head("git@github.com:org/repo.git")
        assert captured[0] == [
            "git",
            "ls-remote",
            "--symref",
            "git@github.com:org/repo.git",
            "HEAD",
        ]


# ---------------------------------------------------------------------------
# find_first_git_repo
# ---------------------------------------------------------------------------


class TestFindFirstGitRepo:
    def test_finds_first_git_repo_in_sorted_order(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        alpha = tmp_path / "alpha"
        beta = tmp_path / "beta"
        alpha.mkdir()
        beta.mkdir()

        # alpha is a git repo, beta is not
        def fake_run_capture(cmd: list[str], **kwargs: object) -> tuple[int, str, str]:
            path_arg = cmd[2]  # "git -C <path> rev-parse ..."
            if path_arg == str(alpha):
                return (0, "true\n", "")
            return (128, "", "not a git repo")

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        result = find_first_git_repo(tmp_path)
        assert result == alpha

    def test_returns_first_alphabetically_when_multiple_repos(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        aaa = tmp_path / "aaa"
        bbb = tmp_path / "bbb"
        aaa.mkdir()
        bbb.mkdir()

        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (0, "true\n", ""),
        )
        result = find_first_git_repo(tmp_path)
        assert result == aaa

    def test_exits_when_no_git_repo_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "notarepo").mkdir()
        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (128, "", "not a git repo"),
        )
        with pytest.raises(SystemExit) as exc_info:
            find_first_git_repo(tmp_path)
        assert exc_info.value.code == 1

    def test_exits_when_parent_dir_is_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (0, "true\n", ""),
        )
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(SystemExit) as exc_info:
            find_first_git_repo(empty_dir)
        assert exc_info.value.code == 1

    def test_searches_nested_directories(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        nested = tmp_path / "outer" / "inner"
        nested.mkdir(parents=True)

        def fake_run_capture(cmd: list[str], **kwargs: object) -> tuple[int, str, str]:
            if cmd[2] == str(nested):
                return (0, "true\n", "")
            return (128, "", "not a git repo")

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        result = find_first_git_repo(tmp_path)
        assert result == nested


# ---------------------------------------------------------------------------
# fetch_output
# ---------------------------------------------------------------------------


class TestFetchOutput:
    def test_delegates_to_run_capture(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        expected = (0, "stdout text", "stderr text")
        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: expected,
        )
        result = fetch_output(["git", "fetch"])
        assert result == expected

    def test_passes_cwd_and_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        received: dict[str, object] = {}

        def fake_run_capture(
            cmd: list[str],
            *,
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> tuple[int, str, str]:
            received["cwd"] = cwd
            received["env"] = env
            return (0, "", "")

        monkeypatch.setattr("agm.vcs.git.run_capture", fake_run_capture)
        custom_env = {"GIT_SSH": "/usr/bin/ssh"}
        fetch_output(["git", "fetch"], cwd=tmp_path, env=custom_env)
        assert received["cwd"] == tmp_path
        assert received["env"] == custom_env

    def test_returns_nonzero_exit_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (1, "", "error occurred"),
        )
        code, stdout, stderr = fetch_output(["git", "fetch"])
        assert code == 1
        assert stderr == "error occurred"


# ---------------------------------------------------------------------------
# WorktreeInfo dataclass
# ---------------------------------------------------------------------------


class TestWorktreeInfo:
    def test_frozen_dataclass_fields(self, tmp_path: Path) -> None:
        wt = WorktreeInfo(path=tmp_path, branch="main")
        assert wt.path == tmp_path
        assert wt.branch == "main"

    def test_branch_can_be_none(self, tmp_path: Path) -> None:
        wt = WorktreeInfo(path=tmp_path, branch=None)
        assert wt.branch is None

    def test_frozen_prevents_mutation(self, tmp_path: Path) -> None:
        wt = WorktreeInfo(path=tmp_path, branch="main")
        with pytest.raises(Exception):
            wt.branch = "other"  # type: ignore[misc]

    def test_equality(self, tmp_path: Path) -> None:
        a = WorktreeInfo(path=tmp_path, branch="main")
        b = WorktreeInfo(path=tmp_path, branch="main")
        assert a == b

    def test_inequality_different_branch(self, tmp_path: Path) -> None:
        a = WorktreeInfo(path=tmp_path, branch="main")
        b = WorktreeInfo(path=tmp_path, branch="develop")
        assert a != b
