"""Comprehensive tests for agm.vcs.git."""

from __future__ import annotations

import shutil
import subprocess
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
    has_commits,
    has_staged_changes,
    is_git_repo,
    local_branch_exists,
    local_branches,
    ls_remote_head,
    merge,
    remote_branch_exists,
    remote_unmerged_branches,
    remotes_with_branch,
    repo_name_from_url,
    symbolic_ref,
    unique_remote_branch_ref,
    worktree_add,
    worktree_list,
    worktree_prune,
    worktree_remove,
)
from tests._git_helpers import (
    clone_local_remote,
    clone_with_fork_remote,
    commit_file,
    git_output,
    git_run,
    init_repo,
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


# ---------------------------------------------------------------------------
# containing_root / exact_repo_root
# ---------------------------------------------------------------------------


class TestContainingRoot:
    def test_returns_none_for_missing_path(self, tmp_path: Path) -> None:
        assert containing_root(tmp_path / "missing") is None

    def test_returns_none_for_plain_dir(self, tmp_path: Path) -> None:
        # A plain directory that is not inside any git repo → None.
        plain = tmp_path / "plain"
        plain.mkdir()
        assert containing_root(plain) is None

    def test_returns_repo_root(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        assert containing_root(repo, env=env) == repo

    def test_returns_repo_root_from_subdir(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        subdir = repo / "src"
        subdir.mkdir()
        assert containing_root(subdir, env=env) == repo


class TestExactRepoRoot:
    def test_returns_path_for_exact_repo_root(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        assert exact_repo_root(repo, env=env) == repo

    def test_returns_none_for_subdir_of_repo(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        subdir = repo / "src"
        subdir.mkdir()
        assert exact_repo_root(subdir, env=env) is None

    def test_returns_none_for_plain_dir(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        assert exact_repo_root(plain) is None


class TestGenericGitProbeHelpers:
    def test_has_commits_returns_false_for_unborn_repo(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main", "-q"], cwd=repo, env=env, check=True)

        assert has_commits(repo, env=env) is False

    def test_has_commits_returns_true_after_initial_commit(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)

        assert has_commits(repo, env=env) is True

    # has_staged_changes: the error-exit path (unexpected returncode → SystemExit)
    # is kept as a behavioral fake because triggering a real git error exit with
    # returncode ≥ 2 from `git diff --cached --quiet` requires contriving a broken
    # git state that is impractical in a deterministic test environment.
    def test_has_staged_changes_exits_on_unexpected_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.run_capture", lambda *_a, **_kw: (128, "", "fatal"))

        with pytest.raises(SystemExit):
            has_staged_changes(tmp_path, [Path("config.toml")])

    def test_has_staged_changes_returns_false_when_nothing_staged(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        # Nothing staged after the initial commit.
        assert has_staged_changes(repo, [repo / "README.md"], env=env) is False

    def test_has_staged_changes_returns_true_when_file_staged(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        new_file = repo / "new.txt"
        new_file.write_text("hello\n", encoding="utf-8")
        subprocess.run(["git", "add", "new.txt"], cwd=repo, env=env, check=True)
        assert has_staged_changes(repo, [new_file], env=env) is True

    def test_repo_name_from_url_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            repo_name_from_url("/")


# ---------------------------------------------------------------------------
# is_git_repo
# ---------------------------------------------------------------------------


class TestIsGitRepo:
    def test_returns_true_for_real_repo(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        assert is_git_repo(repo) is True

    def test_returns_false_for_plain_dir(self, tmp_path: Path) -> None:
        plain = tmp_path / "notarepo"
        plain.mkdir()
        assert is_git_repo(plain) is False


# ---------------------------------------------------------------------------
# checkout_root
# ---------------------------------------------------------------------------


class TestCheckoutRoot:
    def test_returns_toplevel_when_cwd_is_git_repo(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "myrepo"
        init_repo(repo, env)
        assert checkout_root(cwd=repo) == repo

    def test_returns_toplevel_from_subdir(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "myrepo"
        init_repo(repo, env)
        subdir = repo / "src"
        subdir.mkdir()
        assert checkout_root(cwd=subdir) == repo

    def test_falls_back_to_repo_subdir_when_cwd_is_not_git(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        # cwd is a plain dir that contains a "repo/" subdir that IS a git repo.
        project = tmp_path / "project"
        project.mkdir()
        init_repo(project / "repo", env)
        assert checkout_root(cwd=project) == project / "repo"

    def test_exits_when_neither_cwd_nor_repo_subdir_is_git(self, tmp_path: Path) -> None:
        plain = tmp_path / "notarepo"
        plain.mkdir()
        with pytest.raises(SystemExit) as exc_info:
            checkout_root(cwd=plain)
        assert exc_info.value.code == 1

    def test_exits_when_cwd_not_git_and_repo_subdir_not_git(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        (project / "repo").mkdir()  # repo/ exists but is not a git repo
        with pytest.raises(SystemExit) as exc_info:
            checkout_root(cwd=project)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# git_common_dir
# ---------------------------------------------------------------------------


class TestGitCommonDir:
    def test_returns_common_dir_for_main_repo(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        assert git_common_dir(cwd=repo) == repo / ".git"

    def test_returns_common_dir_from_worktree(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        wt = tmp_path / "wt"
        # Create a linked worktree; its --git-common-dir points to the main .git.
        subprocess.run(
            ["git", "worktree", "add", str(wt), "-b", "wt-branch"],
            cwd=repo,
            env=env,
            check=True,
        )
        assert git_common_dir(cwd=wt) == repo / ".git"


# ---------------------------------------------------------------------------
# fetch / fetch_prune_all / fetch_prune_origin
# ---------------------------------------------------------------------------


class TestFetch:
    def test_brings_remote_commits_into_the_tracking_ref(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        source, clone = clone_local_remote(tmp_path, env)
        commit_file(source, env, name="later.txt")
        source_head = git_output(source, ["rev-parse", "HEAD"], env)
        assert git_output(clone, ["rev-parse", "origin/main"], env) != source_head

        fetch(clone, env=env)

        assert git_output(clone, ["rev-parse", "origin/main"], env) == source_head

    def test_uses_the_supplied_environment(self, tmp_path: Path, env: dict[str, str]) -> None:
        # The remote is only reachable through a rewrite rule in the git config
        # of the HOME that *env* points at, so a fetch run under any other
        # environment cannot resolve the remote at all.
        source, clone = clone_local_remote(tmp_path, env)
        git_run(clone, ["remote", "set-url", "origin", "agm-remote/source"], env)
        Path(env["HOME"], ".gitconfig").write_text(
            f'[url "{tmp_path}/"]\n\tinsteadOf = "agm-remote/"\n', encoding="utf-8"
        )
        commit_file(source, env, name="later.txt")

        fetch(clone, env=env)

        assert git_output(clone, ["rev-parse", "origin/main"], env) == git_output(
            source, ["rev-parse", "HEAD"], env
        )

    def test_prune_all_drops_tracking_refs_deleted_on_every_remote(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = clone_with_fork_remote(
            tmp_path, env, branch="branch-x", on_origin=True, on_fork=True
        )
        assert remotes_with_branch(repo, "branch-x", env=env) == ["fork", "origin"]
        git_run(tmp_path / "origin.git", ["branch", "-D", "branch-x", "-q"], env)
        git_run(tmp_path / "fork.git", ["branch", "-D", "branch-x", "-q"], env)

        fetch_prune_all(repo, env=env)

        assert remotes_with_branch(repo, "branch-x", env=env) == []

    def test_prune_origin_leaves_the_other_remotes_alone(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = clone_with_fork_remote(
            tmp_path, env, branch="branch-x", on_origin=True, on_fork=True
        )
        git_run(tmp_path / "origin.git", ["branch", "-D", "branch-x", "-q"], env)
        git_run(tmp_path / "fork.git", ["branch", "-D", "branch-x", "-q"], env)

        fetch_prune_origin(repo, env=env)

        # Only origin was fetched, so only its stale tracking ref disappeared.
        assert remotes_with_branch(repo, "branch-x", env=env) == ["fork"]


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merges_upstream_branch(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)

        # Create a feature branch with one extra commit.
        subprocess.run(["git", "checkout", "-b", "feature", "-q"], cwd=repo, env=env, check=True)
        (repo / "feature.txt").write_text("feature\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, env=env, check=True)
        subprocess.run(
            ["git", "commit", "-m", "feature commit", "-q"], cwd=repo, env=env, check=True
        )

        # Back to main; configure feature as main's merge target so that
        # `git merge` (no args) knows what to merge.
        subprocess.run(["git", "checkout", "main", "-q"], cwd=repo, env=env, check=True)
        subprocess.run(
            ["git", "branch", "--set-upstream-to=feature", "main"],
            cwd=repo,
            env=env,
            check=True,
        )

        merge(repo, env=env)

        # Fast-forward: feature.txt is now present on main.
        assert (repo / "feature.txt").exists()


# ---------------------------------------------------------------------------
# current_branch
# ---------------------------------------------------------------------------


class TestCurrentBranch:
    def test_returns_main_branch_name(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        assert current_branch(repo, env=env) == "main"

    def test_returns_branch_after_checkout(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        subprocess.run(["git", "checkout", "-b", "feature", "-q"], cwd=repo, env=env, check=True)
        assert current_branch(repo, env=env) == "feature"


# ---------------------------------------------------------------------------
# local_branches
# ---------------------------------------------------------------------------


class TestLocalBranches:
    def test_returns_sorted_branches_from_real_repo(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        subprocess.run(["git", "branch", "develop"], cwd=repo, env=env, check=True)
        subprocess.run(["git", "branch", "feature"], cwd=repo, env=env, check=True)
        assert local_branches(repo, env=env) == ["develop", "feature", "main"]

    def test_returns_empty_list_for_repo_with_no_branches(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        # A freshly initialised repo with no commits has no branch refs yet.
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-b", "main", "-q"], cwd=repo, env=env, check=True)
        assert local_branches(repo, env=env) == []

    # Kept as a behavioral fake: real `git for-each-ref` never emits empty
    # lines, so the defensive `if line` guard can only be exercised by feeding
    # synthetic output directly.
    def test_filters_empty_lines(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "\nmain\n\nfeature\n",
        )
        result = local_branches(tmp_path)
        assert result == ["feature", "main"]


# ---------------------------------------------------------------------------
# worktree_add
# ---------------------------------------------------------------------------


class TestWorktreeAdd:
    def test_add_existing_branch_creates_worktree_on_disk(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        subprocess.run(["git", "branch", "feature"], cwd=repo, env=env, check=True)
        wt = tmp_path / "wt-feature"
        worktree_add(repo, wt, "feature", env=env)
        assert wt.is_dir()

    def test_add_with_create_creates_new_branch_and_worktree(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        wt = tmp_path / "wt-new"
        worktree_add(repo, wt, "new-branch", create=True, env=env)
        assert wt.is_dir()
        # The new branch must exist in the main repo.
        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", "refs/heads/new-branch"],
            cwd=repo,
            env=env,
        )
        assert result.returncode == 0

    def test_add_with_create_and_start_point_uses_start_point_content(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        # Create a develop branch with an extra file.
        subprocess.run(["git", "checkout", "-b", "develop", "-q"], cwd=repo, env=env, check=True)
        (repo / "dev.txt").write_text("dev\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, env=env, check=True)
        subprocess.run(["git", "commit", "-m", "dev commit", "-q"], cwd=repo, env=env, check=True)
        subprocess.run(["git", "checkout", "main", "-q"], cwd=repo, env=env, check=True)

        wt = tmp_path / "wt-from-develop"
        worktree_add(repo, wt, "new-branch", create=True, start_point="develop", env=env)

        # The worktree was created from develop, so dev.txt is present.
        assert wt.is_dir()
        assert (wt / "dev.txt").exists()

    def test_add_with_create_and_tag_start_point_uses_tag_content(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        (repo / "tagged.txt").write_text("tagged\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, env=env, check=True)
        subprocess.run(
            ["git", "commit", "-m", "tagged commit", "-q"], cwd=repo, env=env, check=True
        )
        subprocess.run(["git", "tag", "v1"], cwd=repo, env=env, check=True)
        (repo / "tagged.txt").unlink()
        subprocess.run(["git", "add", "."], cwd=repo, env=env, check=True)
        subprocess.run(
            ["git", "commit", "-m", "remove tagged file", "-q"],
            cwd=repo,
            env=env,
            check=True,
        )

        wt = tmp_path / "wt-from-tag"
        worktree_add(repo, wt, "new-branch", create=True, start_point="v1", env=env)

        assert wt.is_dir()
        assert current_branch(wt, env=env) == "new-branch"
        assert (wt / "tagged.txt").exists()

    def test_add_with_create_and_remote_only_start_point_keeps_new_branch_name(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        source = tmp_path / "source"
        init_repo(source, env)
        subprocess.run(
            ["git", "checkout", "-b", "cloud-native", "-q"], cwd=source, env=env, check=True
        )
        (source / "cloud.txt").write_text("cloud\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=source, env=env, check=True)
        subprocess.run(
            ["git", "commit", "-m", "cloud commit", "-q"], cwd=source, env=env, check=True
        )
        bare = tmp_path / "origin.git"
        subprocess.run(["git", "init", "--bare", str(bare)], env=env, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", str(bare)], cwd=source, env=env, check=True
        )
        subprocess.run(["git", "checkout", "main", "-q"], cwd=source, env=env, check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], cwd=source, env=env, check=True)
        subprocess.run(
            ["git", "push", "-u", "origin", "cloud-native"], cwd=source, env=env, check=True
        )
        subprocess.run(
            ["git", "symbolic-ref", "HEAD", "refs/heads/main"], cwd=bare, env=env, check=True
        )
        repo = tmp_path / "repo"
        subprocess.run(["git", "clone", str(bare), str(repo)], env=env, check=True)
        assert local_branch_exists(repo, "cloud-native", env=env) is False
        assert remote_branch_exists(repo, "cloud-native", env=env) is True

        wt = tmp_path / "wt-from-remote-parent"
        worktree_add(
            repo,
            wt,
            "tool-history-cloud-backend",
            create=True,
            start_point="cloud-native",
            env=env,
        )

        upstream_remote = subprocess.run(
            ["git", "config", "--get", "branch.tool-history-cloud-backend.remote"],
            cwd=repo,
            env=env,
            check=False,
        )
        upstream_merge = subprocess.run(
            ["git", "config", "--get", "branch.tool-history-cloud-backend.merge"],
            cwd=repo,
            env=env,
            check=False,
        )

        assert wt.is_dir()
        assert current_branch(wt, env=env) == "tool-history-cloud-backend"
        assert (wt / "cloud.txt").exists()
        assert upstream_remote.returncode != 0
        assert upstream_merge.returncode != 0

    def test_add_existing_remote_branch_tracks_its_remote(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        source = tmp_path / "source"
        init_repo(source, env)
        subprocess.run(
            ["git", "checkout", "-b", "existing-remote", "-q"],
            cwd=source,
            env=env,
            check=True,
        )
        (source / "remote.txt").write_text("remote\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=source, env=env, check=True)
        subprocess.run(
            ["git", "commit", "-m", "remote commit", "-q"], cwd=source, env=env, check=True
        )
        bare = tmp_path / "origin.git"
        subprocess.run(["git", "init", "--bare", str(bare)], env=env, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", str(bare)], cwd=source, env=env, check=True
        )
        subprocess.run(["git", "checkout", "main", "-q"], cwd=source, env=env, check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], cwd=source, env=env, check=True)
        subprocess.run(
            ["git", "push", "-u", "origin", "existing-remote"],
            cwd=source,
            env=env,
            check=True,
        )
        subprocess.run(
            ["git", "symbolic-ref", "HEAD", "refs/heads/main"], cwd=bare, env=env, check=True
        )
        repo = tmp_path / "repo"
        subprocess.run(["git", "clone", str(bare), str(repo)], env=env, check=True)
        assert local_branch_exists(repo, "existing-remote", env=env) is False
        assert remote_branch_exists(repo, "existing-remote", env=env) is True

        wt = tmp_path / "wt-from-existing-remote"
        worktree_add(repo, wt, "existing-remote", env=env)

        upstream_remote = subprocess.run(
            ["git", "config", "--get", "branch.existing-remote.remote"],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        upstream_merge = subprocess.run(
            ["git", "config", "--get", "branch.existing-remote.merge"],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        assert wt.is_dir()
        assert current_branch(wt, env=env) == "existing-remote"
        assert (wt / "remote.txt").exists()
        assert upstream_remote.stdout.strip() == "origin"
        assert upstream_merge.stdout.strip() == "refs/heads/existing-remote"


# ---------------------------------------------------------------------------
# worktree_remove
# ---------------------------------------------------------------------------


class TestWorktreeRemove:
    def test_remove_clean_worktree(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        wt = tmp_path / "wt"
        subprocess.run(
            ["git", "worktree", "add", str(wt), "-b", "wt-branch"],
            cwd=repo,
            env=env,
            check=True,
        )
        assert wt.exists()

        worktree_remove(repo, wt, env=env)

        assert not wt.exists()
        assert wt not in [info.path for info in worktree_list(repo, env=env)]

    def test_force_removes_dirty_worktree(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        wt = tmp_path / "wt"
        subprocess.run(
            ["git", "worktree", "add", str(wt), "-b", "wt-branch"],
            cwd=repo,
            env=env,
            check=True,
        )
        # Dirty the worktree by modifying a tracked file: a non-force removal
        # would refuse this, so success here proves --force was applied.
        (wt / "README.md").write_text("modified\n", encoding="utf-8")

        worktree_remove(repo, wt, force=True, env=env)

        assert not wt.exists()
        assert wt not in [info.path for info in worktree_list(repo, env=env)]


# ---------------------------------------------------------------------------
# worktree_prune
# ---------------------------------------------------------------------------


class TestWorktreePrune:
    def test_prune_removes_stale_worktree_registration(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        wt = tmp_path / "wt"
        subprocess.run(
            ["git", "worktree", "add", str(wt), "-b", "wt-branch"],
            cwd=repo,
            env=env,
            check=True,
        )
        # Delete the worktree directory behind git's back, leaving a stale
        # administrative registration that prune should drop.  We assert
        # against raw `git worktree list` output here rather than through
        # worktree_list(), because our helper already filters prunable
        # entries — only the raw state distinguishes "before prune" from
        # "after prune" and thus actually exercises worktree_prune.
        shutil.rmtree(wt)

        def raw_worktree_paths() -> str:
            return subprocess.run(
                ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            ).stdout

        assert str(wt) in raw_worktree_paths()

        worktree_prune(repo, env=env)

        assert str(wt) not in raw_worktree_paths()


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

    def test_blank_line_before_any_worktree_entry_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A blank line encountered before a 'worktree' line (path is None)
        must not append a spurious WorktreeInfo entry."""
        # Leading blank line followed by a real worktree block
        porcelain = "\nworktree /repo\nbranch refs/heads/main\n\n"
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: porcelain,
        )
        result = worktree_list(tmp_path)
        # Only one entry — the leading blank did not produce a spurious entry
        assert len(result) == 1
        assert result[0] == WorktreeInfo(path=Path("/repo"), branch="main")

    def test_skips_prunable_worktree(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Worktrees whose directory git reports as prunable (gitdir points to a
        non-existent location) must be omitted so callers never operate on a
        missing directory."""
        porcelain = (
            "worktree /repo\nbranch refs/heads/main\n\n"
            "worktree /repo-wt/gone\nbranch refs/heads/gone\n"
            "prunable gitdir file points to non-existent location\n\n"
            "worktree /repo-wt/feature\nbranch refs/heads/feature\n\n"
        )
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: porcelain,
        )
        result = worktree_list(tmp_path)
        assert result == [
            WorktreeInfo(path=Path("/repo"), branch="main"),
            WorktreeInfo(path=Path("/repo-wt/feature"), branch="feature"),
        ]

    def test_skips_prunable_last_worktree_without_trailing_blank_line(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A prunable entry that is the final block (no trailing blank line) must
        also be dropped rather than flushed by the end-of-output handling."""
        porcelain = (
            "worktree /repo\nbranch refs/heads/main\n\n"
            "worktree /repo-wt/gone\nbranch refs/heads/gone\n"
            "prunable gitdir file points to non-existent location\n"
        )
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: porcelain,
        )
        result = worktree_list(tmp_path)
        assert result == [WorktreeInfo(path=Path("/repo"), branch="main")]

    def test_lists_real_worktrees(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        wt = tmp_path / "wt-feature"
        subprocess.run(
            ["git", "worktree", "add", str(wt), "-b", "feature"],
            cwd=repo,
            env=env,
            check=True,
        )
        result = worktree_list(repo, env=env)
        by_path = {info.path: info.branch for info in result}
        assert by_path == {repo: "main", wt: "feature"}


# ---------------------------------------------------------------------------
# branch_delete
# ---------------------------------------------------------------------------


class TestBranchDelete:
    def test_deletes_merged_branch(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        # A branch with no commits beyond main is trivially merged, so the
        # safe `-d` deletion accepts it.
        subprocess.run(["git", "branch", "merged"], cwd=repo, env=env, check=True)
        assert local_branch_exists(repo, "merged", env=env) is True

        branch_delete(repo, "merged", env=env)

        assert local_branch_exists(repo, "merged", env=env) is False

    def test_force_deletes_unmerged_branch(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        # Create a branch with a commit that is not merged into main: a plain
        # `-d` deletion would refuse it, so success proves `-D` (force) ran.
        subprocess.run(["git", "checkout", "-b", "unmerged", "-q"], cwd=repo, env=env, check=True)
        (repo / "extra.txt").write_text("extra\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, env=env, check=True)
        subprocess.run(
            ["git", "commit", "-m", "unmerged commit", "-q"], cwd=repo, env=env, check=True
        )
        subprocess.run(["git", "checkout", "main", "-q"], cwd=repo, env=env, check=True)
        assert local_branch_exists(repo, "unmerged", env=env) is True

        branch_delete(repo, "unmerged", force=True, env=env)

        assert local_branch_exists(repo, "unmerged", env=env) is False


class TestBranchUpstream:
    def test_returns_upstream_of_a_tracking_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _source, clone = clone_local_remote(tmp_path, env)
        assert _branch_upstream(clone, "main", env=env) == "origin/main"

    def test_returns_none_when_the_branch_tracks_nothing(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = init_repo(tmp_path / "repo", env)
        assert _branch_upstream(repo, "main", env=env) is None

    # Kept as a behavioral fake: git either resolves ``@{upstream}`` or fails,
    # so a successful call returning empty output cannot be produced for real.
    def test_returns_none_when_output_is_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.run_capture",
            lambda cmd, **kwargs: (0, "\n", ""),
        )
        result = _branch_upstream(tmp_path, "feature")
        assert result is None


class TestIsAncestor:
    def test_reports_ancestry_in_one_direction_only(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = init_repo(tmp_path / "repo", env)
        git_run(repo, ["checkout", "-b", "feature", "-q"], env)
        commit_file(repo, env, name="feature.txt")

        assert _is_ancestor(repo, "main", "feature", env=env) is True
        assert _is_ancestor(repo, "feature", "main", env=env) is False


class TestBranchCanDelete:
    def test_absent_branch_is_never_deletable(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = init_repo(tmp_path / "repo", env)

        assert branch_can_delete(repo, "missing", env=env) is False
        assert branch_can_delete(repo, "missing", force=True, env=env) is False

    def test_force_accepts_a_branch_safe_deletion_refuses(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = init_repo(tmp_path / "repo", env)
        git_run(repo, ["checkout", "-b", "unmerged", "-q"], env)
        commit_file(repo, env, name="unmerged.txt")
        git_run(repo, ["checkout", "main", "-q"], env)

        assert branch_can_delete(repo, "unmerged", env=env) is False
        assert branch_can_delete(repo, "unmerged", force=True, env=env) is True

    def test_branch_merged_into_head_without_upstream_is_deletable(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = init_repo(tmp_path / "repo", env)
        git_run(repo, ["branch", "merged"], env)
        assert _branch_upstream(repo, "merged", env=env) is None

        assert branch_can_delete(repo, "merged", env=env) is True

    def test_branch_merged_into_its_upstream_is_deletable_though_head_is_behind(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        source, clone = clone_local_remote(tmp_path, env)
        commit_file(source, env, name="later.txt")
        git_run(clone, ["fetch", "-q"], env)
        # feature sits on the advanced origin/main it tracks, while the checked
        # out main still points at the older commit.
        git_run(clone, ["branch", "--track", "feature", "origin/main"], env)

        assert branch_can_delete(clone, "feature", env=env) is True

    def test_branch_ahead_of_its_upstream_is_not_deletable_even_when_checked_out(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _source, clone = clone_local_remote(tmp_path, env)
        git_run(clone, ["checkout", "-b", "feature", "--track", "origin/main", "-q"], env)
        commit_file(clone, env, name="ahead.txt")
        # HEAD is the branch itself, so only consulting the upstream can tell
        # that the branch carries a commit the upstream has not taken.
        assert current_branch(clone, env=env) == "feature"

        assert branch_can_delete(clone, "feature", env=env) is False


# ---------------------------------------------------------------------------
# local_branch_exists / remote_branch_exists
# ---------------------------------------------------------------------------


class TestLocalBranchExists:
    def test_returns_true_when_branch_exists(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        assert local_branch_exists(repo, "main", env=env) is True

    def test_returns_false_when_branch_missing(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        init_repo(repo, env)
        assert local_branch_exists(repo, "nonexistent", env=env) is False


class TestRemoteBranchExists:
    def test_returns_true_when_a_remote_carries_the_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = clone_with_fork_remote(
            tmp_path, env, branch="branch-x", on_origin=True, on_fork=False
        )
        assert remote_branch_exists(repo, "branch-x", env=env) is True

    def test_returns_false_when_no_remote_carries_the_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = clone_with_fork_remote(tmp_path, env, branch="branch-x")
        assert remote_branch_exists(repo, "missing", env=env) is False


# ---------------------------------------------------------------------------
# default_remote_branch_ref
# ---------------------------------------------------------------------------


class TestDefaultRemoteBranchRef:
    def test_returns_the_tracking_ref_of_origins_default_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _source, clone = clone_local_remote(tmp_path, env)
        assert default_remote_branch_ref(clone, env=env) == "origin/main"

    def test_follows_a_non_main_default_branch(self, tmp_path: Path, env: dict[str, str]) -> None:
        _source, clone = clone_local_remote(tmp_path, env, default_branch="develop")
        assert default_remote_branch_ref(clone, env=env) == "origin/develop"

    # Kept as a behavioral fake: git's symbolic-ref either resolves the ref or
    # fails, so the empty-answer guard cannot be reached through a real repo.
    def test_exits_when_symbolic_ref_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "   \n",
        )
        with pytest.raises(SystemExit) as exc_info:
            default_remote_branch_ref(tmp_path)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# remote_unmerged_branches
# ---------------------------------------------------------------------------


class TestRemoteUnmergedBranches:
    def test_lists_remote_branches_not_merged_into_the_base_ref(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _source, clone = clone_local_remote(tmp_path, env, branches=["feature-a", "feature-b"])

        from_main = remote_unmerged_branches(clone, base_ref="origin/main", env=env)
        from_feature_a = remote_unmerged_branches(clone, base_ref="origin/feature-a", env=env)

        assert set(from_main) == {"origin/feature-a", "origin/feature-b"}
        # feature-a and everything it contains drop out once it is the base.
        assert set(from_feature_a) == {"origin/feature-b"}

    def test_returns_empty_when_every_remote_branch_is_merged(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _source, clone = clone_local_remote(tmp_path, env)
        assert remote_unmerged_branches(clone, base_ref="origin/main", env=env) == []

    # Kept as a behavioral fake: real `git for-each-ref` never emits empty
    # lines, so the filtering guard can only be exercised on synthetic output.
    def test_filters_empty_lines(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "\norigin/branch\n\n",
        )
        result = remote_unmerged_branches(tmp_path, base_ref="origin/main")
        assert result == ["origin/branch"]


# ---------------------------------------------------------------------------
# create_tracking_branch
# ---------------------------------------------------------------------------


class TestCreateTrackingBranch:
    def test_creates_a_local_branch_tracking_the_remote_ref(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _source, clone = clone_local_remote(tmp_path, env, branches=["feature"])
        # git's own automatic upstream setup is switched off, so tracking can
        # only come from the helper itself.
        Path(env["HOME"], ".gitconfig").write_text(
            "[branch]\n\tautoSetupMerge = false\n", encoding="utf-8"
        )
        assert local_branch_exists(clone, "feature", env=env) is False

        create_tracking_branch(clone, "feature", "origin/feature", env=env)

        assert local_branch_exists(clone, "feature", env=env) is True
        assert git_output(clone, ["rev-parse", "feature"], env) == git_output(
            clone, ["rev-parse", "origin/feature"], env
        )
        assert _branch_upstream(clone, "feature", env=env) == "origin/feature"


# ---------------------------------------------------------------------------
# symbolic_ref
# ---------------------------------------------------------------------------


class TestSymbolicRef:
    def test_resolves_head_to_the_checked_out_branch(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = init_repo(tmp_path / "repo", env)
        assert symbolic_ref(repo, "HEAD", env=env) == "main"

    def test_resolves_origin_head_to_the_remote_tracking_ref(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        _source, clone = clone_local_remote(tmp_path, env)
        assert symbolic_ref(clone, "refs/remotes/origin/HEAD", env=env) == "origin/main"


# ---------------------------------------------------------------------------
# ls_remote_head
# ---------------------------------------------------------------------------


class TestLsRemoteHead:
    def test_returns_symref_for_local_repo(self, tmp_path: Path, env: dict[str, str]) -> None:
        # `git ls-remote --symref <path> HEAD` resolves HEAD against a local
        # repository path with no network access, so a real repo exercises the
        # helper end to end.
        repo = tmp_path / "repo"
        init_repo(repo, env)
        result = ls_remote_head(str(repo), env=env)
        # HEAD symbolically resolves to refs/heads/main, plus the HEAD sha line.
        assert "ref: refs/heads/main\tHEAD" in result
        assert "\tHEAD" in result


# ---------------------------------------------------------------------------
# find_first_git_repo
# ---------------------------------------------------------------------------


class TestFindFirstGitRepo:
    def test_finds_the_repo_among_plain_directories(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        (tmp_path / "beta").mkdir()
        init_repo(tmp_path / "gamma", env)

        assert find_first_git_repo(tmp_path) == tmp_path / "gamma"

    def test_returns_the_first_repo_in_sorted_order(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        init_repo(tmp_path / "aaa", env)
        init_repo(tmp_path / "bbb", env)

        assert find_first_git_repo(tmp_path) == tmp_path / "aaa"

    def test_searches_nested_directories(self, tmp_path: Path, env: dict[str, str]) -> None:
        nested = init_repo(tmp_path / "outer" / "inner", env)

        assert find_first_git_repo(tmp_path) == nested

    def test_exits_when_no_git_repo_found(self, tmp_path: Path) -> None:
        (tmp_path / "notarepo").mkdir()
        with pytest.raises(SystemExit) as exc_info:
            find_first_git_repo(tmp_path)
        assert exc_info.value.code == 1

    def test_exits_when_parent_dir_is_empty(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(SystemExit) as exc_info:
            find_first_git_repo(empty_dir)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# fetch_output
# ---------------------------------------------------------------------------


class TestFetchOutput:
    def test_captures_the_output_of_a_command_run_in_cwd(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = init_repo(tmp_path / "repo", env)

        returncode, stdout, stderr = fetch_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=repo, env=env
        )

        assert returncode == 0
        assert Path(stdout.strip()) == repo
        assert stderr == ""

    def test_uses_the_supplied_environment(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = init_repo(tmp_path / "repo", env)
        probe_env = {**env, "GIT_AUTHOR_NAME": "Env Probe"}

        returncode, stdout, _stderr = fetch_output(
            ["git", "var", "GIT_AUTHOR_IDENT"], cwd=repo, env=probe_env
        )

        assert returncode == 0
        assert "Env Probe" in stdout

    def test_reports_the_failure_of_the_command(self, tmp_path: Path, env: dict[str, str]) -> None:
        returncode, stdout, stderr = fetch_output(
            ["git", "-C", str(tmp_path / "missing"), "status"], env=env
        )

        assert returncode != 0
        assert stdout == ""
        assert stderr != ""


# ---------------------------------------------------------------------------
# Remote branch resolution across multiple remotes
# ---------------------------------------------------------------------------


class TestRemoteBranchResolution:
    def test_remotes_with_branch_lists_only_remotes_carrying_it(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = clone_with_fork_remote(
            tmp_path, env, branch="branch-x", on_origin=False, on_fork=True
        )
        assert remotes_with_branch(repo, "branch-x", env=env) == ["fork"]
        assert remotes_with_branch(repo, "main", env=env) == ["fork", "origin"]
        assert remotes_with_branch(repo, "absent", env=env) == []

    def test_remote_branch_exists_finds_branch_on_non_origin_remote(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = clone_with_fork_remote(
            tmp_path, env, branch="branch-x", on_origin=False, on_fork=True
        )
        assert remote_branch_exists(repo, "branch-x", env=env) is True

    def test_unique_remote_branch_ref_resolves_single_remote(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = clone_with_fork_remote(
            tmp_path, env, branch="branch-x", on_origin=False, on_fork=True
        )
        assert unique_remote_branch_ref(repo, "branch-x", env=env) == "fork/branch-x"

    def test_unique_remote_branch_ref_is_none_without_any_remote(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = clone_with_fork_remote(
            tmp_path, env, branch="branch-x", on_origin=False, on_fork=True
        )
        assert unique_remote_branch_ref(repo, "absent", env=env) is None

    def test_unique_remote_branch_ref_exits_when_several_remotes_match(
        self, tmp_path: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo = clone_with_fork_remote(
            tmp_path, env, branch="branch-x", on_origin=True, on_fork=True
        )
        with pytest.raises(SystemExit) as exc_info:
            unique_remote_branch_ref(repo, "branch-x", env=env)
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "fork" in err
        assert "origin" in err

    def test_worktree_add_with_track_creates_branch_tracking_the_remote(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = clone_with_fork_remote(
            tmp_path, env, branch="branch-x", on_origin=False, on_fork=True
        )
        wt = tmp_path / "wt-branch-x"

        worktree_add(
            repo, wt, "branch-x", create=True, track=True, start_point="fork/branch-x", env=env
        )

        assert (wt / "fork.txt").exists()
        assert current_branch(wt, env=env) == "branch-x"
        upstream = subprocess.run(
            ["git", "config", "--get", "branch.branch-x.remote"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert upstream.stdout.strip() == "fork"
