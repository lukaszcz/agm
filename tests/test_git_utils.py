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
    worktree_prune,
    worktree_remove,
)


def _init_repo(path: Path, env: dict[str, str]) -> None:
    """Initialize a git repo at *path* with an initial commit.

    Uses *env* for git identity (GIT_AUTHOR_NAME / GIT_COMMITTER_NAME must be
    set).  Mirrors the helper pattern from test_project_utils.py / test_config_git.py.
    """
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", "-q"], cwd=path, env=env, check=True)
    (path / "README.md").write_text("# test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, env=env, check=True)
    subprocess.run(["git", "commit", "-m", "initial", "-q"], cwd=path, env=env, check=True)


def _assert_git_repo_command(cmd: list[str], repo_dir: Path, *parts: str) -> None:
    assert cmd[0] == "git"
    assert "-C" in cmd
    assert cmd[cmd.index("-C") + 1] == str(repo_dir)
    for part in parts:
        assert part in cmd


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
        _init_repo(repo, env)
        assert containing_root(repo, env=env) == repo

    def test_returns_repo_root_from_subdir(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo, env)
        subdir = repo / "src"
        subdir.mkdir()
        assert containing_root(subdir, env=env) == repo


class TestExactRepoRoot:
    def test_returns_path_for_exact_repo_root(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo, env)
        assert exact_repo_root(repo, env=env) == repo

    def test_returns_none_for_subdir_of_repo(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo, env)
        subdir = repo / "src"
        subdir.mkdir()
        assert exact_repo_root(subdir, env=env) is None

    def test_returns_none_for_plain_dir(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        assert exact_repo_root(plain) is None


class TestGenericGitProbeHelpers:
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
        _init_repo(repo, env)
        # Nothing staged after the initial commit.
        assert has_staged_changes(repo, [repo / "README.md"], env=env) is False

    def test_has_staged_changes_returns_true_when_file_staged(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo, env)
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
        _init_repo(repo, env)
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
        _init_repo(repo, env)
        assert checkout_root(cwd=repo) == repo

    def test_returns_toplevel_from_subdir(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "myrepo"
        _init_repo(repo, env)
        subdir = repo / "src"
        subdir.mkdir()
        assert checkout_root(cwd=subdir) == repo

    def test_falls_back_to_repo_subdir_when_cwd_is_not_git(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        # cwd is a plain dir that contains a "repo/" subdir that IS a git repo.
        project = tmp_path / "project"
        project.mkdir()
        _init_repo(project / "repo", env)
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
        _init_repo(repo, env)
        assert git_common_dir(cwd=repo) == repo / ".git"

    def test_returns_common_dir_from_worktree(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo, env)
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
    # These helpers hit a remote (network).  Kept as behavioral fakes: setting
    # up a real local bare-repo remote + clone and asserting fetch effects would
    # work but adds significant complexity for thin wrapper functions whose only
    # observable side-effect is "git fetch ran."  The fake asserts the call is
    # made with the expected arguments.
    def test_fetch_calls_require_success_with_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        fetch(tmp_path)
        _assert_git_repo_command(captured[0], tmp_path, "fetch")

    def test_fetch_passes_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        received_env: list[dict[str, str] | None] = []

        def fake_require_success(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
            received_env.append(env)

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
        _assert_git_repo_command(captured[0], tmp_path, "fetch", "--all", "--prune")

    def test_fetch_prune_origin_passes_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        fetch_prune_origin(tmp_path)
        _assert_git_repo_command(captured[0], tmp_path, "fetch", "--prune", "origin")


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merges_upstream_branch(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo, env)

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
        _init_repo(repo, env)
        assert current_branch(repo, env=env) == "main"

    def test_returns_branch_after_checkout(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo, env)
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
        _init_repo(repo, env)
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
        _init_repo(repo, env)
        subprocess.run(["git", "branch", "feature"], cwd=repo, env=env, check=True)
        wt = tmp_path / "wt-feature"
        worktree_add(repo, wt, "feature", env=env)
        assert wt.is_dir()

    def test_add_with_create_creates_new_branch_and_worktree(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo, env)
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
        _init_repo(repo, env)
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
        _init_repo(repo, env)
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
        _init_repo(source, env)
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
        _init_repo(source, env)
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
        _init_repo(repo, env)
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
        _init_repo(repo, env)
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
        _init_repo(repo, env)
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
        _init_repo(repo, env)
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
        _init_repo(repo, env)
        # A branch with no commits beyond main is trivially merged, so the
        # safe `-d` deletion accepts it.
        subprocess.run(["git", "branch", "merged"], cwd=repo, env=env, check=True)
        assert local_branch_exists(repo, "merged", env=env) is True

        branch_delete(repo, "merged", env=env)

        assert local_branch_exists(repo, "merged", env=env) is False

    def test_force_deletes_unmerged_branch(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo, env)
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


class TestBranchCanDelete:
    def test_returns_false_when_branch_not_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.local_branch_exists", lambda repo, b, env=None: False)
        result = branch_can_delete(tmp_path, "missing")
        assert result is False

    def test_returns_true_when_force_and_branch_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.local_branch_exists", lambda repo, b, env=None: True)
        result = branch_can_delete(tmp_path, "feature", force=True)
        assert result is True

    def test_returns_true_when_merged_into_upstream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.local_branch_exists", lambda repo, b, env=None: True)
        monkeypatch.setattr("agm.vcs.git._branch_upstream", lambda repo, b, env=None: "origin/main")
        monkeypatch.setattr("agm.vcs.git._is_ancestor", lambda repo, a, d, env=None: True)
        result = branch_can_delete(tmp_path, "feature")
        assert result is True

    def test_returns_true_when_merged_into_head_no_upstream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.local_branch_exists", lambda repo, b, env=None: True)
        monkeypatch.setattr("agm.vcs.git._branch_upstream", lambda repo, b, env=None: None)
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
        monkeypatch.setattr("agm.vcs.git.local_branch_exists", lambda repo, b, env=None: True)
        monkeypatch.setattr(
            "agm.vcs.git._branch_upstream",
            lambda repo, b, env=None: "origin/main",
        )
        monkeypatch.setattr("agm.vcs.git._is_ancestor", lambda repo, a, d, env=None: False)
        result = branch_can_delete(tmp_path, "feature")
        assert result is False

    def test_uses_upstream_over_head_when_upstream_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("agm.vcs.git.local_branch_exists", lambda repo, b, env=None: True)
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
    def test_returns_true_when_branch_exists(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo, env)
        assert local_branch_exists(repo, "main", env=env) is True

    def test_returns_false_when_branch_missing(self, tmp_path: Path, env: dict[str, str]) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo, env)
        assert local_branch_exists(repo, "nonexistent", env=env) is False


class TestRemoteBranchExists:
    # remote_branch_exists checks refs/remotes/origin/<branch>; these only
    # exist after a fetch from a real remote.  Kept as behavioral fakes.
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

    def test_checks_correct_remote_ref_path(
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

    def test_queries_origin_head_ref(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    def test_returns_non_empty_lines(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "origin/feature-a\norigin/feature-b\n",
        )
        result = remote_unmerged_branches(tmp_path, base_ref="origin/main")
        assert result == ["origin/feature-a", "origin/feature-b"]

    def test_filters_empty_lines(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    # Kept as a behavioral fake: create_tracking_branch tracks a remote-tracking
    # ref (origin/<branch>), which only exists after fetching from a real remote.
    # Setting up a remote + fetch to assert the tracking config is disproportionate
    # for this thin wrapper, so the fake verifies the invocation.
    def test_calls_require_success_with_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []
        monkeypatch.setattr(
            "agm.vcs.git.require_success",
            lambda cmd, **kwargs: captured.append(cmd),
        )
        create_tracking_branch(tmp_path, "my-branch", "origin/my-branch")
        _assert_git_repo_command(
            captured[0], tmp_path, "branch", "--track", "my-branch", "origin/my-branch"
        )


# ---------------------------------------------------------------------------
# symbolic_ref
# ---------------------------------------------------------------------------


class TestSymbolicRef:
    def test_returns_stripped_output(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "agm.vcs.git.require_capture",
            lambda cmd, **kwargs: "origin/main\n",
        )
        result = symbolic_ref(tmp_path, "refs/remotes/origin/HEAD")
        assert result == "origin/main"

    # Kept as a behavioral fake: the ref symbolic_ref is exercised with here
    # (refs/remotes/origin/HEAD) is a remote-tracking ref that only exists after
    # cloning/fetching from a real remote, so a focused fake of the output is the
    # right tool rather than standing up a remote.
    def test_passes_ref_in_command(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    def test_returns_symref_for_local_repo(self, tmp_path: Path, env: dict[str, str]) -> None:
        # `git ls-remote --symref <path> HEAD` resolves HEAD against a local
        # repository path with no network access, so a real repo exercises the
        # helper end to end.
        repo = tmp_path / "repo"
        _init_repo(repo, env)
        result = ls_remote_head(str(repo), env=env)
        # HEAD symbolically resolves to refs/heads/main, plus the HEAD sha line.
        assert "ref: refs/heads/main\tHEAD" in result
        assert "\tHEAD" in result


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

    def test_returns_nonzero_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
            setattr(wt, "branch", "other")

    def test_equality(self, tmp_path: Path) -> None:
        a = WorktreeInfo(path=tmp_path, branch="main")
        b = WorktreeInfo(path=tmp_path, branch="main")
        assert a == b

    def test_inequality_different_branch(self, tmp_path: Path) -> None:
        a = WorktreeInfo(path=tmp_path, branch="main")
        b = WorktreeInfo(path=tmp_path, branch="develop")
        assert a != b
