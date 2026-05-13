"""Tests for agm.commands.init."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import agm.commands.init as init_module
import agm.vcs.git as git_helpers
from agm.commands.args import InitArgs
from agm.commands.init import (
    configure_project_dir,
    derive_project_name,
    ensure_git_repo,
    ensure_gitignore_entry,
    looks_like_repo_url,
    run,
    use_embedded_layout,
    write_file_if_missing,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_args(
    positional: list[str] | None = None,
    *,
    branch: str | None = None,
    embedded: bool = False,
    workspace: bool = False,
    clone: bool = False,
) -> InitArgs:
    return InitArgs(
        positional=positional or [],
        branch=branch,
        embedded=embedded,
        workspace=workspace,
        clone=clone,
    )


# ---------------------------------------------------------------------------
# looks_like_repo_url
# ---------------------------------------------------------------------------


class TestLooksLikeRepoUrl:
    def test_https_url(self) -> None:
        assert looks_like_repo_url("https://github.com/org/repo") is True

    def test_http_url(self) -> None:
        assert looks_like_repo_url("http://example.com/repo") is True

    def test_ssh_protocol_url(self) -> None:
        assert looks_like_repo_url("ssh://git@github.com/org/repo") is True

    def test_git_at_with_colon(self) -> None:
        assert looks_like_repo_url("git@github.com:org/repo") is True

    def test_git_at_without_colon_is_not_a_url(self) -> None:
        # starts with git@ but no colon — not a URL
        assert looks_like_repo_url("git@nodomain") is False

    def test_github_com_colon(self) -> None:
        assert looks_like_repo_url("github.com:org/repo") is True

    def test_github_com_slash(self) -> None:
        assert looks_like_repo_url("github.com/org/repo") is True

    def test_ends_with_dot_git(self) -> None:
        assert looks_like_repo_url("some-local-path.git") is True

    def test_plain_directory_name(self) -> None:
        assert looks_like_repo_url("myproject") is False

    def test_relative_path(self) -> None:
        assert looks_like_repo_url("../myproject") is False

    def test_absolute_path_without_git_suffix(self) -> None:
        assert looks_like_repo_url("/home/user/project") is False

    def test_absolute_path_with_git_suffix(self) -> None:
        assert looks_like_repo_url("/home/user/project.git") is True

    def test_empty_string(self) -> None:
        assert looks_like_repo_url("") is False

    def test_git_protocol_url(self) -> None:
        assert looks_like_repo_url("git://github.com/org/repo") is True

    def test_scp_like_github(self) -> None:
        assert looks_like_repo_url("git@github.com:org/repo.git") is True


# ---------------------------------------------------------------------------
# derive_project_name
# ---------------------------------------------------------------------------


class TestDeriveProjectName:
    def test_https_url(self) -> None:
        assert derive_project_name("https://github.com/org/myrepo") == "myrepo"

    def test_https_url_with_git_suffix(self) -> None:
        assert derive_project_name("https://github.com/org/myrepo.git") == "myrepo"

    def test_scp_like_url(self) -> None:
        assert derive_project_name("git@github.com:org/myrepo.git") == "myrepo"

    def test_trailing_slash_is_stripped(self) -> None:
        assert derive_project_name("https://github.com/org/myrepo/") == "myrepo"

    def test_multiple_trailing_slashes(self) -> None:
        assert derive_project_name("https://github.com/org/myrepo///") == "myrepo"

    def test_no_git_suffix(self) -> None:
        assert derive_project_name("https://github.com/org/proj") == "proj"

    def test_simple_name_without_extension(self) -> None:
        assert derive_project_name("myrepo") == "myrepo"

    def test_only_dot_git_raises(self) -> None:
        # Path(".git").name == ".git"; removesuffix(".git") == "" → error
        with pytest.raises(SystemExit):
            derive_project_name(".git")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(SystemExit):
            derive_project_name("")

    def test_just_slashes_raises(self) -> None:
        # rstrip("/") == "" → Path("").name == "" → error
        with pytest.raises(SystemExit):
            derive_project_name("///")

    def test_error_message_contains_repo_url(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            derive_project_name(".git")
        captured = capsys.readouterr()
        assert ".git" in captured.err


# ---------------------------------------------------------------------------
# write_file_if_missing
# ---------------------------------------------------------------------------


class TestWriteFileIfMissing:
    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "new.txt"
        write_file_if_missing(target, "hello")
        assert target.read_text(encoding="utf-8") == "hello\n"

    def test_does_not_overwrite_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "existing.txt"
        target.write_text("original\n", encoding="utf-8")
        write_file_if_missing(target, "new content")
        assert target.read_text(encoding="utf-8") == "original\n"

    def test_appends_newline_to_content(self, tmp_path: Path) -> None:
        target = tmp_path / "file.txt"
        write_file_if_missing(target, "# comment")
        assert target.read_text(encoding="utf-8").endswith("\n")


# ---------------------------------------------------------------------------
# ensure_gitignore_entry
# ---------------------------------------------------------------------------


class TestEnsureGitignoreEntry:
    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        ensure_gitignore_entry(gitignore, ".agm")
        assert gitignore.read_text(encoding="utf-8") == ".agm\n"

    def test_adds_entry_when_file_exists_without_it(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n", encoding="utf-8")
        ensure_gitignore_entry(gitignore, ".agm")
        lines = gitignore.read_text(encoding="utf-8").splitlines()
        assert ".agm" in lines
        assert "*.pyc" in lines

    def test_does_not_duplicate_existing_entry(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text(".agm\n*.pyc\n", encoding="utf-8")
        ensure_gitignore_entry(gitignore, ".agm")
        lines = gitignore.read_text(encoding="utf-8").splitlines()
        assert lines.count(".agm") == 1

    def test_adds_newline_before_entry_when_file_has_no_trailing_newline(
        self, tmp_path: Path
    ) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc", encoding="utf-8")
        ensure_gitignore_entry(gitignore, ".agm")
        content = gitignore.read_text(encoding="utf-8")
        assert content == "*.pyc\n.agm\n"

    def test_appends_directly_when_file_already_ends_with_newline(
        self, tmp_path: Path
    ) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\n", encoding="utf-8")
        ensure_gitignore_entry(gitignore, ".agm")
        content = gitignore.read_text(encoding="utf-8")
        assert content == "*.pyc\n.agm\n"

    def test_entry_ends_with_newline(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        ensure_gitignore_entry(gitignore, ".agm")
        assert gitignore.read_text(encoding="utf-8").endswith("\n")


# ---------------------------------------------------------------------------
# ensure_git_repo
# ---------------------------------------------------------------------------


class TestEnsureGitRepo:
    def test_runs_git_init_when_not_a_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[list[str]] = []

        def fake_require_success(cmd: list[str], **_kwargs: object) -> None:
            called.append(cmd)

        monkeypatch.setattr(init_module, "require_success", fake_require_success)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        ensure_git_repo(tmp_path)

        assert called == [["git", "init", "-q", str(tmp_path)]]

    def test_skips_git_init_when_already_a_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        called: list[list[str]] = []

        def fake_require_success(cmd: list[str], **_kwargs: object) -> None:
            called.append(cmd)

        monkeypatch.setattr(init_module, "require_success", fake_require_success)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: True)

        ensure_git_repo(tmp_path)

        assert called == []

    def test_runs_git_init_when_git_dir_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[list[str]] = []

        def fake_require_success(cmd: list[str], **_kwargs: object) -> None:
            called.append(cmd)

        monkeypatch.setattr(init_module, "require_success", fake_require_success)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: True)

        ensure_git_repo(tmp_path)

        assert called == [["git", "init", "-q", str(tmp_path)]]


# ---------------------------------------------------------------------------
# configure_project_dir – workspace layout
# ---------------------------------------------------------------------------


class TestConfigureProjectDirWorkspace:
    def test_creates_expected_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        project_dir = tmp_path / "proj"
        configure_project_dir(project_dir, embedded=False)

        assert (project_dir / "repo").is_dir()
        assert (project_dir / "deps").is_dir()
        assert (project_dir / "notes").is_dir()
        assert (project_dir / "config").is_dir()
        assert (project_dir / "worktrees").is_dir()

    def test_creates_config_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        project_dir = tmp_path / "proj"
        configure_project_dir(project_dir, embedded=False)

        assert (project_dir / "config" / "env.sh").is_file()
        assert (project_dir / "config" / "setup.sh").is_file()

    def test_setup_sh_is_executable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        project_dir = tmp_path / "proj"
        configure_project_dir(project_dir, embedded=False)

        setup_sh = project_dir / "config" / "setup.sh"
        assert setup_sh.stat().st_mode & 0o111 != 0

    def test_does_not_overwrite_existing_config_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "config").mkdir()
        env_sh = project_dir / "config" / "env.sh"
        env_sh.write_text("# custom\n", encoding="utf-8")

        configure_project_dir(project_dir, embedded=False)

        assert env_sh.read_text(encoding="utf-8") == "# custom\n"


# ---------------------------------------------------------------------------
# configure_project_dir – embedded layout
# ---------------------------------------------------------------------------


class TestConfigureProjectDirEmbedded:
    def test_creates_dot_agm_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        configure_project_dir(project_dir, embedded=True)

        assert (project_dir / ".agm").is_dir()

    def test_creates_expected_subdirectories_under_dot_agm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        configure_project_dir(project_dir, embedded=True)

        agm_dir = project_dir / ".agm"
        assert (agm_dir / "deps").is_dir()
        assert (agm_dir / "notes").is_dir()
        assert (agm_dir / "config").is_dir()
        assert (agm_dir / "worktrees").is_dir()

    def test_adds_embedded_layout_entries_to_gitignore(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        configure_project_dir(project_dir, embedded=True)

        gitignore = project_dir / ".gitignore"
        assert gitignore.is_file()
        lines = gitignore.read_text(encoding="utf-8").splitlines()
        assert ".agm" in lines
        assert ".agent-files/" in lines

    def test_adds_agent_files_to_existing_gitignore(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / ".gitignore").write_text("*.pyc\n.agm\n", encoding="utf-8")
        configure_project_dir(project_dir, embedded=True)

        lines = (project_dir / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert ".agm" in lines
        assert ".agent-files/" in lines
        assert lines.count(".agm") == 1

    def test_creates_config_files_under_dot_agm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        configure_project_dir(project_dir, embedded=True)

        config_dir = project_dir / ".agm" / "config"
        assert (config_dir / "env.sh").is_file()
        assert (config_dir / "setup.sh").is_file()

    def test_does_not_create_repo_directory_at_top_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        configure_project_dir(project_dir, embedded=True)

        assert not (project_dir / "repo").exists()


# ---------------------------------------------------------------------------
# use_embedded_layout
# ---------------------------------------------------------------------------


class TestUseEmbeddedLayout:
    def test_returns_true_when_embedded_flag_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        args = make_args(embedded=True)
        assert use_embedded_layout(args, project_dir=tmp_path, repo_url="") is True

    def test_returns_false_when_workspace_flag_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        args = make_args(workspace=True)
        assert use_embedded_layout(args, project_dir=tmp_path, repo_url="") is False

    def test_returns_false_when_repo_url_provided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        args = make_args()
        result = use_embedded_layout(
            args, project_dir=tmp_path, repo_url="https://github.com/org/repo"
        )
        assert result is False

    def test_returns_true_when_project_dir_is_git_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: True)
        args = make_args()
        # project_dir must exist for `exists()` to return True
        result = use_embedded_layout(args, project_dir=tmp_path, repo_url="")
        assert result is True

    def test_returns_false_when_no_git_repo_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        args = make_args()
        non_existing = tmp_path / "newproject"
        result = use_embedded_layout(args, project_dir=non_existing, repo_url="")
        assert result is False

    def test_workspace_flag_takes_priority_over_existing_git_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: True)
        args = make_args(workspace=True)
        result = use_embedded_layout(args, project_dir=tmp_path, repo_url="")
        assert result is False

    def test_repo_url_takes_priority_over_existing_git_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: True)
        args = make_args()
        result = use_embedded_layout(
            args, project_dir=tmp_path, repo_url="https://github.com/org/repo"
        )
        assert result is False

    def test_prints_message_when_git_repo_detected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: True)
        args = make_args()
        use_embedded_layout(args, project_dir=tmp_path, repo_url="")
        captured = capsys.readouterr()
        assert "embedded" in captured.out


# ---------------------------------------------------------------------------
# run – argument validation
# ---------------------------------------------------------------------------


class TestRunArgumentValidation:
    def test_too_many_positional_args_exits(self) -> None:
        args = make_args(["a", "b", "c"])
        with pytest.raises(SystemExit):
            run(args)

    def test_clone_without_repo_url_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = make_args(["myproject"], clone=True)
        with pytest.raises(SystemExit):
            run(args)
        assert "REPO_URL" in capsys.readouterr().err

    def test_branch_without_repo_url_exits(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = make_args(["myproject"], branch="feat")
        with pytest.raises(SystemExit):
            run(args)
        assert "REPO_URL" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# run – no-repo-url paths (local project init)
# ---------------------------------------------------------------------------


class TestRunLocalInit:
    def test_init_in_cwd_when_no_positional_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)

        args = make_args(workspace=True)
        run(args)

        assert (tmp_path / "config").is_dir()

    def test_init_with_project_name_creates_subdir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)

        args = make_args(["myproject"], workspace=True)
        run(args)

        assert (tmp_path / "myproject" / "config").is_dir()

    def test_single_positional_treated_as_project_name_when_not_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)

        args = make_args(["notaurl"], workspace=True)
        run(args)

        assert (tmp_path / "notaurl").is_dir()

    def test_single_positional_treated_as_repo_url_when_looks_like_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        cloned: list[list[str]] = []

        def fake_require_success(cmd: list[str], **_kw: object) -> None:
            if cmd[0] == "git" and "clone" in cmd:
                cloned.append(cmd)
                # simulate a clone by creating the target directory
                Path(cmd[-1]).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(init_module, "require_success", fake_require_success)

        url = "https://github.com/org/myrepo"
        args = make_args([url], workspace=True)
        run(args)

        assert any("clone" in c for c in cloned)

    def test_two_positionals_are_project_and_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        cloned: list[list[str]] = []

        def fake_require_success(cmd: list[str], **_kw: object) -> None:
            if cmd[0] == "git" and "clone" in cmd:
                cloned.append(cmd)
                Path(cmd[-1]).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(init_module, "require_success", fake_require_success)

        url = "https://github.com/org/myrepo"
        args = make_args(["customname", url], workspace=True)
        run(args)

        assert (tmp_path / "customname" / "repo").is_dir() or any(
            "customname" in str(c) for c in cloned
        )


# ---------------------------------------------------------------------------
# run – clone flow
# ---------------------------------------------------------------------------


class TestRunClone:
    def test_clone_derives_project_name_from_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        clone_calls: list[list[str]] = []

        def fake_require_success(cmd: list[str], **_kw: object) -> None:
            if "clone" in cmd:
                clone_calls.append(cmd)
                Path(cmd[-1]).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(init_module, "require_success", fake_require_success)

        url = "https://github.com/org/derived-name.git"
        args = make_args([url], clone=True, workspace=True)
        run(args)

        assert (tmp_path / "derived-name").is_dir()

    def test_clone_passes_branch_when_provided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        clone_calls: list[list[str]] = []

        def fake_require_success(cmd: list[str], **_kw: object) -> None:
            if "clone" in cmd:
                clone_calls.append(cmd)
                Path(cmd[-1]).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(init_module, "require_success", fake_require_success)

        url = "https://github.com/org/myrepo.git"
        args = make_args([url], clone=True, branch="dev", workspace=True)
        run(args)

        assert clone_calls, "git clone should have been called"
        assert "--branch" in clone_calls[0]
        assert "dev" in clone_calls[0]

    def test_clone_errors_when_target_dir_not_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        url = "https://github.com/org/myrepo.git"
        # Pre-populate the repo directory with content
        repo_dir = tmp_path / "myrepo" / "repo"
        repo_dir.mkdir(parents=True)
        (repo_dir / "somefile.txt").write_text("content\n", encoding="utf-8")

        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)
        args = make_args([url], clone=True, workspace=True)

        with pytest.raises(SystemExit):
            run(args)

        assert "already exists" in capsys.readouterr().err

    def test_clone_errors_with_absolute_path_when_repo_dir_outside_cwd(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Use a cwd that is NOT a parent of the repo_dir so that
        # repo_dir.relative_to(Path.cwd()) raises ValueError, exercising the
        # `except ValueError: display_dir = str(repo_dir)` branch.
        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        other_root = tmp_path / "other"
        # Create a non-empty repo dir under a completely separate tree
        repo_dir = other_root / "myrepo" / "repo"
        repo_dir.mkdir(parents=True)
        (repo_dir / "somefile.txt").write_text("content\n", encoding="utf-8")

        monkeypatch.chdir(cwd_dir)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)

        # Use two-positional form so project_dir is an absolute path outside cwd
        url = "https://github.com/org/myrepo.git"
        args = make_args([str(other_root / "myrepo"), url], workspace=True)
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)

        with pytest.raises(SystemExit):
            run(args)

        captured = capsys.readouterr()
        assert "already exists" in captured.err
        # The display should be the absolute path string, not a relative one
        assert str(repo_dir) in captured.err

    def test_clone_without_branch_omits_branch_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        clone_calls: list[list[str]] = []

        def fake_require_success(cmd: list[str], **_kw: object) -> None:
            if "clone" in cmd:
                clone_calls.append(cmd)
                Path(cmd[-1]).mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(init_module, "require_success", fake_require_success)

        url = "https://github.com/org/myrepo.git"
        args = make_args([url], clone=True, workspace=True)
        run(args)

        assert clone_calls, "git clone should have been called"
        assert "--branch" not in clone_calls[0]


# ---------------------------------------------------------------------------
# run – embedded layout selection
# ---------------------------------------------------------------------------


class TestRunEmbeddedLayout:
    def test_embedded_flag_chooses_embedded_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)

        args = make_args(["proj"], embedded=True)
        run(args)

        assert (tmp_path / "proj" / ".agm").is_dir()

    def test_workspace_flag_chooses_workspace_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(git_helpers, "is_git_repo", lambda _p: False)
        monkeypatch.setattr(init_module, "require_success", lambda _cmd, **_kw: None)

        args = make_args(["proj"], workspace=True)
        run(args)

        assert (tmp_path / "proj" / "repo").is_dir()
        assert not (tmp_path / "proj" / ".agm").exists()


# ---------------------------------------------------------------------------
# integration: real git init (workspace layout)
# ---------------------------------------------------------------------------


class TestConfigureProjectDirRealGit:
    def test_workspace_layout_creates_real_git_repos(
        self, tmp_path: Path, env: dict[str, str]
    ) -> None:
        old_env: dict[str, str | None] = {}
        for k, v in env.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            project_dir = tmp_path / "proj"
            configure_project_dir(project_dir, embedded=False)

            # config and notes should be real git repos
            config_dir = project_dir / "config"
            notes_dir = project_dir / "notes"
            assert (config_dir / ".git").exists()
            assert (notes_dir / ".git").exists()
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
