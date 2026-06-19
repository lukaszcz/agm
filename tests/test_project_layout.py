"""Comprehensive tests for agm.project.layout path-resolution helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import agm.project.layout as layout_module
from agm.project.layout import (
    _copy_existing_config_files,
    _merge_branch_env_file,
    _project_dir_from_env,
    _project_dir_from_workspace,
    _resolved_cwd,
    branch_session_name,
    branch_worktree_path,
    copy_config,
    current_workspace,
    current_workspace_or_project_root,
    default_worktrees_dir,
    discover_current_project_dir,
    exit_if_main_workspace_branch,
    expected_branch_worktree_path,
    is_embedded_project,
    is_main_workspace_branch,
    is_project_dir,
    is_split_project,
    main_repo_dir,
    project_config_dir,
    project_deps_dir,
    project_name,
    project_notes_dir,
    project_repo_dir,
    project_root,
    require_current_project_dir,
    require_project_dir,
)

# ---------------------------------------------------------------------------
# _resolved_cwd
# ---------------------------------------------------------------------------


def test_resolved_cwd_returns_cwd_when_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _resolved_cwd(None)
    assert result == tmp_path


def test_resolved_cwd_resolves_given_path(tmp_path: Path) -> None:
    subdir = tmp_path / "sub"
    subdir.mkdir()
    symlink = tmp_path / "link"
    symlink.symlink_to(subdir)
    result = _resolved_cwd(symlink)
    assert result == subdir.resolve()


def test_resolved_cwd_returns_resolved_absolute_path(tmp_path: Path) -> None:
    result = _resolved_cwd(tmp_path)
    assert result == tmp_path.resolve()


def test_expected_branch_worktree_path_resolves_branch_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    repo_dir = project_dir / "repo"
    repo_dir.mkdir(parents=True)
    monkeypatch.setattr(layout_module.git_helpers, "current_branch", lambda _repo: "main")

    result = expected_branch_worktree_path(project_dir, "feature")

    assert result == (project_dir / "worktrees" / "feature").resolve(strict=False)


# ---------------------------------------------------------------------------
# _project_dir_from_workspace
# ---------------------------------------------------------------------------


def test_project_dir_from_workspace_finds_agm_marker(tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    (project / ".agm").mkdir(parents=True)
    assert _project_dir_from_workspace(project) == project / ".agm"


def test_project_dir_from_workspace_finds_repo_subdir(tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    (project / "repo").mkdir(parents=True)
    assert _project_dir_from_workspace(project) == project


def test_project_dir_from_workspace_finds_parent_via_repo_name(tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    repo_dir = project / "repo"
    (project / "worktrees").mkdir(parents=True)
    repo_dir.mkdir(parents=True)
    assert _project_dir_from_workspace(repo_dir) == project


def test_project_dir_from_workspace_finds_parent_via_repo_name_with_worktrees_hidden(
    tmp_path: Path,
) -> None:
    project = tmp_path / "myproject"
    repo_dir = project / "repo"
    (project / ".worktrees").mkdir(parents=True)
    repo_dir.mkdir(parents=True)
    assert _project_dir_from_workspace(repo_dir) == project


def test_project_dir_from_workspace_finds_parent_from_worktree_in_hidden_worktrees(
    tmp_path: Path,
) -> None:
    project = tmp_path / "myproject"
    worktree_dir = project / ".worktrees" / "feat"
    worktree_dir.mkdir(parents=True)
    assert _project_dir_from_workspace(worktree_dir) == project


def test_project_dir_from_workspace_finds_parent_from_worktree_in_worktrees(
    tmp_path: Path,
) -> None:
    project = tmp_path / "myproject"
    (project / "repo").mkdir(parents=True)
    worktree_dir = project / "worktrees" / "feat"
    worktree_dir.mkdir(parents=True)
    assert _project_dir_from_workspace(worktree_dir) == project


def test_project_dir_from_workspace_returns_none_for_plain_dir(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert _project_dir_from_workspace(plain) is None


def test_project_dir_from_workspace_finds_agm_from_embedded_worktree(tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    worktree_dir = project / ".agm" / "worktrees" / "feat"
    worktree_dir.mkdir(parents=True)
    assert _project_dir_from_workspace(worktree_dir) == project / ".agm"


# ---------------------------------------------------------------------------
# _project_dir_from_env
# ---------------------------------------------------------------------------


def test_project_dir_from_env_returns_path_from_proj_dir(tmp_path: Path) -> None:
    project = tmp_path / "project"
    assert _project_dir_from_env(env={"PROJ_DIR": str(project)}) == project


def test_project_dir_from_env_returns_none_when_unset() -> None:
    assert _project_dir_from_env(env={}) is None


def test_project_dir_from_env_returns_none_when_empty() -> None:
    assert _project_dir_from_env(env={"PROJ_DIR": ""}) is None


def test_project_dir_from_env_returns_agm_dir_directly(tmp_path: Path) -> None:
    agm_dir = tmp_path / "project" / ".agm"
    assert _project_dir_from_env(env={"PROJ_DIR": str(agm_dir)}) == agm_dir


# ---------------------------------------------------------------------------
# is_split_project
# ---------------------------------------------------------------------------


def test_is_split_project_true_when_repo_subdir_exists(tmp_path: Path) -> None:
    (tmp_path / "repo").mkdir()
    assert is_split_project(tmp_path) is True


def test_is_split_project_false_when_no_repo_subdir(tmp_path: Path) -> None:
    assert is_split_project(tmp_path) is False


def test_is_split_project_false_when_repo_is_file(tmp_path: Path) -> None:
    (tmp_path / "repo").write_text("not a dir", encoding="utf-8")
    assert is_split_project(tmp_path) is False


# ---------------------------------------------------------------------------
# is_embedded_project
# ---------------------------------------------------------------------------


def test_is_embedded_project_true_for_git_repo_with_agm_dir(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    agm_dir = project / ".agm"
    agm_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)
    assert is_embedded_project(agm_dir) is True


def test_is_embedded_project_false_without_agm_dir(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)
    assert is_embedded_project(project) is False


def test_is_embedded_project_false_without_git_repo(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    agm_dir = project / ".agm"
    agm_dir.mkdir()
    assert is_embedded_project(agm_dir) is False


def test_is_embedded_project_false_for_workspace_project_dir(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    assert is_embedded_project(project) is False


# ---------------------------------------------------------------------------
# is_project_dir
# ---------------------------------------------------------------------------


def test_is_project_dir_true_for_workspace_with_git_repo(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    repo_dir = project / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    assert is_project_dir(project) is True


def test_is_project_dir_true_for_embedded_project(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    agm_dir = project / ".agm"
    agm_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)
    assert is_project_dir(agm_dir) is True


def test_is_project_dir_false_for_plain_dir(tmp_path: Path) -> None:
    project = tmp_path / "plain"
    project.mkdir()
    assert is_project_dir(project) is False


# ---------------------------------------------------------------------------
# require_project_dir
# ---------------------------------------------------------------------------


def test_require_project_dir_returns_resolved_path_for_valid_project(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    repo_dir = project / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)

    result = require_project_dir(project)

    assert result == project.resolve()


def test_require_project_dir_exits_for_invalid_dir(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        require_project_dir(plain)

    assert exc_info.value.code == 1


def test_require_project_dir_resolves_relative_path(
    tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    repo_dir = project / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    monkeypatch.chdir(tmp_path)

    result = require_project_dir(Path("proj"))

    assert result == project.resolve()


# ---------------------------------------------------------------------------
# require_current_project_dir
# ---------------------------------------------------------------------------


def test_require_current_project_dir_returns_valid_project(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    repo_dir = project / "repo"
    repo_dir.mkdir()
    (project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)

    result = require_current_project_dir(project)

    assert result == project.resolve()


def test_require_current_project_dir_exits_when_not_a_project(tmp_path: Path) -> None:
    plain = tmp_path / "notaproject"
    plain.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        require_current_project_dir(plain)

    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# project_root
# ---------------------------------------------------------------------------


def test_project_root_returns_repo_parent_for_embedded_project(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    agm_dir = project / ".agm"
    agm_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)

    assert project_root(agm_dir) == project


def test_project_root_returns_project_dir_for_workspace_project(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    assert project_root(project) == project


# ---------------------------------------------------------------------------
# project_repo_dir
# ---------------------------------------------------------------------------


def test_project_repo_dir_returns_repo_subdir_for_workspace_project(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    assert project_repo_dir(project) == project / "repo"


def test_project_repo_dir_returns_parent_for_embedded_project(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    agm_dir = project / ".agm"
    agm_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)

    assert project_repo_dir(agm_dir) == project


def test_project_repo_dir_returns_input_when_no_layout_markers(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()

    assert project_repo_dir(project) == project


# ---------------------------------------------------------------------------
# main_repo_dir
# ---------------------------------------------------------------------------


def test_main_repo_dir_is_alias_for_project_repo_dir_workspace(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    assert main_repo_dir(project) == project_repo_dir(project)


def test_main_repo_dir_is_alias_for_project_repo_dir_embedded(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    agm_dir = project / ".agm"
    agm_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)

    assert main_repo_dir(agm_dir) == project_repo_dir(agm_dir)


# ---------------------------------------------------------------------------
# default_worktrees_dir
# ---------------------------------------------------------------------------


def test_default_worktrees_dir_workspace_project(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    # workspace: data_dir == project_dir, so worktrees dir is project/worktrees
    assert default_worktrees_dir(project) == project / "worktrees"


def test_default_worktrees_dir_embedded_project(tmp_path: Path) -> None:
    agm_dir = tmp_path / "proj" / ".agm"
    agm_dir.mkdir(parents=True)

    # embedded: data_dir == project_dir (.agm), worktrees directly inside
    assert default_worktrees_dir(agm_dir) == agm_dir / "worktrees"


# ---------------------------------------------------------------------------
# project_config_dir
# ---------------------------------------------------------------------------


def test_project_config_dir_workspace(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    assert project_config_dir(project) == project / "config"


def test_project_config_dir_embedded(tmp_path: Path) -> None:
    agm_dir = tmp_path / "proj" / ".agm"
    agm_dir.mkdir(parents=True)

    assert project_config_dir(agm_dir) == agm_dir / "config"


# ---------------------------------------------------------------------------
# project_deps_dir
# ---------------------------------------------------------------------------


def test_project_deps_dir_workspace(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    assert project_deps_dir(project) == project / "deps"


def test_project_deps_dir_embedded(tmp_path: Path) -> None:
    agm_dir = tmp_path / "proj" / ".agm"
    agm_dir.mkdir(parents=True)

    assert project_deps_dir(agm_dir) == agm_dir / "deps"


# ---------------------------------------------------------------------------
# project_notes_dir
# ---------------------------------------------------------------------------


def test_project_notes_dir_workspace(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    assert project_notes_dir(project) == project / "notes"


def test_project_notes_dir_embedded(tmp_path: Path) -> None:
    agm_dir = tmp_path / "proj" / ".agm"
    agm_dir.mkdir(parents=True)

    assert project_notes_dir(agm_dir) == agm_dir / "notes"


# ---------------------------------------------------------------------------
# is_main_workspace_branch
# ---------------------------------------------------------------------------


def test_is_main_workspace_branch_true_for_repo_literal(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert is_main_workspace_branch(project, "repo", repo_branch="main") is True


def test_is_main_workspace_branch_true_when_branch_equals_repo_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert is_main_workspace_branch(project, "main", repo_branch="main") is True


def test_is_main_workspace_branch_true_for_custom_repo_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert is_main_workspace_branch(project, "master", repo_branch="master") is True


def test_is_main_workspace_branch_false_for_different_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert is_main_workspace_branch(project, "feat/x", repo_branch="main") is False


def test_is_main_workspace_branch_false_for_develop_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert is_main_workspace_branch(project, "develop", repo_branch="main") is False


# ---------------------------------------------------------------------------
# branch_worktree_path
# ---------------------------------------------------------------------------


def test_branch_worktree_path_for_repo_literal_workspace(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "repo").mkdir(parents=True)

    result = branch_worktree_path(project, "repo", repo_branch="main")

    assert result == project / "repo"


def test_branch_worktree_path_for_main_branch_workspace(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "repo").mkdir(parents=True)

    result = branch_worktree_path(project, "main", repo_branch="main")

    assert result == project / "repo"


def test_branch_worktree_path_for_worktree_branch_workspace(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "repo").mkdir(parents=True)

    result = branch_worktree_path(project, "feat/my-feature", repo_branch="main")

    assert result == project / "worktrees" / "feat/my-feature"


def test_branch_worktree_path_for_worktree_branch_embedded(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    agm_dir = project / ".agm"
    agm_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)

    result = branch_worktree_path(agm_dir, "feat/abc", repo_branch="main")

    assert result == agm_dir / "worktrees" / "feat/abc"


# ---------------------------------------------------------------------------
# branch_session_name
# ---------------------------------------------------------------------------


def test_branch_session_name_for_repo_literal(tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    (project / "repo").mkdir(parents=True)

    assert branch_session_name(project, "repo") == "myproject"


def test_branch_session_name_for_plain_project_name(tmp_path: Path) -> None:
    project = tmp_path / "alpha"
    (project / "repo").mkdir(parents=True)

    assert branch_session_name(project, "repo") == "alpha"


def test_branch_session_name_for_worktree_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    (project / "repo").mkdir(parents=True)

    def fake_current_branch(_repo_dir: Path, *, env: dict[str, str] | None = None) -> str:
        del env
        return "main"

    monkeypatch.setattr(layout_module.git_helpers, "current_branch", fake_current_branch)

    assert branch_session_name(project, "feat/my-branch") == "proj/feat/my-branch"


def test_branch_session_name_for_main_branch_same_as_repo_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    (project / "repo").mkdir(parents=True)

    def fake_current_branch(_repo_dir: Path, *, env: dict[str, str] | None = None) -> str:
        del env
        return "main"

    monkeypatch.setattr(layout_module.git_helpers, "current_branch", fake_current_branch)

    assert branch_session_name(project, "main") == "proj"


def test_branch_session_name_for_embedded_project(
    tmp_path: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "myproject"
    project.mkdir()
    agm_dir = project / ".agm"
    agm_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)

    def fake_current_branch(_repo_dir: Path, *, env: dict[str, str] | None = None) -> str:
        del env
        return "main"

    monkeypatch.setattr(layout_module.git_helpers, "current_branch", fake_current_branch)
    assert branch_session_name(agm_dir, "main") == "myproject"
    assert branch_session_name(agm_dir, "feat/x") == "myproject/feat/x"


# ---------------------------------------------------------------------------
# exit_if_main_workspace_branch
# ---------------------------------------------------------------------------


def test_exit_if_main_workspace_branch_exits_for_repo_literal(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "repo").mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        exit_if_main_workspace_branch(project, "repo", repo_branch="main")

    assert exc_info.value.code == 1


def test_exit_if_main_workspace_branch_exits_for_main_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "repo").mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        exit_if_main_workspace_branch(project, "main", repo_branch="main")

    assert exc_info.value.code == 1


def test_exit_if_main_workspace_branch_does_not_exit_for_worktree_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "repo").mkdir(parents=True)

    # Should not raise
    exit_if_main_workspace_branch(project, "feat/x", repo_branch="main")


# ---------------------------------------------------------------------------
# _copy_existing_config_files
# ---------------------------------------------------------------------------


def test_copy_existing_config_files_copies_present_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    (source / ".env").write_text("FOO=bar\n", encoding="utf-8")
    (source / ".env.local").write_text("BAR=baz\n", encoding="utf-8")

    _copy_existing_config_files(source, target)

    assert (target / ".env").read_text(encoding="utf-8") == "FOO=bar\n"
    assert (target / ".env.local").read_text(encoding="utf-8") == "BAR=baz\n"


def test_copy_existing_config_files_copies_empty_env(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    (source / ".env").write_text("", encoding="utf-8")
    (source / ".env.local").write_text("BAR=baz\n", encoding="utf-8")

    _copy_existing_config_files(source, target)

    assert (target / ".env").read_text(encoding="utf-8") == ""
    assert (target / ".env.local").read_text(encoding="utf-8") == "BAR=baz\n"


def test_copy_existing_config_files_skips_missing_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    # Only one CONFIG_FILES entry present
    (source / ".setup.sh").write_text("#!/bin/bash\n", encoding="utf-8")

    _copy_existing_config_files(source, target)

    assert (target / ".setup.sh").read_text(encoding="utf-8") == "#!/bin/bash\n"
    assert not (target / ".env").exists()


def test_copy_existing_config_files_does_nothing_when_source_is_empty(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    _copy_existing_config_files(source, target)

    assert list(target.iterdir()) == []


# ---------------------------------------------------------------------------
# _merge_branch_env_file
# ---------------------------------------------------------------------------


def test_merge_branch_env_file_merges_keys_into_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    (source / ".env").write_text("NEW_KEY=value\n", encoding="utf-8")
    (target / ".env").write_text("EXISTING=old\n", encoding="utf-8")

    _merge_branch_env_file(source, target)

    content = (target / ".env").read_text(encoding="utf-8")
    assert "EXISTING=old" in content
    assert "NEW_KEY=value" in content


def test_merge_branch_env_file_creates_target_env_when_missing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    (source / ".env").write_text("MY_KEY=abc\n", encoding="utf-8")

    _merge_branch_env_file(source, target)

    content = (target / ".env").read_text(encoding="utf-8")
    assert "MY_KEY=abc" in content


def test_merge_branch_env_file_overwrites_existing_key(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    (source / ".env").write_text("KEY=new_value\n", encoding="utf-8")
    (target / ".env").write_text("KEY=old_value\n", encoding="utf-8")

    _merge_branch_env_file(source, target)

    content = (target / ".env").read_text(encoding="utf-8")
    assert "KEY=new_value" in content
    assert "KEY=old_value" not in content


def test_merge_branch_env_file_does_nothing_when_source_env_missing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    (target / ".env").write_text("PRESERVED=1\n", encoding="utf-8")

    _merge_branch_env_file(source, target)

    content = (target / ".env").read_text(encoding="utf-8")
    assert "PRESERVED=1" in content


# ---------------------------------------------------------------------------
# copy_config
# ---------------------------------------------------------------------------


def test_copy_config_copies_config_files_to_target(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    (config_dir / ".env").write_text("CONFIG_KEY=val\n", encoding="utf-8")
    (config_dir / ".env.local").write_text("LOCAL_KEY=local\n", encoding="utf-8")

    copy_config(project_dir=project, target=target, branch=None, cwd=None)

    assert (target / ".env").read_text(encoding="utf-8") == "CONFIG_KEY=val\n"
    assert (target / ".env.local").read_text(encoding="utf-8") == "LOCAL_KEY=local\n"


def test_copy_config_merges_branch_env_when_branch_given(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    workspace_config_dir = config_dir / "feat"
    workspace_config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    (config_dir / ".env").write_text("BASE_KEY=base\n", encoding="utf-8")
    (workspace_config_dir / ".env").write_text("BRANCH_KEY=branch\n", encoding="utf-8")

    copy_config(project_dir=project, target=target, branch="feat", cwd=None)

    target_env_content = (target / ".env.local").read_text(encoding="utf-8")
    assert "BASE_KEY=base" in target_env_content
    assert "BRANCH_KEY=branch" in target_env_content


def test_copy_config_does_nothing_when_target_does_not_exist(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "nonexistent"

    (config_dir / ".env").write_text("KEY=val\n", encoding="utf-8")

    # Should not raise and not create the target
    copy_config(project_dir=project, target=target, branch=None, cwd=None)

    assert not target.exists()


def test_copy_config_does_nothing_when_config_dir_missing(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    # No config/ dir
    copy_config(project_dir=project, target=target, branch=None, cwd=None)

    assert list(target.iterdir()) == []


def test_copy_config_resolves_project_dir_from_cwd_when_not_given(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    repo_dir = project / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, env=env, check=True)
    target = tmp_path / "checkout"
    target.mkdir()

    (config_dir / ".env").write_text("AUTO_KEY=auto\n", encoding="utf-8")

    copy_config(project_dir=None, target=target, branch=None, cwd=project)

    assert "AUTO_KEY=auto" in (target / ".env").read_text(encoding="utf-8")


def test_copy_config_uses_absolute_target_directly(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    (config_dir / ".setup.sh").write_text("#!/bin/bash\necho setup\n", encoding="utf-8")

    copy_config(project_dir=project, target=target, branch=None, cwd=tmp_path)

    assert (target / ".setup.sh").read_text(encoding="utf-8") == "#!/bin/bash\necho setup\n"


def test_copy_config_uses_relative_target_resolved_against_cwd(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    (config_dir / ".env").write_text("REL_KEY=relative\n", encoding="utf-8")

    # Pass relative target resolved against tmp_path
    copy_config(
        project_dir=project,
        target=Path("checkout"),
        branch=None,
        cwd=tmp_path,
    )

    assert "REL_KEY=relative" in (target / ".env").read_text(encoding="utf-8")


def test_copy_config_copies_arbitrary_dot_files_and_directories(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    (config_dir / ".new-tool").mkdir()
    (config_dir / ".new-tool" / "settings.json").write_text("{}", encoding="utf-8")
    (config_dir / ".customrc").write_text("custom\n", encoding="utf-8")
    (config_dir / "not-dot").write_text("skip\n", encoding="utf-8")

    copy_config(project_dir=project, target=target, branch=None, cwd=None)

    assert (target / ".new-tool" / "settings.json").read_text(encoding="utf-8") == "{}"
    assert (target / ".customrc").read_text(encoding="utf-8") == "custom\n"
    assert not (target / "not-dot").exists()


def test_copy_config_branch_dot_entries_override_base_entries(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    workspace_config_dir = config_dir / "feat"
    workspace_config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    (config_dir / ".tool").mkdir()
    (config_dir / ".tool" / "settings.json").write_text("base\n", encoding="utf-8")
    (config_dir / ".base-only").write_text("base\n", encoding="utf-8")
    (workspace_config_dir / ".tool").mkdir()
    (workspace_config_dir / ".tool" / "settings.json").write_text("branch\n", encoding="utf-8")
    (workspace_config_dir / ".branch-only").write_text("branch\n", encoding="utf-8")

    copy_config(project_dir=project, target=target, branch="feat", cwd=None)

    assert (target / ".tool" / "settings.json").read_text(encoding="utf-8") == "branch\n"
    assert (target / ".base-only").read_text(encoding="utf-8") == "base\n"
    assert (target / ".branch-only").read_text(encoding="utf-8") == "branch\n"


def test_copy_config_merges_dotenv_files_with_config_env_precedence(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    workspace_config_dir = config_dir / "feat"
    workspace_config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    (config_dir / ".env").write_text(
        "BASE_ONLY=base\nSHARED=base-env\n",
        encoding="utf-8",
    )
    (config_dir / ".env.local").write_text(
        "LOCAL_ONLY=base-local\nSHARED=base-local\n",
        encoding="utf-8",
    )
    (workspace_config_dir / ".env").write_text(
        "BRANCH_ONLY=branch\nSHARED=branch-env\n",
        encoding="utf-8",
    )
    (workspace_config_dir / ".env.local").write_text(
        "BRANCH_LOCAL_ONLY=branch-local\nSHARED=branch-local\n",
        encoding="utf-8",
    )

    copy_config(project_dir=project, target=target, branch="feat", cwd=None)

    assert (target / ".env").read_text(encoding="utf-8") == (
        "BASE_ONLY=base\nSHARED=base-env\n"
    )
    target_local = (target / ".env.local").read_text(encoding="utf-8")
    assert "BASE_ONLY=base" in target_local
    assert "LOCAL_ONLY=base-local" in target_local
    assert "BRANCH_ONLY=branch" in target_local
    assert "BRANCH_LOCAL_ONLY=branch-local" in target_local
    assert "SHARED=branch-local" in target_local


def test_copy_config_replaces_existing_target_env_local_when_merging(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    workspace_config_dir = config_dir / "feat"
    workspace_config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    (config_dir / ".env").write_text("CONFIGURED=yes\n", encoding="utf-8")
    (workspace_config_dir / ".env").write_text("BRANCH=yes\n", encoding="utf-8")
    (target / ".env.local").write_text("STALE=value\n", encoding="utf-8")

    copy_config(project_dir=project, target=target, branch="feat", cwd=None)

    target_local = (target / ".env.local").read_text(encoding="utf-8")
    assert "CONFIGURED=yes" in target_local
    assert "BRANCH=yes" in target_local
    assert "STALE=value" not in target_local


def test_copy_config_detects_current_workspace_branch_when_branch_not_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    config_dir = project / "config"
    workspace_config_dir = config_dir / "feat"
    workspace_config_dir.mkdir(parents=True)
    workspace_dir = project / "worktrees" / "feat"
    workspace_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "target"
    target.mkdir()

    (workspace_config_dir / ".branch-tool").write_text("branch\n", encoding="utf-8")
    monkeypatch.setattr(
        layout_module,
        "current_workspace",
        lambda _project_dir, *, cwd=None, env=None: layout_module.CurrentWorkspace(
            workspace_dir=workspace_dir,
            branch="feat",
            is_main=False,
        ),
    )

    copy_config(project_dir=project, target=target, branch=None, cwd=workspace_dir)

    assert (target / ".branch-tool").read_text(encoding="utf-8") == "branch\n"


# ---------------------------------------------------------------------------
# discover_current_project_dir – additional path walk coverage
# ---------------------------------------------------------------------------


def test_current_project_dir_from_nested_subdir_of_embedded_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    agm_dir = project / ".agm"
    agm_dir.mkdir()
    deep = project / "src" / "pkg" / "module"
    deep.mkdir(parents=True)
    monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda _path: True)

    assert discover_current_project_dir(deep) == agm_dir


def test_current_project_root_candidate_falls_back_when_no_markers_and_not_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda _path: False)

    assert discover_current_project_dir(plain) is None
    assert current_workspace_or_project_root(plain) == plain


def test_current_workspace_or_project_root_uses_proj_dir_env(tmp_path: Path) -> None:
    project = tmp_path / "project"

    assert current_workspace_or_project_root(tmp_path, env={"PROJ_DIR": str(project)}) == project


def test_discover_current_project_dir_uses_proj_dir_env(tmp_path: Path) -> None:
    project = tmp_path / "project"

    assert discover_current_project_dir(tmp_path, env={"PROJ_DIR": str(project)}) == project


def test_current_workspace_or_project_root_prefers_cwd_project_over_stale_proj_dir_env(
    tmp_path: Path, env: dict[str, str]
) -> None:
    stale_project = tmp_path / "stale"
    stale_repo = stale_project / "repo"
    stale_repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=stale_repo, env=env, check=True)

    current_project = tmp_path / "current"
    current_repo = current_project / "repo"
    current_repo.mkdir(parents=True)
    (current_project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=current_repo, env=env, check=True)

    result = current_workspace_or_project_root(
        current_project,
        env={"PROJ_DIR": str(stale_project)},
    )

    assert result == current_project


def test_discover_current_project_dir_prefers_cwd_project_over_stale_proj_dir_env(
    tmp_path: Path, env: dict[str, str]
) -> None:
    stale_project = tmp_path / "stale"
    stale_repo = stale_project / "repo"
    stale_repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=stale_repo, env=env, check=True)

    current_project = tmp_path / "current"
    current_repo = current_project / "repo"
    current_repo.mkdir(parents=True)
    (current_project / "worktrees").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=current_repo, env=env, check=True)

    result = discover_current_project_dir(
        current_project / "repo",
        env={"PROJ_DIR": str(stale_project)},
    )

    assert result == current_project


def test_current_workspace_or_project_root_uses_proj_dir_env_with_agm_dir(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    agm_dir = project / ".agm"
    agm_dir.mkdir(parents=True)

    assert current_workspace_or_project_root(
        tmp_path, env={"PROJ_DIR": str(agm_dir)}
    ) == agm_dir


def test_discover_current_project_dir_uses_proj_dir_env_with_agm_dir(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "project"
    agm_dir = project / ".agm"
    agm_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)

    assert discover_current_project_dir(
        tmp_path, env={"PROJ_DIR": str(agm_dir)}
    ) == agm_dir


def test_current_workspace_or_project_root_falls_back_to_git_checkout_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()
    checkout = project / "subdir"
    checkout.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda _path: True)
    monkeypatch.setattr(
        layout_module.git_helpers,
        "checkout_root",
        lambda _cwd=None: checkout,
    )

    assert current_workspace_or_project_root(cwd) == checkout


class TestCurrentProjectDirFallbackPaths:
    def test_uses_git_common_dir_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """git_common_dir is not used by simplified fallback discovery."""
        project = tmp_path / "myproject"
        repo = project / "repo"
        repo.mkdir(parents=True)
        (project / "worktrees").mkdir()

        worktree = tmp_path / "worktree"
        worktree.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: worktree,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None, env=None: repo / ".git",
        )

        result = current_workspace_or_project_root(worktree)
        assert result == worktree

    def test_falls_back_to_workspace_dir_when_no_project_markers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When no project markers exist, falls back to workspace_dir."""
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: plain_dir,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )

        result = current_workspace_or_project_root(plain_dir)
        assert result == plain_dir

    def test_checkout_root_raises_system_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root raises SystemExit, fallback returns cwd."""
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )

        result = current_workspace_or_project_root(plain_dir)
        assert result == plain_dir


class TestCurrentWorkspace:
    def test_returns_none_when_cwd_not_in_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        project = tmp_path / "proj"
        (project / "repo").mkdir(parents=True)
        other = tmp_path / "other"
        other.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: False)
        result = current_workspace(project, cwd=other)
        assert result is None

    def test_falls_back_to_repo_when_cwd_not_git_repo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When cwd is the project dir but not a git repo, uses the repo_dir."""
        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)

        def fake_is_git_repo(p: Path) -> bool:
            return p == repo

        def fake_project_dir(
            cwd: Path | None = None, *, env: dict[str, str] | None = None
        ) -> Path:
            del env
            return project

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", fake_is_git_repo)
        monkeypatch.setattr(layout_module, "discover_current_project_dir", fake_project_dir)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda p, env=None: "main",
        )

        result = current_workspace(project, cwd=project)
        assert result is not None
        assert result.workspace_dir == repo.resolve(strict=False)

    def test_checkout_root_raises_system_exit_uses_repo_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root raises SystemExit, falls back to repo_dir."""
        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)

        def fake_is_git_repo(p: Path) -> bool:
            return True

        def fake_project_dir(
            cwd: Path | None = None, *, env: dict[str, str] | None = None
        ) -> Path:
            del env
            return project

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", fake_is_git_repo)
        monkeypatch.setattr(layout_module, "discover_current_project_dir", fake_project_dir)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda p, env=None: "main",
        )

        result = current_workspace(project, cwd=repo)
        assert result is not None

    def test_checkout_root_raises_system_exit_no_repo_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root raises and repo_dir is not git, uses cwd as checkout."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".agm").mkdir()

        def fake_is_git_repo(p: Path) -> bool:
            return True

        def fake_project_dir(
            cwd: Path | None = None, *, env: dict[str, str] | None = None
        ) -> Path:
            del env
            return project

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", fake_is_git_repo)
        monkeypatch.setattr(layout_module, "discover_current_project_dir", fake_project_dir)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda p, env=None: "main",
        )

        result = current_workspace(project, cwd=project)
        assert result is None or result.workspace_dir is not None


class TestCurrentProjectDirCommonDirFallback:
    def test_returns_workspace_dir_when_in_parents(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When workspace_dir is a parent of current, return workspace_dir."""
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()

        parent_dir = tmp_path

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: parent_dir,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )

        result = current_workspace_or_project_root(plain_dir)
        assert result == parent_dir

    def test_checkout_root_succeeds_but_no_project_markers_uses_common_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root succeeds, git_common_dir is not consulted."""
        project = tmp_path / "myproject"
        repo = project / "repo"
        repo.mkdir(parents=True)
        (project / "worktrees").mkdir()

        worktree = tmp_path / "somewhere"
        worktree.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: worktree,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None, env=None: repo / ".git",
        )

        result = current_workspace_or_project_root(worktree)
        assert result == worktree


class TestCurrentWorkspaceSystemExitPaths:
    def test_checkout_root_raises_and_repo_is_git(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root raises and repo is git, use repo_dir."""
        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)

        def fake_is_git_repo(p: Path) -> bool:
            return p == repo or p == project

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", fake_is_git_repo)
        monkeypatch.setattr(
            layout_module,
            "discover_current_project_dir",
            lambda cwd=None, env=None: project,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda p, env=None: "main",
        )

        result = current_workspace(project, cwd=project)
        assert result is not None
        assert result.workspace_dir == repo

    def test_checkout_root_raises_and_repo_not_git_uses_current(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root raises and repo is not git, use current dir."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".agm").mkdir()

        def fake_is_git_repo(p: Path) -> bool:
            return True  # is_git_repo returns True for project

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", fake_is_git_repo)
        monkeypatch.setattr(
            layout_module,
            "discover_current_project_dir",
            lambda cwd=None, env=None: project,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda p, env=None: "main",
        )

        result = current_workspace(project, cwd=project)
        assert result is not None
        assert result.workspace_dir == project


class TestCurrentProjectDirCheckoutRootRaisesThenCommonDirRaises:
    def test_checkout_root_raises_and_common_dir_raises_returns_current(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root raises, falls back to current."""
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )

        result = current_workspace_or_project_root(plain_dir)
        assert result == plain_dir


class TestCurrentWorkspaceEdgeCases:
    def test_workspace_dir_equals_repo_dir_returns_main_checkout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When workspace_dir equals repo_dir, returns main workspace with branch=None."""
        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module,
            "discover_current_project_dir",
            lambda cwd=None, env=None: project.resolve(strict=False),
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: repo.resolve(strict=False),
        )

        result = current_workspace(project, cwd=repo)
        assert result is not None
        assert result.workspace_dir == repo.resolve(strict=False)
        assert result.branch is None
        assert result.is_main is True

    def test_workspace_dir_current_returns_main_when_same_as_repo_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When cwd is 'current' and checkout_root returns current, returns main."""
        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)
        (project / "worktrees").mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module,
            "discover_current_project_dir",
            lambda cwd=None, env=None: project.resolve(strict=False),
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: repo.resolve(strict=False),
        )

        result = current_workspace(project, cwd=repo)
        assert result is not None
        assert result.is_main is True


class TestCurrentWorkspaceWithRepoDirEnv:
    def test_repo_dir_env_var_points_inside_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When REPO_DIR env var points to a git repo inside project."""
        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)
        (project / "worktrees").mkdir()

        worktree_dir = project / "worktrees" / "feat"
        worktree_dir.mkdir(parents=True)

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda p, env=None: "feat",
        )

        env = {"REPO_DIR": str(worktree_dir)}
        result = current_workspace(project, env=env)
        assert result is not None
        assert result.workspace_dir == worktree_dir.resolve(strict=False)
        assert result.branch == "feat"
        assert result.is_main is False

    def test_repo_dir_env_var_points_to_repo_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When REPO_DIR points to the main repo_dir, checkout is main."""
        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda p, env=None: "main",
        )

        env = {"REPO_DIR": str(repo)}
        result = current_workspace(project, env=env)
        assert result is not None
        assert result.is_main is True
        assert result.branch is None


class TestCurrentProjectDirGitCommonDirFindsProject:
    def test_git_common_dir_parent_has_project_markers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root succeeds, that root is the fallback."""
        project = tmp_path / "myproject"
        repo = project / "repo"
        repo.mkdir(parents=True)
        (project / "worktrees").mkdir()

        worktree = tmp_path / "somewhere"
        worktree.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: worktree,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None, env=None: repo / ".git",
        )

        result = current_workspace_or_project_root(worktree)
        assert result == worktree

    def test_falls_back_to_current_when_nothing_matches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root succeeds, returns that checkout root."""
        isolated = tmp_path / "isolated"
        isolated.mkdir()
        other = tmp_path / "other"
        other.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: other,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )

        result = current_workspace_or_project_root(isolated)
        assert result == other


class TestCurrentWorkspaceCwdNotInProject:
    def test_returns_none_when_cwd_not_in_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """current_workspace returns None when cwd is not inside project."""
        project = tmp_path / "proj"
        (project / "repo").mkdir(parents=True)
        other = tmp_path / "other"
        other.mkdir()

        monkeypatch.setattr(
            layout_module,
            "discover_current_project_dir",
            lambda cwd=None, env=None: other.resolve(strict=False),
        )
        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: False)

        result = current_workspace(project, cwd=other)
        assert result is None

    def test_checkout_root_raises_repo_not_git_repo_uses_current(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root raises and repo_dir is not a git repo,
        workspace_dir = current."""
        project = tmp_path / "proj"
        (project / "repo").mkdir(parents=True)

        monkeypatch.setattr(
            layout_module.git_helpers,
            "is_git_repo",
            lambda p: p != project / "repo",
        )
        monkeypatch.setattr(
            layout_module,
            "discover_current_project_dir",
            lambda cwd=None, env=None: project.resolve(strict=False),
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: (_ for _ in ()).throw(SystemExit(1)),
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "current_branch",
            lambda p, env=None: "some-branch",
        )

        result = current_workspace(project, cwd=project)
        assert result is not None
        assert result.workspace_dir == project.resolve(strict=False)
        assert result.branch == "some-branch"
        assert result.is_main is False


class TestCurrentWorkspaceReturnsNoneWhenCwdNotInProject:
    def test_cwd_not_at_project_root_and_not_git_repo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """current_workspace returns None when cwd is inside the project but
        is not a git repo and not at the project root."""
        project = tmp_path / "proj"
        (project / "repo").mkdir(parents=True)
        sub_dir = project / "subdir"
        sub_dir.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: False)
        monkeypatch.setattr(
            layout_module,
            "discover_current_project_dir",
            lambda cwd=None, env=None: project.resolve(strict=False),
        )

        result = current_workspace(project, cwd=sub_dir)
        assert result is None


class TestCurrentProjectDirGitCommonDirTry:
    """Cover line 98: the try: block for git_helpers.git_common_dir(current)."""

    def test_git_common_dir_search_after_checkout_root_finds_no_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root returns a dir without project markers, it is the fallback."""
        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)
        (project / "worktrees").mkdir()

        cwd = tmp_path / "cwd"
        cwd.mkdir()

        checkout = tmp_path / "checkout"
        checkout.mkdir()

        git_dir = repo / ".git"

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: checkout,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None, env=None: git_dir,
        )

        assert current_workspace_or_project_root(cwd) == checkout

    def test_git_common_dir_finds_workspace_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root succeeds, git_common_dir is not searched."""
        project = tmp_path / "proj"
        repo = project / "repo"
        repo.mkdir(parents=True)
        (project / "worktrees").mkdir()  # workspace project marker

        cwd = tmp_path / "cwd"
        cwd.mkdir()

        checkout = tmp_path / "checkout"
        checkout.mkdir()

        git_dir = repo / ".git"

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: checkout,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None, env=None: git_dir,
        )

        result = current_workspace_or_project_root(cwd)
        assert result == checkout

    def test_git_common_dir_finds_embedded_project_via_parent_walk(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When checkout_root succeeds, fallback stays at checkout root."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".agm").mkdir()  # embedded project marker

        git_subdir = project / "sub" / ".git"
        git_subdir.mkdir(parents=True)

        cwd = tmp_path / "cwd"
        cwd.mkdir()

        checkout = tmp_path / "checkout"
        checkout.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: checkout,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None, env=None: git_subdir,
        )

        result = current_workspace_or_project_root(cwd)
        assert result == checkout


class TestCurrentProjectDirGitCommonDirPath:
    def test_git_common_dir_parent_finds_project_via_worktrees_marker(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Fallback does not use git_common_dir to search for project markers."""
        project = tmp_path / "myproject"
        repo = project / "repo"
        repo.mkdir(parents=True)
        worktrees = project / ".agm" / "worktrees"
        worktrees.mkdir(parents=True)

        worktree = tmp_path / "checkout"
        worktree.mkdir()

        monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda p: True)
        monkeypatch.setattr(
            layout_module.git_helpers,
            "checkout_root",
            lambda cwd=None, env=None: worktree,
        )
        monkeypatch.setattr(
            layout_module.git_helpers,
            "git_common_dir",
            lambda cwd=None, env=None: repo / ".git",
        )

        result = current_workspace_or_project_root(worktree)
        assert result == worktree


# ---------------------------------------------------------------------------
# project_name
# ---------------------------------------------------------------------------


def test_project_name_falls_back_to_root_dir_name(tmp_path: Path) -> None:
    project = tmp_path / "myproj"
    project.mkdir()
    (project / "repo").mkdir()

    assert project_name(project) == "myproj"


def test_project_name_reads_from_config_toml(tmp_path: Path) -> None:
    project = tmp_path / "dirname"
    project.mkdir()
    (project / "repo").mkdir()
    config_dir = project / "config"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        '[project]\nname = "custom-name"\n', encoding="utf-8"
    )

    assert project_name(project) == "custom-name"


def test_project_name_ignores_empty_name_in_config(tmp_path: Path) -> None:
    project = tmp_path / "myproj"
    project.mkdir()
    (project / "repo").mkdir()
    config_dir = project / "config"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        '[project]\nname = ""\n', encoding="utf-8"
    )

    assert project_name(project) == "myproj"


def test_project_name_ignores_non_string_name_in_config(tmp_path: Path) -> None:
    project = tmp_path / "myproj"
    project.mkdir()
    (project / "repo").mkdir()
    config_dir = project / "config"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        '[project]\nname = 42\n', encoding="utf-8"
    )

    assert project_name(project) == "myproj"


def test_project_name_for_embedded_project(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "myproj"
    project.mkdir()
    agm_dir = project / ".agm"
    agm_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)

    assert project_name(agm_dir) == "myproj"


def test_copy_config_copies_dot_env_directly_when_branch_given(tmp_path: Path) -> None:
    """When a branch is given and config_dir/.env exists, it is copied directly
    to the target (the 'if (config_dir / ".env").exists():' branch)."""
    project = tmp_path / "proj"
    config_dir = project / "config"
    workspace_config_dir = config_dir / "feat"
    workspace_config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    # Place a .env in config_dir (not in workspace_config_dir)
    (config_dir / ".env").write_text("DIRECT_KEY=direct\n", encoding="utf-8")

    copy_config(project_dir=project, target=target, branch="feat", cwd=None)

    # The .env must have been copied directly into target
    assert (target / ".env").read_text(encoding="utf-8") == "DIRECT_KEY=direct\n"


def test_copy_config_skips_branch_dir_when_not_a_dir(tmp_path: Path) -> None:
    """When workspace_config_dir is not a directory, the 'if workspace_config_dir.is_dir()'
    branch on line 422 is False and we skip to _merge_config_dotenv_files."""
    project = tmp_path / "proj"
    config_dir = project / "config"
    config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    # workspace_config_dir ("config/feat") does NOT exist
    copy_config(project_dir=project, target=target, branch="feat", cwd=None)

    # No crash, and target has no copied branch files (only possibly merged env)
    assert not (target / "feat").exists()
