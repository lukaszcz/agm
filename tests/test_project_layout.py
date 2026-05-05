"""Comprehensive tests for agm.project.layout path-resolution helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import agm.project.layout as layout_module
from agm.project.layout import (
    CONFIG_FILES,
    _copy_existing_config_files,
    _merge_branch_env_file,
    _project_dir_from_checkout,
    _resolved_cwd,
    branch_session_name,
    branch_worktree_path,
    copy_config,
    current_project_dir,
    default_worktrees_dir,
    exit_if_main_checkout_branch,
    is_embedded_project,
    is_main_checkout_branch,
    is_project_dir,
    is_workspace_project,
    main_repo_dir,
    project_config_dir,
    project_data_dir,
    project_deps_dir,
    project_notes_dir,
    project_repo_dir,
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


# ---------------------------------------------------------------------------
# _project_dir_from_checkout
# ---------------------------------------------------------------------------


def test_project_dir_from_checkout_finds_agm_marker(tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    (project / ".agm").mkdir(parents=True)
    assert _project_dir_from_checkout(project) == project


def test_project_dir_from_checkout_finds_repo_subdir(tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    (project / "repo").mkdir(parents=True)
    assert _project_dir_from_checkout(project) == project


def test_project_dir_from_checkout_finds_parent_via_repo_name(tmp_path: Path) -> None:
    project = tmp_path / "myproject"
    repo_dir = project / "repo"
    (project / "worktrees").mkdir(parents=True)
    repo_dir.mkdir(parents=True)
    assert _project_dir_from_checkout(repo_dir) == project


def test_project_dir_from_checkout_finds_parent_via_repo_name_with_worktrees_hidden(
    tmp_path: Path,
) -> None:
    project = tmp_path / "myproject"
    repo_dir = project / "repo"
    (project / ".worktrees").mkdir(parents=True)
    repo_dir.mkdir(parents=True)
    assert _project_dir_from_checkout(repo_dir) == project


def test_project_dir_from_checkout_finds_parent_from_worktree_in_hidden_worktrees(
    tmp_path: Path,
) -> None:
    project = tmp_path / "myproject"
    worktree_dir = project / ".worktrees" / "feat"
    worktree_dir.mkdir(parents=True)
    assert _project_dir_from_checkout(worktree_dir) == project


def test_project_dir_from_checkout_finds_parent_from_worktree_in_worktrees(
    tmp_path: Path,
) -> None:
    project = tmp_path / "myproject"
    (project / "repo").mkdir(parents=True)
    worktree_dir = project / "worktrees" / "feat"
    worktree_dir.mkdir(parents=True)
    assert _project_dir_from_checkout(worktree_dir) == project


def test_project_dir_from_checkout_returns_none_for_plain_dir(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert _project_dir_from_checkout(plain) is None


# ---------------------------------------------------------------------------
# is_workspace_project
# ---------------------------------------------------------------------------


def test_is_workspace_project_true_when_repo_subdir_exists(tmp_path: Path) -> None:
    (tmp_path / "repo").mkdir()
    assert is_workspace_project(tmp_path) is True


def test_is_workspace_project_false_when_no_repo_subdir(tmp_path: Path) -> None:
    assert is_workspace_project(tmp_path) is False


def test_is_workspace_project_false_when_repo_is_file(tmp_path: Path) -> None:
    (tmp_path / "repo").write_text("not a dir", encoding="utf-8")
    assert is_workspace_project(tmp_path) is False


# ---------------------------------------------------------------------------
# is_embedded_project
# ---------------------------------------------------------------------------


def test_is_embedded_project_true_for_git_repo_with_agm_dir(
    tmp_path: Path, env: dict[str, str]
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)
    assert is_embedded_project(project) is True


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
    (project / ".agm").mkdir()
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
    (project / ".agm").mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, env=env, check=True)
    assert is_project_dir(project) is True


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
# project_data_dir
# ---------------------------------------------------------------------------


def test_project_data_dir_returns_agm_subdir_for_embedded_project(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()

    assert project_data_dir(project) == project / ".agm"


def test_project_data_dir_returns_project_dir_for_workspace_project(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    assert project_data_dir(project) == project


def test_project_data_dir_returns_project_dir_when_no_agm_marker(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()

    assert project_data_dir(project) == project


# ---------------------------------------------------------------------------
# project_repo_dir
# ---------------------------------------------------------------------------


def test_project_repo_dir_returns_repo_subdir_for_workspace_project(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    assert project_repo_dir(project) == project / "repo"


def test_project_repo_dir_returns_project_dir_for_embedded_project(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()

    assert project_repo_dir(project) == project


def test_project_repo_dir_returns_project_dir_when_no_repo_subdir(tmp_path: Path) -> None:
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


def test_main_repo_dir_is_alias_for_project_repo_dir_embedded(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()

    assert main_repo_dir(project) == project_repo_dir(project)


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
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()

    # embedded: data_dir == project/.agm
    assert default_worktrees_dir(project) == project / ".agm" / "worktrees"


# ---------------------------------------------------------------------------
# project_config_dir
# ---------------------------------------------------------------------------


def test_project_config_dir_workspace(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    assert project_config_dir(project) == project / "config"


def test_project_config_dir_embedded(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()

    assert project_config_dir(project) == project / ".agm" / "config"


# ---------------------------------------------------------------------------
# project_deps_dir
# ---------------------------------------------------------------------------


def test_project_deps_dir_workspace(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    assert project_deps_dir(project) == project / "deps"


def test_project_deps_dir_embedded(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()

    assert project_deps_dir(project) == project / ".agm" / "deps"


# ---------------------------------------------------------------------------
# project_notes_dir
# ---------------------------------------------------------------------------


def test_project_notes_dir_workspace(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "repo").mkdir()

    assert project_notes_dir(project) == project / "notes"


def test_project_notes_dir_embedded(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()

    assert project_notes_dir(project) == project / ".agm" / "notes"


# ---------------------------------------------------------------------------
# is_main_checkout_branch
# ---------------------------------------------------------------------------


def test_is_main_checkout_branch_true_for_repo_literal(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert is_main_checkout_branch(project, "repo", repo_branch="main") is True


def test_is_main_checkout_branch_true_when_branch_equals_repo_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert is_main_checkout_branch(project, "main", repo_branch="main") is True


def test_is_main_checkout_branch_true_for_custom_repo_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert is_main_checkout_branch(project, "master", repo_branch="master") is True


def test_is_main_checkout_branch_false_for_different_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert is_main_checkout_branch(project, "feat/x", repo_branch="main") is False


def test_is_main_checkout_branch_false_for_develop_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert is_main_checkout_branch(project, "develop", repo_branch="main") is False


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


def test_branch_worktree_path_for_worktree_branch_embedded(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / ".agm").mkdir(parents=True)

    result = branch_worktree_path(project, "feat/abc", repo_branch="main")

    assert result == project / ".agm" / "worktrees" / "feat/abc"


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


# ---------------------------------------------------------------------------
# exit_if_main_checkout_branch
# ---------------------------------------------------------------------------


def test_exit_if_main_checkout_branch_exits_for_repo_literal(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "repo").mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        exit_if_main_checkout_branch(project, "repo", repo_branch="main")

    assert exc_info.value.code == 1


def test_exit_if_main_checkout_branch_exits_for_main_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "repo").mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        exit_if_main_checkout_branch(project, "main", repo_branch="main")

    assert exc_info.value.code == 1


def test_exit_if_main_checkout_branch_does_not_exit_for_worktree_branch(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "repo").mkdir(parents=True)

    # Should not raise
    exit_if_main_checkout_branch(project, "feat/x", repo_branch="main")


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


def test_copy_existing_config_files_skips_empty_env(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    (source / ".env").write_text("", encoding="utf-8")
    (source / ".env.local").write_text("BAR=baz\n", encoding="utf-8")

    _copy_existing_config_files(source, target)

    assert not (target / ".env").exists()
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


def test_copy_existing_config_files_covers_all_config_files_entries(tmp_path: Path) -> None:
    """Verify all CONFIG_FILES names are candidates for copying."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    # Create all config files (non-empty)
    for name in CONFIG_FILES:
        (source / name).write_text(f"# {name}\n", encoding="utf-8")

    _copy_existing_config_files(source, target)

    for name in CONFIG_FILES:
        assert (target / name).exists(), f"Expected {name} to be copied"


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
    branch_config_dir = config_dir / "feat"
    branch_config_dir.mkdir(parents=True)
    (project / "repo").mkdir()
    target = tmp_path / "checkout"
    target.mkdir()

    (config_dir / ".env").write_text("BASE_KEY=base\n", encoding="utf-8")
    (branch_config_dir / ".env").write_text("BRANCH_KEY=branch\n", encoding="utf-8")

    copy_config(project_dir=project, target=target, branch="feat", cwd=None)

    target_env_content = (target / ".env").read_text(encoding="utf-8")
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


# ---------------------------------------------------------------------------
# current_project_dir – additional path walk coverage
# ---------------------------------------------------------------------------


def test_current_project_dir_from_nested_subdir_of_embedded_project(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".agm").mkdir()
    deep = project / "src" / "pkg" / "module"
    deep.mkdir(parents=True)

    assert current_project_dir(deep) == project


def test_current_project_dir_none_when_no_markers_and_not_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    monkeypatch.setattr(layout_module.git_helpers, "is_git_repo", lambda _path: False)

    # Returns the plain dir itself as fallback
    result = current_project_dir(plain)
    assert result == plain


def test_current_project_dir_checkout_root_finds_project_in_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When checkout_root returns a dir whose parent has project markers,
    the second for loop returns the project via return on line 98."""
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

    result = current_project_dir(cwd)
    assert result == project
