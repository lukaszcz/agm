"""Comprehensive tests for agm.project.dependency_env."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

import pytest

import agm.core.fs as fs_mod
import agm.project.dependency_env as dep_env_module
from agm.project.dependency_env import (
    _dependency_config_checkout_name,
    _ensure_config_toml_file,
    _seed_from_parent_config,
    _set_toml_deps_value,
    config_toml_file,
    dep_env_var_name,
    ensure_dependency_configs_for_branch,
    load_dependency_toml_env,
    update_all_project_dependency_configs,
    update_dependency_config,
    update_dependency_configs_for_branch,
    update_dependency_toml_config,
    update_main_dependency_configs,
)
from agm.vcs.git import WorktreeInfo

# ---------------------------------------------------------------------------
# dep_env_var_name
# ---------------------------------------------------------------------------


class TestDepEnvVarName:
    def test_simple_lowercase(self) -> None:
        assert dep_env_var_name("mylib") == "MYLIB_DIR"

    def test_simple_uppercase(self) -> None:
        assert dep_env_var_name("MYLIB") == "MYLIB_DIR"

    def test_mixed_case(self) -> None:
        assert dep_env_var_name("MyLib") == "MYLIB_DIR"

    def test_hyphen_replaced_by_underscore(self) -> None:
        assert dep_env_var_name("my-lib") == "MY_LIB_DIR"

    def test_dot_replaced_by_underscore(self) -> None:
        assert dep_env_var_name("my.lib") == "MY_LIB_DIR"

    def test_space_replaced_by_underscore(self) -> None:
        assert dep_env_var_name("my lib") == "MY_LIB_DIR"

    def test_multiple_non_alnum_collapsed(self) -> None:
        assert dep_env_var_name("my--lib") == "MY_LIB_DIR"

    def test_leading_digit_gets_underscore_prefix(self) -> None:
        assert dep_env_var_name("1lib") == "_1LIB_DIR"

    def test_all_digits_gets_underscore_prefix(self) -> None:
        assert dep_env_var_name("123") == "_123_DIR"

    def test_empty_string_returns_dep(self) -> None:
        assert dep_env_var_name("") == "DEP_DIR"

    def test_only_special_chars_returns_dep(self) -> None:
        assert dep_env_var_name("---") == "DEP_DIR"

    def test_only_special_chars_mixed_returns_dep(self) -> None:
        assert dep_env_var_name("...") == "DEP_DIR"

    def test_underscores_preserved(self) -> None:
        assert dep_env_var_name("my_lib") == "MY_LIB_DIR"

    def test_leading_underscore_stripped_then_uppercased(self) -> None:
        # leading/trailing _ are stripped by .strip("_") before checking digit
        assert dep_env_var_name("_mylib") == "MYLIB_DIR"

    def test_numbers_in_middle(self) -> None:
        assert dep_env_var_name("lib2go") == "LIB2GO_DIR"

    def test_single_letter(self) -> None:
        assert dep_env_var_name("a") == "A_DIR"

    def test_digit_only_after_stripping(self) -> None:
        # "-1-" → strip "_" → "1" → starts with digit → "_1"
        assert dep_env_var_name("-1-") == "_1_DIR"

    def test_complex_name(self) -> None:
        assert dep_env_var_name("some-complex.dep_name") == "SOME_COMPLEX_DEP_NAME_DIR"


# ---------------------------------------------------------------------------
# config_toml_file
# ---------------------------------------------------------------------------


class TestConfigTomlFile:
    def test_no_branch_returns_config_dir_config_toml(self, tmp_path: Path) -> None:
        # split layout: data_dir = project_dir, config_dir = project_dir / "config"
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        result = config_toml_file(project_dir, None)
        assert result == project_dir / "config" / "config.toml"

    def test_branch_returns_config_dir_workspace_config_toml(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        result = config_toml_file(project_dir, "feat/x")
        assert result == project_dir / "config" / "feat/x" / "config.toml"

    def test_embedded_layout_uses_agm_subdir(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        agm_dir = project_dir / ".agm"
        agm_dir.mkdir()
        result = config_toml_file(agm_dir, None)
        assert result == agm_dir / "config" / "config.toml"

    def test_embedded_layout_with_branch(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        agm_dir = project_dir / ".agm"
        agm_dir.mkdir()
        result = config_toml_file(agm_dir, "main")
        assert result == agm_dir / "config" / "main" / "config.toml"


# ---------------------------------------------------------------------------
# _set_toml_deps_value
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _set_toml_deps_value
# ---------------------------------------------------------------------------


class TestSetTomlDepsValue:
    def test_empty_content_creates_deps_section(self) -> None:
        result = _set_toml_deps_value("", "mylib", "main")
        assert "[deps]" in result
        assert 'mylib = "main"' in result

    def test_adds_deps_section_to_existing_content(self) -> None:
        content = '[other]\nkey = "value"\n'
        result = _set_toml_deps_value(content, "mylib", "main")
        assert "[deps]" in result
        assert 'mylib = "main"' in result

    def test_updates_existing_dep_entry(self) -> None:
        content = '[deps]\nmylib = "old"\n'
        result = _set_toml_deps_value(content, "mylib", "new")
        assert 'mylib = "new"' in result
        assert 'mylib = "old"' not in result

    def test_adds_new_dep_to_existing_deps_section(self) -> None:
        content = '[deps]\nexisting = "branch"\n'
        result = _set_toml_deps_value(content, "newlib", "feat")
        assert 'existing = "branch"' in result
        assert 'newlib = "feat"' in result

    def test_dep_with_special_name_is_quoted(self) -> None:
        result = _set_toml_deps_value("", "feat/x", "branch")
        assert '"feat/x"' in result
        assert '"branch"' in result

    def test_preserves_other_sections_after_deps(self) -> None:
        content = '[deps]\nmylib = "main"\n\n[other]\nkey = "value"\n'
        result = _set_toml_deps_value(content, "mylib", "new")
        assert "[other]" in result
        assert 'key = "value"' in result
        assert 'mylib = "new"' in result

    def test_new_dep_inserted_before_next_section(self) -> None:
        content = '[deps]\nexisting = "branch"\n\n[other]\nkey = "value"\n'
        result = _set_toml_deps_value(content, "newlib", "feat")
        # newlib should appear before [other]
        assert result.index('newlib = "feat"') < result.index("[other]")

    def test_deps_section_at_end_of_file_no_trailing_newline(self) -> None:
        # content with no trailing newline — separator should be added
        content = '[other]\nkey = "value"'
        result = _set_toml_deps_value(content, "mylib", "main")
        assert "[deps]" in result
        assert 'mylib = "main"' in result

    def test_content_with_trailing_newline(self) -> None:
        content = '[other]\nkey = "value"\n'
        result = _set_toml_deps_value(content, "mylib", "main")
        assert "[deps]\n" in result

    def test_update_second_dep_in_section(self) -> None:
        content = '[deps]\nfirst = "a"\nsecond = "old"\n'
        result = _set_toml_deps_value(content, "second", "new")
        assert 'first = "a"' in result
        assert 'second = "new"' in result
        assert 'second = "old"' not in result

    def test_result_is_valid_toml(self, tmp_path: Path) -> None:
        import tomllib

        content = '[deps]\nexisting = "branch"\n'
        result = _set_toml_deps_value(content, "newlib", "feat")
        toml_file = tmp_path / "test.toml"
        toml_file.write_bytes(result.encode())
        with toml_file.open("rb") as f:
            parsed = tomllib.load(f)
        deps = cast(dict[str, str], parsed["deps"])
        assert deps["existing"] == "branch"
        assert deps["newlib"] == "feat"

    def test_update_result_is_valid_toml(self, tmp_path: Path) -> None:
        import tomllib

        content = '[deps]\nmylib = "old"\n'
        result = _set_toml_deps_value(content, "mylib", "new")
        toml_file = tmp_path / "test.toml"
        toml_file.write_bytes(result.encode())
        with toml_file.open("rb") as f:
            parsed = tomllib.load(f)
        deps = cast(dict[str, str], parsed["deps"])
        assert deps["mylib"] == "new"

    def test_deps_with_comment_header(self) -> None:
        content = '[deps] # my deps\nmylib = "old"\n'
        result = _set_toml_deps_value(content, "mylib", "new")
        assert 'mylib = "new"' in result
        assert 'mylib = "old"' not in result

    def test_deps_section_with_leading_spaces_in_header(self) -> None:
        content = '  [deps]\nmylib = "old"\n'
        result = _set_toml_deps_value(content, "mylib", "new")
        assert 'mylib = "new"' in result

    def test_multiple_calls_idempotent_on_same_dep(self) -> None:
        content = ""
        result1 = _set_toml_deps_value(content, "mylib", "main")
        result2 = _set_toml_deps_value(result1, "mylib", "main")
        assert result1 == result2


# ---------------------------------------------------------------------------
# load_toml_file
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# load_dependency_toml_env
# ---------------------------------------------------------------------------


class TestLoadDependencyTomlEnv:
    def _workspace_project(self, tmp_path: Path) -> Path:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        return project_dir

    def test_returns_env_unchanged_when_no_config_files(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        env: dict[str, str] = {"EXISTING": "value"}
        result = load_dependency_toml_env(
            project_dir=project_dir,
            config_files=[],
            env=env,
        )
        assert result == {"EXISTING": "value"}

    def test_returns_env_unchanged_when_config_file_missing(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        missing = project_dir / "config" / "config.toml"
        env: dict[str, str] = {}
        result = load_dependency_toml_env(
            project_dir=project_dir,
            config_files=[missing],
            env=env,
        )
        assert result == {}

    def test_loads_dep_branch_from_config_file(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\nmylib = "main"\n', encoding="utf-8")
        env: dict[str, str] = {}
        result = load_dependency_toml_env(
            project_dir=project_dir,
            config_files=[config_file],
            env=env,
        )
        expected_path = str(project_dir / "deps" / "mylib" / "main")
        assert result["MYLIB_DIR"] == expected_path

    def test_skips_empty_dep_branch(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\nmylib = ""\n', encoding="utf-8")
        env: dict[str, str] = {}
        result = load_dependency_toml_env(
            project_dir=project_dir,
            config_files=[config_file],
            env=env,
        )
        assert "MYLIB_DIR" not in result

    def test_skips_non_string_dep_branch(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("[deps]\nmylib = 42\n", encoding="utf-8")
        env: dict[str, str] = {}
        result = load_dependency_toml_env(
            project_dir=project_dir,
            config_files=[config_file],
            env=env,
        )
        assert "MYLIB_DIR" not in result

    def test_multiple_config_files_later_wins(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        workspace_config_dir = config_dir / "feat"
        workspace_config_dir.mkdir()
        main_file = config_dir / "config.toml"
        main_file.write_text('[deps]\nmylib = "main"\n', encoding="utf-8")
        workspace_config_file = workspace_config_dir / "config.toml"
        workspace_config_file.write_text('[deps]\nmylib = "feat"\n', encoding="utf-8")
        env: dict[str, str] = {}
        result = load_dependency_toml_env(
            project_dir=project_dir,
            config_files=[main_file, workspace_config_file],
            env=env,
        )
        expected_path = str(project_dir / "deps" / "mylib" / "feat")
        assert result["MYLIB_DIR"] == expected_path

    def test_multiple_deps_loaded(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\nlib-a = "main"\nlib-b = "dev"\n', encoding="utf-8")
        env: dict[str, str] = {}
        result = load_dependency_toml_env(
            project_dir=project_dir,
            config_files=[config_file],
            env=env,
        )
        assert result["LIB_A_DIR"] == str(project_dir / "deps" / "lib-a" / "main")
        assert result["LIB_B_DIR"] == str(project_dir / "deps" / "lib-b" / "dev")

    def test_preserves_existing_env_vars(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\nmylib = "main"\n', encoding="utf-8")
        env = {"EXISTING": "preserved"}
        result = load_dependency_toml_env(
            project_dir=project_dir,
            config_files=[config_file],
            env=env,
        )
        assert result["EXISTING"] == "preserved"

    def test_does_not_mutate_original_env(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\nmylib = "main"\n', encoding="utf-8")
        env: dict[str, str] = {}
        load_dependency_toml_env(
            project_dir=project_dir,
            config_files=[config_file],
            env=env,
        )
        assert env == {}

    def test_dep_with_special_name_creates_correct_env_var(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\n"my-dep" = "main"\n', encoding="utf-8")
        env: dict[str, str] = {}
        result = load_dependency_toml_env(
            project_dir=project_dir,
            config_files=[config_file],
            env=env,
        )
        assert result["MY_DEP_DIR"] == str(project_dir / "deps" / "my-dep" / "main")

    def test_no_deps_table_in_config(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text('[other]\nkey = "value"\n', encoding="utf-8")
        env: dict[str, str] = {}
        result = load_dependency_toml_env(
            project_dir=project_dir,
            config_files=[config_file],
            env=env,
        )
        assert result == {}


# ---------------------------------------------------------------------------
# update_dependency_toml_config / update_dependency_config
# ---------------------------------------------------------------------------


class TestUpdateDependencyTomlConfig:
    def _workspace_project(self, tmp_path: Path) -> Path:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        return project_dir

    def test_creates_config_toml_for_main(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        update_dependency_toml_config(
            project_dir=project_dir,
            dep_name="mylib",
            dep_branch="main",
            config_branch=None,
        )
        config_file = project_dir / "config" / "config.toml"
        assert config_file.exists()
        content = config_file.read_text(encoding="utf-8")
        assert 'mylib = "main"' in content

    def test_creates_config_toml_for_branch(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        update_dependency_toml_config(
            project_dir=project_dir,
            dep_name="mylib",
            dep_branch="feat",
            config_branch="feat",
        )
        config_file = project_dir / "config" / "feat" / "config.toml"
        assert config_file.exists()
        content = config_file.read_text(encoding="utf-8")
        assert 'mylib = "feat"' in content

    def test_updates_existing_dep_entry(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\nmylib = "old"\n', encoding="utf-8")
        update_dependency_toml_config(
            project_dir=project_dir,
            dep_name="mylib",
            dep_branch="new",
            config_branch=None,
        )
        content = config_file.read_text(encoding="utf-8")
        assert 'mylib = "new"' in content
        assert 'mylib = "old"' not in content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        # Workspace config directories don't exist yet
        update_dependency_toml_config(
            project_dir=project_dir,
            dep_name="mylib",
            dep_branch="main",
            config_branch="feat/nested",
        )
        config_file = project_dir / "config" / "feat" / "nested" / "config.toml"
        assert config_file.exists()

    def test_update_dependency_config_is_alias(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        update_dependency_config(
            project_dir=project_dir,
            dep_name="mylib",
            dep_branch="main",
            config_branch=None,
        )
        config_file = project_dir / "config" / "config.toml"
        assert config_file.exists()
        content = config_file.read_text(encoding="utf-8")
        assert 'mylib = "main"' in content


# ---------------------------------------------------------------------------
# _ensure_config_toml_file
# ---------------------------------------------------------------------------


class TestEnsureConfigTomlFile:
    def _workspace_project(self, tmp_path: Path) -> Path:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        return project_dir

    def test_creates_empty_config_toml_when_missing(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        _ensure_config_toml_file(project_dir, None)
        config_file = project_dir / "config" / "config.toml"
        assert config_file.exists()
        assert config_file.read_text(encoding="utf-8") == ""

    def test_does_not_overwrite_existing_config_toml(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\nmylib = "main"\n', encoding="utf-8")
        _ensure_config_toml_file(project_dir, None)
        content = config_file.read_text(encoding="utf-8")
        assert 'mylib = "main"' in content

    def test_creates_workspace_config_toml_when_missing(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        _ensure_config_toml_file(project_dir, "feat")
        config_file = project_dir / "config" / "feat" / "config.toml"
        assert config_file.exists()
        assert config_file.read_text(encoding="utf-8") == ""

    def test_creates_parent_dirs_for_branch(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        _ensure_config_toml_file(project_dir, "feat/nested")
        config_file = project_dir / "config" / "feat" / "nested" / "config.toml"
        assert config_file.exists()


# ---------------------------------------------------------------------------
# update_main_dependency_configs
# ---------------------------------------------------------------------------


class TestUpdateMainDependencyConfigs:
    def _workspace_project(self, tmp_path: Path) -> Path:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        return project_dir

    def test_does_nothing_when_no_deps_dir(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        # No deps dir — should not raise
        update_main_dependency_configs(project_dir)
        config_file = project_dir / "config" / "config.toml"
        assert not config_file.exists()

    def test_writes_dep_from_git_repo_in_dep_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir()
        branch_dir = dep_dir / "main"
        branch_dir.mkdir()
        # Create a real git repo so is_git_repo returns True
        subprocess.run(["git", "init", "-b", "main"], cwd=branch_dir, env=env, check=True)
        # Mock is_git_repo to avoid calling the real git
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: True)
        update_main_dependency_configs(project_dir)
        config_file = project_dir / "config" / "config.toml"
        assert config_file.exists()
        content = config_file.read_text(encoding="utf-8")
        assert "mylib" in content
        assert "main" in content

    def test_skips_dep_dir_with_no_git_repo(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir()
        # No git repos inside dep_dir
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: False)
        update_main_dependency_configs(project_dir)
        config_file = project_dir / "config" / "config.toml"
        assert not config_file.exists()

    def test_multiple_deps_all_written(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        for dep_name in ["liba", "libb"]:
            dep_dir = deps_dir / dep_name
            dep_dir.mkdir()
            branch_dir = dep_dir / "main"
            branch_dir.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=branch_dir, env=env, check=True)
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: True)
        update_main_dependency_configs(project_dir)
        config_file = project_dir / "config" / "config.toml"
        content = config_file.read_text(encoding="utf-8")
        assert "liba" in content
        assert "libb" in content


# ---------------------------------------------------------------------------
# update_dependency_configs_for_branch
# ---------------------------------------------------------------------------


class TestUpdateDependencyConfigsForBranch:
    def _workspace_project(self, tmp_path: Path) -> Path:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        return project_dir

    def test_does_nothing_when_no_deps_dir(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        # Should not raise
        update_dependency_configs_for_branch(project_dir=project_dir, branch="feat")
        config_file = project_dir / "config" / "feat" / "config.toml"
        assert not config_file.exists()

    def test_writes_dep_config_for_branch_with_matching_checkout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir()
        # Create a branch checkout directory matching the config branch
        branch_dir = dep_dir / "feat"
        branch_dir.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=branch_dir, env=env, check=True)
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: True)
        update_dependency_configs_for_branch(project_dir=project_dir, branch="feat")
        config_file = project_dir / "config" / "feat" / "config.toml"
        assert config_file.exists()
        content = config_file.read_text(encoding="utf-8")
        assert "mylib" in content
        assert "feat" in content

    def test_falls_back_to_main_checkout_when_no_branch_checkout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir()
        # Only a "main" checkout, no "feat" checkout
        main_dir = dep_dir / "main"
        main_dir.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=main_dir, env=env, check=True)
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: True)
        update_dependency_configs_for_branch(project_dir=project_dir, branch="feat")
        config_file = project_dir / "config" / "feat" / "config.toml"
        assert config_file.exists()
        content = config_file.read_text(encoding="utf-8")
        assert "mylib" in content
        assert "main" in content


# ---------------------------------------------------------------------------
# ensure_dependency_configs_for_branch
# ---------------------------------------------------------------------------


class TestEnsureDependencyConfigsForBranch:
    def _workspace_project(self, tmp_path: Path) -> Path:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        return project_dir

    def test_does_nothing_when_no_deps_dir(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        ensure_dependency_configs_for_branch(project_dir=project_dir, branch="feat")
        # No error and no config file created
        config_file = project_dir / "config" / "feat" / "config.toml"
        assert not config_file.exists()

    def test_fills_missing_inherited_dep_entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir()
        main_dir = dep_dir / "main"
        main_dir.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=main_dir, env=env, check=True)
        (project_dir / "config").mkdir()
        (project_dir / "config" / "config.toml").write_text(
            '[deps]\nmylib = "main"\n', encoding="utf-8"
        )
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: True)
        ensure_dependency_configs_for_branch(project_dir=project_dir, branch="feat")
        config_file = project_dir / "config" / "feat" / "config.toml"
        assert config_file.exists()
        content = config_file.read_text(encoding="utf-8")
        assert "mylib" in content

    def test_does_not_overwrite_existing_dep_entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir()
        main_dir = dep_dir / "main"
        main_dir.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=main_dir, env=env, check=True)
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: True)
        main_config_dir = project_dir / "config"
        main_config_dir.mkdir()
        (main_config_dir / "config.toml").write_text('[deps]\nmylib = "main"\n', encoding="utf-8")
        # Pre-create config with existing entry
        config_dir = project_dir / "config" / "feat"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\nmylib = "existing-branch"\n', encoding="utf-8")
        ensure_dependency_configs_for_branch(project_dir=project_dir, branch="feat")
        content = config_file.read_text(encoding="utf-8")
        assert 'mylib = "existing-branch"' in content

    def test_fills_missing_but_preserves_existing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        for dep_name in ["liba", "libb"]:
            dep_dir = deps_dir / dep_name
            dep_dir.mkdir()
            main_dir = dep_dir / "main"
            main_dir.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=main_dir, env=env, check=True)
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: True)
        main_config_dir = project_dir / "config"
        main_config_dir.mkdir()
        (main_config_dir / "config.toml").write_text(
            '[deps]\nliba = "main"\nlibb = "main"\n', encoding="utf-8"
        )
        # Pre-create config with only liba entry
        config_dir = project_dir / "config" / "feat"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\nliba = "custom-branch"\n', encoding="utf-8")
        ensure_dependency_configs_for_branch(project_dir=project_dir, branch="feat")
        content = config_file.read_text(encoding="utf-8")
        assert 'liba = "custom-branch"' in content
        assert "libb" in content

    def test_copies_config_from_parent_branch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        for dep_name in ["mylib"]:
            dep_dir = deps_dir / dep_name
            dep_dir.mkdir()
            main_dir = dep_dir / "main"
            main_dir.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=main_dir, env=env, check=True)
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: True)
        # Create parent workspace config with a specific dep version
        parent_config_dir = project_dir / "config" / "parent-branch"
        parent_config_dir.mkdir(parents=True)
        parent_config_file = parent_config_dir / "config.toml"
        parent_config_file.write_text('[deps]\nmylib = "dev"\n', encoding="utf-8")
        # New branch should inherit parent's dep config, not use filesystem fallback
        ensure_dependency_configs_for_branch(
            project_dir=project_dir,
            branch="child-branch",
            parent_branch="parent-branch",
        )
        config_file = project_dir / "config" / "child-branch" / "config.toml"
        assert config_file.exists()
        content = config_file.read_text(encoding="utf-8")
        assert 'mylib = "dev"' in content

    def test_parent_branch_does_not_inject_undeclared_dependency(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        for dep_name in ["declared", "undeclared"]:
            main_dir = deps_dir / dep_name / "main"
            main_dir.mkdir(parents=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=main_dir, env=env, check=True)
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: True)
        parent_config_dir = project_dir / "config" / "parent-branch"
        parent_config_dir.mkdir(parents=True)
        (parent_config_dir / "config.toml").write_text(
            '[deps]\ndeclared = "main"\n', encoding="utf-8"
        )

        ensure_dependency_configs_for_branch(
            project_dir=project_dir,
            branch="child-branch",
            parent_branch="parent-branch",
        )

        config_file = project_dir / "config" / "child-branch" / "config.toml"
        content = config_file.read_text(encoding="utf-8")
        assert "declared" in content
        assert "undeclared" not in content

    def test_parent_branch_copies_env_file(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        # Create parent workspace config with a .env file
        parent_config_dir = project_dir / "config" / "parent-branch"
        parent_config_dir.mkdir(parents=True)
        (parent_config_dir / "config.toml").write_text('[deps]\nmylib = "dev"\n', encoding="utf-8")
        (parent_config_dir / ".env").write_text("MY_VAR=from_parent\n", encoding="utf-8")
        ensure_dependency_configs_for_branch(
            project_dir=project_dir,
            branch="child-branch",
            parent_branch="parent-branch",
        )
        child_env_file = project_dir / "config" / "child-branch" / ".env"
        assert child_env_file.exists()
        assert "from_parent" in child_env_file.read_text(encoding="utf-8")

    def test_parent_branch_does_not_overwrite_existing_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir()
        main_dir = dep_dir / "main"
        main_dir.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=main_dir, env=env, check=True)
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: True)
        # Create parent workspace config
        parent_config_dir = project_dir / "config" / "parent-branch"
        parent_config_dir.mkdir(parents=True)
        parent_config_file = parent_config_dir / "config.toml"
        parent_config_file.write_text('[deps]\nmylib = "dev"\n', encoding="utf-8")
        # Pre-create child workspace config with different content
        child_config_dir = project_dir / "config" / "child-branch"
        child_config_dir.mkdir(parents=True)
        child_config_file = child_config_dir / "config.toml"
        child_config_file.write_text('[deps]\nmylib = "custom"\n', encoding="utf-8")
        ensure_dependency_configs_for_branch(
            project_dir=project_dir,
            branch="child-branch",
            parent_branch="parent-branch",
        )
        # Existing config should not be overwritten by parent
        content = child_config_file.read_text(encoding="utf-8")
        assert 'mylib = "custom"' in content

    def test_skips_inherited_dep_without_checkout_directory(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        config_dir = project_dir / "config"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[deps]\nmissing = "main"\n', encoding="utf-8")

        ensure_dependency_configs_for_branch(project_dir=project_dir, branch="feat")

        config_file = project_dir / "config" / "feat" / "config.toml"
        assert not config_file.exists()

    def test_no_parent_branch_uses_main_config_not_deps_filesystem(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir()
        main_dir = dep_dir / "main"
        main_dir.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=main_dir, env=env, check=True)
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: True)

        ensure_dependency_configs_for_branch(project_dir=project_dir, branch="feat")

        config_file = project_dir / "config" / "feat" / "config.toml"
        assert not config_file.exists()


# ---------------------------------------------------------------------------
# _seed_from_parent_config
# ---------------------------------------------------------------------------


class TestSeedFromParentConfig:
    def _workspace_project(self, tmp_path: Path) -> Path:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        return project_dir

    def test_copies_config_toml_from_parent(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        # Create parent workspace config
        parent_config_dir = project_dir / "config" / "parent"
        parent_config_dir.mkdir(parents=True)
        (parent_config_dir / "config.toml").write_text(
            '[deps]\nlib1 = "dev"\nlib2 = "v2"\n', encoding="utf-8"
        )
        _seed_from_parent_config(project_dir=project_dir, parent_branch="parent", branch="child")
        child_config_file = project_dir / "config" / "child" / "config.toml"
        assert child_config_file.exists()
        content = child_config_file.read_text(encoding="utf-8")
        assert 'lib1 = "dev"' in content
        assert 'lib2 = "v2"' in content

    def test_copies_env_files_from_parent(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        parent_config_dir = project_dir / "config" / "parent"
        parent_config_dir.mkdir(parents=True)
        (parent_config_dir / "config.toml").write_text("", encoding="utf-8")
        (parent_config_dir / ".env").write_text("FOO=bar\n", encoding="utf-8")
        (parent_config_dir / ".env.local").write_text("BAZ=qux\n", encoding="utf-8")
        _seed_from_parent_config(project_dir=project_dir, parent_branch="parent", branch="child")
        child_config_dir = project_dir / "config" / "child"
        assert (child_config_dir / ".env").exists()
        assert (child_config_dir / ".env.local").exists()
        assert (child_config_dir / ".env").read_text(encoding="utf-8") == "FOO=bar\n"

    def test_does_not_copy_if_parent_config_missing(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        _seed_from_parent_config(
            project_dir=project_dir, parent_branch="nonexistent", branch="child"
        )
        child_config_dir = project_dir / "config" / "child"
        assert not child_config_dir.exists()

    def test_does_not_copy_if_child_config_already_populated(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        # Parent config
        parent_config_dir = project_dir / "config" / "parent"
        parent_config_dir.mkdir(parents=True)
        (parent_config_dir / "config.toml").write_text('[deps]\nlib = "dev"\n', encoding="utf-8")
        # Child config already exists with content
        child_config_dir = project_dir / "config" / "child"
        child_config_dir.mkdir(parents=True)
        (child_config_dir / "config.toml").write_text(
            '[deps]\nlib = "existing"\n', encoding="utf-8"
        )
        _seed_from_parent_config(project_dir=project_dir, parent_branch="parent", branch="child")
        content = (child_config_dir / "config.toml").read_text(encoding="utf-8")
        assert 'lib = "existing"' in content

    def test_copies_to_empty_child_dir(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        parent_config_dir = project_dir / "config" / "parent"
        parent_config_dir.mkdir(parents=True)
        (parent_config_dir / "config.toml").write_text('[deps]\nlib = "dev"\n', encoding="utf-8")
        # Child dir exists but is empty (e.g. created by mkdir)
        child_config_dir = project_dir / "config" / "child"
        child_config_dir.mkdir(parents=True)
        _seed_from_parent_config(project_dir=project_dir, parent_branch="parent", branch="child")
        assert (child_config_dir / "config.toml").exists()

    def test_does_not_copy_subdirectories(self, tmp_path: Path) -> None:
        project_dir = self._workspace_project(tmp_path)
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        parent_config_dir = project_dir / "config" / "parent"
        parent_config_dir.mkdir(parents=True)
        (parent_config_dir / "config.toml").write_text("", encoding="utf-8")
        # A subdirectory in parent config (should not be copied)
        (parent_config_dir / "subdir").mkdir()
        _seed_from_parent_config(project_dir=project_dir, parent_branch="parent", branch="child")
        child_config_dir = project_dir / "config" / "child"
        assert not (child_config_dir / "subdir").exists()


# ---------------------------------------------------------------------------
# update_all_project_dependency_configs
# ---------------------------------------------------------------------------


class TestUpdateAllProjectDependencyConfigs:
    def _workspace_project(self, tmp_path: Path) -> Path:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        return project_dir

    def test_creates_main_config_toml(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = self._workspace_project(tmp_path)

        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "current_branch",
            lambda _repo_dir, **_kwargs: "main",
        )
        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "worktree_list",
            lambda _repo_dir, **_kwargs: [],
        )
        update_all_project_dependency_configs(project_dir)
        config_file = project_dir / "config" / "config.toml"
        assert config_file.exists()

    def test_creates_workspace_config_toml_for_worktrees(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        worktrees_dir = project_dir / "worktrees"
        worktrees_dir.mkdir()
        feat_dir = worktrees_dir / "feat"
        feat_dir.mkdir()

        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "current_branch",
            lambda _repo_dir, **_kwargs: "main",
        )
        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "worktree_list",
            lambda _repo_dir, **_kwargs: [
                WorktreeInfo(path=feat_dir, branch="feat"),
            ],
        )
        update_all_project_dependency_configs(project_dir)
        workspace_config_file = project_dir / "config" / "feat" / "config.toml"
        assert workspace_config_file.exists()

    def test_skips_workspace_config_for_main_branch_worktree(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        worktrees_dir = project_dir / "worktrees"
        worktrees_dir.mkdir()
        main_wt_dir = worktrees_dir / "main"
        main_wt_dir.mkdir()

        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "current_branch",
            lambda _repo_dir, **_kwargs: "main",
        )
        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "worktree_list",
            lambda _repo_dir, **_kwargs: [
                WorktreeInfo(path=main_wt_dir, branch="main"),
            ],
        )
        update_all_project_dependency_configs(project_dir)
        # Workspace config for "main" should NOT be created
        workspace_config_file = project_dir / "config" / "main" / "config.toml"
        assert not workspace_config_file.exists()

    def test_skips_worktree_with_no_branch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        worktrees_dir = project_dir / "worktrees"
        worktrees_dir.mkdir()
        detached_dir = worktrees_dir / "detached"
        detached_dir.mkdir()

        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "current_branch",
            lambda _repo_dir, **_kwargs: "main",
        )
        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "worktree_list",
            lambda _repo_dir, **_kwargs: [
                WorktreeInfo(path=detached_dir, branch=None),
            ],
        )
        update_all_project_dependency_configs(project_dir)
        # No workspace config should be created
        assert not (project_dir / "config" / "detached").exists()

    def test_skips_worktree_outside_worktrees_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        # A worktree that is not in the default worktrees dir
        outside_dir = tmp_path / "outside-worktree"
        outside_dir.mkdir()

        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "current_branch",
            lambda _repo_dir, **_kwargs: "main",
        )
        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "worktree_list",
            lambda _repo_dir, **_kwargs: [
                WorktreeInfo(path=outside_dir, branch="feat"),
            ],
        )
        update_all_project_dependency_configs(project_dir)
        # No workspace config should be created for external worktrees
        workspace_config_file = project_dir / "config" / "feat" / "config.toml"
        assert not workspace_config_file.exists()

    def test_passes_env_to_git_calls(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = self._workspace_project(tmp_path)
        captured_envs: list[dict[str, str] | None] = []

        def fake_current_branch(_repo_dir: Path, *, env: dict[str, str] | None = None) -> str:
            captured_envs.append(env)
            return "main"

        def fake_worktree_list(
            _repo_dir: Path, *, env: dict[str, str] | None = None
        ) -> list[WorktreeInfo]:
            captured_envs.append(env)
            return []

        monkeypatch.setattr(dep_env_module.git_helpers, "current_branch", fake_current_branch)
        monkeypatch.setattr(dep_env_module.git_helpers, "worktree_list", fake_worktree_list)

        test_env = {"MY_VAR": "value"}
        update_all_project_dependency_configs(project_dir, env=test_env)

        assert all(e == test_env for e in captured_envs)

    def test_worktrees_dir_is_the_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Worktrees directory is project_dir/worktrees (split layout)."""
        project_dir = self._workspace_project(tmp_path)
        worktrees_dir = project_dir / "worktrees"
        worktrees_dir.mkdir()
        feat_dir = worktrees_dir / "feat"
        feat_dir.mkdir()

        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "current_branch",
            lambda _repo_dir, **_kwargs: "main",
        )
        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "worktree_list",
            lambda _repo_dir, **_kwargs: [
                WorktreeInfo(path=feat_dir, branch="feat"),
            ],
        )
        update_all_project_dependency_configs(project_dir)
        workspace_config = project_dir / "config" / "feat" / "config.toml"
        assert workspace_config.exists()


class TestDependencyConfigCheckoutNameFallback:
    def test_falls_back_to_main_when_branch_path_not_git_repo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        dep_dir = tmp_path / "dep"
        dep_dir.mkdir()
        main_dir = dep_dir / "main"
        main_dir.mkdir()
        (main_dir / ".git").mkdir()  # .git marker so dependency_repo_paths finds it
        feat_dir = dep_dir / "feat"
        feat_dir.mkdir()
        # main is a git repo, feat is not
        monkeypatch.setattr(
            dep_env_module.git_helpers,
            "is_git_repo",
            lambda p: p == main_dir,
        )
        result = _dependency_config_checkout_name(dep_dir, "feat")
        assert result == "main"

    def test_returns_none_when_no_repos(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        dep_dir = tmp_path / "dep"
        dep_dir.mkdir()
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda p: False)
        result = _dependency_config_checkout_name(dep_dir, "feat")
        assert result is None


class TestEnsureConfigTomlFileCoverage:
    def test_does_not_overwrite_existing_file(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        config_dir = project_dir / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\nfoo = "bar"\n', encoding="utf-8")
        _ensure_config_toml_file(project_dir, None)
        # Content unchanged
        assert 'foo = "bar"' in config_file.read_text(encoding="utf-8")


class TestUpdateMainDependencyConfigsWithExistingBranch:
    def test_skips_dep_when_existing_branch_is_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When dep already has a branch in config, it is skipped."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        config_dir = project_dir / "config"
        config_dir.mkdir()
        deps_dir = project_dir / "deps"
        (deps_dir / "mylib").mkdir(parents=True)

        # Write a config.toml with mylib already set
        config_file = config_dir / "config.toml"
        config_file.write_text('[deps]\nmylib = "feat"\n', encoding="utf-8")

        updated: list[str] = []
        monkeypatch.setattr(
            dep_env_module,
            "project_deps_dir",
            lambda pd: deps_dir,
        )
        monkeypatch.setattr(
            dep_env_module,
            "_dependency_config_checkout_name",
            lambda dep_dir, branch: "main",
        )
        monkeypatch.setattr(
            dep_env_module,
            "update_dependency_toml_config",
            lambda **kwargs: updated.append(kwargs["dep_name"]),
        )
        monkeypatch.setattr(
            dep_env_module,
            "config_toml_file",
            lambda pd, branch: config_file,
        )

        dep_env_module.update_main_dependency_configs(project_dir)
        # mylib already has "feat" branch, so it should be skipped
        assert "mylib" not in updated


class TestDependencyConfigCheckoutNameNotGitRepo:
    def test_falls_back_to_main_when_branch_path_is_not_git_repo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """_dependency_config_checkout_name falls back to _main_dependency_checkout_name
        when branch_path doesn't have .git or isn't a git repo."""
        dep_dir = tmp_path / "dep"
        dep_dir.mkdir()
        feat_dir = dep_dir / "feat"
        feat_dir.mkdir()
        main_subdir = dep_dir / "main"
        main_subdir.mkdir()
        (main_subdir / ".git").mkdir()

        # branch_path/feat/.git doesn't exist => falls back to _main_dependency_checkout_name
        real_exists = fs_mod.exists

        def fake_exists(p: Path) -> bool:
            if str(p) == str(feat_dir / ".git"):
                return False  # Feat is not a git repo at all
            return real_exists(p)

        monkeypatch.setattr(fs_mod, "exists", fake_exists)
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda p: p == main_subdir)

        result = dep_env_module._dependency_config_checkout_name(dep_dir, "feat")
        assert result == "main"


class TestEnsureDependencyConfigsForBranchSkipsNone:
    def test_skips_dep_when_checkout_name_is_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """ensure_dependency_configs_for_branch skips deps where checkout_name is None."""
        from typing import Any

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        deps_dir = project_dir / "deps"
        dep_dir = deps_dir / "mylib"
        dep_dir.mkdir(parents=True)
        config_dir = project_dir / "config"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[deps]\nmylib = "main"\n', encoding="utf-8")

        # Make _dependency_config_checkout_name return None
        monkeypatch.setattr(
            dep_env_module,
            "_dependency_config_checkout_name",
            lambda dep_dir, branch: None,
        )

        # Track whether update_dependency_toml_config is called
        update_calls: list[Any] = []
        monkeypatch.setattr(
            dep_env_module,
            "update_dependency_toml_config",
            lambda **kwargs: update_calls.append(kwargs),
        )

        dep_env_module.ensure_dependency_configs_for_branch(project_dir=project_dir, branch="feat")
        # No update calls since checkout_name was None
        assert update_calls == []


class TestDependencyConfigCheckoutNameHappyPath:
    def test_returns_checkout_name_when_branch_is_git_repo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """_dependency_config_checkout_name returns checkout name when branch is a git repo."""
        dep_dir = tmp_path / "dep"
        dep_dir.mkdir()
        feat_dir = dep_dir / "feat"
        feat_dir.mkdir()
        # Create .git so exists check passes
        (feat_dir / ".git").mkdir()

        monkeypatch.setattr(fs_mod, "exists", lambda p: True)
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda p: True)

        result = dep_env_module._dependency_config_checkout_name(dep_dir, "feat")
        assert result == "feat"


class TestUpdateDependencyConfigsForBranchNoCheckout:
    def test_skips_dep_with_no_matching_checkout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "repo").mkdir()
        deps_dir = project_dir / "deps"
        deps_dir.mkdir()
        dep_dir = deps_dir / "orphan"
        dep_dir.mkdir()
        # No checkout directories at all - _dependency_config_checkout_name returns None
        monkeypatch.setattr(dep_env_module.git_helpers, "is_git_repo", lambda _: False)
        update_dependency_configs_for_branch(project_dir=project_dir, branch="feat")
        # No config should be created
        config_file = project_dir / "config" / "feat" / "config.toml"
        assert not config_file.exists()


# ===========================================================================
# [project].name helpers
# ===========================================================================


class TestSetTomlProjectName:
    def test_adds_to_empty_content(self) -> None:
        result = dep_env_module._set_toml_project_name("", "my-proj")
        assert "[project]" in result
        assert 'name = "my-proj"' in result

    def test_adds_to_existing_content(self) -> None:
        content = '[deps]\nfoo = "bar"\n'
        result = dep_env_module._set_toml_project_name(content, "hello")
        assert "[project]" in result
        assert 'name = "hello"' in result
        assert "[deps]" in result

    def test_updates_existing_name(self) -> None:
        content = '[project]\nname = "old"\n'
        result = dep_env_module._set_toml_project_name(content, "new")
        assert 'name = "new"' in result
        assert "old" not in result

    def test_adds_name_before_next_section(self) -> None:
        content = '[project]\nexisting = "yes"\n[deps]\nfoo = "bar"\n'
        result = dep_env_module._set_toml_project_name(content, "mine")
        lines = result.splitlines()
        project_idx = next(i for i, ln in enumerate(lines) if ln == "[project]")
        deps_idx = next(i for i, ln in enumerate(lines) if ln == "[deps]")
        name_idx = next(i for i, ln in enumerate(lines) if 'name = "mine"' in ln)
        assert project_idx < name_idx < deps_idx

    def test_appends_after_existing_keys(self) -> None:
        content = '[project]\nalpha = "1"\n'
        result = dep_env_module._set_toml_project_name(content, "beta")
        lines = result.splitlines()
        assert any("alpha" in ln for ln in lines)
        assert any('name = "beta"' in ln for ln in lines)


class TestEnsureProjectNameInConfig:
    def test_creates_config_with_project_name(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "config").mkdir()

        dep_env_module.ensure_project_name_in_config(project_dir=project_dir, name="myproj")

        config_file = project_dir / "config" / "config.toml"
        assert config_file.is_file()
        content = config_file.read_text(encoding="utf-8")
        assert "[project]" in content
        assert 'name = "myproj"' in content

    def test_updates_existing_config(self, tmp_path: Path) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        config_dir = project_dir / "config"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[deps]\nfoo = "bar"\n', encoding="utf-8")

        dep_env_module.ensure_project_name_in_config(project_dir=project_dir, name="hello")

        content = (config_dir / "config.toml").read_text(encoding="utf-8")
        assert 'name = "hello"' in content
        assert 'foo = "bar"' in content
