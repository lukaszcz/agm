"""Tests for environment and installation helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agm.core.env import (
    agm_installation_prefix,
    is_safe_shell_env_assignment_name,
    is_shell_identifier,
    source_env_files,
)


def test_agm_installation_prefix_uses_running_executable_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    agm_executable = prefix / "bin" / "agm"
    agm_executable.parent.mkdir(parents=True)
    agm_executable.write_text("", encoding="utf-8")

    monkeypatch.setattr("agm.core.env.sys.argv", [str(agm_executable), "config", "env"])

    assert agm_installation_prefix() == prefix


def test_agm_installation_prefix_uses_symlink_location_without_resolving_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    tool_executable = tmp_path / "uv-tools" / "agm"
    tool_executable.parent.mkdir(parents=True)
    tool_executable.write_text("", encoding="utf-8")
    agm_executable = prefix / "bin" / "agm"
    agm_executable.parent.mkdir(parents=True)
    agm_executable.symlink_to(tool_executable)
    monkeypatch.setattr("agm.core.env.sys.argv", [str(agm_executable)])

    assert agm_installation_prefix() == prefix


def test_agm_installation_prefix_ignores_a_different_agm_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A development build reports its own prefix, never an installed one on PATH."""

    venv_prefix = tmp_path / "worktree" / ".venv"
    venv_executable = venv_prefix / "bin" / "agm"
    venv_executable.parent.mkdir(parents=True)
    venv_executable.write_text("", encoding="utf-8")
    installed_executable = tmp_path / "installed" / "bin" / "agm"
    installed_executable.parent.mkdir(parents=True)
    installed_executable.write_text("", encoding="utf-8")

    monkeypatch.setattr("agm.core.env.sys.argv", [str(venv_executable)])
    monkeypatch.setattr(shutil, "which", lambda _name, **_kwargs: str(installed_executable))

    assert agm_installation_prefix() == venv_prefix


def test_agm_installation_prefix_returns_none_without_an_executable_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agm.core.env.sys.argv", [""])

    assert agm_installation_prefix() is None


def test_agm_installation_prefix_returns_none_for_a_script_outside_a_bin_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script run from a source tree has no installation prefix at all.

    Treating its grandparent as a prefix would invent a runtime tree — and an
    AGM home — beside an arbitrary directory.
    """
    script = tmp_path / "repo" / "tools" / "install_agm_config.py"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")

    monkeypatch.setattr("agm.core.env.sys.argv", [str(script)])

    assert agm_installation_prefix() is None


def test_agm_installation_prefix_returns_none_for_an_executable_beside_its_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "prefix" / "agm"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")

    monkeypatch.setattr("agm.core.env.sys.argv", [str(executable)])

    assert agm_installation_prefix() is None


def test_agm_installation_prefix_accepts_a_relative_bin_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``bin`` check reads the invocation path, however it was spelled."""
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "agm").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr("agm.core.env.sys.argv", ["prefix/bin/agm"])

    assert agm_installation_prefix() == prefix


def test_is_shell_identifier_accepts_shell_variable_names() -> None:
    assert is_shell_identifier("NAME")
    assert is_shell_identifier("_NAME_2")


def test_is_shell_identifier_rejects_non_assignable_names() -> None:
    assert not is_shell_identifier("BAD-NAME")
    assert not is_shell_identifier("1_BAD")
    assert not is_shell_identifier("")


def test_is_safe_shell_env_assignment_name_rejects_shell_managed_names() -> None:
    assert not is_safe_shell_env_assignment_name("_")
    assert not is_safe_shell_env_assignment_name("PWD")
    assert not is_safe_shell_env_assignment_name("OLDPWD")
    assert not is_safe_shell_env_assignment_name("SHLVL")
    assert not is_safe_shell_env_assignment_name("UID")
    assert not is_safe_shell_env_assignment_name("EUID")
    assert not is_safe_shell_env_assignment_name("PPID")
    assert not is_safe_shell_env_assignment_name("BASHOPTS")
    assert not is_safe_shell_env_assignment_name("BASHPID")
    assert not is_safe_shell_env_assignment_name("SHELLOPTS")
    assert not is_safe_shell_env_assignment_name("BAD-NAME")
    assert is_safe_shell_env_assignment_name("PROJECT_ENV")


def test_source_env_files_lets_a_later_file_override_an_earlier_one(tmp_path: Path) -> None:
    """Files are sourced in the order given, so the last assignment wins."""
    first = tmp_path / "first.sh"
    first.write_text("export AGM_TEST_ORDER=first\nexport AGM_TEST_ONLY_FIRST=yes\n")
    second = tmp_path / "second.sh"
    second.write_text("export AGM_TEST_ORDER=second\n")

    forwards = source_env_files([first, second], env={})
    backwards = source_env_files([second, first], env={})

    assert forwards["AGM_TEST_ORDER"] == "second"
    assert forwards["AGM_TEST_ONLY_FIRST"] == "yes"
    assert backwards["AGM_TEST_ORDER"] == "first"


def test_source_env_files_shares_shell_state_between_files(tmp_path: Path) -> None:
    """A single shell sources every file, so a later file sees earlier shell variables."""
    first = tmp_path / "first.sh"
    first.write_text("AGM_TEST_SHELL_LOCAL=shared\n")
    second = tmp_path / "second.sh"
    second.write_text('export AGM_TEST_DERIVED="$AGM_TEST_SHELL_LOCAL"\n')

    result = source_env_files([first, second], env={})

    assert result["AGM_TEST_DERIVED"] == "shared"


def test_source_env_files_skips_a_missing_file_between_existing_ones(tmp_path: Path) -> None:
    first = tmp_path / "first.sh"
    first.write_text("export AGM_TEST_FIRST=1\nexport AGM_TEST_ORDER=first\n")
    last = tmp_path / "last.sh"
    last.write_text("export AGM_TEST_LAST=1\nexport AGM_TEST_ORDER=last\n")

    result = source_env_files([first, tmp_path / "absent.sh", last], env={})

    assert result["AGM_TEST_FIRST"] == "1"
    assert result["AGM_TEST_LAST"] == "1"
    assert result["AGM_TEST_ORDER"] == "last"
