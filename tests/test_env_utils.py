"""Tests for environment and installation helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agm.core.env import (
    agm_installation_prefix,
    is_safe_shell_env_assignment_name,
    is_shell_identifier,
)
from agm.core.prompt import expand_prompt_env_vars


def test_agm_installation_prefix_uses_agm_binary_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    agm_executable = prefix / "bin" / "agm"
    agm_executable.parent.mkdir(parents=True)
    agm_executable.write_text("", encoding="utf-8")

    def fake_which(name: str) -> str | None:
        return str(agm_executable) if name == "agm" else None

    monkeypatch.setattr("agm.core.env.shutil.which", fake_which)

    assert agm_installation_prefix() == prefix


def test_agm_installation_prefix_returns_none_when_agm_is_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_which(_name: str) -> str | None:
        return None

    monkeypatch.setattr("agm.core.env.shutil.which", fake_which)

    assert agm_installation_prefix() is None


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


def test_expand_prompt_env_vars_replaces_known_vars_and_leaves_unknowns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KNOWN", "value")
    monkeypatch.setenv("OTHER_2", "two")
    monkeypatch.delenv("UNKNOWN", raising=False)

    expanded = expand_prompt_env_vars(
        "A=$KNOWN B=${OTHER_2} C=$UNKNOWN D=${UNKNOWN} E=$9 F=${NOT-VALID}",
        env=os.environ,
    )

    assert expanded == "A=value B=two C=$UNKNOWN D=${UNKNOWN} E=$9 F=${NOT-VALID}"
