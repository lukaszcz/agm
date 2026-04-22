"""Tests for environment and installation helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agm.core.env import agm_installation_prefix
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
