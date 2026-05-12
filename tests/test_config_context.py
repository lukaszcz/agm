from __future__ import annotations

from pathlib import Path

import pytest

import agm.config.context as context_module
from agm.config.context import current_config_context


def test_current_config_context_uses_explicit_proj_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"

    context = current_config_context(
        cwd=tmp_path,
        env={"HOME": str(home), "PROJ_DIR": str(project)},
    )

    assert context.home == home
    assert context.proj_dir == project
    assert context.cwd == tmp_path


def test_current_config_context_uses_discovered_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    project = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.setattr(context_module, "current_project_dir", lambda cwd: project)

    context = current_config_context(cwd=cwd, env={"HOME": str(home)})

    assert context.proj_dir == project


def test_current_config_context_uses_none_when_discovery_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"

    def fail(_cwd: Path) -> Path:
        raise SystemExit(1)

    monkeypatch.setattr(context_module, "current_project_dir", fail)

    context = current_config_context(cwd=tmp_path, env={"HOME": str(home)})

    assert context.proj_dir is None
