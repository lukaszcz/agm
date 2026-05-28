"""Tests for runtime package metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_click_is_declared_as_runtime_dependency() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert any(dependency.split(">=", maxsplit=1)[0] == "click" for dependency in dependencies)
