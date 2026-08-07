"""Tests for the shared default-agent-runner resolution helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agent.config import default_agent_runner
from agm.agent.defaults import DEFAULT_AGENT_RUNNER
from agm.config.context import ConfigContext
from agm.config.general import load_merged_config


class TestDefaultAgentRunner:
    def test_falls_back_to_the_canonical_default_without_a_configured_runner(
        self, tmp_path: Path
    ) -> None:
        merged = load_merged_config(home=tmp_path, proj_dir=None, cwd=tmp_path)
        assert default_agent_runner(merged=merged) == DEFAULT_AGENT_RUNNER

    def test_uses_the_configured_loop_runner_from_an_already_merged_config(self) -> None:
        merged = {"loop": {"runner": "codex"}}
        assert default_agent_runner(merged=merged) == "codex"

    def test_without_merged_reads_the_ambient_config_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[loop]\nrunner = 'pi'\n")
        monkeypatch.setattr(
            "agm.agent.config.current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        assert default_agent_runner() == "pi"
