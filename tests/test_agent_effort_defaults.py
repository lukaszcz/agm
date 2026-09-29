"""Configured default agent effort: lookup precedence and strict ``[agent]`` validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agm.agent.effort import (
    AgentConfig,
    AgentConfigError,
    default_effort_resolver,
    with_default_effort,
)
from agm.agent.spec import AgentClaude, AgentCodex, AgentCommand, AgentPi, AgentSpec
from agm.agent.values import parse_agent_shorthand
from agm.config.general import load_merged_config
from agm.core.toml import TomlDict

_AGENT_TABLE: TomlDict = {
    "claude": {"effort": "medium", "opus": {"effort": "high"}},
    "codex": {"effort": "low", "gpt-5.1-codex": {"effort": "xhigh"}},
    "pi": {
        "effort": "low",
        "anthropic": {"effort": "medium", "claude-sonnet-4-5": {"effort": "high"}},
    },
}


def _resolved(spec: AgentSpec, table: TomlDict = _AGENT_TABLE) -> AgentSpec:
    return default_effort_resolver({"agent": table})(spec)


@pytest.mark.parametrize(
    ("spec", "effort"),
    [
        (AgentClaude("opus", ""), "high"),
        (AgentClaude("sonnet", ""), "medium"),
        (AgentClaude("", ""), "medium"),
        (AgentCodex("gpt-5.1-codex", ""), "xhigh"),
        (AgentCodex("o3", ""), "low"),
        (AgentPi("anthropic", "claude-sonnet-4-5", ""), "high"),
        (AgentPi("anthropic", "claude-haiku", ""), "medium"),
        (AgentPi("openai", "gpt", ""), "low"),
        (AgentPi("", "claude-sonnet-4-5", ""), "low"),
    ],
)
def test_the_most_specific_configured_table_supplies_the_effort(
    spec: AgentClaude | AgentCodex | AgentPi, effort: str
) -> None:
    assert _resolved(spec) == replace(spec, thinking=effort)


@pytest.mark.parametrize(
    "spec",
    [
        AgentClaude("opus", "low"),
        AgentCodex("gpt-5.1-codex", "minimal"),
        AgentPi("anthropic", "claude-sonnet-4-5", "off"),
        AgentCommand("runner --effort none"),
    ],
)
def test_an_explicit_thinking_or_a_command_agent_is_never_touched(spec: AgentSpec) -> None:
    assert _resolved(spec) == spec


@pytest.mark.parametrize(
    "spec", [AgentClaude("opus", ""), AgentCodex("o3", ""), AgentPi("openai", "gpt", "")]
)
def test_nothing_configured_leaves_thinking_empty(spec: AgentSpec) -> None:
    assert _resolved(spec, {}) == spec


@pytest.mark.parametrize(
    ("table", "spec"),
    [
        ({"claude": {"effort": "low", "opus": {"effort": ""}}}, AgentClaude("opus", "")),
        (
            {"pi": {"effort": "low", "anthropic": {"effort": "", "m": {}}}},
            AgentPi("anthropic", "m", ""),
        ),
    ],
)
def test_an_explicit_empty_effort_stops_the_fall_through(table: TomlDict, spec: AgentSpec) -> None:
    assert _resolved(spec, table) == spec


def test_a_table_without_an_effort_falls_through() -> None:
    table: TomlDict = {"pi": {"effort": "low", "anthropic": {"m": {}}}}

    assert _resolved(AgentPi("anthropic", "m", ""), table) == AgentPi("anthropic", "m", "low")


@pytest.mark.parametrize(
    "table",
    [
        {"effort": "high"},
        {"opencode": {"effort": "high"}},
        {"claude": "high"},
        {"claude": {"efort": "high"}},
        {"claude": {"opus": {"model": "x"}}},
        {"claude": {"effort": 3}},
        {"codex": {"o3": {"effort": True}}},
        {"claude": {"opus": {"fast": {"effort": "high"}}}},
        {"codex": {"o3": {"x": {}}}},
        {"pi": {"anthropic": {"m": {"deeper": {"effort": "high"}}}}},
    ],
)
def test_a_malformed_agent_section_is_rejected(table: TomlDict) -> None:
    with pytest.raises(AgentConfigError):
        AgentConfig.from_merged({"agent": table})


def test_the_full_documented_nesting_is_accepted() -> None:
    AgentConfig.from_merged({"agent": _AGENT_TABLE})


def test_the_agent_section_merges_across_config_files(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text(
        '[agent.claude]\neffort = "low"\n[agent.codex]\neffort = "low"\n'
    )
    proj_dir = tmp_path / "proj"
    (proj_dir / "config").mkdir(parents=True)
    (proj_dir / "config" / "config.toml").write_text(
        '[agent.claude.opus]\neffort = "high"\n[agent.codex."gpt-5.1-codex"]\neffort = "xhigh"\n'
    )

    config = AgentConfig.from_merged(load_merged_config(home=home, proj_dir=proj_dir, cwd=tmp_path))

    assert with_default_effort(AgentClaude("opus", ""), config) == AgentClaude("opus", "high")
    assert with_default_effort(AgentClaude("haiku", ""), config) == AgentClaude("haiku", "low")
    assert with_default_effort(AgentCodex("gpt-5.1-codex", ""), config) == AgentCodex(
        "gpt-5.1-codex", "xhigh"
    )
    assert with_default_effort(AgentCodex("o3", ""), config) == AgentCodex("o3", "low")


def test_a_bracketed_model_table_configures_its_shorthand(tmp_path: Path) -> None:
    """A quoted ``[agent.claude."opus[1m]"]`` table addresses ``claude/opus[1m]``."""
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text('[agent.claude."opus[1m]"]\neffort = "high"\n')

    config = AgentConfig.from_merged(load_merged_config(home=home, proj_dir=None, cwd=tmp_path))

    spec = parse_agent_shorthand("claude/opus[1m]")
    assert spec is not None
    assert with_default_effort(spec, config) == AgentClaude("opus[1m]", "high")
    assert with_default_effort(AgentClaude("opus", ""), config) == AgentClaude("opus", "")
