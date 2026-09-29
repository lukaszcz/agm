"""Configured agent provider, model and effort defaults; strict ``[agent]`` validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agm.agent.spec import AgentClaude, AgentCodex, AgentCommand, AgentPi, AgentSpec
from agm.agent.spec_defaults import (
    AgentConfig,
    AgentConfigError,
    configured_defaults_resolver,
    with_configured_defaults,
)
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
    return configured_defaults_resolver({"agent": table})(spec)


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


_MODEL_TABLE: TomlDict = {
    "claude": {"model": "opus", "effort": "low", "opus": {"effort": "high"}},
    "codex": {"model": "gpt-5.1-codex", "gpt-5.1-codex": {"effort": "xhigh"}},
    "pi": {
        "provider": "anthropic",
        "model": "fallback",
        "effort": "low",
        "anthropic": {"model": "claude-sonnet-4-5", "claude-sonnet-4-5": {"effort": "high"}},
        "openai": {"effort": "medium"},
    },
}


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (AgentClaude("", ""), AgentClaude("opus", "high")),
        (AgentClaude("", "max"), AgentClaude("opus", "max")),
        (AgentClaude("sonnet", ""), AgentClaude("sonnet", "low")),
        (AgentCodex("", ""), AgentCodex("gpt-5.1-codex", "xhigh")),
        (AgentPi("", "", ""), AgentPi("anthropic", "claude-sonnet-4-5", "high")),
        (AgentPi("", "custom", ""), AgentPi("anthropic", "custom", "low")),
        (AgentPi("openai", "", ""), AgentPi("openai", "fallback", "medium")),
        (AgentPi("anthropic", "", "off"), AgentPi("anthropic", "claude-sonnet-4-5", "off")),
        (AgentPi("groq", "", ""), AgentPi("groq", "fallback", "low")),
    ],
)
def test_defaults_resolve_provider_then_model_then_effort_on_the_resolved_names(
    spec: AgentSpec, expected: AgentSpec
) -> None:
    assert _resolved(spec, _MODEL_TABLE) == expected


@pytest.mark.parametrize(
    "spec",
    [
        AgentClaude("haiku", "low"),
        AgentCodex("o3", "minimal"),
        AgentPi("openai", "gpt", "off"),
    ],
)
def test_explicit_names_and_thinking_are_never_overridden(spec: AgentSpec) -> None:
    assert _resolved(spec, _MODEL_TABLE) == spec


@pytest.mark.parametrize(
    ("table", "spec", "expected"),
    [
        ({"claude": {"model": ""}}, AgentClaude("", ""), AgentClaude("", "")),
        (
            {"claude": {"model": "", "effort": "low"}},
            AgentClaude("", ""),
            AgentClaude("", "low"),
        ),
        (
            {"pi": {"provider": "", "model": "m", "a": {"model": "x"}}},
            AgentPi("", "", ""),
            AgentPi("", "m", ""),
        ),
        (
            {"pi": {"provider": "a", "model": "m", "a": {"model": ""}}},
            AgentPi("", "", ""),
            AgentPi("a", "", ""),
        ),
    ],
)
def test_an_explicit_empty_name_stops_the_fall_through(
    table: TomlDict, spec: AgentSpec, expected: AgentSpec
) -> None:
    assert _resolved(spec, table) == expected


def test_a_provider_table_without_a_model_falls_through_to_the_pi_model() -> None:
    table: TomlDict = {"pi": {"provider": "a", "model": "m", "a": {"effort": "high"}}}

    assert _resolved(AgentPi("", "", ""), table) == AgentPi("a", "m", "high")


@pytest.mark.parametrize(
    "section",
    [
        "claude",
        [],
        ["claude"],
        {"effort": "high"},
        {"opencode": {"effort": "high"}},
        {"claude": "high"},
        {"claude": {"efort": "high"}},
        {"claude": {"opus": {"model": "x"}}},
        {"claude": {"provider": "x"}},
        {"claude": {"model": 3}},
        {"codex": {"o3": {"model": "x"}}},
        {"codex": {"provider": "x"}},
        {"pi": {"provider": True}},
        {"pi": {"model": ["m"]}},
        {"pi": {"anthropic": {"provider": "x"}}},
        {"pi": {"anthropic": {"model": 1}}},
        {"pi": {"anthropic": {"m": {"model": "x"}}}},
        {"pi": {"anthropic": {"m": {"provider": "x"}}}},
        {"claude": {"effort": 3}},
        {"codex": {"o3": {"effort": True}}},
        {"claude": {"opus": {"fast": {"effort": "high"}}}},
        {"codex": {"o3": {"x": {}}}},
        {"pi": {"anthropic": {"m": {"deeper": {"effort": "high"}}}}},
    ],
)
def test_a_malformed_agent_section_is_rejected(section: object) -> None:
    with pytest.raises(AgentConfigError):
        AgentConfig.from_merged({"agent": section})


def test_the_full_documented_nesting_is_accepted() -> None:
    AgentConfig.from_merged({"agent": _AGENT_TABLE})
    AgentConfig.from_merged({"agent": _MODEL_TABLE})


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

    assert with_configured_defaults(AgentClaude("opus", ""), config) == AgentClaude("opus", "high")
    assert with_configured_defaults(AgentClaude("haiku", ""), config) == AgentClaude("haiku", "low")
    assert with_configured_defaults(AgentCodex("gpt-5.1-codex", ""), config) == AgentCodex(
        "gpt-5.1-codex", "xhigh"
    )
    assert with_configured_defaults(AgentCodex("o3", ""), config) == AgentCodex("o3", "low")


def test_a_bracketed_model_table_configures_its_shorthand(tmp_path: Path) -> None:
    """A quoted ``[agent.claude."opus[1m]"]`` table addresses ``claude/opus[1m]``."""
    home = tmp_path / "home"
    (home / ".agm").mkdir(parents=True)
    (home / ".agm" / "config.toml").write_text('[agent.claude."opus[1m]"]\neffort = "high"\n')

    config = AgentConfig.from_merged(load_merged_config(home=home, proj_dir=None, cwd=tmp_path))

    spec = parse_agent_shorthand("claude/opus[1m]")
    assert spec is not None
    assert with_configured_defaults(spec, config) == AgentClaude("opus[1m]", "high")
    assert with_configured_defaults(AgentClaude("opus", ""), config) == AgentClaude("opus", "")
