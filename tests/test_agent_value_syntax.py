"""Host-facing syntax for values whose type is ``Agent``."""

from __future__ import annotations

import pytest

from agm.agent.spec import AgentClaude, AgentCodex, AgentCommand, AgentPi
from agm.agent.values import parse_agent_shorthand, parse_agent_text
from agm.cli_support.agent_values import normalize_agent_source


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("claude/sonnet-medium", AgentClaude("sonnet", "medium")),
        (
            "claude/claude-sonnet-4-5-experimental",
            AgentClaude("claude-sonnet-4-5", "experimental"),
        ),
        ("codex/o3-high", AgentCodex("o3", "high")),
        ("codex/gpt-5.4-extra-high", AgentCodex("gpt-5.4-extra", "high")),
        ("pi/openai/gpt-5-low", AgentPi("openai", "gpt-5", "low")),
        ("anthropic/claude-opus-custom", AgentPi("anthropic", "claude-opus", "custom")),
        ("pi/claude/sonnet-high", AgentPi("claude", "sonnet", "high")),
    ],
)
def test_agent_shorthand_selects_a_native_agent(text: str, expected: object) -> None:
    assert parse_agent_shorthand(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "claude/sonnet",
        "claude/-high",
        "claude/sonnet-",
        "codex/model/high",
        "pi/openai/gpt",
        "pi//gpt-high",
        "pi/openai/-high",
    ],
)
def test_incomplete_or_noncanonical_shorthand_does_not_match(text: str) -> None:
    assert parse_agent_shorthand(text) is None


def test_unrecognized_text_is_an_agent_command_verbatim() -> None:
    command = "custom-agent --model 'some model' \\%{SESSION_ID}"

    assert parse_agent_text(command) == AgentCommand(command)


def test_multiple_agl_expressions_are_command_text_not_a_constructor() -> None:
    source = 'AgentClaude("sonnet", "high")\nAgentCodex("o3", "low")'

    assert normalize_agent_source(source) == (
        'AgentCommand("AgentClaude(\\"sonnet\\", \\"high\\")\\nAgentCodex(\\"o3\\", \\"low\\")")'
    )
