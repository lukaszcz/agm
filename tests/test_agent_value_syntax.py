"""Host-facing syntax for values whose type is ``Agent``."""

from __future__ import annotations

import pytest

from agm.agent.spec import AgentClaude, AgentCodex, AgentPi
from agm.agent.values import agent_spec_shape, parse_agent_shorthand
from agm.agl.runtime.value_decode import host_text_to_json
from agm.agl.semantics.type_table import create_seeded_type_table
from agm.agl.semantics.types import BUILTIN_PRELUDE_TYPES
from agm.agl.type_schema import build_param_decoder
from agm.cli_support.agent_values import normalize_agent_source

_AGENT_DECODER = build_param_decoder(BUILTIN_PRELUDE_TYPES["Agent"], create_seeded_type_table())


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


def test_shorthand_text_decodes_through_host_text_to_json() -> None:
    result = host_text_to_json(
        "codex/o3-high",
        _AGENT_DECODER.decode,
        dict(_AGENT_DECODER.defs),
        agent_command_fallback=True,
    )

    assert result == agent_spec_shape(AgentCodex("o3", "high"))


def test_pi_shorthand_decodes_through_host_text_to_json() -> None:
    result = host_text_to_json(
        "anthropic/claude-opus-custom",
        _AGENT_DECODER.decode,
        dict(_AGENT_DECODER.defs),
        agent_command_fallback=True,
    )

    assert result == agent_spec_shape(AgentPi("anthropic", "claude-opus", "custom"))


def test_tagged_json_object_is_recognized_verbatim() -> None:
    text = '{"$case": "AgentClaude", "model": "sonnet", "thinking": "high"}'

    result = host_text_to_json(
        text, _AGENT_DECODER.decode, dict(_AGENT_DECODER.defs), agent_command_fallback=True
    )

    assert result == {"$case": "AgentClaude", "model": "sonnet", "thinking": "high"}


def test_call_form_constructor_selects_a_member() -> None:
    result = host_text_to_json(
        'AgentClaude(model = "sonnet", thinking = "high")',
        _AGENT_DECODER.decode,
        dict(_AGENT_DECODER.defs),
        agent_command_fallback=True,
    )

    assert result == agent_spec_shape(AgentClaude("sonnet", "high"))


def test_unrecognized_text_is_an_agent_command_verbatim() -> None:
    command = "custom-agent --model 'some model' \\%{SESSION_ID}"

    result = host_text_to_json(
        command, _AGENT_DECODER.decode, dict(_AGENT_DECODER.defs), agent_command_fallback=True
    )

    assert result == {"$case": "AgentCommand", "command": command}


def test_unrecognized_text_is_an_error_without_the_command_fallback() -> None:
    with pytest.raises(ValueError):
        host_text_to_json(
            "not an agent",
            _AGENT_DECODER.decode,
            dict(_AGENT_DECODER.defs),
            agent_command_fallback=False,
        )


def test_qualified_member_call_selects_a_member() -> None:
    result = host_text_to_json(
        'Agent::AgentClaude(model = "sonnet", thinking = "high")',
        _AGENT_DECODER.decode,
        dict(_AGENT_DECODER.defs),
        agent_command_fallback=True,
    )

    assert result == agent_spec_shape(AgentClaude("sonnet", "high"))


def test_unclosed_member_call_is_an_error_not_a_command() -> None:
    """Text opening a member call commits to it: a read failure inside the
    call (an unclosed parenthesis, here) is reported rather than silently
    falling back to a verbatim command, even with the fallback allowed."""
    with pytest.raises(ValueError):
        host_text_to_json(
            'AgentClaude(model = "x"',
            _AGENT_DECODER.decode,
            dict(_AGENT_DECODER.defs),
            agent_command_fallback=True,
        )


def test_misqualified_member_call_is_an_error_not_a_command() -> None:
    """A qualifier other than ``Agent`` on a member call is an error, even
    though its member name and call shape are otherwise well formed."""
    with pytest.raises(ValueError):
        host_text_to_json(
            'Foo::AgentClaude("a", "b")',
            _AGENT_DECODER.decode,
            dict(_AGENT_DECODER.defs),
            agent_command_fallback=True,
        )


@pytest.mark.parametrize("source", ["Agent :: AgentClaude", "Agent:: AgentClaude"])
def test_whitespace_around_double_colon_is_insignificant_in_a_qualified_call(
    source: str,
) -> None:
    result = host_text_to_json(
        f'{source}(model = "sonnet", thinking = "high")',
        _AGENT_DECODER.decode,
        dict(_AGENT_DECODER.defs),
        agent_command_fallback=True,
    )

    assert result == agent_spec_shape(AgentClaude("sonnet", "high"))


def test_leading_whitespace_before_a_call_is_insignificant() -> None:
    result = host_text_to_json(
        '  \n AgentClaude(model = "sonnet", thinking = "high")',
        _AGENT_DECODER.decode,
        dict(_AGENT_DECODER.defs),
        agent_command_fallback=True,
    )

    assert result == agent_spec_shape(AgentClaude("sonnet", "high"))


def test_text_opening_with_a_parenthesis_is_an_agent_command_verbatim() -> None:
    command = "(not a call)"

    result = host_text_to_json(
        command, _AGENT_DECODER.decode, dict(_AGENT_DECODER.defs), agent_command_fallback=True
    )

    assert result == {"$case": "AgentCommand", "command": command}


def test_malformed_qualifier_chain_is_an_agent_command_verbatim() -> None:
    """``Agent::(x)`` never lexically opens a member call (no name after
    ``::``), so it does not commit -- it falls back to a verbatim command."""
    command = "Agent::(x)"

    result = host_text_to_json(
        command, _AGENT_DECODER.decode, dict(_AGENT_DECODER.defs), agent_command_fallback=True
    )

    assert result == {"$case": "AgentCommand", "command": command}


def test_unqualified_non_member_call_is_an_agent_command_verbatim() -> None:
    command = "notamember(x)"

    result = host_text_to_json(
        command, _AGENT_DECODER.decode, dict(_AGENT_DECODER.defs), agent_command_fallback=True
    )

    assert result == {"$case": "AgentCommand", "command": command}


def test_multiple_agl_expressions_are_command_text_not_a_constructor() -> None:
    source = 'AgentClaude("sonnet", "high")\nAgentCodex("o3", "low")'

    assert normalize_agent_source(source) == (
        'AgentCommand("AgentClaude(\\"sonnet\\", \\"high\\")\\nAgentCodex(\\"o3\\", \\"low\\")")'
    )
