"""Host-facing syntax for values whose type is ``Agent``."""

from __future__ import annotations

import pytest

from agm.agent.spec import AgentClaude, AgentCodex, AgentCommand, AgentPi, AgentSpec
from agm.agent.values import AgentShorthandError, agent_spec_shape, parse_agent_shorthand
from agm.agl.runtime.value_decode import ValueDecodeError, host_text_to_json
from agm.agl.semantics.type_table import create_seeded_type_table
from agm.agl.semantics.types import BUILTIN_PRELUDE_TYPES
from agm.agl.type_schema import build_param_decoder

_AGENT_DECODER = build_param_decoder(BUILTIN_PRELUDE_TYPES["Agent"], create_seeded_type_table())


def _decode(text: str, *, fallback: bool = True) -> object:
    return host_text_to_json(
        text, _AGENT_DECODER.decode, dict(_AGENT_DECODER.defs), agent_command_fallback=fallback
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("claude/opus", AgentClaude("opus", "")),
        ("claude/opus:high", AgentClaude("opus", "high")),
        ("claude/opus[1m]", AgentClaude("opus[1m]", "")),
        ("claude/claude-sonnet-4-6[1m]:high", AgentClaude("claude-sonnet-4-6[1m]", "high")),
        ("claude/sonnet-medium", AgentClaude("sonnet-medium", "")),
        (
            "claude/claude-sonnet-4-5:experimental",
            AgentClaude("claude-sonnet-4-5", "experimental"),
        ),
        ("codex/o3", AgentCodex("o3", "")),
        ("codex/o3:high", AgentCodex("o3", "high")),
        ("codex/gpt-5.1-codex:xhigh", AgentCodex("gpt-5.1-codex", "xhigh")),
        ("pi/openai/gpt-5", AgentPi("openai", "gpt-5", "")),
        ("pi/openai/gpt-5:low", AgentPi("openai", "gpt-5", "low")),
        ("pi/claude/sonnet:high", AgentPi("claude", "sonnet", "high")),
        ("anthropic/claude-opus", AgentPi("anthropic", "claude-opus", "")),
        ("anthropic/claude-opus:custom", AgentPi("anthropic", "claude-opus", "custom")),
        ("anthropic/x[1m]", AgentPi("anthropic", "x[1m]", "")),
        ("ollama/llama3:8b:high", AgentPi("ollama", "llama3:8b", "high")),
        ("bin/agent", AgentPi("bin", "agent", "")),
        ("my.host/org@v1+x:a_b-c", AgentPi("my.host", "org@v1+x", "a_b-c")),
        ("Claude/opus", AgentClaude("opus", "")),
        ("CODEX/o3:high", AgentCodex("o3", "high")),
        ("Pi/openai/gpt-5", AgentPi("openai", "gpt-5", "")),
        ("claude", AgentClaude("", "")),
        ("CLAUDE", AgentClaude("", "")),
        ("codex", AgentCodex("", "")),
        ("pi", AgentPi("", "", "")),
        ("claude:high", AgentClaude("", "high")),
        ("Codex:low", AgentCodex("", "low")),
        ("pi:off", AgentPi("", "", "off")),
        ("pi/anthropic", AgentPi("anthropic", "", "")),
        ("pi/anthropic:high", AgentPi("anthropic", "", "high")),
        ("PI/openai", AgentPi("openai", "", "")),
    ],
)
def test_agent_shorthand_selects_a_native_agent(text: str, expected: object) -> None:
    assert parse_agent_shorthand(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "claude -p",
        "claude --model x",
        "claude\t-p",
        "claudex",
        "pix:high",
        "p\u0131",
        "p\u0130:high",
        "p\u0131/openai/gpt",
        "codex-cli --flag",
        "worker --flag",
        "./bin/agent",
        "a/b/c",
        "/model",
        "provider/",
        "provider/model:",
        "provider/:high",
        "scripts/agent --flag",
        "my provider/model",
        "provider/model:high now",
        '"claude/opus"',
        '["a/b"]',
        '{"$case":"AgentPi","model":"a/b"',
        'AgentClaud(model="x/y")',
        "C:/agent",
        "http:/x",
        "provider/model:hi:gh!",
        "provider/mo$del",
        " provider/model",
        "prov[1]/model",
        "provider/model:hi[gh]",
    ],
)
def test_text_outside_the_shorthand_forms_is_not_shorthand(text: str) -> None:
    assert parse_agent_shorthand(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "claude/",
        "claude/a/b",
        "claude/opus:",
        "claude/:high",
        "claude/opus --x",
        "claude/opus:hi gh",
        "codex/",
        "codex/model/high",
        "codex/o3:",
        "pi/a/b/c/d",
        "pi//gpt",
        "pi/openai/",
        "pi/openai/:high",
        "pi/open ai/gpt",
        " claude/opus",
        "claude/opus ",
        "\tpi/openai/gpt-5",
        "claude/op$us",
        "pi/open:ai/gpt",
        "codex/o3:hi:gh!",
        "claude/opus:hi[gh]",
        "pi/open[ai]/gpt",
        "claude:",
        "Codex:",
        "pi:",
        "pi/",
        "pi/anthropic/",
        "pi/anthropic:",
        "pi/anthropic:hi:gh",
        "claude:hi:gh",
        "claude:hi gh",
        "claude:/opus",
        " claude",
        "\tcodex:high",
        " pi/anthropic",
        "claude ",
        "claude\n",
        "codex\t",
        "pi ",
        "Claude  \n",
        "claude:high ",
        "pi/anthropic\t",
    ],
)
def test_malformed_native_prefix_is_a_shorthand_error(text: str) -> None:
    with pytest.raises(AgentShorthandError):
        parse_agent_shorthand(text)


def test_spec_shape_tags_each_variant_with_its_own_fields() -> None:
    assert agent_spec_shape(AgentCommand("run me")) == {
        "$case": "AgentCommand",
        "command": "run me",
    }
    assert agent_spec_shape(AgentClaude("sonnet", "high")) == {
        "$case": "AgentClaude",
        "model": "sonnet",
        "thinking": "high",
    }
    assert agent_spec_shape(AgentCodex("o3", "high")) == {
        "$case": "AgentCodex",
        "model": "o3",
        "thinking": "high",
    }
    assert agent_spec_shape(AgentPi("anthropic", "claude-opus", "custom")) == {
        "$case": "AgentPi",
        "provider": "anthropic",
        "model": "claude-opus",
        "thinking": "custom",
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("claude/opus", AgentClaude("opus", "")),
        ("claude/opus:high", AgentClaude("opus", "high")),
        ("codex/o3:high", AgentCodex("o3", "high")),
        ("anthropic/claude-opus:custom", AgentPi("anthropic", "claude-opus", "custom")),
        ("claude", AgentClaude("", "")),
        ("Codex", AgentCodex("", "")),
        ("pi", AgentPi("", "", "")),
        ("claude:high", AgentClaude("", "high")),
        ("pi:low", AgentPi("", "", "low")),
        ("pi/anthropic", AgentPi("anthropic", "", "")),
        ("pi/anthropic:high", AgentPi("anthropic", "", "high")),
    ],
)
def test_shorthand_decodes_through_host_text_to_json(text: str, expected: AgentSpec) -> None:
    assert _decode(text) == agent_spec_shape(expected)


@pytest.mark.parametrize("fallback", [True, False])
@pytest.mark.parametrize(
    "text",
    [
        "claude/",
        "claude/opus:",
        "codex/a/b",
        "pi/anthropic/",
        " claude/opus",
        "Claude/",
        "claude:",
        "pi/",
        " codex",
        "claude ",
        "codex\n",
    ],
)
def test_malformed_native_prefix_is_a_decode_error_not_a_command(text: str, fallback: bool) -> None:
    with pytest.raises(ValueDecodeError):
        _decode(text, fallback=fallback)


@pytest.mark.parametrize(
    "command",
    [
        "claude -p",
        "claude --model x",
        "claudex",
        "p\u0131",
        "p\u0130:high",
        "worker --flag",
        "./bin/agent",
        "a/b/c",
        "C:/agent",
        "http:/x",
        '"claude/opus"',
        '["a/b"]',
        '{"$case":"AgentPi","model":"a/b"',
        'AgentClaud(model="x/y")',
    ],
)
def test_non_shorthand_text_stays_an_agent_command(command: str) -> None:
    """Regression: slash-bearing JSON, typo'd calls, and drive/scheme spellings
    were once read as Pi shorthand."""
    result = _decode(command)

    assert result == {"$case": "AgentCommand", "command": command}


@pytest.mark.parametrize(
    "text",
    ['"claude/opus"', '["a/b"]', '{"$case":"AgentPi","model":"a/b"', 'AgentClaud(model="x/y")'],
)
def test_non_shorthand_text_is_an_error_without_the_command_fallback(text: str) -> None:
    with pytest.raises(ValueDecodeError):
        _decode(text, fallback=False)


@pytest.mark.parametrize(
    "text",
    [
        'AgentCommand(command="bin/agent")',
        '{"$case":"AgentCommand","command":"bin/agent"}',
    ],
)
def test_slash_inside_a_call_or_json_object_is_not_shorthand(text: str) -> None:
    result = _decode(text)

    assert result == {"$case": "AgentCommand", "command": "bin/agent"}


def test_tagged_json_object_is_recognized_verbatim() -> None:
    text = '{"$case": "AgentClaude", "model": "sonnet", "thinking": "high"}'

    result = _decode(text)

    assert result == {"$case": "AgentClaude", "model": "sonnet", "thinking": "high"}


def test_call_form_constructor_selects_a_member() -> None:
    result = _decode('AgentClaude(model = "sonnet", thinking = "high")')

    assert result == agent_spec_shape(AgentClaude("sonnet", "high"))


def test_unrecognized_text_is_an_agent_command_verbatim() -> None:
    command = "custom-agent --model 'some model' \\%{SESSION_ID}"

    result = _decode(command)

    assert result == {"$case": "AgentCommand", "command": command}


def test_unrecognized_text_is_an_error_without_the_command_fallback() -> None:
    with pytest.raises(ValueError):
        _decode("not an agent", fallback=False)


def test_qualified_member_call_selects_a_member() -> None:
    result = _decode('Agent::AgentClaude(model = "sonnet", thinking = "high")')

    assert result == agent_spec_shape(AgentClaude("sonnet", "high"))


def test_unclosed_member_call_is_an_error_not_a_command() -> None:
    """Text opening a member call commits to it: a read failure inside the
    call (an unclosed parenthesis, here) is reported rather than silently
    falling back to a verbatim command, even with the fallback allowed."""
    with pytest.raises(ValueError):
        _decode('AgentClaude(model = "x"')


def test_misqualified_member_call_is_an_error_not_a_command() -> None:
    """A qualifier other than ``Agent`` on a member call is an error, even
    though its member name and call shape are otherwise well formed."""
    with pytest.raises(ValueError):
        _decode('Foo::AgentClaude("a", "b")')


@pytest.mark.parametrize("source", ["Agent :: AgentClaude", "Agent:: AgentClaude"])
def test_whitespace_around_double_colon_is_insignificant_in_a_qualified_call(
    source: str,
) -> None:
    result = _decode(f'{source}(model = "sonnet", thinking = "high")')

    assert result == agent_spec_shape(AgentClaude("sonnet", "high"))


def test_leading_whitespace_before_a_call_is_insignificant() -> None:
    result = _decode('  \n AgentClaude(model = "sonnet", thinking = "high")')

    assert result == agent_spec_shape(AgentClaude("sonnet", "high"))


def test_text_opening_with_a_parenthesis_is_an_agent_command_verbatim() -> None:
    command = "(not a call)"

    result = _decode(command)

    assert result == {"$case": "AgentCommand", "command": command}


def test_malformed_qualifier_chain_is_an_agent_command_verbatim() -> None:
    """``Agent::(x)`` never lexically opens a member call (no name after
    ``::``), so it does not commit -- it falls back to a verbatim command."""
    command = "Agent::(x)"

    result = _decode(command)

    assert result == {"$case": "AgentCommand", "command": command}


def test_unqualified_non_member_call_is_an_agent_command_verbatim() -> None:
    command = "notamember(x)"

    result = _decode(command)

    assert result == {"$case": "AgentCommand", "command": command}


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_text_is_an_error(text: str) -> None:
    """Whitespace-only text is always an error, even with the command fallback."""
    with pytest.raises(ValueError):
        _decode(text)
