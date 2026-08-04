"""Typed host-side specifications and command builders for AgL agents.

The builders return argv lists only.  Dispatch supplies that argv to the shared
prompt preparation and process-execution helpers in :mod:`agm.agent.runner`.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import TypeAlias

from agm.agl.semantics.values import EnumValue, TextValue

__all__ = [
    "AgentClaude",
    "AgentCodex",
    "AgentCommand",
    "AgentPi",
    "AgentSpec",
    "build_claude",
    "build_codex",
    "build_command",
    "build_pi",
    "decode",
]


@dataclass(frozen=True, slots=True)
class AgentCommand:
    """A command supplied directly by an AgL ``AgentCommand`` value."""

    command: str


@dataclass(frozen=True, slots=True)
class AgentClaude:
    """The model and thinking settings for a Claude prompt invocation."""

    model: str
    thinking: str


@dataclass(frozen=True, slots=True)
class AgentCodex:
    """The model and thinking settings for a Codex prompt invocation."""

    model: str
    thinking: str


@dataclass(frozen=True, slots=True)
class AgentPi:
    """The provider, model, and thinking settings for a Pi prompt invocation."""

    provider: str
    model: str
    thinking: str


AgentSpec: TypeAlias = AgentCommand | AgentClaude | AgentCodex | AgentPi


def decode(value: EnumValue) -> AgentSpec:
    """Decode a runtime ``Agent`` enum value into its host-side specification."""
    match value.variant:
        case "AgentCommand":
            return AgentCommand(_text_field(value, "command"))
        case "AgentClaude":
            return AgentClaude(_text_field(value, "model"), _text_field(value, "thinking"))
        case "AgentCodex":
            return AgentCodex(_text_field(value, "model"), _text_field(value, "thinking"))
        case "AgentPi":
            return AgentPi(
                _text_field(value, "provider"),
                _text_field(value, "model"),
                _text_field(value, "thinking"),
            )
        case variant:
            raise ValueError(f"unsupported Agent variant: {variant}")


def build_command(agent: AgentCommand) -> list[str]:
    """Build an ``AgentCommand`` argv, retaining its prompt-file semantics.

    Placeholder substitution or appending the prompt-file argument is performed
    later by :func:`agm.agent.runner.command_with_prompt_target`, exactly as for
    configured runner commands.
    """
    command = shlex.split(agent.command)
    if not command:
        raise ValueError("agent command is empty")
    return command


def build_claude(agent: AgentClaude) -> list[str]:
    """Build the argv for a one-shot Claude prompt invocation."""
    command = ["claude", "-p"]
    if agent.model:
        command.extend(("--model", agent.model))
    if agent.thinking:
        command.extend(("--effort", agent.thinking))
    return command


def build_codex(agent: AgentCodex) -> list[str]:
    """Build the argv for a one-shot Codex prompt invocation."""
    command = ["codex", "exec"]
    if agent.model:
        command.extend(("--model", agent.model))
    if agent.thinking:
        command.extend(("-c", f"model_reasoning_effort={agent.thinking}"))
    return command


def build_pi(agent: AgentPi) -> list[str]:
    """Build the argv for a one-shot Pi prompt invocation."""
    command = ["pi", "-p"]
    if agent.provider:
        command.extend(("--provider", agent.provider))
    if agent.model:
        command.extend(("--model", agent.model))
    if agent.thinking:
        command.extend(("--thinking", agent.thinking))
    return command


def _text_field(value: EnumValue, name: str) -> str:
    """Read a text payload field from a typechecked runtime enum value."""
    field = value.fields[name]
    if not isinstance(field, TextValue):
        raise ValueError(f"Agent field {name!r} must be text")
    return field.value
