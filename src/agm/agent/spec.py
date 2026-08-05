"""Typed host-side specifications and command builders for AgL agents.

A pure host data leaf: each specification knows only how to build its own argv.
Decoding a runtime ``Agent`` enum value into one of these specifications lives on
the AgL side, in :mod:`agm.agl.runtime.agents`.  Dispatch supplies the built argv
to the shared prompt preparation and process-execution helpers in
:mod:`agm.agent.runner`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from agm.agent.runner import parse_command

__all__ = [
    "AgentClaude",
    "AgentCodex",
    "AgentCommand",
    "AgentPi",
    "AgentSpec",
]


@dataclass(frozen=True, slots=True)
class AgentCommand:
    """A command supplied directly by an AgL ``AgentCommand`` value."""

    command: str

    def argv(self) -> list[str]:
        """Split the configured command, retaining its prompt-file semantics.

        Placeholder substitution or appending the prompt-file argument is
        performed later by :func:`agm.agent.runner.command_with_prompt_target`,
        exactly as for configured runner commands.
        """
        return parse_command(self.command, kind="agent")


@dataclass(frozen=True, slots=True)
class AgentClaude:
    """The model and thinking settings for a Claude prompt invocation."""

    model: str
    thinking: str

    def argv(self) -> list[str]:
        """Build the argv for a one-shot Claude prompt invocation."""
        return ["claude", "-p", *_flag("--model", self.model), *_flag("--effort", self.thinking)]


@dataclass(frozen=True, slots=True)
class AgentCodex:
    """The model and thinking settings for a Codex prompt invocation."""

    model: str
    thinking: str

    def argv(self) -> list[str]:
        """Build the argv for a one-shot Codex prompt invocation."""
        command = ["codex", "exec", *_flag("--model", self.model)]
        if self.thinking:
            command.extend(("-c", f"model_reasoning_effort={self.thinking}"))
        return command


@dataclass(frozen=True, slots=True)
class AgentPi:
    """The provider, model, and thinking settings for a Pi prompt invocation."""

    provider: str
    model: str
    thinking: str

    def argv(self) -> list[str]:
        """Build the argv for a one-shot Pi prompt invocation."""
        return [
            "pi",
            "-p",
            *_flag("--provider", self.provider),
            *_flag("--model", self.model),
            *_flag("--thinking", self.thinking),
        ]


AgentSpec: TypeAlias = AgentCommand | AgentClaude | AgentCodex | AgentPi


def _flag(flag: str, value: str) -> tuple[str, ...]:
    """Return the flag/value pair, or nothing when the setting is unset."""
    return (flag, value) if value else ()
