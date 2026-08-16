"""Typed host-side specifications and command builders for AgL agents.

A pure host data leaf: each specification knows only how to build its own argv.
Decoding a runtime ``Agent`` enum value into one of these specifications lives on
the AgL side, in :mod:`agm.agl.runtime.agents`.  Dispatch supplies the built argv
to the shared prompt preparation and process-execution helpers in
:mod:`agm.agent.runner`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, TypeAlias

from agm.agent.runner import parse_command

__all__ = [
    "AgentClaude",
    "AgentCodex",
    "AgentCommand",
    "AgentPi",
    "AGENT_SPECS",
    "AgentSpec",
]


@dataclass(frozen=True, slots=True)
class AgentCommand:
    """A command supplied directly by an AgL ``AgentCommand`` value."""

    command: str

    PAYLOAD_FIELDS: ClassVar[tuple[str, ...]] = ("command",)
    prompt_via_stdin: ClassVar[bool] = False

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

    PAYLOAD_FIELDS: ClassVar[tuple[str, ...]] = ("model", "thinking")
    prompt_via_stdin: ClassVar[bool] = False

    def argv(self) -> list[str]:
        """Build the argv for a one-shot Claude prompt invocation."""
        return ["claude", "-p", *_claude_options(self.model, self.thinking)]

    def session_argv(
        self,
        session_id: str,
        *,
        resume: bool = False,
        fork: bool = False,
        json_output: bool = False,
        name: str = "",
    ) -> list[str]:
        """Build a Claude prompt argv for an existing or newly named session."""
        command = ["claude", "-p", "--resume" if resume else "--session-id", session_id]
        if fork:
            command.append("--fork-session")
        if json_output:
            command.extend(("--output-format", "json"))
        if name:
            command.extend(("-n", name))
        return [*command, *_claude_options(self.model, self.thinking)]


@dataclass(frozen=True, slots=True)
class AgentCodex:
    """The model and thinking settings for a Codex prompt invocation.

    ``codex exec`` treats a positional ``@<path>`` argument as literal prompt
    text rather than expanding it (that expansion is a TUI-only feature), so
    the prompt is delivered on stdin instead: the argv ends with ``-`` and
    ``prompt_via_stdin`` tells the runner to pipe the rendered prompt in
    rather than appending a prompt-file target.
    """

    model: str
    thinking: str

    PAYLOAD_FIELDS: ClassVar[tuple[str, ...]] = ("model", "thinking")
    prompt_via_stdin: ClassVar[bool] = True

    def argv(self) -> list[str]:
        """Build the argv for a one-shot Codex prompt invocation, reading stdin."""
        return self._exec_argv()

    def session_argv(self, session_id: str | None = None) -> list[str]:
        """Build the argv that starts or resumes a Codex CLI session."""
        command = ["codex", "exec"]
        if session_id is None:
            command.append("--json")
        else:
            command.extend(("resume", session_id))
        return self._exec_argv(command)

    def _exec_argv(self, command: list[str] | None = None) -> list[str]:
        """Add this specification's model settings and stdin marker to *command*."""
        result = command or ["codex", "exec"]
        result.extend(_codex_options(self.model, self.thinking))
        result.append("-")
        return result


@dataclass(frozen=True, slots=True)
class AgentPi:
    """The provider, model, and thinking settings for a Pi prompt invocation."""

    provider: str
    model: str
    thinking: str

    PAYLOAD_FIELDS: ClassVar[tuple[str, ...]] = ("provider", "model", "thinking")
    prompt_via_stdin: ClassVar[bool] = False

    def argv(self) -> list[str]:
        """Build the argv for a one-shot Pi prompt invocation."""
        return ["pi", "-p", *_pi_options(self.provider, self.model, self.thinking)]

    def session_argv(
        self, session_id: str, *, fork_from: str | None = None, name: str = ""
    ) -> list[str]:
        """Build the argv for a Pi prompt in a named or forked CLI session."""
        command = ["pi", "-p", "--session-id", session_id]
        if fork_from is not None:
            command.extend(("--fork", fork_from))
        if name:
            command.extend(("--name", name))
        return [*command, *_pi_options(self.provider, self.model, self.thinking)]

    def rpc_argv(self, *, name: str = "") -> list[str]:
        """Build the argv for a persistent Pi RPC session."""
        command = ["pi", "--mode", "rpc"]
        if name:
            command.extend(("--name", name))
        return [*command, *_pi_options(self.provider, self.model, self.thinking)]


AgentSpec: TypeAlias = AgentCommand | AgentClaude | AgentCodex | AgentPi

#: Runtime projection of the checked ``std/core::Agent`` variants.
AGENT_SPECS: Mapping[str, type[AgentSpec]] = MappingProxyType(
    {
        "AgentCommand": AgentCommand,
        "AgentClaude": AgentClaude,
        "AgentCodex": AgentCodex,
        "AgentPi": AgentPi,
    }
)


def _claude_options(model: str, thinking: str) -> list[str]:
    """Build the model settings shared by Claude one-shot and session runs."""
    return [*_flag("--model", model), *_flag("--effort", thinking)]


def _codex_options(model: str, thinking: str) -> list[str]:
    """Build the model settings shared by Codex one-shot and session runs."""
    options = [*_flag("--model", model)]
    if thinking:
        options.extend(("-c", f"model_reasoning_effort={thinking}"))
    return options


def _pi_options(provider: str, model: str, thinking: str) -> list[str]:
    """Build the model settings shared by Pi one-shot and session runs."""
    return [
        *_flag("--provider", provider),
        *_flag("--model", model),
        *_flag("--thinking", thinking),
    ]


def _flag(flag: str, value: str) -> tuple[str, ...]:
    """Return the flag/value pair, or nothing when the setting is unset."""
    return (flag, value) if value else ()
