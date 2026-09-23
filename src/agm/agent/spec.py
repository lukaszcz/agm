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
from enum import StrEnum
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
    "PermissionMode",
    "SessionTransport",
]


class SessionTransport(StrEnum):
    """The ways a session backend can drive an agent.

    The member values are also the standard ``SessionTransport`` member names,
    so a specification's default crosses into AgL without translation.
    """

    CLI = "Cli"
    RPC = "Rpc"


class PermissionMode(StrEnum):
    """How much an agent CLI is allowed to do without asking, inside its run.

    ``NATIVE`` selects the agent's own "don't ask me" mode; ``UNRESTRICTED``
    selects its most permissive mode, meant for use inside a sandbox;
    ``NONE`` appends no flag, leaving the agent's default prompting behavior
    in place. Every spec's ``argv``/``session_argv``/``rpc_argv`` takes this
    keyword-only, defaulting to ``NONE`` so unmigrated callers are unaffected.
    """

    NATIVE = "native"
    UNRESTRICTED = "unrestricted"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class AgentCommand:
    """A command supplied directly by an AgL ``AgentCommand`` value."""

    command: str

    PAYLOAD_FIELDS: ClassVar[tuple[str, ...]] = ("command",)
    prompt_via_stdin: ClassVar[bool] = False
    DEFAULT_SESSION_TRANSPORT: ClassVar[SessionTransport] = SessionTransport.CLI

    def argv(self, *, permission_mode: PermissionMode = PermissionMode.NONE) -> list[str]:
        """Split the configured command, retaining its prompt-file semantics.

        *permission_mode* is ignored: an ``AgentCommand`` is verbatim in every
        mode, since its author controls its flags directly.

        Placeholder substitution or appending the prompt-file argument is
        performed later by :func:`agm.agent.runner.command_with_prompt_target`,
        exactly as for configured runner commands.
        """
        del permission_mode
        return parse_command(self.command, kind="agent")

    def payload_values(self) -> tuple[str, ...]:
        """This specification's ``PAYLOAD_FIELDS`` values, in that order."""
        return (self.command,)


@dataclass(frozen=True, slots=True)
class AgentClaude:
    """The model and thinking settings for a Claude prompt invocation."""

    model: str
    thinking: str

    PAYLOAD_FIELDS: ClassVar[tuple[str, ...]] = ("model", "thinking")
    prompt_via_stdin: ClassVar[bool] = False
    DEFAULT_SESSION_TRANSPORT: ClassVar[SessionTransport] = SessionTransport.CLI

    def argv(self, *, permission_mode: PermissionMode = PermissionMode.NONE) -> list[str]:
        """Build the argv for a one-shot Claude prompt invocation."""
        return [
            "claude",
            "-p",
            *_claude_options(self.model, self.thinking),
            *_CLAUDE_PERMISSION_FLAGS[permission_mode],
        ]

    def payload_values(self) -> tuple[str, ...]:
        """This specification's ``PAYLOAD_FIELDS`` values, in that order."""
        return (self.model, self.thinking)

    def session_argv(
        self,
        session_id: str,
        *,
        resume: bool = False,
        fork: bool = False,
        json_output: bool = False,
        name: str = "",
        permission_mode: PermissionMode = PermissionMode.NONE,
    ) -> list[str]:
        """Build a Claude prompt argv for an existing or newly named session."""
        command = ["claude", "-p", "--resume" if resume else "--session-id", session_id]
        if fork:
            command.append("--fork-session")
        if json_output:
            command.extend(("--output-format", "json"))
        if name:
            command.extend(("-n", name))
        return [
            *command,
            *_claude_options(self.model, self.thinking),
            *_CLAUDE_PERMISSION_FLAGS[permission_mode],
        ]


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
    DEFAULT_SESSION_TRANSPORT: ClassVar[SessionTransport] = SessionTransport.CLI

    def argv(self, *, permission_mode: PermissionMode = PermissionMode.NONE) -> list[str]:
        """Build the argv for a one-shot Codex prompt invocation, reading stdin."""
        return self._exec_argv(permission_mode=permission_mode)

    def payload_values(self) -> tuple[str, ...]:
        """This specification's ``PAYLOAD_FIELDS`` values, in that order."""
        return (self.model, self.thinking)

    def session_argv(
        self,
        session_id: str | None = None,
        *,
        permission_mode: PermissionMode = PermissionMode.NONE,
    ) -> list[str]:
        """Build the argv that starts or resumes a Codex CLI session."""
        command = ["codex", "exec"]
        if session_id is None:
            command.append("--json")
        else:
            command.extend(("resume", session_id))
        return self._exec_argv(
            command, permission_mode=permission_mode, resuming=session_id is not None
        )

    def _exec_argv(
        self,
        command: list[str] | None = None,
        *,
        permission_mode: PermissionMode = PermissionMode.NONE,
        resuming: bool = False,
    ) -> list[str]:
        """Add this specification's model settings and stdin marker to *command*.

        The permission flag goes before the trailing ``-`` stdin marker, which
        must stay last: codex reads the prompt from the first thing after it.
        *resuming* selects the flag table, because the two invocation forms
        accept different approval flags.
        """
        result = command or ["codex", "exec"]
        result.extend(_codex_options(self.model, self.thinking))
        table = _CODEX_RESUME_PERMISSION_FLAGS if resuming else _CODEX_PERMISSION_FLAGS
        result.extend(table[permission_mode])
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
    DEFAULT_SESSION_TRANSPORT: ClassVar[SessionTransport] = SessionTransport.RPC

    def argv(self, *, permission_mode: PermissionMode = PermissionMode.NONE) -> list[str]:
        """Build the argv for a one-shot Pi prompt invocation.

        *permission_mode* is accepted for a uniform spec surface but never
        adds a flag: Pi has no permission-mode CLI option in any mode.
        """
        del permission_mode
        return ["pi", "-p", *_pi_options(self.provider, self.model, self.thinking)]

    def payload_values(self) -> tuple[str, ...]:
        """This specification's ``PAYLOAD_FIELDS`` values, in that order."""
        return (self.provider, self.model, self.thinking)

    def session_argv(
        self,
        session_id: str,
        *,
        fork_from: str | None = None,
        name: str = "",
        permission_mode: PermissionMode = PermissionMode.NONE,
    ) -> list[str]:
        """Build the argv for a Pi prompt in a named or forked CLI session."""
        del permission_mode
        command = ["pi", "-p", "--session-id", session_id]
        if fork_from is not None:
            command.extend(("--fork", fork_from))
        if name:
            command.extend(("--name", name))
        return [*command, *_pi_options(self.provider, self.model, self.thinking)]

    def rpc_argv(
        self,
        *,
        name: str = "",
        session_id: str = "",
        permission_mode: PermissionMode = PermissionMode.NONE,
    ) -> list[str]:
        """Build the argv for a persistent Pi RPC session."""
        del permission_mode
        command = ["pi", "--mode", "rpc"]
        if session_id:
            command.extend(("--session-id", session_id))
        if name:
            command.extend(("--name", name))
        return [*command, *_pi_options(self.provider, self.model, self.thinking)]


AgentSpec: TypeAlias = AgentCommand | AgentClaude | AgentCodex | AgentPi

#: Runtime projection of the checked standard ``Agent`` variants.
AGENT_SPECS: Mapping[str, type[AgentSpec]] = MappingProxyType(
    {
        "AgentCommand": AgentCommand,
        "AgentClaude": AgentClaude,
        "AgentCodex": AgentCodex,
        "AgentPi": AgentPi,
    }
)


#: Claude's permission flag per mode, appended after its model options.
_CLAUDE_PERMISSION_FLAGS: Mapping[PermissionMode, tuple[str, ...]] = MappingProxyType(
    {
        PermissionMode.NATIVE: ("--permission-mode", "auto"),
        PermissionMode.UNRESTRICTED: ("--dangerously-skip-permissions",),
        PermissionMode.NONE: (),
    }
)

#: Codex's permission flag per mode, appended after its model options and
#: before the trailing ``-`` stdin marker.
_CODEX_PERMISSION_FLAGS: Mapping[PermissionMode, tuple[str, ...]] = MappingProxyType(
    {
        PermissionMode.NATIVE: ("--approve-for-me",),
        PermissionMode.UNRESTRICTED: ("--dangerously-bypass-approvals-and-sandbox",),
        PermissionMode.NONE: (),
    }
)

#: Codex's permission flag per mode on the ``resume`` form. A resumed thread
#: continues under the approval policy, sandbox policy and permission profile
#: recorded when its session was created, so the native mode re-asserts
#: nothing -- and ``codex exec resume`` accepts no approval flag at all. The
#: bypass flag stays available as an explicit escalation, which it does accept.
_CODEX_RESUME_PERMISSION_FLAGS: Mapping[PermissionMode, tuple[str, ...]] = MappingProxyType(
    {
        PermissionMode.NATIVE: (),
        PermissionMode.UNRESTRICTED: ("--dangerously-bypass-approvals-and-sandbox",),
        PermissionMode.NONE: (),
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
