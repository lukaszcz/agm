"""Value-driven AgL ``Agent`` dispatch.

The legacy registry remains as a compatibility container for the old language
surface, but ``ask`` dispatches only encoded ``Agent`` enum values through its
value dispatcher.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NoReturn

from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS, BuiltinNominals
from agm.agl.ir.ids import AgentId
from agm.agl.runtime.render import render_value
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.semantics.values import EnumValue, JsonValue, TextValue
from agm.core.env import clone_env

AgentFn = Callable[[AgentRequest], AgentResponse | str]


class AgentCallHostError(Exception):
    """Python-level transport failure mapped to catchable ``AgentCallError``."""

    def __init__(
        self, *, cause: str, exit_code: int | None, stderr_tail: str, elapsed: float
    ) -> None:
        super().__init__(cause)
        self.cause = cause
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail
        self.elapsed = elapsed


class AgentRegistry:
    """Compatibility registry plus the dispatcher for typed agent values."""

    def __init__(
        self,
        *,
        named: Mapping[AgentId, AgentFn] | Mapping[str, AgentFn],
        default_agent: AgentFn | None,
        value_agent: AgentFn | None = None,
    ) -> None:
        self._named = {
            key if isinstance(key, AgentId) else AgentId(key): fn for key, fn in named.items()
        }
        self._default = default_agent
        self._value_agent = value_agent

    def set_default_agent(self, fn: AgentFn | None) -> None:
        self._default = fn

    @property
    def has_default_agent(self) -> bool:
        """Return whether a callable backs default ``ask`` dispatch."""
        return self._default is not None

    @property
    def agent_names(self) -> frozenset[str]:
        return frozenset(agent.declared_name for agent in self._named if not agent.scope_path)

    @property
    def agent_ids(self) -> frozenset[AgentId]:
        return frozenset(self._named)

    def backs(self, agent_id: AgentId) -> bool:
        return agent_id in self._named

    @property
    def value_dispatcher(self) -> AgentFn | None:
        """Return the typed-value dispatcher supplied by the host.

        Named registrations are retained only for compatibility with the old
        declaration surface. They never select a runner for an encoded
        ``Agent`` value: an ``AgentCommand`` always dispatches its own command
        through the value dispatcher or the default value-driven fallback.
        """
        return self._value_agent or self._default

    def dispatch(
        self,
        agent: EnumValue | AgentId | str,
        request: AgentRequest,
        *,
        nominals: BuiltinNominals = NO_BUILTIN_DECLARATIONS,
    ) -> AgentResponse:
        """Compatibility wrapper for legacy registry callers."""
        if isinstance(agent, (AgentId, str)):
            identity = agent if isinstance(agent, AgentId) else AgentId(agent)
            fn = self._named.get(identity) or self._default
            if fn is None:
                raise KeyError(f"No agent registered for {identity.display_name!r}")
            raw = fn(request)
            return AgentResponse(content=raw) if isinstance(raw, str) else raw
        return dispatch_agent_value(agent, request, self.value_dispatcher, nominals=nominals)


def dispatch_agent_value(
    agent: EnumValue,
    request: AgentRequest,
    dispatcher: AgentFn | None,
    *,
    nominals: BuiltinNominals = NO_BUILTIN_DECLARATIONS,
) -> AgentResponse:
    """Execute an encoded ``Agent`` without consulting named registrations."""
    agent_label = render_value(agent)
    if dispatcher is None:
        _raise_agent_call_error(
            agent,
            agent_label,
            AgentCallHostError(cause="no_dispatcher", exit_code=None, stderr_tail="", elapsed=0.0),
            nominals=nominals,
        )
    try:
        raw = dispatcher(request)
    except AgentCallHostError as error:
        _raise_agent_call_error(agent, agent_label, error, nominals=nominals)
    return AgentResponse(content=raw) if isinstance(raw, str) else raw


def _raise_agent_call_error(
    agent: EnumValue,
    agent_label: str,
    error: AgentCallHostError,
    *,
    nominals: BuiltinNominals,
) -> NoReturn:
    from agm.agl.runtime.trace import new_trace_id
    from agm.agl.semantics.exceptions import AglRaise
    from agm.agl.semantics.values import ExceptionValue

    nominal = nominals.nominal("AgentCallError")
    raise AglRaise(
        ExceptionValue(
            nominal=nominal,
            display_name=nominal.display_name,
            fields={
                "message": TextValue(f"Agent {agent_label!r} failed: {error.cause}"),
                "trace_id": TextValue(new_trace_id()),
                "agent": agent,
                "cause": TextValue(error.cause),
                "metadata": JsonValue(
                    {
                        "exit_code": error.exit_code,
                        "stderr_tail": error.stderr_tail,
                        "elapsed": error.elapsed,
                    }
                ),
            },
        )
    )


def _run_request(
    request: AgentRequest, command: str | list[str], idle_timeout: float | None
) -> AgentResponse:
    """Compose a request and run already-built argv through the shared runner seam."""
    from agm.agent.runner import (
        cleanup_temp_files,
        prepare_rendered_prompt_run,
        run_prepared_prompt_result,
    )

    parts = [request.prompt]
    if request.output_contract is not None and request.output_contract.format_instructions:
        parts.append(request.output_contract.format_instructions)
    if request.attempt:
        errors = "\n".join(f"- {error.message}" for error in request.validation_errors) or "(none)"
        parts.append(
            "Your previous response did not match the required output format.\n\n"
            f"Validation errors:\n{errors}\n\nPrevious response:\n"
            f"{request.previous_invalid_output or ''}\n\n"
            "Return only valid JSON matching the schema."
        )
    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            "\n\n".join(parts), runner=command, temp_files=temp_files, env=clone_env()
        )
        result = run_prepared_prompt_result(prepared, idle_timeout=idle_timeout)
    finally:
        cleanup_temp_files(temp_files)
    if result.spawn_error is not None:
        raise AgentCallHostError(
            cause="spawn_failure",
            exit_code=result.returncode,
            stderr_tail=_stderr_tail(result.stderr),
            elapsed=result.elapsed,
        )
    if result.timed_out:
        raise AgentCallHostError(
            cause="timeout",
            exit_code=result.returncode,
            stderr_tail=_stderr_tail(result.stderr),
            elapsed=result.elapsed,
        )
    if result.returncode not in (None, 0):
        raise AgentCallHostError(
            cause="nonzero_exit",
            exit_code=result.returncode,
            stderr_tail=_stderr_tail(result.stderr),
            elapsed=result.elapsed,
        )
    return AgentResponse(content=result.stdout, metadata={"elapsed": result.elapsed})


def value_driven_agent_factory(*, idle_timeout: float | None) -> AgentFn:
    """Return a dispatcher which builds an invocation from ``request.agent``."""
    from agm.agent.spec import (
        AgentClaude,
        AgentCodex,
        AgentCommand,
        AgentPi,
        build_claude,
        build_codex,
        build_command,
        build_pi,
        decode,
    )

    def dispatch(request: AgentRequest) -> AgentResponse:
        try:
            spec = decode(request.agent)
            match spec:
                case AgentCommand():
                    command = build_command(spec)
                case AgentClaude():
                    command = build_claude(spec)
                case AgentCodex():
                    command = build_codex(spec)
                case AgentPi():
                    command = build_pi(spec)
                case _:
                    raise ValueError("unsupported decoded Agent specification")
        except ValueError as exc:
            raise AgentCallHostError(
                cause="invalid_agent",
                exit_code=None,
                stderr_tail=str(exc),
                elapsed=0.0,
            ) from exc
        return _run_request(request, command, idle_timeout)

    return dispatch


def runner_backed_agent_factory(
    *,
    default_runner_cmd: str,
    per_agent_cmds: Mapping[AgentId, str] | Mapping[str, str],
    idle_timeout: float | None,
) -> AgentFn:
    """Build the value-driven dispatcher while retaining the legacy signature.

    ``default_runner_cmd`` and ``per_agent_cmds`` remain accepted during the
    deprecated declaration/config transition, but cannot override an encoded
    ``AgentCommand``. Agent enum values are decoded exclusively by their own
    builders.
    """
    del default_runner_cmd, per_agent_cmds
    return value_driven_agent_factory(idle_timeout=idle_timeout)


def _stderr_tail(stderr: str, *, max_chars: int = 500) -> str:
    return stderr[-max_chars:] if len(stderr) > max_chars else stderr
