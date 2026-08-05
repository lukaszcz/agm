"""Value-driven AgL ``Agent`` dispatch."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS, BuiltinNominals
from agm.agl.runtime.render import render_value
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.semantics.values import EnumValue, JsonValue, TextValue
from agm.core.env import clone_env

if TYPE_CHECKING:
    from agm.agent.spec import AgentSpec

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


def dispatch_agent_value(
    agent: EnumValue,
    request: AgentRequest,
    dispatcher: AgentFn | None,
    *,
    nominals: BuiltinNominals = NO_BUILTIN_DECLARATIONS,
) -> AgentResponse:
    """Execute an encoded ``Agent`` through the host dispatcher."""
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
    request: AgentRequest, command: list[str], idle_timeout: float | None
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


def decode_agent_value(value: EnumValue) -> "AgentSpec":
    """Decode a runtime ``Agent`` enum value into its host-side specification."""
    from agm.agent.spec import AgentClaude, AgentCodex, AgentCommand, AgentPi

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


def _text_field(value: EnumValue, name: str) -> str:
    """Read a text payload field from a typechecked runtime enum value."""
    field = value.fields[name]
    if not isinstance(field, TextValue):
        raise ValueError(f"Agent field {name!r} must be text")
    return field.value


def value_driven_agent_factory(*, idle_timeout: float | None) -> AgentFn:
    """Return a dispatcher which builds an invocation from ``request.agent``."""

    def dispatch(request: AgentRequest) -> AgentResponse:
        try:
            command = decode_agent_value(request.agent).argv()
        except ValueError as exc:
            raise AgentCallHostError(
                cause="invalid_agent",
                exit_code=None,
                stderr_tail=str(exc),
                elapsed=0.0,
            ) from exc
        return _run_request(request, command, idle_timeout)

    return dispatch


def _stderr_tail(stderr: str, *, max_chars: int = 500) -> str:
    return stderr[-max_chars:] if len(stderr) > max_chars else stderr
