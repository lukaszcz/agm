"""Value-driven AgL ``Agent`` transport dispatch."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from agm.agent.transport import AgentCallInfo, stderr_tail
from agm.agl.runtime.request import AgentCallHostError, AgentRequest, AgentResponse
from agm.agl.semantics.types import terminal_name
from agm.agl.semantics.values import RecordValue, TextValue
from agm.core.env import clone_env

if TYPE_CHECKING:
    from agm.agent.runner import PromptDelivery
    from agm.agent.spec import AgentSpec

AgentFn = Callable[[AgentRequest], AgentResponse | str]


def dispatch_agent_value(request: AgentRequest, dispatcher: AgentFn) -> AgentResponse:
    """Dispatch one request, preserving host failures for the effects seam."""
    raw = dispatcher(request)
    return AgentResponse(content=raw) if isinstance(raw, str) else raw


def _run_request(
    request: AgentRequest,
    command: list[str],
    idle_timeout: float | None,
    *,
    delivery: "PromptDelivery",
) -> AgentResponse:
    """Send the already-composed request prompt through the shared runner seam."""
    from agm.agent.runner import (
        cleanup_temp_files,
        prepare_rendered_prompt_run,
        prompt_run_result_error,
        run_prepared_prompt_result,
    )
    from agm.util.interp import InterpolationError

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            request.prompt,
            runner=command,
            temp_files=temp_files,
            env=clone_env(),
            delivery=delivery,
        )
        result = run_prepared_prompt_result(prepared, idle_timeout=idle_timeout)
        call_info = AgentCallInfo(
            argv=prepared.argv or [],
            prompt_via_stdin=prepared.prompt_via_stdin,
            elapsed=result.elapsed,
            exit_code=result.returncode,
        )
    except InterpolationError as exc:
        raise AgentCallHostError(
            cause="spawn_failure", exit_code=None, stderr_tail=str(exc), elapsed=0.0
        ) from exc
    finally:
        cleanup_temp_files(temp_files)
    failure = prompt_run_result_error(result)
    if failure is not None:
        raise AgentCallHostError(
            cause=failure.cause,
            exit_code=result.returncode,
            stderr_tail=stderr_tail(result.stderr),
            elapsed=result.elapsed,
            call_info=call_info,
        )
    return AgentResponse(
        content=result.stdout,
        metadata={"elapsed": result.elapsed},
        call_info=call_info,
    )


def agent_spec_type(value: RecordValue) -> "type[AgentSpec]":
    """Resolve the host specification class an ``Agent`` member is projected onto.

    The single seam through which AgL asks what kind of agent a value is, so no
    caller has to recognize a variant by its name.
    """
    from agm.agent.spec import AGENT_SPECS

    member_name = terminal_name(value.display_name)
    spec_cls = AGENT_SPECS.get(member_name)
    if spec_cls is None:
        raise ValueError(f"unsupported Agent member: {member_name}")
    return spec_cls


def decode_agent_value(value: RecordValue) -> "AgentSpec":
    """Decode an ``Agent`` member record into its host-side specification."""
    spec_cls = agent_spec_type(value)
    return spec_cls(*(_text_field(value, name) for name in spec_cls.PAYLOAD_FIELDS))


def _text_field(value: RecordValue, name: str) -> str:
    field = value.fields[name]
    if not isinstance(field, TextValue):
        raise ValueError(f"Agent field {name!r} must be text")
    return field.value


def value_driven_agent_factory(*, idle_timeout: float | None) -> AgentFn:
    """Return a dispatcher which builds an invocation from ``request.agent``."""

    def dispatch(request: AgentRequest) -> AgentResponse:
        try:
            spec = decode_agent_value(request.agent)
            command = spec.argv()
        except ValueError as exc:
            raise AgentCallHostError(
                cause="invalid_agent", exit_code=None, stderr_tail=str(exc), elapsed=0.0
            ) from exc
        return _run_request(request, command, idle_timeout, delivery=_spec_delivery(spec))

    return dispatch


def _spec_delivery(spec: "AgentSpec") -> "PromptDelivery":
    """Return how *spec* wants its rendered prompt delivered."""
    from agm.agent.runner import PromptDelivery

    return PromptDelivery.STDIN if spec.prompt_via_stdin else PromptDelivery.FILE
