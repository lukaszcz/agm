"""Value-driven AgL ``Agent`` transport dispatch."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from agm.agent.stream import ClaudeOutputStream, decode_claude_stream_json
from agm.agent.transport import AgentCallInfo
from agm.agl.ir.builtin_nominals import BuiltinNominals, standard_member_name
from agm.agl.runtime.request import AgentCallHostError, AgentRequest, AgentResponse
from agm.agl.semantics.values import RecordValue, TextValue, Value
from agm.sandbox.request import PreparedSandboxCommand

if TYPE_CHECKING:
    from agm.agent.runner import PromptDelivery
    from agm.agent.spec import AgentSpec
    from agm.sandbox.prepare import SandboxContext, SandboxRun

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
    sandbox: "SandboxRun | None" = None,
    stdout_callback: Callable[[str], None] | None = None,
    stdout_finalizer: Callable[[], None] | None = None,
    decode_stdout: Callable[[str], str] | None = None,
    interactive: bool = False,
) -> AgentResponse:
    """Send the already-composed request prompt through the shared runner seam."""
    from agm.agent.runner import (
        cleanup_temp_files,
        prepare_rendered_prompt_run,
        prompt_run_result_error,
        result_stderr_tail,
        run_prepared_prompt_result,
    )
    from agm.agent.spec import AgentCodex
    from agm.util.interp import InterpolationError

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            request.prompt,
            runner=command,
            temp_files=temp_files,
            env=request.env,
            delivery=delivery,
            sandbox=sandbox,
            pty=interactive,
        )
        output_callback = request.output_callback
        result = run_prepared_prompt_result(
            prepared,
            idle_timeout=idle_timeout,
            stdout_callback=stdout_callback,
            stdout_to_file=isinstance(request.agent, AgentCodex),
            stderr_callback=(
                None if output_callback is None else lambda text: output_callback("stderr", text)
            ),
            interactive=interactive,
        )
        if stdout_finalizer is not None:
            stdout_finalizer()
        call_info = AgentCallInfo(
            argv=prepared.argv or [],
            prompt_via_stdin=prepared.prompt_via_stdin,
            elapsed=result.elapsed,
            exit_code=result.returncode,
            # A prepared but never-started sandbox (preparation failed) never
            # actually ran the call under the sandbox, so this must reflect
            # what happened, not merely what was requested.
            sandboxed=isinstance(prepared.sandbox, PreparedSandboxCommand),
            permission_mode=request.permission_mode.value,
        )
    except InterpolationError as exc:
        raise AgentCallHostError(
            cause="interpolation_failure" if interactive else "spawn_failure",
            exit_code=None,
            stderr_tail=str(exc),
            elapsed=0.0,
        ) from exc
    finally:
        cleanup_temp_files(temp_files)
    failure = prompt_run_result_error(result)
    if failure is not None:
        raise AgentCallHostError(
            cause=failure.cause,
            exit_code=result.returncode,
            stderr_tail=result_stderr_tail(result),
            elapsed=result.elapsed,
            call_info=call_info,
            detail=failure.detail,
        )
    content = result.stdout.text()
    if decode_stdout is not None:
        try:
            content = decode_stdout(content)
        except ValueError as exc:
            raise AgentCallHostError(
                cause="protocol_failure",
                exit_code=result.returncode,
                stderr_tail=result_stderr_tail(result),
                elapsed=result.elapsed,
                call_info=call_info,
                detail=str(exc),
            ) from exc
    return AgentResponse(
        content=content,
        metadata={"elapsed": result.elapsed},
        call_info=call_info,
    )


def run_agent_chat(
    request: AgentRequest, *, get_sandbox_context: "Callable[[], SandboxContext] | None"
) -> AgentResponse:
    """Launch a fresh foreground conversation without a response contract or timeout."""
    from agm.agent.runner import PromptDelivery
    from agm.agent.spec import AgentCommand
    from agm.sandbox.prepare import sandbox_run_for

    try:
        command = request.agent.interactive_argv(permission_mode=request.permission_mode)
    except ValueError as error:
        raise AgentCallHostError(
            cause="invalid_agent", exit_code=None, stderr_tail=str(error), elapsed=0.0
        ) from error
    if request.sandbox is not None and get_sandbox_context is None:
        raise AgentCallHostError(
            cause="spawn_failure",
            exit_code=None,
            stderr_tail="The host does not provide sandbox preparation.",
            elapsed=0.0,
        )
    sandbox = (
        None
        if get_sandbox_context is None
        else sandbox_run_for(request.sandbox, get_sandbox_context)
    )
    return _run_request(
        request,
        command,
        None,
        delivery=PromptDelivery.FILE
        if isinstance(request.agent, AgentCommand)
        else PromptDelivery.LITERAL,
        sandbox=sandbox,
        interactive=True,
    )


def agent_member_name(value: RecordValue, nominals: BuiltinNominals) -> str:
    """Return the bare ``Agent`` member name (e.g. ``"AgentClaude"``) *value* projects onto.

    Dispatch is by nominal identity, resolved through *nominals* (an ``Agent``
    member's ``NominalId`` is program-specific), never by a name read off the
    value itself.
    """
    from agm.agent.spec import AGENT_SPECS

    return standard_member_name(value.nominal, "Agent", AGENT_SPECS, nominals)


def agent_spec_type(value: RecordValue, nominals: BuiltinNominals) -> type[AgentSpec]:
    """Resolve the host specification class an ``Agent`` member is projected onto.

    The single seam through which AgL asks what kind of agent a value is, so no
    caller has to recognize a variant by its name.
    """
    from agm.agent.spec import AGENT_SPECS

    return AGENT_SPECS[agent_member_name(value, nominals)]


def decode_agent_value(value: RecordValue, nominals: BuiltinNominals) -> AgentSpec:
    """Decode an ``Agent`` member record into its host-side specification."""
    from agm.agent.spec import payload_fields

    spec_cls = agent_spec_type(value, nominals)
    return spec_cls(*(_text_field(value, name) for name in payload_fields(spec_cls)))


def agent_value(spec: AgentSpec, nominals: BuiltinNominals) -> RecordValue:
    """Encode a host agent specification back into its ``Agent`` member value.

    The inverse of :func:`decode_agent_value`, used where a host hands a spec
    it already holds back to AgL (e.g. a default session's snapshot agent).
    """
    from agm.agent.spec import payload_items

    declared = nominals.resolve_standard_member("Agent", type(spec).__name__)
    fields: dict[str, Value] = {
        name: TextValue(value) for name, value in payload_items(spec).items()
    }
    return RecordValue(nominal=declared.nominal, fields=fields)


def _text_field(value: RecordValue, name: str) -> str:
    return cast(TextValue, value.fields[name]).value


def value_driven_agent_factory(
    *, idle_timeout: float | None, get_sandbox_context: "Callable[[], SandboxContext]"
) -> AgentFn:
    """Return a dispatcher which builds an invocation from ``request.agent``.

    *get_sandbox_context* is a lazily-caching `SandboxContext` builder (see
    `sandbox.prepare.lazy_sandbox_context`); the host passes in one shared
    callable so this factory, `create_agl_session_host`, and the execution
    services all resolve sandbox configuration from the same context, and an
    agent-free program, or one whose every call runs unsandboxed, never pays
    for that config I/O.
    """

    def dispatch(request: AgentRequest) -> AgentResponse:
        from agm.agent.spec import AgentClaude
        from agm.sandbox.prepare import sandbox_run_for

        spec = request.agent
        stream = (
            ClaudeOutputStream(request.output_callback)
            if request.output_callback is not None and isinstance(spec, AgentClaude)
            else None
        )
        try:
            command = (
                spec.argv(
                    permission_mode=request.permission_mode,
                    verbose=True,
                    stream_output=True,
                )
                if stream is not None and isinstance(spec, AgentClaude)
                else spec.argv(permission_mode=request.permission_mode)
            )
        except ValueError as exc:
            raise AgentCallHostError(
                cause="invalid_agent", exit_code=None, stderr_tail=str(exc), elapsed=0.0
            ) from exc
        sandbox = sandbox_run_for(request.sandbox, get_sandbox_context)
        return _run_request(
            request,
            command,
            idle_timeout,
            delivery=_spec_delivery(spec),
            sandbox=sandbox,
            stdout_callback=None if stream is None else stream.feed,
            stdout_finalizer=None if stream is None else stream.finish,
            decode_stdout=decode_claude_stream_json if stream is not None else None,
        )

    return dispatch


def _spec_delivery(spec: AgentSpec) -> "PromptDelivery":
    """Return how *spec* wants its rendered prompt delivered."""
    from agm.agent.runner import PromptDelivery

    return PromptDelivery.STDIN if spec.prompt_via_stdin else PromptDelivery.FILE
