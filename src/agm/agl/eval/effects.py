"""Host-effect handler collaborator for the AgL IR evaluator.

``EffectHandlers`` implements agent dispatch (``ask``, ``ask-request``) and
``exec`` (shell execution + output parsing/retry).  It is driven by
``IrInterpreter`` via the narrow ``EffectCtx`` Protocol and must NOT import
``ir_interpreter`` (no cycle).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import NoReturn, Protocol, cast

from agm.agl.ir.ids import ContractId, Location
from agm.agl.ir.nodes import (
    IrAsk,
    IrAskRequest,
    IrExec,
    IrExpr,
    IrSessionAsk,
    IrSessionDefault,
    IrSessionOp,
    IrSessionOpen,
)
from agm.agl.ir.program import ExecutableProgram, ExternFunctionBody
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.agents import AgentFn, dispatch_agent_value
from agm.agl.runtime.codec import ParseResult
from agm.agl.runtime.contract import OutputContract
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.runtime.option import none_value, some_value
from agm.agl.runtime.render import render_value
from agm.agl.runtime.request import (
    AgentCallHostError,
    AgentCancelled,
    AgentRequest,
    AgentResponse,
    compose_agent_prompt,
    compose_initial_agent_prompt,
    compose_session_corrective_follow_up,
)
from agm.agl.runtime.request import (
    ValidationError as ReqValidationError,
)
from agm.agl.runtime.sessions import (
    SessionAskError,
    SessionHost,
    SessionHostError,
    SessionTransport,
)
from agm.agl.runtime.trace import TraceStore
from agm.agl.semantics.cycles import AglCyclicValue, cyclic_value_raise
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.exceptions import make_builtin_exception as _make_exc_value
from agm.agl.semantics.values import (
    VOID_VALUE,
    BoolValue,
    DecimalValue,
    EnumValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
    Value,
)

# ---------------------------------------------------------------------------
# Narrow context Protocol
# ---------------------------------------------------------------------------


class EffectCtx(Protocol):
    """The minimal IrInterpreter surface the effect handlers need."""

    _program: ExecutableProgram
    _trace: TraceStore
    _agent_dispatcher: AgentFn | None
    _session_host: SessionHost | None
    _strict_json: bool
    _shell_exec_timeout: float | None
    _host_contracts: Mapping[ContractId, OutputContract]
    _extern_registry: ExternRegistry

    def _eval(self, expr: IrExpr) -> Value: ...

    def _parse_host_output(
        self, raw: str, contract_id: ContractId, *, effective_strict: bool
    ) -> ParseResult: ...


# ---------------------------------------------------------------------------
# Collaborator class
# ---------------------------------------------------------------------------


class EffectHandlers:
    """Effect-handler collaborator for ``IrInterpreter``.

    Instantiated once per ``IrInterpreter`` instance (``self._effects``).
    All agent-dispatch and exec dispatch in the IR op dispatcher is delegated
    here.
    """

    def __init__(self, ctx: EffectCtx) -> None:
        self._ctx = ctx

    def _text_of(self, value: Value) -> str:
        """Return the prompt/command text *value* stands for.

        A ``text`` value is used as-is; anything else is rendered. Rendering
        walks the value's containers, so a cyclic array or dict surfaces here
        as the walk's sentinel and becomes the catchable ``CyclicValueError``
        every other rendering site raises. Shared by ``ask``, ``ask``-request,
        and ``exec``, which all turn one operand into text this way.
        """
        if isinstance(value, TextValue):
            return value.value
        try:
            return render_value(value)
        except AglCyclicValue as exc:
            raise cyclic_value_raise(nominals=self._ctx._program.builtin_nominals) from exc

    # ------------------------------------------------------------------
    # Extern (Python FFI) call helper
    # ------------------------------------------------------------------

    def eval_extern_call(
        self, module_id: ModuleId, extern: ExternFunctionBody, args: Sequence[Value]
    ) -> Value:
        """Handle a call to an ``extern def``: resolve and invoke.

        The companion callable was already imported at program load
        (``pipeline._wire_extern_registry``); this only looks it up by name.
        ``ExternRegistry.invoke`` is the single chokepoint that turns every
        runtime failure crossing the boundary — a raising callable, an
        argument-conversion failure, or a return-contract violation — into
        ``AglRaise(ExternError)``, mirroring the ``exec`` model.
        """
        fn = self._ctx._extern_registry.resolve(module_id, extern.name)
        return self._ctx._extern_registry.invoke(
            extern.name, fn, args, nominals=self._ctx._program.builtin_nominals
        )

    # ------------------------------------------------------------------
    # Agent call helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_trace_value(agent: EnumValue) -> dict[str, object]:
        """Return the agent variant and payload without re-decoding it."""
        return {
            "variant": agent.variant,
            "payload": {
                name: value.value if isinstance(value, TextValue) else render_value(value)
                for name, value in agent.fields.items()
            },
        }

    def _dispatch_agent(
        self,
        request: AgentRequest,
        node: IrAsk,
        *,
        max_attempts: int,
        target_type: str,
        codec: str,
        strict_json: bool | None,
        json_schema: object | None,
    ) -> AgentResponse:
        """Trace, dispatch, and map every agent outcome at one boundary."""
        self._ctx._trace.agent_request(
            agent=self._agent_trace_value(request.agent),
            attempt=request.attempt,
            max_attempts=max_attempts,
            prompt=request.prompt,
            target_type=target_type,
            codec=codec,
            strict_json=strict_json,
            json_schema=json_schema,
            span=node.location,
        )
        try:
            if self._ctx._agent_dispatcher is None:
                raise AgentCallHostError(
                    cause="no_dispatcher", exit_code=None, stderr_tail="", elapsed=0.0
                )
            response = dispatch_agent_value(request, self._ctx._agent_dispatcher)
        except AgentCallHostError as exc:
            call_info = exc.call_info.to_trace() if exc.call_info is not None else None
            if call_info is None:
                call_info = {"exit_code": exc.exit_code, "elapsed": exc.elapsed}
            call_info["stderr_tail"] = exc.stderr_tail
            self._ctx._trace.agent_response(
                ok=False, cause=exc.cause, call_info=call_info, span=node.location
            )
            self._raise_agent_call_error(request.agent, exc)
        except AgentCancelled as exc:
            self._ctx._trace.agent_response(
                ok=False, cancelled=True, reason=exc.reason, span=node.location
            )
            exc.span = node.location
            raise
        except KeyboardInterrupt as exc:
            cancelled = AgentCancelled(
                render_value(request.agent), "interrupted", span=node.location
            )
            self._ctx._trace.agent_response(
                ok=False, cancelled=True, reason=cancelled.reason, span=node.location
            )
            raise cancelled from exc
        self._ctx._trace.agent_response(
            ok=True,
            content=response.content,
            metadata=response.metadata,
            call_info=response.call_info.to_trace() if response.call_info is not None else None,
            span=node.location,
        )
        return response

    def _raise_agent_call_error(self, agent: EnumValue, error: AgentCallHostError) -> NoReturn:
        """Convert a transport failure after it was recorded in the trace."""
        declared = self._ctx._program.builtin_nominals.resolve("AgentCallError")
        agent_label = render_value(agent)
        raise AglRaise(
            ExceptionValue(
                nominal=declared.nominal,
                display_name=declared.display_name,
                fields={
                    "message": TextValue(
                        f"Agent {agent_label!r} failed: {error.cause}"
                        + (f" (exit {error.exit_code})" if error.exit_code is not None else "")
                    ),
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

    @staticmethod
    def _classify_parse_errors(result: ParseResult) -> tuple[ReqValidationError, ...]:
        """Reduce a failed parse ``result`` to validation errors for the next retry."""
        if result.errors:
            return result.errors
        if result.error_msg:
            return (
                ReqValidationError(
                    category="invalid_json",
                    message=result.error_msg,
                    path="$",
                    field=None,
                ),
            )
        return ()

    def _raise_agent_parse_error(
        self,
        *,
        message: str,
        agent: Value,
        last_raw: str | None,
        last_normalized: str | None,
        last_errors: tuple[ReqValidationError, ...],
        max_attempts: int,
        target_type_label: str,
        json_schema: str | None,
    ) -> NoReturn:
        """Raise ``AgentParseError`` once all parse/retry attempts are exhausted."""
        errors_json: list[object] = [e.to_json_obj() for e in last_errors]
        normalized_text = last_normalized if last_normalized is not None else (last_raw or "")
        raise AglRaise(
            _make_exc_value(
                "AgentParseError",
                message,
                nominals=self._ctx._program.builtin_nominals,
                raw=TextValue(last_raw or ""),
                normalized_raw=TextValue(normalized_text),
                agent=agent,
                attempts=IntValue(max_attempts),
                target_type=TextValue(target_type_label),
                expected_schema=JsonValue(
                    None if json_schema is None else cast(object, json.loads(json_schema))
                ),
                validation_errors=JsonValue(errors_json),
                metadata=JsonValue(None),
            )
        )

    def eval_ir_ask(
        self,
        _node: IrAsk,
        agent_expr: IrExpr,
        prompt_expr: IrExpr,
        contract_id: ContractId,
        max_attempts: int,
    ) -> Value:
        """Handle IrAsk: dispatch an Agent enum value and parse output."""
        agent_val = self._ctx._eval(agent_expr)
        if not isinstance(agent_val, EnumValue):
            raise TypeError(
                f"IrAsk agent must evaluate to an Agent enum value, got {type(agent_val).__name__}"
            )
        agent_name = render_value(agent_val)

        prompt_text = self._text_of(self._ctx._eval(prompt_expr))

        contract = self._ctx._program.contracts[contract_id]

        # Unit-typed asks still produce a complete request/response trace pair.
        if contract.is_unit:
            request = AgentRequest(agent=agent_val, prompt=prompt_text, output_contract=None)
            request.prompt = compose_agent_prompt(request)
            self._dispatch_agent(
                request,
                _node,
                max_attempts=1,
                target_type=contract.target_type_label,
                codec=contract.codec_name,
                strict_json=contract.strict_json,
                json_schema=None,
            )
            return VOID_VALUE

        effective_strict = (
            contract.strict_json if contract.strict_json is not None else self._ctx._strict_json
        )

        from agm.agl.runtime.contract import TypelessOutputContract

        output_contract: OutputContract | TypelessOutputContract = self._ctx._host_contracts.get(
            contract_id
        ) or TypelessOutputContract(
            target_type=contract.target_type_label,
            codec_name=contract.codec_name,
            strict_json=contract.strict_json,
            format_instructions=contract.format_instructions,
            json_schema=(
                None
                if contract.json_schema is None
                else cast(object, json.loads(contract.json_schema))
            ),
            structured_exec=contract.structured_exec,
        )

        last_raw: str | None = None
        last_normalized: str | None = None
        last_errors: tuple[ReqValidationError, ...] = ()

        json_schema = (
            None if contract.json_schema is None else cast(object, json.loads(contract.json_schema))
        )
        for attempt in range(max_attempts):
            request = AgentRequest(
                agent=agent_val,
                prompt=prompt_text,
                attempt=attempt,
                previous_invalid_output=last_raw,
                validation_errors=list(last_errors),
                output_contract=output_contract,
            )
            request.prompt = compose_agent_prompt(request)
            response = self._dispatch_agent(
                request,
                _node,
                max_attempts=max_attempts,
                target_type=contract.target_type_label,
                codec=contract.codec_name,
                strict_json=contract.strict_json,
                json_schema=json_schema,
            )
            raw = response.content

            result = self._ctx._parse_host_output(
                raw, contract_id, effective_strict=effective_strict
            )
            self._ctx._trace.parse_result(
                ok=result.ok,
                raw=raw,
                normalized_raw=result.normalized_raw or raw,
                error_summary=result.error_msg
                or "; ".join(error.message for error in result.errors),
                span=_node.location,
            )

            if result.ok and result.value is not None:
                return result.value

            last_raw = raw
            last_normalized = result.normalized_raw
            last_errors = self._classify_parse_errors(result)

        self._raise_agent_parse_error(
            message=(
                f"Agent {agent_name!r} failed to produce a valid "
                f"{contract.target_type_label} after {max_attempts} attempt(s). "
                f"Last output: {last_raw!r}"
            ),
            agent=agent_val,
            last_raw=last_raw,
            last_normalized=last_normalized,
            last_errors=last_errors,
            max_attempts=max_attempts,
            target_type_label=contract.target_type_label,
            json_schema=contract.json_schema,
        )

    def _session_error(self, error: SessionHostError) -> NoReturn:
        """Map a host lifecycle failure to the catchable SessionError shape."""
        raise AglRaise(
            _make_exc_value(
                "SessionError",
                error.message,
                nominals=self._ctx._program.builtin_nominals,
                operation=TextValue(error.operation),
            )
        ) from error

    def _require_session_host(self, operation: str) -> SessionHost:
        host = self._ctx._session_host
        if host is None:
            self._session_error(SessionHostError("session host is unavailable", operation))
        return host

    def _session_value(self, handle: str, agent: EnumValue, transport: str) -> RecordValue:
        declared = self._ctx._program.builtin_nominals.resolve("Session")
        transport_declared = self._ctx._program.builtin_nominals.resolve("SessionTransport")
        return RecordValue(
            nominal=declared.nominal,
            display_name=declared.display_name,
            fields={
                "id": TextValue(handle),
                "agent": agent,
                "transport": EnumValue(
                    nominal=transport_declared.nominal,
                    display_name=transport_declared.display_name,
                    variant=transport,
                    fields={},
                ),
            },
        )

    def _session_parts(self, value: Value, _operation: str) -> tuple[str, EnumValue, str]:
        """Extract the statically guaranteed fields from a ``Session`` record."""
        session = cast(RecordValue, value)
        return (
            cast(TextValue, session.fields["id"]).value,
            cast(EnumValue, session.fields["agent"]),
            cast(EnumValue, session.fields["transport"]).variant,
        )

    def _resolve_session_transport(self, agent: EnumValue, transport: Value | None) -> str:
        if transport is None:
            return SessionTransport.RPC if agent.variant == "AgentPi" else SessionTransport.CLI
        selected_transport = cast(EnumValue, transport)
        if selected_transport.variant == "None":
            return SessionTransport.RPC if agent.variant == "AgentPi" else SessionTransport.CLI
        return cast(EnumValue, selected_transport.fields["value"]).variant

    def eval_ir_session_open(self, node: IrSessionOpen) -> Value:
        """Open a host-backed session and mint its opaque AgL record."""
        agent = cast(EnumValue, self._ctx._eval(node.agent))
        transport = self._resolve_session_transport(
            agent, None if node.transport is None else self._ctx._eval(node.transport)
        )
        name = self._text_of(self._ctx._eval(node.name))
        try:
            handle = self._require_session_host("open").open(agent, transport, name=name)
        except SessionHostError as error:
            self._session_error(error)
        return self._session_value(handle, agent, transport)

    def eval_ir_session_default(self, _node: IrSessionDefault, default_agent: Value) -> Value:
        """Lazily obtain the session whose agent is current at first use."""
        agent = cast(EnumValue, default_agent)
        transport = self._resolve_session_transport(agent, None)
        try:
            host = self._require_session_host("default")
            handle = host.default(agent, transport)
            snapshot = host.snapshot(handle)
        except SessionHostError as error:
            self._session_error(error)
        return self._session_value(handle, snapshot.agent, snapshot.transport)

    def _dispatch_session_agent(
        self,
        handle: str,
        request: AgentRequest,
        node: IrSessionAsk,
        *,
        max_attempts: int,
        target_type: str,
        codec: str,
        strict_json: bool | None,
        json_schema: object | None,
    ) -> str:
        """Trace, dispatch, and map one request sent through a session."""
        self._ctx._trace.agent_request(
            agent=self._agent_trace_value(request.agent),
            attempt=request.attempt,
            max_attempts=max_attempts,
            prompt=request.prompt,
            target_type=target_type,
            codec=codec,
            strict_json=strict_json,
            json_schema=json_schema,
            span=node.location,
        )
        try:
            raw = self._require_session_host("ask").ask(handle, request.prompt)
        except SessionAskError as error:
            call_info = error.call_info.to_trace() if error.call_info is not None else None
            if call_info is None:
                call_info = {"exit_code": error.exit_code, "elapsed": error.elapsed}
            call_info["stderr_tail"] = error.stderr_tail
            self._ctx._trace.agent_response(
                ok=False, cause=error.cause, call_info=call_info, span=node.location
            )
            self._raise_agent_call_error(
                request.agent,
                AgentCallHostError(
                    cause=error.cause,
                    exit_code=error.exit_code,
                    stderr_tail=error.stderr_tail,
                    elapsed=error.elapsed,
                    call_info=error.call_info,
                ),
            )
        except SessionHostError as error:
            self._session_error(error)
        except AgentCancelled as error:
            self._ctx._trace.agent_response(
                ok=False, cancelled=True, reason=error.reason, span=node.location
            )
            error.span = node.location
            raise
        except KeyboardInterrupt as error:
            cancelled = AgentCancelled(
                render_value(request.agent), "interrupted", span=node.location
            )
            self._ctx._trace.agent_response(
                ok=False, cancelled=True, reason=cancelled.reason, span=node.location
            )
            raise cancelled from error
        self._ctx._trace.agent_response(
            ok=True, content=raw, metadata={}, call_info=None, span=node.location
        )
        return raw

    def eval_ir_session_ask(self, node: IrSessionAsk) -> Value:
        """Send a prompt through a session and retry invalid typed responses in place."""
        handle, agent, _transport = self._session_parts(self._ctx._eval(node.session), "ask")
        prompt = self._text_of(self._ctx._eval(node.prompt))
        contract = self._ctx._program.contracts[node.contract_id]
        effective_strict = (
            contract.strict_json if contract.strict_json is not None else self._ctx._strict_json
        )
        output_contract: OutputContract | None = (
            None if contract.is_unit else self._ctx._host_contracts[node.contract_id]
        )
        json_schema = (
            None if contract.json_schema is None else cast(object, json.loads(contract.json_schema))
        )
        last_raw: str | None = None
        last_normalized: str | None = None
        last_errors: tuple[ReqValidationError, ...] = ()

        for attempt in range(node.max_attempts):
            request = AgentRequest(
                agent=agent,
                prompt=prompt,
                attempt=attempt,
                validation_errors=list(last_errors),
                output_contract=output_contract,
            )
            request.prompt = (
                compose_initial_agent_prompt(request)
                if attempt == 0
                else compose_session_corrective_follow_up(request)
            )
            raw = self._dispatch_session_agent(
                handle,
                request,
                node,
                max_attempts=node.max_attempts,
                target_type=contract.target_type_label,
                codec=contract.codec_name,
                strict_json=contract.strict_json,
                json_schema=json_schema,
            )
            if contract.is_unit:
                return VOID_VALUE
            result = self._ctx._parse_host_output(
                raw, node.contract_id, effective_strict=effective_strict
            )
            self._ctx._trace.parse_result(
                ok=result.ok,
                raw=raw,
                normalized_raw=result.normalized_raw or raw,
                error_summary=result.error_msg
                or "; ".join(error.message for error in result.errors),
                span=node.location,
            )
            if result.ok and result.value is not None:
                return result.value
            last_raw = raw
            last_normalized = result.normalized_raw
            last_errors = self._classify_parse_errors(result)

        self._raise_agent_parse_error(
            message=(
                f"Agent {render_value(agent)!r} failed to produce a valid "
                f"{contract.target_type_label} after {node.max_attempts} attempt(s). "
                f"Last output: {last_raw!r}"
            ),
            agent=agent,
            last_raw=last_raw,
            last_normalized=last_normalized,
            last_errors=last_errors,
            max_attempts=node.max_attempts,
            target_type_label=contract.target_type_label,
            json_schema=contract.json_schema,
        )

    def eval_ir_session_op(self, node: IrSessionOp) -> Value:
        """Dispatch one lifecycle operation through the session host."""
        handle, agent, transport = self._session_parts(self._ctx._eval(node.session), node.op)
        argument = self._text_of(self._ctx._eval(node.arg)) if node.arg is not None else ""
        host = self._require_session_host(node.op)
        try:
            if node.op == "compact":
                host.compact(handle, argument)
                return VOID_VALUE
            if node.op == "reset":
                host.reset(handle)
                return VOID_VALUE
            if node.op == "fork":
                return self._session_value(host.fork(handle), agent, transport)
            if node.op == "stats":
                stats = host.stats(handle)
                declared = self._ctx._program.builtin_nominals.resolve("SessionStats")
                return RecordValue(
                    nominal=declared.nominal,
                    display_name=declared.display_name,
                    fields={
                        "input-tokens": IntValue(stats.input_tokens),
                        "output-tokens": IntValue(stats.output_tokens),
                        "cost": DecimalValue(stats.cost),
                        "context-percent": DecimalValue(stats.context_percent),
                    },
                )
            if node.op == "set-name":
                host.set_name(handle, argument)
                return VOID_VALUE
            assert node.op == "close"
            host.close(handle)
            return VOID_VALUE
        except SessionHostError as error:
            self._session_error(error)

    def eval_ir_ask_request(
        self,
        _node: IrAskRequest,
        agent_expr: IrExpr,
        prompt_expr: IrExpr,
    ) -> Value:
        """Handle IrAskRequest: build AgentRequest record without dispatching."""
        request_agent = self._ctx._eval(agent_expr)
        if not isinstance(request_agent, EnumValue):
            raise TypeError(
                "IrAskRequest agent must evaluate to an Agent enum value, "
                f"got {type(request_agent).__name__}"
            )

        prompt_text = self._text_of(self._ctx._eval(prompt_expr))

        agent_request = self._ctx._program.builtin_nominals.resolve("AgentRequest")
        return RecordValue(
            nominal=agent_request.nominal,
            display_name=agent_request.display_name,
            fields={
                "agent": request_agent,
                "prompt": TextValue(prompt_text),
                "target_type": some_value(TextValue("text")),
                "format_instructions": none_value(),
                "json_schema": none_value(),
                "attempt": IntValue(0),
                "previous_error": none_value(),
                "metadata": JsonValue(
                    {
                        "codec_name": "text",
                        "strict_json": None,
                        "structured_exec": False,
                    }
                ),
            },
        )

    # ------------------------------------------------------------------
    # Exec call helper
    # ------------------------------------------------------------------

    def _raise_exec_error(
        self,
        message: str,
        *,
        command: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        timed_out: bool = False,
    ) -> NoReturn:
        """Raise the builtin ``ExecError`` carrying one shell invocation's outcome.

        Every ``exec`` failure — spawn, timeout, non-zero exit, and typed-output
        parse failure — reports through this one shape.
        """
        raise AglRaise(
            _make_exc_value(
                "ExecError",
                message,
                nominals=self._ctx._program.builtin_nominals,
                command=TextValue(command),
                exit_code=IntValue(exit_code),
                stdout=TextValue(stdout),
                stderr=TextValue(stderr),
                timed_out=BoolValue(timed_out),
            )
        )

    def _raise_exec_parse_error(
        self, *, command: str, last_raw: str | None, last_errors: tuple[ReqValidationError, ...]
    ) -> NoReturn:
        """Raise ``ExecError`` for a typed shell-output parse failure.

        ``AgentParseError`` is reserved for failures produced by an ``Agent``
        value. Shell execution has no agent value to carry, so representing it
        as an agent parse failure would violate that exception's typed shape.
        """
        detail = "; ".join(error.message for error in last_errors)
        self._raise_exec_error(
            f"exec output failed to parse: {detail}" if detail else "exec output failed to parse",
            command=command,
            exit_code=0,
            stdout=last_raw or "",
            stderr=detail,
        )

    def _run_exec_shell(self, cmd: str, location: Location) -> tuple[str, str, int | None]:
        """Run *cmd* via the shell; raise ``ExecError`` on spawn failure or timeout.

        Returns ``(stdout, stderr, returncode)`` — a non-zero exit code is NOT
        raised here so that the structured-exec path can treat it as data.
        Mirrors legacy ``_run_shell_capture`` (without the trace event).
        """
        from agm.core.process import run_capture_result

        result = run_capture_result(
            ["sh", "-c", cmd],
            idle_timeout=self._ctx._shell_exec_timeout,
            isolate_process_group=True,
        )
        if result.spawn_error is not None:
            spawn_error = str(result.spawn_error)
            self._ctx._trace.exec_command(
                command=cmd,
                exit_code=-1,
                duration=result.elapsed,
                stdout=result.stdout,
                stderr=spawn_error,
                timed_out=False,
                span=location,
            )
            self._raise_exec_error(
                f"Failed to spawn shell: {spawn_error}",
                command=cmd,
                exit_code=-1,
                stdout="",
                stderr=spawn_error,
            )
        if result.timed_out:
            exit_code = result.returncode if result.returncode is not None else -1
            self._ctx._trace.exec_command(
                command=cmd,
                exit_code=exit_code,
                duration=result.elapsed,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=True,
                span=location,
            )
            self._raise_exec_error(
                f"Shell command timed out (idle timeout exceeded): {cmd!r}",
                command=cmd,
                exit_code=exit_code,
                stdout=result.stdout.rstrip("\n"),
                stderr=result.stderr.rstrip("\n"),
                timed_out=True,
            )
        self._ctx._trace.exec_command(
            command=cmd,
            exit_code=result.returncode if result.returncode is not None else 0,
            duration=result.elapsed,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=False,
            span=location,
        )
        return result.stdout, result.stderr, result.returncode

    def eval_ir_exec(
        self,
        _node: IrExec,
        command_expr: IrExpr,
        contract_id: ContractId,
        max_attempts: int,
    ) -> Value:
        """Handle IrExec: run shell command and parse output."""
        # 1. Evaluate command expression
        cmd = self._text_of(self._ctx._eval(command_expr))

        contract = self._ctx._program.contracts[contract_id]

        # 2. Run shell once (raises on spawn error or timeout)
        stdout, stderr, returncode = self._run_exec_shell(cmd, _node.location)

        # 3. Structured exec: return ExecResult regardless of exit code
        if contract.structured_exec:
            actual_exit_code = returncode if returncode is not None else 0
            exec_result = self._ctx._program.builtin_nominals.resolve("ExecResult")
            return RecordValue(
                nominal=exec_result.nominal,
                display_name=exec_result.display_name,
                fields={
                    "stdout": TextValue(stdout.rstrip("\n")),
                    "exit_code": IntValue(actual_exit_code),
                    "stderr": TextValue(stderr.rstrip("\n")),
                    "timed_out": BoolValue(False),
                },
            )

        # 4. Non-zero exit raises ExecError (for text/typed execs)
        if returncode is not None and returncode != 0:
            self._raise_exec_error(
                f"Shell command exited with code {returncode}: {cmd!r}",
                command=cmd,
                exit_code=returncode,
                stdout=stdout.rstrip("\n"),
                stderr=stderr.rstrip("\n"),
            )

        # 5. Unit contract: successful output is deliberately discarded.
        if contract.is_unit:
            return VOID_VALUE

        # 6. Text codec: return stdout directly
        captured = stdout.rstrip("\n")
        if contract.codec_name == "text":
            return TextValue(captured)

        # 7. Parse/retry loop for typed exec
        effective_strict = (
            contract.strict_json if contract.strict_json is not None else self._ctx._strict_json
        )

        last_raw: str | None = captured
        last_errors: tuple[ReqValidationError, ...] = ()

        for attempt in range(max_attempts):
            if attempt > 0:
                # Re-run shell on retry (raises on spawn error / timeout / non-zero exit)
                stdout2, stderr2, rc2 = self._run_exec_shell(cmd, _node.location)
                if rc2 is not None and rc2 != 0:
                    raise AglRaise(
                        _make_exc_value(
                            "ExecError",
                            f"Shell command exited with code {rc2}: {cmd!r}",
                            nominals=self._ctx._program.builtin_nominals,
                            command=TextValue(cmd),
                            exit_code=IntValue(rc2),
                            stdout=TextValue(stdout2.rstrip("\n")),
                            stderr=TextValue(stderr2.rstrip("\n")),
                            timed_out=BoolValue(False),
                        )
                    )
                last_raw = stdout2.rstrip("\n")

            result = self._ctx._parse_host_output(
                last_raw or "", contract_id, effective_strict=effective_strict
            )
            self._ctx._trace.parse_result(
                ok=result.ok,
                raw=last_raw or "",
                normalized_raw=result.normalized_raw or (last_raw or ""),
                error_summary=result.error_msg
                or "; ".join(error.message for error in result.errors),
                span=_node.location,
            )

            if result.ok and result.value is not None:
                return result.value

            last_errors = self._classify_parse_errors(result)

        self._raise_exec_parse_error(
            command=cmd,
            last_raw=last_raw,
            last_errors=last_errors,
        )
