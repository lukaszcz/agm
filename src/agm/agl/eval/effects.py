"""Host-effect handler collaborator for the AgL IR evaluator.

``EffectHandlers`` implements agent dispatch (``ask``, ``ask-request``) and
``exec`` (shell execution + output parsing/retry).  It is driven by
``IrInterpreter`` via the narrow ``EffectCtx`` Protocol and must NOT import
``ir_interpreter`` (no cycle).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ContextManager, NoReturn, Protocol, assert_never, cast

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
    IrSessionOpKind,
)
from agm.agl.ir.program import ExecutableProgram, ExternFunctionBody
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.codec import ParseResult
from agm.agl.runtime.contract import OutputContract, TypelessOutputContract
from agm.agl.runtime.externs import ExternRegistry, ExternRuntimeState
from agm.agl.runtime.option import none_value, option_text, some_value
from agm.agl.runtime.render import render_value
from agm.agl.runtime.request import (
    AgentCallHostError,
    AgentCancelled,
    AgentRequest,
    compose_agent_prompt,
    compose_initial_agent_prompt,
    compose_session_corrective_follow_up,
)
from agm.agl.runtime.request import (
    ValidationError as ReqValidationError,
)
from agm.agl.runtime.sessions import (
    AgentDispatcherSessionHost,
    SessionAgentError,
    SessionAskError,
    SessionHost,
    SessionHostError,
    SessionRequestHost,
    default_session_transport,
    with_ephemeral_session,
)
from agm.agl.runtime.trace import TraceStore
from agm.agl.semantics.cycles import AglCyclicValue, cyclic_value_raise
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.exceptions import make_builtin_exception as _make_exc_value
from agm.agl.semantics.types import terminal_name
from agm.agl.semantics.values import (
    VOID_VALUE,
    BoolValue,
    DecimalValue,
    DictValue,
    ExceptionValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.core.parse import parse_timeout

# ---------------------------------------------------------------------------
# Narrow context Protocol
# ---------------------------------------------------------------------------


class EffectCtx(Protocol):
    """The minimal IrInterpreter surface the effect handlers need."""

    _program: ExecutableProgram
    _trace: TraceStore
    _agent_dispatcher: AgentFn | None
    _session_host: SessionHost
    _strict_json: bool
    _host_contracts: Mapping[ContractId, OutputContract]
    _extern_registry: ExternRegistry
    _extern_runtime_state: ExternRuntimeState

    def _eval(self, expr: IrExpr) -> Value: ...

    def _encode_extern_value(self, value: Value) -> object: ...

    def _extern_call_window(self) -> ContextManager[None]: ...

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
        with self._ctx._extern_call_window():
            return self._ctx._extern_registry.invoke(
                extern.name,
                fn,
                args,
                nominals=self._ctx._program.builtin_nominals,
                function_encoder=self._ctx._encode_extern_value,
                runtime_state=self._ctx._extern_runtime_state,
            )

    # ------------------------------------------------------------------
    # Agent call helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_trace_value(agent: RecordValue) -> dict[str, object]:
        """Return the agent variant and payload without re-decoding it."""
        return {
            "variant": terminal_name(agent.display_name),
            "payload": {
                name: value.value if isinstance(value, TextValue) else render_value(value)
                for name, value in agent.fields.items()
            },
        }

    def _raise_agent_call_error(self, agent: RecordValue, error: AgentCallHostError) -> NoReturn:
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
        if not isinstance(agent_val, RecordValue):
            raise TypeError(
                "IrAsk agent must evaluate to an Agent member record, "
                f"got {type(agent_val).__name__}"
            )
        prompt_text = self._text_of(self._ctx._eval(prompt_expr))

        output_contract, json_schema = self._contract_carriers(contract_id)
        return self._eval_agent_method_ask(
            agent=agent_val,
            prompt=prompt_text,
            contract_id=contract_id,
            max_attempts=max_attempts,
            node=_node,
            output_contract=output_contract,
            json_schema=json_schema,
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

    def _invalid_agent_error(self, agent: RecordValue, error: SessionAgentError) -> NoReturn:
        """Preserve agent-value failures across the session-opening boundary."""
        self._raise_agent_call_error(
            agent,
            AgentCallHostError(
                cause="invalid_agent",
                exit_code=None,
                stderr_tail=error.message,
                elapsed=0.0,
            ),
        )

    def _contract_carriers(
        self, contract_id: ContractId
    ) -> tuple[OutputContract | TypelessOutputContract | None, object | None]:
        """Return the output contract and decoded JSON schema carried by an ask."""
        contract = self._ctx._program.contracts[contract_id]
        if contract.is_unit:
            return None, None
        json_schema = (
            None if contract.json_schema is None else cast(object, json.loads(contract.json_schema))
        )
        output_contract = self._ctx._host_contracts.get(contract_id) or TypelessOutputContract(
            target_type=contract.target_type_label,
            codec_name=contract.codec_name,
            strict_json=contract.strict_json,
            format_instructions=contract.format_instructions,
            json_schema=json_schema,
            structured_exec=contract.structured_exec,
        )
        return output_contract, json_schema

    def _session_value(self, handle: str, agent: RecordValue, transport: str) -> RecordValue:
        declared = self._ctx._program.builtin_nominals.resolve("Session")
        return RecordValue(
            nominal=declared.nominal,
            display_name=declared.display_name,
            fields={
                "id": TextValue(handle),
                "agent": agent,
                "transport": RecordValue(
                    nominal=(
                        transport_member
                        := self._ctx._program.builtin_nominals.resolve_standard_member(
                            "SessionTransport", transport
                        )
                    ).nominal,
                    display_name=transport_member.display_name,
                    fields={},
                ),
            },
        )

    def _session_parts(self, value: Value) -> tuple[str, RecordValue, str]:
        """Extract the statically guaranteed fields from a ``Session`` record."""
        session = cast(RecordValue, value)
        return (
            cast(TextValue, session.fields["id"]).value,
            cast(RecordValue, session.fields["agent"]),
            terminal_name(cast(RecordValue, session.fields["transport"]).display_name),
        )

    def _resolve_session_transport(self, agent: RecordValue, transport: Value | None) -> str:
        selected = None if transport is None else cast(RecordValue, transport)
        if selected is None or terminal_name(selected.display_name) == "None":
            return default_session_transport(agent)
        return terminal_name(cast(RecordValue, selected.fields["value"]).display_name)

    def eval_ir_session_open(self, node: IrSessionOpen) -> Value:
        """Open a host-backed session and mint its opaque AgL record."""
        agent = cast(RecordValue, self._ctx._eval(node.agent))
        transport = self._resolve_session_transport(
            agent, None if node.transport is None else self._ctx._eval(node.transport)
        )
        name = self._text_of(self._ctx._eval(node.name))
        try:
            handle = self._ctx._session_host.open(agent, transport, name=name)
        except SessionHostError as error:
            self._session_error(error)
        return self._session_value(handle, agent, transport)

    def eval_ir_session_default(self, _node: IrSessionDefault, default_agent: Value) -> Value:
        """Lazily obtain the session whose agent is current at first use."""
        agent = cast(RecordValue, default_agent)
        transport = self._resolve_session_transport(agent, None)
        try:
            host = self._ctx._session_host
            handle = host.default(agent, transport)
            snapshot = host.snapshot(handle)
        except SessionHostError as error:
            self._session_error(error)
        return self._session_value(handle, snapshot.agent, snapshot.transport)

    def _dispatch_session_agent(
        self,
        handle: str,
        request: AgentRequest,
        node: IrAsk | IrSessionAsk,
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
        metadata: dict[str, object] = {}
        call_info: dict[str, object] | None = None
        try:
            host = self._ctx._session_host
            if isinstance(host, SessionRequestHost):
                response = host.ask_request(handle, request)
                raw = response.content
                metadata = response.metadata
                call_info = (
                    response.call_info.to_trace() if response.call_info is not None else None
                )
            else:
                raw = host.ask(handle, request.prompt)
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
            ok=True, content=raw, metadata=metadata, call_info=call_info, span=node.location
        )
        return raw

    def _eval_agent_method_ask(
        self,
        *,
        agent: RecordValue,
        prompt: str,
        contract_id: ContractId,
        max_attempts: int,
        node: IrAsk,
        output_contract: OutputContract | TypelessOutputContract | None,
        json_schema: object | None,
    ) -> Value:
        """Run one ``Agent::ask`` call in a short-lived conversation.

        The handle lives for the complete retry loop, so a corrective retry is
        a follow-up rather than a fresh one-shot prompt. It is always released
        once the call completes or fails.
        """
        contract = self._ctx._program.contracts[contract_id]
        transport = self._resolve_session_transport(agent, None)

        def ask_in_session(handle: str) -> Value:
            return self._eval_session_ask_attempts(
                agent=agent,
                prompt=prompt,
                contract_id=contract_id,
                max_attempts=max_attempts,
                node=node,
                output_contract=output_contract,
                dispatch=lambda request: self._dispatch_session_agent(
                    handle,
                    request,
                    node,
                    max_attempts=max_attempts,
                    target_type=contract.target_type_label,
                    codec=contract.codec_name,
                    strict_json=contract.strict_json,
                    json_schema=json_schema,
                ),
            )

        try:
            return with_ephemeral_session(
                self._ctx._session_host,
                agent,
                transport,
                ask_in_session,
                single_prompt=max_attempts == 1,
            )
        except SessionAgentError as error:
            self._invalid_agent_error(agent, error)
        except SessionHostError as error:
            self._session_error(error)

    def eval_ir_session_ask(self, node: IrSessionAsk) -> Value:
        """Send a prompt through a session and run its shared retry engine."""
        handle, agent, _transport = self._session_parts(self._ctx._eval(node.session))
        prompt = self._text_of(self._ctx._eval(node.prompt))
        contract = self._ctx._program.contracts[node.contract_id]
        output_contract, json_schema = self._contract_carriers(node.contract_id)
        return self._eval_session_ask_attempts(
            agent=agent,
            prompt=prompt,
            contract_id=node.contract_id,
            max_attempts=node.max_attempts,
            node=node,
            output_contract=output_contract,
            dispatch=lambda request: self._dispatch_session_agent(
                handle,
                request,
                node,
                max_attempts=node.max_attempts,
                target_type=contract.target_type_label,
                codec=contract.codec_name,
                strict_json=contract.strict_json,
                json_schema=json_schema,
            ),
        )

    def _compose_session_prompt(self, request: AgentRequest) -> str:
        if isinstance(self._ctx._session_host, AgentDispatcherSessionHost):
            return compose_agent_prompt(request)
        if request.attempt == 0:
            return compose_initial_agent_prompt(request)
        return compose_session_corrective_follow_up(request)

    def _eval_session_ask_attempts(
        self,
        *,
        agent: RecordValue,
        prompt: str,
        contract_id: ContractId,
        max_attempts: int,
        node: IrAsk | IrSessionAsk,
        output_contract: OutputContract | TypelessOutputContract | None,
        dispatch: Callable[[AgentRequest], str],
    ) -> Value:
        """Run the session ask retry loop."""
        contract = self._ctx._program.contracts[contract_id]
        effective_strict = (
            contract.strict_json if contract.strict_json is not None else self._ctx._strict_json
        )
        last_raw: str | None = None
        last_normalized: str | None = None
        last_errors: tuple[ReqValidationError, ...] = ()

        for attempt in range(max_attempts):
            request = AgentRequest(
                agent=agent,
                prompt=prompt,
                attempt=attempt,
                previous_invalid_output=last_raw,
                validation_errors=list(last_errors),
                output_contract=output_contract,
            )
            request.prompt = self._compose_session_prompt(request)
            raw = dispatch(request)
            if contract.is_unit:
                return VOID_VALUE
            result = self._ctx._parse_host_output(
                raw, contract_id, effective_strict=effective_strict
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
                f"{contract.target_type_label} after {max_attempts} attempt(s). "
                f"Last output: {last_raw!r}"
            ),
            agent=agent,
            last_raw=last_raw,
            last_normalized=last_normalized,
            last_errors=last_errors,
            max_attempts=max_attempts,
            target_type_label=contract.target_type_label,
            json_schema=contract.json_schema,
        )

    def eval_ir_session_op(self, node: IrSessionOp) -> Value:
        """Dispatch one lifecycle operation through the session host."""
        handle, agent, transport = self._session_parts(self._ctx._eval(node.session))
        argument = self._text_of(self._ctx._eval(node.arg)) if node.arg is not None else ""
        host = self._ctx._session_host
        try:
            match node.op:
                case IrSessionOpKind.COMPACT:
                    host.compact(handle, argument)
                case IrSessionOpKind.RESET:
                    host.reset(handle)
                case IrSessionOpKind.SET_NAME:
                    host.set_name(handle, argument)
                case IrSessionOpKind.CLOSE:
                    host.close(handle)
                case IrSessionOpKind.FORK:
                    return self._session_value(host.fork(handle), agent, transport)
                case IrSessionOpKind.STATS:
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
                case _ as unreachable:  # pragma: no cover
                    assert_never(unreachable)
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
        agent_value = self._ctx._eval(agent_expr)
        if not isinstance(agent_value, RecordValue):
            raise TypeError(
                "IrAskRequest agent must evaluate to an Agent member record, "
                f"got {type(agent_value).__name__}"
            )
        prompt_text = self._text_of(self._ctx._eval(prompt_expr))

        agent_request = self._ctx._program.builtin_nominals.resolve("AgentRequest")
        nominals = self._ctx._program.builtin_nominals
        return RecordValue(
            nominal=agent_request.nominal,
            display_name=agent_request.display_name,
            fields={
                "agent": agent_value,
                "prompt": TextValue(prompt_text),
                "target_type": some_value(TextValue("text"), nominals=nominals),
                "format_instructions": none_value(nominals=nominals),
                "json_schema": none_value(nominals=nominals),
                "attempt": IntValue(0),
                "previous_error": none_value(nominals=nominals),
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

    def _run_exec_shell(
        self,
        cmd: str,
        env: dict[str, str],
        cwd: Path | None,
        timeout: float | None,
        location: Location,
    ) -> tuple[str, str, int | None]:
        """Run *cmd* via the shell; raise ``ExecError`` on spawn failure or timeout.

        Returns ``(stdout, stderr, returncode)`` — a non-zero exit code is NOT
        raised here so that the structured-exec path can treat it as data.
        Mirrors legacy ``_run_shell_capture`` (without the trace event).
        """
        from agm.core.process import run_capture_result

        result = run_capture_result(
            ["sh", "-c", cmd],
            idle_timeout=timeout,
            cwd=cwd,
            env=env,
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
        env_expr: IrExpr,
        cwd_expr: IrExpr,
        timeout_expr: IrExpr,
        contract_id: ContractId,
        max_attempts: int,
    ) -> Value:
        """Handle IrExec: run shell command and parse output."""
        # Evaluate every call operand once. Retried parsing reruns the shell,
        # not the argument expressions, just as an ordinary call would.
        cmd = self._text_of(self._ctx._eval(command_expr))
        environ = self._ctx._eval(env_expr)
        vars_value: Value
        if isinstance(environ, DictValue):
            vars_value = environ
        else:
            assert isinstance(environ, RecordValue)
            vars_value = environ.fields["vars"]
        assert isinstance(vars_value, DictValue)
        env: dict[str, str] = {}
        for name, value in vars_value.entries.items():
            assert isinstance(value, TextValue)
            env[name] = value.value
        cwd_value = self._ctx._eval(cwd_expr)
        assert isinstance(cwd_value, RecordValue)
        cwd_text = option_text(cwd_value, nominals=self._ctx._program.builtin_nominals)
        timeout_value = self._ctx._eval(timeout_expr)
        assert isinstance(timeout_value, RecordValue)
        timeout_text = option_text(timeout_value, nominals=self._ctx._program.builtin_nominals)
        if timeout_text is None:
            timeout = None
        else:
            try:
                timeout = parse_timeout(timeout_text)
            except ValueError as exc:
                raise AglRaise(
                    _make_exc_value(
                        "TypeError",
                        f"invalid timeout: {exc}",
                        nominals=self._ctx._program.builtin_nominals,
                    )
                ) from exc
        cwd = None if cwd_text is None else Path(cwd_text)

        contract = self._ctx._program.contracts[contract_id]

        # Run shell once (raises on spawn error or timeout).
        stdout, stderr, returncode = self._run_exec_shell(cmd, env, cwd, timeout, _node.location)

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
                stdout2, stderr2, rc2 = self._run_exec_shell(cmd, env, cwd, timeout, _node.location)
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
