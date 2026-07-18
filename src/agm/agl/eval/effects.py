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

from agm.agl.ir.ids import ContractId, Location, NominalId
from agm.agl.ir.nodes import IrAsk, IrAskRequest, IrExec, IrExpr
from agm.agl.ir.program import ExecutableProgram, ExternFunctionBody
from agm.agl.modules.ids import PRELUDE_ID, ModuleId
from agm.agl.runtime.agents import AgentRegistry
from agm.agl.runtime.codec import ParseResult
from agm.agl.runtime.contract import OutputContract
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.runtime.option import none_value, some_value
from agm.agl.runtime.render import render_value
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.runtime.request import ValidationError as ReqValidationError
from agm.agl.runtime.trace import TraceStore
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.exceptions import make_builtin_exception as _make_exc_value
from agm.agl.semantics.values import (
    VOID_VALUE,
    AgentValue,
    BoolValue,
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
    _registry: AgentRegistry
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

    # ------------------------------------------------------------------
    # Extern (Python FFI) call helper
    # ------------------------------------------------------------------

    def eval_extern_call(
        self, module_id: ModuleId, extern: ExternFunctionBody, args: Sequence[Value]
    ) -> Value:
        """Handle a call to an ``extern def``: resolve, mint a trace id, invoke.

        The companion callable was already imported at program load
        (``pipeline._wire_extern_registry``); this only looks it up by name.
        ``ExternRegistry.invoke`` is the single chokepoint that turns every
        runtime failure crossing the boundary — a raising callable, an
        argument-conversion failure, or a return-contract violation — into
        ``AglRaise(ExternError)``, mirroring the ``exec`` model.
        """
        fn = self._ctx._extern_registry.resolve(module_id, extern.name)
        trace_id = self._ctx._trace.new_event_id()
        return self._ctx._extern_registry.invoke(extern.name, extern.contract, fn, args, trace_id)

    # ------------------------------------------------------------------
    # Agent call helpers
    # ------------------------------------------------------------------

    def _dispatch_agent(self, agent_name: str, request: AgentRequest, node: IrAsk) -> AgentResponse:
        """Dispatch an agent call, annotating cancellation with the ask span.

        ``AgentCancelled`` (and a bare ``KeyboardInterrupt`` from an unwrapped
        default agent) propagates out of ``AgentRegistry.dispatch`` without source
        context. The REPL session needs the cancelled ``ask`` node's location to
        apply partial-effects promotion by source position (just like ``AglRaise``
        ), so we attach it here. A raw ``KeyboardInterrupt`` is normalized to
        ``AgentCancelled(reason="interrupted")`` to match the
        ``ConfirmingAgent`` conversion and give the session a uniform carrier.
        """
        from agm.agl.runtime.request import AgentCancelled

        try:
            return self._ctx._registry.dispatch(agent_name, request)
        except AgentCancelled as exc:
            exc.span = node.location
            raise
        except KeyboardInterrupt as exc:
            raise AgentCancelled(agent_name, "interrupted", span=node.location) from exc

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
        agent_name: str,
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
                trace_id=self._ctx._trace.new_event_id(),
                raw=TextValue(last_raw or ""),
                normalized_raw=TextValue(normalized_text),
                agent=TextValue(agent_name),
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
        """Handle IrAsk: dispatch agent and parse output."""
        agent_val = self._ctx._eval(agent_expr)
        agent_name = agent_val.name if isinstance(agent_val, AgentValue) else "ask"

        prompt_val = self._ctx._eval(prompt_expr)
        if not isinstance(prompt_val, TextValue):
            prompt_text = render_value(prompt_val)
        else:
            prompt_text = prompt_val.value

        contract = self._ctx._program.contracts[contract_id]

        # Unit-typed ask: dispatch once, no output parsing.
        if contract.is_unit:
            request = AgentRequest(
                agent=agent_name,
                prompt=prompt_text,
                output_contract=None,
            )
            self._dispatch_agent(agent_name, request, _node)
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

        for attempt in range(max_attempts):
            self._ctx._trace.agent_call_attempt(
                agent=agent_name,
                attempt=attempt,
                prompt=prompt_text,
                span=_node.location,
            )
            request = AgentRequest(
                agent=agent_name,
                prompt=prompt_text,
                attempt=attempt,
                previous_invalid_output=last_raw,
                validation_errors=list(last_errors),
                output_contract=output_contract,
            )
            response = self._dispatch_agent(agent_name, request, _node)
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
            agent_name=agent_name,
            last_raw=last_raw,
            last_normalized=last_normalized,
            last_errors=last_errors,
            max_attempts=max_attempts,
            target_type_label=contract.target_type_label,
            json_schema=contract.json_schema,
        )

    def eval_ir_ask_request(
        self,
        _node: IrAskRequest,
        agent_expr: IrExpr,
        prompt_expr: IrExpr,
        contract_id: ContractId,
    ) -> Value:
        """Handle IrAskRequest: build AgentRequest record without dispatching."""
        agent_val = self._ctx._eval(agent_expr)
        agent_name = agent_val.name if isinstance(agent_val, AgentValue) else "ask"

        prompt_val = self._ctx._eval(prompt_expr)
        if not isinstance(prompt_val, TextValue):
            prompt_text = render_value(prompt_val)
        else:
            prompt_text = prompt_val.value

        contract = self._ctx._program.contracts[contract_id]

        json_schema_value: Value
        if contract.json_schema is None:
            json_schema_value = none_value()
        else:
            json_schema_value = some_value(
                JsonValue(cast(object, json.loads(contract.json_schema)))
            )

        return RecordValue(
            nominal=NominalId(PRELUDE_ID, "AgentRequest"),
            display_name="AgentRequest",
            fields={
                "agent": TextValue(agent_name),
                "prompt": TextValue(prompt_text),
                "target_type": none_value()
                if contract.is_unit
                else some_value(TextValue(contract.target_type_label)),
                "format_instructions": none_value()
                if not contract.format_instructions
                else some_value(TextValue(contract.format_instructions)),
                "json_schema": json_schema_value,
                "attempt": IntValue(0),
                "previous_error": none_value(),
                "metadata": JsonValue(
                    {
                        "codec_name": contract.codec_name,
                        "strict_json": contract.strict_json,
                        "structured_exec": contract.structured_exec,
                    }
                ),
            },
        )

    # ------------------------------------------------------------------
    # Exec call helper
    # ------------------------------------------------------------------

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
            trace_id = self._ctx._trace.exec_command(
                command=cmd,
                exit_code=-1,
                duration=result.elapsed,
                stdout=result.stdout,
                stderr=spawn_error,
                timed_out=False,
                span=location,
            )
            raise AglRaise(
                _make_exc_value(
                    "ExecError",
                    f"Failed to spawn shell: {spawn_error}",
                    trace_id=trace_id,
                    command=TextValue(cmd),
                    exit_code=IntValue(-1),
                    stdout=TextValue(""),
                    stderr=TextValue(spawn_error),
                    timed_out=BoolValue(False),
                )
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
            raise AglRaise(
                _make_exc_value(
                    "ExecError",
                    f"Shell command timed out (idle timeout exceeded): {cmd!r}",
                    trace_id=self._ctx._trace.new_event_id(),
                    command=TextValue(cmd),
                    exit_code=IntValue(exit_code),
                    stdout=TextValue(result.stdout.rstrip("\n")),
                    stderr=TextValue(result.stderr.rstrip("\n")),
                    timed_out=BoolValue(True),
                )
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
        command_val = self._ctx._eval(command_expr)
        if not isinstance(command_val, TextValue):
            cmd = render_value(command_val)
        else:
            cmd = command_val.value

        contract = self._ctx._program.contracts[contract_id]

        # 2. Run shell once (raises on spawn error or timeout)
        stdout, stderr, returncode = self._run_exec_shell(cmd, _node.location)

        # 3. Structured exec: return ExecResult regardless of exit code
        if contract.structured_exec:
            actual_exit_code = returncode if returncode is not None else 0
            return RecordValue(
                nominal=NominalId(PRELUDE_ID, "ExecResult"),
                display_name="ExecResult",
                fields={
                    "stdout": TextValue(stdout.rstrip("\n")),
                    "exit_code": IntValue(actual_exit_code),
                    "stderr": TextValue(stderr.rstrip("\n")),
                    "timed_out": BoolValue(False),
                },
            )

        # 4. Non-zero exit raises ExecError (for text/typed execs)
        if returncode is not None and returncode != 0:
            raise AglRaise(
                _make_exc_value(
                    "ExecError",
                    f"Shell command exited with code {returncode}: {cmd!r}",
                    trace_id=self._ctx._trace.new_event_id(),
                    command=TextValue(cmd),
                    exit_code=IntValue(returncode),
                    stdout=TextValue(stdout.rstrip("\n")),
                    stderr=TextValue(stderr.rstrip("\n")),
                    timed_out=BoolValue(False),
                )
            )

        # 5. Text codec: return stdout directly
        captured = stdout.rstrip("\n")
        if contract.codec_name == "text":
            return TextValue(captured)

        # 6. Parse/retry loop for typed exec
        effective_strict = (
            contract.strict_json if contract.strict_json is not None else self._ctx._strict_json
        )

        last_raw: str | None = captured
        last_normalized: str | None = None
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
                            trace_id=self._ctx._trace.new_event_id(),
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

            last_normalized = result.normalized_raw
            last_errors = self._classify_parse_errors(result)

        self._raise_agent_parse_error(
            message=(
                f"exec output failed to parse as {contract.target_type_label} "
                f"after {max_attempts} attempt(s). Last output: {last_raw!r}"
            ),
            agent_name="exec",
            last_raw=last_raw,
            last_normalized=last_normalized,
            last_errors=last_errors,
            max_attempts=max_attempts,
            target_type_label=contract.target_type_label,
            json_schema=contract.json_schema,
        )
