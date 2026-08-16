"""End-to-end behavior tests for AgL (the `agm exec` workflow DSL).

This suite is the TDD specification for the AgL implementation. Every
tests/agl/programs/**/*.agl file is a complete AgL program executed under
each scenario in its sidecar `<name>.scenarios.json`; every
tests/agl/rejections/**/*.agl file is an invalid program that the static
pipeline must reject before executing anything. The data format is documented
in tests/agl/README.md.

Public contract exercised here (AgL implementation requirements , the AgL DSL design ):

    from agm.agl import PipelineDriver

    runtime = PipelineDriver(
        default_strict_json=False,  # lenient JSON recovery is the default
        agent_dispatcher=fn,        # fn(request) -> str
    )
    result = runtime.run(source, param_values={...})

RunResult surface asserted:

    result.ok           True iff static checks and param validation passed and
                        no uncaught AgL exception was raised
    result.diagnostics  pre-execution failures (static errors, param
                        validation), each with `.message: str` and
                        `.line: int` (1-based source line)
    result.error        the uncaught AgL exception or None, exposing
                        `.type_name: str` and `.fields` — a mapping of the
                        exception's declared fields (including "message") to
                        JSON-shaped Python values

`print` writes to the process stdout (captured with capsys).
"""

from __future__ import annotations

import json
import unittest.mock
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from agm.agent.session import (
    SessionCapabilities,
    SessionHostError,
    SessionOperation,
    SessionService,
)
from tests._agl_helpers import prepare_inline_command
from tests._process_helpers import FakeShell

AGL_DIR = Path(__file__).parent / "agl"
PROGRAMS_DIR = AGL_DIR / "programs"
REJECTIONS_DIR = AGL_DIR / "rejections"
REPO_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"
EXTERNS_PROGRAMS_DIR = PROGRAMS_DIR / "externs"
RESOURCE_PROGRAMS_DIR = PROGRAMS_DIR / "resources"
_SESSION_STATIC_DECLARATIONS = """\
builtin def Session::open(
  agent: Agent,
  transport: Option[SessionTransport] = Option[SessionTransport]::None,
  name: text = "",
) -> Session
builtin def Session::default() -> Session
"""


def _load_json(path: Path) -> Any:
    # parse_float=Decimal: AgL has no binary floats; scenario
    # params and expected exception fields must round-trip decimals exactly.
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


@dataclass
class _ScriptedSession:
    """One deterministic session observation owned by a scripted agent."""

    tag: str
    parent: str | None
    transport: str = ""
    one_shot: bool = False
    opened: bool = False
    closed: bool = False
    prompts: list[str] = field(default_factory=list)
    operations: list[tuple[str, str | None, str]] = field(default_factory=list)
    backend: Any = field(init=False, repr=False)


_SESSION_CAPABILITY_NAMES = frozenset(operation.value for operation in SessionOperation)


def _outcome_name(outcome: Any) -> str:
    return str(outcome.get("outcome", "success")) if isinstance(outcome, dict) else str(outcome)


@dataclass
class ScriptedAgent:
    """Replays scripted responses and records ordinary and session prompts.

    ``schemas`` records, alongside each ordinary call's ``prompt``, the
    structured JSON Schema from the output contract for that same call. It is
    separate from ``prompts`` so existing literal-prompt assertions are
    unaffected. Session observations use deterministic tags, letting scenarios
    assert conversation identity without depending on host handles.
    """

    name: str
    responses: list[str]
    repeat_last: bool = False
    session_capabilities: frozenset[str] = field(default_factory=lambda: _SESSION_CAPABILITY_NAMES)
    session_operations: dict[str, list[Any]] = field(default_factory=dict)
    session_ask_outcomes: list[Any] | None = None
    prompts: list[str] = field(default_factory=list)
    schemas: list[Any] = field(default_factory=list)
    prompt_events: list[tuple[str, Any]] = field(default_factory=list)
    sessions: list[_ScriptedSession] = field(default_factory=list)
    overflowed: bool = False
    session_operations_overflowed: bool = False
    session_ask_outcomes_overflowed: bool = False
    require_session_id: bool = False
    _response_count: int = 0

    def __post_init__(self) -> None:
        if any(
            _outcome_name(outcome) == "unsupported"
            for outcome in self.session_operations.get("reset", [])
        ):
            raise ValueError("reset does not support an unsupported outcome")

    def __call__(self, request: Any) -> str:
        self.prompts.append(request.prompt)
        contract = request.output_contract
        schema = contract.json_schema if contract is not None else None
        self.schemas.append(schema)
        self.prompt_events.append((request.prompt, schema))
        return self._next_response()

    def session_service(self) -> Any:
        """Create an observable wrapper around the production session service."""
        return _ScriptedSessionService(self)

    def _new_session_backend(self, agent: object, transport: str) -> Any:
        """Build the same transport-specific backend production would select."""
        from agm.agent.runner import command_targets_session_id
        from agm.agent.spec import AgentClaude, AgentCodex, AgentCommand, AgentPi
        from agm.agl.runtime.agents import decode_agent_value

        if transport == "scripted":
            session = _ScriptedSession(tag=f"session-{len(self.sessions) + 1}", parent=None)
            self.sessions.append(session)
            return _ScriptedSessionBackend(
                self, session, frozenset(SessionOperation), supports_name=True
            )
        try:
            spec = decode_agent_value(agent)
        except (AttributeError, ValueError) as error:
            raise SessionHostError(str(error), "open") from error
        if transport == "rpc":
            if not isinstance(spec, AgentPi):
                raise SessionHostError("RPC transport is only supported by AgentPi", "open")
            capabilities = frozenset(SessionOperation)
            backend_type: type[_ScriptedSessionBackend] = _ScriptedPiRpcSessionBackend
            supports_name = True
        elif transport == "cli":
            backend_type = _ScriptedSessionBackend
            if isinstance(spec, AgentCommand):
                try:
                    argv = spec.argv()
                except ValueError as error:
                    raise SessionHostError(str(error), "open") from error
                if self.require_session_id and not command_targets_session_id(argv):
                    raise SessionHostError(
                        "command session requires a %{SESSION_ID} placeholder; "
                        "use [exec] default-agent instead",
                        "open",
                    )
                capabilities = frozenset({SessionOperation.ASK})
                supports_name = False
            elif isinstance(spec, AgentClaude):
                capabilities = frozenset(
                    {SessionOperation.ASK, SessionOperation.COMPACT, SessionOperation.FORK}
                )
                supports_name = True
            elif isinstance(spec, AgentCodex):
                capabilities = frozenset({SessionOperation.ASK})
                supports_name = False
            elif isinstance(spec, AgentPi):
                capabilities = frozenset({SessionOperation.ASK, SessionOperation.FORK})
                supports_name = True
            else:
                raise SessionHostError("unsupported session agent", "open")
        else:
            raise SessionHostError(f"unsupported session transport {transport!r}", "open")
        session = _ScriptedSession(tag=f"session-{len(self.sessions) + 1}", parent=None)
        self.sessions.append(session)
        return backend_type(self, session, capabilities, supports_name=supports_name)

    def _fork_session(self, parent: _ScriptedSession) -> Any:
        session = _ScriptedSession(
            tag=f"session-{len(self.sessions) + 1}", parent=parent.tag, opened=True
        )
        self.sessions.append(session)
        backend = parent.backend
        return type(backend)(
            self,
            session,
            backend._native_capabilities,
            supports_name=backend._supports_name,
        )

    def _next_response(self) -> str:
        index = self._response_count
        self._response_count += 1
        if index < len(self.responses):
            return self.responses[index]
        if self.repeat_last and self.responses:
            return self.responses[-1]
        self.overflowed = True
        return ""

    def _next_session_ask_outcome(self) -> Any:
        if self.session_ask_outcomes is None:
            return "success"
        if not self.session_ask_outcomes:
            self.session_ask_outcomes_overflowed = True
            return "success"
        return self.session_ask_outcomes.pop(0)

    def _next_operation(self, operation: str) -> Any:
        script = self.session_operations.get(operation)
        if script is None:
            return "success"
        if not script:
            self.session_operations_overflowed = True
            return "success"
        return script.pop(0)

    def _operation_outcome(self, operation: str) -> str:
        script = self.session_operations.get(operation)
        if not script:
            return "success"
        return _outcome_name(script[0])

    def _supports_operation(self, operation: str) -> bool:
        return (
            operation in self.session_capabilities
            and self._operation_outcome(operation) != "unsupported"
        )

    def _consume_capability_rejection(self, operation: str) -> None:
        if self._operation_outcome(operation) == "unsupported":
            self._next_operation(operation)


class _ScriptedSessionService:
    """Record test-only session attempts while delegating behavior to ``SessionService``."""

    def __init__(self, agent: ScriptedAgent) -> None:
        self._agent = agent
        self._service = SessionService(agent._new_session_backend)
        self._sessions: dict[str, _ScriptedSession] = {}
        self._backends: dict[str, _ScriptedSessionBackend] = {}
        self._ephemeral_handles: set[str] = set()

    def open(self, agent: object, transport: str, *, name: str = "") -> str:
        handle = self._service.open(agent, transport, name=name)
        self._sessions[handle] = self._agent.sessions[-1]
        self._backends[handle] = self._agent.sessions[-1].backend
        return handle

    def open_ephemeral(self, agent: object, transport: str, *, one_shot: bool = False) -> str:
        handle = self._service.open(agent, transport, ephemeral=True, one_shot=one_shot)
        self._sessions[handle] = self._agent.sessions[-1]
        self._backends[handle] = self._agent.sessions[-1].backend
        self._ephemeral_handles.add(handle)
        return handle

    def with_ephemeral(
        self,
        agent: object,
        transport: str,
        action: Callable[[str], Any],
        *,
        on_closed: Callable[[str], None] | None = None,
        one_shot: bool = False,
    ) -> Any:
        def register(handle: str) -> Any:
            self._sessions[handle] = self._agent.sessions[-1]
            self._backends[handle] = self._agent.sessions[-1].backend
            self._ephemeral_handles.add(handle)
            return action(handle)

        def retire(handle: str) -> None:
            self._retire_ephemeral(handle)
            if on_closed is not None:
                on_closed(handle)

        return self._service.with_ephemeral(
            agent, transport, register, on_closed=retire, one_shot=one_shot
        )

    def default(self, agent: object, transport: str, *, name: str = "") -> str:
        handle = self._service.default(agent, transport, name=name)
        if handle not in self._sessions:
            self._sessions[handle] = self._agent.sessions[-1]
            self._backends[handle] = self._agent.sessions[-1].backend
        return handle

    def ask(self, handle: str, request: Any) -> Any:
        return self._service.ask(handle, request)

    def compact(self, handle: str, instructions: str = "") -> None:
        self._attempt(
            handle, "compact", instructions, lambda: self._service.compact(handle, instructions)
        )

    def reset(self, handle: str) -> None:
        self._attempt(handle, "reset", None, lambda: self._service.reset(handle))

    def fork(self, handle: str) -> str:
        forked = self._attempt(handle, "fork", None, lambda: self._service.fork(handle))
        self._sessions[forked] = self._agent.sessions[-1]
        self._backends[forked] = self._agent.sessions[-1].backend
        return forked

    def set_name(self, handle: str, name: str) -> None:
        self._attempt(handle, "set-name", name, lambda: self._service.set_name(handle, name))

    def stats(self, handle: str) -> Any:
        return self._attempt(handle, "stats", None, lambda: self._service.stats(handle))

    def close(self, handle: str) -> None:
        self._service.close(handle)
        self._retire_ephemeral(handle)

    def close_all(self) -> None:
        try:
            self._service.close_all()
        finally:
            for handle in tuple(self._ephemeral_handles):
                if not self._service.is_known(handle):
                    self._retire_ephemeral(handle)

    def _retire_ephemeral(self, handle: str) -> None:
        if handle in self._ephemeral_handles:
            self._ephemeral_handles.remove(handle)
            del self._sessions[handle]
            del self._backends[handle]

    def ask_ephemeral(self, agent: object, transport: str, request: Any, *, name: str = "") -> Any:
        return self._service.ask_ephemeral(agent, transport, request, name=name)

    def _attempt(
        self,
        handle: str,
        operation: str,
        arg: str | None,
        action: Callable[[], Any],
    ) -> Any:
        session = self._sessions.get(handle)
        backend = self._backends.get(handle)
        rejected = (
            session is not None
            and backend is not None
            and operation != "reset"
            and not backend.supports(operation)
        )
        if rejected:
            session.operations.append((operation, arg, "unsupported"))
        try:
            return action()
        except SessionHostError as error:
            if rejected and error.operation == operation:
                self._agent._consume_capability_rejection(operation)
            raise


class _ScenarioSessionHost:
    """Route session requests to the scripted agent selected by an AgL value."""

    def __init__(self, agents: dict[str, ScriptedAgent]) -> None:
        self._services = {name: agent.session_service() for name, agent in agents.items()}
        self._handles: dict[str, _ScriptedSessionService] = {}
        self._snapshots: dict[str, tuple[Any, str]] = {}
        self._default_handle: str | None = None

    def open(self, agent: Any, transport: str, *, name: str = "") -> str:
        service = self._service_for(agent)
        try:
            handle = service.open(agent, transport.lower(), name=name)
        except SessionHostError as error:
            self._raise_host_error(error)
        self._handles[handle] = service
        self._snapshots[handle] = (agent, transport)
        return handle

    def open_ephemeral(self, agent: Any, transport: str, *, one_shot: bool = False) -> str:
        service = self._service_for(agent)
        try:
            handle = service.open_ephemeral(agent, transport.lower(), one_shot=one_shot)
        except SessionHostError as error:
            self._raise_host_error(error)
        self._handles[handle] = service
        self._snapshots[handle] = (agent, transport)
        return handle

    def with_ephemeral(
        self,
        agent: Any,
        transport: str,
        action: Callable[[str], Any],
        *,
        one_shot: bool = False,
    ) -> Any:
        service = self._service_for(agent)

        def register(handle: str) -> Any:
            self._handles[handle] = service
            self._snapshots[handle] = (agent, transport)
            return action(handle)

        def retire(handle: str) -> None:
            del self._handles[handle]
            del self._snapshots[handle]

        try:
            return service.with_ephemeral(
                agent, transport.lower(), register, on_closed=retire, one_shot=one_shot
            )
        except SessionHostError as error:
            self._raise_host_error(error)

    def default(self, agent: Any, transport: str, *, name: str = "") -> str:
        if self._default_handle is not None:
            return self._default_handle
        service = self._service_for(agent)
        try:
            handle = service.default(agent, transport.lower(), name=name)
        except SessionHostError as error:
            self._raise_host_error(error)
        self._handles[handle] = service
        self._snapshots[handle] = (agent, transport)
        self._default_handle = handle
        return handle

    def ask(self, handle: str, prompt: str) -> str:
        from agm.agent.session import SessionAskRequest
        from agm.agl.runtime.sessions import SessionAskError as AglSessionAskError

        try:
            return (
                self._service_for_handle(handle, "ask")
                .ask(handle, SessionAskRequest(prompt))
                .content
            )
        except SessionHostError as error:
            self._raise_host_error(error)
        except Exception as error:
            if hasattr(error, "cause"):
                raise AglSessionAskError(
                    cause=error.cause,
                    exit_code=error.exit_code,
                    stderr_tail=error.stderr_tail,
                    elapsed=error.elapsed,
                    call_info=None,
                ) from error
            raise

    def ask_request(self, handle: str, request: Any) -> Any:
        from agm.agl.runtime.request import AgentResponse

        content = self.ask(handle, request.prompt)
        service = self._service_for_handle(handle, "ask")
        schema = None if request.output_contract is None else request.output_contract.json_schema
        service._agent.prompt_events[-1] = (request.prompt, schema)
        return AgentResponse(content)

    def compact(self, handle: str, instructions: str = "") -> None:
        self._operation(handle, "compact", instructions)

    def reset(self, handle: str) -> None:
        self._operation(handle, "reset")

    def fork(self, handle: str) -> str:
        service = self._service_for_handle(handle, "fork")
        try:
            forked = service.fork(handle)
        except SessionHostError as error:
            self._raise_host_error(error)
        self._handles[forked] = service
        self._snapshots[forked] = self._snapshots[handle]
        return forked

    def set_name(self, handle: str, name: str) -> None:
        self._operation(handle, "set_name", name)

    def stats(self, handle: str) -> Any:
        try:
            return self._service_for_handle(handle, "stats").stats(handle)
        except SessionHostError as error:
            self._raise_host_error(error)

    def snapshot(self, handle: str) -> Any:
        from agm.agl.runtime.sessions import SessionHostError as AglSessionHostError
        from agm.agl.runtime.sessions import SessionSnapshot

        try:
            agent, transport = self._snapshots[handle]
        except KeyError:
            raise AglSessionHostError("unknown session", "snapshot") from None
        return SessionSnapshot(agent, transport)

    def close(self, handle: str) -> None:
        service = self._service_for_handle(handle, "close")
        try:
            service.close(handle)
        except SessionHostError as error:
            self._raise_host_error(error)
        if handle not in service._sessions:
            del self._handles[handle]
            del self._snapshots[handle]

    def close_all(self) -> None:
        failures: list[Exception] = []
        for service in self._services.values():
            try:
                service.close_all()
            except ExceptionGroup as error:
                failures.extend(error.exceptions)
        if failures:
            raise ExceptionGroup("failed to close one or more scripted sessions", failures)

    def _operation(self, handle: str, name: str, arg: str = "") -> None:
        service = self._service_for_handle(handle, name.replace("_", "-"))
        try:
            if name == "compact":
                service.compact(handle, arg)
            elif name == "set_name":
                service.set_name(handle, arg)
            else:
                service.reset(handle)
        except SessionHostError as error:
            self._raise_host_error(error)

    def _service_for(self, agent: Any) -> _ScriptedSessionService:
        from agm.agl.runtime.sessions import SessionHostError as AglSessionHostError

        try:
            return self._services[_scripted_agent_name(agent)]
        except KeyError:
            raise AglSessionHostError("unknown scripted agent", "open") from None

    def _service_for_handle(self, handle: str, operation: str) -> _ScriptedSessionService:
        try:
            return self._handles[handle]
        except KeyError:
            from agm.agl.runtime.sessions import SessionHostError as AglSessionHostError

            raise AglSessionHostError("unknown session", operation) from None

    @staticmethod
    def _raise_host_error(error: SessionHostError) -> None:
        from agm.agl.runtime.sessions import SessionHostError as AglSessionHostError

        raise AglSessionHostError(error.message, error.operation) from error


class _ScriptedSessionBackend:
    """In-memory backend constrained to one production transport's surface."""

    def __init__(
        self,
        agent: ScriptedAgent,
        session: _ScriptedSession,
        native_capabilities: frozenset[SessionOperation],
        *,
        supports_name: bool,
    ) -> None:
        self._agent = agent
        self._session = session
        self._native_capabilities = native_capabilities
        self._supports_name = supports_name
        session.backend = self

    @property
    def capabilities(self) -> SessionCapabilities:
        return SessionCapabilities(
            frozenset(
                operation
                for operation in self._native_capabilities
                if self._agent._supports_operation(operation.value)
            )
        )

    def supports(self, operation: str) -> bool:
        return self.capabilities.supports(SessionOperation(operation))

    def open(self, request: Any) -> None:
        if request.name and not self._supports_name:
            self._agent.sessions.remove(self._session)
            raise SessionHostError("scripted session does not support names", "set-name")
        self._session.transport = request.transport
        self._session.one_shot = request.one_shot
        self._session.opened = True

    def ask(self, request: Any) -> Any:
        from agm.agent.session import SessionAskError, SessionAskResponse
        from agm.agent.transport import AgentCallInfo

        self._session.prompts.append(request.prompt)
        self._agent.prompt_events.append((request.prompt, None))
        outcome = self._agent._next_session_ask_outcome()
        if isinstance(outcome, dict):
            elapsed = float(outcome.get("elapsed", 0.0))
            exit_code = outcome.get("exit_code")
            if not isinstance(exit_code, int | None):
                raise ValueError("session ask outcome exit_code must be an integer or null")
            raise SessionAskError(
                cause=outcome.get("cause", "timeout"),
                exit_code=exit_code,
                stderr_tail=str(outcome.get("stderr_tail", "")),
                elapsed=elapsed,
                call_info=AgentCallInfo(
                    argv=[], prompt_via_stdin=False, elapsed=elapsed, exit_code=exit_code
                ),
            )
        if outcome != "success":
            raise ValueError("session ask outcome must be 'success' or a transport failure")
        return SessionAskResponse(content=self._agent._next_response())

    def compact(self, instructions: str) -> None:
        self._apply_operation("compact", instructions)

    def reset(self) -> None:
        self._apply_operation("reset", None)

    def fork(self) -> Any:
        self._apply_operation("fork", None)
        return self._agent._fork_session(self._session)

    def set_name(self, name: str) -> None:
        self._apply_operation("set-name", name)

    def stats(self) -> Any:
        from agm.agent.session import SessionStats

        outcome = self._apply_operation("stats", None)
        if not isinstance(outcome, dict):
            return SessionStats(0, 0, Decimal("0"), Decimal("0"))
        return SessionStats(
            input_tokens=int(outcome.get("input_tokens", 0)),
            output_tokens=int(outcome.get("output_tokens", 0)),
            cost=Decimal(str(outcome.get("cost", "0"))),
            context_percent=Decimal(str(outcome.get("context_percent", "0"))),
        )

    def close(self) -> None:
        self._session.closed = True

    def _apply_operation(self, operation: str, arg: str | None) -> Any:
        outcome = self._agent._next_operation(operation)
        outcome_name = _outcome_name(outcome)
        self._session.operations.append((operation, arg, outcome_name))
        if outcome_name == "unsupported":
            from agm.agent.session import SessionHostError

            raise SessionHostError(f"scripted session does not support {operation}", operation)
        return outcome


class _ScriptedPiRpcSessionBackend(_ScriptedSessionBackend):
    """Native Pi RPC mock used by lifecycle scenarios without a real agent."""


def _scripted_agent_name(agent: Any) -> str:
    """Map an AgL agent value onto one scenario's scripted service."""
    import shlex

    command = agent.fields.get("command")
    if command is not None:
        try:
            return shlex.split(command.value)[0]
        except (TypeError, ValueError):
            return command.value
    provider = agent.fields.get("provider")
    return provider.value if provider is not None else "ask"


def _agent_from_spec(name: str, spec: Any) -> ScriptedAgent:
    if isinstance(spec, list):
        return ScriptedAgent(name=name, responses=[str(r) for r in spec])
    session = spec.get("session", {})
    capability_spec = session.get("capabilities")
    capabilities = (
        _SESSION_CAPABILITY_NAMES
        if capability_spec is None
        else frozenset(str(operation) for operation in capability_spec)
    )
    unknown_capabilities = capabilities - _SESSION_CAPABILITY_NAMES
    if unknown_capabilities:
        raise ValueError(f"unknown session capabilities: {sorted(unknown_capabilities)}")
    operations = {
        str(operation): list(outcomes)
        for operation, outcomes in session.get("operations", {}).items()
    }
    ask_outcomes = session.get("ask")
    if ask_outcomes is not None and not isinstance(ask_outcomes, list):
        raise ValueError("session ask outcomes must be a list")
    if capability_spec is not None:
        for operation, outcomes in operations.items():
            if any(_outcome_name(outcome) == "unsupported" for outcome in outcomes):
                raise ValueError(
                    "unsupported outcomes must be expressed through session capabilities"
                )
    return ScriptedAgent(
        name=name,
        responses=[str(r) for r in spec["responses"]],
        repeat_last=bool(spec.get("repeat_last", False)),
        session_capabilities=capabilities,
        session_operations=operations,
        session_ask_outcomes=None if ask_outcomes is None else list(ask_outcomes),
    )


def _run_prepared_entry(runtime: Any, prepared: Any, *, param_values: dict[str, Any]) -> Any:
    """Run the sole selected file-style entry through the public pipeline seams."""
    discovery = runtime.discover_params(prepared)
    if discovery.checked is None:
        return runtime.run_prepared(prepared, param_values=param_values)
    preflight = runtime.preflight_params(
        prepared,
        param_values=param_values,
        compiled=discovery.compiled,
    )
    if not preflight.result.ok:
        return preflight.result
    entry_programs = [item for item in discovery.programs if item.module.is_entry]
    assert len(entry_programs) == 1
    assert preflight.executable is not None
    return runtime.run_prepared(
        prepared,
        param_values=param_values,
        compiled=discovery.compiled,
        executable=preflight.executable,
        program_symbol=preflight.executable.program_symbols[entry_programs[0].node_id],
    )


def _run_source_entry(
    runtime: Any,
    source: str,
    *,
    roots: Any | None = None,
    default_stdlib: bool = True,
) -> Any:
    """Prepare and invoke a sole explicit entry for focused file-style tests."""
    from agm.agl import PipelineDriver

    prepared = PipelineDriver.prepare_program(source, roots=roots, default_stdlib=default_stdlib)
    return _run_prepared_entry(runtime, prepared, param_values={})


def _run_program(
    source: str, scenario: dict[str, Any], program: Path
) -> tuple[Any, dict[str, ScriptedAgent], FakeShell]:
    from agm.agl import PipelineDriver

    agents = {
        name: _agent_from_spec(name, spec) for name, spec in scenario.get("agents", {}).items()
    }
    shell = FakeShell(scenario.get("shell", []))
    runtime_options: dict[str, Any] = {}
    runtime_config = scenario.get("runtime", {})
    if runtime_config.get("enforce_command_sessions") is True:
        for agent in agents.values():
            agent.require_session_id = True
    if "default_call_depth_limit" in runtime_config:
        runtime_options["default_call_depth_limit"] = runtime_config["default_call_depth_limit"]
    if "default_strict_json" in runtime_config:
        runtime_options["default_strict_json"] = runtime_config["default_strict_json"]

    def dispatch_agent(request: Any) -> str:
        return agents[_scripted_agent_name(request.agent)](request)

    if agents:
        runtime_options["agent_dispatcher"] = dispatch_agent
    if agents:
        runtime_options["session_host"] = _ScenarioSessionHost(agents)
    runtime = PipelineDriver(**runtime_options)
    module_roots = scenario.get("module_roots", [])
    default_stdlib = not scenario.get("no_stdlib", False)
    entry_path: Path | None = None
    roots: Any | None = None
    if module_roots:
        from agm.agl.modules.roots import RootSet

        roots = RootSet(
            roots=frozenset(
                {
                    *((AGL_DIR / str(root)).resolve() for root in module_roots),
                    REPO_STDLIB_ROOT,
                }
            )
        )
    elif program.is_relative_to(EXTERNS_PROGRAMS_DIR) or program.is_relative_to(
        RESOURCE_PROGRAMS_DIR
    ):
        from agm.agl.modules.roots import RootSet

        entry_path = program
        roots = RootSet(roots=frozenset({program.parent.resolve(), REPO_STDLIB_ROOT}))
    # `inline_entry` sources carry no `program def`: they run through the same
    # synthetic-entry transform as `agm exec -c`.
    prepare = (
        prepare_inline_command if scenario.get("inline_entry") else PipelineDriver.prepare_program
    )
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        prepared = prepare(
            source, entry_path=entry_path, roots=roots, default_stdlib=default_stdlib
        )

        result = _run_prepared_entry(runtime, prepared, param_values=scenario.get("params", {}))
    return result, agents, shell


def _assert_host_error(result: Any, agents: dict[str, ScriptedAgent], spec: dict[str, Any]) -> None:
    assert not result.ok, "expected the run to fail param validation"
    assert result.error is None, "param validation failure is not an AgL exception"
    messages = " | ".join(d.message for d in result.diagnostics)
    for needle in spec.get("message_contains", []):
        assert needle.lower() in messages.lower(), (
            f"no diagnostic mentions {needle!r}; diagnostics: {messages!r}"
        )
    for agent in agents.values():
        assert agent.prompts == [], (
            f"agent {agent.name!r} was called despite param validation failing"
        )


def _assert_outcome(result: Any, expect: dict[str, Any]) -> None:
    diags = " | ".join(d.message for d in result.diagnostics)
    assert list(result.diagnostics) == [], f"unexpected static diagnostics: {diags}"
    if "raises" in expect:
        spec = expect["raises"]
        assert result.error is not None, f"expected uncaught {spec['type']}, got none"
        assert result.error.type_name == spec["type"]
        for key, value in spec.get("fields", {}).items():
            actual = result.error.fields[key]
            assert actual == value, f"{spec['type']}.{key}: expected {value!r}, got {actual!r}"
        message = str(result.error.fields.get("message", ""))
        for needle in spec.get("message_contains", []):
            assert needle in message, f"{needle!r} not in message {message!r}"
        assert not result.ok
    else:
        if result.error is not None:
            raise AssertionError(
                f"unexpected uncaught {result.error.type_name}: {result.error.fields!r}"
            )
        assert result.ok


def _assert_output(out: str, expect: dict[str, Any]) -> None:
    if "stdout" in expect:
        assert out == expect["stdout"]
    for needle in expect.get("stdout_contains", []):
        assert needle in out, f"{needle!r} not in stdout {out!r}"
    for needle in expect.get("stdout_not_contains", []):
        assert needle not in out, f"{needle!r} unexpectedly in stdout {out!r}"


def _schema_contains(schema: Any, needle: str) -> bool:
    if isinstance(schema, dict):
        return any(
            str(key) == needle or _schema_contains(value, needle) for key, value in schema.items()
        )
    if isinstance(schema, list):
        return any(_schema_contains(item, needle) for item in schema)
    return schema == needle


def _assert_schema_paths(schema: Any, assertions: list[dict[str, Any]]) -> None:
    """Assert exact JSON Schema values at dictionary-key paths."""
    for assertion in assertions:
        actual = schema
        path = assertion["path"]
        for key in path:
            assert isinstance(actual, dict), f"{path!r} is not a dictionary path in {schema!r}"
            assert key in actual, f"{path!r} is absent from schema {schema!r}"
            actual = actual[key]
        assert actual == assertion["equals"], (
            f"schema at {path!r}: expected {assertion['equals']!r}, got {actual!r}"
        )


def test_schema_paths_assert_exact_nested_values() -> None:
    schema = {
        "$ref": "#/$defs/Workflow_Task",
        "$defs": {
            "Workflow_Task": {
                "properties": {"children": {"items": {"$ref": "#/$defs/Workflow_Task"}}}
            }
        },
    }

    _assert_schema_paths(
        schema,
        [
            {"path": ["$ref"], "equals": "#/$defs/Workflow_Task"},
            {
                "path": [
                    "$defs",
                    "Workflow_Task",
                    "properties",
                    "children",
                    "items",
                    "$ref",
                ],
                "equals": "#/$defs/Workflow_Task",
            },
        ],
    )


def _assert_prompt_text(prompt: str, spec: dict[str, Any]) -> None:
    if "equals" in spec:
        assert prompt == spec["equals"]
    if "starts_with" in spec:
        assert prompt.startswith(spec["starts_with"])
    for needle in spec.get("contains", []):
        assert needle in prompt, f"{needle!r} not in prompt {prompt!r}"
    for needle in spec.get("not_contains", []):
        assert needle not in prompt, f"{needle!r} unexpectedly in prompt {prompt!r}"


def _assert_calls(agents: dict[str, ScriptedAgent], expect: dict[str, Any]) -> None:
    for name, agent in agents.items():
        assert not agent.overflowed, f"agent {name!r} was called more times than scripted"
    expected_calls = expect.get("calls", {})
    assert set(expected_calls) == set(agents), (
        "every scripted agent must have an exact call-count assertion; "
        f"expected entries for {sorted(agents)}, got {sorted(expected_calls)}"
    )
    for name, agent in agents.items():
        count = expected_calls[name]
        actual = len(agent.prompt_events) if "session_prompts" not in expect else len(agent.prompts)
        assert actual == count, f"agent {name!r}: expected {count} calls, got {actual}"
    for spec in expect.get("prompts", []):
        agent_spec = spec["agent"]
        if isinstance(agent_spec, dict):
            name = agent_spec.get("command", "ask")
        else:
            name = agent_spec
        assert isinstance(name, str)
        prompt_events = agents[name].prompt_events
        prompts = [prompt for prompt, _schema in prompt_events]
        call = spec["call"]
        assert call < len(prompts), (
            f"agent {spec['agent']!r} made only {len(prompts)} calls, no call {call}"
        )
        _assert_prompt_text(prompts[call], spec)
        schema = prompt_events[call][1]
        for needle in spec.get("schema_contains", []):
            assert _schema_contains(schema, needle), f"{needle!r} not in schema {schema!r}"
        _assert_schema_paths(schema, spec.get("schema_paths", []))


def _assert_sessions(agents: dict[str, ScriptedAgent], expect: dict[str, Any]) -> None:
    for name, agent in agents.items():
        assert not agent.session_operations_overflowed, (
            f"session operation script for agent {name!r} was used more times than scripted"
        )
        assert not agent.session_ask_outcomes_overflowed, (
            f"session ask script for agent {name!r} was used more times than scripted"
        )
        assert not agent.session_ask_outcomes, (
            f"session ask script for agent {name!r} was not fully consumed"
        )
        unconsumed = {
            operation: len(outcomes)
            for operation, outcomes in agent.session_operations.items()
            if outcomes
        }
        assert not unconsumed, (
            f"session operation script for agent {name!r} was not fully consumed: {unconsumed}"
        )

    def session_for(spec: dict[str, Any]) -> _ScriptedSession:
        agent_name = spec["agent"]
        tag = spec.get("session", spec.get("tag"))
        assert isinstance(agent_name, str)
        assert isinstance(tag, str)
        matches = [session for session in agents[agent_name].sessions if session.tag == tag]
        assert len(matches) == 1, f"no unique session {tag!r} for agent {agent_name!r}"
        return matches[0]

    if "sessions" in expect:
        expected_sessions = {
            (spec["agent"], spec.get("session", spec.get("tag"))) for spec in expect["sessions"]
        }
        observed_sessions = {
            (agent_name, session.tag)
            for agent_name, agent in agents.items()
            for session in agent.sessions
        }
        assert expected_sessions == observed_sessions, (
            f"session set: expected {sorted(expected_sessions)}, got {sorted(observed_sessions)}"
        )

    for spec in expect.get("sessions", []):
        session = session_for(spec)
        for key in ("opened", "closed", "parent", "transport", "one_shot"):
            if key in spec:
                assert getattr(session, key) == spec[key], (
                    f"session {session.tag!r} {key}: expected {spec[key]!r}, "
                    f"got {getattr(session, key)!r}"
                )

    prompt_specs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for spec in expect.get("session_prompts", []):
        agent_name = spec["agent"]
        tag = spec.get("session", spec.get("tag"))
        assert isinstance(agent_name, str)
        assert isinstance(tag, str)
        prompt_specs.setdefault((agent_name, tag), []).append(spec)

    if "sessions" in expect or "session_prompts" in expect:
        for agent_name, agent in agents.items():
            for session in agent.sessions:
                specs = prompt_specs.get((agent_name, session.tag), [])
                assert len(specs) == len(session.prompts), (
                    f"session {session.tag!r} prompt count: expected {len(specs)}, "
                    f"got {len(session.prompts)}"
                )
                calls = sorted(spec["call"] for spec in specs)
                assert calls == list(range(len(session.prompts))), (
                    f"session {session.tag!r} prompt calls must cover each prompt exactly once"
                )

    for spec in expect.get("session_prompts", []):
        session = session_for(spec)
        call = spec["call"]
        assert isinstance(call, int)
        if "follow_up" in spec:
            assert call > 0, "a follow-up must be later than the first session prompt"
            follow_up = spec["follow_up"]
            assert isinstance(follow_up, dict)
            _assert_prompt_text(session.prompts[call], follow_up)
        else:
            _assert_prompt_text(session.prompts[call], spec)

    if "session_operations" not in expect:
        return

    operation_specs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for spec in expect["session_operations"]:
        agent_name = spec["agent"]
        tag = spec.get("session", spec.get("tag"))
        operation = spec["operation"]
        call = spec.get("call", 0)
        assert isinstance(agent_name, str)
        assert isinstance(tag, str)
        assert isinstance(operation, str)
        assert isinstance(call, int)
        operation_specs.setdefault((agent_name, tag), []).append(spec)

    for agent_name, agent in agents.items():
        for session in agent.sessions:
            specs = operation_specs.get((agent_name, session.tag), [])
            assert len(specs) == len(session.operations), (
                f"session {session.tag!r} operation count: expected {len(specs)}, "
                f"got {len(session.operations)}"
            )
            operation_names = {operation for operation, _, _ in session.operations}
            operation_names.update(spec["operation"] for spec in specs)
            for operation in operation_names:
                calls = [call for call in session.operations if call[0] == operation]
                expected_calls = sorted(
                    spec.get("call", 0) for spec in specs if spec["operation"] == operation
                )
                assert expected_calls == list(range(len(calls))), (
                    f"session {session.tag!r} {operation!r} operation calls must cover each "
                    "operation exactly once"
                )

    for spec in expect["session_operations"]:
        session = session_for(spec)
        operation = spec["operation"]
        call = spec.get("call", 0)
        assert isinstance(call, int)
        calls = [recorded for recorded in session.operations if recorded[0] == operation]
        _, arg, outcome = calls[call]
        if "arg" in spec:
            assert arg == spec["arg"]
        if "outcome" in spec:
            assert outcome == spec["outcome"]


def test_scripted_sessions_observe_identity_operations_and_lifecycle() -> None:
    from agm.agent.session import SessionAskRequest, SessionHostError

    agent = _agent_from_spec(
        "writer",
        {
            "responses": ["answer"],
            "repeat_last": True,
            "session": {
                "operations": {
                    "compact": ["success"],
                    "reset": ["success"],
                    "fork": ["success"],
                    "stats": [
                        {
                            "input_tokens": 3,
                            "output_tokens": 5,
                            "cost": "0.12",
                            "context_percent": "4.5",
                        }
                    ],
                    "set-name": ["unsupported"],
                },
            },
        },
    )
    service = agent.session_service()

    parent = service.open(object(), "scripted", name="primary")
    assert service.ask(parent, SessionAskRequest("first")).content == "answer"
    service.compact(parent, "summarize")
    service.reset(parent)
    child = service.fork(parent)
    assert service.ask(parent, SessionAskRequest("second")).content == "answer"
    assert service.ask(child, SessionAskRequest("child prompt")).content == "answer"
    stats = service.stats(parent)
    with pytest.raises(SessionHostError) as raised:
        service.set_name(parent, "renamed")
    service.close(parent)
    service.close(child)

    assert stats.input_tokens == 3
    assert raised.value.operation == "set-name"
    _assert_sessions(
        {"writer": agent},
        {
            "sessions": [
                {"agent": "writer", "tag": "session-1", "opened": True, "closed": True},
                {
                    "agent": "writer",
                    "tag": "session-2",
                    "parent": "session-1",
                    "closed": True,
                },
            ],
            "session_prompts": [
                {"agent": "writer", "session": "session-1", "call": 0, "equals": "first"},
                {
                    "agent": "writer",
                    "session": "session-1",
                    "call": 1,
                    "follow_up": {"equals": "second"},
                },
                {"agent": "writer", "session": "session-2", "call": 0, "equals": "child prompt"},
            ],
            "session_operations": [
                {
                    "agent": "writer",
                    "session": "session-1",
                    "operation": "compact",
                    "arg": "summarize",
                    "outcome": "success",
                },
                {
                    "agent": "writer",
                    "session": "session-1",
                    "operation": "reset",
                    "outcome": "success",
                },
                {
                    "agent": "writer",
                    "session": "session-1",
                    "operation": "fork",
                    "outcome": "success",
                },
                {
                    "agent": "writer",
                    "session": "session-1",
                    "operation": "stats",
                    "outcome": "success",
                },
                {
                    "agent": "writer",
                    "session": "session-1",
                    "operation": "set-name",
                    "arg": "renamed",
                    "outcome": "unsupported",
                },
            ],
        },
    )


def test_scripted_sessions_record_capability_rejections_without_outcomes() -> None:
    from agm.agent.session import SessionHostError

    agent = _agent_from_spec(
        "writer",
        {
            "responses": [],
            "session": {"capabilities": ["compact"]},
        },
    )
    service = agent.session_service()
    handle = service.open(object(), "scripted")

    with pytest.raises(SessionHostError) as raised:
        service.set_name(handle, "renamed")

    assert raised.value.operation == "set-name"
    _assert_sessions(
        {"writer": agent},
        {
            "session_operations": [
                {
                    "agent": "writer",
                    "session": "session-1",
                    "operation": "set-name",
                    "arg": "renamed",
                    "outcome": "unsupported",
                }
            ]
        },
    )


def test_scripted_sessions_reconcile_ordered_operation_outcomes() -> None:
    agent = _agent_from_spec(
        "writer",
        {
            "responses": [],
            "session": {"operations": {"compact": ["success", "success"]}},
        },
    )
    service = agent.session_service()
    handle = service.open(object(), "scripted")

    service.compact(handle, "first")
    with pytest.raises(AssertionError, match="not fully consumed"):
        _assert_sessions({"writer": agent}, {})

    service.compact(handle, "second")
    _assert_sessions(
        {"writer": agent},
        {
            "session_operations": [
                {
                    "agent": "writer",
                    "session": "session-1",
                    "operation": "compact",
                    "call": 0,
                    "arg": "first",
                    "outcome": "success",
                },
                {
                    "agent": "writer",
                    "session": "session-1",
                    "operation": "compact",
                    "call": 1,
                    "arg": "second",
                    "outcome": "success",
                },
            ]
        },
    )


def test_session_expectations_reject_unlisted_empty_sessions() -> None:
    agent = _agent_from_spec("writer", {"responses": []})
    service = agent.session_service()
    service.open(object(), "scripted")
    service.open(object(), "scripted")

    with pytest.raises(AssertionError, match="session set"):
        _assert_sessions(
            {"writer": agent},
            {
                "sessions": [
                    {"agent": "writer", "tag": "session-1", "opened": True, "closed": False}
                ]
            },
        )


def test_scenario_host_reports_unknown_handles_by_attempted_operation() -> None:
    from agm.agl.runtime.sessions import SessionHostError as AglSessionHostError

    host = _ScenarioSessionHost({})
    operations: tuple[tuple[str, Callable[[], object]], ...] = (
        ("ask", lambda: host.ask("unknown", "prompt")),
        ("compact", lambda: host.compact("unknown")),
        ("reset", lambda: host.reset("unknown")),
        ("fork", lambda: host.fork("unknown")),
        ("set-name", lambda: host.set_name("unknown", "name")),
        ("stats", lambda: host.stats("unknown")),
        ("snapshot", lambda: host.snapshot("unknown")),
        ("close", lambda: host.close("unknown")),
    )

    for operation, attempt in operations:
        with pytest.raises(AglSessionHostError) as raised:
            attempt()
        assert raised.value.operation == operation


def test_scripted_sessions_reject_extra_capability_rejection_operation() -> None:
    from agm.agent.session import SessionHostError

    agent = _agent_from_spec(
        "writer",
        {
            "responses": [],
            "session": {"capabilities": ["compact"]},
        },
    )
    service = agent.session_service()
    handle = service.open(object(), "scripted")
    service.compact(handle, "summarize")
    _assert_sessions({"writer": agent}, {})
    with pytest.raises(SessionHostError):
        service.set_name(handle, "renamed")

    with pytest.raises(AssertionError, match="operation count"):
        _assert_sessions(
            {"writer": agent},
            {
                "session_operations": [
                    {
                        "agent": "writer",
                        "session": "session-1",
                        "operation": "compact",
                        "arg": "summarize",
                        "outcome": "success",
                    }
                ]
            },
        )


def test_scripted_sessions_reject_unsupported_reset_outcomes() -> None:
    with pytest.raises(ValueError, match="reset"):
        _agent_from_spec(
            "writer",
            {"responses": [], "session": {"operations": {"reset": ["unsupported"]}}},
        )


def test_scripted_sessions_detect_an_extra_retry_prompt() -> None:
    from agm.agent.session import SessionAskRequest

    agent = _agent_from_spec("writer", {"responses": ["answer"], "repeat_last": True})
    service = agent.session_service()
    handle = service.open(object(), "scripted")
    service.ask(handle, SessionAskRequest("first"))
    service.ask(handle, SessionAskRequest("retry"))

    with pytest.raises(AssertionError, match="prompt count"):
        _assert_sessions(
            {"writer": agent},
            {
                "session_prompts": [
                    {"agent": "writer", "session": "session-1", "call": 0, "equals": "first"}
                ]
            },
        )


def test_scripted_session_ask_closes_an_ephemeral_session() -> None:
    from agm.agent.session import SessionAskRequest

    agent = _agent_from_spec("writer", ["answer"])
    response = agent.session_service().ask_ephemeral(
        object(), "scripted", SessionAskRequest("one-off")
    )

    assert response.content == "answer"
    _assert_sessions(
        {"writer": agent},
        {
            "sessions": [{"agent": "writer", "tag": "session-1", "opened": True, "closed": True}],
            "session_prompts": [
                {"agent": "writer", "session": "session-1", "call": 0, "equals": "one-off"}
            ],
        },
    )


def _scenario_params() -> list[Any]:
    params: list[Any] = []
    for program in sorted(PROGRAMS_DIR.rglob("*.agl")):
        sidecar = program.with_name(program.stem + ".scenarios.json")
        rel = program.relative_to(PROGRAMS_DIR).with_suffix("")
        for scenario in _load_json(sidecar)["scenarios"]:
            params.append(pytest.param(program, scenario, id=f"{rel}::{scenario['name']}"))
    return params


def _rejection_params() -> list[Any]:
    return [
        pytest.param(program, id=str(program.relative_to(REJECTIONS_DIR).with_suffix("")))
        for program in sorted(REJECTIONS_DIR.rglob("*.agl"))
    ]


@pytest.mark.parametrize(("program", "scenario"), _scenario_params())
def test_program_scenario(
    program: Path, scenario: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    result, agents, shell = _run_program(program.read_text(encoding="utf-8"), scenario, program)
    out = capsys.readouterr().out
    expect = scenario["expect"]
    if "host_error" in expect:
        _assert_host_error(result, agents, expect["host_error"])
        shell.assert_complete()
        return
    _assert_outcome(result, expect)
    _assert_output(out, expect)
    _assert_calls(agents, expect)
    _assert_sessions(agents, expect)
    shell.assert_complete()


@pytest.mark.parametrize("program", _rejection_params())
def test_static_rejection(program: Path) -> None:
    from agm.agl import PipelineDriver

    expect = _load_json(program.with_name(program.stem + ".expect.json"))["diagnostic"]
    result = PipelineDriver().run(program.read_text(encoding="utf-8"), param_values={})
    assert not result.ok, "expected the program to be rejected statically"
    assert result.error is None, "static rejection must happen before execution"
    diagnostics = list(result.diagnostics)
    assert diagnostics, "expected at least one diagnostic"
    if "line" in expect:
        lines = [d.line for d in diagnostics]
        assert expect["line"] in lines, f"no diagnostic on line {expect['line']}; lines: {lines}"
    joined = " | ".join(d.message for d in diagnostics)
    for needle in expect.get("message_contains", []):
        assert needle.lower() in joined.lower(), (
            f"no diagnostic mentions {needle!r}; diagnostics: {joined!r}"
        )


def test_pipeline_run_invokes_the_single_entry_program(capsys: pytest.CaptureFixture[str]) -> None:
    from agm.agl import PipelineDriver

    result = PipelineDriver().run('program def main() -> unit = print "hello"')

    assert result.ok
    assert capsys.readouterr().out == "hello\n"


def test_pipeline_check_only_rejects_ambiguous_default_program() -> None:
    from agm.agl import PipelineDriver

    result = PipelineDriver().run(
        "program def first() -> unit = ()\nprogram def second() -> unit = ()\n",
        check_only=True,
    )

    assert not result.ok
    assert result.error is None
    assert [diagnostic.message for diagnostic in result.diagnostics] == [
        "multiple programs declared; select one"
    ]


def test_qualified_std_core_print_still_works(capsys: pytest.CaptureFixture[str]) -> None:
    """A fully qualified ``std/core::print(...)`` call still runs, exactly as
    the bare form does — a built-in call is classified once its callee
    resolves to a ``builtin def``, and ``std/core::print`` reaches the same
    declaration a bare ``print`` does, just by a qualified route."""
    from agm.agl import PipelineDriver

    runtime = PipelineDriver()
    result = _run_source_entry(runtime, 'program def main() -> unit = std/core::print("hi")\n')

    assert list(result.diagnostics) == [], (
        f"unexpected static diagnostics: {' | '.join(d.message for d in result.diagnostics)}"
    )
    assert result.error is None, f"unexpected error: {result.error}"
    assert capsys.readouterr().out == "hi\n"


def _scoped_stdlib_root(tmp_path: Path) -> Path:
    """Build a throwaway module root with ``std/core.agl`` wrapped in ``scope Std``.

    Never the installed/repo stdlib root — the standard ``module_roots``
    scenario mechanism always adds ``REPO_STDLIB_ROOT`` too, which would
    collide with this substitute ``std/core``, so callers build their own
    ``RootSet`` from the returned root instead.
    """
    core_source = (
        (REPO_STDLIB_ROOT / "std" / "core.agl")
        .read_text(encoding="utf-8")
        .replace("import std/config\n\n", "")
        .replace(_SESSION_STATIC_DECLARATIONS, "")
        .replace("std/config::default-agent", 'AgentClaude("sonnet", "medium")')
    )
    scoped_stdlib_root = tmp_path / "scoped_stdlib"
    (scoped_stdlib_root / "std").mkdir(parents=True)
    (scoped_stdlib_root / "std" / "core.agl").write_text(
        f"scope Std\n{core_source}end Std\n", encoding="utf-8"
    )
    return scoped_stdlib_root


def test_scoped_stdlib_arrangement_runs_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A whole stdlib module wrapped in a named scope region works end to end.

    Wraps the runtime-lowerable subset of ``stdlib/std/core.agl`` in a
    ``scope Std ... end Std`` region under a throwaway module root (never the installed/repo
    stdlib — the standard ``module_roots`` scenario mechanism always adds
    ``REPO_STDLIB_ROOT`` too, which would collide with this substitute
    ``std/core``, so this test builds its own ``RootSet`` instead), then runs
    a program against it through parsing, scope resolution, typechecking,
    lowering, and evaluation. Exercises a scoped ``builtin def`` (``exec``)
    dispatching to a scoped ``builtin record`` (``ExecResult``) at a real host
    boundary, and a fully scoped exception hierarchy (``Exception``/
    ``RangeError`` both declared inside the region) raised and left uncaught.
    """
    from agm.agl import PipelineDriver
    from agm.agl.modules.roots import RootSet

    scoped_stdlib_root = _scoped_stdlib_root(tmp_path)

    program = (
        "program def main() -> unit =\n"
        '  let r = Std::exec("echo hi")\n  Std::print(r.stdout)\n'
        '  raise Std::RangeError(message = "boom")\n'
    )

    shell = FakeShell([{"command": "echo hi", "stdout": "hi\n"}])
    runtime = PipelineDriver()
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        result = _run_source_entry(
            runtime, program, roots=RootSet(roots=frozenset({scoped_stdlib_root}))
        )
    shell.assert_complete()

    assert list(result.diagnostics) == [], (
        f"unexpected static diagnostics: {' | '.join(d.message for d in result.diagnostics)}"
    )
    assert capsys.readouterr().out == "hi\n"
    assert result.error is not None, "expected the uncaught scoped RangeError"
    assert result.error.type_name == "Std::RangeError"
    assert result.error.fields["message"] == "boom"


def test_scoped_stdlib_arrangement_structured_exec_result_is_the_scoped_nominal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A structured ``exec`` result is tagged with its own scoped nominal.

    The result is annotated ``Std::ExecResult`` explicitly, so the static
    target type is the scoped nominal by the annotation rather than by
    ``exec``'s own default. The field access below then only succeeds if the
    host-minted value carries that very same scoped nominal — an exact
    nominal projection rejects any other identity.
    """
    from agm.agl import PipelineDriver
    from agm.agl.modules.roots import RootSet

    scoped_stdlib_root = _scoped_stdlib_root(tmp_path)

    program = (
        "program def main() -> unit =\n"
        '  let r: Std::ExecResult = Std::exec("echo hi")\n  Std::print(r.stdout)\n'
    )

    shell = FakeShell([{"command": "echo hi", "stdout": "hi\n"}])
    runtime = PipelineDriver()
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        result = _run_source_entry(
            runtime, program, roots=RootSet(roots=frozenset({scoped_stdlib_root}))
        )
    shell.assert_complete()

    assert list(result.diagnostics) == [], (
        f"unexpected static diagnostics: {' | '.join(d.message for d in result.diagnostics)}"
    )
    assert result.error is None, f"unexpected error: {result.error}"
    assert capsys.readouterr().out == "hi\n"


def test_scoped_stdlib_arrangement_uncaught_host_raised_exec_error_reports_scoped_spelling(
    tmp_path: Path,
) -> None:
    """An uncaught host-raised exception reports its declared, scoped spelling.

    A failed (non-structured, ``text``-typed) ``Std::exec`` call raises
    ``ExecError`` at the host boundary; left uncaught, it now reports
    ``Std::ExecError`` — the outside-the-region half of the catchability fix
    observable without a qualified ``catch`` (the grammar admits none)."""
    from agm.agl import PipelineDriver
    from agm.agl.modules.roots import RootSet

    scoped_stdlib_root = _scoped_stdlib_root(tmp_path)

    program = (
        'program def main() -> unit =\n  let out: text = Std::exec("false")\n  Std::print(out)\n'
    )

    shell = FakeShell([{"command": "false", "returncode": 1}])
    runtime = PipelineDriver()
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        result = _run_source_entry(
            runtime, program, roots=RootSet(roots=frozenset({scoped_stdlib_root}))
        )
    shell.assert_complete()

    assert list(result.diagnostics) == [], (
        f"unexpected static diagnostics: {' | '.join(d.message for d in result.diagnostics)}"
    )
    assert result.error is not None, "expected the uncaught scoped ExecError"
    assert result.error.type_name == "Std::ExecError"


def test_scoped_stdlib_arrangement_bare_print_is_undefined_but_qualified_works(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the whole standard library wrapped in ``scope Std``, a bare
    ``print`` call has no reachable declaration — only ``Std::print`` does.

    A built-in call is classified only once its callee resolves to a
    ``builtin def`` declaration, the same as any other reference; wrapping
    the whole standard library in a named region takes the bare route away
    from every name it declares, ``print`` included, leaving only the
    region's own qualified path."""
    from agm.agl import PipelineDriver
    from agm.agl.modules.roots import RootSet

    scoped_stdlib_root = _scoped_stdlib_root(tmp_path)
    roots = RootSet(roots=frozenset({scoped_stdlib_root}))
    runtime = PipelineDriver()

    bare_result = _run_source_entry(
        runtime, 'program def main() -> unit = print("hi")\n', roots=roots
    )
    assert not bare_result.ok, "expected the bare 'print' call to be statically rejected"
    assert bare_result.diagnostics, "expected at least one diagnostic"

    qualified_result = _run_source_entry(
        runtime, 'program def main() -> unit = Std::print("hi")\n', roots=roots
    )
    assert list(qualified_result.diagnostics) == [], (
        f"unexpected static diagnostics: "
        f"{' | '.join(d.message for d in qualified_result.diagnostics)}"
    )
    assert qualified_result.error is None, f"unexpected error: {qualified_result.error}"
    assert capsys.readouterr().out == "hi\n"


def test_scoped_builtin_hierarchy_declared_in_the_entry_module_catches_a_host_raise(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A host-raised exception is catchable by its declared, scoped spelling.

    ``catch`` accepts only an unqualified NAME (the grammar admits no
    qualified ``A::B`` there), so the catching code lives inside the same
    ``scope Host`` region that declares the hierarchy, where the bare name
    resolves to the region's own declaration — the same arrangement
    ``scoped_builtin_stdlib_arrangement.agl``'s own ``report``/``RangeError``
    catch already uses. This program declares the whole exec/exception
    surface itself and loads without the standard library (``default_stdlib
    =False``), so its own ``scope Host`` module lowers normally (unlike the
    real ``std/core`` module, which is excluded from per-module function
    lowering). A failed (``text``-typed, non-structured) ``exec`` call
    raises ``ExecError`` at the host boundary; before per-path identity was
    restored the host would have minted a path-free nominal while the type
    kept its declared path, so this bare ``catch`` could never match. Now
    both sides agree, and the program handles it.
    """
    from agm.agl import PipelineDriver

    program = (
        "scope Host\n"
        "builtin\n"
        "exception Exception\n"
        "  *\n"
        "  message: text\n"
        "builtin\n"
        "exception ExecError extends Exception\n"
        "  *\n"
        "  command: text\n"
        "  exit_code: int\n"
        "  stdout: text\n"
        "  stderr: text\n"
        "  timed_out: bool\n"
        "builtin record ExecResult\n"
        "  stdout: text\n"
        "  exit_code: int\n"
        "  stderr: text\n"
        "  timed_out: bool\n"
        "builtin def exec(command: text) -> ExecResult\n"
        "def run(cmd: text) -> text =\n"
        "  try\n"
        "    let out: text = exec(cmd)\n"
        "    out\n"
        "  catch ExecError as e =>\n"
        '    "caught"\n'
        "end Host\n"
        "builtin def print[T](value: T) -> unit\n"
        "program def main() -> unit =\n"
        '  print(Host::run("false"))\n'
    )

    shell = FakeShell([{"command": "false", "returncode": 1}])
    runtime = PipelineDriver()
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell):
        result = _run_source_entry(runtime, program, default_stdlib=False)
    shell.assert_complete()

    assert list(result.diagnostics) == [], (
        f"unexpected static diagnostics: {' | '.join(d.message for d in result.diagnostics)}"
    )
    assert result.error is None, f"expected the raised ExecError to be caught: {result.error}"
    assert capsys.readouterr().out == "caught\n"


def test_program_def_entries_execute_via_agm_exec(tmp_path: Path) -> None:
    """Exec runs a sole program and selects a declaration path among several."""
    from click.testing import CliRunner
    from typer.main import get_command

    import agm.cli as cli

    sole = tmp_path / "sole.agl"
    sole.write_text('program def main() -> unit = print "sole"\n')
    selected = tmp_path / "selected.agl"
    selected.write_text(
        'program def first() -> unit = print "first"\n'
        "scope review\n"
        'program def main() -> unit = print "review"\n'
        "end review\n"
    )

    runner = CliRunner()
    sole_result = runner.invoke(
        get_command(cli.app), ["exec", "--no-log", str(sole)], catch_exceptions=False
    )
    selected_result = runner.invoke(
        get_command(cli.app),
        ["exec", "--no-log", "-p", "review::main", str(selected)],
        catch_exceptions=False,
    )

    assert sole_result.exit_code == 0
    assert sole_result.output == "sole\n"
    assert selected_result.exit_code == 0
    assert selected_result.output == "review\n"
