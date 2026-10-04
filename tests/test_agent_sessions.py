"""Unit tests for host-managed agent sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from uuid import UUID

import pytest

from agm.agent.session import (
    AglSessionHost,
    SessionAskError,
    SessionAskRequest,
    SessionAskResponse,
    SessionBackend,
    SessionHostError,
    SessionOpenRequest,
    SessionOperation,
    SessionOperations,
    SessionService,
    SessionStats,
    create_agl_session_host,
)
from agm.agent.spec import AgentCommand, AgentPi, PermissionMode, SessionTransport
from agm.agent.transport import AgentCallInfo, AgentOutputPhase
from agm.agl.runtime.request import AgentRequest
from agm.agl.runtime.sessions import SessionAskError as AglSessionAskError
from agm.agl.runtime.sessions import SessionHostError as AglSessionHostError
from agm.sandbox.request import SandboxLimits
from tests._agl_helpers import hermetic_get_sandbox_context

ALL_OPERATIONS = frozenset(SessionOperation)


@dataclass
class FakeBackend:
    """In-memory backend used to observe the session service's behavior."""

    supported: frozenset[SessionOperation] = ALL_OPERATIONS
    continues_conversation: bool = True
    response: SessionAskResponse = field(
        default_factory=lambda: SessionAskResponse(content="answer")
    )
    ask_error: Exception | None = None
    fork_result: SessionBackend | None = None
    open_requests: list[SessionOpenRequest] = field(default_factory=list)
    ask_requests: list[SessionAskRequest] = field(default_factory=list)
    compact_requests: list[str] = field(default_factory=list)
    reset_calls: int = 0
    names: list[str] = field(default_factory=list)
    close_calls: int = 0
    close_error: Exception | None = None

    @property
    def operations(self) -> SessionOperations:
        def native[T](operation: SessionOperation, method: T) -> T | None:
            return method if operation in self.supported else None

        return SessionOperations(
            compact=native(SessionOperation.COMPACT, self.compact),
            fork=native(SessionOperation.FORK, self.fork),
            set_name=native(SessionOperation.SET_NAME, self.set_name),
            stats=native(SessionOperation.STATS, self.stats),
        )

    def ask(self, request: SessionAskRequest) -> SessionAskResponse:
        self.ask_requests.append(request)
        if self.ask_error is not None:
            raise self.ask_error
        return self.response

    def compact(self, instructions: str) -> None:
        self.compact_requests.append(instructions)

    def reset(self) -> None:
        self.reset_calls += 1

    def fork(self) -> SessionBackend:
        if self.fork_result is None:
            raise AssertionError("test backend must be given a fork result")
        return self.fork_result

    def set_name(self, name: str) -> None:
        self.names.append(name)

    def stats(self) -> SessionStats:
        return SessionStats(
            input_tokens=3,
            output_tokens=5,
            cost=Decimal("0.12"),
            context_percent=Decimal("4.5"),
        )

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeBackendFactory:
    def __init__(self, supported: frozenset[SessionOperation] = ALL_OPERATIONS) -> None:
        self.supported = supported
        self.backends: list[FakeBackend] = []

    def __call__(self, request: SessionOpenRequest) -> SessionBackend:
        backend = FakeBackend(self.supported, open_requests=[request])
        self.backends.append(backend)
        return backend


def _service(
    factory: FakeBackendFactory | None = None,
) -> tuple[SessionService, FakeBackendFactory]:
    actual_factory = factory or FakeBackendFactory()
    return SessionService(actual_factory), actual_factory


def _assert_error(operation: str, action: object) -> None:
    if not callable(action):
        raise AssertionError("action must be callable")
    with pytest.raises(SessionHostError) as raised:
        action()
    assert raised.value.operation == operation


def test_agl_session_host_adapts_all_lifecycle_operations() -> None:
    parent = FakeBackend()
    child = FakeBackend()
    parent.fork_result = child
    created: list[SessionOpenRequest] = []

    def backend_for(request: SessionOpenRequest) -> SessionBackend:
        created.append(request)
        return parent

    host = AglSessionHost(SessionService(backend_for))
    agent = AgentCommand(command="worker")
    handle = host.open(agent, "Cli", name="primary", env={})
    assert host.ask(handle, "hello") == "answer"
    host.compact(handle, "retain")
    host.reset(handle)
    child_handle = host.fork(handle)
    host.set_name(child_handle, "child")
    stats = host.stats(child_handle)
    snapshot = host.snapshot(child_handle)
    host.close(handle)
    host.close_all()

    assert created[0].transport == SessionTransport.CLI
    assert parent.ask_requests == [SessionAskRequest("hello")]
    assert parent.compact_requests == ["retain"]
    assert parent.reset_calls == 1
    assert child.names == ["child"]
    assert stats.input_tokens == 3
    assert snapshot.agent == agent
    assert snapshot.transport == "Cli"
    with pytest.raises(AglSessionHostError):
        host.snapshot("unknown")
    assert parent.close_calls == 1
    assert child.close_calls == 1


def test_production_session_host_selects_and_rejects_transports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agm.agent.session import rpc

    spawned: list[list[str]] = []

    def spawn(agent: AgentPi, command: list[str], operation: str, **_kwargs: object) -> object:
        spawned.append(command)
        return object()

    monkeypatch.setattr(rpc, "_spawn", spawn)
    monkeypatch.setattr(rpc, "_terminate", lambda _child: None)
    host = create_agl_session_host(
        idle_timeout=1.0, get_sandbox_context=hermetic_get_sandbox_context()
    )
    pi = AgentPi(provider="provider", model="model", thinking="think")
    handle = host.open(pi, "Rpc", env={})
    host.close(handle)
    assert len(spawned) == 1
    command = AgentCommand(command="echo %{SESSION_ID}")
    command_handle = host.open(command, "Cli", env={})
    host.close(command_handle)
    with pytest.raises(AglSessionHostError):
        host.open(command, "Rpc", env={})


def test_agl_session_host_rejects_closed_and_unknown_sessions() -> None:
    """Agent decoding now happens in the evaluator; the host only sees specs."""
    backend = FakeBackend(frozenset())
    host = AglSessionHost(SessionService(lambda _request: backend))
    agent = AgentCommand(command="worker")
    handle = host.open(agent, "Cli", env={})
    assert host.ask(handle, "hello") == "answer"
    host.close(handle)
    with pytest.raises(AglSessionHostError):
        host.ask(handle, "closed")
    with pytest.raises(AglSessionHostError) as unknown:
        host.ask("unknown", "missing")
    assert unknown.value.operation == "ask"


def test_agl_session_host_preserves_success_response_metadata_and_call_info() -> None:
    backend = FakeBackend(
        frozenset(),
        response=SessionAskResponse(
            content="answer",
            metadata={"elapsed": 1.5, "provider": "pi"},
            call_info=AgentCallInfo(
                argv=["pi", "--rpc"],
                prompt_via_stdin=True,
                elapsed=1.5,
                exit_code=0,
                sandboxed=False,
                permission_mode="none",
            ),
        ),
    )
    host = AglSessionHost(SessionService(lambda _request: backend))
    agent = AgentPi(provider="provider", model="model", thinking="think")
    handle = host.open(agent, "Rpc", env={})

    def callback(_phase: AgentOutputPhase, _text: str, **_metadata: object) -> None:
        return None

    response = host.ask_request(
        handle, AgentRequest(agent=agent, prompt="hello", env={}, output_callback=callback)
    )

    assert response.content == "answer"
    assert response.metadata == {"elapsed": 1.5, "provider": "pi"}
    assert response.call_info is not None
    assert response.call_info.to_trace() == {
        "argv": ["pi", "--rpc"],
        "prompt_via_stdin": True,
        "elapsed": 1.5,
        "exit_code": 0,
        "sandboxed": False,
        "permission_mode": "none",
    }
    assert backend.ask_requests[0].output_callback is callback


def test_agl_session_host_translates_host_and_ask_failures() -> None:
    backend = FakeBackend(frozenset())
    host = AglSessionHost(SessionService(lambda _request: backend))
    agent = AgentCommand(command="worker")
    handle = host.default(agent, "Cli", env={})
    assert host.default(agent, "Cli", env={}) == handle

    backend.ask_error = SessionAskError(
        cause="failed",
        exit_code=2,
        stderr_tail="tail",
        elapsed=1.5,
        call_info=AgentCallInfo(
            argv=("worker",),
            prompt_via_stdin=True,
            elapsed=1.5,
            exit_code=2,
            sandboxed=False,
            permission_mode="none",
        ),
    )
    with pytest.raises(AglSessionAskError) as ask_error:
        host.ask(handle, "hello")
    assert ask_error.value.cause == "failed"

    host.close(handle)
    with pytest.raises(AglSessionHostError) as host_error:
        host.reset(handle)
    assert host_error.value.operation == "reset"


def test_open_ask_and_close_manage_a_live_session() -> None:
    service, factory = _service()
    agent = object()

    handle = service.open(agent, SessionTransport.CLI, name="named", env={})
    response = service.ask(handle, SessionAskRequest(prompt="hello"))
    service.close(handle)

    UUID(handle)
    backend = factory.backends[0]
    assert backend.open_requests == [
        SessionOpenRequest(agent=agent, transport=SessionTransport.CLI, name="named", env={})
    ]
    assert response == SessionAskResponse(content="answer")
    assert backend.ask_requests == [SessionAskRequest(prompt="hello")]
    assert backend.close_calls == 1


def test_close_is_idempotent_but_closed_unknown_and_forged_handles_cannot_be_used() -> None:
    service, _ = _service()
    handle = service.open(object(), SessionTransport.CLI, env={})

    service.close(handle)
    service.close(handle)

    _assert_error("ask", lambda: service.ask(handle, SessionAskRequest(prompt="again")))
    _assert_error("stats", lambda: service.stats("unknown"))
    _assert_error("reset", lambda: service.reset(str(UUID(int=0))))
    _assert_error("close", lambda: service.close("forged"))


def test_unsupported_operations_fail_before_the_backend_is_called() -> None:
    supported = ALL_OPERATIONS - {SessionOperation.COMPACT}
    service, factory = _service(FakeBackendFactory(supported))
    handle = service.open(object(), SessionTransport.CLI, env={})

    _assert_error("compact", lambda: service.compact(handle, "make room"))

    assert factory.backends[0].compact_requests == []


def test_default_session_snapshots_the_first_agent_and_returns_it_after_changes() -> None:
    service, factory = _service()
    first_agent = object()

    first = service.default(first_agent, SessionTransport.CLI, env={})
    second = service.default(object(), SessionTransport.RPC, env={})

    assert second == first
    assert len(factory.backends) == 1
    assert factory.backends[0].open_requests[0].agent is first_agent


def test_reset_all_discards_handles_and_starts_a_fresh_default_generation() -> None:
    service, factory = _service()
    first_default = service.default(AgentCommand("first"), SessionTransport.CLI, env={})
    explicit = service.open(AgentCommand("other"), SessionTransport.CLI, env={})
    service.close(explicit)

    service.reset_all()

    assert [backend.close_calls for backend in factory.backends] == [1, 1]
    assert not service.is_known(first_default)
    assert not service.is_known(explicit)
    next_default = service.default(AgentCommand("second"), SessionTransport.CLI, env={})
    assert next_default != first_default
    assert len(factory.backends) == 3


def test_reset_all_retains_failed_closes_for_cleanup_retry() -> None:
    service, factory = _service()
    failed = service.open(AgentCommand("failed"), SessionTransport.CLI, env={})
    closed = service.open(AgentCommand("closed"), SessionTransport.CLI, env={})
    factory.backends[0].close_error = RuntimeError("busy")

    with pytest.raises(ExceptionGroup):
        service.reset_all()

    assert service.is_known(failed)
    assert not service.is_known(closed)
    factory.backends[0].close_error = None
    service.reset_all()
    assert not service.is_known(failed)


def test_reset_all_starts_a_fresh_default_even_when_another_close_fails() -> None:
    service, factory = _service()
    first_default = service.default(AgentCommand("default"), SessionTransport.CLI, env={})
    other = service.open(AgentCommand("other"), SessionTransport.CLI, env={})
    factory.backends[1].close_error = RuntimeError("busy")

    with pytest.raises(ExceptionGroup):
        service.reset_all()

    assert not service.is_known(first_default)
    assert service.is_known(other)

    next_default = service.default(AgentCommand("second"), SessionTransport.CLI, env={})
    assert next_default != first_default
    assert service.ask(next_default, SessionAskRequest(prompt="hello")).content == "answer"


def test_close_all_closes_live_backends_once_and_is_safe_to_repeat() -> None:
    service, factory = _service()
    first = service.open(object(), SessionTransport.CLI, env={})
    second = service.open(object(), SessionTransport.RPC, env={})
    service.close(first)

    service.close_all()
    service.close_all()

    assert factory.backends[0].close_calls == 1
    assert factory.backends[1].close_calls == 1
    _assert_error("ask", lambda: service.ask(second, SessionAskRequest(prompt="nope")))


def test_close_all_attempts_every_session_and_leaves_failures_retryable() -> None:
    service, factory = _service()
    errors = [RuntimeError("first close failed"), None, RuntimeError("third close failed")]
    handles = [service.open(object(), SessionTransport.CLI, env={}) for _ in errors]
    for backend, error in zip(factory.backends, errors, strict=True):
        backend.close_error = error

    with pytest.raises(ExceptionGroup) as raised:
        service.close_all()

    assert raised.value.exceptions == (errors[0], errors[2])
    assert [backend.close_calls for backend in factory.backends] == [1, 1, 1]
    _assert_error("ask", lambda: service.ask(handles[1], SessionAskRequest(prompt="closed")))

    factory.backends[0].close_error = None
    factory.backends[2].close_error = None
    service.close_all()

    assert [backend.close_calls for backend in factory.backends] == [2, 1, 2]
    for handle in handles:
        _assert_error(
            "ask", lambda handle=handle: service.ask(handle, SessionAskRequest(prompt="closed"))
        )


def test_ephemeral_lifecycle_retires_its_handle_after_closing() -> None:
    service, factory = _service()
    agent = object()

    response = service.with_ephemeral(
        agent,
        SessionTransport.CLI,
        lambda handle: service.ask(handle, SessionAskRequest(prompt="hello")),
        env={},
    )

    assert response == SessionAskResponse(content="answer")
    assert factory.backends[0].close_calls == 1
    assert factory.backends[0].open_requests == [
        SessionOpenRequest(agent=agent, transport=SessionTransport.CLI, ephemeral=True, env={})
    ]
    assert service._entries == {}


def test_agl_host_single_prompt_ephemeral_lifecycle_retires_its_agent_mapping() -> None:
    service, factory = _service()
    host = AglSessionHost(service)
    agent = AgentCommand(command="worker")

    assert (
        host.with_ephemeral(
            agent, "Cli", lambda handle: host.ask(handle, "hello"), single_prompt=True, env={}
        )
        == "answer"
    )
    assert factory.backends[0].open_requests == [
        SessionOpenRequest(
            agent=AgentCommand("worker"),
            transport=SessionTransport.CLI,
            single_prompt=True,
            ephemeral=True,
            env={},
        )
    ]
    assert host._sessions == {}


def test_agl_host_ephemeral_lifecycle_retires_its_agent_mapping() -> None:
    service, factory = _service()
    host = AglSessionHost(service)
    agent = AgentCommand(command="worker")
    handles: list[str] = []

    response = host.with_ephemeral(
        agent,
        "Cli",
        lambda handle: handles.append(handle) or host.ask(handle, "hello"),
        env={},
    )

    assert response == "answer"
    assert factory.backends[0].close_calls == 1
    assert service._entries == {}
    assert host._sessions == {}
    with pytest.raises(AglSessionHostError):
        host.close(handles[0])


def test_agl_host_reset_all_retires_all_successfully_closed_mappings() -> None:
    service, _factory = _service()
    host = AglSessionHost(service)
    agent = AgentCommand(command="worker")
    persistent = host.open(agent, "Cli", env={})
    ephemeral = host.open_ephemeral(agent, "Cli", env={})

    host.reset_all()

    assert not service.is_known(persistent)
    assert not service.is_known(ephemeral)
    assert host._sessions == {}


def test_agl_host_reset_all_keeps_only_failed_close_mappings() -> None:
    service, factory = _service()
    host = AglSessionHost(service)
    agent = AgentCommand(command="worker")
    failed = host.open(agent, "Cli", env={})
    closed = host.open(agent, "Cli", env={})
    factory.backends[0].close_error = RuntimeError("busy")

    with pytest.raises(ExceptionGroup):
        host.reset_all()

    assert set(host._sessions) == {failed}
    assert service.is_known(failed)
    assert not service.is_known(closed)


def test_close_all_retires_closed_ephemeral_host_mappings() -> None:
    service, factory = _service()
    host = AglSessionHost(service)
    agent = AgentCommand(command="worker")
    handle = host.open_ephemeral(agent, "Cli", single_prompt=True, env={})

    host.close_all()

    assert factory.backends[0].open_requests == [
        SessionOpenRequest(
            agent=AgentCommand("worker"),
            transport=SessionTransport.CLI,
            single_prompt=True,
            ephemeral=True,
            env={},
        )
    ]
    assert factory.backends[0].close_calls == 1
    assert service._entries == {}
    assert host._sessions == {}
    with pytest.raises(AglSessionHostError):
        host.close(handle)


def test_close_all_keeps_an_ephemeral_mapping_when_its_close_can_be_retried() -> None:
    service, factory = _service()
    host = AglSessionHost(service)
    handle = host.open_ephemeral(AgentCommand(command="worker"), "Cli", env={})
    factory.backends[0].close_error = RuntimeError("close failed")

    with pytest.raises(ExceptionGroup):
        host.close_all()

    assert handle in service._entries
    assert host._sessions[handle].ephemeral

    factory.backends[0].close_error = None
    host.close(handle)


def _ask_ephemeral(
    service: SessionService, agent: object, request: SessionAskRequest
) -> SessionAskResponse:
    """Ask through one short-lived session, as the AgL ephemeral ask does."""
    return service.with_ephemeral(
        agent,
        SessionTransport.CLI,
        lambda handle: service.ask(handle, request),
        single_prompt=True,
        env={},
    )


def test_ephemeral_ask_returns_its_response_after_closing() -> None:
    service, factory = _service()
    agent = object()

    response = _ask_ephemeral(service, agent, SessionAskRequest(prompt="hello"))

    assert response == SessionAskResponse(content="answer")
    assert factory.backends[0].open_requests == [
        SessionOpenRequest(
            agent=agent, transport=SessionTransport.CLI, single_prompt=True, ephemeral=True, env={}
        )
    ]
    assert factory.backends[0].close_calls == 1


def test_ephemeral_ask_closes_after_a_backend_failure() -> None:
    factory = FakeBackendFactory()
    error = RuntimeError("transport failed")

    def make_failing_backend(request: SessionOpenRequest) -> SessionBackend:
        backend = FakeBackend(factory.supported, ask_error=error)
        factory.backends.append(backend)
        return backend

    failing_service = SessionService(make_failing_backend)

    with pytest.raises(RuntimeError, match="transport failed"):
        _ask_ephemeral(failing_service, object(), SessionAskRequest(prompt="hello"))

    assert factory.backends[0].close_calls == 1


def test_ephemeral_ask_preserves_an_ask_error_when_close_also_fails() -> None:
    factory = FakeBackendFactory()
    ask_error = SessionAskError(
        cause="nonzero_exit",
        exit_code=1,
        stderr_tail="agent failed",
        elapsed=0.1,
        call_info=AgentCallInfo(
            argv=["runner"],
            prompt_via_stdin=False,
            elapsed=0.1,
            exit_code=1,
            sandboxed=False,
            permission_mode="none",
        ),
    )

    def make_failing_backend(request: SessionOpenRequest) -> SessionBackend:
        backend = FakeBackend(
            factory.supported,
            ask_error=ask_error,
            close_error=RuntimeError("close failed"),
        )
        factory.backends.append(backend)
        return backend

    service = SessionService(make_failing_backend)

    with pytest.raises(SessionAskError) as raised:
        _ask_ephemeral(service, object(), SessionAskRequest(prompt="hello"))

    assert raised.value is ask_error
    assert raised.value.call_info.to_trace() == {
        "argv": ["runner"],
        "prompt_via_stdin": False,
        "elapsed": 0.1,
        "exit_code": 1,
        "sandboxed": False,
        "permission_mode": "none",
    }
    assert factory.backends[0].close_calls == 1


def test_ephemeral_ask_surfaces_a_close_failure_after_a_successful_ask() -> None:
    factory = FakeBackendFactory()
    close_error = RuntimeError("close failed")

    def make_failing_backend(request: SessionOpenRequest) -> SessionBackend:
        backend = FakeBackend(factory.supported, close_error=close_error)
        factory.backends.append(backend)
        return backend

    service = SessionService(make_failing_backend)

    with pytest.raises(RuntimeError) as raised:
        _ask_ephemeral(service, object(), SessionAskRequest(prompt="hello"))

    assert raised.value is close_error
    assert factory.backends[0].close_calls == 1


def test_fork_returns_a_distinct_live_handle_and_reset_preserves_the_original_handle() -> None:
    service, factory = _service()
    original = service.open(object(), SessionTransport.CLI, env={})
    backend = factory.backends[0]
    forked_backend = FakeBackend()
    backend.fork_result = forked_backend

    forked = service.fork(original)
    service.reset(original)

    assert forked != original
    assert backend.reset_calls == 1
    assert service.ask(original, SessionAskRequest(prompt="original")).content == "answer"
    assert service.ask(forked, SessionAskRequest(prompt="forked")).content == "answer"


def test_service_stores_and_forks_the_open_time_sandbox_mode() -> None:
    """A session's permission mode and sandbox limits are fixed at ``open``,
    reach the backend's own ``SessionOpenRequest``, and are copied forward to
    a forked session's entry -- independent of whatever internal bookkeeping
    the backend itself keeps for the same values."""
    limits = SandboxLimits(memory="4G")
    service, factory = _service()

    handle = service.open(
        AgentCommand("cat"),
        "cli",
        permission_mode=PermissionMode.UNRESTRICTED,
        sandbox=limits,
        env={},
    )

    backend = factory.backends[0]
    assert backend.open_requests[0].permission_mode == PermissionMode.UNRESTRICTED
    assert backend.open_requests[0].sandbox == limits
    assert service._entries[handle].permission_mode == PermissionMode.UNRESTRICTED
    assert service._entries[handle].sandbox == limits

    backend.fork_result = FakeBackend()
    forked = service.fork(handle)

    assert service._entries[forked].permission_mode == PermissionMode.UNRESTRICTED
    assert service._entries[forked].sandbox == limits
    # A session opened with no override keeps the "no sandbox" defaults.
    default_handle = service.open(AgentCommand("cat"), "cli", env={})
    assert service._entries[default_handle].permission_mode == PermissionMode.NONE
    assert service._entries[default_handle].sandbox is None


def test_agl_session_host_snapshot_exposes_the_open_time_sandbox_mode() -> None:
    """The AgL-facing snapshot carries the same fixed-at-open mode a fork inherits."""
    limits = SandboxLimits(memory="4G")
    service, factory = _service()
    host = AglSessionHost(service)
    agent = AgentCommand(command="worker")

    handle = host.open(
        agent, "Cli", permission_mode=PermissionMode.UNRESTRICTED, sandbox=limits, env={}
    )
    factory.backends[0].fork_result = FakeBackend()
    forked = host.fork(handle)

    snapshot = host.snapshot(handle)
    forked_snapshot = host.snapshot(forked)

    assert snapshot.permission_mode == PermissionMode.UNRESTRICTED
    assert snapshot.sandbox == limits
    assert forked_snapshot.permission_mode == PermissionMode.UNRESTRICTED
    assert forked_snapshot.sandbox == limits

    default_handle = host.open(agent, "Cli", env={})
    assert host.snapshot(default_handle).permission_mode == PermissionMode.NONE
    assert host.snapshot(default_handle).sandbox is None


def test_a_fork_of_an_ephemeral_session_outlives_its_own_close() -> None:
    """A fork inherits its parent's agent and transport but never its ephemerality."""
    service, factory = _service()
    host = AglSessionHost(service)
    agent = AgentCommand(command="worker")
    parent = host.open_ephemeral(agent, "Cli", env={})
    factory.backends[0].fork_result = FakeBackend()

    forked = host.fork(parent)
    host.close(forked)
    host.close(parent)

    assert service.is_known(forked)
    snapshot = host.snapshot(forked)
    assert (snapshot.agent, snapshot.transport) == (agent, "Cli")
    host.close(forked)
    assert not service.is_known(parent)
    with pytest.raises(AglSessionHostError):
        host.close(parent)


def test_supported_operations_dispatch_to_the_backend() -> None:
    service, factory = _service()
    handle = service.open(object(), SessionTransport.RPC, env={})

    service.compact(handle)
    service.set_name(handle, "renamed")
    stats = service.stats(handle)

    assert factory.backends[0].compact_requests == [""]
    assert factory.backends[0].names == ["renamed"]
    assert stats.output_tokens == 5
