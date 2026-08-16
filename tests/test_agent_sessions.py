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
    SessionCapabilities,
    SessionHostError,
    SessionOpenRequest,
    SessionOperation,
    SessionService,
    SessionStats,
    create_agl_session_host,
)
from agm.agent.spec import AgentCommand
from agm.agent.transport import AgentCallInfo
from agm.agl.runtime.request import AgentRequest
from agm.agl.runtime.sessions import SessionAskError as AglSessionAskError
from agm.agl.runtime.sessions import SessionHostError as AglSessionHostError
from tests._agl_helpers import agent_value


@dataclass
class FakeBackend:
    """In-memory backend used to observe the session service's behavior."""

    capabilities: SessionCapabilities
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

    def open(self, request: SessionOpenRequest) -> None:
        self.open_requests.append(request)

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
    def __init__(self, capabilities: SessionCapabilities | None = None) -> None:
        self.capabilities = capabilities or SessionCapabilities.all()
        self.backends: list[FakeBackend] = []

    def __call__(self, agent: object, transport: str) -> SessionBackend:
        backend = FakeBackend(capabilities=self.capabilities)
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
    capabilities = SessionCapabilities(frozenset(SessionOperation))
    parent = FakeBackend(capabilities)
    child = FakeBackend(capabilities)
    parent.fork_result = child
    created: list[tuple[object, str]] = []

    def backend_for(agent: object, transport: str) -> SessionBackend:
        created.append((agent, transport))
        return parent

    host = AglSessionHost(SessionService(backend_for))
    agent = agent_value("AgentCommand", command="worker")
    handle = host.open(agent, "Cli", name="primary")
    assert host.ask(handle, "hello") == "answer"
    host.compact(handle, "retain")
    host.reset(handle)
    child_handle = host.fork(handle)
    host.set_name(child_handle, "child")
    stats = host.stats(child_handle)
    snapshot = host.snapshot(child_handle)
    host.close(handle)
    host.close_all()

    assert created[0][1] == "cli"
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
    from agm.agent.session.rpc import PiRpcSessionBackend

    monkeypatch.setattr(PiRpcSessionBackend, "_start", lambda self, _argv, _operation: None)
    host = create_agl_session_host(idle_timeout=1.0)
    pi = agent_value("AgentPi", provider="provider", model="model", thinking="think")
    handle = host.open(pi, "Rpc")
    host.close(handle)
    command = agent_value("AgentCommand", command="echo %{SESSION_ID}")
    command_handle = host.open(command, "Cli")
    host.close(command_handle)
    with pytest.raises(AglSessionHostError):
        host.open(command, "Rpc")
    with pytest.raises(AglSessionHostError):
        host.open(command, "bad")


def test_agl_session_host_confirmation_and_invalid_agent_are_handled() -> None:
    capabilities = SessionCapabilities(frozenset({SessionOperation.ASK}))
    backend = FakeBackend(capabilities)
    confirmed: list[tuple[object, str]] = []
    host = AglSessionHost(
        SessionService(lambda _agent, _transport: backend),
        confirm_session=lambda agent, prompt: confirmed.append((agent, prompt)),
    )
    agent = agent_value("AgentCommand", command="worker")
    handle = host.open(agent, "Cli")
    assert host.ask(handle, "hello") == "answer"
    assert confirmed == [(agent, "hello")]
    host.close(handle)
    with pytest.raises(AglSessionHostError):
        host.ask(handle, "closed")
    with pytest.raises(AglSessionHostError) as unknown:
        host.ask("unknown", "missing")
    assert unknown.value.operation == "ask"
    with pytest.raises(TypeError):
        host._agent_spec(object())
    with pytest.raises(AglSessionHostError):
        host._agent_spec(agent_value("Unknown"))


def test_agl_session_host_preserves_success_response_metadata_and_call_info() -> None:
    capabilities = SessionCapabilities(frozenset({SessionOperation.ASK}))
    backend = FakeBackend(
        capabilities,
        response=SessionAskResponse(
            content="answer",
            metadata={"elapsed": 1.5, "provider": "pi"},
            call_info=AgentCallInfo(
                argv=["pi", "--rpc"],
                prompt_via_stdin=True,
                elapsed=1.5,
                exit_code=0,
            ),
        ),
    )
    host = AglSessionHost(SessionService(lambda _agent, _transport: backend))
    agent = agent_value("AgentPi", provider="provider", model="model", thinking="think")
    handle = host.open(agent, "Rpc")

    response = host.ask_request(handle, AgentRequest(agent=agent, prompt="hello"))

    assert response.content == "answer"
    assert response.metadata == {"elapsed": 1.5, "provider": "pi"}
    assert response.call_info is not None
    assert response.call_info.to_trace() == {
        "argv": ["pi", "--rpc"],
        "prompt_via_stdin": True,
        "elapsed": 1.5,
        "exit_code": 0,
    }


def test_agl_session_host_translates_host_and_ask_failures() -> None:
    capabilities = SessionCapabilities(frozenset({SessionOperation.ASK}))
    backend = FakeBackend(capabilities)
    host = AglSessionHost(SessionService(lambda _agent, _transport: backend))
    agent = agent_value("AgentCommand", command="worker")
    handle = host.default(agent, "Cli")
    assert host.default(agent, "Cli") == handle

    backend.ask_error = SessionAskError(
        cause="failed",
        exit_code=2,
        stderr_tail="tail",
        elapsed=1.5,
        call_info=AgentCallInfo(argv=("worker",), prompt_via_stdin=True, elapsed=1.5, exit_code=2),
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

    handle = service.open(agent, "cli", name="named")
    response = service.ask(handle, SessionAskRequest(prompt="hello"))
    service.close(handle)

    UUID(handle)
    backend = factory.backends[0]
    assert backend.open_requests == [SessionOpenRequest(agent=agent, transport="cli", name="named")]
    assert response == SessionAskResponse(content="answer")
    assert backend.ask_requests == [SessionAskRequest(prompt="hello")]
    assert backend.close_calls == 1


def test_close_is_idempotent_but_closed_unknown_and_forged_handles_cannot_be_used() -> None:
    service, _ = _service()
    handle = service.open(object(), "cli")

    service.close(handle)
    service.close(handle)

    _assert_error("ask", lambda: service.ask(handle, SessionAskRequest(prompt="again")))
    _assert_error("stats", lambda: service.stats("unknown"))
    _assert_error("reset", lambda: service.reset(str(UUID(int=0))))
    _assert_error("close", lambda: service.close("forged"))


def test_unsupported_operations_fail_before_the_backend_is_called() -> None:
    capabilities = SessionCapabilities.all() - {SessionOperation.COMPACT}
    service, factory = _service(FakeBackendFactory(capabilities))
    handle = service.open(object(), "cli")

    _assert_error("compact", lambda: service.compact(handle, "make room"))

    assert factory.backends[0].compact_requests == []


def test_default_session_snapshots_the_first_agent_and_returns_it_after_changes() -> None:
    service, factory = _service()
    first_agent = object()

    first = service.default(first_agent, "cli")
    second = service.default(object(), "rpc")

    assert second == first
    assert len(factory.backends) == 1
    assert factory.backends[0].open_requests[0].agent is first_agent


def test_close_all_closes_live_backends_once_and_is_safe_to_repeat() -> None:
    service, factory = _service()
    first = service.open(object(), "cli")
    second = service.open(object(), "rpc")
    service.close(first)

    service.close_all()
    service.close_all()

    assert factory.backends[0].close_calls == 1
    assert factory.backends[1].close_calls == 1
    _assert_error("ask", lambda: service.ask(second, SessionAskRequest(prompt="nope")))


def test_close_all_attempts_every_session_and_leaves_failures_retryable() -> None:
    service, factory = _service()
    errors = [RuntimeError("first close failed"), None, RuntimeError("third close failed")]
    handles = [service.open(object(), "cli") for _ in errors]
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
        "cli",
        lambda handle: service.ask(handle, SessionAskRequest(prompt="hello")),
    )

    assert response == SessionAskResponse(content="answer")
    assert factory.backends[0].close_calls == 1
    assert factory.backends[0].open_requests == [SessionOpenRequest(agent=agent, transport="cli")]
    assert service._entries == {}


def test_agl_host_one_shot_ephemeral_lifecycle_retires_its_agent_mapping() -> None:
    service, factory = _service()
    host = AglSessionHost(service)
    agent = agent_value("AgentCommand", command="worker")

    assert (
        host.with_ephemeral(agent, "Cli", lambda handle: host.ask(handle, "hello"), one_shot=True)
        == "answer"
    )
    assert factory.backends[0].open_requests == [
        SessionOpenRequest(agent=AgentCommand("worker"), transport="cli", one_shot=True)
    ]
    assert host._agents == {}


def test_agl_host_ephemeral_lifecycle_retires_its_agent_mapping() -> None:
    service, factory = _service()
    host = AglSessionHost(service)
    agent = agent_value("AgentCommand", command="worker")
    handles: list[str] = []

    response = host.with_ephemeral(
        agent,
        "Cli",
        lambda handle: handles.append(handle) or host.ask(handle, "hello"),
    )

    assert response == "answer"
    assert factory.backends[0].close_calls == 1
    assert service._entries == {}
    assert host._agents == {}
    assert host._ephemeral_handles == set()
    with pytest.raises(AglSessionHostError):
        host.close(handles[0])


def test_close_all_retires_closed_ephemeral_host_mappings() -> None:
    service, factory = _service()
    host = AglSessionHost(service)
    agent = agent_value("AgentCommand", command="worker")
    handle = host.open_ephemeral(agent, "Cli", one_shot=True)

    host.close_all()

    assert factory.backends[0].open_requests == [
        SessionOpenRequest(agent=AgentCommand("worker"), transport="cli", one_shot=True)
    ]
    assert factory.backends[0].close_calls == 1
    assert service._entries == {}
    assert host._agents == {}
    assert host._ephemeral_handles == set()
    with pytest.raises(AglSessionHostError):
        host.close(handle)


def test_close_all_keeps_an_ephemeral_mapping_when_its_close_can_be_retried() -> None:
    service, factory = _service()
    host = AglSessionHost(service)
    handle = host.open_ephemeral(agent_value("AgentCommand", command="worker"), "Cli")
    factory.backends[0].close_error = RuntimeError("close failed")

    with pytest.raises(ExceptionGroup):
        host.close_all()

    assert handle in service._entries
    assert handle in host._agents
    assert handle in host._ephemeral_handles

    factory.backends[0].close_error = None
    host.close(handle)


def test_ephemeral_ask_returns_its_response_after_closing() -> None:
    service, factory = _service()
    agent = object()

    response = service.ask_ephemeral(agent, "cli", SessionAskRequest(prompt="hello"))

    assert response == SessionAskResponse(content="answer")
    assert factory.backends[0].open_requests == [
        SessionOpenRequest(agent=agent, transport="cli", one_shot=True)
    ]
    assert factory.backends[0].close_calls == 1


def test_ephemeral_ask_closes_after_a_backend_failure() -> None:
    factory = FakeBackendFactory()
    error = RuntimeError("transport failed")

    def make_failing_backend(agent: object, transport: str) -> SessionBackend:
        backend = FakeBackend(capabilities=factory.capabilities, ask_error=error)
        factory.backends.append(backend)
        return backend

    failing_service = SessionService(make_failing_backend)

    with pytest.raises(RuntimeError, match="transport failed"):
        failing_service.ask_ephemeral(object(), "cli", SessionAskRequest(prompt="hello"))

    assert factory.backends[0].close_calls == 1


def test_ephemeral_ask_preserves_an_ask_error_when_close_also_fails() -> None:
    factory = FakeBackendFactory()
    ask_error = SessionAskError(
        cause="nonzero_exit",
        exit_code=1,
        stderr_tail="agent failed",
        elapsed=0.1,
        call_info=AgentCallInfo(argv=["runner"], prompt_via_stdin=False, elapsed=0.1, exit_code=1),
    )

    def make_failing_backend(agent: object, transport: str) -> SessionBackend:
        backend = FakeBackend(
            capabilities=factory.capabilities,
            ask_error=ask_error,
            close_error=RuntimeError("close failed"),
        )
        factory.backends.append(backend)
        return backend

    service = SessionService(make_failing_backend)

    with pytest.raises(SessionAskError) as raised:
        service.ask_ephemeral(object(), "cli", SessionAskRequest(prompt="hello"))

    assert raised.value is ask_error
    assert raised.value.call_info.to_trace() == {
        "argv": ["runner"],
        "prompt_via_stdin": False,
        "elapsed": 0.1,
        "exit_code": 1,
    }
    assert factory.backends[0].close_calls == 1


def test_ephemeral_ask_surfaces_a_close_failure_after_a_successful_ask() -> None:
    factory = FakeBackendFactory()
    close_error = RuntimeError("close failed")

    def make_failing_backend(agent: object, transport: str) -> SessionBackend:
        backend = FakeBackend(capabilities=factory.capabilities, close_error=close_error)
        factory.backends.append(backend)
        return backend

    service = SessionService(make_failing_backend)

    with pytest.raises(RuntimeError) as raised:
        service.ask_ephemeral(object(), "cli", SessionAskRequest(prompt="hello"))

    assert raised.value is close_error
    assert factory.backends[0].close_calls == 1


def test_fork_returns_a_distinct_live_handle_and_reset_preserves_the_original_handle() -> None:
    service, factory = _service()
    original = service.open(object(), "cli")
    backend = factory.backends[0]
    forked_backend = FakeBackend(capabilities=SessionCapabilities.all())
    backend.fork_result = forked_backend

    forked = service.fork(original)
    service.reset(original)

    assert forked != original
    assert backend.reset_calls == 1
    assert service.ask(original, SessionAskRequest(prompt="original")).content == "answer"
    assert service.ask(forked, SessionAskRequest(prompt="forked")).content == "answer"


def test_supported_operations_dispatch_to_the_backend() -> None:
    service, factory = _service()
    handle = service.open(object(), "rpc")

    service.compact(handle)
    service.set_name(handle, "renamed")
    stats = service.stats(handle)

    assert factory.backends[0].compact_requests == [""]
    assert factory.backends[0].names == ["renamed"]
    assert stats.output_tokens == 5
