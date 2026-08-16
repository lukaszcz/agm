"""Host-owned session table and lifecycle service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, Protocol, TypeVar, cast
from uuid import uuid4

if TYPE_CHECKING:
    from agm.agl.runtime.request import AgentRequest, AgentResponse
    from agm.agl.runtime.sessions import SessionSnapshot
    from agm.agl.runtime.sessions import SessionStats as AglSessionStats
    from agm.agl.semantics.values import EnumValue

from agm.agent.session.protocol import (
    SessionAgentError as AgentSessionAgentError,
)
from agm.agent.session.protocol import (
    SessionAskError,
    SessionAskRequest,
    SessionAskResponse,
    SessionBackend,
    SessionHostError,
    SessionOpenRequest,
    SessionOperation,
    SessionStats,
)
from agm.core.cleanup import preserve_primary_error

_T = TypeVar("_T")
SessionConfirmation = Callable[["EnumValue", str], None]

SessionBackendFactory = Callable[[object, str], SessionBackend]


class _SessionBackendConstructor(Protocol):
    def __call__(self, *, idle_timeout: float | None = None) -> SessionBackend: ...


class AglSessionHost:
    """Adapt the AGM session service to the firewall-safe AgL host protocol."""

    def __init__(
        self, service: SessionService, *, confirm_session: SessionConfirmation | None = None
    ) -> None:
        self._service = service
        self._confirm_session = confirm_session
        self._agents: dict[str, tuple[EnumValue, str]] = {}
        self._ephemeral_handles: set[str] = set()

    def open(self, agent: EnumValue, transport: str, *, name: str = "") -> str:
        handle = self._open(agent, transport, name=name)
        self._agents[handle] = (agent, transport)
        return handle

    def open_ephemeral(self, agent: EnumValue, transport: str, *, one_shot: bool = False) -> str:
        """Open one short-lived session for an AgL ask lifecycle."""
        handle = self._open(agent, transport, ephemeral=True, one_shot=one_shot)
        self._agents[handle] = (agent, transport)
        self._ephemeral_handles.add(handle)
        return handle

    def with_ephemeral(
        self,
        agent: EnumValue,
        transport: str,
        action: Callable[[str], _T],
        *,
        one_shot: bool = False,
    ) -> _T:
        """Run *action* in one ephemeral session and release it afterward."""
        spec = self._agent_spec(agent)

        def register(handle: str) -> _T:
            self._agents[handle] = (agent, transport)
            self._ephemeral_handles.add(handle)
            return action(handle)

        if one_shot:
            return self._call_host(
                lambda: self._service.with_ephemeral(
                    spec,
                    transport.lower(),
                    register,
                    on_closed=self._retire_ephemeral,
                    one_shot=True,
                )
            )
        return self._call_host(
            lambda: self._service.with_ephemeral(
                spec, transport.lower(), register, on_closed=self._retire_ephemeral
            )
        )

    def _open(
        self,
        agent: EnumValue,
        transport: str,
        *,
        name: str = "",
        ephemeral: bool = False,
        one_shot: bool = False,
    ) -> str:
        spec = self._agent_spec(agent)
        if ephemeral:
            if one_shot:
                return self._call_host(
                    lambda: self._service.open(
                        spec, transport.lower(), name=name, ephemeral=True, one_shot=True
                    )
                )
            return self._call_host(
                lambda: self._service.open(spec, transport.lower(), name=name, ephemeral=True)
            )
        return self._call_host(lambda: self._service.open(spec, transport.lower(), name=name))

    def default(self, agent: EnumValue, transport: str, *, name: str = "") -> str:
        handle = self._call_host(
            lambda: self._service.default(self._agent_spec(agent), transport.lower(), name=name)
        )
        self._agents.setdefault(handle, (agent, transport))
        return handle

    def ask(self, handle: str, prompt: str) -> str:
        """Send a plain session prompt for hosts without request envelopes."""
        return self._ask(handle, prompt).content

    def ask_request(self, handle: str, request: "AgentRequest") -> "AgentResponse":
        """Preserve a session response's metadata across the AgL firewall."""
        from agm.agl.runtime.request import AgentCallInfo, AgentResponse

        response = self._ask(handle, request.prompt)
        call_info = response.call_info
        return AgentResponse(
            content=response.content,
            metadata=dict(response.metadata),
            call_info=(
                None
                if call_info is None
                else AgentCallInfo(
                    argv=call_info.argv,
                    prompt_via_stdin=call_info.prompt_via_stdin,
                    elapsed=call_info.elapsed,
                    exit_code=call_info.exit_code,
                )
            ),
        )

    def _ask(self, handle: str, prompt: str) -> SessionAskResponse:
        try:
            if self._confirm_session is not None:
                try:
                    agent, _transport = self._agents[handle]
                except KeyError:
                    from agm.agl.runtime.sessions import SessionHostError as AglSessionHostError

                    raise AglSessionHostError("unknown session", "ask") from None
                self._confirm_session(agent, prompt)
            return self._service.ask(handle, SessionAskRequest(prompt))
        except SessionAskError as error:
            self._raise_ask_error(error)
        except SessionHostError as error:
            self._raise_host_error(error)

    def compact(self, handle: str, instructions: str = "") -> None:
        self._call_host(lambda: self._service.compact(handle, instructions))

    def reset(self, handle: str) -> None:
        self._call_host(lambda: self._service.reset(handle))

    def fork(self, handle: str) -> str:
        forked = self._call_host(lambda: self._service.fork(handle))
        self._agents[forked] = self._agents[handle]
        return forked

    def set_name(self, handle: str, name: str) -> None:
        self._call_host(lambda: self._service.set_name(handle, name))

    def stats(self, handle: str) -> "AglSessionStats":
        from agm.agl.runtime.sessions import SessionStats as AglSessionStats

        stats = self._call_host(lambda: self._service.stats(handle))
        return AglSessionStats(
            input_tokens=stats.input_tokens,
            output_tokens=stats.output_tokens,
            cost=stats.cost,
            context_percent=stats.context_percent,
        )

    def snapshot(self, handle: str) -> "SessionSnapshot":
        from agm.agl.runtime.sessions import SessionHostError as AglSessionHostError
        from agm.agl.runtime.sessions import SessionSnapshot

        try:
            agent, transport = self._agents[handle]
        except KeyError:
            raise AglSessionHostError("unknown session", "snapshot") from None
        return SessionSnapshot(agent=agent, transport=transport)

    def close(self, handle: str) -> None:
        self._call_host(lambda: self._service.close(handle))
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
            self._agents.pop(handle, None)

    @staticmethod
    def _agent_spec(agent: object) -> object:
        from agm.agl.runtime.agents import decode_agent_value
        from agm.agl.semantics.values import EnumValue

        if not isinstance(agent, EnumValue):
            raise TypeError(f"session agent must be an EnumValue, got {type(agent).__name__}")
        try:
            return decode_agent_value(agent)
        except ValueError as error:
            from agm.agl.runtime.sessions import SessionAgentError

            raise SessionAgentError(str(error), "open") from error

    @staticmethod
    def _raise_host_error(error: SessionHostError) -> NoReturn:
        from agm.agl.runtime.sessions import SessionHostError as AglSessionHostError

        raise AglSessionHostError(error.message, error.operation) from error

    @staticmethod
    def _raise_ask_error(error: SessionAskError) -> NoReturn:
        from agm.agl.runtime.request import AgentCallInfo
        from agm.agl.runtime.sessions import SessionAskError as AglSessionAskError

        call_info = (
            AgentCallInfo(
                argv=error.call_info.argv,
                prompt_via_stdin=error.call_info.prompt_via_stdin,
                elapsed=error.call_info.elapsed,
                exit_code=error.call_info.exit_code,
            )
            if error.call_info is not None
            else None
        )
        raise AglSessionAskError(
            cause=error.cause,
            exit_code=error.exit_code,
            stderr_tail=error.stderr_tail,
            elapsed=error.elapsed,
            call_info=call_info,
        ) from error

    @staticmethod
    def _call_host(action: Callable[[], _T]) -> _T:
        try:
            return action()
        except AgentSessionAgentError as error:
            from agm.agl.runtime.sessions import SessionAgentError

            raise SessionAgentError(error.message, error.operation) from error
        except SessionHostError as error:
            AglSessionHost._raise_host_error(error)


def create_agl_session_host(
    *, idle_timeout: float | None, confirm_session: SessionConfirmation | None = None
) -> AglSessionHost:
    """Create the production AgL session host with transport-aware backends."""
    from agm.agent.session.cli_adapters import CLI_SESSION_BACKENDS
    from agm.agent.session.rpc import PiRpcSessionBackend
    from agm.agent.spec import AgentPi

    def backend_for(agent: object, transport: str) -> SessionBackend:
        if transport == "rpc":
            if isinstance(agent, AgentPi):
                return PiRpcSessionBackend(idle_timeout=idle_timeout)
            raise SessionHostError("RPC transport is only supported by AgentPi", "open")
        if transport != "cli":
            raise SessionHostError(f"unsupported session transport {transport!r}", "open")
        backend_type = CLI_SESSION_BACKENDS[type(agent).__name__]
        return cast(_SessionBackendConstructor, backend_type)(idle_timeout=idle_timeout)

    return AglSessionHost(SessionService(backend_for), confirm_session=confirm_session)


@dataclass(slots=True)
class _SessionEntry:
    """The host state associated with one opaque session handle."""

    backend: SessionBackend
    agent: object
    transport: str
    ephemeral: bool
    closed: bool = False


class SessionService:
    """Own live session backends behind AGM-generated opaque handle ids."""

    def __init__(self, backend_factory: SessionBackendFactory) -> None:
        self._backend_factory = backend_factory
        self._entries: dict[str, _SessionEntry] = {}
        self._default_handle: str | None = None

    def open(
        self,
        agent: object,
        transport: str,
        *,
        name: str = "",
        ephemeral: bool = False,
        one_shot: bool = False,
    ) -> str:
        """Open a backend session and return its host-generated handle id."""
        backend = self._backend_factory(agent, transport)
        backend.open(
            SessionOpenRequest(agent=agent, transport=transport, name=name, one_shot=one_shot)
        )
        handle = str(uuid4())
        self._entries[handle] = _SessionEntry(
            backend=backend,
            agent=agent,
            transport=transport,
            ephemeral=ephemeral,
        )
        return handle

    def default(self, agent: object, transport: str, *, name: str = "") -> str:
        """Return the lazily opened default session, snapshotting its first agent."""
        if self._default_handle is None:
            self._default_handle = self.open(agent, transport, name=name)
        return self._default_handle

    def ask(self, handle: str, request: SessionAskRequest) -> SessionAskResponse:
        """Send *request* through a live session."""
        entry = self._entry_for(handle, SessionOperation.ASK)
        self._require_capability(entry, SessionOperation.ASK)
        return entry.backend.ask(request)

    def compact(self, handle: str, instructions: str = "") -> None:
        """Compact a live session when its backend supports compaction."""
        entry = self._entry_for(handle, SessionOperation.COMPACT)
        self._require_capability(entry, SessionOperation.COMPACT)
        self._run_lifecycle(SessionOperation.COMPACT, lambda: entry.backend.compact(instructions))

    def reset(self, handle: str) -> None:
        """Reset a live session while retaining its handle, agent, and transport."""
        entry = self._entry_for(handle, "reset")
        self._run_lifecycle("reset", entry.backend.reset)

    def fork(self, handle: str) -> str:
        """Fork a live session, returning a distinct host handle for the child."""
        entry = self._entry_for(handle, SessionOperation.FORK)
        self._require_capability(entry, SessionOperation.FORK)
        forked_backend = self._run_lifecycle(SessionOperation.FORK, entry.backend.fork)
        forked_handle = str(uuid4())
        self._entries[forked_handle] = _SessionEntry(
            backend=forked_backend,
            agent=entry.agent,
            transport=entry.transport,
            ephemeral=False,
        )
        return forked_handle

    def set_name(self, handle: str, name: str) -> None:
        """Set a live session's backend-visible name."""
        entry = self._entry_for(handle, SessionOperation.SET_NAME)
        self._require_capability(entry, SessionOperation.SET_NAME)
        self._run_lifecycle(SessionOperation.SET_NAME, lambda: entry.backend.set_name(name))

    def stats(self, handle: str) -> SessionStats:
        """Return usage statistics for a live session."""
        entry = self._entry_for(handle, SessionOperation.STATS)
        self._require_capability(entry, SessionOperation.STATS)
        return self._run_lifecycle(SessionOperation.STATS, entry.backend.stats)

    def close(self, handle: str) -> None:
        """Close a live session; closing an already closed known handle is a no-op."""
        entry = self._entries.get(handle)
        if entry is None:
            raise self._unknown_handle_error(handle, "close")
        if entry.closed:
            return
        self._run_lifecycle("close", entry.backend.close)
        if entry.ephemeral:
            del self._entries[handle]
        else:
            entry.closed = True

    def is_known(self, handle: str) -> bool:
        """Whether *handle* is still retained by this service."""
        return handle in self._entries

    def close_all(self) -> None:
        """Close every live backend, raising grouped failures after all attempts.

        Sessions are marked closed only after their backend closes successfully, so a
        later call can retry failed closes.
        """
        failures: list[Exception] = []
        for handle, entry in tuple(self._entries.items()):
            if entry.closed:
                continue
            try:
                self.close(handle)
            except Exception as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup("failed to close one or more agent sessions", failures)

    def with_ephemeral(
        self,
        agent: object,
        transport: str,
        action: Callable[[str], _T],
        *,
        name: str = "",
        on_closed: Callable[[str], None] | None = None,
        one_shot: bool = False,
    ) -> _T:
        """Run *action* in a short-lived session and release it afterward."""
        handle = self.open(agent, transport, name=name, ephemeral=True, one_shot=one_shot)

        def close() -> None:
            self.close(handle)
            if on_closed is not None:
                on_closed(handle)

        with preserve_primary_error(close, label="ephemeral session cleanup"):
            return action(handle)

    def ask_ephemeral(
        self,
        agent: object,
        transport: str,
        request: SessionAskRequest,
        *,
        name: str = "",
    ) -> SessionAskResponse:
        """Ask through a short-lived session and close it even when asking fails."""
        return self.with_ephemeral(
            agent,
            transport,
            lambda handle: self.ask(handle, request),
            name=name,
            one_shot=True,
        )

    @staticmethod
    def _run_lifecycle(operation: SessionOperation | str, action: Callable[[], _T]) -> _T:
        """Map backend transport failures to the public lifecycle error model."""
        operation_name = SessionService._operation_name(operation)
        try:
            return action()
        except SessionAskError as error:
            raise SessionHostError(
                f"session {operation_name} transport failed ({error.cause}): {error.stderr_tail}",
                operation_name,
            ) from error

    def _entry_for(self, handle: str, operation: SessionOperation | str) -> _SessionEntry:
        entry = self._entries.get(handle)
        if entry is None:
            raise self._unknown_handle_error(handle, operation)
        if entry.closed:
            raise SessionHostError(
                f"session handle {handle!r} is closed",
                self._operation_name(operation),
            )
        return entry

    def _require_capability(self, entry: _SessionEntry, operation: SessionOperation) -> None:
        if not entry.backend.capabilities.supports(operation):
            raise SessionHostError(
                f"session backend does not support {operation.value}",
                operation.value,
            )

    @staticmethod
    def _unknown_handle_error(handle: str, operation: SessionOperation | str) -> SessionHostError:
        return SessionHostError(
            f"unknown session handle {handle!r}",
            SessionService._operation_name(operation),
        )

    @staticmethod
    def _operation_name(operation: SessionOperation | str) -> str:
        return operation.value if isinstance(operation, SessionOperation) else operation
