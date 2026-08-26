"""Host-owned session table and lifecycle service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NoReturn, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from agm.agl.runtime.request import AgentRequest, AgentResponse
    from agm.agl.runtime.sessions import SessionSnapshot
    from agm.agl.runtime.sessions import SessionStats as AglSessionStats
    from agm.agl.semantics.values import RecordValue

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
SessionConfirmation = Callable[["RecordValue", str], None]

SessionBackendFactory = Callable[[object, str], SessionBackend]


@dataclass(frozen=True, slots=True)
class _HostSession:
    """The AgL-facing identity this host retains for one opaque handle."""

    agent: "RecordValue"
    transport: str
    ephemeral: bool = False


class AglSessionHost:
    """Adapt the AGM session service to the firewall-safe AgL host protocol."""

    def __init__(
        self, service: SessionService, *, confirm_session: SessionConfirmation | None = None
    ) -> None:
        self._service = service
        self._confirm_session = confirm_session
        self._sessions: dict[str, _HostSession] = {}

    def open(self, agent: RecordValue, transport: str, *, name: str = "") -> str:
        handle = self._open(agent, transport, name=name)
        self._sessions[handle] = _HostSession(agent, transport)
        return handle

    def open_ephemeral(
        self, agent: RecordValue, transport: str, *, single_prompt: bool = False
    ) -> str:
        """Open one short-lived session for an AgL ask lifecycle."""
        handle = self._open(agent, transport, ephemeral=True, single_prompt=single_prompt)
        self._sessions[handle] = _HostSession(agent, transport, ephemeral=True)
        return handle

    def with_ephemeral(
        self,
        agent: RecordValue,
        transport: str,
        action: Callable[[str], _T],
        *,
        single_prompt: bool = False,
    ) -> _T:
        """Run *action* in one ephemeral session and release it afterward."""
        spec = self._agent_spec(agent)

        def register(handle: str) -> _T:
            self._sessions[handle] = _HostSession(agent, transport, ephemeral=True)
            return action(handle)

        return self._call_host(
            lambda: self._service.with_ephemeral(
                spec,
                transport.lower(),
                register,
                on_closed=self._retire_ephemeral,
                single_prompt=single_prompt,
            )
        )

    def _open(
        self,
        agent: RecordValue,
        transport: str,
        *,
        name: str = "",
        ephemeral: bool = False,
        single_prompt: bool = False,
    ) -> str:
        spec = self._agent_spec(agent)
        return self._call_host(
            lambda: self._service.open(
                spec, transport.lower(), name=name, ephemeral=ephemeral, single_prompt=single_prompt
            )
        )

    def default(self, agent: RecordValue, transport: str, *, name: str = "") -> str:
        handle = self._call_host(
            lambda: self._service.default(self._agent_spec(agent), transport.lower(), name=name)
        )
        self._sessions.setdefault(handle, _HostSession(agent, transport))
        return handle

    def ask(self, handle: str, prompt: str) -> str:
        """Send a plain session prompt for hosts without request envelopes."""
        return self._ask(handle, prompt).content

    def ask_request(self, handle: str, request: "AgentRequest") -> "AgentResponse":
        """Preserve a session response's metadata across the AgL firewall."""
        from agm.agl.runtime.request import AgentResponse

        response = self._ask(handle, request.prompt)
        return AgentResponse(
            content=response.content,
            metadata=dict(response.metadata),
            call_info=response.call_info,
        )

    def _ask(self, handle: str, prompt: str) -> SessionAskResponse:
        try:
            if self._confirm_session is not None:
                self._confirm_session(self._session_for(handle, "ask").agent, prompt)
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
        self._sessions[forked] = replace(self._sessions[handle], ephemeral=False)
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
        from agm.agl.runtime.sessions import SessionSnapshot

        session = self._session_for(handle, "snapshot")
        return SessionSnapshot(agent=session.agent, transport=session.transport)

    def close(self, handle: str) -> None:
        self._call_host(lambda: self._service.close(handle))
        self._retire_ephemeral(handle)

    def reset_all(self) -> None:
        try:
            self._service.reset_all()
        finally:
            self._forget_released()

    def close_all(self) -> None:
        try:
            self._service.close_all()
        finally:
            self._forget_released()

    def _forget_released(self) -> None:
        """Drop the identities of handles the service no longer retains."""
        for handle in tuple(self._sessions):
            if not self._service.is_known(handle):
                del self._sessions[handle]

    def _retire_ephemeral(self, handle: str) -> None:
        session = self._sessions.get(handle)
        if session is not None and session.ephemeral:
            del self._sessions[handle]

    def _session_for(self, handle: str, operation: str) -> _HostSession:
        from agm.agl.runtime.sessions import SessionHostError as AglSessionHostError

        try:
            return self._sessions[handle]
        except KeyError:
            raise AglSessionHostError("unknown session", operation) from None

    @staticmethod
    def _agent_spec(agent: object) -> object:
        from agm.agl.runtime.agents import decode_agent_value
        from agm.agl.semantics.values import RecordValue

        if not isinstance(agent, RecordValue):
            raise TypeError(f"session agent must be an RecordValue, got {type(agent).__name__}")
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
        from agm.agl.runtime.sessions import SessionAskError as AglSessionAskError

        raise AglSessionAskError(
            cause=error.cause,
            exit_code=error.exit_code,
            stderr_tail=error.stderr_tail,
            elapsed=error.elapsed,
            call_info=error.call_info,
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
        return CLI_SESSION_BACKENDS[type(agent).__name__](idle_timeout=idle_timeout)

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
        single_prompt: bool = False,
    ) -> str:
        """Open a backend session and return its host-generated handle id."""
        backend = self._backend_factory(agent, transport)
        backend.open(
            SessionOpenRequest(
                agent=agent, transport=transport, name=name, single_prompt=single_prompt
            )
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

    def reset_all(self) -> None:
        """Close and forget all sessions so the next default starts fresh.

        Successfully closed and previously closed entries are discarded. Entries
        whose close fails remain available for a later cleanup attempt.
        """
        self._close_every_entry("failed to reset one or more agent sessions", discard=True)
        self._default_handle = None

    def close_all(self) -> None:
        """Close every live backend, raising grouped failures after all attempts.

        Sessions are marked closed only after their backend closes successfully, so a
        later call can retry failed closes.
        """
        self._close_every_entry("failed to close one or more agent sessions", discard=False)

    def _close_every_entry(self, message: str, *, discard: bool) -> None:
        """Close every open entry, reporting all failures once the sweep finishes."""
        failures: list[Exception] = []
        for handle, entry in tuple(self._entries.items()):
            if not entry.closed:
                try:
                    self.close(handle)
                except Exception as error:
                    failures.append(error)
                    continue
            if discard:
                self._entries.pop(handle, None)
        if failures:
            raise ExceptionGroup(message, failures)

    def with_ephemeral(
        self,
        agent: object,
        transport: str,
        action: Callable[[str], _T],
        *,
        name: str = "",
        on_closed: Callable[[str], None] | None = None,
        single_prompt: bool = False,
    ) -> _T:
        """Run *action* in a short-lived session and release it afterward."""
        handle = self.open(agent, transport, name=name, ephemeral=True, single_prompt=single_prompt)

        def close() -> None:
            self.close(handle)
            if on_closed is not None:
                on_closed(handle)

        with preserve_primary_error(close, label="ephemeral session cleanup"):
            return action(handle)

    @staticmethod
    def _run_lifecycle(operation: str, action: Callable[[], _T]) -> _T:
        """Map backend transport failures to the public lifecycle error model."""
        try:
            return action()
        except SessionAskError as error:
            raise SessionHostError(
                f"session {operation} transport failed ({error.cause}): {error.stderr_tail}",
                operation,
            ) from error

    def _entry_for(self, handle: str, operation: str) -> _SessionEntry:
        entry = self._entries.get(handle)
        if entry is None:
            raise self._unknown_handle_error(handle, operation)
        if entry.closed:
            raise SessionHostError(f"session handle {handle!r} is closed", operation)
        return entry

    def _require_capability(self, entry: _SessionEntry, operation: SessionOperation) -> None:
        if not entry.backend.capabilities.supports(operation):
            raise SessionHostError(f"session backend does not support {operation}", operation)

    @staticmethod
    def _unknown_handle_error(handle: str, operation: str) -> SessionHostError:
        return SessionHostError(f"unknown session handle {handle!r}", operation)
