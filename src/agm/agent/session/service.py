"""Host-owned session table and lifecycle service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from agm.agent.session.protocol import (
    SessionAskRequest,
    SessionAskResponse,
    SessionBackend,
    SessionHostError,
    SessionOpenRequest,
    SessionOperation,
    SessionStats,
)

SessionBackendFactory = Callable[[object, str], SessionBackend]


@dataclass(slots=True)
class _SessionEntry:
    """The host state associated with one opaque session handle."""

    backend: SessionBackend
    agent: object
    transport: str
    closed: bool = False


class SessionService:
    """Own live session backends behind AGM-generated opaque handle ids."""

    def __init__(self, backend_factory: SessionBackendFactory) -> None:
        self._backend_factory = backend_factory
        self._entries: dict[str, _SessionEntry] = {}
        self._default_handle: str | None = None

    def open(self, agent: object, transport: str, *, name: str = "") -> str:
        """Open a backend session and return its host-generated handle id."""
        backend = self._backend_factory(agent, transport)
        backend.open(SessionOpenRequest(agent=agent, transport=transport, name=name))
        handle = str(uuid4())
        self._entries[handle] = _SessionEntry(
            backend=backend,
            agent=agent,
            transport=transport,
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
        entry.backend.compact(instructions)

    def reset(self, handle: str) -> None:
        """Reset a live session while retaining its handle, agent, and transport."""
        entry = self._entry_for(handle, "reset")
        entry.backend.reset()

    def fork(self, handle: str) -> str:
        """Fork a live session, returning a distinct host handle for the child."""
        entry = self._entry_for(handle, SessionOperation.FORK)
        self._require_capability(entry, SessionOperation.FORK)
        forked_backend = entry.backend.fork()
        forked_handle = str(uuid4())
        self._entries[forked_handle] = _SessionEntry(
            backend=forked_backend,
            agent=entry.agent,
            transport=entry.transport,
        )
        return forked_handle

    def set_name(self, handle: str, name: str) -> None:
        """Set a live session's backend-visible name."""
        entry = self._entry_for(handle, SessionOperation.SET_NAME)
        self._require_capability(entry, SessionOperation.SET_NAME)
        entry.backend.set_name(name)

    def stats(self, handle: str) -> SessionStats:
        """Return usage statistics for a live session."""
        entry = self._entry_for(handle, SessionOperation.STATS)
        self._require_capability(entry, SessionOperation.STATS)
        return entry.backend.stats()

    def close(self, handle: str) -> None:
        """Close a live session; closing an already closed known handle is a no-op."""
        entry = self._entries.get(handle)
        if entry is None:
            raise self._unknown_handle_error(handle, "close")
        if entry.closed:
            return
        entry.backend.close()
        entry.closed = True

    def close_all(self) -> None:
        """Close every live backend, raising grouped failures after all attempts.

        Sessions are marked closed only after their backend closes successfully, so a
        later call can retry failed closes.
        """
        failures: list[Exception] = []
        for entry in self._entries.values():
            if entry.closed:
                continue
            try:
                entry.backend.close()
            except Exception as error:
                failures.append(error)
            else:
                entry.closed = True
        if failures:
            raise ExceptionGroup("failed to close one or more agent sessions", failures)

    def ask_ephemeral(
        self,
        agent: object,
        transport: str,
        request: SessionAskRequest,
        *,
        name: str = "",
    ) -> SessionAskResponse:
        """Ask through a short-lived session and close it even when asking fails."""
        handle = self.open(agent, transport, name=name)
        try:
            response = self.ask(handle, request)
        except BaseException:
            try:
                self.close(handle)
            except BaseException:
                pass
            raise
        self.close(handle)
        return response

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
