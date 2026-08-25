"""Firewall-safe host protocol for persistent AgL agent sessions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, NoReturn, Protocol, TypeVar, runtime_checkable

from agm.agl.runtime.request import (
    AgentCallHostError,
    AgentCallInfo,
    AgentRequest,
    AgentResponse,
)
from agm.agl.semantics.values import RecordValue
from agm.core.cleanup import preserve_primary_error

if TYPE_CHECKING:
    from agm.agl.runtime.agents import AgentFn

__all__ = [
    "AgentDispatcherSessionHost",
    "EphemeralSessionHost",
    "SessionAgentError",
    "SessionAskError",
    "SessionHost",
    "SessionRequestHost",
    "SessionHostError",
    "SessionSnapshot",
    "SessionStats",
    "SessionTransport",
    "with_ephemeral_session",
]

_T = TypeVar("_T")


class SessionTransport:
    """Canonical transport labels carried by AgL ``Session`` values."""

    CLI = "Cli"
    RPC = "Rpc"


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """The opening identity the host associates with an opaque handle."""

    agent: RecordValue
    transport: str


@dataclass(frozen=True, slots=True)
class SessionStats:
    """Backend-neutral usage information for one session."""

    input_tokens: int
    output_tokens: int
    cost: Decimal
    context_percent: Decimal


class SessionHostError(Exception):
    """A lifecycle or capability failure exposed to the AgL evaluator."""

    def __init__(self, message: str, operation: str) -> None:
        self.message = message
        self.operation = operation
        super().__init__(message)


class SessionAgentError(SessionHostError):
    """An invalid agent value encountered while opening a session."""


class SessionAskError(Exception):
    """A transport failure from a session prompt."""

    def __init__(
        self,
        *,
        cause: str,
        exit_code: int | None,
        stderr_tail: str,
        elapsed: float,
        call_info: AgentCallInfo | None,
    ) -> None:
        super().__init__(cause)
        self.cause = cause
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail
        self.elapsed = elapsed
        self.call_info = call_info


@runtime_checkable
class SessionRequestHost(Protocol):
    """Optional session-host seam that preserves a one-shot request envelope.

    Hosts that adapt an ordinary agent dispatcher use this to retain the
    dispatcher-facing request metadata while the evaluator still owns the
    opaque session lifecycle.
    """

    def ask_request(self, handle: str, request: AgentRequest) -> AgentResponse: ...


@runtime_checkable
class EphemeralSessionHost(Protocol):
    """Optional host seam that owns an ephemeral session's complete scope."""

    def with_ephemeral(
        self,
        agent: RecordValue,
        transport: str,
        action: Callable[[str], _T],
        *,
        one_shot: bool = False,
    ) -> _T: ...


class SessionHost(Protocol):
    """Host-owned lifecycle service addressed by opaque AgL session ids."""

    def open(self, agent: RecordValue, transport: str, *, name: str = "") -> str: ...

    def open_ephemeral(
        self, agent: RecordValue, transport: str, *, one_shot: bool = False
    ) -> str: ...

    def default(self, agent: RecordValue, transport: str, *, name: str = "") -> str: ...

    def ask(self, handle: str, prompt: str) -> str: ...

    def compact(self, handle: str, instructions: str = "") -> None: ...

    def reset(self, handle: str) -> None: ...

    def fork(self, handle: str) -> str: ...

    def set_name(self, handle: str, name: str) -> None: ...

    def stats(self, handle: str) -> SessionStats: ...

    def snapshot(self, handle: str) -> SessionSnapshot: ...

    def close(self, handle: str) -> None: ...

    def close_all(self) -> None: ...


def with_ephemeral_session(
    host: SessionHost,
    agent: RecordValue,
    transport: str,
    action: Callable[[str], _T],
    *,
    one_shot: bool = False,
) -> _T:
    """Run *action* through one host ephemeral handle and always release it."""
    if isinstance(host, EphemeralSessionHost):
        if one_shot:
            return host.with_ephemeral(agent, transport, action, one_shot=True)
        return host.with_ephemeral(agent, transport, action)
    if one_shot:
        handle = host.open_ephemeral(agent, transport, one_shot=True)
    else:
        handle = host.open_ephemeral(agent, transport)
    with preserve_primary_error(
        lambda: host.close(handle), label="ephemeral agent session cleanup"
    ):
        return action(handle)


class AgentDispatcherSessionHost(SessionHost):
    """Dispatcher-backed session host used when no native session service exists.

    Its default handle snapshots an agent for the run, while each dispatch still
    uses the legacy one-shot dispatcher. Explicit Agent-method calls retain
    their short-lived lifecycle.
    """

    def __init__(self, dispatcher: "AgentFn | None") -> None:
        self._dispatcher = dispatcher
        self._sessions: dict[str, SessionSnapshot] = {}
        self._default_handle: str | None = None
        self._next_handle = 0

    @property
    def active_session_count(self) -> int:
        """Return the number of outstanding ephemeral sessions."""
        return len(self._sessions)

    def open(self, _agent: RecordValue, _transport: str, *, name: str = "") -> str:
        del name
        self._unavailable("open")

    def open_ephemeral(self, agent: RecordValue, transport: str, *, one_shot: bool = False) -> str:
        del one_shot
        handle = self._new_handle()
        self._sessions[handle] = SessionSnapshot(agent, transport)
        return handle

    def default(self, agent: RecordValue, transport: str, *, name: str = "") -> str:
        del name
        if self._default_handle is None:
            self._default_handle = self._new_handle()
            self._sessions[self._default_handle] = SessionSnapshot(agent, transport)
        return self._default_handle

    def ask(self, handle: str, prompt: str) -> str:
        request = AgentRequest(agent=self._agent_for(handle, "ask"), prompt=prompt)
        return self.ask_request(handle, request).content

    def ask_request(self, handle: str, request: AgentRequest) -> AgentResponse:
        self._agent_for(handle, "ask")
        if self._dispatcher is None:
            raise SessionAskError(
                cause="no_dispatcher",
                exit_code=None,
                stderr_tail="",
                elapsed=0.0,
                call_info=None,
            )
        try:
            from agm.agl.runtime.agents import dispatch_agent_value

            return dispatch_agent_value(request, self._dispatcher)
        except AgentCallHostError as error:
            raise SessionAskError(
                cause=error.cause,
                exit_code=error.exit_code,
                stderr_tail=error.stderr_tail,
                elapsed=error.elapsed,
                call_info=error.call_info,
            ) from error

    def compact(self, _handle: str, _instructions: str = "") -> None:
        self._unavailable("compact")

    def reset(self, _handle: str) -> None:
        self._unavailable("reset")

    def fork(self, _handle: str) -> str:
        self._unavailable("fork")

    def set_name(self, _handle: str, _name: str) -> None:
        self._unavailable("set-name")

    def stats(self, _handle: str) -> SessionStats:
        self._unavailable("stats")

    def snapshot(self, handle: str) -> SessionSnapshot:
        try:
            return self._sessions[handle]
        except KeyError:
            raise SessionHostError("unknown session", "snapshot") from None

    def close(self, handle: str) -> None:
        self._agent_for(handle, "close")
        del self._sessions[handle]

    def close_all(self) -> None:
        self._sessions.clear()

    def _new_handle(self) -> str:
        handle = f"ephemeral-{self._next_handle}"
        self._next_handle += 1
        return handle

    def _agent_for(self, handle: str, operation: str) -> RecordValue:
        try:
            return self._sessions[handle].agent
        except KeyError:
            raise SessionHostError("unknown session", operation) from None

    @staticmethod
    def _unavailable(operation: str) -> NoReturn:
        raise SessionHostError("persistent session host is unavailable", operation)
