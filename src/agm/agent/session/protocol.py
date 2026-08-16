"""Protocol and data exchanged by host-managed agent session backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from agm.agent.transport import AgentCallInfo, AgentTransportError


class SessionOperation(StrEnum):
    """Operations a session backend can advertise as supported."""

    ASK = "ask"
    COMPACT = "compact"
    FORK = "fork"
    SET_NAME = "set-name"
    STATS = "stats"


@dataclass(frozen=True, slots=True)
class SessionCapabilities:
    """The optional operations implemented natively by a session backend."""

    operations: frozenset[SessionOperation]

    @classmethod
    def all(cls) -> SessionCapabilities:
        """Return capabilities for a backend implementing every optional operation."""
        return cls(frozenset(SessionOperation))

    def supports(self, operation: SessionOperation) -> bool:
        """Whether this backend supports *operation*."""
        return operation in self.operations

    def __sub__(self, operations: set[SessionOperation]) -> SessionCapabilities:
        """Return these capabilities without *operations*."""
        return SessionCapabilities(self.operations - operations)


@dataclass(frozen=True, slots=True)
class SessionOpenRequest:
    """The host-owned metadata supplied while opening a backend session.

    ``one_shot`` selects a backend's legacy one-shot invocation. Handle
    lifetime is owned separately by the session service.
    """

    agent: object
    transport: str
    name: str = ""
    one_shot: bool = False


@dataclass(frozen=True, slots=True)
class SessionAskRequest:
    """A rendered prompt to send within an existing backend session."""

    prompt: str


@dataclass(frozen=True, slots=True)
class SessionAskResponse:
    """The backend response to one session prompt and its call details."""

    content: str
    metadata: dict[str, object] = field(default_factory=dict)
    call_info: AgentCallInfo | None = None


@dataclass(frozen=True, slots=True)
class SessionStats:
    """Usage information reported by a session backend."""

    input_tokens: int
    output_tokens: int
    cost: Decimal
    context_percent: Decimal


class SessionHostError(Exception):
    """A lifecycle or capability failure raised by the host session service."""

    def __init__(self, message: str, operation: str) -> None:
        self.message = message
        self.operation = operation
        super().__init__(message)


class SessionAgentError(SessionHostError):
    """An invalid agent configuration rejected while opening a session."""


class SessionAskError(AgentTransportError):
    """A backend-neutral transport failure from one session ``ask`` call.

    Unlike :class:`SessionHostError`, this retains process diagnostics for the
    host effect boundary that maps a failed agent request to its user-facing
    error model.
    """


class SessionBackend(Protocol):
    """One native session implementation selected for an agent transport."""

    capabilities: SessionCapabilities

    def open(self, request: SessionOpenRequest) -> None:
        """Open the backend's underlying session."""

    def ask(self, request: SessionAskRequest) -> SessionAskResponse:
        """Send a prompt and return the backend response."""

    def compact(self, instructions: str) -> None:
        """Compact the conversation using optional instructions."""

    def reset(self) -> None:
        """Discard conversation history while retaining this backend object."""

    def fork(self) -> SessionBackend:
        """Return a new backend whose history starts from this session."""

    def set_name(self, name: str) -> None:
        """Assign a backend-visible session name."""

    def stats(self) -> SessionStats:
        """Return the backend's current usage statistics."""

    def close(self) -> None:
        """Release the backend's resources."""
