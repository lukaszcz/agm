"""Protocol and data exchanged by host-managed agent session backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from agm.agent.spec import AgentSpec, SessionTransport
from agm.agent.transport import AgentCallInfo, AgentTransportError


class SessionOperation(StrEnum):
    """Session operations, as named in lifecycle errors."""

    ASK = "ask"
    COMPACT = "compact"
    FORK = "fork"
    SET_NAME = "set-name"
    STATS = "stats"


@dataclass(frozen=True, slots=True)
class SessionOpenRequest:
    """The host-owned metadata supplied while opening a backend session.

    ``single_prompt`` states that this session serves exactly one prompt, so a
    backend need not establish a conversation it will never continue. Handle
    lifetime is owned separately by the session service.
    """

    agent: AgentSpec
    transport: SessionTransport
    name: str = ""
    single_prompt: bool = False


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


@dataclass(frozen=True, slots=True)
class SessionOperations:
    """The optional operations a backend implements natively; ``None`` is unsupported."""

    compact: Callable[[str], None] | None = None
    fork: Callable[[], SessionBackend] | None = None
    set_name: Callable[[str], None] | None = None
    stats: Callable[[], SessionStats] | None = None


class SessionBackend(Protocol):
    """One open native session implementation selected for an agent transport."""

    @property
    def operations(self) -> SessionOperations:
        """The optional operations this session supports."""

    def ask(self, request: SessionAskRequest) -> SessionAskResponse:
        """Send a prompt and return the backend response."""

    def reset(self) -> None:
        """Discard conversation history while retaining this backend object."""

    def close(self) -> None:
        """Release the backend's resources."""
