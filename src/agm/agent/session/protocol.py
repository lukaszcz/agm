"""Protocol and data exchanged by host-managed agent session backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from agm.agent.spec import AgentSpec, PermissionMode, SessionTransport
from agm.agent.transport import AgentCallInfo, AgentOutputCallback, AgentTransportError
from agm.sandbox.prepare import SandboxContext
from agm.sandbox.request import SandboxLimits


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
    lifetime is owned separately by the session service. ``ephemeral`` marks a
    session that lives for one ask's retry loop, so a backend that cannot
    continue a conversation may serve each prompt as an independent invocation
    instead of rejecting the open. ``permission_mode``/
    ``sandbox``/``env`` fix the sandboxing and environment every process this
    session spawns runs under, for the session's whole lifetime; the
    ``permission_mode``/``sandbox`` defaults keep a caller that does not
    decode either unaffected. ``env`` is required: every session resolves an
    environment at open (the ambient one by default), so no backend ever
    falls back to the host process environment silently. An empty dict is an
    explicit empty environment, not "unspecified".
    """

    agent: AgentSpec
    transport: SessionTransport
    name: str = ""
    single_prompt: bool = False
    ephemeral: bool = False
    permission_mode: PermissionMode = PermissionMode.NONE
    sandbox: SandboxLimits | None = None
    env: dict[str, str] = field(repr=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class SessionAskRequest:
    """A rendered prompt to send within an existing backend session.

    Carries no sandboxing of its own: a session's ``permission_mode``/
    ``sandbox`` are fixed once, at open (see ``SessionOpenRequest``), for its
    whole lifetime -- there is no per-ask override.
    """

    prompt: str
    output_callback: AgentOutputCallback | None = field(default=None, repr=False, compare=False)


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

    continues_conversation: bool
    """Whether a later prompt continues the earlier ones; fixed by ``open``."""

    @property
    def operations(self) -> SessionOperations:
        """The optional operations this session supports."""

    def ask(self, request: SessionAskRequest) -> SessionAskResponse:
        """Send a prompt and return the backend response."""

    def reset(self) -> None:
        """Discard conversation history while retaining this backend object."""

    def close(self) -> None:
        """Release the backend's resources."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendSettings:
    """Settings fixed once, at ``open``, for a backend's whole lifetime.

    Every process a session spawns -- its first prompt and every later native
    call (fork's replacement, compaction, ...) -- reuses the same settings;
    there is no per-call override.
    """

    get_sandbox_context: Callable[[], SandboxContext]
    env: dict[str, str]
    idle_timeout: float | None = None
    permission_mode: PermissionMode = PermissionMode.NONE
    sandbox: SandboxLimits | None = None


class SandboxFixture:
    """Holds the :class:`BackendSettings` shared by every session backend family (CLI, RPC)."""

    def __init__(self, settings: BackendSettings) -> None:
        self._settings = settings
