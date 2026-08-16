"""Firewall-safe host protocol for persistent AgL agent sessions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from agm.agl.runtime.request import AgentCallInfo
from agm.agl.semantics.values import EnumValue

__all__ = [
    "SessionAskError",
    "SessionHost",
    "SessionHostError",
    "SessionSnapshot",
    "SessionStats",
    "SessionTransport",
]


class SessionTransport:
    """Canonical transport labels carried by AgL ``Session`` values."""

    CLI = "Cli"
    RPC = "Rpc"


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """The opening identity the host associates with an opaque handle."""

    agent: EnumValue
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


class SessionHost(Protocol):
    """Host-owned lifecycle service addressed by opaque AgL session ids."""

    def open(self, agent: EnumValue, transport: str, *, name: str = "") -> str: ...

    def default(self, agent: EnumValue, transport: str, *, name: str = "") -> str: ...

    def ask(self, handle: str, prompt: str) -> str: ...

    def compact(self, handle: str, instructions: str = "") -> None: ...

    def reset(self, handle: str) -> None: ...

    def fork(self, handle: str) -> str: ...

    def set_name(self, handle: str, name: str) -> None: ...

    def stats(self, handle: str) -> SessionStats: ...

    def snapshot(self, handle: str) -> SessionSnapshot: ...

    def close(self, handle: str) -> None: ...

    def close_all(self) -> None: ...
