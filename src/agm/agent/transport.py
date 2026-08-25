"""Host-neutral diagnostics shared by agent transport implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

AgentTransportFailureCause: TypeAlias = Literal[
    "spawn_failure", "timeout", "nonzero_exit", "interpolation_failure", "protocol_failure"
]


@dataclass(frozen=True, slots=True)
class AgentCallInfo:
    """Process details from one agent invocation, when a process was prepared."""

    argv: list[str]
    prompt_via_stdin: bool
    elapsed: float
    exit_code: int | None

    def to_trace(self) -> dict[str, object]:
        """Return the host-neutral call details in trace-record form."""
        return {
            "argv": self.argv,
            "prompt_via_stdin": self.prompt_via_stdin,
            "elapsed": self.elapsed,
            "exit_code": self.exit_code,
        }


def stderr_tail(stderr: str, *, max_chars: int = 500) -> str:
    """Return the bounded stderr diagnostic retained for an agent failure."""
    return stderr[-max_chars:] if len(stderr) > max_chars else stderr


class AgentTransportError(Exception):
    """A transport failure with diagnostics independent of a host's error model."""

    def __init__(
        self,
        *,
        cause: AgentTransportFailureCause,
        exit_code: int | None,
        stderr_tail: str,
        elapsed: float,
        call_info: AgentCallInfo,
    ) -> None:
        super().__init__(cause)
        self.cause = cause
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail
        self.elapsed = elapsed
        self.call_info = call_info
