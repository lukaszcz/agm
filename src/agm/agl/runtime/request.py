"""Runtime request and response types for host-registered agent calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from agm.agl.ir.ids import Location
from agm.agl.semantics.values import EnumValue

if TYPE_CHECKING:
    from agm.agl.runtime.contract import OutputContract, TypelessOutputContract

ValidationErrorCategory = Literal[
    "missing_field", "unknown_field", "wrong_type", "bad_case", "invalid_json"
]


class AgentCancelled(Exception):
    """Signal that a host agent call was declined or interrupted."""

    def __init__(self, callee: str, reason: str, *, span: Location | None = None) -> None:
        super().__init__(f"Agent call to {callee!r} cancelled ({reason}).")
        self.callee = callee
        self.reason = reason
        self.span = span


@dataclass(frozen=True, slots=True)
class ValidationError:
    """A structured JSON-output validation error used for retry feedback."""

    category: ValidationErrorCategory
    message: str
    path: str = "$"
    field: str | None = None

    def to_json_obj(self) -> dict[str, object]:
        return {
            "category": self.category,
            "message": self.message,
            "path": self.path,
            "field": self.field,
        }


@dataclass(frozen=True, slots=True)
class AgentCallInfo:
    """Transport details from one agent invocation, when a process was prepared."""

    argv: list[str]
    prompt_via_stdin: bool
    elapsed: float
    exit_code: int | None

    def to_trace(self) -> dict[str, object]:
        return {
            "argv": self.argv,
            "prompt_via_stdin": self.prompt_via_stdin,
            "elapsed": self.elapsed,
            "exit_code": self.exit_code,
        }


class AgentCallHostError(Exception):
    """Python-level transport failure mapped to catchable ``AgentCallError``."""

    def __init__(
        self,
        *,
        cause: str,
        exit_code: int | None,
        stderr_tail: str,
        elapsed: float,
        call_info: AgentCallInfo | None = None,
    ) -> None:
        super().__init__(cause)
        self.cause = cause
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail
        self.elapsed = elapsed
        self.call_info = call_info


@dataclass(slots=True)
class AgentRequest:
    """The fully composed request passed verbatim to a host dispatcher."""

    agent: EnumValue
    prompt: str
    attempt: int = 0
    previous_invalid_output: str | None = None
    validation_errors: list[ValidationError] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    output_contract: "OutputContract | TypelessOutputContract | None" = None


@dataclass(slots=True)
class AgentResponse:
    """A host agent response, optionally including process-call information."""

    content: str
    metadata: dict[str, object] = field(default_factory=dict)
    call_info: AgentCallInfo | None = None


def compose_agent_prompt(request: AgentRequest) -> str:
    """Compose the exact prompt sent to the agent for one attempt."""
    parts = [request.prompt]
    if request.output_contract is not None and request.output_contract.format_instructions:
        parts.append(request.output_contract.format_instructions)
    if request.attempt:
        errors = "\n".join(f"- {error.message}" for error in request.validation_errors) or "(none)"
        parts.append(
            "Your previous response did not match the required output format.\n\n"
            f"Validation errors:\n{errors}\n\nPrevious response:\n"
            f"{request.previous_invalid_output or ''}\n\n"
            "Return only valid JSON matching the schema."
        )
    return "\n\n".join(parts)
