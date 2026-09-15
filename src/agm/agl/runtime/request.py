"""Runtime request and response types for host-registered agent calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from agm.agent.transport import AgentCallInfo
from agm.agl.ir.ids import Location
from agm.agl.semantics.values import RecordValue

if TYPE_CHECKING:
    from agm.agl.runtime.contract import OutputContract, TypelessOutputContract

ValidationErrorCategory = Literal[
    "missing_field", "unknown_field", "wrong_type", "bad_case", "invalid_json"
]

_VALIDATION_SUMMARIES: dict[ValidationErrorCategory, str] = {
    "missing_field": "The response is missing required data.",
    "unknown_field": "The response contains fields not permitted by the schema.",
    "wrong_type": "The response contains a value with an incorrect type.",
    "bad_case": "The response does not select a valid case.",
    "invalid_json": "The response is not valid JSON.",
}
_DEFAULT_VALIDATION_SUMMARY = "The response does not match the required output format."


class AgentCancelled(Exception):
    """Signal that a host agent call was interrupted."""

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

    agent: RecordValue
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


def compose_initial_agent_prompt(request: AgentRequest) -> str:
    """Compose the initial prompt and its optional output-format instructions."""
    parts = [request.prompt]
    if request.output_contract is not None and request.output_contract.format_instructions:
        parts.append(request.output_contract.format_instructions)
    return "\n\n".join(parts)


def _validation_summary(errors: list[ValidationError]) -> str:
    """Summarize validation categories without exposing response-derived details."""
    categories: set[ValidationErrorCategory] = set()
    summaries: list[str] = []
    for error in errors:
        if error.category not in categories:
            categories.add(error.category)
            summaries.append(_VALIDATION_SUMMARIES[error.category])
    if not summaries:
        summaries.append(_DEFAULT_VALIDATION_SUMMARY)
    return "\n".join(f"- {summary}" for summary in summaries)


def _compose_corrective_feedback(
    request: AgentRequest, *, include_previous_output: bool, include_intro: bool
) -> str:
    """Compose corrective feedback with a sanitized validation summary."""
    parts = [f"Validation errors:\n{_validation_summary(request.validation_errors)}"]
    if include_intro:
        parts.insert(0, "Your previous response did not match the required output format.")
    if include_previous_output:
        parts.append(f"Previous response:\n{request.previous_invalid_output or ''}")
    parts.append("Return only valid JSON matching the schema.")
    return "\n\n".join(parts)


def compose_corrective_follow_up(request: AgentRequest) -> str:
    """Compose one-shot retry feedback, including the failed response."""
    return _compose_corrective_feedback(request, include_previous_output=True, include_intro=True)


def compose_session_corrective_follow_up(request: AgentRequest) -> str:
    """Compose session retry feedback without replaying conversation content."""
    return _compose_corrective_feedback(request, include_previous_output=False, include_intro=False)


def compose_agent_prompt(request: AgentRequest) -> str:
    """Compose a complete one-shot agent prompt, including retry feedback."""
    initial = compose_initial_agent_prompt(request)
    if not request.attempt:
        return initial
    return "\n\n".join((initial, compose_corrective_follow_up(request)))
