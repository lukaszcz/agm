"""AgentRequest, AgentResponse and related runtime request types.

These are the objects passed to host-registered agent callables.
``AgentRequest.prompt`` is the already-rendered prompt template (the rendered
text that the agent should receive as its user message).

Design §7.5 / §7.6 shape: ``AgentRequest`` carries the rendered prompt,
attempt counter, retry context, and an ``OutputContract`` so agents can
inspect format instructions and the JSON schema for native structured output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from agm.agl.ir.ids import Location

if TYPE_CHECKING:
    from agm.agl.runtime.contract import OutputContract, TypelessOutputContract


# The documented validation-error categories (design §7.5, extended for F4):
# - ``missing_field``  — a required field was absent.
# - ``unknown_field``  — an undeclared field was present (``additionalProperties``).
# - ``wrong_type``     — a field's JSON type did not match the schema.
# - ``bad_case``       — an enum object's ``$case`` did not name a known variant
#                        (or was missing / not a string).
# - ``invalid_json``   — the agent response contained no extractable JSON value
#                        at all (design §7.5 extension, F4: an honest category for
#                        totally-unparseable output so the reason is fed back on
#                        the next retry attempt rather than being silently dropped).
ValidationErrorCategory = Literal[
    "missing_field",
    "unknown_field",
    "wrong_type",
    "bad_case",
    "invalid_json",
]


class AgentCancelled(Exception):
    """Signal that a host agent call was declined or interrupted."""

    def __init__(
        self,
        callee: str,
        reason: str,
        *,
        span: Location | None = None,
    ) -> None:
        super().__init__(f"Agent call to {callee!r} cancelled ({reason}).")
        self.callee = callee
        self.reason = reason
        self.span = span


@dataclass(frozen=True, slots=True)
class ValidationError:
    """A structured parse/validation error (design §7.5 / §7.7).

    Produced by the JSON codec when an agent response parses as JSON but fails
    strict schema validation.  Carries enough structure that retry feedback and
    ``AgentParseError.validation_errors`` can describe *what* went wrong without
    leaking jsonschema-internal phrasing (e.g. "is not valid under any of the
    given schemas").

    ``category``
        One of the documented categories (see :data:`ValidationErrorCategory`).
    ``message``
        A human-readable, type-directed description of the failure.
    ``path``
        A JSON-path-like location of the offending value (``"$"`` for the root,
        ``"$.field"`` for a record field, etc.).
    ``field``
        The offending field name when applicable (``None`` for root-level or
        ``$case`` failures).
    """

    category: ValidationErrorCategory
    message: str
    path: str = "$"
    field: str | None = None

    def to_json_obj(self) -> dict[str, object]:
        """JSON-shaped representation (for tracing / retry-feedback prompts)."""
        return {
            "category": self.category,
            "message": self.message,
            "path": self.path,
            "field": self.field,
        }


@dataclass(slots=True)
class AgentRequest:
    """The request object passed to a host-registered agent callable.

    ``agent``
        The agent name as it appears in the AgL source: ``"ask"`` for the
        built-in default agent, or the registered name for named agents.
    ``prompt``
        The fully rendered user-authored prompt template.  Interpolated
        values have already been processed by the renderer pipeline.  The
        agent should use this verbatim as its user message.
    ``attempt``
        0-based attempt counter (0 = first call, 1 = first retry, …).
    ``previous_invalid_output``
        The raw text returned by the previous (failed) attempt, or ``None``
        on the first attempt. Useful for retry-feedback messages.
    ``validation_errors``
        Structured :class:`ValidationError` records from the previous failed
        attempt (design §7.5 / §7.8).  Empty on the first attempt; populated on
        retries so the agent can be told *what* was wrong.
    ``output_contract``
        The materialized output contract for this call site (design §7.5).
        Carries ``format_instructions`` and ``json_schema`` so agents can
        relay them to the underlying model.  ``None`` for ``unit`` calls,
        whose response is intentionally ignored.
    """

    agent: str
    prompt: str
    attempt: int = 0
    previous_invalid_output: str | None = None
    validation_errors: list[ValidationError] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    output_contract: "OutputContract | TypelessOutputContract | None" = None


@dataclass(slots=True)
class AgentResponse:
    """A structured response from a host agent callable.

    A host agent may return either a plain ``str`` (treated as
    ``AgentResponse(content=value, metadata={})``) or an ``AgentResponse``
    directly.
    """

    content: str
    metadata: dict[str, object] = field(default_factory=dict)
