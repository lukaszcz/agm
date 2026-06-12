"""AgentRequest, AgentResponse and related runtime request types.

These are the objects passed to host-registered agent callables.
``AgentRequest.prompt`` is the already-rendered prompt template (the rendered
text that the agent should receive as its user message).

Design §7.5 shape — M1 minimal fields; M2 adds ``output_contract`` so agents
can inspect format instructions and the JSON schema for native structured
output (design §7.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.agl.runtime.contract import OutputContract


@dataclass(slots=True)
class AgentRequest:
    """The request object passed to a host-registered agent callable.

    ``agent``
        The agent name as it appears in the AgL source: ``"prompt"`` for the
        built-in default agent, or the registered name for named agents.
    ``prompt``
        The fully rendered user-authored prompt template.  Interpolated
        values have already been processed by the renderer pipeline.  The
        agent should use this verbatim as its user message.
    ``attempt``
        0-based attempt counter (0 = first call, 1 = first retry, …).
    ``previous_invalid_output``
        The raw text returned by the previous (failed) attempt, or ``None``
        on the first attempt.  Useful for retry-feedback messages (M4+).
    ``output_contract``
        The materialized output contract for this call site (design §7.5).
        Carries ``format_instructions`` and ``json_schema`` so agents can
        relay them to the underlying model.  ``None`` for untyped text calls
        that have no explicit contract (rare; normally always set in M2+).
    """

    agent: str
    prompt: str
    attempt: int = 0
    previous_invalid_output: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    output_contract: "OutputContract | None" = None


@dataclass(slots=True)
class AgentResponse:
    """A structured response from a host agent callable.

    A host agent may return either a plain ``str`` (treated as
    ``AgentResponse(content=value, metadata={})``) or an ``AgentResponse``
    directly.
    """

    content: str
    metadata: dict[str, object] = field(default_factory=dict)
