"""Host-facing text syntax for ``Agent`` values."""

from __future__ import annotations

from agm.agent.spec import AgentClaude, AgentCodex, AgentCommand, AgentPi, AgentSpec

__all__ = ["agent_spec_shape", "parse_agent_shorthand"]


def _model_effort(text: str) -> tuple[str, str] | None:
    """Split a non-empty model and opaque effort suffix at the final hyphen."""
    model, separator, effort = text.rpartition("-")
    if not separator or not model or not effort:
        return None
    return model, effort


def parse_agent_shorthand(text: str) -> AgentSpec | None:
    """Parse compact native-agent syntax, returning ``None`` when it does not match.

    Exact ``claude/`` and ``codex/`` prefixes select those CLIs. ``pi/`` takes
    an explicit provider, while any other ``provider/model-effort`` spelling
    defaults to Pi. The effort is the non-empty final hyphen suffix and remains
    opaque to AGM.
    """
    parts = text.split("/")
    if parts[0] == "claude":
        if len(parts) != 2 or (model_effort := _model_effort(parts[1])) is None:
            return None
        return AgentClaude(*model_effort)
    if parts[0] == "codex":
        if len(parts) != 2 or (model_effort := _model_effort(parts[1])) is None:
            return None
        return AgentCodex(*model_effort)
    if parts[0] == "pi":
        if len(parts) != 3 or not parts[1] or (model_effort := _model_effort(parts[2])) is None:
            return None
        model, effort = model_effort
        return AgentPi(parts[1], model, effort)
    if len(parts) == 2 and parts[0] and (model_effort := _model_effort(parts[1])) is not None:
        model, effort = model_effort
        return AgentPi(parts[0], model, effort)
    return None


def agent_spec_shape(spec: AgentSpec) -> dict[str, object]:
    """Return the canonical external tagged-object shape for *spec*."""
    if isinstance(spec, AgentCommand):
        return {"$case": "AgentCommand", "command": spec.command}
    if isinstance(spec, AgentClaude):
        return {"$case": "AgentClaude", "model": spec.model, "thinking": spec.thinking}
    if isinstance(spec, AgentCodex):
        return {"$case": "AgentCodex", "model": spec.model, "thinking": spec.thinking}
    return {
        "$case": "AgentPi",
        "provider": spec.provider,
        "model": spec.model,
        "thinking": spec.thinking,
    }
