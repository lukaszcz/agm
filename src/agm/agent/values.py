"""Host-facing text syntax for ``Agent`` values."""

from __future__ import annotations

import re
from collections.abc import Callable

from agm.agent.spec import AgentClaude, AgentCodex, AgentPi, AgentSpec, payload_items

__all__ = ["AgentShorthandError", "agent_spec_shape", "parse_agent_shorthand"]

# Provider and effort characters; a model may also contain ``:``.
_SEGMENT = re.compile(r"[A-Za-z0-9._@+-]+")
_MODEL = re.compile(r"[A-Za-z0-9._@+:-]+")


class AgentShorthandError(ValueError):
    """Text opens a native-agent prefix (``claude/``, ``codex/``, ``pi/``) but breaks its form."""


def _model_effort(text: str) -> tuple[str, str] | None:
    """Split ``MODEL[:EFFORT]`` at the final colon (omitted effort: ``""``); ``None`` if invalid."""
    model, colon, effort = text.rpartition(":")
    if not colon:
        model, effort = text, ""
    if not _MODEL.fullmatch(model) or (colon and not _SEGMENT.fullmatch(effort)):
        return None
    return model, effort


def _provider_shorthand(parts: list[str]) -> AgentPi | None:
    """Read ``PROVIDER/MODEL[:EFFORT]`` segments as a Pi agent."""
    if len(parts) != 2 or not _SEGMENT.fullmatch(parts[0]):
        return None
    if (model_effort := _model_effort(parts[1])) is None:
        return None
    return AgentPi(parts[0], *model_effort)


def _model_shorthand(
    agent: Callable[[str, str], AgentSpec],
) -> Callable[[list[str]], AgentSpec | None]:
    """Return a reader of one ``MODEL[:EFFORT]`` segment into *agent*."""

    def read(parts: list[str]) -> AgentSpec | None:
        if len(parts) != 1 or (model_effort := _model_effort(parts[0])) is None:
            return None
        return agent(*model_effort)

    return read


# Native prefix (matched case-insensitively) -> (expected form, segment reader).
_NATIVE_SHORTHANDS: dict[str, tuple[str, Callable[[list[str]], AgentSpec | None]]] = {
    "claude": ("claude/MODEL[:EFFORT]", _model_shorthand(AgentClaude)),
    "codex": ("codex/MODEL[:EFFORT]", _model_shorthand(AgentCodex)),
    "pi": ("pi/PROVIDER/MODEL[:EFFORT]", _provider_shorthand),
}


def parse_agent_shorthand(text: str) -> AgentSpec | None:
    """Parse compact native-agent syntax, returning ``None`` when *text* is not shorthand.

    Forms: ``claude/MODEL[:EFFORT]``, ``codex/MODEL[:EFFORT]``,
    ``pi/PROVIDER/MODEL[:EFFORT]``, and any other ``PROVIDER/MODEL[:EFFORT]``
    (Pi). The effort follows the final colon, is opaque to AGM, and defaults
    to ``""``. Provider and effort are ``[A-Za-z0-9._@+-]+``; the model also
    admits ``:``. Text whose left-stripped first segment is a native prefix
    (any case) but which breaks that form -- surrounding whitespace included
    -- raises :class:`AgentShorthandError`.
    """
    stripped = text.lstrip()
    prefix, *parts = stripped.split("/")
    native = _NATIVE_SHORTHANDS.get(prefix.lower()) if parts else None
    if native is None:
        return _provider_shorthand(text.split("/"))
    form, read = native
    if stripped != text or (spec := read(parts)) is None:
        raise AgentShorthandError(f"malformed agent shorthand {text!r}; expected {form}")
    return spec


def agent_spec_shape(spec: AgentSpec) -> dict[str, object]:
    """Return the canonical external tagged-object shape for *spec*."""
    return {"$case": type(spec).__name__, **payload_items(spec)}
