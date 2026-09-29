"""Host-facing text syntax for ``Agent`` values."""

from __future__ import annotations

import re
from collections.abc import Callable

from agm.agent.spec import (
    NATIVE_AGENT_SPECS,
    AgentPi,
    AgentSpec,
    NativeAgentSpec,
    model_fields,
    payload_items,
)

__all__ = ["AgentShorthandError", "agent_spec_shape", "parse_agent_shorthand"]

# Provider and effort characters; a model may also contain ``:``, ``[`` and ``]``
# (``opus[1m]``).
_SEGMENT = re.compile(r"[A-Za-z0-9._@+-]+")
_MODEL = re.compile(r"[A-Za-z0-9._@+:\[\]-]+")


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


def _shorthand_reader(
    spec_cls: type[NativeAgentSpec],
) -> Callable[[list[str]], NativeAgentSpec | None]:
    """Return a reader of *spec_cls*'s ``[PROVIDER/]MODEL[:EFFORT]`` segments."""
    levels = len(model_fields(spec_cls))

    def read(parts: list[str]) -> NativeAgentSpec | None:
        *names, last = parts
        if len(parts) != levels or not all(_SEGMENT.fullmatch(name) for name in names):
            return None
        if (model_effort := _model_effort(last)) is None:
            return None
        names.extend(model_effort)
        return spec_cls(*names)

    return read


def _shorthand_form(spec_cls: type[NativeAgentSpec]) -> str:
    """The expected-form text for *spec_cls*, e.g. ``pi/PROVIDER/MODEL[:EFFORT]``."""
    return "/".join((spec_cls.CLI_NAME, *map(str.upper, model_fields(spec_cls)))) + "[:EFFORT]"


# Native prefix (matched case-insensitively) -> (expected form, segment reader).
_NATIVE_SHORTHANDS: dict[str, tuple[str, Callable[[list[str]], NativeAgentSpec | None]]] = {
    spec_cls.CLI_NAME: (_shorthand_form(spec_cls), _shorthand_reader(spec_cls))
    for spec_cls in NATIVE_AGENT_SPECS
}

# Any other ``PROVIDER/MODEL[:EFFORT]`` text reads as a Pi agent.
_provider_shorthand = _shorthand_reader(AgentPi)


def parse_agent_shorthand(text: str) -> AgentSpec | None:
    """Parse compact native-agent syntax, returning ``None`` when *text* is not shorthand.

    Forms: ``claude/MODEL[:EFFORT]``, ``codex/MODEL[:EFFORT]``,
    ``pi/PROVIDER/MODEL[:EFFORT]``, and any other ``PROVIDER/MODEL[:EFFORT]``
    (Pi). The effort follows the final colon, is opaque to AGM, and defaults
    to ``""``. Provider and effort are ``[A-Za-z0-9._@+-]+``; the model also
    admits ``:``, ``[`` and ``]``. Text whose left-stripped first segment is a
    native prefix (any case) but which breaks that form -- surrounding
    whitespace included -- raises :class:`AgentShorthandError`.
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
