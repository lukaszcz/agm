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
    """Text opens a native agent name (``claude``, ``codex``, ``pi``) but breaks its form."""


def _split_effort(text: str) -> tuple[str, str | None]:
    """Split ``NAMES[:EFFORT]`` at the final colon; the effort is ``None`` when omitted."""
    names, colon, effort = text.rpartition(":")
    return (names, effort) if colon else (text, None)


def _shorthand_reader(
    spec_cls: type[NativeAgentSpec],
) -> Callable[[list[str], str | None], NativeAgentSpec | None]:
    """Return a reader of *spec_cls*'s leading name segments and effort; ``None`` if invalid.

    Omitted trailing names and effort are ``""``; only the last name field (the model) is
    read as a model.
    """
    fields = model_fields(spec_cls)
    patterns = (*(_SEGMENT,) * (len(fields) - 1), _MODEL)

    def read(names: list[str], effort: str | None) -> NativeAgentSpec | None:
        if len(names) > len(fields) or (effort is not None and not _SEGMENT.fullmatch(effort)):
            return None
        if not all(pattern.fullmatch(name) for pattern, name in zip(patterns, names)):
            return None
        padded = [*names, *[""] * (len(fields) - len(names)), effort or ""]
        return spec_cls(*padded)

    return read


def _shorthand_form(spec_cls: type[NativeAgentSpec]) -> str:
    """The expected-form text for *spec_cls*, e.g. ``pi[/PROVIDER[/MODEL]][:EFFORT]``."""
    fields = model_fields(spec_cls)
    names = "".join(f"[/{field.upper()}" for field in fields) + "]" * len(fields)
    return f"{spec_cls.CLI_NAME}{names}[:EFFORT]"


# Native name (matched case-insensitively) -> (expected form, reader).
_NATIVE_SHORTHANDS: dict[
    str, tuple[str, Callable[[list[str], str | None], NativeAgentSpec | None]]
] = {
    spec_cls.CLI_NAME: (_shorthand_form(spec_cls), _shorthand_reader(spec_cls))
    for spec_cls in NATIVE_AGENT_SPECS
}

# A native name ending the text or followed by ``/`` or ``:``.
_NATIVE_NAME = re.compile(
    "(?:" + "|".join(map(re.escape, _NATIVE_SHORTHANDS)) + r")(?=[/:]|\Z)",
    re.IGNORECASE | re.ASCII,
)

# Any other ``PROVIDER/MODEL[:EFFORT]`` text reads as a Pi agent.
_provider_shorthand = _shorthand_reader(AgentPi)
_PROVIDER_LEVELS = len(model_fields(AgentPi))


def parse_agent_shorthand(text: str) -> AgentSpec | None:
    """Parse compact native-agent syntax, returning ``None`` when *text* is not shorthand.

    Forms: ``claude[/MODEL][:EFFORT]``, ``codex[/MODEL][:EFFORT]``,
    ``pi[/PROVIDER[/MODEL]][:EFFORT]``, and any other ``PROVIDER/MODEL[:EFFORT]``
    (Pi). Omitted native names and effort are ``""``. The effort follows the final
    colon and is opaque to AGM. Provider and effort are ``[A-Za-z0-9._@+-]+``; the
    model also admits ``:``, ``[`` and ``]``. Native text is stripped text starting
    with a native name (ASCII, any case) followed by its end, ``/`` or ``:``; native
    text breaking its form -- surrounding whitespace included -- raises
    :class:`AgentShorthandError`.
    """
    stripped = text.strip()
    native = _NATIVE_NAME.match(stripped)
    if native is None:
        names, effort = _split_effort(text)
        segments = names.split("/")
        return _provider_shorthand(segments, effort) if len(segments) == _PROVIDER_LEVELS else None
    form, read = _NATIVE_SHORTHANDS[native.group().lower()]
    names, effort = _split_effort(stripped[native.end() :])
    head, *segments = names.split("/")
    if stripped != text or head or (spec := read(segments, effort)) is None:
        raise AgentShorthandError(f"malformed agent shorthand {text!r}; expected {form}")
    return spec


def agent_spec_shape(spec: AgentSpec) -> dict[str, object]:
    """Return the canonical external tagged-object shape for *spec*."""
    return {"$case": type(spec).__name__, **payload_items(spec)}
