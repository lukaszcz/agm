"""Immutable host-capability catalog for the AgL static pipeline.

``HostCapabilities`` is a frozen, data-only dataclass that describes which
agents, codecs, and renderers the host has registered.  It is constructed by
``WorkflowRuntime.run()`` before the static passes execute and is consumed by
the type checker (Component 5) — the checker never imports agent/codec/renderer
*implementations*, only their capability descriptors.

Design
------
- ``agent_names``: the set of explicitly registered named agents.
- ``has_fallback_agent``: when ``True`` an implicit fallback handles any
  unregistered agent name, so the checker accepts unknown names.  When
  ``False`` an unrecognised agent name is a static error.
- ``codec_kinds``: mapping from codec name → frozenset of semantic type-kind
  strings the codec supports.  In v1 the only registered codec is ``"text"``
  supporting ``{"text"}``; the ``"json"`` codec (M2) adds
  ``{"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}``.
- ``renderer_names``: frozenset of known renderer names (``"default"``,
  ``"raw"`` in v1; extended in M5).

The string type-kind identifiers used in ``codec_kinds`` match the names of
the semantic ``Type`` subclasses in ``agm.agl.typecheck.types`` (lower-cased,
with the ``"Type"`` suffix stripped).  For example ``TextType`` → ``"text"``,
``RecordType`` → ``"record"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class HostCapabilities:
    """Immutable capability catalog consumed by the type-checker.

    Parameters
    ----------
    agent_names:
        Names of explicitly registered agents (does not include ``"prompt"``
        or ``"exec"`` — those are built-ins handled separately).
    has_fallback_agent:
        When ``True``, any agent name is accepted regardless of whether it
        appears in ``agent_names``.
    codec_kinds:
        Mapping from codec name to the frozenset of semantic type-kind strings
        the codec can handle.  Type-kind strings are the lower-cased class name
        without the ``"Type"`` suffix (e.g. ``"text"``, ``"json"``, ``"int"``).
    renderer_names:
        The set of renderer names the host has registered.  The checker verifies
        that every explicit ``as <name>`` renderer reference in an interpolation
        segment names a renderer in this set.
    """

    agent_names: frozenset[str] = field(default_factory=frozenset)
    has_fallback_agent: bool = False
    codec_kinds: dict[str, frozenset[str]] = field(default_factory=dict)
    renderer_names: frozenset[str] = field(default_factory=frozenset)


def default_capabilities() -> HostCapabilities:
    """Return the v1 default capabilities (text codec + default/raw renderers).

    This is the baseline used by ``WorkflowRuntime`` when no codecs or
    renderers have been explicitly registered.  M2 extends this with the JSON
    codec and additional renderers.
    """
    return HostCapabilities(
        agent_names=frozenset(),
        has_fallback_agent=True,
        codec_kinds={
            "text": frozenset({"text"}),
        },
        renderer_names=frozenset({"default", "raw"}),
    )
