"""Immutable host-capability catalog for the AgL static pipeline.

``HostCapabilities`` is a frozen, data-only dataclass that describes which
agents and codecs the host has registered.  It is constructed by
``WorkflowRuntime.run()`` before the static passes execute and is consumed by
the type checker (Component 5) — the checker never imports agent/codec
*implementations*, only their capability descriptors.

Design
------
- ``agent_names``: the set of named agents the host can *back* (the registered
  backings).  This is NOT the set of valid names: name validity is owned by the
  scope pass — an undeclared named agent is a scope binding error.  The runtime
  cross-checks ``agent_names`` against the source-declared set.
- ``has_default_agent``: when ``True`` the host has a default agent that backs
  the built-in ``prompt`` keyword.  When ``False`` a ``prompt`` call is a static
  error.
- ``codec_kinds``: mapping from codec name → frozenset of semantic type-kind
  strings the codec supports.  Built-in codecs: ``"text"`` (supports
  ``{"text"}``); ``"json"`` (supports
  ``{"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}``).
  Hosts may register additional codecs via ``WorkflowRuntime``.

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
        Names of agents the host can *back* (does not include ``"prompt"``
        or ``"exec"`` — those are built-ins handled separately).  This is the
        set of host-supplied backings, not the set of valid names: scope owns
        name validity (an undeclared named agent is a binding error).
    has_default_agent:
        When ``True``, a default agent backs the built-in ``prompt`` keyword.
        When ``False``, a ``prompt`` call is a static error.
    supports_shell_exec:
        When ``True``, the host can execute ``exec`` (shell) calls.  When
        ``False``, any ``exec`` call site is a static error.  ``WorkflowRuntime``
        sets this to ``True``; test harnesses that do not want shell execution
        may set it to ``False``.
    codec_kinds:
        Mapping from codec name to the frozenset of semantic type-kind strings
        the codec can handle.  Type-kind strings are the lower-cased class name
        without the ``"Type"`` suffix (e.g. ``"text"``, ``"json"``, ``"int"``).
    """

    agent_names: frozenset[str] = field(default_factory=frozenset)
    has_default_agent: bool = False
    supports_shell_exec: bool = False
    codec_kinds: dict[str, frozenset[str]] = field(default_factory=dict)
