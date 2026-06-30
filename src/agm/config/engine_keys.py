"""Canonical schema of AGM's AgL engine config keys.

Pure data leaf (no ``agm`` imports) shared by the config layer — which reads
these keys from the ``[exec]`` / ``[<program>]`` TOML tables — and the AgL
semantics layer, which maps each key to a concrete AgL type.  Keeping the key
catalog here lets both layers depend on one definition without coupling.
"""

from __future__ import annotations

from enum import Enum


class EngineKeyKind(Enum):
    """Value kind of an engine config key.

    The AgL semantics layer maps each kind to a concrete AgL type; the config
    layer only reads key names, so the kind is opaque to it.
    """

    BOOL = "bool"
    INT = "int"
    TEXT = "text"
    OPTION_TEXT = "option_text"


# Ordered catalog of every engine key: kebab-case name -> value kind.
ENGINE_KEY_KINDS: tuple[tuple[str, EngineKeyKind], ...] = (
    ("log", EngineKeyKind.BOOL),
    ("strict-json", EngineKeyKind.BOOL),
    ("max-iters", EngineKeyKind.INT),
    ("runner", EngineKeyKind.TEXT),
    ("log-file", EngineKeyKind.OPTION_TEXT),
    ("timeout", EngineKeyKind.OPTION_TEXT),
)
