"""Engine-key registry for AgL config declarations.

Maps fixed config engine-key names (kebab) to their resolved AgL types.
This module is the single source of truth for the fixed set of AgL engine
config keys and for program names reserved by AGM.
"""

from __future__ import annotations

from agm.agl.semantics.types import OPTION_TEXT_TYPE, BoolType, IntType, TextType, Type
from agm.command_catalog import COMMAND_NAMES
from agm.config.engine_keys import ENGINE_KEY_KINDS, EngineKeyKind
from agm.config.sections import RESERVED_CONFIG_SECTIONS

# Concrete AgL type for each engine-key value kind.  The kebab key names and
# their kinds are the single source of truth in :mod:`agm.config.engine_keys`;
# this layer only knows how each kind maps onto an AgL type.
_TYPE_BY_KIND: dict[EngineKeyKind, Type] = {
    EngineKeyKind.BOOL: BoolType(),
    EngineKeyKind.INT: IntType(),
    EngineKeyKind.TEXT: TextType(),
    EngineKeyKind.OPTION_TEXT: OPTION_TEXT_TYPE,
}

# Lookup: kebab key name → resolved AgL type.
_ENGINE_KEY_TYPES: dict[str, Type] = {
    name: _TYPE_BY_KIND[kind] for name, kind in ENGINE_KEY_KINDS
}

# Frozenset of all valid engine key names (kebab).
ENGINE_KEY_NAMES: frozenset[str] = frozenset(_ENGINE_KEY_TYPES)


def get_engine_key_type(name: str) -> Type | None:
    """Return the AgL type for engine key *name*, or ``None`` if unknown."""
    return _ENGINE_KEY_TYPES.get(name)


# ---------------------------------------------------------------------------
# Reserved program names
# ---------------------------------------------------------------------------

# Program names that collide with AGM top-level command and TOML config-section
# names.  A ``program NAME`` declaration whose name is in this set is a scope
# error because it would conflict with an existing ``[NAME]`` section in the
# config file schema.
#
# Sources: the shared CLI command catalog (:mod:`agm.command_catalog`) plus the
# reserved structural config-section names (:mod:`agm.config.sections`).
RESERVED_PROGRAM_NAMES: frozenset[str] = frozenset(COMMAND_NAMES) | RESERVED_CONFIG_SECTIONS
