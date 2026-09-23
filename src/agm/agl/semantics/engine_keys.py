"""Engine-key registry for AgL engine settings.

Maps fixed engine-key names (kebab) to their resolved AgL types. This module
is the single source of truth for the fixed set of AgL engine-setting keys
declared as ``builtin var`` bindings in ``std/config``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from agm.agl.semantics.types import (
    BUILTIN_PRELUDE_TYPES,
    OPTION_TEXT_TYPE,
    BoolType,
    TextType,
    Type,
)
from agm.config.engine_keys import ENGINE_KEY_KINDS, EngineKeyKind
from agm.config.engine_keys import ENGINE_KEY_NAMES as ENGINE_KEY_NAMES

# Concrete AgL type for each engine-key value kind.  The kebab key names and
# their kinds are the single source of truth in :mod:`agm.config.engine_keys`;
# this layer only knows how each kind maps onto an AgL type.
_TYPE_BY_KIND: dict[EngineKeyKind, Type] = {
    EngineKeyKind.BOOL: BoolType(),
    EngineKeyKind.TEXT: TextType(),
    EngineKeyKind.OPTION_TEXT: OPTION_TEXT_TYPE,
    EngineKeyKind.AGENT: BUILTIN_PRELUDE_TYPES["Agent"],
    EngineKeyKind.AGENT_SANDBOX: BUILTIN_PRELUDE_TYPES["AgentSandbox"],
}

# Lookup: kebab key name → resolved AgL type. Built from the same
# ``ENGINE_KEY_KINDS`` tuple ``cli_support.program_options.engine_key_flags``
# iterates, so that lookup is total by construction: every name it visits is
# a key of this mapping.
ENGINE_KEY_TYPES: Mapping[str, Type] = MappingProxyType(
    {name: _TYPE_BY_KIND[kind] for name, kind in ENGINE_KEY_KINDS}
)


def get_engine_key_type(name: str) -> Type | None:
    """Return the AgL type for engine key *name*, or ``None`` if unknown."""
    return ENGINE_KEY_TYPES.get(name)
