"""Engine-key registry for AgL config declarations.

Maps fixed config engine-key names (kebab) to their resolved AgL types.
This module is the single source of truth for the fixed set of AgL engine
config keys and for program names reserved by AGM.
"""

from __future__ import annotations

from dataclasses import dataclass

from agm.agl.semantics.types import OPTION_TEXT_TYPE, BoolType, IntType, TextType, Type


@dataclass(frozen=True, slots=True)
class EngineKeyEntry:
    """Metadata for a single engine config key."""

    name: str       # kebab-case key name
    agl_type: Type  # resolved AgL type


# Ordered catalog of all fixed config engine keys (plan _PLAN_CONTEXT.md §3.3).
_ENGINE_KEY_CATALOG: tuple[EngineKeyEntry, ...] = (
    EngineKeyEntry(name="log",         agl_type=BoolType()),
    EngineKeyEntry(name="strict-json", agl_type=BoolType()),
    EngineKeyEntry(name="max-iters",   agl_type=IntType()),
    EngineKeyEntry(name="runner",      agl_type=TextType()),
    EngineKeyEntry(name="log-file",    agl_type=OPTION_TEXT_TYPE),
    EngineKeyEntry(name="timeout",     agl_type=OPTION_TEXT_TYPE),
)

# Lookup: kebab name → EngineKeyEntry
_ENGINE_KEY_MAP: dict[str, EngineKeyEntry] = {e.name: e for e in _ENGINE_KEY_CATALOG}

# Frozenset of all valid engine key names (kebab).
ENGINE_KEY_NAMES: frozenset[str] = frozenset(_ENGINE_KEY_MAP)


def get_engine_key_type(name: str) -> Type | None:
    """Return the AgL type for engine key *name*, or ``None`` if unknown."""
    entry = _ENGINE_KEY_MAP.get(name)
    return entry.agl_type if entry is not None else None


# ---------------------------------------------------------------------------
# Reserved program names
# ---------------------------------------------------------------------------

# Program names that collide with AGM top-level command and TOML config-section
# names.  A ``program NAME`` declaration whose name is in this set is a scope
# error because it would conflict with an existing ``[NAME]`` section in the
# config file schema (plan §15 / D7).
#
# Sources: top-level CLI command names (parser.py _COMMAND_OVERVIEW) and
# TOML config section names (config/general.py).
RESERVED_PROGRAM_NAMES: frozenset[str] = frozenset({
    # Top-level AGM CLI commands (parser.py _COMMAND_OVERVIEW)
    "open",
    "close",
    "workspace",
    "init",
    "sync",
    "dep",
    "loop",
    "review",
    "revise",
    "refine",
    "exec",
    "repl",
    "run",
    "config",
    "worktree",
    "tmux",
    "help",
    # Additional TOML config section names (config/general.py)
    "agents",
    "params",
})
