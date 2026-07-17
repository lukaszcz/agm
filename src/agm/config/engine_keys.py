"""Canonical schema of AGM's AgL engine config keys.

Pure data leaf (no ``agm`` imports) shared by the config layer — which reads
these keys from the ``[exec]`` / ``[<program>]`` TOML tables — AgL semantics,
which maps each key to a concrete type, deep IR validation, and the AgL
evaluator/REPL, which route a key's write by its consuming side. Keeping the
catalog here lets all consumers depend on one definition without coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
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


class EngineKeyConsumer(Enum):
    """Which side of the host boundary owns an engine key's live value.

    ``RUNTIME_LIVE``
        The AgL evaluator backs the key with a live interpreter field (the loop
        cap, the strict-json mode, the shell timeout), so a write takes effect
        inside the evaluator itself.
    ``HOST_CONSUMED``
        The key has no interpreter field; its value lives in a register and a
        write is reflected back into a live host service (the agent runner, the
        trace destination) by the host.
    """

    RUNTIME_LIVE = "runtime_live"
    HOST_CONSUMED = "host_consumed"


@dataclass(frozen=True)
class EngineKeySpec:
    """One engine key: its kebab-case name, value kind, and consuming side."""

    name: str
    kind: EngineKeyKind
    consumer: EngineKeyConsumer


# Ordered catalog of every engine key.  This is the one place a key is declared;
# every projection below is derived from it.
ENGINE_KEYS: tuple[EngineKeySpec, ...] = (
    EngineKeySpec("log", EngineKeyKind.BOOL, EngineKeyConsumer.HOST_CONSUMED),
    EngineKeySpec("strict-json", EngineKeyKind.BOOL, EngineKeyConsumer.RUNTIME_LIVE),
    EngineKeySpec("max-iters", EngineKeyKind.INT, EngineKeyConsumer.RUNTIME_LIVE),
    EngineKeySpec("runner", EngineKeyKind.TEXT, EngineKeyConsumer.HOST_CONSUMED),
    EngineKeySpec("log-file", EngineKeyKind.OPTION_TEXT, EngineKeyConsumer.HOST_CONSUMED),
    EngineKeySpec("timeout", EngineKeyKind.OPTION_TEXT, EngineKeyConsumer.RUNTIME_LIVE),
)

# Ordered projection for consumers that only need name -> value kind.
ENGINE_KEY_KINDS: tuple[tuple[str, EngineKeyKind], ...] = tuple(
    (spec.name, spec.kind) for spec in ENGINE_KEYS
)

# Closed set shared by configuration, AgL semantics, and deep IR validation.
ENGINE_KEY_NAMES: frozenset[str] = frozenset(spec.name for spec in ENGINE_KEYS)


def engine_keys_for(consumer: EngineKeyConsumer) -> frozenset[str]:
    """Return the names of every engine key consumed by *consumer*."""
    return frozenset(spec.name for spec in ENGINE_KEYS if spec.consumer is consumer)


# Keys whose write applies a live effect inside the AgL evaluator.
RUNTIME_LIVE_ENGINE_KEYS: frozenset[str] = engine_keys_for(EngineKeyConsumer.RUNTIME_LIVE)

# Keys backed by a register and reflected into a live host service on write.
HOST_CONSUMED_ENGINE_KEYS: frozenset[str] = engine_keys_for(EngineKeyConsumer.HOST_CONSUMED)
