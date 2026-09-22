"""Canonical schema of AGM's AgL engine config keys.

Pure data leaf (no ``agm`` imports) shared by the config layer — which reads
these keys from ``[exec]`` and qualified program TOML tables — AgL semantics,
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
    TEXT = "text"
    OPTION_TEXT = "option_text"
    AGENT = "agent"


class EngineKeyConsumer(Enum):
    """Which side of the host boundary owns an engine key's live value.

    ``RUNTIME_LIVE``
        The AgL evaluator backs the key with a live interpreter field (the
        strict-json mode, the shell timeout), so a write takes effect
        inside the evaluator itself.
    ``HOST_CONSUMED``
        The key has no interpreter field; its value lives in a register for the
        host to read on demand.
    """

    RUNTIME_LIVE = "runtime_live"
    HOST_CONSUMED = "host_consumed"


@dataclass(frozen=True)
class EngineKeySpec:
    """One engine key: its shape, config accessor, and host default.

    ``config_attr`` names the corresponding ``ExecConfig`` attribute without
    making this pure data leaf import the config layer. ``has_default``
    distinguishes an absent host default from an ``Option`` default whose value
    is ``None``.
    """

    name: str
    kind: EngineKeyKind
    consumer: EngineKeyConsumer
    config_attr: str | None = None
    default: object = None
    has_default: bool = True


# Ordered catalog of every engine key.  This is the one place a key is declared;
# every projection below is derived from it.
ENGINE_KEYS: tuple[EngineKeySpec, ...] = (
    EngineKeySpec(
        "trace",
        EngineKeyKind.BOOL,
        EngineKeyConsumer.HOST_CONSUMED,
        config_attr="trace",
        default=False,
    ),
    EngineKeySpec(
        "strict-json",
        EngineKeyKind.BOOL,
        EngineKeyConsumer.RUNTIME_LIVE,
        config_attr="strict_json",
        default=False,
    ),
    EngineKeySpec(
        "default-agent",
        EngineKeyKind.AGENT,
        EngineKeyConsumer.HOST_CONSUMED,
        config_attr="default_agent",
        has_default=False,
    ),
    EngineKeySpec(
        "trace-file",
        EngineKeyKind.OPTION_TEXT,
        EngineKeyConsumer.HOST_CONSUMED,
        config_attr="trace_file",
        default=None,
    ),
    EngineKeySpec(
        "timeout",
        EngineKeyKind.OPTION_TEXT,
        EngineKeyConsumer.RUNTIME_LIVE,
        config_attr="timeout",
        default=None,
    ),
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

# Keys backed by a host-owned register.
HOST_CONSUMED_ENGINE_KEYS: frozenset[str] = engine_keys_for(EngineKeyConsumer.HOST_CONSUMED)

#: The register pair backing the trace destination. A write to either repoints
#: the same trace store; see ``IrInterpreter._reconfigure_host_service``.
TRACE_ENGINE_KEYS: frozenset[str] = frozenset(
    spec.name
    for spec in ENGINE_KEYS
    if spec.consumer is EngineKeyConsumer.HOST_CONSUMED and spec.name in {"trace", "trace-file"}
)


def trace_write_implies_enabled(key: str, value_is_some: bool) -> bool:
    """Return whether a trace-register write also enables trace logging.

    A ``Some`` write to ``trace-file`` supplies a trace destination and therefore
    implies the ``trace`` register is enabled.
    """
    return key == "trace-file" and value_is_some
