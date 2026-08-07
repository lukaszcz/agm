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
    AGENT = "agent"


class EngineKeyConsumer(Enum):
    """Which side of the host boundary owns an engine key's live value.

    ``RUNTIME_LIVE``
        The AgL evaluator backs the key with a live interpreter field (the loop
        cap, the strict-json mode, the shell timeout), so a write takes effect
        inside the evaluator itself.
    ``HOST_CONSUMED``
        The key has no interpreter field; its value lives in a register.  A
        host-consumed key additionally sets ``reconfigures_host`` when a write
        must be reflected into a live host service rather than only stored.
    """

    RUNTIME_LIVE = "runtime_live"
    HOST_CONSUMED = "host_consumed"


@dataclass(frozen=True)
class EngineKeySpec:
    """One engine key: its shape, config accessor, and host default.

    ``reconfigures_host`` marks a host-consumed key whose write reconfigures a
    live host service (the trace destination). ``config_attr`` names the
    corresponding ``ExecConfig`` attribute without making this pure data leaf
    import the config layer. ``has_default`` distinguishes an absent host
    default from an ``Option`` default whose value is ``None``.
    """

    name: str
    kind: EngineKeyKind
    consumer: EngineKeyConsumer
    reconfigures_host: bool = False
    config_attr: str | None = None
    default: object = None
    has_default: bool = True


# Ordered catalog of every engine key.  This is the one place a key is declared;
# every projection below is derived from it.
ENGINE_KEYS: tuple[EngineKeySpec, ...] = (
    EngineKeySpec(
        "log",
        EngineKeyKind.BOOL,
        EngineKeyConsumer.HOST_CONSUMED,
        reconfigures_host=True,
        config_attr="log",
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
        "max-iters",
        EngineKeyKind.INT,
        EngineKeyConsumer.RUNTIME_LIVE,
        config_attr="default_loop_limit",
        default=0,
    ),
    EngineKeySpec(
        "default-agent",
        EngineKeyKind.AGENT,
        EngineKeyConsumer.HOST_CONSUMED,
        config_attr="default_agent",
        has_default=False,
    ),
    EngineKeySpec(
        "log-file",
        EngineKeyKind.OPTION_TEXT,
        EngineKeyConsumer.HOST_CONSUMED,
        reconfigures_host=True,
        config_attr="log_file",
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

# Keys whose write must be reflected into a live host service.
HOST_RECONFIGURING_ENGINE_KEYS: frozenset[str] = frozenset(
    spec.name for spec in ENGINE_KEYS if spec.reconfigures_host
)
