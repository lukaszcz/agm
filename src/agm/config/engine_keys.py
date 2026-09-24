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
    AGENT_SANDBOX = "agent_sandbox"


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
    """One engine key: its shape, config accessor, host default, and relations.

    ``config_attr`` names the corresponding ``ExecConfig`` attribute without
    making this pure data leaf import the config layer. ``has_default``
    distinguishes an absent host default from an ``Option`` default whose value
    is ``None``.

    ``register`` names a host register several keys share: a write to any of
    them repoints the same store, and the register's own on/off switch is the
    key whose ``name`` is that register. ``enables_register`` marks a key whose
    ``Some`` value implies the switch is on. ``is_path`` marks a key whose
    value is a filesystem path, so the config layer interpolates, expands and
    anchors it like any other path-valued field.
    """

    name: str
    kind: EngineKeyKind
    consumer: EngineKeyConsumer
    config_attr: str | None = None
    default: object = None
    has_default: bool = True
    register: str | None = None
    enables_register: bool = False
    is_path: bool = False


#: The register backing the trace destination, named after its on/off switch.
TRACE_REGISTER = "trace"

# Ordered catalog of every engine key.  This is the one place a key is declared;
# every projection below is derived from it.
ENGINE_KEYS: tuple[EngineKeySpec, ...] = (
    EngineKeySpec(
        "trace",
        EngineKeyKind.BOOL,
        EngineKeyConsumer.HOST_CONSUMED,
        config_attr="trace",
        default=False,
        register=TRACE_REGISTER,
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
        "default-sandbox",
        EngineKeyKind.AGENT_SANDBOX,
        EngineKeyConsumer.HOST_CONSUMED,
        config_attr="default_sandbox",
        has_default=False,
    ),
    EngineKeySpec(
        "trace-file",
        EngineKeyKind.OPTION_TEXT,
        EngineKeyConsumer.HOST_CONSUMED,
        config_attr="trace_file",
        default=None,
        register=TRACE_REGISTER,
        enables_register=True,
        is_path=True,
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

#: Every declared register name.
ENGINE_REGISTERS: tuple[str, ...] = tuple(
    sorted({spec.register for spec in ENGINE_KEYS if spec.register is not None})
)


def engine_keys_for_register(register: str) -> frozenset[str]:
    """Return the names of every engine key sharing *register*."""
    return frozenset(spec.name for spec in ENGINE_KEYS if spec.register == register)


#: The key pair backing the trace destination. A write to either repoints the
#: same trace store; see ``IrInterpreter._reconfigure_host_service``.
TRACE_ENGINE_KEYS: frozenset[str] = engine_keys_for_register(TRACE_REGISTER)

_TRACE_ENABLING_KEYS: frozenset[str] = frozenset(
    spec.name for spec in ENGINE_KEYS if spec.register == TRACE_REGISTER and spec.enables_register
)

#: Ordered names of the engine keys whose value is a filesystem path.
PATH_ENGINE_KEYS: tuple[str, ...] = tuple(spec.name for spec in ENGINE_KEYS if spec.is_path)


def trace_write_implies_enabled(key: str, value_is_some: bool) -> bool:
    """Return whether a trace-register write also enables trace logging.

    A ``Some`` write to a key marked ``enables_register`` supplies a trace
    destination and therefore implies the ``trace`` switch is on.
    """
    return value_is_some and key in _TRACE_ENABLING_KEYS
