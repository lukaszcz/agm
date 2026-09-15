"""Shared CLI-over-config resolution of the AgL engine settings a host controls.

``agm exec`` and ``agm repl`` both hand the AgL engine a seed mapping for the
settings the host explicitly controls.  The layering rule is the same for both —
CLI flag, then the command's configuration tables — so it lives here rather than
in either command.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

from agm.agl.runtime.engine_config import convert_config_value, raw_option_str
from agm.config.engine_keys import ENGINE_KEY_NAMES, ENGINE_KEYS, EngineKeyKind, EngineKeySpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agm.agl.semantics.values import Value
    from agm.config.general import ExecConfig

__all__ = ["build_host_engine_seeds"]


def _configured_value(
    spec: EngineKeySpec,
    config: "ExecConfig",
    primary_table: "Mapping[str, object]",
    fallback_table: "Mapping[str, object]",
) -> object | None:
    """Return one configured engine value, preserving raw ``Option[text]`` spelling."""
    assert spec.config_attr is not None
    configured_value = cast(object | None, getattr(config, spec.config_attr))
    if configured_value is None:
        return None
    if spec.kind is EngineKeyKind.OPTION_TEXT:
        return raw_option_str(primary_table, fallback_table, spec.name)
    return configured_value


def build_host_engine_seeds(
    *,
    config: "ExecConfig",
    primary_table: "Mapping[str, object]",
    fallback_table: "Mapping[str, object] | None" = None,
    cli_values: "Mapping[str, object | None]",
) -> "dict[str, Value]":
    """Decode the engine settings the host explicitly controls into seed values.

    Only explicit controls are seeded: a setting left to its default stays
    absent from the result, so a ``builtin var`` initializer supplies it instead
    of being suppressed by a host-side floor. ``cli_values`` contains only
    explicitly supplied CLI values; its present ``None`` values represent an
    explicit empty ``Option``. CLI values win over configuration tables, which
    are consulted in *primary_table* then *fallback_table* order.

    Every key, ``default-agent`` included, decodes through
    :func:`~agm.agl.runtime.engine_config.convert_config_value` against one
    shared seeded ``TypeTable``. A decode failure prints an error naming the
    offending key and its origin (``--<name>`` for a CLI flag, otherwise
    ``configuration key <name>``) to stderr and exits 1 here, before anything
    runs.
    """
    from agm.agl.semantics.engine_keys import ENGINE_KEY_TYPES
    from agm.agl.semantics.type_table import create_seeded_type_table

    fallback: "Mapping[str, object]" = fallback_table if fallback_table is not None else {}
    configured = {key for key in ENGINE_KEY_NAMES if key in primary_table or key in fallback}

    seed_raw: dict[str, object] = {}
    origins: dict[str, str] = {}
    for spec in ENGINE_KEYS:
        # ``log`` is implied by log-file, so it follows its own rule below
        # instead of this per-key resolution.
        if spec.name == "log":
            continue
        if spec.name in cli_values:
            seed_raw[spec.name] = cli_values[spec.name]
            origins[spec.name] = f"--{spec.name}"
        elif spec.name in configured:
            value = _configured_value(spec, config, primary_table, fallback)
            # A ``None`` config result is absent, not an explicit control. In
            # particular this lets a builtin initializer supply invalid/empty
            # Option values.
            if value is not None:
                seed_raw[spec.name] = value
                origins[spec.name] = f"configuration key {spec.name}"

    # ``log`` is not an ordinary setting: a supplied log-file implies it. Keep
    # that relationship explicit instead of encoding it in the catalog loop.
    if "log" in cli_values:
        seed_raw["log"] = cli_values["log"]
        origins["log"] = "--log"
    elif cli_values.get("log-file") is not None:
        seed_raw["log"] = True
        origins["log"] = "--log-file"
    elif configured & {"log", "log-file"}:
        seed_raw["log"] = config.log or config.log_file is not None
        origins["log"] = "configuration key log"

    type_table = create_seeded_type_table()
    seeds: dict[str, Value] = {}
    for key_name, raw in seed_raw.items():
        # ``seed_raw``'s keys are always drawn from ``ENGINE_KEYS`` above, so
        # this lookup is total: never a missing key to guard against.
        key_type = ENGINE_KEY_TYPES[key_name]
        try:
            seeds[key_name] = convert_config_value(key_name, raw, key_type, type_table)
        except ValueError as exc:
            print(
                f"Error: invalid {key_name} value from {origins[key_name]}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    return seeds
