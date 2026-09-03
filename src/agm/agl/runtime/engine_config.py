"""Host engine-config decoding helpers (CLI flags, config-file entries)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agm.config.engine_keys import ENGINE_KEYS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.types import Type as AglType
    from agm.agl.semantics.values import Value

__all__ = [
    "build_engine_config_seeds",
    "convert_config_value",
    "convert_host_value",
    "engine_default_settings",
    "raw_option_str",
]


def raw_option_str(
    primary: "Mapping[str, object]",
    fallback: "Mapping[str, object]",
    key: str,
) -> str | None:
    """Return the raw TOML value for *key* as a string, checking primary then fallback.

    Preserves the exact string written in the config file (e.g. ``"30s"``).
    For numeric values (int/float), converts to string (e.g. ``60`` → ``"60"``),
    but only when the value is positive (a zero/negative numeric config value is
    treated as absent).
    Returns ``None`` when the key is absent or empty/invalid in both tables.

    Used by ``commands/exec.py`` and ``commands/repl.py`` to extract the raw
    timeout/log-file strings before passing them to :func:`convert_config_value`.
    """
    for table in (primary, fallback):
        val = table.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return str(val)
        if isinstance(val, float) and val > 0:
            from agm.core.parse import format_timeout

            return format_timeout(val)
    return None


def build_engine_config_seeds(raw_values: "Mapping[str, object]") -> "dict[str, Value]":
    """Decode explicitly supplied scalar or ``Option`` host engine settings.

    Callers seed ``default-agent`` separately as a typed ``Agent`` value. The
    returned mapping deliberately omits absent keys. This preserves the
    distinction between a host control and the runtime fallback, letting a
    ``builtin var`` initializer supply the latter.  A present value of
    ``None`` remains meaningful for ``Option`` settings such as ``timeout``.
    """
    from agm.agl.semantics.engine_keys import get_engine_key_type
    from agm.agl.semantics.type_table import create_seeded_type_table

    type_table = create_seeded_type_table()
    result: dict[str, Value] = {}
    for key_name, raw in raw_values.items():
        key_type = get_engine_key_type(key_name)
        if key_type is None:
            raise ValueError(f"unknown engine key: {key_name!r}")
        result[key_name] = convert_config_value(key_name, raw, key_type, type_table)
    return result


def engine_default_settings() -> "dict[str, Value]":
    """Build the typed engine-default value for every scalar/``Option[text]`` engine key.

    Derives the raw values from the shared engine-key catalog, then decodes
    them via :func:`convert_config_value` (``false``/``false``/``0``/``none``/
    ``none``, where zero represents the disabled ``max-iters`` safety valve),
    building its own fresh seeded ``TypeTable`` rather than requiring one from
    the caller.

    ``default-agent`` is an ``Agent`` value rather than a scalar or
    ``Option[text]`` one and has no host-side default: it comes from the
    ``std/config`` ``builtin var`` declaration like any other declared default.
    """
    return build_engine_config_seeds(
        {spec.name: spec.default for spec in ENGINE_KEYS if spec.has_default}
    )


def convert_host_value(
    name: str, raw: object, type_obj: "AglType", type_table: "TypeTable"
) -> "Value":
    """Decode a raw host value against a declared AgL type.

    Builds the same :class:`~agm.agl.ir.contracts.ParamDecoder` the lowerer
    embeds in the compiled IR (via :func:`~agm.agl.type_schema.build_param_decoder`)
    and runs the shared :func:`~agm.agl.runtime.arguments.decode_param_value`
    path, so this and the compiled-IR value-decode boundary go through one
    mechanism. Its only caller is :func:`convert_config_value`, which decodes
    host engine-setting values (CLI flags, config-file entries) through it.

    ``text`` values are taken verbatim; every other value crosses the canonical
    JSON boundary — either a JSON string or a JSON-compatible Python value, both
    parsed strictly (no json-repair of user typos).  Types with no wire
    schema (unit/agent/exception/…) are rejected up front.  *type_table*
    resolves record/enum field/variant shapes for *type_obj*.
    """
    from agm.agl.runtime.arguments import decode_param_value
    from agm.agl.runtime.convert import StrictJsonParseError
    from agm.agl.type_schema import build_param_decoder

    try:
        decoder = build_param_decoder(type_obj, type_table)
    except TypeError as exc:
        raise ValueError(f"Setting {name!r} has unsupported type {type_obj!r}.") from exc
    try:
        return decode_param_value(decoder, raw)
    except (StrictJsonParseError, ValueError) as exc:
        raise ValueError(f"Setting {name!r}: could not parse as {type_obj!r}: {exc}") from exc


def convert_config_value(
    name: str, raw: object, key_type: "AglType", type_table: "TypeTable | None" = None
) -> "Value":
    """Convert a raw scalar or ``Option`` host engine value to its AgL type.

    ``default-agent`` is an ``Agent`` value, not a scalar or ``Option`` setting;
    a host-supplied AgL literal (``--agent``/``[exec] default-agent``) is parsed
    separately as an engine-setting override rather than through this helper.
    The one exception is ``[exec] runner``, a bare host command string with no
    AgL syntax of its own: it is decoded here as an ``{"$case":
    "AgentCommand", "command": ...}`` shape, the same decode path any other
    engine key uses. For ``Option[T]`` engine keys (``timeout``, ``log-file``) the raw value is
    projected into the Option enum: a present *raw* becomes ``some(value)`` with
    its inner ``T`` decoded via :func:`convert_host_value`, and ``None`` becomes
    ``none``.  Non-Option keys fall back to :func:`convert_host_value`.
    *type_table* is threaded through to both; the Option unwrap itself reads
    ``key_type.type_args`` directly and never needs variant shapes from it.

    The settings accepted here are built-in scalar or ``Option[text]`` types,
    never user-declared nominal types, so *type_table* defaults to a fresh
    seeded ``TypeTable`` when the caller has none in hand (e.g. CLI-flag config
    projection); callers that already hold the session/program table (the
    REPL) pass it explicitly.
    """
    from agm.agl.runtime.option import none_value, some_value
    from agm.agl.semantics.type_table import create_seeded_type_table
    from agm.agl.semantics.types import TextType, is_standard_option_enum

    table = type_table if type_table is not None else create_seeded_type_table()
    if is_standard_option_enum(key_type):
        if raw is None:
            return none_value()
        inner: AglType = key_type.type_args[0] if key_type.type_args else TextType()
        return some_value(convert_host_value(name, raw, inner, table))
    return convert_host_value(name, raw, key_type, table)
