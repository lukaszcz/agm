"""Host engine-config decoding helpers (CLI flags, config-file entries)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agm.agent.spec import AGENT_SPECS
from agm.agl.semantics.values import RecordValue
from agm.config.engine_keys import ENGINE_KEYS, EngineKeyKind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agm.agl.ir.builtin_nominals import BuiltinNominals
    from agm.agl.ir.ids import NominalId
    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.types import Type as AglType
    from agm.agl.semantics.values import Value

__all__ = [
    "build_engine_config_seeds",
    "convert_config_value",
    "convert_host_value",
    "engine_default_settings",
    "raw_option_str",
    "restamp_engine_setting",
]


def _engine_key_shape(kind: EngineKeyKind) -> tuple[str, tuple[str, ...]] | None:
    """Return the ``(enum name, member names)`` an engine key *kind* restamps, if any."""
    if kind is EngineKeyKind.AGENT:
        return ("Agent", tuple(AGENT_SPECS))
    if kind is EngineKeyKind.AGENT_SANDBOX:
        return ("AgentSandbox", ("Disabled", "Native", "Sandbox"))
    if kind is EngineKeyKind.OPTION_TEXT:
        return ("Option", ("None", "Some"))
    return None


#: Engine key name -> ``(enum name, member names)``, built once from ``ENGINE_KEYS``.
_ENGINE_KEY_ENUM_SHAPES: dict[str, tuple[str, tuple[str, ...]]] = {
    spec.name: shape for spec in ENGINE_KEYS if (shape := _engine_key_shape(spec.kind)) is not None
}


def _restamp_value_tree(
    value: "Value", *, from_table: "BuiltinNominals", to_table: "BuiltinNominals"
) -> "Value":
    """Recursively restamp every nominal identity in a record tree from *from_table* to *to_table*.

    A restamped record can itself carry other host-known nominal values (e.g.
    ``AgentSandbox``'s ``Sandbox`` member has ``Optional``/``Option``-typed
    fields), so restamping the outer identity alone would leave a nested
    value's identity unrecognized by *to_table*'s owner. Only records recurse:
    every engine-setting value is a scalar, an ``Option``/``Optional``, or a
    record built from those (see ``semantics.engine_keys.ENGINE_KEY_TYPES``)
    -- never an array or dict -- so this never needs to look inside a
    collection. A non-record value, including one nested in a field this
    table has no name for, crosses unchanged.
    """
    if not isinstance(value, RecordValue):
        return value
    fields = {
        name: _restamp_value_tree(field_value, from_table=from_table, to_table=to_table)
        for name, field_value in value.fields.items()
    }
    located = from_table.reverse(value.nominal)
    if located is None:
        return RecordValue(nominal=value.nominal, fields=fields)
    name, member_name = located
    target = (
        to_table.resolve(name)
        if member_name is None
        else to_table.resolve_standard_member(name, member_name)
    )
    return RecordValue(nominal=target.nominal, fields=fields)


def _restamp_host_enum_member(
    value: RecordValue,
    *,
    enum_name: str,
    member_names: tuple[str, ...],
    from_table: "BuiltinNominals",
    to_table: "BuiltinNominals",
) -> RecordValue:
    """Restamp *value*'s identity from *from_table* to *to_table*, if it is an *enum_name* member.

    Used both to bind a persisted engine-setting seed (reserved fallback
    identity) onto a program's own nominal table, and to persist a
    post-run engine-setting value (a program's own identity) back onto the
    reserved fallback table so it survives past that program's own lifetime.
    Once membership is established, restamping the outer identity together
    with its nested fields is exactly :func:`_restamp_value_tree`'s job, so
    this delegates to it rather than rebuilding the fields dict itself.
    """
    for member_name in member_names:
        source = from_table.resolve_standard_member(enum_name, member_name)
        if value.nominal == source.nominal:
            restamped = _restamp_value_tree(value, from_table=from_table, to_table=to_table)
            assert isinstance(restamped, RecordValue)
            return restamped
    return value


def restamp_engine_setting(
    key: str, value: "Value", *, from_table: "BuiltinNominals", to_table: "BuiltinNominals"
) -> "Value":
    """Restamp *value* onto *to_table*'s identity when *key* is enum-backed.

    An enum-backed engine key (``AGENT``/``AGENT_SANDBOX``/``OPTION_TEXT`` kind) carries the
    identity of whichever nominal table stamped it; every reader of such a
    value needs it in its own table's identity to recognize the value's
    members with :func:`~agm.agl.ir.builtin_nominals.resolve_standard_member_name`
    (or, as here, directly). A non-enum-backed key, or a value that is not a
    ``RecordValue``, crosses through unchanged.
    """
    if not isinstance(value, RecordValue):
        return value
    shape = _ENGINE_KEY_ENUM_SHAPES.get(key)
    if shape is None:
        return value
    enum_name, member_names = shape
    return _restamp_host_enum_member(
        value,
        enum_name=enum_name,
        member_names=member_names,
        from_table=from_table,
        to_table=to_table,
    )


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
    timeout/trace-file strings before passing them to :func:`convert_config_value`.
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
    """Decode explicitly supplied scalar, ``Option``, or ``Agent`` host engine settings.

    The returned mapping deliberately omits absent keys. This preserves the
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
    them via :func:`convert_config_value` (``false``/``false``/``none``/``none``),
    building its own fresh seeded ``TypeTable`` rather than requiring one from
    the caller.

    ``default-agent`` is an ``Agent`` value rather than a scalar or
    ``Option[text]`` one and has no host-side default: it comes from the
    ``std/config`` ``builtin var`` declaration like any other declared default.
    """
    return build_engine_config_seeds(
        {spec.name: spec.default for spec in ENGINE_KEYS if spec.has_default}
    )


def _reserved_default_resolver(nominal: "NominalId", field_index: int) -> "Value":
    """Fill an omitted defaulted field of a reserved record with its host-side constant.

    Passed to :func:`~agm.agl.runtime.arguments.decode_param_value` as the
    ``default_resolver`` for every host engine-config decode
    (:func:`convert_host_value`). No program — and so no evaluator — exists
    yet at this boundary, unlike an ordinary program's own field default
    (filled by ``IrInterpreter.default_for_field`` against the real, fully
    linked ``NominalDescriptor`` table); a reserved record's default is
    instead a plain constant the seeded ``TypeDef`` carries directly (see
    :func:`~agm.agl.semantics.type_table.reserved_field_default`), which is
    all ``Sandbox`` (the first engine-key type with defaulted fields) needs.
    """
    from agm.agl.semantics.type_table import reserved_field_default

    return reserved_field_default(nominal.value, field_index)


def convert_host_value(
    name: str, raw: object, type_obj: AglType, type_table: "TypeTable"
) -> "Value":
    """Decode a raw host value against a declared AgL type.

    Builds the same :class:`~agm.agl.ir.contracts.ParamDecoder` the lowerer
    embeds in the compiled IR (via :func:`~agm.agl.type_schema.build_param_decoder`)
    and runs the shared :func:`~agm.agl.runtime.arguments.decode_param_value`
    path, so this and the compiled-IR value-decode boundary go through one
    mechanism. Its only caller is :func:`convert_config_value`, which decodes
    host engine-setting values (CLI flags, config-file entries) through it.

    ``text`` values are taken verbatim; a JSON-compatible Python value (not a
    string) crosses the canonical JSON boundary directly. A raw string is read
    through the shared strict-JSON-or-value-syntax dispatch
    (:func:`~agm.agl.runtime.value_decode.host_text_to_json`): strict JSON
    first, then one AgL value-syntax literal — no repair of user typos either
    way. *raw* may instead be an
    :class:`~agm.agl.runtime.arguments.OptionSome` box, for an ``Option[T]``
    *type_obj*: the boxed payload decodes against ``T``'s own field schema and
    is wrapped into the enum's ``Some`` shape, exactly as a program's own
    ``Option[T]`` parameter decodes. Types with no wire schema
    (unit/function/exception/…) are rejected up front; the builtin ``Agent``
    enum has an ordinary wire schema, dispatched through its own shorthand
    and constructor-call reading. *type_table* resolves record/enum
    field/variant shapes for *type_obj*. An omitted defaulted field (e.g.
    ``Sandbox``'s) fills through :func:`_reserved_default_resolver`.
    """
    from agm.agl.runtime.arguments import decode_param_value
    from agm.agl.runtime.convert import StrictJsonParseError
    from agm.agl.type_schema import build_param_decoder

    try:
        decoder = build_param_decoder(type_obj, type_table)
    except TypeError as exc:
        raise ValueError(f"Setting {name!r} has unsupported type {type_obj!r}.") from exc
    try:
        return decode_param_value(decoder, raw, default_resolver=_reserved_default_resolver)
    except (StrictJsonParseError, ValueError) as exc:
        raise ValueError(f"Setting {name!r}: could not parse as {type_obj!r}: {exc}") from exc


def convert_config_value(
    name: str, raw: object, key_type: AglType, type_table: "TypeTable | None" = None
) -> "Value":
    """Convert a raw scalar, ``Option``, or ``Agent`` host engine value to its AgL type.

    ``default-agent`` is an ``Agent`` value, not a scalar or ``Option`` setting;
    its host-supplied text or JSON-shaped data (``--default-agent``/config)
    decodes through :func:`convert_host_value` exactly like every other key.
    For ``Option[T]`` engine keys (``timeout``, ``trace-file``) a present *raw*
    is boxed as an :class:`~agm.agl.runtime.arguments.OptionSome` and decoded
    through :func:`convert_host_value` against the *whole* ``Option[T]``
    type — the same deferred-decode path a program's own ``Option[T]``
    parameter uses — and an absent (``None``) *raw* is ``none`` directly.
    Non-Option keys decode directly through :func:`convert_host_value`.
    *type_table* is threaded through to it.

    The settings accepted here are built-in scalar, ``Option[text]``, or the
    builtin ``Agent`` enum, never user-declared nominal types, so *type_table*
    defaults to a fresh
    seeded ``TypeTable`` when the caller has none in hand (e.g. CLI-flag config
    projection); callers that already hold the session/program table (the
    REPL) pass it explicitly.
    """
    from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
    from agm.agl.runtime.arguments import OptionSome
    from agm.agl.runtime.option import none_value
    from agm.agl.semantics.type_table import create_seeded_type_table
    from agm.agl.semantics.types import is_standard_option_enum

    table = type_table if type_table is not None else create_seeded_type_table()
    if is_standard_option_enum(key_type):
        if raw is None:
            return none_value(nominals=NO_BUILTIN_DECLARATIONS)
        return convert_host_value(name, OptionSome(raw), key_type, table)
    return convert_host_value(name, raw, key_type, table)
