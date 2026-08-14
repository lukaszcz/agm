"""Runtime param decoding/validation + contract materialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agm.agl.diagnostics import Diagnostic
from agm.config.engine_keys import ENGINE_KEYS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agm.agl.ir.contracts import ParamDecoder
    from agm.agl.ir.ids import ContractId, SymbolId
    from agm.agl.ir.program import ExecutableProgram, IrParam
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.contract import OutputContract
    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.types import Type as AglType
    from agm.agl.semantics.values import Value

__all__ = [
    "build_engine_config_seeds",
    "convert_config_value",
    "convert_param_value",
    "decode_or_diagnose_param",
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


def decode_param_value(decoder: "ParamDecoder", raw: object) -> "Value":
    """Decode a raw host param value against *decoder* into a typed ``Value``.

    The single decode path shared by IR param binding (:func:`_prepare_ir_params`)
    and the REPL/config param path (:func:`convert_param_value`).  ``text`` params
    are taken verbatim; every other value crosses the canonical JSON boundary
    (strict parse, integral-decimal normalization, JSON-Schema validation, then
    the typeless :func:`decode_value` walk).

    :raises StrictJsonParseError: if a textual/native value is not strict JSON.
    :raises ValueError: on a type/shape mismatch or schema-validation failure.
    """
    from agm.agl.runtime.convert import (
        _clean_validation_message,
        decode_value,
        normalize_integral_decimals,
        parse_json_strict,
        validator_for_schema,
    )
    from agm.agl.runtime.serialize import dumps_exact

    if decoder.text_verbatim:
        if not isinstance(raw, str):
            raise ValueError(f"expected a text value (str), got {type(raw).__name__}")
        obj: object = raw
    elif isinstance(raw, str):
        obj = parse_json_strict(raw)
    elif _is_json_shaped(raw):
        # Native host values cross the same canonical JSON boundary as textual
        # values. In particular, Python floats become Decimal through
        # parse_float=Decimal before typed decoding.
        obj = parse_json_strict(dumps_exact(raw, indent=None))
    else:
        raise ValueError(f"expected a JSON-compatible value, got {type(raw).__name__}")
    normalized = normalize_integral_decimals(obj)
    validation_errors = list(validator_for_schema(decoder.json_schema).iter_errors(normalized))
    if validation_errors:
        raise ValueError(_clean_validation_message(validation_errors[0]))
    return decode_value(decoder.decode, normalized, dict(decoder.defs))


def decode_or_diagnose_param(
    param: "IrParam",
    display_name: str,
    supplied: bool,
    raw: object,
    *,
    missing_message: str,
) -> "tuple[Value, None] | tuple[None, Diagnostic] | tuple[None, None]":
    """Decode one host-supplied param value, or build its diagnostic.

    The single per-param boundary shared by IR param binding
    (:func:`_prepare_ir_params`) and the REPL's incremental param path
    (:meth:`~agm.agl.repl.session.ReplSession._pre_eval_param_values`).
    Returns ``(value, None)`` on success, ``(None, diagnostic)`` when the
    param is missing-and-required or fails to parse, and ``(None, None)``
    when the param is missing but optional (nothing to record).

    *missing_message* is caller-supplied because the two call sites phrase
    "missing" differently: compiled-IR binding reports a plain missing-param
    error, while the REPL's imported-param path points at supplying a
    default expression instead.
    """
    from agm.agl.runtime.convert import StrictJsonParseError

    if not supplied:
        if param.required:
            return None, Diagnostic(
                message=missing_message,
                line=param.location.start_line,
                column=param.location.start_col,
            )
        return None, None
    decoder = param.external_decoder
    assert decoder is not None, "lowerer must provide an external param decoder"
    try:
        return decode_param_value(decoder, raw), None
    except (StrictJsonParseError, ValueError) as exc:
        return None, Diagnostic(
            message=(
                f"Param {display_name!r}: could not parse as {decoder.target_type_label}: {exc}"
            ),
            line=param.location.start_line,
            column=param.location.start_col,
        )


def _prepare_ir_params(
    executable: "ExecutableProgram", param_values: "Mapping[str, object]"
) -> "tuple[dict[SymbolId, Value], list[Diagnostic]]":
    """Validate and typelessly decode external params from IR metadata."""
    decoded: "dict[SymbolId, Value]" = {}
    errors: list[Diagnostic] = []
    name_counts: dict[str, int] = {}
    for param in executable.params:
        name_counts[param.public_name] = name_counts.get(param.public_name, 0) + 1
    for param in executable.params:
        value_name = (
            param.qualified_public_name if name_counts[param.public_name] > 1 else param.public_name
        )
        # The CLI accepts a module-qualified spelling even where the short
        # spelling is unavailable because it collides with an AGM option.
        # Prefer that identity-preserving key whenever supplied.
        supplied_name = (
            param.qualified_public_name
            if param.qualified_public_name in param_values
            else value_name
        )
        supplied = supplied_name in param_values
        value, diagnostic = decode_or_diagnose_param(
            param,
            value_name,
            supplied,
            param_values.get(supplied_name),
            missing_message=f"Missing required param: {value_name!r}",
        )
        if diagnostic is not None:
            errors.append(diagnostic)
        elif value is not None:
            decoded[param.symbol] = value
    return decoded, errors


def _materialize_ir_contracts(
    executable: "ExecutableProgram", codecs: "Mapping[str, OutputCodec]"
) -> "tuple[dict[ContractId, OutputContract], list[Diagnostic]]":
    """Materialize host codec contracts exclusively from linked IR metadata."""
    from agm.agl.runtime.contract import materialize_ir_contract

    materialized: "dict[ContractId, OutputContract]" = {}
    errors: list[Diagnostic] = []
    for contract_id, request in executable.contracts.items():
        try:
            contract = materialize_ir_contract(request, codecs)
        except ValueError as exc:
            errors.append(Diagnostic(message=f"Contract error: {exc}", line=1))
            continue
        if contract is not None:
            materialized[contract_id] = contract
    return materialized, errors


def convert_param_value(
    name: str, raw: object, type_obj: "AglType", type_table: "TypeTable"
) -> "Value":
    """Convert a raw host param value to the declared AgL type.

    Builds the same :class:`~agm.agl.ir.contracts.ParamDecoder` the lowerer
    embeds in the compiled IR (via :func:`~agm.agl.type_schema.build_param_decoder`)
    and runs the shared :func:`decode_param_value` path, so the REPL/config param
    boundary and the compiled-IR param boundary decode through one mechanism.

    ``text`` params are taken verbatim; every other value crosses the canonical
    JSON boundary — either a JSON string or a JSON-compatible Python value, both
    parsed strictly (no json-repair of user typos).  Types with no wire
    schema (unit/agent/exception/…) are rejected up front.  *type_table*
    resolves record/enum field/variant shapes for *type_obj*.
    """
    from agm.agl.runtime.convert import StrictJsonParseError
    from agm.agl.type_schema import build_param_decoder

    try:
        decoder = build_param_decoder(type_obj, type_table)
    except TypeError as exc:
        raise ValueError(f"Param {name!r} has unsupported type {type_obj!r}.") from exc
    try:
        return decode_param_value(decoder, raw)
    except (StrictJsonParseError, ValueError) as exc:
        raise ValueError(f"Param {name!r}: could not parse as {type_obj!r}: {exc}") from exc


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
    its inner ``T`` decoded via :func:`convert_param_value`, and ``None`` becomes
    ``none``.  Non-Option keys fall back to :func:`convert_param_value`.
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
    from agm.agl.semantics.types import EnumType, TextType

    table = type_table if type_table is not None else create_seeded_type_table()
    if isinstance(key_type, EnumType) and key_type.name == "Option":
        if raw is None:
            return none_value()
        inner: AglType = key_type.type_args[0] if key_type.type_args else TextType()
        return some_value(convert_param_value(name, raw, inner, table))
    return convert_param_value(name, raw, key_type, table)


def _is_json_shaped(obj: object) -> bool:
    """Return ``True`` iff *obj* is a JSON-compatible Python value.

    The closed set: ``None``, ``bool``, ``int``, ``float``,
    ``decimal.Decimal``, ``str``, ``list`` (elements recursively JSON-shaped),
    and ``dict`` (str keys, values recursively JSON-shaped).

    Used by :func:`convert_param_value` to detect non-JSON-shaped host objects
    (e.g. sets or custom classes) before attempting serialisation, so the
    caller can emit a clean diagnostic instead of a cryptic traceback.
    """
    import decimal as _decimal_mod

    if obj is None or isinstance(obj, (bool, int, float, str, _decimal_mod.Decimal)):
        return True
    if isinstance(obj, list):
        return all(_is_json_shaped(e) for e in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _is_json_shaped(v) for k, v in obj.items())
    return False
