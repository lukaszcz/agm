"""Runtime param decoding/validation + contract materialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agm.agl.diagnostics import Diagnostic

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agm.agl.ir.contracts import ParamDecoder
    from agm.agl.ir.ids import ContractId, SymbolId
    from agm.agl.ir.program import ExecutableProgram
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.contract import OutputContract
    from agm.agl.semantics.types import Type as AglType
    from agm.agl.semantics.values import Value

__all__ = ["convert_config_value", "convert_param_value"]


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
    return decode_value(decoder.decode, obj)


def _prepare_ir_params(
    executable: "ExecutableProgram", param_values: "Mapping[str, object]"
) -> "tuple[dict[SymbolId, Value], list[Diagnostic]]":
    """Validate and typelessly decode external params from IR metadata."""
    from agm.agl.runtime.convert import StrictJsonParseError

    decoded: "dict[SymbolId, Value]" = {}
    errors: list[Diagnostic] = []
    for param in executable.params:
        if param.public_name not in param_values:
            if param.required:
                errors.append(
                    Diagnostic(
                        message=f"Missing required param: {param.public_name!r}",
                        line=param.location.start_line,
                        column=param.location.start_col,
                    )
                )
            continue
        decoder = param.external_decoder
        assert decoder is not None, "lowerer must provide an external param decoder"
        try:
            decoded[param.symbol] = decode_param_value(decoder, param_values[param.public_name])
        except (StrictJsonParseError, ValueError) as exc:
            errors.append(
                Diagnostic(
                    message=(
                        f"Param {param.public_name!r}: could not parse as "
                        f"{decoder.target_type_label}: {exc}"
                    ),
                    line=param.location.start_line,
                    column=param.location.start_col,
                )
            )
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


def convert_param_value(name: str, raw: object, type_obj: "AglType") -> "Value":
    """Convert a raw host param value to the declared AgL type.

    Builds the same :class:`~agm.agl.ir.contracts.ParamDecoder` the lowerer
    embeds in the compiled IR (via :func:`~agm.agl.type_schema.build_param_decoder`)
    and runs the shared :func:`decode_param_value` path, so the REPL/config param
    boundary and the compiled-IR param boundary decode through one mechanism.

    ``text`` params are taken verbatim; every other value crosses the canonical
    JSON boundary — either a JSON string or a JSON-compatible Python value, both
    parsed strictly (F7: no json-repair of user typos).  Types with no wire
    schema (unit/agent/exception/…) are rejected up front.
    """
    from agm.agl.runtime.convert import StrictJsonParseError
    from agm.agl.type_schema import build_param_decoder

    try:
        decoder = build_param_decoder(type_obj)
    except TypeError as exc:
        raise ValueError(f"Param {name!r} has unsupported type {type_obj!r}.") from exc
    try:
        return decode_param_value(decoder, raw)
    except (StrictJsonParseError, ValueError) as exc:
        raise ValueError(
            f"Param {name!r}: could not parse as {type_obj!r}: {exc}"
        ) from exc


def convert_config_value(name: str, raw: object, key_type: "AglType") -> "Value":
    """Convert a raw host config value to the declared engine-key AgL type.

    For ``Option[T]`` engine keys (``timeout``, ``log-file``) the raw value is
    projected into the Option enum: a present *raw* becomes ``some(value)`` with
    its inner ``T`` decoded via :func:`convert_param_value`, and ``None`` becomes
    ``none``.  Non-Option keys fall back to :func:`convert_param_value`.
    """
    from agm.agl.ir.ids import NominalId
    from agm.agl.modules.ids import STD_CORE_ID
    from agm.agl.semantics.types import EnumType, TextType
    from agm.agl.semantics.values import EnumValue

    if isinstance(key_type, EnumType) and key_type.name == "Option":
        nominal = NominalId(STD_CORE_ID, "Option")
        if raw is None:
            return EnumValue(
                nominal=nominal, display_name="Option", variant="None", fields={}
            )
        inner: AglType = key_type.type_args[0] if key_type.type_args else TextType()
        inner_val = convert_param_value(name, raw, inner)
        return EnumValue(
            nominal=nominal,
            display_name="Option",
            variant="Some",
            fields={"value": inner_val},
        )
    return convert_param_value(name, raw, key_type)


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
