"""Runtime param decoding/validation + contract materialization helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from agm.agl.diagnostics import Diagnostic

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agm.agl.ir.ids import ContractId, SymbolId
    from agm.agl.ir.program import ExecutableProgram
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.contract import OutputContract
    from agm.agl.semantics.types import Type as AglType
    from agm.agl.semantics.values import Value

__all__ = ["convert_param_value"]


def _prepare_ir_params(
    executable: "ExecutableProgram", param_values: "Mapping[str, object]"
) -> "tuple[dict[SymbolId, Value], list[Diagnostic]]":
    """Validate and typelessly decode external params from IR metadata."""
    from jsonschema import Draft202012Validator

    from agm.agl.runtime.convert import (
        StrictJsonParseError,
        decode_value,
        normalize_integral_decimals,
        parse_json_strict,
    )
    from agm.agl.runtime.serialize import dumps_exact

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
        raw = param_values[param.public_name]
        try:
            if decoder.text_verbatim:
                if not isinstance(raw, str):
                    raise ValueError(
                        f"expected a text value (str), got {type(raw).__name__}"
                    )
                obj: object = raw
            elif isinstance(raw, str):
                obj = parse_json_strict(raw)
            elif _is_json_shaped(raw):
                # Native host values cross the same canonical JSON boundary as
                # textual values. In particular, Python floats become Decimal
                # through parse_float=Decimal before typed decoding.
                obj = parse_json_strict(dumps_exact(raw, indent=None))
            else:
                raise ValueError(f"expected a JSON-compatible value, got {type(raw).__name__}")
            normalized = normalize_integral_decimals(obj)
            schema = cast(object, json.loads(decoder.json_schema))
            validation_errors = list(Draft202012Validator(schema).iter_errors(normalized))
            if validation_errors:
                raise ValueError(validation_errors[0].message)
            decoded[param.symbol] = decode_value(decoder.decode, obj)
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

    Supported types:
    - ``text``: verbatim (the value must already be a ``str``).
    - ``int``/``decimal``/``bool``/``json``: parsed via stdlib ``json`` with
      ``parse_float=Decimal`` (design §5.1: no binary floats) and validated.
    - ``list``/``dict``/``record``/``enum``: parsed from a JSON string via the
      ``JsonCodec``.
    """
    import decimal as _decimal

    from agm.agl.semantics.types import (
        BoolType,
        DecimalType,
        DictType,
        EnumType,
        IntType,
        JsonType,
        ListType,
        RecordType,
        TextType,
    )
    from agm.agl.semantics.values import (
        BoolValue,
        DecimalValue,
        IntValue,
        JsonValue,
        TextValue,
    )

    # Text: verbatim.
    if isinstance(type_obj, TextType):
        if not isinstance(raw, str):
            raise ValueError(
                f"Param {name!r}: expected a text value (str), got {type(raw).__name__}"
            )
        return TextValue(raw)

    # Structured types (list/dict/record/enum): delegate to JsonCodec.
    if isinstance(type_obj, (ListType, DictType, RecordType, EnumType)):
        from agm.agl.runtime.codec import JsonCodec
        from agm.agl.type_schema import derive_schema

        # Accept either a JSON string or a Python native object (dict/list/
        # scalar) that was already parsed from JSON.  For native objects we
        # re-serialize to a JSON string so the codec can validate and convert
        # using the full type-aware path.  Decimal values are serialized
        # losslessly using dumps_exact, which emits them as unquoted numeric
        # text so the codec's json.loads(parse_float=Decimal) round-trip is
        # exact — avoiding the old default=str bug that turned Decimal("1.5")
        # into the JSON string "1.5" and failed schema validation.
        if isinstance(raw, str):
            json_str = raw
        elif _is_json_shaped(raw):
            from agm.agl.runtime.serialize import dumps_exact

            json_str = dumps_exact(raw, indent=None)
        else:
            raise ValueError(
                f"Param {name!r} has type {type_obj!r}; structured params must be "
                "provided as a JSON string or a JSON-compatible Python value "
                f"(got {type(raw).__name__!r})."
            )
        codec = JsonCodec()
        # Precompute schema once (CARRY-IN 2: avoids re-derivation inside parse).
        schema = derive_schema(type_obj)
        # Host-supplied param values are not chatty agent output: they must be
        # exactly one bare JSON value (F7).  Strict parsing avoids json-repair
        # silently "fixing" user typos.
        result = codec.parse(json_str, type_obj, strict_json=True, schema=schema)
        if not result.ok or result.value is None:
            raise ValueError(
                f"Param {name!r}: could not parse as {type_obj!r}; structured "
                f"params must be exactly one valid JSON value: {result.error_msg}"
            )
        return result.value

    # Scalar non-text (int/decimal/bool/json): parse from JSON if given as string.
    if not isinstance(type_obj, (IntType, DecimalType, BoolType, JsonType)):
        raise ValueError(
            f"Param {name!r} has unsupported type {type_obj!r}."
        )

    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_float=_decimal.Decimal)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Param {name!r}: could not parse as JSON: {exc}"
            ) from exc

    if isinstance(type_obj, IntType):
        if isinstance(value, int) and not isinstance(value, bool):
            return IntValue(value)
        if isinstance(value, _decimal.Decimal) and value == int(value):
            return IntValue(int(value))
        raise ValueError(
            f"Param {name!r}: expected an integer, got {type(value).__name__} {value!r}"
        )

    if isinstance(type_obj, DecimalType):
        if isinstance(value, _decimal.Decimal):
            return DecimalValue(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return DecimalValue(_decimal.Decimal(value))
        raise ValueError(
            f"Param {name!r}: expected a decimal, got {type(value).__name__} {value!r}"
        )

    if isinstance(type_obj, BoolType):
        if isinstance(value, bool):
            return BoolValue(value)
        raise ValueError(
            f"Param {name!r}: expected a bool, got {type(value).__name__} {value!r}"
        )

    # JsonType: accept any parsed JSON value.
    return JsonValue(value)


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
