"""Strict-parse primitives and the typeless decode walk for AgL.

This module is the canonical home for the reusable building blocks shared by
the cast path (``as`` / ``as?`` operators), the agent/exec output codec, and
host parameter decoding:

- :exc:`StrictJsonParseError` — raised by :func:`parse_json_strict` on any
  malformed or non-conforming input.
- :func:`parse_json_strict` — strict ``json.loads`` with ``parse_float=Decimal``
  and ``parse_constant`` that rejects non-standard constants
  (``NaN`` / ``Infinity`` / ``-Infinity``) even when nested inside containers.
  Also rejects any trailing/leading non-whitespace and a lone surrogate
  escape.  Returns the raw parsed Python object.
- :func:`agl_validator_class` / :func:`validator_for_schema` — the Draft
  2020-12 validator AgL uses everywhere, with a Decimal-aware ``integer``
  check.
- :func:`_clean_validation_message` — strip Python ``Decimal(...)`` reprs from
  jsonschema error messages before surfacing them to users.
- :func:`decode_value` / :func:`_decode_scalar` — the typeless
  ``DecodeSchema``-driven decode walk.  The single decode path shared by
  casts, the agent/exec codec, and host param decoding.  Raises ``ValueError``
  on any type mismatch; callers convert this into domain errors.  A
  ``RefDecode`` node is resolved through a *defs* mapping threaded alongside
  the walk (see :func:`decode_value`'s *defs* parameter) — the runtime mirror
  of a JSON Schema ``$ref``/``$defs`` pair for a recursive type.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, assert_never

from agm.agl.ir.contracts import (
    ArrayDecode,
    DecodeSchema,
    DictDecode,
    EnumDecode,
    RecordDecode,
    RefDecode,
    ScalarDecode,
    ScalarKind,
    resolve_schema_ref,
)
from agm.agl.ir.ids import NominalId
from agm.agl.semantics.values import (
    ArrayValue,
    BoolValue,
    DecimalValue,
    DictValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.util.unicode import loads_json

if TYPE_CHECKING:
    from jsonschema import TypeChecker
    from jsonschema import ValidationError as JsonschemaValidationError
    from jsonschema.protocols import Validator

# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------


class StrictJsonParseError(Exception):
    """Raised by :func:`parse_json_strict` when input is not a single valid JSON value.

    Covers:
    - Malformed JSON (syntax errors).
    - Non-standard constants: ``NaN``, ``Infinity``, ``-Infinity`` (including
      when nested inside containers such as ``[NaN]`` or ``{"x": Infinity}``).
    - Trailing or leading non-whitespace beyond the JSON value.
    - Empty / whitespace-only input.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# The AgL validator — Draft 2020-12 with a Decimal-aware "integer" type check
# ---------------------------------------------------------------------------


_VALIDATOR_CLASS: type[Validator] | None = None


def agl_validator_class() -> type[Validator]:
    """Return the Draft 2020-12 validator class every AgL schema check uses.

    Built on first use, so a run that validates nothing never imports
    ``jsonschema``: it and its dependencies cost more to import than every
    other third-party package the runtime loads put together.

    Its ``integer`` check also accepts an integral ``Decimal``: a wire number
    written with a fraction or exponent parses as ``Decimal``
    (``parse_float=Decimal``), and an ``int`` target accepts it when integral,
    as ``decimal as int`` would (:func:`_decode_scalar` then narrows it).
    Everything else uses the base Draft 2020-12 check; ``bool`` is never
    accepted.
    """
    global _VALIDATOR_CLASS
    if _VALIDATOR_CLASS is None:
        from jsonschema import Draft202012Validator
        from jsonschema.validators import extend

        base = Draft202012Validator.TYPE_CHECKER

        def is_integer_or_integral_decimal(checker: TypeChecker, instance: object) -> bool:
            if isinstance(instance, Decimal):
                return instance == instance.to_integral_value()
            return base.is_type(instance, "integer")

        _VALIDATOR_CLASS = extend(
            Draft202012Validator,
            type_checker=base.redefine("integer", is_integer_or_integral_decimal),
        )
    return _VALIDATOR_CLASS


_VALIDATOR_CACHE: dict[str, Validator] = {}


def validator_for_schema(json_schema: str) -> Validator:
    """Compile (and cache) an AgL validator from its canonical JSON string.

    Shared by the cast path (``conversions``) and host param decoding
    (``params``) so identical schemas are compiled once.
    """
    validator = _VALIDATOR_CACHE.get(json_schema)
    if validator is None:
        schema_obj: object = json.loads(json_schema)
        validator = agl_validator_class()(schema_obj)
        _VALIDATOR_CACHE[json_schema] = validator
    return validator


# ---------------------------------------------------------------------------
# parse_json_strict
# ---------------------------------------------------------------------------


def _reject_constant(c: str) -> object:
    """Raise :exc:`StrictJsonParseError` for any non-standard JSON constant.

    Passed as ``parse_constant`` to :func:`json.loads` so that ``NaN``,
    ``Infinity``, and ``-Infinity`` are rejected even when they appear nested
    inside containers such as ``[NaN]`` or ``{"x": Infinity}``.
    """
    raise StrictJsonParseError(f"Non-standard JSON constant {c!r} is not permitted in strict mode")


def parse_json_strict(text: str) -> object:
    """Parse *text* as a single strict JSON value.

    Rules:
    - Leading and trailing whitespace are ignored (as per JSON specification).
    - Exactly one JSON value must be present — no trailing junk.
    - Non-standard constants ``NaN``, ``Infinity``, and ``-Infinity`` are
      rejected even when nested inside containers (e.g. ``[NaN]``,
      ``{"x": Infinity}``).  They are not valid JSON.
    - Floating-point numbers are parsed as :class:`decimal.Decimal` (never
      ``float``), preserving exact precision.
    - A ``\\uD8xx``/``\\uDCxx`` escape that does not combine with an adjacent
      partner into one scalar character is rejected.

    :returns: The parsed Python object (``dict``, ``list``, ``str``, ``int``,
              :class:`decimal.Decimal`, ``bool``, or ``None``).
    :raises StrictJsonParseError: On any malformed or non-conforming input.
    """
    stripped = text.strip()
    if not stripped:
        raise StrictJsonParseError("Empty input: no JSON value found")

    try:
        # json.JSONDecoder.decode (used by json.loads) calls raw_decode and then
        # verifies that only whitespace follows the first value — so trailing junk
        # such as "42 extra" is already rejected with JSONDecodeError.
        #
        # parse_constant=_reject_constant ensures NaN/Infinity/-Infinity raise
        # StrictJsonParseError even when nested inside containers like [NaN].
        obj: object = loads_json(stripped, parse_float=Decimal, parse_constant=_reject_constant)
    except StrictJsonParseError:
        raise
    except ValueError as exc:
        raise StrictJsonParseError(f"JSON parse error: {exc}") from exc

    return obj


# ---------------------------------------------------------------------------
# _clean_validation_message
# ---------------------------------------------------------------------------


def _clean_validation_message(error: JsonschemaValidationError) -> str:
    """Return a clean validation error message without Python internal reprs.

    jsonschema error messages embed the Python repr of the ``instance``
    (e.g. ``Decimal('3.5') is not of type 'integer'``).  We reconstruct
    a cleaner message from the error's ``message`` with Decimal reprs
    replaced by their string equivalents.
    """
    # Replace occurrences of Decimal('...') with the bare number string.
    return re.sub(r"Decimal\('([^']*)'\)", r"\1", error.message)


# ---------------------------------------------------------------------------
# decode_value / _decode_scalar — the typeless DecodeSchema-driven decode walk
# ---------------------------------------------------------------------------

#: Shared empty *defs* mapping — the default for a non-recursive decode walk
#: (no ``RefDecode`` node can occur, so no ``$defs`` table is needed).
_EMPTY_DEFS: Mapping[str, DecodeSchema] = MappingProxyType({})

#: Resolves one omitted defaulted field's value at decode time. An ordinary
#: program's own default resolves from the declaring nominal's real,
#: fully-linked descriptor (``IrInterpreter.default_for_field`` is the
#: canonical implementation). A reserved record decoded before any program
#: exists (a host engine setting from a CLI flag or config entry) instead
#: resolves from its seeded ``TypeDef``'s own host-side constant
#: (``semantics.type_table.reserved_field_default``, wrapped by
#: ``runtime.engine_config``'s own resolver) — no evaluator is reachable
#: there. ``None`` (a compile-time contract preview, which decodes nothing):
#: a defaulted-but-omitted field then reports the ordinary "missing field"
#: error, same as an undefaulted one.
DefaultResolver = Callable[[NominalId, int], Value]


def decode_value(
    schema: DecodeSchema,
    obj: object,
    defs: Mapping[str, DecodeSchema] = _EMPTY_DEFS,
    *,
    default_resolver: DefaultResolver | None = None,
) -> Value:
    """Construct a typed ``Value`` from JSON-shaped *obj* per *schema*.

    The typeless ``DecodeSchema``-driven decode walk shared by the cast path
    (``as`` / ``as?``), the agent/exec output codec, and host param decoding.
    Raises ``ValueError`` on any type mismatch; callers convert this into the
    appropriate domain error (``AglCastConversion``, ``ValidationError``, etc.).
    A record/enum-variant field missing from *obj* fills from *default_resolver*
    when its ``default_index`` is set and a resolver is given; otherwise it
    raises the ordinary missing-field error (see *default_resolver*'s own
    docstring).

    *defs* resolves ``RefDecode`` nodes for a recursive target type — the
    ``$defs`` table built alongside *schema* by ``type_schema.derive_schema_and_decode``
    (see ``DecodePlan``); empty for a non-recursive *schema*, which then never
    contains a ``RefDecode`` node. Ref resolution follows chains until a
    non-ref body is reached, then decodes that body; this allows ordinary
    recursive bodies while rejecting malformed ref-only cycles. An unknown key
    or ref-only cycle indicates an inconsistent decode plan (a lowering bug,
    not a user-facing condition) since a well-formed plan's keys always match
    its own ``RefDecode`` occurrences one-to-one and always name real bodies.
    """
    match schema:
        case RefDecode(key=key):
            resolved = resolve_decode_ref(key, defs)
            return decode_value(resolved, obj, defs, default_resolver=default_resolver)
        case ScalarDecode(kind=kind):
            return _decode_scalar(kind, obj)
        case ArrayDecode(elem=elem):
            if not isinstance(obj, list):
                raise ValueError(f"Expected array, got {type(obj).__name__}")
            return ArrayValue(
                [decode_value(elem, e, defs, default_resolver=default_resolver) for e in obj]
            )
        case DictDecode(value=value_schema):
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object, got {type(obj).__name__}")
            entries: dict[str, Value] = {}
            for k, v in obj.items():
                if not isinstance(k, str):
                    raise ValueError(f"Dict key must be string, got {type(k).__name__}")
                entries[k] = decode_value(value_schema, v, defs, default_resolver=default_resolver)
            return DictValue(entries=entries)
        case RecordDecode(nominal=nominal, fields=fields):
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object for record, got {type(obj).__name__}")
            record_fields: dict[str, Value] = {}
            for rfield in fields:
                if rfield.json_name not in obj:
                    if rfield.default_index is None or default_resolver is None:
                        raise ValueError(f"Missing field {rfield.json_name!r}")
                    record_fields[rfield.name] = default_resolver(nominal, rfield.default_index)
                    continue
                record_fields[rfield.name] = decode_value(
                    rfield.schema, obj[rfield.json_name], defs, default_resolver=default_resolver
                )
            return RecordValue(nominal=nominal, fields=record_fields)
        case EnumDecode(display_name=display_name, variants=variants):
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object for enum, got {type(obj).__name__}")
            case_val = obj.get("$case")
            if not isinstance(case_val, str):
                raise ValueError("Enum object must have a string '$case' field")
            variant = next((v for v in variants if v.json_name == case_val), None)
            if variant is None:
                raise ValueError(
                    f"Unknown enum variant {case_val!r} for {display_name!r}. "
                    f"Valid variants: {[v.json_name for v in variants]}"
                )
            payload: dict[str, Value] = {}
            for vfield in variant.fields:
                if vfield.json_name not in obj:
                    if vfield.default_index is None or default_resolver is None:
                        raise ValueError(
                            f"Enum variant {case_val!r} is missing field {vfield.json_name!r}"
                        )
                    payload[vfield.name] = default_resolver(variant.nominal, vfield.default_index)
                    continue
                payload[vfield.name] = decode_value(
                    vfield.schema, obj[vfield.json_name], defs, default_resolver=default_resolver
                )
            return RecordValue(nominal=variant.nominal, fields=payload)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def resolve_decode_ref(key: str, defs: Mapping[str, DecodeSchema]) -> DecodeSchema:
    """Resolve a ``RefDecode`` key to a non-ref body, rejecting malformed cycles."""
    return resolve_schema_ref(
        key,
        defs,
        lambda schema: schema.key if isinstance(schema, RefDecode) else None,
        subject="decode_value: RefDecode",
    )


def _decode_scalar(kind: ScalarKind, obj: object) -> Value:
    """Decode a JSON scalar into the matching leaf ``Value``."""
    match kind:
        case ScalarKind.TEXT:
            if isinstance(obj, str):
                return TextValue(obj)
            raise ValueError(f"Expected string, got {type(obj).__name__}")
        case ScalarKind.INT:
            if isinstance(obj, bool):
                raise ValueError("Expected integer, got bool")
            if isinstance(obj, int):
                return IntValue(obj)
            if isinstance(obj, Decimal) and obj == obj.to_integral_value():
                return IntValue(int(obj))
            raise ValueError(f"Expected integer, got {type(obj).__name__} {obj!r}")
        case ScalarKind.DECIMAL:
            if isinstance(obj, bool):
                raise ValueError("Expected decimal, got bool")
            if isinstance(obj, Decimal):
                return DecimalValue(obj)
            if isinstance(obj, int):
                return DecimalValue(Decimal(obj))
            raise ValueError(f"Expected decimal, got {type(obj).__name__} {obj!r}")
        case ScalarKind.BOOL:
            if isinstance(obj, bool):
                return BoolValue(obj)
            raise ValueError(f"Expected bool, got {type(obj).__name__}")
        case ScalarKind.JSON:
            return JsonValue(obj)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)
