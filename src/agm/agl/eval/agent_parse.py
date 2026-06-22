"""Typeless agent-output parser for the IR evaluator (M6b).

``parse_agent_output`` parses a raw agent response string into a typed
``Value`` using a typeless ``ContractRequest`` — no checker ``Type`` reaches
this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonschemaValidationError

from agm.agl.eval.conversions import decode_value
from agm.agl.eval.values import TextValue, Value
from agm.agl.ir.contracts import ContractRequest, EnumDecode
from agm.agl.runtime.codec import _AMBIGUOUS_MULTI_VALUE, _extract_json_text
from agm.agl.runtime.convert import normalize_integral_decimals
from agm.agl.runtime.request import ValidationError

if TYPE_CHECKING:
    pass

__all__ = ["AgentParseResult", "parse_agent_output"]

# Cache compiled JSON-Schema validators (same caching pattern as eval/conversions.py)
_VALIDATOR_CACHE: dict[str, Draft202012Validator] = {}


def _validator_for(json_schema: str) -> Draft202012Validator:
    """Compile (and cache) a JSON-Schema validator from its canonical JSON string."""
    v = _VALIDATOR_CACHE.get(json_schema)
    if v is None:
        schema_obj: object = json.loads(json_schema)
        v = Draft202012Validator(schema_obj)
        _VALIDATOR_CACHE[json_schema] = v
    return v


@dataclass(slots=True)
class AgentParseResult:
    """Outcome of parsing a raw agent response through a contract.

    ``ok``              — True iff parsing and validation succeeded.
    ``value``           — The typed Value on success; None on failure.
    ``error_msg``       — Human-readable failure description (empty on success).
    ``errors``          — Structured ValidationError records (empty on success
                          and for non-validation failures like missing JSON).
    ``normalized_raw``  — Canonical JSON text that was actually parsed (after
                          fence stripping / repair), or None on failure.
    """

    ok: bool
    value: Value | None
    error_msg: str
    errors: tuple[ValidationError, ...] = field(default_factory=tuple)
    normalized_raw: str | None = None

    @classmethod
    def success(cls, value: Value, normalized_raw: str | None = None) -> "AgentParseResult":
        return cls(ok=True, value=value, error_msg="", normalized_raw=normalized_raw)

    @classmethod
    def failure(
        cls,
        msg: str,
        errors: tuple[ValidationError, ...] = (),
        normalized_raw: str | None = None,
    ) -> "AgentParseResult":
        return cls(
            ok=False, value=None, error_msg=msg, errors=errors, normalized_raw=normalized_raw
        )


def parse_agent_output(
    raw: str,
    contract: ContractRequest,
    *,
    effective_strict: bool,
) -> AgentParseResult:
    """Parse a raw agent response string per the typeless contract descriptor.

    Parameters
    ----------
    raw:
        The raw string returned by the agent.
    contract:
        The ``ContractRequest`` for this call site (no checker Type).
    effective_strict:
        The effective strict_json flag (contract.strict_json overrides evaluator
        default; the caller computes and passes the resolved value).

    Returns
    -------
    ``AgentParseResult`` with ``ok=True`` and a typed ``Value`` on success,
    or ``ok=False`` with error details on failure.
    """
    if contract.codec_name == "text":
        # Text codec: passthrough — raw string is the value.
        return AgentParseResult.success(TextValue(raw))

    # JSON codec path.
    schema_str = contract.json_schema
    if schema_str is None:
        return AgentParseResult.failure("ContractRequest has no json_schema for json codec")

    if effective_strict:
        return _parse_strict(raw, contract, schema_str)
    return _parse_lenient(raw, contract, schema_str)


def _parse_strict(
    raw: str, contract: ContractRequest, schema_str: str
) -> AgentParseResult:
    """Strict JSON parsing: stdlib json.loads, no repair or extraction."""
    try:
        parsed_obj: object = json.loads(raw, parse_float=Decimal)
    except json.JSONDecodeError as exc:
        return AgentParseResult.failure(f"Strict JSON parse failed: {exc}")
    return _validate_and_convert(raw.strip(), parsed_obj, contract, schema_str)


def _parse_lenient(
    raw: str, contract: ContractRequest, schema_str: str
) -> AgentParseResult:
    """Lenient JSON recovery: fence stripping + json-repair + re-parse."""
    json_text = _extract_json_text(raw)
    if json_text is _AMBIGUOUS_MULTI_VALUE:
        return AgentParseResult.failure(
            "Ambiguous agent response: multiple JSON values were found, but "
            "exactly one is required (design §2.8)."
        )
    if json_text is None or not isinstance(json_text, str):
        return AgentParseResult.failure(
            f"Could not extract a JSON value from the agent response: {raw!r}"
        )
    try:
        parsed_obj: object = json.loads(json_text, parse_float=Decimal)
    except json.JSONDecodeError as exc:
        return AgentParseResult.failure(f"JSON parse failed after repair attempt: {exc}")
    return _validate_and_convert(json_text, parsed_obj, contract, schema_str)


def _validate_and_convert(
    json_text: str,
    parsed_obj: object,
    contract: ContractRequest,
    schema_str: str,
) -> AgentParseResult:
    """Validate parsed_obj against schema, then decode to typed Value."""
    normalized_obj = normalize_integral_decimals(parsed_obj)

    schema_obj: object = json.loads(schema_str)
    if not isinstance(schema_obj, dict):
        return AgentParseResult.failure("Invalid JSON schema: not a dict")
    validator = _validator_for(schema_str)
    raw_errors = list(validator.iter_errors(normalized_obj))
    if raw_errors:
        sorted_errors = sorted(raw_errors, key=_path_sort_key)
        errors = tuple(
            _make_validation_error(e, contract) for e in sorted_errors
        )
        summary = "; ".join(e.message for e in sorted_errors)
        return AgentParseResult.failure(
            f"Schema validation failed: {summary}",
            errors=errors,
            normalized_raw=json_text,
        )

    if contract.decode is None:
        return AgentParseResult.failure("ContractRequest has no decode schema for json codec")
    try:
        value = decode_value(contract.decode, normalized_obj)
    except ValueError as exc:
        return AgentParseResult.failure(
            f"Value conversion failed: {exc}", normalized_raw=json_text
        )
    return AgentParseResult.success(value, normalized_raw=json_text)


def _path_sort_key(error: object) -> str:
    """A stable, comparable sort key for a jsonschema error (by path)."""
    if isinstance(error, JsonschemaValidationError):
        return "/".join(str(p) for p in error.path)
    return ""


def _make_validation_error(error: object, contract: ContractRequest) -> ValidationError:
    """Build a ValidationError from a jsonschema ValidationError.

    For oneOf/enum failures, uses the contract's DecodeSchema to identify
    which enum type is at the failing path so enum variant errors are classified
    identically to the legacy path's ``_classify_enum_failure``.
    """
    if not isinstance(error, JsonschemaValidationError):
        return ValidationError(category="wrong_type", message=str(error), path="$", field=None)

    path = "$" + "".join(
        f".{p}" if isinstance(p, str) else f"[{p}]" for p in error.path
    )

    if error.validator == "required":
        required = error.validator_value
        instance = error.instance
        name: str | None = None
        if isinstance(required, list) and isinstance(instance, dict):
            for n in required:
                if isinstance(n, str) and n not in instance:
                    name = n
                    break
        return ValidationError(
            category="missing_field", message=error.message, path=path, field=name
        )

    if error.validator == "additionalProperties":
        return ValidationError(
            category="unknown_field", message=error.message, path=path, field=None
        )

    if error.validator == "type":
        field_elem = error.path[-1] if error.path else None
        fname: str | None = field_elem if isinstance(field_elem, str) else None
        return ValidationError(category="wrong_type", message=error.message, path=path, field=fname)

    if error.validator == "oneOf":
        return _classify_enum_failure_typeless(error, path, contract)

    return ValidationError(category="wrong_type", message=error.message, path=path, field=None)


def _classify_enum_failure_typeless(
    error: JsonschemaValidationError,
    path: str,
    contract: ContractRequest,
) -> ValidationError:
    """Classify a oneOf enum validation failure using the DecodeSchema.

    Mirrors ``_classify_enum_failure`` in ``runtime/codec.py`` message-for-message,
    using ``EnumDecode`` (from the typeless ``ContractRequest.decode``) instead of
    the checker's ``EnumType``.  Results are byte-identical to the legacy path.
    """
    instance = error.instance
    if not isinstance(instance, dict):
        return ValidationError(
            category="bad_case",
            message="Enum object did not match any known variant.",
            path=path, field=None,
        )

    case_val = instance.get("$case")
    if not isinstance(case_val, str):
        return ValidationError(
            category="bad_case",
            message='Enum object is missing a string "$case" tag.',
            path=path, field="$case",
        )

    # Navigate the DecodeSchema to find the EnumDecode at this path.
    # If we can't find it, fall back to the missing-$case-tag message.
    enum_decode = _find_enum_decode_at_path(contract, list(error.absolute_path))
    if enum_decode is None:
        return ValidationError(
            category="bad_case",
            message='Enum object is missing a string "$case" tag.',
            path=path, field="$case",
        )

    # Check if case_val is a known variant — mirror legacy's "Valid variants: ..." message.
    known_variants = {v.name: v for v in enum_decode.variants}
    if case_val not in known_variants:
        valid = ", ".join(v.name for v in enum_decode.variants)
        return ValidationError(
            category="bad_case",
            message=f'Unknown "$case" {case_val!r} for enum {enum_decode.display_name!r}. '
            f"Valid variants: {valid}.",
            path=path, field="$case",
        )

    # case_val is a known variant — inspect the instance directly against declared fields
    # (mirrors legacy's loop over variant_fields, NOT via jsonschema sub-errors).
    variant = known_variants[case_val]
    variant_field_names = [fname for fname, _ in variant.fields]
    # First missing field → "Enum variant ... is missing field ..."
    for field_name in variant_field_names:
        if field_name not in instance:
            return ValidationError(
                category="missing_field",
                message=f"Enum variant {case_val!r} is missing field {field_name!r}.",
                path=path, field=field_name,
            )
    # Unknown instance key → "Enum variant ... has an unexpected field ..."
    declared = set(variant_field_names) | {"$case"}
    for key in instance:
        if key not in declared:
            return ValidationError(
                category="unknown_field",
                message=f"Enum variant {case_val!r} has an unexpected field {key!r}.",
                path=path, field=key,
            )

    return ValidationError(
        category="bad_case",
        message="Enum object did not match the selected variant schema.",
        path=path, field=None,
    )


def _find_enum_decode_at_path(
    contract: ContractRequest,
    path_elements: list[object],
) -> "EnumDecode | None":
    """Navigate the contract's DecodeSchema to find an EnumDecode at the given JSON path.

    Returns ``None`` when the path can't be navigated or the node isn't an EnumDecode.
    """
    from agm.agl.ir.contracts import (
        DecodeSchema,
        DictDecode,
        ListDecode,
        RecordDecode,
    )

    decode = contract.decode
    if decode is None:
        return None

    for elem in path_elements:
        if isinstance(decode, ListDecode):
            decode = decode.elem
        elif isinstance(decode, DictDecode):
            decode = decode.value
        elif isinstance(decode, RecordDecode):
            if not isinstance(elem, str):
                return None
            field_decode: DecodeSchema | None = None
            for fname, fschema in decode.fields:
                if fname == elem:
                    field_decode = fschema
                    break
            if field_decode is None:
                return None
            decode = field_decode
        elif isinstance(decode, EnumDecode):
            # Inside an enum field — shouldn't happen at this level but handle it.
            return None
        else:
            return None

    if isinstance(decode, EnumDecode):
        return decode
    return None
