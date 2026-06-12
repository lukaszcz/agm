"""Output codecs for the AgL runtime.

``OutputCodec`` is a protocol that every codec must satisfy.  The built-in
codecs are:

- ``TextCodec`` — passthrough for the ``text`` type.
- ``JsonCodec`` — structured output with lenient-by-default JSON recovery
  (design §2.8 / §9.3): extracts exactly one JSON value from chatty output
  (fences/prose) using ``json-repair``, then validates strictly via
  ``jsonschema``.

Codec names (e.g. ``TextCodec.name == "text"``) are the values used in
``HostCapabilities.codec_kinds`` and ``OutputContractSpec.codec_name``.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

import json_repair
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonschemaValidationError

from agm.agl.eval.values import (
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.agl.runtime.request import ValidationError
from agm.agl.typecheck.types import (
    BoolType,
    DecimalType,
    DictType,
    EnumType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
)

if TYPE_CHECKING:
    from agm.agl.runtime.contract import OutputContract
    from agm.agl.typecheck.env import TypeEnvironment

# ---------------------------------------------------------------------------
# ParseResult — outcome of codec.parse()
# ---------------------------------------------------------------------------


class ParseResult:
    """The result of parsing a raw agent-response string through a codec.

    ``ok``              — True iff parsing and validation succeeded.
    ``value``           — The typed Value on success; ``None`` on failure.
    ``error_msg``       — A human-readable failure description (empty on success).
    ``errors``          — Structured :class:`ValidationError` records describing
                          schema-validation failures (design §7.5 / §7.7).  Empty
                          on success and for non-validation failures (e.g. no JSON
                          could be extracted, ambiguous multi-value output).
    ``normalized_raw``  — The canonical JSON text that was actually parsed (after
                          fence stripping / repair), or ``None`` on failure.
                          Design §2.8: "the normalized (recovered) value is traced
                          alongside the raw output."
    """

    __slots__ = ("ok", "value", "error_msg", "errors", "normalized_raw")

    def __init__(
        self,
        *,
        ok: bool,
        value: Value | None,
        error_msg: str,
        errors: tuple[ValidationError, ...] = (),
        normalized_raw: str | None = None,
    ) -> None:
        self.ok = ok
        self.value = value
        self.error_msg = error_msg
        self.errors = errors
        self.normalized_raw = normalized_raw

    @classmethod
    def success(cls, value: Value, normalized_raw: str | None = None) -> "ParseResult":
        return cls(ok=True, value=value, error_msg="", normalized_raw=normalized_raw)

    @classmethod
    def failure(
        cls, msg: str, errors: tuple[ValidationError, ...] = ()
    ) -> "ParseResult":
        return cls(ok=False, value=None, error_msg=msg, errors=errors)


# ---------------------------------------------------------------------------
# OutputCodec protocol
# ---------------------------------------------------------------------------


class OutputCodec(Protocol):
    """Protocol for AgL output codecs.

    Every codec exposes:
    - ``name`` — the codec identifier (e.g. ``"text"``, ``"json"``).
    - ``supports_type(t)`` — True iff this codec can handle the given type.
    - ``make_contract(type_ref, env)`` — build an ``OutputContract``.
    - ``parse(raw, target_type, strict_json)`` — parse a raw string.
    """

    @property
    def name(self) -> str: ...

    def supports_type(self, t: Type) -> bool: ...

    def make_contract(self, type_ref: Type, env: "TypeEnvironment") -> "OutputContract": ...

    def parse(self, raw: str, target_type: Type, *, strict_json: bool = False) -> ParseResult: ...


# ---------------------------------------------------------------------------
# TextCodec — passthrough codec for the text type
# ---------------------------------------------------------------------------


class TextCodec:
    """The built-in ``text`` codec: passthrough, no parsing needed.

    For a ``text`` target, the raw agent response is returned as-is, wrapped
    in a ``TextValue``.  ``strict_json`` is ignored (inapplicable).
    """

    @property
    def name(self) -> str:
        return "text"

    def supports_type(self, t: Type) -> bool:
        return isinstance(t, TextType)

    def make_contract(self, type_ref: Type, env: "TypeEnvironment") -> "OutputContract":
        from agm.agl.runtime.contract import OutputContract

        return OutputContract(
            target_type=type_ref,
            codec=self,
            strict_json=None,
            format_instructions="Return plain text.",
            json_schema=None,
        )

    def parse(self, raw: str, target_type: Type, *, strict_json: bool = False) -> ParseResult:
        # Text codec: always succeeds; the raw string is the value.
        return ParseResult.success(TextValue(raw))


# ---------------------------------------------------------------------------
# JSON extraction helpers (lenient mode)
# ---------------------------------------------------------------------------

# Regex to capture a fenced block: ```json ... ``` or ``` ... ```
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


def _try_direct_parse(text: str) -> tuple[bool, str]:
    """Attempt a direct stdlib json.loads on *text* (stripped).

    Returns ``(success, text)`` where:
    - ``success=True`` and ``text`` is the stripped input if it is valid JSON.
    - ``success=False`` and ``text`` is empty if parsing fails.

    The returned text is the original *text* (not re-serialised), so
    ``json.loads(text, parse_float=Decimal)`` will preserve decimal precision.
    """
    try:
        json.loads(text, parse_float=Decimal)
        return True, text
    except json.JSONDecodeError:
        return False, ""


# Sentinel returned by extraction when the candidate yields an *ambiguous*
# multi-value result (design §2.8: "exactly one JSON value").  See F3 ruling.
_AMBIGUOUS_MULTI_VALUE = object()


def _first_significant_char(text: str) -> str:
    """Return the first non-whitespace character of *text* (``""`` if blank)."""
    stripped = text.lstrip()
    return stripped[0] if stripped else ""


def _repaired_is_ambiguous_multi_value(candidate: str, repaired: str) -> bool:
    """Detect ambiguous multi-value output recovered by ``json-repair`` (F3).

    ``json-repair`` collapses several concatenated top-level JSON values (e.g.
    ``{"a":1} {"b":2}``) into a single JSON *array*.  That is a genuine
    multi-value response, not a single value, and design §2.8 requires exactly
    one JSON value.

    We treat the recovery as ambiguous when **all** of:

    1. json-repair produced a top-level array, but
    2. the candidate's first significant character is not ``[`` (a bare or
       fenced array legitimately starts with ``[`` and must still parse), and
    3. the source candidate contains no ``[`` at all — i.e. json-repair
       *synthesized* the array brackets to wrap several fused values.  This
       distinguishes ``{"a":1} {"b":2}`` (synthetic wrapper → ambiguous) from
       prose-wrapped single arrays like ``Here you go:\\n[1, 2]`` (the brackets
       are already in the source → a single value, recovered normally).
    """
    if _first_significant_char(repaired) != "[":
        return False
    if _first_significant_char(candidate) == "[":
        return False
    if "[" in candidate:
        return False
    try:
        parsed: object = json.loads(repaired)
    except json.JSONDecodeError:  # pragma: no cover — repaired is valid JSON
        return False
    return isinstance(parsed, list)


def _extract_json_text(raw: str) -> str | None | object:
    """Extract a single JSON text from potentially chatty agent output.

    Strategy (lenient mode — design §2.8):
    0. Try direct stdlib ``json.loads`` on the stripped input — if it succeeds
       (bare valid JSON, possibly with surrounding whitespace), return the
       stripped text verbatim.  This preserves full decimal precision since
       we never route through ``json-repair`` for already-valid JSON.
    1. Check for a Markdown code fence (```json ... ``` or ``` ... ```).
       If found, try direct parse on the fenced content; if that fails,
       try ``repair_json`` on the fenced content.
    2. Fall back to ``repair_json`` on the whole raw string (handles
       prose-wrapped JSON such as "Here you go:\\n{...}").
    3. Return ``None`` if no JSON value could be extracted, or the
       ``_AMBIGUOUS_MULTI_VALUE`` sentinel if json-repair fused several
       top-level values into an array (F3 — design §2.8 "exactly one JSON
       value").

    When ``json-repair`` is needed, it returns the repaired JSON *text*
    (without ``return_objects=True``), which is then re-parsed with
    ``json.loads(parse_float=Decimal)``.  Note that ``json-repair`` may
    lose decimal precision for very high-precision numbers; the direct-parse
    path (step 0 / step 1 inner) avoids this.
    """
    stripped = raw.strip()

    # Step 0: already valid JSON → return as-is, full precision preserved.
    ok, direct = _try_direct_parse(stripped)
    if ok:
        return direct

    # Step 1: fenced block.
    fence_match = _FENCE_RE.search(stripped)
    if fence_match:
        candidate: str = fence_match.group(1).strip()
        # Try direct parse on fenced content first.
        ok2, direct2 = _try_direct_parse(candidate)
        if ok2:
            return direct2
        # Fall back to repair within the fence.
        repaired = json_repair.repair_json(candidate)
        if isinstance(repaired, str) and repaired and repaired not in ('""', "null"):
            if _repaired_is_ambiguous_multi_value(candidate, repaired):
                return _AMBIGUOUS_MULTI_VALUE
            return repaired

    # Step 2: repair_json on the whole string (handles prose-wrapped).
    repaired_full = json_repair.repair_json(stripped)
    if isinstance(repaired_full, str) and repaired_full and repaired_full not in ('""', "null"):
        if _repaired_is_ambiguous_multi_value(stripped, repaired_full):
            return _AMBIGUOUS_MULTI_VALUE
        return repaired_full

    return None


# ---------------------------------------------------------------------------
# JSON → Value conversion helpers
# ---------------------------------------------------------------------------


def _json_to_value(obj: object, typ: Type) -> Value:
    """Convert a JSON-shaped Python object to the appropriate typed ``Value``.

    ``obj`` is the result of ``json.loads(parse_float=Decimal)`` — it may be
    ``dict``, ``list``, ``str``, ``int``, ``Decimal``, ``bool``, or ``None``.
    ``Decimal`` is never converted to ``float`` (design §5.1).

    Raises ``ValueError`` for type mismatches (the caller handles these).
    """
    if isinstance(typ, TextType):
        if isinstance(obj, str):
            return TextValue(obj)
        raise ValueError(f"Expected string, got {type(obj).__name__}")

    if isinstance(typ, IntType):
        if isinstance(obj, bool):
            raise ValueError("Expected integer, got bool")
        if isinstance(obj, int):
            return IntValue(obj)
        # Integral Decimals are normalized to ``int`` before validation/conversion
        # (see ``_normalize_integral_decimals``), so any Decimal reaching here is
        # non-integral and rejected for an int target.
        raise ValueError(f"Expected integer, got {type(obj).__name__} {obj!r}")

    if isinstance(typ, DecimalType):
        if isinstance(obj, bool):
            raise ValueError("Expected decimal, got bool")
        if isinstance(obj, Decimal):
            return DecimalValue(obj)
        if isinstance(obj, int):
            return DecimalValue(Decimal(obj))
        raise ValueError(f"Expected decimal, got {type(obj).__name__} {obj!r}")

    if isinstance(typ, BoolType):
        if isinstance(obj, bool):
            return BoolValue(obj)
        raise ValueError(f"Expected bool, got {type(obj).__name__}")

    if isinstance(typ, JsonType):
        # Accept any JSON-shaped value.
        return JsonValue(obj)

    if isinstance(typ, ListType):
        if not isinstance(obj, list):
            raise ValueError(f"Expected array, got {type(obj).__name__}")
        elements = tuple(_json_to_value(e, typ.elem) for e in obj)
        return ListValue(elements=elements)

    if isinstance(typ, DictType):
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object, got {type(obj).__name__}")
        entries: dict[str, Value] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"Dict key must be string, got {type(k).__name__}")
            entries[k] = _json_to_value(v, typ.value)
        return DictValue(entries=entries)

    if isinstance(typ, RecordType):
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object for record, got {type(obj).__name__}")
        fields: dict[str, Value] = {}
        for field_name, field_type in typ.fields.items():
            if field_name not in obj:
                raise ValueError(f"Missing field {field_name!r}")
            fields[field_name] = _json_to_value(obj[field_name], field_type)
        return RecordValue(type_name=typ.name, fields=fields)

    if isinstance(typ, EnumType):
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object for enum, got {type(obj).__name__}")
        case_val = obj.get("$case")
        if not isinstance(case_val, str):
            raise ValueError("Enum object must have a string '$case' field")
        variant_fields = typ.variants.get(case_val)
        if variant_fields is None:
            raise ValueError(
                f"Unknown enum variant {case_val!r} for {typ.name!r}. "
                f"Valid variants: {list(typ.variants.keys())}"
            )
        payload: dict[str, Value] = {}
        for field_name, field_type in variant_fields.items():
            if field_name not in obj:
                raise ValueError(
                    f"Enum variant {case_val!r} is missing field {field_name!r}"
                )
            payload[field_name] = _json_to_value(obj[field_name], field_type)
        return EnumValue(type_name=typ.name, variant=case_val, fields=payload)

    # ExceptionType is not wire-serialised by the JSON codec.
    raise ValueError(f"Cannot deserialise type {typ!r} from JSON")


# ---------------------------------------------------------------------------
# Decimal normalization (F2) and validation-error mapping (F1)
# ---------------------------------------------------------------------------


def _normalize_integral_decimals(obj: object) -> object:
    """Convert integral ``Decimal`` values to ``int`` throughout *obj*.

    Walks the JSON-shaped tree produced by ``json.loads(parse_float=Decimal)``
    and replaces any ``Decimal`` whose value is integral and lossless
    (``d == int(d)``) with the equivalent ``int``.  This lets a wire value of
    ``1.0`` satisfy an ``{"type": "integer"}`` schema; non-integral decimals
    such as ``1.5`` are preserved and continue to fail integer targets.

    Decimal targets re-widen ``int`` → ``Decimal`` via ``_json_to_value`` and a
    ``json`` passthrough sees a JSON-equal value, so this is loss-free.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, Decimal):
        if obj == obj.to_integral_value():
            return int(obj)
        return obj
    if isinstance(obj, list):
        items: list[object] = obj
        return [_normalize_integral_decimals(e) for e in items]
    if isinstance(obj, dict):
        mapping: dict[object, object] = obj
        return {k: _normalize_integral_decimals(v) for k, v in mapping.items()}
    return obj


def _missing_required_field(error: JsonschemaValidationError) -> str | None:
    """Return the absent field for a ``required`` validation error.

    jsonschema reports the full ``required`` list as ``validator_value``; the
    actually-missing field is whichever required name is not present in the
    (object) instance.
    """
    required = error.validator_value
    instance = error.instance
    if isinstance(required, list) and isinstance(instance, dict):
        for name in required:
            if isinstance(name, str) and name not in instance:
                return name
    return None


def _classify_jsonschema_error(
    error: JsonschemaValidationError, target_type: Type
) -> ValidationError:
    """Map a single jsonschema error into a structured :class:`ValidationError`.

    The mapping is type-directed so that opaque ``oneOf`` failures (enums) are
    reported as ``bad_case`` / ``missing_field`` rather than leaking the
    jsonschema phrasing "is not valid under any of the given schemas".
    """
    path = "$" + "".join(f".{p}" if isinstance(p, str) else f"[{p}]" for p in error.path)

    if error.validator == "required":
        name = _missing_required_field(error)
        return ValidationError(
            category="missing_field",
            message=error.message,
            path=path,
            field=name,
        )
    if error.validator == "additionalProperties":
        return ValidationError(
            category="unknown_field",
            message=error.message,
            path=path,
            field=None,
        )
    if error.validator == "type":
        # The offending field (if any) is the last path element.
        field = error.path[-1] if error.path else None
        name = field if isinstance(field, str) else None
        return ValidationError(
            category="wrong_type",
            message=error.message,
            path=path,
            field=name,
        )
    if error.validator == "oneOf":
        return _classify_enum_failure(error, target_type, path)
    # Any other validator (e.g. const inside a non-enum context): treat the
    # mismatch as a wrong-type failure with the original message.
    return ValidationError(
        category="wrong_type",
        message=error.message,
        path=path,
        field=None,
    )


def _classify_enum_failure(
    error: JsonschemaValidationError, target_type: Type, path: str
) -> ValidationError:
    """Type-directed classification of an enum ``oneOf`` validation failure.

    Inspects the failing instance against the *target_type* (an ``EnumType``
    when this branch is reached for an enum target) to decide whether the
    ``$case`` tag is missing/invalid (``bad_case``) or a known variant is
    missing a payload field (``missing_field``).
    """
    instance = error.instance
    # Locate the enum type governing the offending value.
    enum_type = _enum_type_at_path(target_type, list(error.path))
    if not isinstance(instance, dict) or enum_type is None:
        return ValidationError(
            category="bad_case",
            message="Enum object did not match any known variant.",
            path=path,
            field=None,
        )

    case = instance.get("$case")
    if not isinstance(case, str):
        return ValidationError(
            category="bad_case",
            message='Enum object is missing a string "$case" tag.',
            path=path,
            field="$case",
        )
    variant_fields = enum_type.variants.get(case)
    if variant_fields is None:
        valid = ", ".join(enum_type.variants.keys())
        return ValidationError(
            category="bad_case",
            message=f'Unknown "$case" {case!r} for enum {enum_type.name!r}. '
            f"Valid variants: {valid}.",
            path=path,
            field="$case",
        )
    # Known variant → report the first missing payload field (if any).
    for field_name in variant_fields:
        if field_name not in instance:
            return ValidationError(
                category="missing_field",
                message=f'Enum variant {case!r} is missing field {field_name!r}.',
                path=path,
                field=field_name,
            )
    # Otherwise an unknown payload field was supplied.
    declared = set(variant_fields) | {"$case"}
    for key in instance:
        if key not in declared:
            return ValidationError(
                category="unknown_field",
                message=f"Enum variant {case!r} has an unexpected field {key!r}.",
                path=path,
                field=key,
            )
    return ValidationError(  # pragma: no cover — defensive fallback
        category="bad_case",
        message="Enum object did not match the selected variant schema.",
        path=path,
        field=None,
    )


def _enum_type_at_path(target_type: Type, path: list[object]) -> EnumType | None:
    """Resolve the ``EnumType`` governing the value at *path* within *target_type*."""
    current: Type = target_type
    for step in path:
        if isinstance(current, ListType):
            current = current.elem
        elif isinstance(current, DictType):
            current = current.value
        elif isinstance(current, RecordType) and isinstance(step, str):
            field_type = current.fields.get(step)
            if field_type is None:
                return None
            current = field_type
        else:
            return None
    return current if isinstance(current, EnumType) else None


def _collect_validation_errors(
    obj: object, target_type: Type, schema: dict[str, object]
) -> tuple[ValidationError, ...]:
    """Validate *obj* against *schema*, returning structured errors (empty if OK)."""
    validator = Draft202012Validator(schema)
    raw_errors: list[JsonschemaValidationError] = list(validator.iter_errors(obj))
    # Deterministic ordering by the (string) JSON path; path elements may mix
    # ``str`` and ``int`` and are not directly comparable.
    errors = sorted(raw_errors, key=_path_sort_key)
    return tuple(_classify_jsonschema_error(e, target_type) for e in errors)


def _path_sort_key(error: JsonschemaValidationError) -> str:
    """A stable, comparable sort key for a jsonschema error (by path)."""
    return "/".join(str(p) for p in error.path)


# ---------------------------------------------------------------------------
# Format instructions builders
# ---------------------------------------------------------------------------


def _field_kind_label(typ: Type) -> str:
    """Return a human-readable type label for a field in format instructions."""
    if isinstance(typ, TextType):
        return "string"
    if isinstance(typ, IntType):
        return "integer"
    if isinstance(typ, DecimalType):
        return "number"
    if isinstance(typ, BoolType):
        return "boolean"
    if isinstance(typ, JsonType):
        return "any JSON value"
    if isinstance(typ, ListType):
        return f"array of {_field_kind_label(typ.elem)}"
    if isinstance(typ, DictType):
        return f"object with {_field_kind_label(typ.value)} values"
    if isinstance(typ, RecordType):
        return typ.name
    if isinstance(typ, EnumType):
        return typ.name
    return repr(typ)


def _build_format_instructions(typ: Type) -> str:
    """Build human-readable format instructions for *typ* (design §7.3/§7.4)."""
    if isinstance(typ, RecordType):
        field_lines = "\n".join(
            f"- {name}: {_field_kind_label(ftype)}" for name, ftype in typ.fields.items()
        )
        return (
            "Return exactly one JSON object.\n"
            "Do not include Markdown, prose, or code fences.\n"
            "The JSON must have exactly these fields:\n"
            f"{field_lines}"
        )
    if isinstance(typ, EnumType):
        variant_lines: list[str] = []
        for variant_name, variant_fields in typ.variants.items():
            if not variant_fields:
                variant_lines.append(f'{{ "$case": "{variant_name}" }}')
            else:
                field_parts = ", ".join(
                    f'"{fn}": [...]' if isinstance(ft, ListType) else f'"{fn}": ...'
                    for fn, ft in variant_fields.items()
                )
                variant_lines.append(f'{{ "$case": "{variant_name}", {field_parts} }}')
        valid_shapes = "\n".join(variant_lines)
        return (
            "Return exactly one JSON object.\n"
            "Do not include Markdown, prose, or code fences.\n"
            'Use "$case" to identify the selected variant.\n'
            "\n"
            "Valid shapes:\n"
            f"{valid_shapes}"
        )
    # Scalar / list / dict / json: generic instructions.
    return (
        "Return exactly one JSON value.\n"
        "Do not include Markdown, prose, or code fences."
    )


# ---------------------------------------------------------------------------
# JsonCodec — M2 structured-output codec
# ---------------------------------------------------------------------------

# The type kinds this codec handles (matches Type.kind property strings).
_JSON_CODEC_KINDS: frozenset[str] = frozenset(
    {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
)


class JsonCodec:
    """Built-in ``json`` codec for structured AgL outputs (M2).

    Parsing strategy (design §2.8):
    - **Lenient** (default): extract exactly one JSON value from chatty output
      (fences/prose), repair trivially malformed JSON via ``json-repair``
      (which returns a repaired JSON *string*), then re-parse with
      ``json.loads(parse_float=Decimal)`` to preserve decimal exactness
      (design §5.1).  Validate against the derived JSON Schema.
    - **Strict** (``strict_json=True``): parse exactly one bare JSON value
      via stdlib ``json.loads`` (surrounding whitespace permitted; nothing
      else).  No fence stripping or repair.

    Schema validation is always strict in both modes (rules 3–6 of §2.8 are
    never relaxed).
    """

    @property
    def name(self) -> str:
        return "json"

    def supports_type(self, t: Type) -> bool:
        return t.kind in _JSON_CODEC_KINDS

    def make_contract(self, type_ref: Type, env: "TypeEnvironment") -> "OutputContract":
        """Build an ``OutputContract`` for *type_ref* (design §7.7)."""
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.runtime.schema import derive_schema

        schema = derive_schema(type_ref)
        instructions = _build_format_instructions(type_ref)
        return OutputContract(
            target_type=type_ref,
            codec=self,
            strict_json=False,  # default; overridden per call-site
            format_instructions=instructions,
            json_schema=schema,
        )

    def parse(self, raw: str, target_type: Type, *, strict_json: bool = False) -> ParseResult:
        """Parse *raw* agent output into the typed ``Value`` for *target_type*.

        Lenient mode (``strict_json=False``, the default per design §2.8):
          1. Attempt to extract/repair exactly one JSON text from *raw*.
          2. Re-parse the repaired text with ``json.loads(parse_float=Decimal)``.
          3. Validate against the derived JSON Schema.
          4. Convert to the appropriate typed ``Value``.

        Strict mode (``strict_json=True``):
          1. ``json.loads`` on the stripped raw string — no repair, no fence
             stripping.  Fails if there is any surrounding non-whitespace.
          2. Validate and convert as in lenient mode.

        Decimal exactness (design §5.1): ``json-repair`` always produces a
        JSON *string* (not Python objects), which is then re-parsed via
        ``json.loads(parse_float=Decimal)``.  Decimal values are never
        routed through Python ``float``.
        """
        from agm.agl.runtime.schema import derive_schema

        schema = derive_schema(target_type)

        if strict_json:
            return self._parse_strict(raw, target_type, schema)
        return self._parse_lenient(raw, target_type, schema)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_strict(
        self, raw: str, target_type: Type, schema: dict[str, object]
    ) -> ParseResult:
        """Strict JSON parsing: stdlib json.loads, no repair or extraction."""
        try:
            parsed_obj: object = json.loads(raw, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            return ParseResult.failure(f"Strict JSON parse failed: {exc}")

        return self._validate_and_convert(raw.strip(), parsed_obj, target_type, schema)

    def _parse_lenient(
        self, raw: str, target_type: Type, schema: dict[str, object]
    ) -> ParseResult:
        """Lenient JSON recovery: fence stripping + json-repair + re-parse."""
        json_text = _extract_json_text(raw)
        if json_text is _AMBIGUOUS_MULTI_VALUE:
            return ParseResult.failure(
                "Ambiguous agent response: multiple JSON values were found, but "
                "exactly one is required (design §2.8)."
            )
        if json_text is None or not isinstance(json_text, str):
            return ParseResult.failure(
                f"Could not extract a JSON value from the agent response: {raw!r}"
            )

        try:
            parsed_obj: object = json.loads(json_text, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            return ParseResult.failure(
                f"JSON parse failed after repair attempt: {exc}"
            )

        return self._validate_and_convert(json_text, parsed_obj, target_type, schema)

    def _validate_and_convert(
        self,
        json_text: str,
        parsed_obj: object,
        target_type: Type,
        schema: dict[str, object],
    ) -> ParseResult:
        """Validate *parsed_obj* against *schema*, then convert to typed Value."""
        # F2: normalize integral Decimals to int before validation so that e.g.
        # ``1.0`` satisfies ``{"type": "integer"}`` (and decimal targets re-widen
        # via the int→Decimal path in ``_json_to_value``).  ``1.5`` is left as a
        # Decimal and still fails integer targets.
        normalized_obj = _normalize_integral_decimals(parsed_obj)

        # Schema validation (always strict — design §2.8 rules 3–6).
        errors = _collect_validation_errors(normalized_obj, target_type, schema)
        if errors:
            summary = "; ".join(e.message for e in errors)
            return ParseResult.failure(
                f"Schema validation failed: {summary}", errors=errors
            )

        # Convert to typed Value.
        try:
            value = _json_to_value(normalized_obj, target_type)
        except ValueError as exc:
            return ParseResult.failure(f"Value conversion failed: {exc}")

        return ParseResult.success(value, normalized_raw=json_text)
