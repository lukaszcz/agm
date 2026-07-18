"""Output codecs for the AgL runtime.

``OutputCodec`` is a protocol that every codec must satisfy.  The built-in
codecs are:

- ``TextCodec`` — passthrough for the ``text`` type.
- ``JsonCodec`` — structured output with lenient-by-default JSON recovery:
  extracts exactly one JSON value from chatty output (fences/prose) using
  ``json-repair``, then validates strictly via ``jsonschema``.

Codec names (e.g. ``TextCodec.name == "text"``) are the values used in
``HostCapabilities.codec_kinds`` and ``OutputContractSpec.codec_name``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

import json_repair
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonschemaValidationError

from agm.agl.ir.contracts import (
    ContractRequest,
    DecodeSchema,
    DictDecode,
    EnumDecode,
    ListDecode,
    RecordDecode,
    RefDecode,
)
from agm.agl.runtime.convert import _EMPTY_DEFS, decode_value, normalize_integral_decimals
from agm.agl.runtime.request import ValidationError
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import TextType, Type
from agm.agl.semantics.values import TextValue, Value
from agm.agl.type_schema import build_format_instructions, derive_schema_and_decode

if TYPE_CHECKING:
    from agm.agl.runtime.contract import OutputContract

DecodeDefsInput = Mapping[str, DecodeSchema] | tuple[tuple[str, DecodeSchema], ...]


# ---------------------------------------------------------------------------
# ParseResult — outcome of codec.parse()
# ---------------------------------------------------------------------------


class ParseResult:
    """The result of parsing a raw agent-response string through a codec.

    ``ok``              — True iff parsing and validation succeeded.
    ``value``           — The typed Value on success; ``None`` on failure.
    ``error_msg``       — A human-readable failure description (empty on success).
    ``errors``          — Structured :class:`ValidationError` records describing
                          schema-validation failures.  Empty
                          on success and for non-validation failures (e.g. no JSON
                          could be extracted, ambiguous multi-value output).
    ``normalized_raw``  — The canonical JSON text that was actually parsed (after
                          fence stripping / repair), or ``None`` on failure.
                          Design : "the normalized (recovered) value is traced
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
        cls,
        msg: str,
        errors: tuple[ValidationError, ...] = (),
        normalized_raw: str | None = None,
    ) -> "ParseResult":
        return cls(
            ok=False,
            value=None,
            error_msg=msg,
            errors=errors,
            normalized_raw=normalized_raw,
        )


# ---------------------------------------------------------------------------
# OutputCodec protocol
# ---------------------------------------------------------------------------


class OutputCodec(Protocol):
    """Protocol for AgL output codecs.

    Every codec exposes:
    - ``name`` — the codec identifier (e.g. ``"text"``, ``"json"``).
    - ``supported_kinds`` — frozenset of semantic type-kind strings this codec handles.
      This is the authoritative source for ``HostCapabilities.codec_kinds``.
    - ``supports_type(t)`` — True iff this codec can handle the given type.
    - ``make_contract(type_ref, type_table)`` — build an ``OutputContract``.
      Runs at check time (or REPL contract-preview time), when a real checker
      ``Type`` is in hand.  ``type_table`` resolves record/enum field/variant
      shapes for *type_ref* (or one nested inside it); ``None`` is only valid
      when *type_ref* carries no nominal type.
    - ``parse(raw, *, strict_json, schema, decode, defs)`` — parse a raw string.
      Runs at execution time against the typeless contract data the lowerer
      already compiled (``schema`` is the JSON Schema dict, ``decode`` the
      typeless ``DecodeSchema`` walk, ``defs`` its ``$defs`` table for a
      recursive target type — empty/absent for a non-recursive one); a codec
      never sees a checker ``Type`` at parse time.
    """

    @property
    def name(self) -> str: ...

    @property
    def supported_kinds(self) -> frozenset[str]: ...

    def supports_type(self, t: Type) -> bool: ...

    def make_contract(
        self, type_ref: Type, type_table: TypeTable | None = None
    ) -> "OutputContract": ...

    def parse(
        self,
        raw: str,
        *,
        strict_json: bool = False,
        schema: dict[str, object] | None = None,
        decode: DecodeSchema | None = None,
        defs: DecodeDefsInput | None = None,
    ) -> ParseResult: ...


# ---------------------------------------------------------------------------
# Built-in codec names
# ---------------------------------------------------------------------------

# The names of the built-in codecs (``TextCodec.name`` / ``JsonCodec.name``).
# These are reserved: host-registered codecs cannot shadow them.
BUILTIN_CODEC_NAMES: frozenset[str] = frozenset({"text", "json"})


# ---------------------------------------------------------------------------
# TextCodec — passthrough codec for the text type
# ---------------------------------------------------------------------------


class TextCodec:
    """The built-in ``text`` codec: passthrough, no parsing needed.

    For a ``text`` target, the raw agent response is returned as-is, wrapped
    in a ``TextValue``.  ``strict_json`` and ``schema`` are ignored (inapplicable).
    """

    @property
    def name(self) -> str:
        return "text"

    @property
    def supported_kinds(self) -> frozenset[str]:
        """The set of type-kind strings this codec can handle.

        Single source of truth for ``HostCapabilities.codec_kinds["text"]``,
        avoiding a duplicated literal at the host-environment assembly site.
        """
        return frozenset({"text"})

    def supports_type(self, t: Type) -> bool:
        return isinstance(t, TextType)

    def make_contract(
        self, type_ref: Type, type_table: TypeTable | None = None
    ) -> "OutputContract":
        """Build an ``OutputContract`` for *type_ref*.

        For ``text`` targets ``format_instructions`` is left empty (absent):
        a text target imposes no format on the agent's response, so there are
        no instructions to relay.  ``decode`` is ``None``: a text target has
        no schema-driven decode walk, since the raw string is the value.
        *type_table* is accepted for protocol conformance but unused — a
        ``text`` target never carries a nominal type.
        """
        from agm.agl.runtime.contract import OutputContract

        return OutputContract(
            target_type_label=repr(type_ref),
            codec=self,
            strict_json=None,
            format_instructions="",
            json_schema=None,
            decode=None,
        )

    def parse(
        self,
        raw: str,
        *,
        strict_json: bool = False,
        schema: dict[str, object] | None = None,
        decode: DecodeSchema | None = None,
        defs: DecodeDefsInput | None = None,
    ) -> ParseResult:
        # Text codec: always succeeds; the raw string is the value.
        # ``strict_json``/``schema``/``decode``/``defs`` are inapplicable for text targets.
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
# multi-value result.  See  ruling.
_AMBIGUOUS_MULTI_VALUE = object()


def _count_top_level_values(candidate: str) -> int:
    """Count complete top-level JSON values in *candidate*.

    Scans the stripped candidate with :meth:`json.JSONDecoder.raw_decode` in a
    whitespace-skipping loop.  ``{"a": 1} {"b": 2}`` and ``{"a": [1]} {"b": 2}``
    both yield ``2`` regardless of which brackets appear; a single value (even a
    fenced or bracketed one) yields ``1``.  Returns the count, capped once two
    values are seen (the caller only distinguishes 0 / 1 / 2+).

    Design ; 2+ top-level values are
    ambiguous.  If the candidate is not parseable as a run of JSON values
    (e.g. trailing prose ``json-repair`` already cleaned up), the count
    reflects only the leading values it could decode.
    """
    decoder = json.JSONDecoder()
    index = 0
    length = len(candidate)
    count = 0
    while index < length:
        # Skip whitespace between top-level values.  A run of trailing
        # whitespace makes the next ``raw_decode`` raise, ending the scan.
        while index < length and candidate[index].isspace():
            index += 1
        try:
            decoded: tuple[object, int] = decoder.raw_decode(candidate, index)
        except json.JSONDecodeError:
            break
        end: int = decoded[1]
        count += 1
        if count >= 2:
            return count
        index = end
    return count


def _candidate_is_ambiguous_multi_value(candidate: str) -> bool:
    """Return True if *candidate* contains 2+ complete top-level JSON values."""
    return _count_top_level_values(candidate.strip()) >= 2


# JSON scalar keywords recoverable from prose (bool / null).
_SCALAR_KEYWORD_RE = re.compile(r"(?<![A-Za-z0-9_])(true|false|null)(?![A-Za-z0-9_])")
# JSON numbers recoverable from prose.
_SCALAR_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9_.])")


def _scan_bare_scalar(text: str) -> str | None | object:
    """Recover a single bare JSON scalar (bool/null/number) embedded in prose.

    Lenient recovery strips prose around a single JSON value.
    ``json-repair`` does not pull bare scalars out of prose (e.g.
    ``"The flag is:\\nfalse"``), so this fallback finds keyword/number tokens
    via word-boundary-anchored regexes and returns the value's JSON text when
    **exactly one** is present.  Two or more distinct scalar tokens are
    ambiguous (``_AMBIGUOUS_MULTI_VALUE``); none yields ``None``.
    """
    matches = [m.group(1) for m in _SCALAR_KEYWORD_RE.finditer(text)]
    matches += [m.group(1) for m in _SCALAR_NUMBER_RE.finditer(text)]
    if not matches:
        return None
    if len(matches) >= 2:
        return _AMBIGUOUS_MULTI_VALUE
    return matches[0]


def _extract_json_text(raw: str) -> str | None | object:
    """Extract a single JSON text from potentially chatty agent output.

    Strategy (lenient mode — design ):
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
       top-level values into an array.

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
        # Ambiguity: the fenced candidate holds 2+ top-level JSON values.
        if _candidate_is_ambiguous_multi_value(candidate):
            return _AMBIGUOUS_MULTI_VALUE
        # Try direct parse on fenced content first.
        ok2, direct2 = _try_direct_parse(candidate)
        if ok2:
            return direct2
        # Fall back to repair within the fence.
        repaired = json_repair.repair_json(candidate)
        if isinstance(repaired, str) and repaired and repaired not in ('""', "null"):
            return repaired

    # Step 2: repair_json on the whole string (handles prose-wrapped).
    # Ambiguity: the whole response holds 2+ top-level JSON values
    # (e.g. ``{"a":1} {"b":2}`` or ``{"a":[1]} {"b":2}``).
    if _candidate_is_ambiguous_multi_value(stripped):
        return _AMBIGUOUS_MULTI_VALUE
    repaired_full = json_repair.repair_json(stripped)
    if isinstance(repaired_full, str) and repaired_full and repaired_full not in ('""', "null"):
        return repaired_full

    # Step 3: recover a single bare scalar (bool/null/number) from prose that
    # ``json-repair`` cannot extract (e.g. ``"The flag is:\nfalse"``).
    return _scan_bare_scalar(stripped)


# ---------------------------------------------------------------------------
# Typeless validation-error classification and shared JSON parse core
# ---------------------------------------------------------------------------


def _path_sort_key(error: JsonschemaValidationError) -> str:
    """A stable, comparable sort key for a jsonschema error (by path)."""
    return "/".join(str(p) for p in error.path)


def _resolve_ref(decode: DecodeSchema, defs: Mapping[str, DecodeSchema]) -> DecodeSchema:
    """Resolve a ``RefDecode`` node through *defs*; return *decode* unchanged otherwise.

    Shared by the classification walkers below, which navigate a finite JSON
    error PATH (not the value graph) — resolving a ref as encountered always
    terminates, so no visited-set is needed here (contrast
    ``ir/validate.py::_check_decode_nominals``, which walks the whole decode
    plan and does track visited ``defs`` keys). An unknown key (should never
    happen for a well-formed contract) makes the ref opaque to the caller's
    ``isinstance`` checks, so navigation fails soft into the generic fallback
    message rather than raising — these walkers only refine an already-failed
    validation's message, never gate correctness.
    """
    while isinstance(decode, RefDecode):
        resolved = defs.get(decode.key)
        if resolved is None:
            return decode
        decode = resolved
    return decode


def _decode_contains_ref(decode: DecodeSchema) -> bool:
    """Return whether *decode* contains any ``RefDecode`` node."""
    if isinstance(decode, RefDecode):
        return True
    if isinstance(decode, ListDecode):
        return _decode_contains_ref(decode.elem)
    if isinstance(decode, DictDecode):
        return _decode_contains_ref(decode.value)
    if isinstance(decode, RecordDecode):
        return any(_decode_contains_ref(field_decode) for _name, field_decode in decode.fields)
    if isinstance(decode, EnumDecode):
        return any(
            _decode_contains_ref(field_decode)
            for variant in decode.variants
            for _name, field_decode in variant.fields
        )
    return False


def _coerce_decode_defs(defs: DecodeDefsInput | None) -> Mapping[str, DecodeSchema]:
    """Normalize parse-time decode defs from either contract storage shape."""
    if defs is None:
        return _EMPTY_DEFS
    if isinstance(defs, Mapping):
        return defs
    return dict(defs)


def _find_enum_decode_at_path(
    decode: DecodeSchema,
    path_elements: list[object],
    defs: Mapping[str, DecodeSchema] = _EMPTY_DEFS,
) -> EnumDecode | None:
    """Navigate the decode schema to find an ``EnumDecode`` at the given JSON path."""
    decode = _resolve_ref(decode, defs)
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
        else:
            return None
        decode = _resolve_ref(decode, defs)
    return decode if isinstance(decode, EnumDecode) else None


def _make_validation_error(
    error: object, decode_schema: DecodeSchema, defs: Mapping[str, DecodeSchema] = _EMPTY_DEFS
) -> ValidationError:
    """Map a jsonschema error into a structured :class:`ValidationError`."""
    if not isinstance(error, JsonschemaValidationError):
        return ValidationError(category="wrong_type", message=str(error), path="$", field=None)

    path = "$" + "".join(f".{p}" if isinstance(p, str) else f"[{p}]" for p in error.path)

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
        return _classify_enum_failure(error, path, decode_schema, defs)
    return ValidationError(category="wrong_type", message=error.message, path=path, field=None)


def _classify_enum_failure(
    error: JsonschemaValidationError,
    path: str,
    decode_schema: DecodeSchema,
    defs: Mapping[str, DecodeSchema] = _EMPTY_DEFS,
) -> ValidationError:
    """Classify a oneOf enum validation failure using the typeless ``DecodeSchema``."""
    instance = error.instance
    if not isinstance(instance, dict):
        return ValidationError(
            category="bad_case",
            message="Enum object did not match any known variant.",
            path=path,
            field=None,
        )

    case_val = instance.get("$case")
    if not isinstance(case_val, str):
        return ValidationError(
            category="bad_case",
            message='Enum object is missing a string "$case" tag.',
            path=path,
            field="$case",
        )

    enum_decode = _find_enum_decode_at_path(decode_schema, list(error.absolute_path), defs)
    if enum_decode is None:
        return ValidationError(
            category="bad_case",
            message='Enum object is missing a string "$case" tag.',
            path=path,
            field="$case",
        )

    known_variants = {v.name: v for v in enum_decode.variants}
    if case_val not in known_variants:
        valid = ", ".join(v.name for v in enum_decode.variants)
        return ValidationError(
            category="bad_case",
            message=f'Unknown "$case" {case_val!r} for enum {enum_decode.display_name!r}. '
            f"Valid variants: {valid}.",
            path=path,
            field="$case",
        )

    variant = known_variants[case_val]
    variant_field_names = [fname for fname, _ in variant.fields]
    for field_name in variant_field_names:
        if field_name not in instance:
            return ValidationError(
                category="missing_field",
                message=f"Enum variant {case_val!r} is missing field {field_name!r}.",
                path=path,
                field=field_name,
            )
    declared = set(variant_field_names) | {"$case"}
    for key in instance:
        if key not in declared:
            return ValidationError(
                category="unknown_field",
                message=f"Enum variant {case_val!r} has an unexpected field {key!r}.",
                path=path,
                field=key,
            )

    return ValidationError(
        category="bad_case",
        message="Enum object did not match the selected variant schema.",
        path=path,
        field=None,
    )


def _parse_json_core(
    raw: str,
    schema_dict: dict[str, object],
    decode_schema: DecodeSchema,
    defs: Mapping[str, DecodeSchema] = _EMPTY_DEFS,
    *,
    strict: bool,
) -> ParseResult:
    """Shared JSON parse core used by ``JsonCodec`` and the IR evaluator.

    Takes pre-compiled *schema_dict* (a JSON Schema dict) and *decode_schema*
    (a typeless ``DecodeSchema``) so the IR evaluator can call it with values
    already embedded in the ``ContractRequest`` without holding checker types.
    *defs* is *decode_schema*'s ``$defs`` table for a recursive target type
    (empty for a non-recursive one).
    """
    if strict:
        try:
            parsed_obj: object = json.loads(raw, parse_float=Decimal)
        except json.JSONDecodeError as exc:
            return ParseResult.failure(f"Strict JSON parse failed: {exc}")
        return _validate_and_decode_core(raw.strip(), parsed_obj, schema_dict, decode_schema, defs)

    json_text = _extract_json_text(raw)
    if json_text is _AMBIGUOUS_MULTI_VALUE:
        return ParseResult.failure(
            "Ambiguous agent response: multiple JSON values were found, but "
            "exactly one is required."
        )
    if json_text is None or not isinstance(json_text, str):
        return ParseResult.failure(
            f"Could not extract a JSON value from the agent response: {raw!r}"
        )
    try:
        parsed_obj = json.loads(json_text, parse_float=Decimal)
    except json.JSONDecodeError as exc:
        return ParseResult.failure(f"JSON parse failed after repair attempt: {exc}")
    return _validate_and_decode_core(json_text, parsed_obj, schema_dict, decode_schema, defs)


def _validate_and_decode_core(
    json_text: str,
    parsed_obj: object,
    schema_dict: dict[str, object],
    decode_schema: DecodeSchema,
    defs: Mapping[str, DecodeSchema] = _EMPTY_DEFS,
) -> ParseResult:
    """Validate *parsed_obj* against *schema_dict*, then decode to typed ``Value``."""
    normalized = normalize_integral_decimals(parsed_obj)
    validator = Draft202012Validator(schema_dict)
    raw_errors: list[JsonschemaValidationError] = list(validator.iter_errors(normalized))
    if raw_errors:
        errors_sorted = sorted(raw_errors, key=_path_sort_key)
        errors = tuple(_make_validation_error(e, decode_schema, defs) for e in errors_sorted)
        summary = "; ".join(e.message for e in errors)
        return ParseResult.failure(
            f"Schema validation failed: {summary}",
            errors=errors,
            normalized_raw=json_text,
        )
    try:
        value = decode_value(decode_schema, normalized, defs)
    except ValueError as exc:
        return ParseResult.failure(f"Value conversion failed: {exc}", normalized_raw=json_text)
    return ParseResult.success(value, normalized_raw=json_text)


def _parse_contract_output(
    raw: str,
    contract: ContractRequest,
    *,
    effective_strict: bool,
) -> ParseResult:
    """Parse a raw agent/exec response per a built-in-codec ``ContractRequest``.

    Handles the ``text`` passthrough and the ``json`` parse path, including
    defensive checks for missing ``json_schema`` and ``decode`` fields.
    Called by ``IrInterpreter._parse_host_output`` for built-in codecs.
    """
    if contract.codec_name == "text":
        return ParseResult.success(TextValue(raw))
    # json codec
    if contract.json_schema is None:
        return ParseResult.failure("ContractRequest has no json_schema for json codec")
    schema_raw: object = json.loads(contract.json_schema)
    if not isinstance(schema_raw, dict):
        return ParseResult.failure("ContractRequest json_schema is not a JSON object")
    if contract.decode is None:
        return ParseResult.failure("ContractRequest has no decode schema for json codec")
    return _parse_json_core(
        raw, schema_raw, contract.decode, dict(contract.defs), strict=effective_strict
    )


# ---------------------------------------------------------------------------
# Format instructions builder
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JsonCodec structured-output codec
# ---------------------------------------------------------------------------

# The type kinds this codec handles (matches Type.kind property strings).
_JSON_CODEC_KINDS: frozenset[str] = frozenset(
    {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
)


class JsonCodec:
    """Built-in ``json`` codec for structured AgL outputs.

    Parsing strategy:
    - **Lenient** (default): extract exactly one JSON value from chatty output
      (fences/prose), repair trivially malformed JSON via ``json-repair``
      (which returns a repaired JSON *string*), then re-parse with
      ``json.loads(parse_float=Decimal)`` to preserve decimal exactness.
      Validate against the derived JSON Schema.
    - **Strict** (``strict_json=True``): parse exactly one bare JSON value
      via stdlib ``json.loads`` (surrounding whitespace permitted; nothing
      else).  No fence stripping or repair.

    Schema validation is always strict in both modes (rules 3–6 of
    never relaxed).

    ``make_contract`` derives the JSON Schema and typeless decode walk once
    from a real checker ``Type`` (compile time / REPL contract preview).
    ``parse`` never derives them: it takes the schema dict and ``DecodeSchema``
    explicitly, so execution-time parsing runs entirely off the typeless
    contract data the lowerer already compiled — no checker ``Type`` involved.
    """

    @property
    def name(self) -> str:
        return "json"

    @property
    def supported_kinds(self) -> frozenset[str]:
        """The set of type-kind strings this codec can handle.

        Single source of truth for ``HostCapabilities.codec_kinds["json"]``,
        avoiding a duplicated literal at the host-environment assembly site.
        Matches ``_JSON_CODEC_KINDS`` (kept in this module as a local constant
        to drive ``supports_type``; the runtime no longer duplicates it).
        """
        return _JSON_CODEC_KINDS

    def supports_type(self, t: Type) -> bool:
        return t.kind in _JSON_CODEC_KINDS

    def make_contract(
        self, type_ref: Type, type_table: TypeTable | None = None
    ) -> "OutputContract":
        """Build an ``OutputContract`` for *type_ref*.

        Derives the JSON Schema, format instructions, and typeless decode
        walk once, from the real checker ``Type``.  Compile time / REPL
        contract-preview use only: execution-time parsing never calls this —
        it uses the ``json_schema``/``decode`` the lowerer already compiled
        into the IR contract request.

        *type_table* resolves record/enum field/variant shapes.  ``None`` is
        only valid when *type_ref* carries no nominal type: passing ``None``
        for a record/enum target is an internal error, surfaced as the
        ``KeyError`` an empty table's lookup naturally raises rather than a
        user-facing diagnostic.
        """
        from agm.agl.runtime.contract import OutputContract

        table = type_table if type_table is not None else TypeTable()
        schema, decode_plan = derive_schema_and_decode(type_ref, table)
        instructions = build_format_instructions(schema)
        return OutputContract(
            target_type_label=repr(type_ref),
            codec=self,
            strict_json=False,  # default; overridden per call-site
            format_instructions=instructions,
            json_schema=schema,
            decode=decode_plan.root,
            defs=decode_plan.defs,
        )

    def parse(
        self,
        raw: str,
        *,
        strict_json: bool = False,
        schema: dict[str, object] | None = None,
        decode: DecodeSchema | None = None,
        defs: DecodeDefsInput | None = None,
    ) -> ParseResult:
        """Parse *raw* agent output into the typed ``Value`` described by *schema*/*decode*.

        Lenient mode (``strict_json=False``, the default per design ):
          1. Attempt to extract/repair exactly one JSON text from *raw*.
          2. Re-parse the repaired text with ``json.loads(parse_float=Decimal)``.
          3. Validate against *schema*.
          4. Convert to the appropriate typed ``Value`` per *decode*.

        Strict mode (``strict_json=True``):
          1. ``json.loads`` on the stripped raw string — no repair, no fence
             stripping.  Fails if there is any surrounding non-whitespace.
          2. Validate and convert as in lenient mode.

        *schema* and *decode* are the JSON Schema dict and typeless
        ``DecodeSchema`` walk for the target type — both required.  *defs* is
        *decode*'s ``$defs`` table for a recursive target type; absent (or
        ``None``) for a non-recursive one.  Callers (the IR evaluator, or a
        test exercising this codec directly) must supply *schema*/*decode*
        explicitly; this method never derives them from a checker ``Type``,
        so there is no re-derivation cost per parse attempt.

        Decimal exactness: ``json-repair`` always produces a
        JSON *string* (not Python objects), which is then re-parsed via
        ``json.loads(parse_float=Decimal)``.  Decimal values are never
        routed through Python ``float``.

        :raises ValueError: if *schema* or *decode* is ``None``.
        """
        if schema is None or decode is None:
            raise ValueError(
                "JsonCodec.parse requires an explicit schema and decode walk; "
                "it no longer derives them from a checker Type. Pass the "
                "contract-carried json_schema/decode (see ContractRequest)."
            )
        effective_defs = _coerce_decode_defs(defs)
        if not effective_defs and _decode_contains_ref(decode):
            raise ValueError(
                "JsonCodec.parse requires defs when the decode walk contains RefDecode nodes."
            )
        return _parse_json_core(raw, schema, decode, effective_defs, strict=strict_json)
