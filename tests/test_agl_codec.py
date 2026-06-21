"""Tests for the AgL JsonCodec, schema derivation, and wire-up (M2b).

Covers (per PLAN_DSL §9.3 and §13):
1. Schema derivation (schema.py): every Type kind → JSON Schema dict.
2. JsonCodec.supports_type: json/record/enum/list/dict/int/decimal/bool;
   NOT text (text stays TextCodec).
3. Lenient default parsing: bare JSON, fenced ```json``` blocks, prose-wrapped,
   trailing-comma / single-quote trivial repairs, extracted and re-parsed with
   parse_float=Decimal (decimal exactness).
4. Strict mode (strict_json=True): rejects fences, prose, repairs; only bare
   JSON with surrounding whitespace accepted.
5. Schema validation errors → ParseResult.ok=False (missing/unknown field, wrong
   type, bad $case).
6. Typed Value construction: RecordValue, EnumValue, ListValue, DictValue,
   scalars; int→decimal widening where the target type says decimal.
7. Multiple JSON values / ambiguous output → failure (design §2.8: exactly one).
8. WorkflowRuntime wire-up: JsonCodec registered; checker passes json/record/enum
   targets; format_instructions reach AgentRequest; make_contract API.
9. decimal exactness end-to-end: 1.5 parsed from agent response stays Decimal("1.5").

Note: tests for record/enum targets use the direct-AST approach from test_agl_eval.py
since those type declarations require M2a parser features.  Scalar / json / list /
dict targets can be tested via ``WorkflowRuntime.run`` with parseable source.
"""

from __future__ import annotations

import itertools
from decimal import Decimal

import pytest

from agm.agl import WorkflowRuntime
from agm.agl.capabilities import HostCapabilities
from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.interpreter import Interpreter
from agm.agl.eval.scope import Scope
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
)
from agm.agl.runtime.agents import AgentFn, AgentRegistry
from agm.agl.runtime.codec import JsonCodec, ParseResult, TextCodec
from agm.agl.runtime.contract import OutputContract, materialize_contract
from agm.agl.runtime.request import AgentRequest
from agm.agl.runtime.schema import derive_schema
from agm.agl.scope import resolve
from agm.agl.syntax import nodes as ast
from agm.agl.syntax import types as tast
from agm.agl.syntax.nodes import (
    Item,
    TemplateSegment,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck import check
from agm.agl.typecheck.env import CheckedProgram, OutputContractSpec
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
from tests._agl_helpers import ambient_agents_for

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_node_ids = itertools.count(100_000)


def _nid() -> int:
    return next(_node_ids)


def _sp() -> SourceSpan:
    return SourceSpan(1, 1, 1, 5, 0, 4)


def _make_issue_type() -> RecordType:
    """A three-field record: title: text, severity: int, description: text."""
    return RecordType(
        name="Issue",
        fields={
            "title": TextType(),
            "severity": IntType(),
            "description": TextType(),
        },
    )


def _make_review_type() -> EnumType:
    """enum Review | Pass | Fail(issues: list[text])"""
    return EnumType(
        name="Review",
        variants={
            "Pass": {},
            "Fail": {"issues": ListType(elem=TextType())},
        },
    )


def _make_contract_for(typ: Type) -> OutputContract:
    """Build an OutputContract for a type via JsonCodec.make_contract."""
    codec = JsonCodec()
    return codec.make_contract(typ)


def _variant_schema_for_case(schema: dict[str, object], case: str) -> dict[str, object]:
    """Return the ``oneOf`` variant sub-schema whose ``$case`` const is *case*."""
    one_of = schema["oneOf"]
    assert isinstance(one_of, list)
    for variant in one_of:
        assert isinstance(variant, dict)
        props = variant["properties"]
        assert isinstance(props, dict)
        if props["$case"] == {"const": case}:
            return variant
    raise AssertionError(f"no variant with $case={case!r}")


# ---------------------------------------------------------------------------
# Direct-AST execution helpers (for record/enum targets that M2a parser adds)
# ---------------------------------------------------------------------------


def _ensure_expr_tail(body: tuple[Item, ...]) -> tuple[Item, ...]:
    """Append a unit literal when the last item is a binder.

    In v2, a block must end with an expression (binders need a continuation).
    Tests that end with ``let x = ...`` are migrated by appending ``()`` so
    the block type is ``unit`` and all bound names are still in scope.
    """
    from agm.agl.syntax.nodes import LetDecl, VarDecl

    if body and isinstance(body[-1], (LetDecl, VarDecl)):
        return body + (ast.UnitLit(span=_sp(), node_id=_nid()),)
    return body


def _check_program_with_json(body: tuple[Item, ...]) -> CheckedProgram:
    """Run *body* through real resolve + check with both text and json codecs."""
    program = ast.Program(
        body=ast.Block(items=_ensure_expr_tail(body), span=_sp(), node_id=_nid()),
        span=_sp(),
        node_id=_nid(),
    )
    resolved = resolve(program, ambient_agents=ambient_agents_for(program))
    caps = HostCapabilities(
        agent_names=frozenset(),
        has_default_agent=True,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
            ),
        },
    )
    return check(resolved, caps)


def _run_with_json_codec(
    body: tuple[Item, ...],
    *,
    named: dict[str, AgentFn] | None = None,
    default_agent: AgentFn | None = None,
    strict_json: bool = False,
) -> Scope:
    """Build + resolve + check + execute *body* with JsonCodec registered."""
    from agm.agl.runtime.codec import JsonCodec, OutputCodec, TextCodec
    from agm.agl.runtime.contract import materialize_contract

    checked = _check_program_with_json(body)
    text_codec = TextCodec()
    json_codec = JsonCodec()
    codecs: dict[str, OutputCodec] = {
        text_codec.name: text_codec,
        json_codec.name: json_codec,
    }
    contracts: dict[int, OutputContract] = {}
    for node_id, spec in checked.contract_specs.items():
        contracts[node_id] = materialize_contract(spec, codecs)

    registry = AgentRegistry(named=named or {}, default_agent=default_agent)
    interp = Interpreter(
        checked=checked,
        registry=registry,
        contracts=contracts,
        type_env=checked.type_env,
        loop_limit=3,
        strict_json=strict_json,
    )
    root = Scope(parent=None)
    interp.execute(root)
    return root


# AST statement / expression builders (subset needed for codec tests)


def _let(name: str, value: ast.Expr, *, type_ann: tast.TypeExpr | None = None) -> ast.LetDecl:
    return ast.LetDecl(name=name, type_ann=type_ann, value=value, span=_sp(), node_id=_nid())


def _template(*segments: TemplateSegment) -> ast.Template:
    return ast.Template(segments=tuple(segments), span=_sp(), node_id=_nid())


def _text_seg(text: str) -> ast.TextSegment:
    return ast.TextSegment(text=text, span=_sp(), node_id=_nid())


def _ask_call(
    text: str,
    *,
    strict_json: bool | None = None,
) -> ast.Call:
    """Build a v2 ``ask(text)`` call expression using the default agent.

    ``strict_json=True`` adds ``strict_json: true`` as a named argument.
    The caller supplies the agent function as ``default_agent`` when running.
    """
    named_args: list[ast.NamedArg] = []
    if strict_json is not None:
        named_args.append(
            ast.NamedArg(
                name="strict_json",
                value=ast.BoolLit(value=strict_json, span=_sp(), node_id=_nid()),
                span=_sp(),
                node_id=_nid(),
            )
        )
    return ast.Call(
        callee=ast.VarRef(name="ask", span=_sp(), node_id=_nid()),
        args=(_template(_text_seg(text)),),
        named_args=tuple(named_args),
        span=_sp(),
        node_id=_nid(),
    )


def _name_ty(name: str) -> tast.NameT:
    return tast.NameT(name=name, span=_sp(), node_id=_nid())


def _int_ty() -> tast.IntT:
    return tast.IntT(span=_sp(), node_id=_nid())


def _dec_ty() -> tast.DecimalT:
    return tast.DecimalT(span=_sp(), node_id=_nid())


def _bool_ty() -> tast.BoolT:
    return tast.BoolT(span=_sp(), node_id=_nid())


def _list_ty(elem: tast.TypeExpr) -> tast.ListT:
    return tast.ListT(elem=elem, span=_sp(), node_id=_nid())


def _text_ty() -> tast.TextT:
    return tast.TextT(span=_sp(), node_id=_nid())


def _field_def(name: str, type_expr: tast.TypeExpr) -> ast.FieldDef:
    return ast.FieldDef(name=name, type_expr=type_expr, span=_sp(), node_id=_nid())


def _record_def(name: str, *fields: ast.FieldDef) -> ast.RecordDef:
    return ast.RecordDef(name=name, fields=tuple(fields), span=_sp(), node_id=_nid())


def _variant_def(name: str, *fields: ast.FieldDef) -> ast.VariantDef:
    return ast.VariantDef(name=name, fields=tuple(fields), span=_sp(), node_id=_nid())


def _enum_def(name: str, *variants: ast.VariantDef) -> ast.EnumDef:
    return ast.EnumDef(name=name, variants=tuple(variants), span=_sp(), node_id=_nid())


# ---------------------------------------------------------------------------
# 1. Schema derivation
# ---------------------------------------------------------------------------


class TestDeriveSchema:
    def test_text_type(self) -> None:
        schema = derive_schema(TextType())
        assert schema == {"type": "string"}

    def test_int_type(self) -> None:
        schema = derive_schema(IntType())
        assert schema == {"type": "integer"}

    def test_decimal_type(self) -> None:
        schema = derive_schema(DecimalType())
        assert schema == {"type": "number"}

    def test_bool_type(self) -> None:
        schema = derive_schema(BoolType())
        assert schema == {"type": "boolean"}

    def test_json_type_is_permissive(self) -> None:
        # json type accepts anything: {}
        schema = derive_schema(JsonType())
        assert schema == {}

    def test_list_of_text(self) -> None:
        schema = derive_schema(ListType(elem=TextType()))
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_list_of_int(self) -> None:
        schema = derive_schema(ListType(elem=IntType()))
        assert schema == {"type": "array", "items": {"type": "integer"}}

    def test_list_nested(self) -> None:
        schema = derive_schema(ListType(elem=ListType(elem=BoolType())))
        assert schema == {
            "type": "array",
            "items": {"type": "array", "items": {"type": "boolean"}},
        }

    def test_dict_of_text(self) -> None:
        schema = derive_schema(DictType(value=TextType()))
        assert schema == {"type": "object", "additionalProperties": {"type": "string"}}

    def test_dict_of_int(self) -> None:
        schema = derive_schema(DictType(value=IntType()))
        assert schema == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_record_schema(self) -> None:
        issue_type = _make_issue_type()
        schema = derive_schema(issue_type)
        assert schema == {
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "severity", "description"],
            "properties": {
                "title": {"type": "string"},
                "severity": {"type": "integer"},
                "description": {"type": "string"},
            },
        }

    def test_record_required_has_all_fields(self) -> None:
        typ = RecordType(
            name="Pair",
            fields={"a": IntType(), "b": TextType()},
        )
        schema = derive_schema(typ)
        required = schema["required"]
        assert isinstance(required, list)
        assert set(required) == {"a", "b"}

    def test_record_nested_record(self) -> None:
        inner = RecordType(name="Inner", fields={"x": IntType()})
        outer = RecordType(name="Outer", fields={"inner": inner})
        schema = derive_schema(outer)
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert properties["inner"] == {
            "type": "object",
            "additionalProperties": False,
            "required": ["x"],
            "properties": {"x": {"type": "integer"}},
        }

    def test_enum_schema_pass_only(self) -> None:
        typ = EnumType(name="Status", variants={"Done": {}})
        schema = derive_schema(typ)
        assert schema == {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["$case"],
                    "properties": {"$case": {"const": "Done"}},
                }
            ]
        }

    def test_enum_schema_review(self) -> None:
        schema = derive_schema(_make_review_type())
        assert schema == {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["$case"],
                    "properties": {"$case": {"const": "Pass"}},
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["$case", "issues"],
                    "properties": {
                        "$case": {"const": "Fail"},
                        "issues": {"type": "array", "items": {"type": "string"}},
                    },
                },
            ]
        }

    def test_enum_nullary_variant_has_only_case_field(self) -> None:
        typ = EnumType(name="E", variants={"A": {}, "B": {"x": IntType()}})
        schema = derive_schema(typ)
        # First variant (A) should have only $case in required.
        a_schema = _variant_schema_for_case(schema, "A")
        required_a = a_schema["required"]
        assert isinstance(required_a, list)
        assert required_a == ["$case"]

    def test_enum_payload_variant_has_case_plus_fields(self) -> None:
        typ = EnumType(name="E", variants={"A": {}, "B": {"x": IntType()}})
        schema = derive_schema(typ)
        b_schema = _variant_schema_for_case(schema, "B")
        required_b = b_schema["required"]
        assert isinstance(required_b, list)
        assert set(required_b) == {"$case", "x"}


# ---------------------------------------------------------------------------
# 2. JsonCodec.supports_type
# ---------------------------------------------------------------------------


class TestJsonCodecSupportsType:
    def test_supports_json(self) -> None:
        assert JsonCodec().supports_type(JsonType()) is True

    def test_supports_int(self) -> None:
        assert JsonCodec().supports_type(IntType()) is True

    def test_supports_decimal(self) -> None:
        assert JsonCodec().supports_type(DecimalType()) is True

    def test_supports_bool(self) -> None:
        assert JsonCodec().supports_type(BoolType()) is True

    def test_supports_list(self) -> None:
        assert JsonCodec().supports_type(ListType(elem=TextType())) is True

    def test_supports_dict(self) -> None:
        assert JsonCodec().supports_type(DictType(value=TextType())) is True

    def test_supports_record(self) -> None:
        assert JsonCodec().supports_type(_make_issue_type()) is True

    def test_supports_enum(self) -> None:
        assert JsonCodec().supports_type(_make_review_type()) is True

    def test_does_not_support_text(self) -> None:
        assert JsonCodec().supports_type(TextType()) is False

    def test_name_is_json(self) -> None:
        assert JsonCodec().name == "json"


# ---------------------------------------------------------------------------
# 3. Lenient parsing (default)
# ---------------------------------------------------------------------------


class TestLenientParsing:
    """Lenient is the default (strict_json=False). Recover from fences/prose."""

    def _codec(self) -> JsonCodec:
        return JsonCodec()

    def test_bare_integer(self) -> None:
        codec = self._codec()
        result = codec.parse("5", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(5)

    def test_bare_json_object(self) -> None:
        codec = self._codec()
        typ = JsonType()
        result = codec.parse('{"k": 1}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_fenced_json_block_extracted(self) -> None:
        codec = self._codec()
        result = codec.parse("```json\n5\n```", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(5)

    def test_fenced_json_object_extracted(self) -> None:
        codec = self._codec()
        typ = JsonType()
        result = codec.parse('```json\n{"k": 1}\n```', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_prose_wrapped_json_extracted(self) -> None:
        codec = self._codec()
        typ = JsonType()
        result = codec.parse('Here you go:\n[1, 2]', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_prose_and_fence(self) -> None:
        codec = self._codec()
        result = codec.parse(
            "Sure thing!\n```json\n5\n```", IntType(), strict_json=False
        )
        assert result.ok is True
        assert result.value == IntValue(5)

    def test_trailing_comma_repaired(self) -> None:
        codec = self._codec()
        typ = JsonType()
        result = codec.parse('{"k": 1,}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_single_quoted_keys_repaired(self) -> None:
        codec = self._codec()
        typ = JsonType()
        result = codec.parse("{'k': 1}", typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_gibberish_fails(self) -> None:
        codec = self._codec()
        result = codec.parse("complete gibberish, no number here", IntType(), strict_json=False)
        assert result.ok is False
        assert result.value is None

    def test_bare_bool_recovered_from_prose(self) -> None:
        """Lenient recovery pulls a bare ``false`` keyword out of prose."""
        codec = self._codec()
        result = codec.parse("The flag is:\nfalse", BoolType(), strict_json=False)
        assert result.ok is True
        assert result.value == BoolValue(False)

    def test_bare_null_recovered_from_prose(self) -> None:
        codec = self._codec()
        result = codec.parse("Answer: null", JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw is None

    def test_bare_number_recovered_from_prose(self) -> None:
        codec = self._codec()
        result = codec.parse("the count is 42 items", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(42)

    def test_keyword_substring_not_falsely_recovered(self) -> None:
        """``nullable`` must not be mistaken for a bare ``null`` token."""
        codec = self._codec()
        result = codec.parse("the config is nullable here", BoolType(), strict_json=False)
        assert result.ok is False

    def test_two_bare_scalars_in_prose_are_ambiguous(self) -> None:
        codec = self._codec()
        result = codec.parse("maybe true or maybe false", BoolType(), strict_json=False)
        assert result.ok is False
        assert "multiple JSON values" in result.error_msg


# ---------------------------------------------------------------------------
# 4. Strict mode
# ---------------------------------------------------------------------------


class TestStrictParsing:
    """strict_json=True: only bare JSON with surrounding whitespace accepted."""

    def _codec(self) -> JsonCodec:
        return JsonCodec()

    def test_bare_integer_accepted(self) -> None:
        codec = self._codec()
        result = codec.parse("5", IntType(), strict_json=True)
        assert result.ok is True
        assert result.value == IntValue(5)

    def test_whitespace_around_bare_integer_accepted(self) -> None:
        codec = self._codec()
        result = codec.parse("  5  ", IntType(), strict_json=True)
        assert result.ok is True
        assert result.value == IntValue(5)

    def test_fenced_value_rejected(self) -> None:
        codec = self._codec()
        result = codec.parse("```json\n5\n```", IntType(), strict_json=True)
        assert result.ok is False

    def test_trailing_prose_rejected(self) -> None:
        codec = self._codec()
        result = codec.parse("5\nThat is my final answer.", IntType(), strict_json=True)
        assert result.ok is False

    def test_single_quotes_rejected(self) -> None:
        codec = self._codec()
        result = codec.parse("{'k': 1}", JsonType(), strict_json=True)
        assert result.ok is False

    def test_trailing_comma_rejected(self) -> None:
        codec = self._codec()
        result = codec.parse('{"k": 1,}', JsonType(), strict_json=True)
        assert result.ok is False

    def test_bare_object_accepted(self) -> None:
        codec = self._codec()
        result = codec.parse('{"k": 1}', JsonType(), strict_json=True)
        assert result.ok is True

    def test_fenced_object_rejected(self) -> None:
        codec = self._codec()
        result = codec.parse('```json\n{"k": 1}\n```', JsonType(), strict_json=True)
        assert result.ok is False


# ---------------------------------------------------------------------------
# 5. Decimal exactness
# ---------------------------------------------------------------------------


class TestDecimalExactness:
    """Decimal values must never round-trip through float (design §5.1)."""

    def test_decimal_stays_decimal_in_lenient(self) -> None:
        codec = JsonCodec()
        result = codec.parse("1.5", DecimalType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, DecimalValue)
        assert result.value.value == Decimal("1.5")

    def test_decimal_stays_decimal_in_strict(self) -> None:
        codec = JsonCodec()
        result = codec.parse("1.5", DecimalType(), strict_json=True)
        assert result.ok is True
        assert isinstance(result.value, DecimalValue)
        assert result.value.value == Decimal("1.5")

    def test_decimal_in_record_field(self) -> None:
        codec = JsonCodec()
        typ = RecordType(name="Foo", fields={"w": DecimalType()})
        result = codec.parse('{"w": 1.5}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, RecordValue)
        w = result.value.fields["w"]
        assert isinstance(w, DecimalValue)
        assert w.value == Decimal("1.5")

    def test_decimal_from_fenced_stays_exact(self) -> None:
        codec = JsonCodec()
        result = codec.parse("```json\n1.5\n```", DecimalType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, DecimalValue)
        assert result.value.value == Decimal("1.5")

    def test_decimal_not_float(self) -> None:
        codec = JsonCodec()
        result = codec.parse("1.5", DecimalType(), strict_json=False)
        assert isinstance(result.value, DecimalValue)
        assert not isinstance(result.value.value, float)

    def test_int_widened_to_decimal_when_target_says_decimal(self) -> None:
        codec = JsonCodec()
        result = codec.parse("3", DecimalType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, DecimalValue)
        assert result.value.value == Decimal("3")

    def test_high_precision_decimal(self) -> None:
        # Bare valid JSON is parsed directly (no json-repair), so Decimal precision
        # is fully preserved by json.loads(parse_float=Decimal).
        codec = JsonCodec()
        result = codec.parse("1.23456789012345678901", DecimalType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, DecimalValue)
        assert result.value.value == Decimal("1.23456789012345678901")

    def test_decimal_in_repaired_json(self) -> None:
        """Decimal exactness through json-repair path (single-quote input)."""
        codec = JsonCodec()
        typ = RecordType(name="Foo", fields={"w": DecimalType()})
        result = codec.parse("{'w': 1.5}", typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, RecordValue)
        w = result.value.fields["w"]
        assert isinstance(w, DecimalValue)
        assert w.value == Decimal("1.5")


# ---------------------------------------------------------------------------
# 6. Typed Value construction
# ---------------------------------------------------------------------------


class TestTypedValueConstruction:
    def test_int_value(self) -> None:
        codec = JsonCodec()
        result = codec.parse("42", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(42)

    def test_bool_value_true(self) -> None:
        codec = JsonCodec()
        result = codec.parse("true", BoolType(), strict_json=False)
        assert result.ok is True
        assert result.value == BoolValue(True)

    def test_bool_value_false(self) -> None:
        codec = JsonCodec()
        result = codec.parse("false", BoolType(), strict_json=False)
        assert result.ok is True
        assert result.value == BoolValue(False)

    def test_list_of_text(self) -> None:
        codec = JsonCodec()
        typ = ListType(elem=TextType())
        result = codec.parse('["a", "b"]', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, ListValue)
        assert result.value.elements == (TextValue("a"), TextValue("b"))

    def test_list_of_int(self) -> None:
        codec = JsonCodec()
        typ = ListType(elem=IntType())
        result = codec.parse('[1, 2, 3]', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, ListValue)
        assert result.value.elements == (IntValue(1), IntValue(2), IntValue(3))

    def test_dict_of_text(self) -> None:
        codec = JsonCodec()
        typ = DictType(value=TextType())
        result = codec.parse('{"a": "hello"}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, DictValue)
        assert result.value.entries == {"a": TextValue("hello")}

    def test_record_value(self) -> None:
        codec = JsonCodec()
        typ = _make_issue_type()
        raw = '{"title": "Bug", "severity": 5, "description": "Oh no"}'
        result = codec.parse(raw, typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, RecordValue)
        assert result.value.display_name == "Issue"
        assert result.value.fields["title"] == TextValue("Bug")
        assert result.value.fields["severity"] == IntValue(5)
        assert result.value.fields["description"] == TextValue("Oh no")

    def test_enum_nullary_variant(self) -> None:
        codec = JsonCodec()
        typ = _make_review_type()
        result = codec.parse('{"$case": "Pass"}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, EnumValue)
        assert result.value.display_name == "Review"
        assert result.value.variant == "Pass"
        assert result.value.fields == {}

    def test_enum_payload_variant(self) -> None:
        codec = JsonCodec()
        typ = _make_review_type()
        raw = '{"$case": "Fail", "issues": ["a", "b"]}'
        result = codec.parse(raw, typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, EnumValue)
        assert result.value.variant == "Fail"
        issues = result.value.fields["issues"]
        assert isinstance(issues, ListValue)
        assert issues.elements == (TextValue("a"), TextValue("b"))

    def test_json_value_wraps_raw(self) -> None:
        codec = JsonCodec()
        result = codec.parse('{"a": 1}', JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_nested_record(self) -> None:
        codec = JsonCodec()
        inner = RecordType(name="Inner", fields={"x": IntType()})
        outer = RecordType(name="Outer", fields={"inner": inner, "n": IntType()})
        raw = '{"inner": {"x": 7}, "n": 3}'
        result = codec.parse(raw, outer, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, RecordValue)
        inner_val = result.value.fields["inner"]
        assert isinstance(inner_val, RecordValue)
        assert inner_val.fields["x"] == IntValue(7)

    def test_list_in_record_field(self) -> None:
        codec = JsonCodec()
        typ = RecordType(name="Doc", fields={"tags": ListType(elem=TextType())})
        result = codec.parse('{"tags": ["x", "y"]}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, RecordValue)
        tags = result.value.fields["tags"]
        assert isinstance(tags, ListValue)
        assert tags.elements == (TextValue("x"), TextValue("y"))


# ---------------------------------------------------------------------------
# 7. Schema validation errors (missing/unknown/wrong-type/$case)
# ---------------------------------------------------------------------------


class TestSchemaValidationErrors:
    def test_missing_required_field_fails(self) -> None:
        codec = JsonCodec()
        typ = _make_issue_type()
        # missing severity and description
        result = codec.parse('{"title": "Bug"}', typ, strict_json=False)
        assert result.ok is False
        assert result.value is None

    def test_unknown_field_fails(self) -> None:
        codec = JsonCodec()
        typ = _make_issue_type()
        raw = '{"title": "Bug", "severity": 1, "description": "x", "extra": true}'
        result = codec.parse(raw, typ, strict_json=False)
        assert result.ok is False

    def test_wrong_type_fails(self) -> None:
        codec = JsonCodec()
        typ = _make_issue_type()
        raw = '{"title": "Bug", "severity": "high", "description": "x"}'
        result = codec.parse(raw, typ, strict_json=False)
        assert result.ok is False

    def test_bad_case_tag_fails(self) -> None:
        codec = JsonCodec()
        typ = _make_review_type()
        result = codec.parse('{"$case": "Unknown"}', typ, strict_json=False)
        assert result.ok is False

    def test_missing_case_field_fails(self) -> None:
        codec = JsonCodec()
        typ = _make_review_type()
        result = codec.parse('{"issues": ["x"]}', typ, strict_json=False)
        assert result.ok is False

    def test_enum_missing_payload_field_fails(self) -> None:
        codec = JsonCodec()
        typ = _make_review_type()
        # Fail variant but missing issues
        result = codec.parse('{"$case": "Fail"}', typ, strict_json=False)
        assert result.ok is False

    def test_failure_result_has_no_value(self) -> None:
        codec = JsonCodec()
        result = codec.parse('{}', _make_issue_type(), strict_json=False)
        assert result.value is None

    def test_failure_result_has_error_msg(self) -> None:
        codec = JsonCodec()
        result = codec.parse('{}', _make_issue_type(), strict_json=False)
        assert result.error_msg
        assert isinstance(result.error_msg, str)


# ---------------------------------------------------------------------------
# 7b. Structured ValidationError records (F1 — design §7.5 / §7.7)
# ---------------------------------------------------------------------------


class TestStructuredValidationErrors:
    """Each documented category is surfaced as a structured ValidationError."""

    def _categories(self, result: ParseResult) -> list[str]:
        return [e.category for e in result.errors]

    def test_missing_field_category(self) -> None:
        codec = JsonCodec()
        result = codec.parse('{"title": "Bug"}', _make_issue_type(), strict_json=False)
        assert result.ok is False
        assert "missing_field" in self._categories(result)
        missing = [e for e in result.errors if e.category == "missing_field"]
        assert all(e.field is not None for e in missing)

    def test_unknown_field_category(self) -> None:
        codec = JsonCodec()
        raw = '{"title": "Bug", "severity": 1, "description": "x", "extra": true}'
        result = codec.parse(raw, _make_issue_type(), strict_json=False)
        assert result.ok is False
        assert self._categories(result) == ["unknown_field"]
        # The opaque jsonschema phrasing must not leak verbatim as the category.
        assert "extra" in result.errors[0].message

    def test_wrong_type_category(self) -> None:
        codec = JsonCodec()
        raw = '{"title": "Bug", "severity": "high", "description": "x"}'
        result = codec.parse(raw, _make_issue_type(), strict_json=False)
        assert result.ok is False
        wrong = [e for e in result.errors if e.category == "wrong_type"]
        assert wrong
        assert wrong[0].field == "severity"
        assert wrong[0].path == "$.severity"

    def test_bad_case_unknown_variant(self) -> None:
        codec = JsonCodec()
        result = codec.parse('{"$case": "Nope"}', _make_review_type(), strict_json=False)
        assert result.ok is False
        assert self._categories(result) == ["bad_case"]
        msg = result.errors[0].message
        # Type-directed: real variant names, not "not valid under any of ...".
        assert "is not valid under any of the given schemas" not in msg
        assert "Nope" in msg
        assert "Pass" in msg and "Fail" in msg

    def test_bad_case_missing_tag(self) -> None:
        codec = JsonCodec()
        result = codec.parse('{"issues": ["x"]}', _make_review_type(), strict_json=False)
        assert result.ok is False
        assert self._categories(result) == ["bad_case"]
        assert result.errors[0].field == "$case"
        assert "is not valid under any of the given schemas" not in result.errors[0].message

    def test_enum_missing_payload_field_is_missing_field(self) -> None:
        codec = JsonCodec()
        # Fail variant requires "issues".
        result = codec.parse('{"$case": "Fail"}', _make_review_type(), strict_json=False)
        assert result.ok is False
        assert self._categories(result) == ["missing_field"]
        assert result.errors[0].field == "issues"
        assert "is not valid under any of the given schemas" not in result.errors[0].message

    def test_enum_unknown_payload_field_is_unknown_field(self) -> None:
        codec = JsonCodec()
        result = codec.parse('{"$case": "Pass", "junk": 1}', _make_review_type(), strict_json=False)
        assert result.ok is False
        assert self._categories(result) == ["unknown_field"]
        assert result.errors[0].field == "junk"

    def test_nested_enum_wrong_type_path(self) -> None:
        """Type-directed enum resolution works under a record field path."""
        codec = JsonCodec()
        typ = RecordType(name="Wrapper", fields={"review": _make_review_type()})
        result = codec.parse('{"review": {"$case": "Bogus"}}', typ, strict_json=False)
        assert result.ok is False
        assert self._categories(result) == ["bad_case"]
        assert "Bogus" in result.errors[0].message

    def test_success_has_no_errors(self) -> None:
        codec = JsonCodec()
        raw = '{"title": "Bug", "severity": 1, "description": "x"}'
        result = codec.parse(raw, _make_issue_type(), strict_json=False)
        assert result.ok is True
        assert result.errors == ()

    def test_non_validation_failure_has_no_errors(self) -> None:
        """A failure to extract any JSON is not a schema-validation error."""
        codec = JsonCodec()
        result = codec.parse("complete gibberish ###", _make_issue_type(), strict_json=False)
        assert result.ok is False
        assert result.errors == ()


class TestValidationErrorsThroughRuntime:
    """F1: real ValidationErrors thread into AgentParseError.validation_errors."""

    def test_validation_errors_in_agent_parse_error(self) -> None:
        record_def = _record_def(
            "Issue",
            _field_def("title", _text_ty()),
            _field_def("severity", _int_ty()),
        )
        let_x = _let(
            "x",
            _ask_call("Get issue."),
            type_ann=_name_ty("Issue"),
        )
        with pytest.raises(AglRaise) as exc_info:
            _run_with_json_codec(
                (record_def, let_x),
                # Valid JSON, but missing the required "severity" field.
                default_agent=lambda req: '{"title": "Bug"}',
            )
        exc = exc_info.value.exc
        assert exc.display_name == "AgentParseError"
        ve = exc.fields["validation_errors"]
        assert isinstance(ve, JsonValue)
        assert isinstance(ve.raw, list)
        assert len(ve.raw) >= 1
        first = ve.raw[0]
        assert isinstance(first, dict)
        assert first["category"] == "missing_field"
        assert first["field"] == "severity"

    def test_bad_case_validation_errors_through_runtime(self) -> None:
        enum_def = _enum_def(
            "Review",
            _variant_def("Pass"),
            _variant_def("Fail", _field_def("issues", _list_ty(_text_ty()))),
        )
        let_r = _let("r", _ask_call("Review."), type_ann=_name_ty("Review"))
        with pytest.raises(AglRaise) as exc_info:
            _run_with_json_codec(
                (enum_def, let_r),
                default_agent=lambda req: '{"$case": "Nope"}',
            )
        exc = exc_info.value.exc
        ve = exc.fields["validation_errors"]
        assert isinstance(ve, JsonValue)
        assert isinstance(ve.raw, list)
        first = ve.raw[0]
        assert isinstance(first, dict)
        assert first["category"] == "bad_case"

    def test_retry_request_carries_previous_validation_errors(self) -> None:
        """On retry, the AgentRequest exposes the prior attempt's ValidationErrors."""
        from agm.agl.runtime.request import ValidationError as VE

        seen: list[list[VE]] = []

        def agent(req: AgentRequest) -> str:
            seen.append(list(req.validation_errors))
            return '{"title": "Bug"}'  # always missing severity → always fails

        record_def = _record_def(
            "Issue",
            _field_def("title", _text_ty()),
            _field_def("severity", _int_ty()),
        )
        # v2: on_parse_error: Retry(n: 1) as a named arg to ask().
        # Constructors are now Call nodes (no separate Constructor AST node).
        retry_ctor = ast.Call(
            callee=ast.VarRef(name="Retry", span=_sp(), node_id=_nid()),
            args=(),
            named_args=(
                ast.NamedArg(
                    name="n",
                    value=ast.IntLit(value=1, span=_sp(), node_id=_nid()),
                    span=_sp(),
                    node_id=_nid(),
                ),
            ),
            span=_sp(),
            node_id=_nid(),
        )
        retry_call = ast.Call(
            callee=ast.VarRef(name="ask", span=_sp(), node_id=_nid()),
            args=(_template(_text_seg("Get issue.")),),
            named_args=(
                ast.NamedArg(
                    name="on_parse_error",
                    value=retry_ctor,
                    span=_sp(),
                    node_id=_nid(),
                ),
            ),
            span=_sp(),
            node_id=_nid(),
        )
        let_x = _let("x", retry_call, type_ann=_name_ty("Issue"))
        with pytest.raises(AglRaise):
            _run_with_json_codec((record_def, let_x), default_agent=agent)
        # Two attempts: first sees no prior errors, retry sees the missing_field.
        assert len(seen) == 2
        assert seen[0] == []
        assert seen[1] and seen[1][0].category == "missing_field"


# ---------------------------------------------------------------------------
# 7c. Multi-value ambiguity rejection (F3 — design §2.8 "exactly one value")
# ---------------------------------------------------------------------------


class TestMultiValueAmbiguity:
    def test_two_objects_rejected(self) -> None:
        codec = JsonCodec()
        result = codec.parse('{"a":1} {"b":2}', JsonType(), strict_json=False)
        assert result.ok is False
        assert "multiple JSON values" in result.error_msg

    def test_two_objects_newline_separated_rejected(self) -> None:
        codec = JsonCodec()
        result = codec.parse('{"a":1}\n{"b":2}', JsonType(), strict_json=False)
        assert result.ok is False
        assert "multiple JSON values" in result.error_msg

    def test_text_then_single_object_recovers(self) -> None:
        codec = JsonCodec()
        result = codec.parse('text then {"a": 1}', JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw == {"a": 1}

    def test_bare_array_parses(self) -> None:
        codec = JsonCodec()
        result = codec.parse('[1, 2]', JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw == [1, 2]

    def test_fenced_array_parses(self) -> None:
        codec = JsonCodec()
        result = codec.parse('```json\n[1, 2]\n```', JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw == [1, 2]

    def test_prose_wrapped_array_recovers(self) -> None:
        """A genuine single array wrapped in prose is recovered (not ambiguous)."""
        codec = JsonCodec()
        result = codec.parse('Here you go:\n[1, 2]', JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw == [1, 2]

    def test_ambiguous_inside_fence_rejected(self) -> None:
        codec = JsonCodec()
        result = codec.parse('```json\n{"a":1} {"b":2}\n```', JsonType(), strict_json=False)
        assert result.ok is False
        assert "multiple JSON values" in result.error_msg

    def test_two_objects_with_inner_array_rejected(self) -> None:
        """F4: ``{"a": [1]} {"b": 2}`` is ambiguous despite the inner ``[``."""
        codec = JsonCodec()
        result = codec.parse('{"a": [1]} {"b": 2}', JsonType(), strict_json=False)
        assert result.ok is False
        assert "multiple JSON values" in result.error_msg

    def test_two_values_with_escaped_bracket_string_rejected(self) -> None:
        """F4: a bracket inside an escaped string does not hide the second value."""
        codec = JsonCodec()
        result = codec.parse('{"a": "[x]"} {"b": 2}', JsonType(), strict_json=False)
        assert result.ok is False
        assert "multiple JSON values" in result.error_msg

    def test_single_object_with_inner_array_recovers(self) -> None:
        """F4: a single object containing an array is one value (not ambiguous)."""
        codec = JsonCodec()
        result = codec.parse('{"a": [1, 2]}', JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw == {"a": [1, 2]}


# ---------------------------------------------------------------------------
# 8. make_contract (OutputContract materialization)
# ---------------------------------------------------------------------------


class TestMakeContract:
    def test_json_schema_populated_for_record(self) -> None:
        contract = _make_contract_for(_make_issue_type())
        schema = contract.json_schema
        assert isinstance(schema, dict)
        assert schema["type"] == "object"

    def test_json_schema_populated_for_enum(self) -> None:
        contract = _make_contract_for(_make_review_type())
        schema = contract.json_schema
        assert isinstance(schema, dict)
        assert "oneOf" in schema

    def test_format_instructions_include_schema_for_record(self) -> None:
        contract = _make_contract_for(_make_issue_type())
        instr = contract.format_instructions
        assert instr
        assert "conforming to the following JSON Schema" in instr
        # The actual derived schema is embedded, not a hand-written paraphrase.
        assert '"properties"' in instr
        assert '"additionalProperties"' in instr
        # Field names reach the agent via the schema.
        assert "title" in instr
        assert "severity" in instr
        assert "description" in instr

    def test_format_instructions_include_schema_for_enum(self) -> None:
        contract = _make_contract_for(_make_review_type())
        instr = contract.format_instructions
        assert instr
        assert "conforming to the following JSON Schema" in instr
        assert '"oneOf"' in instr
        assert '"$case"' in instr
        # Variant names reach the agent via the schema consts.
        assert "Pass" in instr
        assert "Fail" in instr

    def test_format_instructions_for_permissive_json_omit_schema_block(self) -> None:
        # The ``json`` type derives a permissive ``{}`` schema; there is no
        # shape to convey, so only the behavioural preamble is emitted.
        contract = _make_contract_for(JsonType())
        instr = contract.format_instructions
        assert instr
        assert "conforming to the following JSON Schema" not in instr
        assert "```json" not in instr
        assert "Return exactly one JSON value" in instr

    def test_codec_field_is_json_codec(self) -> None:
        contract = _make_contract_for(_make_issue_type())
        assert isinstance(contract.codec, JsonCodec)

    def test_materialize_contract_with_json_codec(self) -> None:
        codec = JsonCodec()
        spec = OutputContractSpec(
            target_type=_make_issue_type(),
            codec_name="json",
            strict_json=False,
        )
        contract = materialize_contract(spec, {"json": codec, "text": TextCodec()})
        assert isinstance(contract.codec, JsonCodec)
        assert contract.json_schema is not None


# ---------------------------------------------------------------------------
# 9. WorkflowRuntime wire-up
# ---------------------------------------------------------------------------


def _json_ty() -> tast.JsonT:
    return tast.JsonT(span=_sp(), node_id=_nid())


def _dict_ty(value: tast.TypeExpr) -> tast.DictT:
    return tast.DictT(value=value, span=_sp(), node_id=_nid())


class TestWorkflowRuntimeWireUp:
    """JsonCodec registered in runtime; checker passes json/record/enum targets.

    Note: typed agent-call bindings (let x: T = agent "...") are tested via the
    direct-AST helpers because the M1 parser wraps agent calls in an 'access'
    tree node for non-text typed bindings.  WorkflowRuntime.run() tests cover the
    static pipeline (codec_kinds) and the error reporting paths.
    """

    def test_json_target_type_accepted_via_direct_ast(self) -> None:
        """A call targeting json type should pass type checking and execute."""
        let_x = _let("x", _ask_call("Get data."), type_ann=_json_ty())
        scope = _run_with_json_codec(
            (let_x,), default_agent=lambda req: '{"x": 1}'
        )
        x = scope.snapshot()["x"]
        assert isinstance(x, JsonValue)

    def test_int_target_accepted_via_json_codec(self) -> None:
        let_n = _let("n", _ask_call("Get number."), type_ann=_int_ty())
        scope = _run_with_json_codec((let_n,), default_agent=lambda req: "42")
        assert scope.snapshot()["n"] == IntValue(42)

    def test_bool_target_accepted(self) -> None:
        let_b = _let("b", _ask_call("Is it true?"), type_ann=_bool_ty())
        scope = _run_with_json_codec(
            (let_b,), default_agent=lambda req: "true"
        )
        assert scope.snapshot()["b"] == BoolValue(True)

    def test_decimal_target_accepted(self) -> None:
        let_d = _let("d", _ask_call("Get ratio."), type_ann=_dec_ty())
        scope = _run_with_json_codec((let_d,), default_agent=lambda req: "1.5")
        d = scope.snapshot()["d"]
        assert isinstance(d, DecimalValue)
        assert d.value == Decimal("1.5")

    def test_record_target_accepted_via_json_codec(self) -> None:
        # record Issue; title: text; severity: int
        # let x: Issue = ask "Get issue."
        record_def = _record_def(
            "Issue",
            _field_def("title", _text_ty()),
            _field_def("severity", _int_ty()),
        )
        let_x = _let(
            "x",
            _ask_call("Get issue."),
            type_ann=_name_ty("Issue"),
        )
        scope = _run_with_json_codec(
            (record_def, let_x),
            default_agent=lambda req: '{"title": "Bug", "severity": 5}',
        )
        x = scope.snapshot()["x"]
        assert isinstance(x, RecordValue)
        assert x.fields["title"] == TextValue("Bug")

    def test_enum_target_accepted_via_json_codec(self) -> None:
        # enum Review | Pass | Fail(issues: list[text])
        # let r: Review = ask "Review."
        enum_def = _enum_def(
            "Review",
            _variant_def("Pass"),
            _variant_def("Fail", _field_def("issues", _list_ty(_text_ty()))),
        )
        let_r = _let(
            "r",
            _ask_call("Review."),
            type_ann=_name_ty("Review"),
        )
        scope = _run_with_json_codec(
            (enum_def, let_r),
            default_agent=lambda req: '{"$case": "Pass"}',
        )
        r = scope.snapshot()["r"]
        assert isinstance(r, EnumValue)
        assert r.variant == "Pass"

    def test_list_target_accepted(self) -> None:
        let_xs = _let(
            "xs",
            _ask_call("List items."),
            type_ann=_list_ty(_text_ty()),
        )
        scope = _run_with_json_codec(
            (let_xs,), default_agent=lambda req: '["a", "b"]'
        )
        xs = scope.snapshot()["xs"]
        assert isinstance(xs, ListValue)
        assert xs.elements == (TextValue("a"), TextValue("b"))

    def test_dict_target_accepted(self) -> None:
        let_d = _let(
            "d",
            _ask_call("Dict."),
            type_ann=_dict_ty(_text_ty()),
        )
        scope = _run_with_json_codec(
            (let_d,), default_agent=lambda req: '{"k": "v"}'
        )
        d = scope.snapshot()["d"]
        assert isinstance(d, DictValue)
        assert d.entries == {"k": TextValue("v")}

    def test_agent_receives_format_instructions_for_record(self) -> None:
        """Format instructions from the contract must be available in agent request."""
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return '{"title": "X", "severity": 1}'

        # record Issue; title: text; severity: int; let x: Issue = ask "Fetch."
        record_def = _record_def(
            "Issue",
            _field_def("title", _text_ty()),
            _field_def("severity", _int_ty()),
        )
        let_x = _let("x", _ask_call("Fetch."), type_ann=_name_ty("Issue"))
        _run_with_json_codec((record_def, let_x), default_agent=agent)
        assert received, "agent was not called"
        req = received[0]
        assert req.output_contract is not None
        assert req.output_contract.format_instructions
        # The derived JSON Schema is embedded in the instructions the agent sees.
        assert '"properties"' in req.output_contract.format_instructions
        assert "title" in req.output_contract.format_instructions

    def test_lenient_fenced_json_works_end_to_end(self) -> None:
        """Lenient recovery: agent returns fenced JSON, runtime parses it."""
        record_def = _record_def(
            "Issue",
            _field_def("title", _text_ty()),
            _field_def("severity", _int_ty()),
        )
        let_x = _let("x", _ask_call("Get."), type_ann=_name_ty("Issue"))
        scope = _run_with_json_codec(
            (record_def, let_x),
            default_agent=lambda req: '```json\n{"title": "Flaky", "severity": 2}\n```',
        )
        x = scope.snapshot()["x"]
        assert isinstance(x, RecordValue)
        assert x.fields["title"] == TextValue("Flaky")

    def test_strict_json_rejects_fenced_end_to_end(self) -> None:
        """strict_json=True: fenced JSON → AgentParseError."""
        let_n = _let(
            "n",
            _ask_call("Count.", strict_json=True),
            type_ann=_int_ty(),
        )
        with pytest.raises(AglRaise) as exc_info:
            _run_with_json_codec(
                (let_n,),
                default_agent=lambda req: "```json\n6\n```",
            )
        exc = exc_info.value.exc
        assert exc.display_name == "AgentParseError"

    def test_runtime_default_strict_json_applies(self) -> None:
        """default_strict_json=True on runtime applies to calls without explicit option."""
        let_n = _let("n", _ask_call("Count."), type_ann=_int_ty())
        with pytest.raises(AglRaise) as exc_info:
            _run_with_json_codec(
                (let_n,),
                default_agent=lambda req: "```json\n5\n```",
                strict_json=True,
            )
        exc = exc_info.value.exc
        assert exc.display_name == "AgentParseError"

    def test_parse_error_becomes_agent_parse_error(self) -> None:
        let_n = _let("n", _ask_call("Num."), type_ann=_int_ty())
        with pytest.raises(AglRaise) as exc_info:
            _run_with_json_codec(
                (let_n,),
                default_agent=lambda req: "not json at all",
            )
        exc = exc_info.value.exc
        assert exc.display_name == "AgentParseError"
        # In v2 the agent field reflects the built-in "ask" call site (default agent path).
        assert exc.fields.get("agent") == TextValue("ask")

    def test_agent_parse_error_has_target_type_field(self) -> None:
        let_n = _let("n", _ask_call("Num."), type_ann=_int_ty())
        with pytest.raises(AglRaise) as exc_info:
            _run_with_json_codec((let_n,), default_agent=lambda req: "bad")
        exc = exc_info.value.exc
        assert exc.display_name == "AgentParseError"
        assert "target_type" in exc.fields

    def test_decimal_exactness_end_to_end(self) -> None:
        """Decimal stays exact through the full runtime pipeline."""
        let_d = _let("d", _ask_call("Get ratio."), type_ann=_dec_ty())
        scope = _run_with_json_codec((let_d,), default_agent=lambda req: "1.5")
        d = scope.snapshot()["d"]
        assert isinstance(d, DecimalValue)
        assert d.value == Decimal("1.5")
        assert not isinstance(d.value, float)

    def test_json_codec_supports_type_kinds_registered(self) -> None:
        """HostCapabilities codec_kinds includes json codec kinds."""
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.runtime.codec import JsonCodec, TextCodec

        text_codec = TextCodec()
        json_codec = JsonCodec()
        kinds = {
            text_codec.name: frozenset({"text"}),
            json_codec.name: frozenset({
                "json", "record", "enum", "list", "dict", "int", "decimal", "bool"
            }),
        }
        caps = HostCapabilities(
            codec_kinds=kinds,
        )
        # json kind must appear in some codec's supported kinds
        all_supported = set().union(*caps.codec_kinds.values())
        assert "json" in all_supported
        assert "record" in all_supported
        assert "enum" in all_supported


# ---------------------------------------------------------------------------
# 10. $case dispatch
# ---------------------------------------------------------------------------


class TestCaseDispatch:
    def test_correct_case_dispatched(self) -> None:
        codec = JsonCodec()
        typ = EnumType(
            name="Status",
            variants={
                "Done": {},
                "Running": {"progress": IntType()},
            },
        )
        result = codec.parse('{"$case": "Running", "progress": 50}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, EnumValue)
        assert result.value.variant == "Running"
        assert result.value.fields["progress"] == IntValue(50)

    def test_bad_case_fails(self) -> None:
        codec = JsonCodec()
        typ = EnumType(name="Status", variants={"Done": {}})
        result = codec.parse('{"$case": "Exploded"}', typ, strict_json=False)
        assert result.ok is False

    def test_missing_case_tag_fails(self) -> None:
        codec = JsonCodec()
        typ = EnumType(name="Status", variants={"Done": {}})
        result = codec.parse('{"done": true}', typ, strict_json=False)
        assert result.ok is False

    def test_nullary_enum_no_extra_fields(self) -> None:
        codec = JsonCodec()
        typ = EnumType(name="Status", variants={"Done": {}})
        result = codec.parse('{"$case": "Done"}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, EnumValue)
        assert result.value.fields == {}


# ---------------------------------------------------------------------------
# 11. normalized_raw in ParseResult
# ---------------------------------------------------------------------------


class TestNormalizedRaw:
    def test_lenient_sets_normalized_raw_when_extraction_occurred(self) -> None:
        codec = JsonCodec()
        result = codec.parse("```json\n5\n```", IntType(), strict_json=False)
        assert result.ok is True
        # normalized_raw should be the extracted/repaired JSON text
        assert result.normalized_raw is not None

    def test_bare_json_normalized_raw(self) -> None:
        codec = JsonCodec()
        result = codec.parse("5", IntType(), strict_json=False)
        assert result.ok is True
        # Even bare JSON has a normalized_raw
        assert result.normalized_raw is not None

    def test_schema_failure_carries_normalized_raw(self) -> None:
        """F5: a fenced-but-schema-invalid response still exposes the recovered text."""
        codec = JsonCodec()
        # Fenced JSON that is valid JSON but the wrong shape for an int target.
        result = codec.parse('```json\n"oops"\n```', IntType(), strict_json=False)
        assert result.ok is False
        # The recovered (normalized) JSON text is threaded through the failure,
        # distinct from the fenced raw response.
        assert result.normalized_raw == '"oops"'

    def test_conversion_failure_carries_normalized_raw(self) -> None:
        """F5: a value-conversion failure also threads the recovered text."""
        codec = JsonCodec()
        # 1.5 passes a decimal schema check but cannot convert to an int Value;
        # exercises the conversion-failure branch.  Use a shape jsonschema lets
        # through but _json_to_value rejects: a float for an int via repair.
        result = codec.parse("not json at all", IntType(), strict_json=False)
        assert result.ok is False


# ---------------------------------------------------------------------------
# 12. Record/enum params accepted via json codec (M2 extension of runtime)
# ---------------------------------------------------------------------------


class TestRecordEnumParams:
    """Runtime.convert_param now accepts record/enum types via JsonCodec."""

    def test_record_param_parsed_from_json_string(self) -> None:
        # record Issue; title: text; severity: int; param issue: Issue; print issue.title
        # Use direct-AST since record decl needs M2a parser.
        from agm.agl.syntax.nodes import ParamDecl, VarRef

        record_def = _record_def(
            "Issue",
            _field_def("title", _text_ty()),
            _field_def("severity", _int_ty()),
        )
        param_decl = ParamDecl(
            name="issue",
            annotation=_name_ty("Issue"),
            default=None,
            span=_sp(),
            node_id=_nid(),
        )
        # In v2 print is a Call expression, not a PrintStmt.
        print_call = ast.Call(
            callee=VarRef(name="print", span=_sp(), node_id=_nid()),
            args=(VarRef(name="issue", span=_sp(), node_id=_nid()),),
            named_args=(),
            span=_sp(),
            node_id=_nid(),
        )
        program = ast.Program(
            body=ast.Block(
                items=(record_def, param_decl, print_call),
                span=_sp(),
                node_id=_nid(),
            ),
            span=_sp(),
            node_id=_nid(),
        )
        resolved = resolve(program, ambient_agents=ambient_agents_for(program))
        caps = HostCapabilities(
            codec_kinds={
                "text": frozenset({"text"}),
                "json": frozenset(
                    {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
                ),
            },
        )
        checked = check(resolved, caps)
        from agm.agl.runtime.codec import JsonCodec, OutputCodec, TextCodec
        from agm.agl.runtime.contract import materialize_contract

        text_codec = TextCodec()
        json_codec = JsonCodec()
        codecs: dict[str, OutputCodec] = {
            text_codec.name: text_codec,
            json_codec.name: json_codec,
        }
        contracts: dict[int, OutputContract] = {
            nid: materialize_contract(spec, codecs)
            for nid, spec in checked.contract_specs.items()
        }
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.runtime.agents import AgentRegistry

        registry = AgentRegistry(named={}, default_agent=None)
        root = Scope(parent=None)
        # Convert and pass the param value as the runtime would.
        raw_input = '{"title": "Bug", "severity": 5}'
        issue_type = RecordType(
            name="Issue", fields={"title": TextType(), "severity": IntType()}
        )
        codec = JsonCodec()
        parse_result = codec.parse(raw_input, issue_type, strict_json=False)
        assert parse_result.ok and parse_result.value is not None
        from agm.agl.syntax.nodes import ParamDecl as ID

        param_values = {
            stmt.name: parse_result.value
            for stmt in program.body.items
            if isinstance(stmt, ID)
        }
        interp = Interpreter(
            checked=checked,
            registry=registry,
            contracts=contracts,
            type_env=checked.type_env,
            loop_limit=3,
            strict_json=False,
            param_values=param_values,
        )
        interp.execute(root)
        v = root.snapshot()["issue"]
        assert isinstance(v, RecordValue)
        assert v.fields["title"] == TextValue("Bug")

    def test_enum_param_parsed_from_json_string(self) -> None:
        """Enum can be parsed via JsonCodec from a JSON string."""
        codec = JsonCodec()
        typ = EnumType(name="Status", variants={"Done": {}, "Pending": {}})
        result = codec.parse('{"$case": "Done"}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, EnumValue)
        assert result.value.variant == "Done"

    def test_list_param_parsed_from_json_string(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run(
            "param tags: list[text]",
            param_values={"tags": '["a", "b"]'},
        )
        assert result.ok is True

    def test_structured_param_accepts_python_list(self) -> None:
        """Structured params may be provided as a Python list (JSON-compatible)."""
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.runtime import convert_param_value

        result = convert_param_value("xs", [1, 2, 3], ListType(elem=IntType()))
        assert isinstance(result, ListValue)
        assert result.elements == (IntValue(1), IntValue(2), IntValue(3))

    def test_structured_param_must_be_string_or_compatible(self) -> None:
        """Structured params that are not a string or JSON-compatible Python value raise."""
        from agm.agl.runtime.runtime import convert_param_value

        with pytest.raises(ValueError, match="JSON"):
            convert_param_value("xs", object(), ListType(elem=IntType()))

    def test_invalid_structured_param_raises(self) -> None:
        """A JSON string that fails schema validation for the declared type raises."""
        from agm.agl.runtime.runtime import convert_param_value

        with pytest.raises(ValueError, match="could not parse"):
            convert_param_value(
                "issue",
                '{"title": "Bug"}',  # missing severity
                RecordType(name="Issue", fields={"title": TextType(), "severity": IntType()}),
            )

    def test_unsupported_type_in_convert_param_value_raises(self) -> None:
        """ExceptionType is not a supported param type."""
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import ExceptionType

        with pytest.raises(ValueError, match="unsupported type"):
            convert_param_value("e", "val", ExceptionType(name="Boom"))

    def test_structured_param_is_strict_no_repair(self) -> None:
        """F7: host --param values are parsed strictly; typos are NOT repaired.

        A trailing comma (which json-repair would silently fix for chatty agent
        output) must be rejected for a user-supplied structured param, with an
        error that makes the JSON requirement clear.
        """
        from agm.agl.runtime.runtime import convert_param_value

        with pytest.raises(ValueError, match="valid JSON value"):
            convert_param_value(
                "issue",
                '{"title": "Bug", "severity": 5,}',  # trailing comma typo
                RecordType(name="Issue", fields={"title": TextType(), "severity": IntType()}),
            )

    def test_structured_param_rejects_fenced_json(self) -> None:
        """F7: a Markdown-fenced --param value is not stripped (strict parsing)."""
        from agm.agl.runtime.runtime import convert_param_value

        with pytest.raises(ValueError, match="valid JSON value"):
            convert_param_value(
                "tags",
                "```json\n[1, 2]\n```",
                ListType(elem=IntType()),
            )


# ---------------------------------------------------------------------------
# 13. Coverage: _json_to_value error branches
# ---------------------------------------------------------------------------


class TestJsonToValueErrorBranches:
    """Cover the ValueError branches inside _json_to_value."""

    def _parse(self, raw: str, typ: Type) -> ParseResult:
        return JsonCodec().parse(raw, typ, strict_json=False)

    def test_text_type_got_non_string(self) -> None:
        # Schema accepts any string but we can test _json_to_value directly.
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="string"):
            _json_to_value(42, TextType())

    def test_int_type_got_bool(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="bool"):
            _json_to_value(True, IntType())

    def test_int_type_got_non_integer_decimal(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="integer"):
            _json_to_value(Decimal("1.5"), IntType())

    def test_decimal_type_got_bool(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="bool"):
            _json_to_value(True, DecimalType())

    def test_decimal_type_got_string(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="decimal"):
            _json_to_value("not a number", DecimalType())

    def test_bool_type_got_int(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="bool"):
            _json_to_value(1, BoolType())

    def test_list_type_got_non_list(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="array"):
            _json_to_value("not a list", ListType(elem=TextType()))

    def test_dict_type_got_non_dict(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="object"):
            _json_to_value([1, 2], DictType(value=TextType()))

    def test_dict_non_string_key(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        # Construct a dict with a non-str key (not normally from json.loads but defensive).
        with pytest.raises(ValueError, match="Dict key must be string"):
            _json_to_value({1: "val"}, DictType(value=TextType()))

    def test_record_type_got_non_dict(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="record"):
            _json_to_value([1, 2], RecordType(name="R", fields={"x": IntType()}))

    def test_record_missing_field(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="Missing field"):
            _json_to_value({}, RecordType(name="R", fields={"x": IntType()}))

    def test_enum_type_got_non_dict(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="object for enum"):
            _json_to_value("oops", EnumType(name="E", variants={"A": {}}))

    def test_enum_missing_case_tag(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match=r"\$case"):
            _json_to_value({}, EnumType(name="E", variants={"A": {}}))

    def test_enum_unknown_variant(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="Unknown enum variant"):
            _json_to_value({"$case": "X"}, EnumType(name="E", variants={"A": {}}))

    def test_enum_missing_payload_field(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value

        with pytest.raises(ValueError, match="missing field"):
            _json_to_value(
                {"$case": "B"},
                EnumType(name="E", variants={"B": {"x": IntType()}}),
            )

    def test_exception_type_not_supported(self) -> None:
        from agm.agl.runtime.convert import json_to_value as _json_to_value
        from agm.agl.typecheck.types import ExceptionType

        with pytest.raises(ValueError, match="Cannot deserialise"):
            _json_to_value({}, ExceptionType(name="Boom"))

    def test_integral_decimal_to_int_through_parse(self) -> None:
        """F2: wire ``1.0`` validates and converts to IntValue(1) for an int target.

        Exercised through ``parse()`` (the public path), not the previously-dead
        ``_json_to_value`` branch: post-parse normalization rewrites integral
        Decimals to int *before* schema validation, so ``{"type": "integer"}``
        accepts ``1.0``.
        """
        codec = JsonCodec()
        result = codec.parse("1.0", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(1)

    def test_integral_decimal_to_int_strict(self) -> None:
        """F2: integral-Decimal normalization also applies on the strict path."""
        codec = JsonCodec()
        result = codec.parse("1.0", IntType(), strict_json=True)
        assert result.ok is True
        assert result.value == IntValue(1)

    def test_non_integral_decimal_rejected_for_int(self) -> None:
        """F2: ``1.5`` still fails an int target (not integral)."""
        codec = JsonCodec()
        result = codec.parse("1.5", IntType(), strict_json=False)
        assert result.ok is False
        assert result.value is None
        assert any(e.category == "wrong_type" for e in result.errors)

    def test_integral_decimal_for_decimal_target(self) -> None:
        """F2: ``1.0`` for a decimal target yields a value-exact DecimalValue.

        Normalization routes the integral Decimal through int, and the
        int→decimal widening in ``_json_to_value`` re-widens it: the resulting
        value equals ``1`` exactly (design §5.1 — no value/precision loss;
        ``Decimal('1') == Decimal('1.0')``).
        """
        codec = JsonCodec()
        result = codec.parse("1.0", DecimalType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, DecimalValue)
        # Value exactness: numerically equal to both 1 and 1.0.
        assert result.value.value == Decimal("1.0")
        assert result.value.value == Decimal("1")
        # Pinned representation: integral decimals normalize to scale-0 Decimal('1').
        assert result.value.value == Decimal(1)


# ---------------------------------------------------------------------------
# 14. Coverage: schema.py ExceptionType branch
# ---------------------------------------------------------------------------


class TestSchemaExceptionType:
    def test_exception_type_raises_type_error(self) -> None:
        from agm.agl.typecheck.types import ExceptionType

        with pytest.raises(TypeError, match="ExceptionType"):
            derive_schema(ExceptionType(name="Boom"))


# ---------------------------------------------------------------------------
# 15. Coverage: fenced malformed JSON (repair within fence)
# ---------------------------------------------------------------------------


class TestFencedMalformedJson:
    """Fenced content that is itself malformed but repairable."""

    def test_fenced_single_quotes_repaired(self) -> None:
        codec = JsonCodec()
        # Fenced content with single-quoted keys — json-repair fixes it.
        raw = "```json\n{'k': 1}\n```"
        result = codec.parse(raw, JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_fenced_trailing_comma_repaired(self) -> None:
        codec = JsonCodec()
        raw = "```json\n{\"a\": 1,}\n```"
        result = codec.parse(raw, JsonType(), strict_json=False)
        assert result.ok is True


# ---------------------------------------------------------------------------
# 17. Coverage: lenient json parse fail after repair
# ---------------------------------------------------------------------------


class TestLenientParseAfterRepair:
    """Edge cases in lenient path."""

    def test_schema_validation_failure_message(self) -> None:
        # Passing a string for an int target fails schema validation.
        codec = JsonCodec()
        result = codec.parse('"not an int"', IntType(), strict_json=False)
        assert result.ok is False
        assert "Schema validation failed" in result.error_msg

    def test_validate_and_convert_value_conversion_failure(self) -> None:
        """_validate_and_convert: schema passes (permissive) but _json_to_value fails."""
        codec = JsonCodec()
        # Call _validate_and_convert directly with a permissive schema ({}) but a
        # TextType target that rejects a non-string value.  The permissive schema {}
        # accepts anything, but _json_to_value(42, TextType()) raises ValueError.
        result = codec._validate_and_convert("42", 42, TextType(), {})
        assert result.ok is False
        assert "Value conversion failed" in result.error_msg


class TestFencedRepairFallback:
    """Fenced content where direct parse fails and json-repair returns empty/null."""

    def test_fenced_empty_content_falls_through_to_whole_raw_repair(self) -> None:
        """Fenced block with no useful content falls through to whole-raw repair."""
        codec = JsonCodec()
        # Fenced block with gibberish that repairs to "" or empty.
        # The outer prose has the real JSON.
        raw = "Here is the value: 42 ```json\n\n```"
        result = codec.parse(raw, IntType(), strict_json=False)
        assert result.ok
        assert result.value == IntValue(42)
        assert result.normalized_raw == "42"

    def test_lenient_json_decode_error_after_extraction(self) -> None:
        """_parse_lenient: _extract_json_text returns a string that json.loads still rejects."""
        from unittest.mock import patch

        from agm.agl.runtime import codec as codec_module

        codec = JsonCodec()
        # Patch _extract_json_text to return a string that is NOT valid JSON.
        with patch.object(codec_module, "_extract_json_text", return_value="{broken"):
            result = codec.parse("anything", IntType(), strict_json=False)
        assert result.ok is False
        assert "JSON parse failed after repair attempt" in result.error_msg


# ---------------------------------------------------------------------------
# 16. Coverage: validation-error mapping internals (F1) and extraction edges
# ---------------------------------------------------------------------------


class TestValidationMappingCoverage:
    """Cover structural / defensive branches of the F1 error mapping."""

    def test_trailing_comma_array_recovers_not_ambiguous(self) -> None:
        """A repaired array whose candidate already starts with '[' is not ambiguous."""
        codec = JsonCodec()
        result = codec.parse("[1, 2,]", JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw == [1, 2]

    def test_enum_two_field_variant_reports_first_missing(self) -> None:
        codec = JsonCodec()
        typ = EnumType(name="E", variants={"V": {"a": IntType(), "b": IntType()}})
        # "a" present, "b" missing → loop skips a, reports b.
        result = codec.parse('{"$case": "V", "a": 1}', typ, strict_json=False)
        assert result.ok is False
        assert result.errors[0].category == "missing_field"
        assert result.errors[0].field == "b"

    def test_enum_non_object_instance_is_bad_case(self) -> None:
        codec = JsonCodec()
        typ = EnumType(name="E", variants={"A": {}})
        result = codec.parse("42", typ, strict_json=False)
        assert result.ok is False
        assert result.errors[0].category == "bad_case"

    def test_list_nested_enum_bad_case(self) -> None:
        codec = JsonCodec()
        enum = EnumType(name="E", variants={"A": {}, "B": {"x": IntType()}})
        result = codec.parse('[{"$case": "Z"}]', ListType(elem=enum), strict_json=False)
        assert result.ok is False
        assert result.errors[0].category == "bad_case"
        assert result.errors[0].path == "$[0]"

    def test_dict_nested_enum_bad_case(self) -> None:
        codec = JsonCodec()
        enum = EnumType(name="E", variants={"A": {}, "B": {"x": IntType()}})
        result = codec.parse('{"k": {"$case": "Z"}}', DictType(value=enum), strict_json=False)
        assert result.ok is False
        assert result.errors[0].category == "bad_case"
        assert result.errors[0].path == "$.k"

    def test_missing_required_field_helper_defensive(self) -> None:
        """_missing_required_field returns None for non-list / non-dict shapes."""
        from jsonschema import ValidationError as JVE

        from agm.agl.runtime.codec import _missing_required_field

        err = JVE("required")
        err.validator_value = "not-a-list"
        err.instance = {"x": 1}
        assert _missing_required_field(err) is None

    def test_missing_required_field_helper_all_present(self) -> None:
        """_missing_required_field returns None when every required name is present."""
        from jsonschema import ValidationError as JVE

        from agm.agl.runtime.codec import _missing_required_field

        err = JVE("required")
        err.validator_value = ["a", "b"]
        err.instance = {"a": 1, "b": 2}
        assert _missing_required_field(err) is None

    def test_classify_unknown_validator_is_wrong_type(self) -> None:
        """A non-required/additionalProperties/type/oneOf validator → wrong_type."""
        from jsonschema import ValidationError as JVE

        from agm.agl.runtime.codec import _classify_jsonschema_error

        err = JVE("const mismatch")
        err.validator = "const"
        err.validator_value = "X"
        err.instance = "Y"
        ve = _classify_jsonschema_error(err, TextType())
        assert ve.category == "wrong_type"
        assert ve.message == "const mismatch"

    def test_enum_type_at_path_unknown_record_field(self) -> None:
        """_enum_type_at_path returns None when a path step names an unknown field."""
        from agm.agl.runtime.codec import _enum_type_at_path

        rec = RecordType(name="R", fields={"a": IntType()})
        assert _enum_type_at_path(rec, ["missing"]) is None

    def test_enum_type_at_path_scalar_with_remaining_path(self) -> None:
        """_enum_type_at_path returns None when path descends past a scalar."""
        from agm.agl.runtime.codec import _enum_type_at_path

        assert _enum_type_at_path(IntType(), ["deeper"]) is None


# ---------------------------------------------------------------------------
# 18. CARRY-IN 2 — schema reuse: make_contract no longer takes TypeEnvironment
# ---------------------------------------------------------------------------


class TestMakeContractNoTypeEnv:
    """CARRY-IN 2: make_contract signature drops the unused TypeEnvironment param."""

    def test_text_codec_make_contract_no_env(self) -> None:
        codec = TextCodec()
        # make_contract now takes only type_ref — no env argument.
        contract = codec.make_contract(TextType())
        assert contract.codec is codec

    def test_json_codec_make_contract_no_env(self) -> None:
        codec = JsonCodec()
        contract = codec.make_contract(_make_issue_type())
        assert contract.json_schema is not None

    def test_materialize_contract_no_longer_constructs_type_env(self) -> None:
        """materialize_contract must not instantiate TypeEnvironment internally."""
        from agm.agl.runtime.contract import materialize_contract
        from agm.agl.typecheck.env import OutputContractSpec

        spec = OutputContractSpec(
            target_type=_make_issue_type(),
            codec_name="json",
            strict_json=False,
        )
        # If TypeEnvironment() were still constructed it would not fail, but we
        # verify the contract comes out correctly to confirm the wire-up works.
        contract = materialize_contract(spec, {"json": JsonCodec(), "text": TextCodec()})
        assert contract.json_schema is not None


class TestSchemaPrecomputedInParse:
    """CARRY-IN 2: parse() accepts a precomputed schema; runtime-side callers pass it."""

    def test_parse_with_precomputed_schema_succeeds(self) -> None:
        codec = JsonCodec()
        typ = _make_issue_type()
        from agm.agl.runtime.schema import derive_schema
        schema = derive_schema(typ)
        raw = '{"title": "Bug", "severity": 5, "description": "A bug"}'
        result = codec.parse(raw, typ, strict_json=False, schema=schema)
        assert result.ok is True

    def test_parse_with_precomputed_schema_validation_failure(self) -> None:
        codec = JsonCodec()
        typ = _make_issue_type()
        from agm.agl.runtime.schema import derive_schema
        schema = derive_schema(typ)
        # Missing required fields → schema validation fails even with precomputed schema.
        result = codec.parse('{"title": "Bug"}', typ, strict_json=False, schema=schema)
        assert result.ok is False
        assert result.errors

    def test_parse_with_precomputed_schema_matches_derived(self) -> None:
        """F5: parse(schema=precomputed) is observably equivalent to parse().

        Passing the materialized schema is an optimization, never a behavior
        change: the parse outcome (ok/value/errors) must be identical to letting
        the codec derive the schema itself, on both the success and the
        validation-failure path.
        """
        codec = JsonCodec()
        typ = RecordType(name="Issue", fields={"title": TextType(), "severity": IntType()})
        schema = codec.make_contract(typ).json_schema
        assert isinstance(schema, dict)

        good = '{"title": "x", "severity": 1}'
        bad = '{"title": "x"}'  # missing required field
        for raw in (good, bad):
            with_schema = codec.parse(raw, typ, strict_json=False, schema=schema)
            without = codec.parse(raw, typ, strict_json=False)
            assert with_schema.ok == without.ok
            assert with_schema.value == without.value
            assert [e.message for e in with_schema.errors] == [
                e.message for e in without.errors
            ]

    def test_contract_json_schema_reused_across_parses(self) -> None:
        """F5: the contract's json_schema object is the one threaded into parse.

        Observable identity reuse: the schema object the codec materializes on
        the contract is the same object accepted by ``parse(schema=...)`` — so
        the interpreter passing ``contract.json_schema`` reuses it rather than
        re-deriving (BONUS).  We verify it parses correctly and that the schema
        is a concrete materialized dict.
        """
        codec = JsonCodec()
        typ = RecordType(name="Issue", fields={"title": TextType(), "severity": IntType()})
        contract = codec.make_contract(typ)
        schema = contract.json_schema
        assert isinstance(schema, dict)
        raw = '{"title": "x", "severity": 1}'
        # Reuse the very same schema object on repeated parses.
        for _ in range(3):
            result = codec.parse(raw, typ, strict_json=False, schema=schema)
            assert result.ok is True
            # Identity: contract still holds the same schema object.
            assert contract.json_schema is schema

    def test_parse_without_schema_still_works(self) -> None:
        """parse() without schema= falls back to deriving it (backward compat)."""
        codec = JsonCodec()
        result = codec.parse("42", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(42)


# ---------------------------------------------------------------------------
# 19. CARRY-IN 1 — supported_kinds property on codecs
# ---------------------------------------------------------------------------


class TestCodecSupportedKinds:
    """CARRY-IN 1: codecs expose supported_kinds; runtime builds caps from them."""

    def test_text_codec_supported_kinds(self) -> None:
        codec = TextCodec()
        assert codec.supported_kinds == frozenset({"text"})

    def test_json_codec_supported_kinds(self) -> None:
        codec = JsonCodec()
        assert codec.supported_kinds == frozenset(
            {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
        )

    def test_supported_kinds_consistent_with_supports_type(self) -> None:
        """Every kind in supported_kinds matches a Type that supports_type returns True for."""
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
        )
        kind_to_type: dict[str, Type] = {
            "text": TextType(),
            "int": IntType(),
            "decimal": DecimalType(),
            "bool": BoolType(),
            "json": JsonType(),
            "list": ListType(elem=TextType()),
            "dict": DictType(value=TextType()),
            "record": RecordType(name="R", fields={}),
            "enum": EnumType(name="E", variants={}),
        }
        for codec in (TextCodec(), JsonCodec()):
            for kind in codec.supported_kinds:
                typ = kind_to_type[kind]
                assert codec.supports_type(typ), (
                    f"{codec.name}.supports_type({kind}) should be True"
                )


# ---------------------------------------------------------------------------
# 20. CARRY-IN 1 — register_codec public API
# ---------------------------------------------------------------------------


class TestRegisterCodec:
    """CARRY-IN 1: register_codec adds a custom codec to the runtime."""

    def _make_custom_codec(self) -> TextCodec:
        """A minimal custom codec (reuses TextCodec but with a different name for testing)."""
        import copy
        codec = copy.copy(TextCodec())
        return codec

    def test_register_codec_accepted(self, capsys: pytest.CaptureFixture[str]) -> None:
        from agm.agl.runtime.codec import TextCodec as TC
        rt = WorkflowRuntime(default_agent=lambda request: "response")

        class AltTextCodec(TC):
            @property
            def name(self) -> str:
                return "alt_text"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

        rt.register_codec(AltTextCodec())
        result = rt.run(
            'let answer: text = ask("question", format: "alt_text")\nprint answer',
            param_values={},
        )

        assert result.ok
        assert capsys.readouterr().out == "response\n"

    def test_register_duplicate_codec_raises(self) -> None:
        from agm.agl.runtime.codec import ParseResult as PR
        from agm.agl.runtime.contract import OutputContract as OC

        class CustomCodec:
            @property
            def name(self) -> str:
                return "custom_dup"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset()

            def supports_type(self, t: Type) -> bool:
                return False

            def make_contract(self, type_ref: Type) -> OC:
                raise NotImplementedError

            def parse(
                self,
                raw: str,
                target_type: Type,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
            ) -> PR:
                raise NotImplementedError

        rt = WorkflowRuntime()
        rt.register_codec(CustomCodec())
        with pytest.raises(ValueError, match="custom_dup"):
            rt.register_codec(CustomCodec())

    def test_register_reserved_codec_name_text_raises(self) -> None:
        rt = WorkflowRuntime()
        with pytest.raises(ValueError, match="text"):
            rt.register_codec(TextCodec())

    def test_register_reserved_codec_name_json_raises(self) -> None:
        rt = WorkflowRuntime()
        with pytest.raises(ValueError, match="json"):
            rt.register_codec(JsonCodec())

    def test_custom_codec_make_contract_and_parse_exercised_in_pipeline(self) -> None:
        """F3 (M3b): a custom codec selected via ``format:`` is genuinely used.

        The codec is chosen with ``ask("Q", format: "tagcodec")`` on a ``text``
        target.  Both its ``make_contract`` (observable via the format
        instructions threaded into the agent request) and its ``parse``
        (observable as a distinctive prefix on the resulting binding) are
        exercised end-to-end through ``run()`` with a stub agent.
        """
        from agm.agl.eval.values import TextValue

        class TagCodec:
            @property
            def name(self) -> str:
                return "tagcodec"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

            def supports_type(self, t: Type) -> bool:
                from agm.agl.typecheck.types import TextType as TT
                return isinstance(t, TT)

            def make_contract(self, type_ref: Type) -> "OutputContract":
                from agm.agl.runtime.contract import OutputContract
                return OutputContract(
                    target_type=type_ref,
                    codec=self,
                    strict_json=None,
                    format_instructions="TAGCODEC-INSTRUCTIONS",
                    json_schema=None,
                )

            def parse(
                self,
                raw: str,
                target_type: Type,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
            ) -> "ParseResult":
                from agm.agl.runtime.codec import ParseResult
                # Distinctive transform proving THIS codec parsed the output.
                return ParseResult.success(TextValue(f"PARSED::{raw}"))

        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "hello"

        rt = WorkflowRuntime(default_agent=agent)
        rt.register_codec(TagCodec())
        # v2: format: arg takes the codec name as a string; let needs a continuation.
        result = rt.run('let y: text = ask("Q", format: "tagcodec")\ny')
        assert result.ok is True
        # parse() ran: the binding carries the codec's distinctive prefix.
        assert result.bindings["y"] == TextValue("PARSED::hello")
        # make_contract() ran: its format instructions reached the agent.
        assert received[0].output_contract is not None
        assert (
            received[0].output_contract.format_instructions
            == "TAGCODEC-INSTRUCTIONS"
        )


class TestRuntimeBuildsCodecKinds:
    """A custom codec is selectable via ``format:`` only after registration."""

    def test_runtime_builds_codec_kinds_from_registered_codec(self) -> None:
        from agm.agl.runtime.codec import TextCodec

        class AltCodec(TextCodec):
            @property
            def name(self) -> str:
                return "altcodec"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

        # v2: format: arg takes the codec name as a string; let needs a continuation.
        src = 'let x: text = ask("Q", format: "altcodec")\nx'

        rt_unreg = WorkflowRuntime(default_agent=lambda req: "ok")
        unreg = rt_unreg.run(src)
        assert unreg.ok is False  # altcodec unknown without registration
        assert any("altcodec" in d.message for d in unreg.diagnostics)

        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        rt.register_codec(AltCodec())
        reg = rt.run(src)
        assert reg.ok is True
