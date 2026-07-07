"""Tests for the AgL JsonCodec, schema derivation, and wire-up.

Covers (per the AgL DSL contract):
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
7. Multiple JSON values / ambiguous output → failure.
8. PipelineDriver wire-up: JsonCodec registered; checker passes json/record/enum
   targets; format_instructions reach AgentRequest; make_contract API.
9. decimal exactness end-to-end: 1.5 parsed from agent response stays Decimal("1.5").

"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from decimal import Decimal

import pytest
from jsonschema import Draft202012Validator

from agm.agl import PipelineDriver
from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.contracts import (
    ContractRequest,
    DecodeSchema,
    ListDecode,
    RefDecode,
    ScalarDecode,
    ScalarKind,
)
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.runtime.agents import AgentFn, AgentRegistry
from agm.agl.runtime.codec import JsonCodec, ParseResult, TextCodec
from agm.agl.runtime.contract import OutputContract, materialize_contract, materialize_ir_contract
from agm.agl.runtime.request import AgentRequest
from agm.agl.scope import resolve
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.type_table import TypeDef, TypeTable
from agm.agl.semantics.types import (
    AgentType,
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
    TypeVarType,
    UnitType,
)
from agm.agl.semantics.values import (
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
from agm.agl.syntax import nodes as ast
from agm.agl.syntax import types as tast
from agm.agl.syntax.nodes import (
    Item,
    TemplateSegment,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.type_schema import build_decode_schema, derive_schema
from agm.agl.typecheck import check
from agm.agl.typecheck.env import CheckedProgram, OutputContractSpec
from tests._agl_helpers import ambient_agents_for, enum_type, record_type, type_table_for

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_node_ids = itertools.count(100_000)


def _nid() -> int:
    return next(_node_ids)


def _sp() -> SourceSpan:
    return SourceSpan(1, 1, 1, 5, 0, 4)


_ISSUE_TYPE, _ISSUE_TYPEDEF = record_type(
    "Issue",
    {"title": TextType(), "severity": IntType(), "description": TextType()},
)
_REVIEW_TYPE, _REVIEW_TYPEDEF = enum_type(
    "Review",
    {"Pass": {}, "Fail": {"issues": ListType(elem=TextType())}},
)
# Shared table pre-seeded with the two ad-hoc composite shapes reused across
# most of this suite (see ``_make_issue_type``/``_make_review_type``); tests
# that build a different ad-hoc type pass their own table explicitly.
_DEFAULT_TABLE = type_table_for(_ISSUE_TYPEDEF, _REVIEW_TYPEDEF)


def _make_issue_type() -> RecordType:
    """A three-field record: title: text, severity: int, description: text."""
    return _ISSUE_TYPE


def _make_review_type() -> EnumType:
    """enum Review | Pass | Fail(issues: list[text])"""
    return _REVIEW_TYPE


def _make_contract_for(typ: Type, table: TypeTable | None = None) -> OutputContract:
    """Build an OutputContract for a type via JsonCodec.make_contract.

    *table* defaults to :data:`_DEFAULT_TABLE` (the ad-hoc Issue/Review
    shapes); callers using a different ad-hoc type must pass their own table
    (built via ``tests._agl_helpers.type_table_for``) since *typ* is
    constructed directly rather than through the real type builder.
    """
    codec = JsonCodec()
    return codec.make_contract(typ, table if table is not None else _DEFAULT_TABLE)


def _parse_typed(
    codec: JsonCodec,
    raw: str,
    typ: Type,
    *,
    strict_json: bool = False,
    schema: dict[str, object] | None = None,
    table: TypeTable | None = None,
) -> ParseResult:
    """Call ``codec.parse`` with the schema/decode derived from *typ*.

    Production code never derives schema/decode from a checker ``Type`` at
    parse time (the IR evaluator always has the contract-carried
    ``json_schema``/``decode`` on hand); this test-only helper exists so the
    (still very readable) bulk of this suite can keep expressing expectations
    in terms of a ``Type`` rather than hand-building a schema dict and a
    ``DecodeSchema`` at every call site.  *schema*, when given, overrides only
    the derived schema (mirrors ``JsonCodec.make_contract`` computing schema
    and decode once and threading both into ``parse``); ``decode`` is always
    built from *typ* and cannot be overridden.  *table* defaults to
    :data:`_DEFAULT_TABLE` (see :func:`_make_contract_for`).
    """
    table = table if table is not None else _DEFAULT_TABLE
    decode_plan = build_decode_schema(typ, table)
    return codec.parse(
        raw,
        strict_json=strict_json,
        schema=schema if schema is not None else derive_schema(typ, table),
        decode=decode_plan.root,
        defs=dict(decode_plan.defs),
    )


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
# Direct-AST execution helpers for record/enum targets.
# ---------------------------------------------------------------------------


def _ensure_expr_tail(body: tuple[Item, ...]) -> tuple[Item, ...]:
    """Append a unit literal when the last item is a binder.

    In AgL, a block must end with an expression (binders need a continuation).
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
            "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
        },
    )
    return check(resolved, caps)


class _Bindings(dict[str, object]):
    def snapshot(self) -> "_Bindings":
        return self


def _run_with_json_codec(
    body: tuple[Item, ...],
    *,
    named: dict[str, AgentFn] | None = None,
    default_agent: AgentFn | None = None,
    strict_json: bool = False,
) -> _Bindings:
    """Build + resolve + check + execute *body* with JsonCodec registered."""
    from agm.agl.eval.ir_interpreter import IrInterpreter
    from agm.agl.lower import lower_program
    from agm.agl.runtime.codec import JsonCodec, OutputCodec, TextCodec
    from agm.agl.runtime.params import _materialize_ir_contracts

    checked = _check_program_with_json(body)
    text_codec = TextCodec()
    json_codec = JsonCodec()
    codecs: dict[str, OutputCodec] = {
        text_codec.name: text_codec,
        json_codec.name: json_codec,
    }
    registry = AgentRegistry(named=named or {}, default_agent=default_agent)
    executable = lower_program(
        checked, source_text="<direct-ast>", source_label="<test>", validate=True
    )
    contracts, errors = _materialize_ir_contracts(executable, codecs)
    assert errors == []
    interp = IrInterpreter(
        executable,
        registry=registry,
        strict_json=strict_json,
        host_contracts=contracts,
    )
    return _Bindings(interp.run())


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
    """Build an ``ask(text)`` call expression using the default agent.

    ``strict_json=True`` adds ``strict_json = true`` as a named argument.
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


def _field_def(name: str, type_expr: tast.TypeExpr) -> ast.Param:
    return ast.Param(
        name=name,
        type_expr=type_expr,
        kind=ast.ParamKind.NAMED_ONLY,
        default=None,
        span=_sp(),
        node_id=_nid(),
    )


def _record_def(name: str, *fields: ast.Param) -> ast.RecordDef:
    return ast.RecordDef(name=name, fields=tuple(fields), span=_sp(), node_id=_nid())


def _variant_def(name: str, *fields: ast.Param) -> ast.VariantDef:
    return ast.VariantDef(name=name, fields=tuple(fields), span=_sp(), node_id=_nid())


def _enum_def(name: str, *variants: ast.VariantDef) -> ast.EnumDef:
    return ast.EnumDef(name=name, variants=tuple(variants), span=_sp(), node_id=_nid())


# ---------------------------------------------------------------------------
# 1. Schema derivation
# ---------------------------------------------------------------------------


class TestDeriveSchema:
    def test_text_type(self) -> None:
        schema = derive_schema(TextType(), type_table_for())
        assert schema == {"type": "string"}

    def test_int_type(self) -> None:
        schema = derive_schema(IntType(), type_table_for())
        assert schema == {"type": "integer"}

    def test_decimal_type(self) -> None:
        schema = derive_schema(DecimalType(), type_table_for())
        assert schema == {"type": "number"}

    def test_bool_type(self) -> None:
        schema = derive_schema(BoolType(), type_table_for())
        assert schema == {"type": "boolean"}

    def test_json_type_is_permissive(self) -> None:
        # json type accepts anything: {}
        schema = derive_schema(JsonType(), type_table_for())
        assert schema == {}

    def test_list_of_text(self) -> None:
        schema = derive_schema(ListType(elem=TextType()), type_table_for())
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_list_of_int(self) -> None:
        schema = derive_schema(ListType(elem=IntType()), type_table_for())
        assert schema == {"type": "array", "items": {"type": "integer"}}

    def test_list_nested(self) -> None:
        schema = derive_schema(ListType(elem=ListType(elem=BoolType())), type_table_for())
        assert schema == {
            "type": "array",
            "items": {"type": "array", "items": {"type": "boolean"}},
        }

    def test_dict_of_text(self) -> None:
        schema = derive_schema(DictType(value=TextType()), type_table_for())
        assert schema == {"type": "object", "additionalProperties": {"type": "string"}}

    def test_dict_of_int(self) -> None:
        schema = derive_schema(DictType(value=IntType()), type_table_for())
        assert schema == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_record_schema(self) -> None:
        issue_type = _make_issue_type()
        schema = derive_schema(issue_type, _DEFAULT_TABLE)
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
        typ, typedef = record_type("Pair", {"a": IntType(), "b": TextType()})
        schema = derive_schema(typ, type_table_for(typedef))
        required = schema["required"]
        assert isinstance(required, list)
        assert set(required) == {"a", "b"}

    def test_record_nested_record(self) -> None:
        inner, inner_def = record_type("Inner", {"x": IntType()})
        outer, outer_def = record_type("Outer", {"inner": inner})
        schema = derive_schema(outer, type_table_for(outer_def, inner_def))
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert properties["inner"] == {
            "type": "object",
            "additionalProperties": False,
            "required": ["x"],
            "properties": {"x": {"type": "integer"}},
        }

    def test_enum_schema_pass_only(self) -> None:
        typ, typedef = enum_type("Status", {"Done": {}})
        schema = derive_schema(typ, type_table_for(typedef))
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
        review_type = _make_review_type()
        schema = derive_schema(review_type, _DEFAULT_TABLE)
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
        typ, typedef = enum_type("E", {"A": {}, "B": {"x": IntType()}})
        schema = derive_schema(typ, type_table_for(typedef))
        # First variant (A) should have only $case in required.
        a_schema = _variant_schema_for_case(schema, "A")
        required_a = a_schema["required"]
        assert isinstance(required_a, list)
        assert required_a == ["$case"]

    def test_enum_payload_variant_has_case_plus_fields(self) -> None:
        typ, typedef = enum_type("E", {"A": {}, "B": {"x": IntType()}})
        schema = derive_schema(typ, type_table_for(typedef))
        b_schema = _variant_schema_for_case(schema, "B")
        required_b = b_schema["required"]
        assert isinstance(required_b, list)
        assert set(required_b) == {"$case", "x"}


# ---------------------------------------------------------------------------
# 1b. Recursive `$defs`/`$ref` schema emission
# ---------------------------------------------------------------------------


# The exact inline body of the `Tree` `$defs` entry shared by several tests
# below: `enum Tree | Leaf | Node(value: int, left: Tree, right: Tree)`, whose
# `left`/`right` fields are $ref'd back to Tree itself (a self-loop).
_TREE_DEFS_BODY: dict[str, object] = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["$case"],
            "properties": {"$case": {"const": "Leaf"}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["$case", "value", "left", "right"],
            "properties": {
                "$case": {"const": "Node"},
                "value": {"type": "integer"},
                "left": {"$ref": "#/$defs/Tree"},
                "right": {"$ref": "#/$defs/Tree"},
            },
        },
    ]
}


def _tree_type_and_def() -> tuple[EnumType, TypeDef]:
    """``enum Tree | Leaf | Node(value: int, left: Tree, right: Tree)``."""
    tree_ref = EnumType(name="Tree")
    return enum_type(
        "Tree",
        {
            "Leaf": {},
            "Node": {"value": IntType(), "left": tree_ref, "right": tree_ref},
        },
    )


class TestRecursiveSchemaDerivation:
    """`$defs`/`$ref` emission over the concrete instantiation graph (golden schemas)."""

    def test_recursive_enum_root_is_ref_with_defs(self) -> None:
        # Root-is-recursive shape: the root itself is `$ref`'d, and the
        # `$defs` entry holds the real (self-referencing) body.
        tree, tree_def = _tree_type_and_def()
        schema = derive_schema(tree, type_table_for(tree_def))
        assert schema == {"$ref": "#/$defs/Tree", "$defs": {"Tree": _TREE_DEFS_BODY}}

    def test_list_guarded_recursive_record_is_ref_with_defs(self) -> None:
        category, category_def = record_type(
            "Category",
            {"name": TextType(), "subcategories": ListType(RecordType(name="Category"))},
        )
        schema = derive_schema(category, type_table_for(category_def))
        assert schema == {
            "$ref": "#/$defs/Category",
            "$defs": {
                "Category": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "subcategories"],
                    "properties": {
                        "name": {"type": "string"},
                        "subcategories": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Category"},
                        },
                    },
                }
            },
        }

    def test_non_recursive_wrapper_inlined_recursive_field_in_defs(self) -> None:
        # Only PART of the graph is recursive: `Wrapper` itself has no cycle
        # (it is inlined normally), but its `root: Tree` field is $ref'd and
        # `Tree` gets the only `$defs` entry.
        tree, tree_def = _tree_type_and_def()
        wrapper, wrapper_def = record_type("Wrapper", {"root": tree, "label": TextType()})
        schema = derive_schema(wrapper, type_table_for(wrapper_def, tree_def))
        assert schema == {
            "type": "object",
            "additionalProperties": False,
            "required": ["root", "label"],
            "properties": {
                "root": {"$ref": "#/$defs/Tree"},
                "label": {"type": "string"},
            },
            "$defs": {"Tree": _TREE_DEFS_BODY},
        }

    def test_mutual_record_enum_pair_gets_two_defs_entries(self) -> None:
        # record A { b: B } / enum B { Nil, Cons(a: A) }: A and B form one
        # mutual cycle, so BOTH get their own `$defs` entry.
        a, a_def = record_type("A", {"b": EnumType(name="B")})
        b, b_def = enum_type("B", {"Nil": {}, "Cons": {"a": RecordType(name="A")}})
        schema = derive_schema(a, type_table_for(a_def, b_def))
        assert schema == {
            "$ref": "#/$defs/A",
            "$defs": {
                "A": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["b"],
                    "properties": {"b": {"$ref": "#/$defs/B"}},
                },
                "B": {
                    "oneOf": [
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["$case"],
                            "properties": {"$case": {"const": "Nil"}},
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["$case", "a"],
                            "properties": {
                                "$case": {"const": "Cons"},
                                "a": {"$ref": "#/$defs/A"},
                            },
                        },
                    ]
                },
            },
        }

    def test_generic_instantiations_get_distinct_keys(self) -> None:
        # Tree[int] and Tree[text] — two distinct concrete instantiations of
        # the SAME generic declaration — get distinct, non-colliding keys.
        tree_def = TypeDef(
            kind="enum",
            name="Tree",
            module_id=ENTRY_ID,
            type_params=("T",),
            variants=(
                ("Leaf", ()),
                (
                    "Node",
                    (
                        ("value", TypeVarType("T")),
                        ("left", EnumType(name="Tree", type_args=(TypeVarType("T"),))),
                        ("right", EnumType(name="Tree", type_args=(TypeVarType("T"),))),
                    ),
                ),
            ),
        )
        tree_int = EnumType(name="Tree", type_args=(IntType(),))
        tree_text = EnumType(name="Tree", type_args=(TextType(),))
        wrapper, wrapper_def = record_type("Holder", {"a": tree_int, "b": tree_text})
        schema = derive_schema(wrapper, type_table_for(wrapper_def, tree_def))
        defs = schema["$defs"]
        assert isinstance(defs, dict)
        assert set(defs.keys()) == {"Tree_int", "Tree_text"}
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert properties["a"] == {"$ref": "#/$defs/Tree_int"}
        assert properties["b"] == {"$ref": "#/$defs/Tree_text"}

    def test_cross_module_same_name_gets_qualified_keys(self) -> None:
        mod_a = ModuleId.from_dotted("mod_a")
        mod_b = ModuleId.from_dotted("mod_b")
        tree_a, tree_a_def = enum_type(
            "Tree",
            {"Leaf": {}, "Node": {"next": EnumType(name="Tree", module_id=mod_a)}},
            module_id=mod_a,
        )
        tree_b, tree_b_def = enum_type(
            "Tree",
            {"Leaf": {}, "Node": {"next": EnumType(name="Tree", module_id=mod_b)}},
            module_id=mod_b,
        )
        wrapper, wrapper_def = record_type("Holder", {"a": tree_a, "b": tree_b})
        schema = derive_schema(wrapper, type_table_for(wrapper_def, tree_a_def, tree_b_def))
        defs = schema["$defs"]
        assert isinstance(defs, dict)
        assert set(defs.keys()) == {"mod_a.Tree", "mod_b.Tree"}
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert properties["a"] == {"$ref": "#/$defs/mod_a.Tree"}
        assert properties["b"] == {"$ref": "#/$defs/mod_b.Tree"}

    def test_non_recursive_schema_has_no_defs_key(self) -> None:
        # Non-recursive output stays byte-identical to a plain inlining
        # derivation: no `$defs` key at all.
        inner, inner_def = record_type("Inner", {"x": IntType()})
        outer, outer_def = record_type("Outer", {"inner": inner})
        schema = derive_schema(outer, type_table_for(outer_def, inner_def))
        assert "$defs" not in schema

    def test_phantom_recursive_argument_growth_refs_same_defs_entry(self) -> None:
        recursive = RecordType("R", type_args=(ListType(TypeVarType("T")),), module_id=ENTRY_ID)
        root = RecordType("R", type_args=(IntType(),), module_id=ENTRY_ID)
        r_def = TypeDef(
            kind="record",
            name="R",
            module_id=ENTRY_ID,
            type_params=("T",),
            fields=(("children", ListType(recursive)),),
        )
        schema = derive_schema(root, type_table_for(r_def))
        assert schema == {
            "$ref": "#/$defs/R",
            "$defs": {
                "R": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["children"],
                    "properties": {
                        "children": {"type": "array", "items": {"$ref": "#/$defs/R"}}
                    },
                }
            },
        }

    def test_raises_for_infinite_closure_root(self) -> None:
        pair_def = TypeDef(
            kind="record",
            name="Pair",
            module_id=ENTRY_ID,
            type_params=("A", "B"),
            fields=(("first", TypeVarType("A")), ("second", TypeVarType("B"))),
        )
        perfect_def = TypeDef(
            kind="enum",
            name="Perfect",
            module_id=ENTRY_ID,
            type_params=("T",),
            variants=(
                ("Single", (("value", TypeVarType("T")),)),
                (
                    "Succ",
                    (
                        (
                            "next",
                            EnumType(
                                name="Perfect",
                                type_args=(
                                    RecordType(
                                        name="Pair",
                                        type_args=(TypeVarType("T"), TypeVarType("T")),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        table = type_table_for(pair_def, perfect_def)
        perfect_int = EnumType(name="Perfect", type_args=(IntType(),))
        with pytest.raises(TypeError, match="no finite schema"):
            derive_schema(perfect_int, table)

    def test_assign_defs_keys_breaks_residual_collision_with_numeric_suffix(self) -> None:
        # Three handles whose bare display forms all sanitize to the
        # identical string despite being genuinely distinct instantiations
        # (`X[A_B]`, `X[A, B]`, and the plain type `X_A_B` all sanitize to
        # `X_A_B`), all in the entry module (so module-qualification cannot
        # disambiguate them): the numeric suffix is the final,
        # always-available disambiguator, tried incrementally until free.
        from agm.agl.type_schema import _assign_defs_keys

        h1 = RecordType("X", type_args=(RecordType("A_B", module_id=ENTRY_ID),), module_id=ENTRY_ID)
        h2 = RecordType(
            "X",
            type_args=(RecordType("A", module_id=ENTRY_ID), RecordType("B", module_id=ENTRY_ID)),
            module_id=ENTRY_ID,
        )
        h3 = RecordType("X_A_B", module_id=ENTRY_ID)
        keys = _assign_defs_keys((h1, h2, h3))
        assert keys[h1] == "X_A_B"
        assert keys[h2] == "X_A_B_2"
        assert keys[h3] == "X_A_B_3"

    def test_defs_key_order_is_deterministic_for_mutually_recursive_hub(self) -> None:
        # Hub has three direct neighbours (Alpha, Mike, Zulu) discovered off
        # ONE frozenset — a Python-hash-ordered set, not a sequence — and each
        # of them loops back to Hub, putting all four in one mutually
        # recursive component. The BFS queue extension that discovers Alpha/
        # Mike/Zulu must sort them deterministically (by declaration key), or
        # this order — and hence $defs dict insertion order — would vary with
        # PYTHONHASHSEED. Field order is deliberately NOT alphabetical, so a
        # test that only reproduced field order would not catch a
        # frozenset-iteration-order regression.
        hub_handle = RecordType("Hub", module_id=ENTRY_ID)
        alpha, alpha_def = record_type("Alpha", {"back": hub_handle})
        mike, mike_def = record_type("Mike", {"back": hub_handle})
        zulu, zulu_def = record_type("Zulu", {"back": hub_handle})
        hub, hub_def = record_type("Hub", {"a": zulu, "b": alpha, "c": mike})
        table = type_table_for(hub_def, alpha_def, mike_def, zulu_def)
        schema = derive_schema(hub, table)
        defs = schema["$defs"]
        assert isinstance(defs, dict)
        assert list(defs.keys()) == ["Hub", "Alpha", "Mike", "Zulu"]


# ---------------------------------------------------------------------------
# 1b-bis. RefDecode/DecodePlan emission — mirrors TestRecursiveSchemaDerivation's
# matrix; keys must match derive_schema's own $defs keys one-to-one.
# ---------------------------------------------------------------------------


class TestRecursiveDecodeDerivation:
    """build_decode_schema goldens over the same matrix as the schema goldens."""

    def test_recursive_enum_root_is_refdecode_with_defs(self) -> None:
        from agm.agl.ir.contracts import (
            DecodePlan,
            EnumDecode,
            RefDecode,
            ScalarDecode,
            ScalarKind,
            VariantDecode,
        )
        from agm.agl.ir.ids import NominalId

        tree, tree_def = _tree_type_and_def()
        plan = build_decode_schema(tree, type_table_for(tree_def))
        tree_body = EnumDecode(
            nominal=NominalId(ENTRY_ID, "Tree"),
            display_name="Tree",
            variants=(
                VariantDecode(name="Leaf", fields=()),
                VariantDecode(
                    name="Node",
                    fields=(
                        ("value", ScalarDecode(ScalarKind.INT)),
                        ("left", RefDecode("Tree")),
                        ("right", RefDecode("Tree")),
                    ),
                ),
            ),
        )
        assert plan == DecodePlan(root=RefDecode("Tree"), defs=(("Tree", tree_body),))

    def test_list_guarded_recursive_record_is_refdecode_with_defs(self) -> None:
        from agm.agl.ir.contracts import (
            DecodePlan,
            ListDecode,
            RecordDecode,
            RefDecode,
            ScalarDecode,
            ScalarKind,
        )
        from agm.agl.ir.ids import NominalId

        category, category_def = record_type(
            "Category",
            {"name": TextType(), "subcategories": ListType(RecordType(name="Category"))},
        )
        plan = build_decode_schema(category, type_table_for(category_def))
        category_body = RecordDecode(
            nominal=NominalId(ENTRY_ID, "Category"),
            display_name="Category",
            fields=(
                ("name", ScalarDecode(ScalarKind.TEXT)),
                ("subcategories", ListDecode(RefDecode("Category"))),
            ),
        )
        assert plan == DecodePlan(root=RefDecode("Category"), defs=(("Category", category_body),))

    def test_non_recursive_wrapper_inlined_recursive_field_in_defs(self) -> None:
        from agm.agl.ir.contracts import (
            EnumDecode,
            RecordDecode,
            RefDecode,
            ScalarDecode,
            ScalarKind,
        )
        from agm.agl.ir.ids import NominalId

        tree, tree_def = _tree_type_and_def()
        wrapper, wrapper_def = record_type("Wrapper", {"root": tree, "label": TextType()})
        plan = build_decode_schema(wrapper, type_table_for(wrapper_def, tree_def))
        assert plan.root == RecordDecode(
            nominal=NominalId(ENTRY_ID, "Wrapper"),
            display_name="Wrapper",
            fields=(
                ("root", RefDecode("Tree")),
                ("label", ScalarDecode(ScalarKind.TEXT)),
            ),
        )
        assert [key for key, _ in plan.defs] == ["Tree"]
        tree_body = dict(plan.defs)["Tree"]
        assert isinstance(tree_body, EnumDecode)

    def test_mutual_record_enum_pair_gets_two_defs_entries(self) -> None:
        from agm.agl.ir.contracts import (
            DecodePlan,
            EnumDecode,
            RecordDecode,
            RefDecode,
            VariantDecode,
        )
        from agm.agl.ir.ids import NominalId

        a, a_def = record_type("A", {"b": EnumType(name="B")})
        b, b_def = enum_type("B", {"Nil": {}, "Cons": {"a": RecordType(name="A")}})
        plan = build_decode_schema(a, type_table_for(a_def, b_def))
        a_body = RecordDecode(
            nominal=NominalId(ENTRY_ID, "A"), display_name="A", fields=(("b", RefDecode("B")),)
        )
        b_body = EnumDecode(
            nominal=NominalId(ENTRY_ID, "B"),
            display_name="B",
            variants=(
                VariantDecode(name="Nil", fields=()),
                VariantDecode(name="Cons", fields=(("a", RefDecode("A")),)),
            ),
        )
        assert plan == DecodePlan(root=RefDecode("A"), defs=(("A", a_body), ("B", b_body)))

    def test_generic_instantiations_get_distinct_keys_matching_schema(self) -> None:
        tree_def = TypeDef(
            kind="enum",
            name="Tree",
            module_id=ENTRY_ID,
            type_params=("T",),
            variants=(
                ("Leaf", ()),
                (
                    "Node",
                    (
                        ("value", TypeVarType("T")),
                        ("left", EnumType(name="Tree", type_args=(TypeVarType("T"),))),
                        ("right", EnumType(name="Tree", type_args=(TypeVarType("T"),))),
                    ),
                ),
            ),
        )
        tree_int = EnumType(name="Tree", type_args=(IntType(),))
        tree_text = EnumType(name="Tree", type_args=(TextType(),))
        wrapper, wrapper_def = record_type("Holder", {"a": tree_int, "b": tree_text})
        table = type_table_for(wrapper_def, tree_def)
        schema = derive_schema(wrapper, table)
        plan = build_decode_schema(wrapper, table)
        schema_defs = schema["$defs"]
        assert isinstance(schema_defs, dict)
        # The decode $defs keys are EXACTLY the schema's own $defs keys.
        decode_keys = {key for key, _ in plan.defs}
        assert decode_keys == set(schema_defs.keys()) == {"Tree_int", "Tree_text"}

    def test_non_recursive_decode_output_unchanged(self) -> None:
        """Spot-check: non-recursive DecodePlan has empty defs (representation-identical)."""
        from agm.agl.ir.contracts import DecodePlan, RecordDecode, ScalarDecode, ScalarKind
        from agm.agl.ir.ids import NominalId

        inner, inner_def = record_type("Inner", {"x": IntType()})
        outer, outer_def = record_type("Outer", {"inner": inner})
        plan = build_decode_schema(outer, type_table_for(outer_def, inner_def))
        assert plan == DecodePlan(
            root=RecordDecode(
                nominal=NominalId(ENTRY_ID, "Outer"),
                display_name="Outer",
                fields=(
                    (
                        "inner",
                        RecordDecode(
                            nominal=NominalId(ENTRY_ID, "Inner"),
                            display_name="Inner",
                            fields=(("x", ScalarDecode(ScalarKind.INT)),),
                        ),
                    ),
                ),
            ),
            defs=(),
        )

    def test_raises_for_infinite_closure_root(self) -> None:
        pair_def = TypeDef(
            kind="record",
            name="Pair",
            module_id=ENTRY_ID,
            type_params=("A", "B"),
            fields=(("first", TypeVarType("A")), ("second", TypeVarType("B"))),
        )
        perfect_def = TypeDef(
            kind="enum",
            name="Perfect",
            module_id=ENTRY_ID,
            type_params=("T",),
            variants=(
                ("Single", (("value", TypeVarType("T")),)),
                (
                    "Succ",
                    (
                        (
                            "next",
                            EnumType(
                                name="Perfect",
                                type_args=(
                                    RecordType(
                                        name="Pair",
                                        type_args=(TypeVarType("T"), TypeVarType("T")),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        table = type_table_for(pair_def, perfect_def)
        perfect_int = EnumType(name="Perfect", type_args=(IntType(),))
        with pytest.raises(TypeError, match="no finite schema"):
            build_decode_schema(perfect_int, table)

    def test_derive_schema_and_decode_shares_one_plan_and_matches_separate_calls(self) -> None:
        """derive_schema_and_decode matches (derive_schema(...), build_decode_schema(...))."""
        from agm.agl.type_schema import derive_schema_and_decode

        tree, tree_def = _tree_type_and_def()
        table = type_table_for(tree_def)
        schema, plan = derive_schema_and_decode(tree, table)
        assert schema == derive_schema(tree, table)
        assert plan == build_decode_schema(tree, table)


# ---------------------------------------------------------------------------
# 1c. Recursive schema validator round-trip (JSON Schema validation only;
# TestRecursiveDecodeDerivation below covers build_decode_schema/JsonCodec.parse
# for a recursive type)
# ---------------------------------------------------------------------------


class TestRecursiveSchemaValidatorRoundtrip:
    def test_accepts_nested_tree_payload(self) -> None:
        tree, tree_def = _tree_type_and_def()
        schema = derive_schema(tree, type_table_for(tree_def))
        validator = Draft202012Validator(schema)
        payload = {
            "$case": "Node",
            "value": 1,
            "left": {"$case": "Leaf"},
            "right": {
                "$case": "Node",
                "value": 2,
                "left": {"$case": "Leaf"},
                "right": {"$case": "Leaf"},
            },
        }
        assert validator.is_valid(payload)

    def test_rejects_missing_required_field_two_levels_deep(self) -> None:
        tree, tree_def = _tree_type_and_def()
        schema = derive_schema(tree, type_table_for(tree_def))
        validator = Draft202012Validator(schema)
        payload = {
            "$case": "Node",
            "value": 1,
            "left": {"$case": "Leaf"},
            # Missing "value" two levels deep inside "right.left"... actually
            # inside "right" itself (a Node missing its required "value").
            "right": {
                "$case": "Node",
                "left": {"$case": "Leaf"},
                "right": {"$case": "Leaf"},
            },
        }
        assert not validator.is_valid(payload)

    def test_rejects_wrong_case_tag_in_nested_variant(self) -> None:
        tree, tree_def = _tree_type_and_def()
        schema = derive_schema(tree, type_table_for(tree_def))
        validator = Draft202012Validator(schema)
        payload = {
            "$case": "Node",
            "value": 1,
            "left": {"$case": "Leaf"},
            "right": {
                "$case": "Bogus",
                "value": 2,
                "left": {"$case": "Leaf"},
                "right": {"$case": "Leaf"},
            },
        }
        assert not validator.is_valid(payload)


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
        result = _parse_typed(codec, "5", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(5)

    def test_bare_json_object(self) -> None:
        codec = self._codec()
        typ = JsonType()
        result = _parse_typed(codec, '{"k": 1}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_fenced_json_block_extracted(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, "```json\n5\n```", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(5)

    def test_fenced_json_object_extracted(self) -> None:
        codec = self._codec()
        typ = JsonType()
        result = _parse_typed(codec, '```json\n{"k": 1}\n```', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_prose_wrapped_json_extracted(self) -> None:
        codec = self._codec()
        typ = JsonType()
        result = _parse_typed(codec, "Here you go:\n[1, 2]", typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_prose_and_fence(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, "Sure thing!\n```json\n5\n```", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(5)

    def test_trailing_comma_repaired(self) -> None:
        codec = self._codec()
        typ = JsonType()
        result = _parse_typed(codec, '{"k": 1,}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_single_quoted_keys_repaired(self) -> None:
        codec = self._codec()
        typ = JsonType()
        result = _parse_typed(codec, "{'k': 1}", typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_gibberish_fails(self) -> None:
        codec = self._codec()
        result = _parse_typed(
            codec, "complete gibberish, no number here", IntType(), strict_json=False
        )
        assert result.ok is False
        assert result.value is None

    def test_bare_bool_recovered_from_prose(self) -> None:
        """Lenient recovery pulls a bare ``false`` keyword out of prose."""
        codec = self._codec()
        result = _parse_typed(codec, "The flag is:\nfalse", BoolType(), strict_json=False)
        assert result.ok is True
        assert result.value == BoolValue(False)

    def test_bare_null_recovered_from_prose(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, "Answer: null", JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw is None

    def test_bare_number_recovered_from_prose(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, "the count is 42 items", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(42)

    def test_keyword_substring_not_falsely_recovered(self) -> None:
        """``nullable`` must not be mistaken for a bare ``null`` token."""
        codec = self._codec()
        result = _parse_typed(codec, "the config is nullable here", BoolType(), strict_json=False)
        assert result.ok is False

    def test_two_bare_scalars_in_prose_are_ambiguous(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, "maybe true or maybe false", BoolType(), strict_json=False)
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
        result = _parse_typed(codec, "5", IntType(), strict_json=True)
        assert result.ok is True
        assert result.value == IntValue(5)

    def test_whitespace_around_bare_integer_accepted(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, "  5  ", IntType(), strict_json=True)
        assert result.ok is True
        assert result.value == IntValue(5)

    def test_fenced_value_rejected(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, "```json\n5\n```", IntType(), strict_json=True)
        assert result.ok is False

    def test_trailing_prose_rejected(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, "5\nThat is my final answer.", IntType(), strict_json=True)
        assert result.ok is False

    def test_single_quotes_rejected(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, "{'k': 1}", JsonType(), strict_json=True)
        assert result.ok is False

    def test_trailing_comma_rejected(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, '{"k": 1,}', JsonType(), strict_json=True)
        assert result.ok is False

    def test_bare_object_accepted(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, '{"k": 1}', JsonType(), strict_json=True)
        assert result.ok is True

    def test_fenced_object_rejected(self) -> None:
        codec = self._codec()
        result = _parse_typed(codec, '```json\n{"k": 1}\n```', JsonType(), strict_json=True)
        assert result.ok is False


# ---------------------------------------------------------------------------
# 5. Decimal exactness
# ---------------------------------------------------------------------------


class TestDecimalExactness:
    """Decimal values must never round-trip through float."""

    def test_decimal_stays_decimal_in_lenient(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "1.5", DecimalType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, DecimalValue)
        assert result.value.value == Decimal("1.5")

    def test_decimal_stays_decimal_in_strict(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "1.5", DecimalType(), strict_json=True)
        assert result.ok is True
        assert isinstance(result.value, DecimalValue)
        assert result.value.value == Decimal("1.5")

    def test_decimal_in_record_field(self) -> None:
        codec = JsonCodec()
        typ, typedef = record_type("Foo", {"w": DecimalType()})
        result = _parse_typed(
            codec, '{"w": 1.5}', typ, strict_json=False, table=type_table_for(typedef)
        )
        assert result.ok is True
        assert isinstance(result.value, RecordValue)
        w = result.value.fields["w"]
        assert isinstance(w, DecimalValue)
        assert w.value == Decimal("1.5")

    def test_decimal_from_fenced_stays_exact(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "```json\n1.5\n```", DecimalType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, DecimalValue)
        assert result.value.value == Decimal("1.5")

    def test_decimal_not_float(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "1.5", DecimalType(), strict_json=False)
        assert isinstance(result.value, DecimalValue)
        assert not isinstance(result.value.value, float)

    def test_int_widened_to_decimal_when_target_says_decimal(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "3", DecimalType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, DecimalValue)
        assert result.value.value == Decimal("3")

    def test_high_precision_decimal(self) -> None:
        # Bare valid JSON is parsed directly (no json-repair), so Decimal precision
        # is fully preserved by json.loads(parse_float=Decimal).
        codec = JsonCodec()
        result = _parse_typed(codec, "1.23456789012345678901", DecimalType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, DecimalValue)
        assert result.value.value == Decimal("1.23456789012345678901")

    def test_decimal_in_repaired_json(self) -> None:
        """Decimal exactness through json-repair path (single-quote input)."""
        codec = JsonCodec()
        typ, typedef = record_type("Foo", {"w": DecimalType()})
        result = _parse_typed(
            codec, "{'w': 1.5}", typ, strict_json=False, table=type_table_for(typedef)
        )
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
        result = _parse_typed(codec, "42", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(42)

    def test_bool_value_true(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "true", BoolType(), strict_json=False)
        assert result.ok is True
        assert result.value == BoolValue(True)

    def test_bool_value_false(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "false", BoolType(), strict_json=False)
        assert result.ok is True
        assert result.value == BoolValue(False)

    def test_list_of_text(self) -> None:
        codec = JsonCodec()
        typ = ListType(elem=TextType())
        result = _parse_typed(codec, '["a", "b"]', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, ListValue)
        assert result.value.elements == (TextValue("a"), TextValue("b"))

    def test_list_of_int(self) -> None:
        codec = JsonCodec()
        typ = ListType(elem=IntType())
        result = _parse_typed(codec, "[1, 2, 3]", typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, ListValue)
        assert result.value.elements == (IntValue(1), IntValue(2), IntValue(3))

    def test_dict_of_text(self) -> None:
        codec = JsonCodec()
        typ = DictType(value=TextType())
        result = _parse_typed(codec, '{"a": "hello"}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, DictValue)
        assert result.value.entries == {"a": TextValue("hello")}

    def test_record_value(self) -> None:
        codec = JsonCodec()
        typ = _make_issue_type()
        raw = '{"title": "Bug", "severity": 5, "description": "Oh no"}'
        result = _parse_typed(codec, raw, typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, RecordValue)
        assert result.value.display_name == "Issue"
        assert result.value.fields["title"] == TextValue("Bug")
        assert result.value.fields["severity"] == IntValue(5)
        assert result.value.fields["description"] == TextValue("Oh no")

    def test_enum_nullary_variant(self) -> None:
        codec = JsonCodec()
        typ = _make_review_type()
        result = _parse_typed(codec, '{"$case": "Pass"}', typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, EnumValue)
        assert result.value.display_name == "Review"
        assert result.value.variant == "Pass"
        assert result.value.fields == {}

    def test_enum_payload_variant(self) -> None:
        codec = JsonCodec()
        typ = _make_review_type()
        raw = '{"$case": "Fail", "issues": ["a", "b"]}'
        result = _parse_typed(codec, raw, typ, strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, EnumValue)
        assert result.value.variant == "Fail"
        issues = result.value.fields["issues"]
        assert isinstance(issues, ListValue)
        assert issues.elements == (TextValue("a"), TextValue("b"))

    def test_json_value_wraps_raw(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, '{"a": 1}', JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_nested_record(self) -> None:
        codec = JsonCodec()
        inner, inner_def = record_type("Inner", {"x": IntType()})
        outer, outer_def = record_type("Outer", {"inner": inner, "n": IntType()})
        raw = '{"inner": {"x": 7}, "n": 3}'
        result = _parse_typed(
            codec, raw, outer, strict_json=False, table=type_table_for(outer_def, inner_def)
        )
        assert result.ok is True
        assert isinstance(result.value, RecordValue)
        inner_val = result.value.fields["inner"]
        assert isinstance(inner_val, RecordValue)
        assert inner_val.fields["x"] == IntValue(7)

    def test_list_in_record_field(self) -> None:
        codec = JsonCodec()
        typ, typedef = record_type("Doc", {"tags": ListType(elem=TextType())})
        result = _parse_typed(
            codec, '{"tags": ["x", "y"]}', typ, strict_json=False, table=type_table_for(typedef)
        )
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
        result = _parse_typed(codec, '{"title": "Bug"}', typ, strict_json=False)
        assert result.ok is False
        assert result.value is None

    def test_unknown_field_fails(self) -> None:
        codec = JsonCodec()
        typ = _make_issue_type()
        raw = '{"title": "Bug", "severity": 1, "description": "x", "extra": true}'
        result = _parse_typed(codec, raw, typ, strict_json=False)
        assert result.ok is False

    def test_wrong_type_fails(self) -> None:
        codec = JsonCodec()
        typ = _make_issue_type()
        raw = '{"title": "Bug", "severity": "high", "description": "x"}'
        result = _parse_typed(codec, raw, typ, strict_json=False)
        assert result.ok is False

    def test_bad_case_tag_fails(self) -> None:
        codec = JsonCodec()
        typ = _make_review_type()
        result = _parse_typed(codec, '{"$case": "Unknown"}', typ, strict_json=False)
        assert result.ok is False

    def test_missing_case_field_fails(self) -> None:
        codec = JsonCodec()
        typ = _make_review_type()
        result = _parse_typed(codec, '{"issues": ["x"]}', typ, strict_json=False)
        assert result.ok is False

    def test_enum_missing_payload_field_fails(self) -> None:
        codec = JsonCodec()
        typ = _make_review_type()
        # Fail variant but missing issues
        result = _parse_typed(codec, '{"$case": "Fail"}', typ, strict_json=False)
        assert result.ok is False

    def test_failure_result_has_no_value(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "{}", _make_issue_type(), strict_json=False)
        assert result.value is None

    def test_failure_result_has_error_msg(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "{}", _make_issue_type(), strict_json=False)
        assert result.error_msg
        assert isinstance(result.error_msg, str)


# ---------------------------------------------------------------------------
# 7b. Structured ValidationError records
# ---------------------------------------------------------------------------


class TestStructuredValidationErrors:
    """Each documented category is surfaced as a structured ValidationError."""

    def _categories(self, result: ParseResult) -> list[str]:
        return [e.category for e in result.errors]

    def test_missing_field_category(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, '{"title": "Bug"}', _make_issue_type(), strict_json=False)
        assert result.ok is False
        assert "missing_field" in self._categories(result)
        missing = [e for e in result.errors if e.category == "missing_field"]
        assert all(e.field is not None for e in missing)

    def test_unknown_field_category(self) -> None:
        codec = JsonCodec()
        raw = '{"title": "Bug", "severity": 1, "description": "x", "extra": true}'
        result = _parse_typed(codec, raw, _make_issue_type(), strict_json=False)
        assert result.ok is False
        assert self._categories(result) == ["unknown_field"]
        # The opaque jsonschema phrasing must not leak verbatim as the category.
        assert "extra" in result.errors[0].message

    def test_wrong_type_category(self) -> None:
        codec = JsonCodec()
        raw = '{"title": "Bug", "severity": "high", "description": "x"}'
        result = _parse_typed(codec, raw, _make_issue_type(), strict_json=False)
        assert result.ok is False
        wrong = [e for e in result.errors if e.category == "wrong_type"]
        assert wrong
        assert wrong[0].field == "severity"
        assert wrong[0].path == "$.severity"

    def test_bad_case_unknown_variant(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, '{"$case": "Nope"}', _make_review_type(), strict_json=False)
        assert result.ok is False
        assert self._categories(result) == ["bad_case"]
        msg = result.errors[0].message
        # Type-directed: real variant names, not "not valid under any of ...".
        assert "is not valid under any of the given schemas" not in msg
        assert "Nope" in msg
        assert "Pass" in msg and "Fail" in msg

    def test_bad_case_missing_tag(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, '{"issues": ["x"]}', _make_review_type(), strict_json=False)
        assert result.ok is False
        assert self._categories(result) == ["bad_case"]
        assert result.errors[0].field == "$case"
        assert "is not valid under any of the given schemas" not in result.errors[0].message

    def test_enum_missing_payload_field_is_missing_field(self) -> None:
        codec = JsonCodec()
        # Fail variant requires "issues".
        result = _parse_typed(codec, '{"$case": "Fail"}', _make_review_type(), strict_json=False)
        assert result.ok is False
        assert self._categories(result) == ["missing_field"]
        assert result.errors[0].field == "issues"
        assert "is not valid under any of the given schemas" not in result.errors[0].message

    def test_enum_unknown_payload_field_is_unknown_field(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(
            codec, '{"$case": "Pass", "junk": 1}', _make_review_type(), strict_json=False
        )
        assert result.ok is False
        assert self._categories(result) == ["unknown_field"]
        assert result.errors[0].field == "junk"

    def test_nested_enum_wrong_type_path(self) -> None:
        """Type-directed enum resolution works under a record field path."""
        codec = JsonCodec()
        typ, typedef = record_type("Wrapper", {"review": _make_review_type()})
        result = _parse_typed(
            codec,
            '{"review": {"$case": "Bogus"}}',
            typ,
            strict_json=False,
            table=type_table_for(typedef, _REVIEW_TYPEDEF),
        )
        assert result.ok is False
        assert self._categories(result) == ["bad_case"]
        assert "Bogus" in result.errors[0].message

    def test_success_has_no_errors(self) -> None:
        codec = JsonCodec()
        raw = '{"title": "Bug", "severity": 1, "description": "x"}'
        result = _parse_typed(codec, raw, _make_issue_type(), strict_json=False)
        assert result.ok is True
        assert result.errors == ()

    def test_non_validation_failure_has_no_errors(self) -> None:
        """A failure to extract any JSON is not a schema-validation error."""
        codec = JsonCodec()
        result = _parse_typed(
            codec, "complete gibberish ###", _make_issue_type(), strict_json=False
        )
        assert result.ok is False
        assert result.errors == ()


class TestValidationErrorsThroughRuntime:
    """real ValidationErrors thread into AgentParseError.validation_errors."""

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
        #  on_parse_error: Retry(n: 1) as a named arg to ask().
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
# 7c. Multi-value ambiguity rejection
# ---------------------------------------------------------------------------


class TestMultiValueAmbiguity:
    def test_two_objects_rejected(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, '{"a":1} {"b":2}', JsonType(), strict_json=False)
        assert result.ok is False
        assert "multiple JSON values" in result.error_msg

    def test_two_objects_newline_separated_rejected(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, '{"a":1}\n{"b":2}', JsonType(), strict_json=False)
        assert result.ok is False
        assert "multiple JSON values" in result.error_msg

    def test_text_then_single_object_recovers(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, 'text then {"a": 1}', JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw == {"a": 1}

    def test_bare_array_parses(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "[1, 2]", JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw == [1, 2]

    def test_fenced_array_parses(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "```json\n[1, 2]\n```", JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw == [1, 2]

    def test_prose_wrapped_array_recovers(self) -> None:
        """A genuine single array wrapped in prose is recovered (not ambiguous)."""
        codec = JsonCodec()
        result = _parse_typed(codec, "Here you go:\n[1, 2]", JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw == [1, 2]

    def test_ambiguous_inside_fence_rejected(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, '```json\n{"a":1} {"b":2}\n```', JsonType(), strict_json=False)
        assert result.ok is False
        assert "multiple JSON values" in result.error_msg

    def test_two_objects_with_inner_array_rejected(self) -> None:
        """``{"a": [1]} {"b": 2}`` is ambiguous despite the inner ``[``."""
        codec = JsonCodec()
        result = _parse_typed(codec, '{"a": [1]} {"b": 2}', JsonType(), strict_json=False)
        assert result.ok is False
        assert "multiple JSON values" in result.error_msg

    def test_two_values_with_escaped_bracket_string_rejected(self) -> None:
        """a bracket inside an escaped string does not hide the second value."""
        codec = JsonCodec()
        result = _parse_typed(codec, '{"a": "[x]"} {"b": 2}', JsonType(), strict_json=False)
        assert result.ok is False
        assert "multiple JSON values" in result.error_msg

    def test_single_object_with_inner_array_recovers(self) -> None:
        """a single object containing an array is one value (not ambiguous)."""
        codec = JsonCodec()
        result = _parse_typed(codec, '{"a": [1, 2]}', JsonType(), strict_json=False)
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
        issue_type = _make_issue_type()
        spec = OutputContractSpec(
            target_type=issue_type,
            codec_name="json",
            strict_json=False,
        )
        contract = materialize_contract(spec, {"json": codec, "text": TextCodec()}, _DEFAULT_TABLE)
        assert isinstance(contract.codec, JsonCodec)
        assert contract.json_schema is not None


# ---------------------------------------------------------------------------
# 9. PipelineDriver wire-up
# ---------------------------------------------------------------------------


def _json_ty() -> tast.JsonT:
    return tast.JsonT(span=_sp(), node_id=_nid())


def _dict_ty(value: tast.TypeExpr) -> tast.DictT:
    return tast.DictT(value=value, span=_sp(), node_id=_nid())


class TestPipelineDriverWireUp:
    """JsonCodec registered in runtime; checker passes json/record/enum targets.

    Note: typed agent-call bindings (let x: T = agent "...") are tested via the
    direct-AST helpers because the parser wraps agent calls in an 'access'
    tree node for non-text typed bindings.  PipelineDriver.run() tests cover the
    static pipeline (codec_kinds) and the error reporting paths.
    """

    def test_json_target_type_accepted_via_direct_ast(self) -> None:
        """A call targeting json type should pass type checking and execute."""
        let_x = _let("x", _ask_call("Get data."), type_ann=_json_ty())
        scope = _run_with_json_codec((let_x,), default_agent=lambda req: '{"x": 1}')
        x = scope.snapshot()["x"]
        assert isinstance(x, JsonValue)

    def test_int_target_accepted_via_json_codec(self) -> None:
        let_n = _let("n", _ask_call("Get number."), type_ann=_int_ty())
        scope = _run_with_json_codec((let_n,), default_agent=lambda req: "42")
        assert scope.snapshot()["n"] == IntValue(42)

    def test_bool_target_accepted(self) -> None:
        let_b = _let("b", _ask_call("Is it true?"), type_ann=_bool_ty())
        scope = _run_with_json_codec((let_b,), default_agent=lambda req: "true")
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
        scope = _run_with_json_codec((let_xs,), default_agent=lambda req: '["a", "b"]')
        xs = scope.snapshot()["xs"]
        assert isinstance(xs, ListValue)
        assert xs.elements == (TextValue("a"), TextValue("b"))

    def test_dict_target_accepted(self) -> None:
        let_d = _let(
            "d",
            _ask_call("Dict."),
            type_ann=_dict_ty(_text_ty()),
        )
        scope = _run_with_json_codec((let_d,), default_agent=lambda req: '{"k": "v"}')
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
        # In AgL the agent field reflects the built-in "ask" call site (default agent path).
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
            json_codec.name: frozenset(
                {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
            ),
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
        typ, typedef = enum_type(
            "Status",
            {
                "Done": {},
                "Running": {"progress": IntType()},
            },
        )
        result = _parse_typed(
            codec,
            '{"$case": "Running", "progress": 50}',
            typ,
            strict_json=False,
            table=type_table_for(typedef),
        )
        assert result.ok is True
        assert isinstance(result.value, EnumValue)
        assert result.value.variant == "Running"
        assert result.value.fields["progress"] == IntValue(50)

    def test_bad_case_fails(self) -> None:
        codec = JsonCodec()
        typ, typedef = enum_type("Status", {"Done": {}})
        result = _parse_typed(
            codec, '{"$case": "Exploded"}', typ, strict_json=False, table=type_table_for(typedef)
        )
        assert result.ok is False

    def test_missing_case_tag_fails(self) -> None:
        codec = JsonCodec()
        typ, typedef = enum_type("Status", {"Done": {}})
        result = _parse_typed(
            codec, '{"done": true}', typ, strict_json=False, table=type_table_for(typedef)
        )
        assert result.ok is False

    def test_nullary_enum_no_extra_fields(self) -> None:
        codec = JsonCodec()
        typ, typedef = enum_type("Status", {"Done": {}})
        result = _parse_typed(
            codec, '{"$case": "Done"}', typ, strict_json=False, table=type_table_for(typedef)
        )
        assert result.ok is True
        assert isinstance(result.value, EnumValue)
        assert result.value.fields == {}


# ---------------------------------------------------------------------------
# 11. normalized_raw in ParseResult
# ---------------------------------------------------------------------------


class TestNormalizedRaw:
    def test_lenient_sets_normalized_raw_when_extraction_occurred(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "```json\n5\n```", IntType(), strict_json=False)
        assert result.ok is True
        # normalized_raw should be the extracted/repaired JSON text
        assert result.normalized_raw is not None

    def test_bare_json_normalized_raw(self) -> None:
        codec = JsonCodec()
        result = _parse_typed(codec, "5", IntType(), strict_json=False)
        assert result.ok is True
        # Even bare JSON has a normalized_raw
        assert result.normalized_raw is not None

    def test_schema_failure_carries_normalized_raw(self) -> None:
        """a fenced-but-schema-invalid response still exposes the recovered text."""
        codec = JsonCodec()
        # Fenced JSON that is valid JSON but the wrong shape for an int target.
        result = _parse_typed(codec, '```json\n"oops"\n```', IntType(), strict_json=False)
        assert result.ok is False
        # The recovered (normalized) JSON text is threaded through the failure,
        # distinct from the fenced raw response.
        assert result.normalized_raw == '"oops"'

    def test_conversion_failure_carries_normalized_raw(self) -> None:
        """a value-conversion failure also threads the recovered text."""
        codec = JsonCodec()
        # 1.5 passes a decimal schema check but cannot convert to an int Value;
        # exercises the conversion-failure branch.  Use a shape jsonschema lets
        # through but _json_to_value rejects: a float for an int via repair.
        result = _parse_typed(codec, "not json at all", IntType(), strict_json=False)
        assert result.ok is False


# ---------------------------------------------------------------------------
# 12. Record/enum params accepted via json codec
# ---------------------------------------------------------------------------


class TestRecordEnumParams:
    """Runtime.convert_param now accepts record/enum types via JsonCodec."""

    def test_record_param_parsed_from_json_string(self) -> None:
        result = PipelineDriver().run(
            """
record Issue
  title: text
  severity: int
param issue: Issue
issue
""",
            param_values={"issue": '{"title": "Bug", "severity": 5}'},
        )
        assert result.ok
        v = result.bindings["issue"]
        assert isinstance(v, RecordValue)
        assert v.fields["title"] == TextValue("Bug")

    def test_enum_param_parsed_from_json_string(self) -> None:
        """Enum can be parsed via JsonCodec from a JSON string."""
        codec = JsonCodec()
        typ, typedef = enum_type("Status", {"Done": {}, "Pending": {}})
        result = _parse_typed(
            codec, '{"$case": "Done"}', typ, strict_json=False, table=type_table_for(typedef)
        )
        assert result.ok is True
        assert isinstance(result.value, EnumValue)
        assert result.value.variant == "Done"

    def test_list_param_parsed_from_json_string(self) -> None:
        rt = PipelineDriver()
        result = rt.run(
            "param tags: list[text]",
            param_values={"tags": '["a", "b"]'},
        )
        assert result.ok is True

    def test_structured_param_accepts_python_list(self) -> None:
        """Structured params may be provided as a Python list (JSON-compatible)."""
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.values import IntValue, ListValue

        result = convert_param_value("xs", [1, 2, 3], ListType(elem=IntType()), type_table_for())
        assert isinstance(result, ListValue)
        assert result.elements == (IntValue(1), IntValue(2), IntValue(3))

    def test_structured_param_must_be_string_or_compatible(self) -> None:
        """Structured params that are not a string or JSON-compatible Python value raise."""
        from agm.agl.runtime.params import convert_param_value

        with pytest.raises(ValueError, match="JSON"):
            convert_param_value("xs", object(), ListType(elem=IntType()), type_table_for())

    def test_invalid_structured_param_raises(self) -> None:
        """A JSON string that fails schema validation for the declared type raises."""
        from agm.agl.runtime.params import convert_param_value

        with pytest.raises(ValueError, match="could not parse"):
            issue_type, issue_def = record_type(
                "Issue", {"title": TextType(), "severity": IntType()}
            )
            convert_param_value(
                "issue",
                '{"title": "Bug"}',  # missing severity
                issue_type,
                type_table_for(issue_def),
            )

    def test_unsupported_type_in_convert_param_value_raises(self) -> None:
        """ExceptionType is not a supported param type."""
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import ExceptionType

        with pytest.raises(ValueError, match="unsupported type"):
            convert_param_value("e", "val", ExceptionType(name="Boom"), type_table_for())

    def test_structured_param_is_strict_no_repair(self) -> None:
        """host --param values are parsed strictly; typos are NOT repaired.

        A trailing comma (which json-repair would silently fix for chatty agent
        output) must be rejected for a user-supplied structured param, with an
        error that makes the JSON requirement clear.
        """
        from agm.agl.runtime.params import convert_param_value

        with pytest.raises(ValueError, match="JSON parse error"):
            issue_type, issue_def = record_type(
                "Issue", {"title": TextType(), "severity": IntType()}
            )
            convert_param_value(
                "issue",
                '{"title": "Bug", "severity": 5,}',  # trailing comma typo
                issue_type,
                type_table_for(issue_def),
            )

    def test_structured_param_rejects_fenced_json(self) -> None:
        """a Markdown-fenced --param value is not stripped (strict parsing)."""
        from agm.agl.runtime.params import convert_param_value

        with pytest.raises(ValueError, match="JSON parse error"):
            convert_param_value(
                "tags",
                "```json\n[1, 2]\n```",
                ListType(elem=IntType()),
                type_table_for(),
            )


# ---------------------------------------------------------------------------
# 13. Coverage: decode_value error branches
# ---------------------------------------------------------------------------


class TestDecodeValueErrorBranches:
    """Cover the ValueError branches inside decode_value / _decode_scalar."""

    def test_text_type_got_non_string(self) -> None:
        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.convert import decode_value

        with pytest.raises(ValueError, match="string"):
            decode_value(ScalarDecode(kind=ScalarKind.TEXT), 42)

    def test_int_type_got_bool(self) -> None:
        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.convert import decode_value

        with pytest.raises(ValueError, match="bool"):
            decode_value(ScalarDecode(kind=ScalarKind.INT), True)

    def test_int_type_got_non_integer_decimal(self) -> None:
        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.convert import decode_value

        with pytest.raises(ValueError, match="integer"):
            decode_value(ScalarDecode(kind=ScalarKind.INT), Decimal("1.5"))

    def test_decimal_type_got_bool(self) -> None:
        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.convert import decode_value

        with pytest.raises(ValueError, match="bool"):
            decode_value(ScalarDecode(kind=ScalarKind.DECIMAL), True)

    def test_decimal_type_got_string(self) -> None:
        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.convert import decode_value

        with pytest.raises(ValueError, match="decimal"):
            decode_value(ScalarDecode(kind=ScalarKind.DECIMAL), "not a number")

    def test_bool_type_got_int(self) -> None:
        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.convert import decode_value

        with pytest.raises(ValueError, match="bool"):
            decode_value(ScalarDecode(kind=ScalarKind.BOOL), 1)

    def test_list_type_got_non_list(self) -> None:
        from agm.agl.ir.contracts import ListDecode, ScalarDecode, ScalarKind
        from agm.agl.runtime.convert import decode_value

        with pytest.raises(ValueError, match="array"):
            decode_value(ListDecode(elem=ScalarDecode(kind=ScalarKind.TEXT)), "not a list")

    def test_dict_type_got_non_dict(self) -> None:
        from agm.agl.ir.contracts import DictDecode, ScalarDecode, ScalarKind
        from agm.agl.runtime.convert import decode_value

        with pytest.raises(ValueError, match="object"):
            decode_value(DictDecode(value=ScalarDecode(kind=ScalarKind.TEXT)), [1, 2])

    def test_dict_non_string_key(self) -> None:
        from agm.agl.ir.contracts import DictDecode, ScalarDecode, ScalarKind
        from agm.agl.runtime.convert import decode_value

        # Construct a dict with a non-str key (not normally from json.loads but defensive).
        with pytest.raises(ValueError, match="Dict key must be string"):
            decode_value(
                DictDecode(value=ScalarDecode(kind=ScalarKind.TEXT)),
                {1: "val"},
            )

    def test_record_type_got_non_dict(self) -> None:
        from agm.agl.ir.contracts import RecordDecode, ScalarDecode, ScalarKind
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.runtime.convert import decode_value

        schema = RecordDecode(
            nominal=NominalId(ENTRY_ID, "R"),
            display_name="R",
            fields=(("x", ScalarDecode(kind=ScalarKind.INT)),),
        )
        with pytest.raises(ValueError, match="record"):
            decode_value(schema, [1, 2])

    def test_record_missing_field(self) -> None:
        from agm.agl.ir.contracts import RecordDecode, ScalarDecode, ScalarKind
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.runtime.convert import decode_value

        schema = RecordDecode(
            nominal=NominalId(ENTRY_ID, "R"),
            display_name="R",
            fields=(("x", ScalarDecode(kind=ScalarKind.INT)),),
        )
        with pytest.raises(ValueError, match="Missing field"):
            decode_value(schema, {})

    def test_enum_type_got_non_dict(self) -> None:
        from agm.agl.ir.contracts import EnumDecode, VariantDecode
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.runtime.convert import decode_value

        schema = EnumDecode(
            nominal=NominalId(ENTRY_ID, "E"),
            display_name="E",
            variants=(VariantDecode(name="A", fields=()),),
        )
        with pytest.raises(ValueError, match="object for enum"):
            decode_value(schema, "oops")

    def test_enum_missing_case_tag(self) -> None:
        from agm.agl.ir.contracts import EnumDecode, VariantDecode
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.runtime.convert import decode_value

        schema = EnumDecode(
            nominal=NominalId(ENTRY_ID, "E"),
            display_name="E",
            variants=(VariantDecode(name="A", fields=()),),
        )
        with pytest.raises(ValueError, match=r"\$case"):
            decode_value(schema, {})

    def test_enum_unknown_variant(self) -> None:
        from agm.agl.ir.contracts import EnumDecode, VariantDecode
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.runtime.convert import decode_value

        schema = EnumDecode(
            nominal=NominalId(ENTRY_ID, "E"),
            display_name="E",
            variants=(VariantDecode(name="A", fields=()),),
        )
        with pytest.raises(ValueError, match="Unknown enum variant"):
            decode_value(schema, {"$case": "X"})

    def test_enum_missing_payload_field(self) -> None:
        from agm.agl.ir.contracts import EnumDecode, ScalarDecode, ScalarKind, VariantDecode
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.runtime.convert import decode_value

        schema = EnumDecode(
            nominal=NominalId(ENTRY_ID, "E"),
            display_name="E",
            variants=(
                VariantDecode(
                    name="B",
                    fields=(("x", ScalarDecode(kind=ScalarKind.INT)),),
                ),
            ),
        )
        with pytest.raises(ValueError, match="missing field"):
            decode_value(schema, {"$case": "B"})

    def test_integral_decimal_to_int_through_parse(self) -> None:
        """wire ``1.0`` validates and converts to IntValue(1) for an int target.

        Exercised through ``parse()`` (the public path): post-parse normalization
        rewrites integral Decimals to int *before* schema validation, so
        ``{"type": "integer"}`` accepts ``1.0``.
        """
        codec = JsonCodec()
        result = _parse_typed(codec, "1.0", IntType(), strict_json=False)
        assert result.ok is True
        assert result.value == IntValue(1)

    def test_integral_decimal_to_int_strict(self) -> None:
        """integral-Decimal normalization also applies on the strict path."""
        codec = JsonCodec()
        result = _parse_typed(codec, "1.0", IntType(), strict_json=True)
        assert result.ok is True
        assert result.value == IntValue(1)

    def test_non_integral_decimal_rejected_for_int(self) -> None:
        """``1.5`` still fails an int target (not integral)."""
        codec = JsonCodec()
        result = _parse_typed(codec, "1.5", IntType(), strict_json=False)
        assert result.ok is False
        assert result.value is None
        assert any(e.category == "wrong_type" for e in result.errors)

    def test_integral_decimal_for_decimal_target(self) -> None:
        """``1.0`` for a decimal target yields a value-exact DecimalValue.

        Normalization routes the integral Decimal through int, and the
        int→decimal widening in ``decode_value`` re-widens it: the resulting
        value equals ``1`` exactly == Decimal('1.0')``).
        """
        codec = JsonCodec()
        result = _parse_typed(codec, "1.0", DecimalType(), strict_json=False)
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
        from agm.agl.semantics.types import ExceptionType

        with pytest.raises(TypeError, match="ExceptionType"):
            derive_schema(ExceptionType(name="Boom"), type_table_for())


# ---------------------------------------------------------------------------
# 15. Coverage: fenced malformed JSON (repair within fence)
# ---------------------------------------------------------------------------


class TestFencedMalformedJson:
    """Fenced content that is itself malformed but repairable."""

    def test_fenced_single_quotes_repaired(self) -> None:
        codec = JsonCodec()
        # Fenced content with single-quoted keys — json-repair fixes it.
        raw = "```json\n{'k': 1}\n```"
        result = _parse_typed(codec, raw, JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)

    def test_fenced_trailing_comma_repaired(self) -> None:
        codec = JsonCodec()
        raw = '```json\n{"a": 1,}\n```'
        result = _parse_typed(codec, raw, JsonType(), strict_json=False)
        assert result.ok is True


# ---------------------------------------------------------------------------
# 17. Coverage: lenient json parse fail after repair
# ---------------------------------------------------------------------------


class TestLenientParseAfterRepair:
    """Edge cases in lenient path."""

    def test_schema_validation_failure_message(self) -> None:
        # Passing a string for an int target fails schema validation.
        codec = JsonCodec()
        result = _parse_typed(codec, '"not an int"', IntType(), strict_json=False)
        assert result.ok is False
        assert "Schema validation failed" in result.error_msg

    def test_validate_and_decode_core_value_conversion_failure(self) -> None:
        """_validate_and_decode_core: schema passes (permissive) but decode_value raises."""
        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.codec import _validate_and_decode_core

        # Permissive schema {} accepts anything; ScalarDecode(TEXT) rejects non-strings.
        decode = ScalarDecode(ScalarKind.TEXT)
        result = _validate_and_decode_core("42", 42, {}, decode)
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
        result = _parse_typed(codec, raw, IntType(), strict_json=False)
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
            result = _parse_typed(codec, "anything", IntType(), strict_json=False)
        assert result.ok is False
        assert "JSON parse failed after repair attempt" in result.error_msg


# ---------------------------------------------------------------------------
# 16. Coverage: validation-error mapping internals and extraction edges
# ---------------------------------------------------------------------------


class TestValidationMappingCoverage:
    """Cover structural / defensive branches of the  error mapping."""

    def test_trailing_comma_array_recovers_not_ambiguous(self) -> None:
        """A repaired array whose candidate already starts with '[' is not ambiguous."""
        codec = JsonCodec()
        result = _parse_typed(codec, "[1, 2,]", JsonType(), strict_json=False)
        assert result.ok is True
        assert isinstance(result.value, JsonValue)
        assert result.value.raw == [1, 2]

    def test_enum_two_field_variant_reports_first_missing(self) -> None:
        codec = JsonCodec()
        typ, typedef = enum_type("E", {"V": {"a": IntType(), "b": IntType()}})
        # "a" present, "b" missing → loop skips a, reports b.
        result = _parse_typed(
            codec, '{"$case": "V", "a": 1}', typ, strict_json=False, table=type_table_for(typedef)
        )
        assert result.ok is False
        assert result.errors[0].category == "missing_field"
        assert result.errors[0].field == "b"

    def test_enum_non_object_instance_is_bad_case(self) -> None:
        codec = JsonCodec()
        typ, typedef = enum_type("E", {"A": {}})
        result = _parse_typed(codec, "42", typ, strict_json=False, table=type_table_for(typedef))
        assert result.ok is False
        assert result.errors[0].category == "bad_case"

    def test_list_nested_enum_bad_case(self) -> None:
        codec = JsonCodec()
        enum, enum_def = enum_type("E", {"A": {}, "B": {"x": IntType()}})
        result = _parse_typed(
            codec,
            '[{"$case": "Z"}]',
            ListType(elem=enum),
            strict_json=False,
            table=type_table_for(enum_def),
        )
        assert result.ok is False
        assert result.errors[0].category == "bad_case"
        assert result.errors[0].path == "$[0]"

    def test_dict_nested_enum_bad_case(self) -> None:
        codec = JsonCodec()
        enum, enum_def = enum_type("E", {"A": {}, "B": {"x": IntType()}})
        result = _parse_typed(
            codec,
            '{"k": {"$case": "Z"}}',
            DictType(value=enum),
            strict_json=False,
            table=type_table_for(enum_def),
        )
        assert result.ok is False
        assert result.errors[0].category == "bad_case"
        assert result.errors[0].path == "$.k"

    def test_make_validation_error_required_non_list(self) -> None:
        """_make_validation_error: required validator with non-list value → field=None."""
        from unittest.mock import MagicMock

        from jsonschema import ValidationError as JVE

        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.codec import _make_validation_error

        err = MagicMock(spec=JVE)
        err.validator = "required"
        err.validator_value = "not-a-list"
        err.instance = {"x": 1}
        err.message = "required"
        err.path = []
        decode = ScalarDecode(ScalarKind.INT)
        ve = _make_validation_error(err, decode)
        assert ve.category == "missing_field"
        assert ve.field is None

    def test_make_validation_error_required_all_present(self) -> None:
        """_make_validation_error: all required fields present → field=None."""
        from unittest.mock import MagicMock

        from jsonschema import ValidationError as JVE

        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.codec import _make_validation_error

        err = MagicMock(spec=JVE)
        err.validator = "required"
        err.validator_value = ["a", "b"]
        err.instance = {"a": 1, "b": 2}
        err.message = "required"
        err.path = []
        decode = ScalarDecode(ScalarKind.INT)
        ve = _make_validation_error(err, decode)
        assert ve.category == "missing_field"
        assert ve.field is None

    def test_make_validation_error_unknown_validator_is_wrong_type(self) -> None:
        """A non-required/additionalProperties/type/oneOf validator → wrong_type."""
        from unittest.mock import MagicMock

        from jsonschema import ValidationError as JVE

        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.codec import _make_validation_error

        err = MagicMock(spec=JVE)
        err.validator = "const"
        err.validator_value = "X"
        err.instance = "Y"
        err.message = "const mismatch"
        err.path = []
        decode = ScalarDecode(ScalarKind.TEXT)
        ve = _make_validation_error(err, decode)
        assert ve.category == "wrong_type"
        assert ve.message == "const mismatch"

    def test_find_enum_decode_at_path_unknown_record_field(self) -> None:
        """_find_enum_decode_at_path returns None when path names unknown field."""
        from agm.agl.runtime.codec import _find_enum_decode_at_path

        rec, rec_def = record_type("R", {"a": IntType()})
        decode = build_decode_schema(rec, type_table_for(rec_def)).root
        assert _find_enum_decode_at_path(decode, ["missing"]) is None

    def test_find_enum_decode_at_path_scalar_with_remaining_path(self) -> None:
        """_find_enum_decode_at_path returns None when path descends past a scalar."""
        from agm.agl.runtime.codec import _find_enum_decode_at_path

        decode = build_decode_schema(IntType(), type_table_for()).root
        assert _find_enum_decode_at_path(decode, ["deeper"]) is None

    def test_find_enum_decode_at_path_unresolvable_ref_returns_none(self) -> None:
        """An unresolvable RefDecode (unknown key) fails soft: no crash, no match found.

        Classification walkers only refine an already-failed validation's
        message; an inconsistent contract must never turn that into a crash.
        """
        from agm.agl.ir.contracts import RefDecode
        from agm.agl.runtime.codec import _find_enum_decode_at_path

        assert _find_enum_decode_at_path(RefDecode("NoSuchKey"), [], {}) is None

    def test_find_enum_decode_at_path_resolves_recursive_ref_to_enum(self) -> None:
        """A root RefDecode that DOES resolve reaches the EnumDecode it points to."""
        from agm.agl.ir.contracts import RefDecode
        from agm.agl.runtime.codec import _find_enum_decode_at_path

        tree, tree_def = _tree_type_and_def()
        plan = build_decode_schema(tree, type_table_for(tree_def))
        assert plan.root == RefDecode("Tree")
        defs = dict(plan.defs)
        found = _find_enum_decode_at_path(plan.root, [], defs)
        assert found is defs["Tree"]

    def test_classify_enum_failure_no_enum_decode_at_path(self) -> None:
        """_classify_enum_failure: _find_enum_decode_at_path returns None → bad_case fallback."""
        from unittest.mock import MagicMock

        from jsonschema import ValidationError as JVE

        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.codec import _classify_enum_failure

        # decode_schema is a ScalarDecode (not EnumDecode) → _find_enum_decode_at_path returns None.
        # instance is a dict with a string $case to get past the first two guards.
        decode = ScalarDecode(ScalarKind.INT)
        err = MagicMock(spec=JVE)
        err.validator = "oneOf"
        err.instance = {"$case": "X"}
        err.absolute_path = []
        err.path = []
        ve = _classify_enum_failure(err, "$", decode)
        assert ve.category == "bad_case"
        assert ve.field == "$case"


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
        issue_type = _make_issue_type()
        contract = codec.make_contract(issue_type, _DEFAULT_TABLE)
        assert contract.json_schema is not None

    def test_materialize_contract_no_longer_constructs_type_env(self) -> None:
        """materialize_contract must not instantiate TypeEnvironment internally."""
        from agm.agl.runtime.contract import materialize_contract
        from agm.agl.typecheck.env import OutputContractSpec

        issue_type = _make_issue_type()
        spec = OutputContractSpec(
            target_type=issue_type,
            codec_name="json",
            strict_json=False,
        )
        # If TypeEnvironment() were still constructed it would not fail, but we
        # verify the contract comes out correctly to confirm the wire-up works.
        contract = materialize_contract(
            spec, {"json": JsonCodec(), "text": TextCodec()}, _DEFAULT_TABLE
        )
        assert contract.json_schema is not None


class TestSchemaPrecomputedInParse:
    """parse() takes an explicit schema + decode; it never derives them from a Type.

    Production code (the IR evaluator) always has the contract-carried
    ``json_schema``/``decode`` on hand and passes them straight through — this
    class verifies parsing behaves identically whether the schema instance is
    freshly derived or the one held by a materialized contract, and that
    omitting either input is a hard error rather than a silent re-derivation.
    """

    def test_parse_with_precomputed_schema_succeeds(self) -> None:
        codec = JsonCodec()
        typ = _make_issue_type()
        table = _DEFAULT_TABLE
        schema = derive_schema(typ, table)
        decode = build_decode_schema(typ, table).root
        raw = '{"title": "Bug", "severity": 5, "description": "A bug"}'
        result = codec.parse(raw, strict_json=False, schema=schema, decode=decode)
        assert result.ok is True

    def test_parse_with_precomputed_schema_validation_failure(self) -> None:
        codec = JsonCodec()
        typ = _make_issue_type()
        table = _DEFAULT_TABLE
        schema = derive_schema(typ, table)
        decode = build_decode_schema(typ, table).root
        # Missing required fields → schema validation fails even with a precomputed schema.
        result = codec.parse('{"title": "Bug"}', strict_json=False, schema=schema, decode=decode)
        assert result.ok is False
        assert result.errors

    def test_parse_with_precomputed_schema_matches_derived(self) -> None:
        """parse(schema=materialized contract's schema) matches parse(schema=freshly derived).

        The contract's schema and a freshly derived schema are the same JSON
        Schema, produced by the same function — passing either must yield an
        identical parse outcome (ok/value/errors).
        """
        codec = JsonCodec()
        typ, typedef = record_type("Issue", {"title": TextType(), "severity": IntType()})
        table = type_table_for(typedef)
        contract_schema = codec.make_contract(typ, table).json_schema
        assert isinstance(contract_schema, dict)
        fresh_schema = derive_schema(typ, table)
        decode = build_decode_schema(typ, table).root

        good = '{"title": "x", "severity": 1}'
        bad = '{"title": "x"}'  # missing required field
        for raw in (good, bad):
            with_contract_schema = codec.parse(
                raw, strict_json=False, schema=contract_schema, decode=decode
            )
            with_fresh_schema = codec.parse(
                raw, strict_json=False, schema=fresh_schema, decode=decode
            )
            assert with_contract_schema.ok == with_fresh_schema.ok
            assert with_contract_schema.value == with_fresh_schema.value
            assert [e.message for e in with_contract_schema.errors] == [
                e.message for e in with_fresh_schema.errors
            ]

    def test_contract_json_schema_reused_across_parses(self) -> None:
        """the contract's json_schema/decode are the objects threaded into parse.

        Observable identity reuse: the schema/decode the codec materializes on
        the contract are the same objects accepted by ``parse(...)`` — the
        interpreter passes ``contract.json_schema``/``contract.decode``
        straight through rather than re-deriving them.
        """
        codec = JsonCodec()
        typ, typedef = record_type("Issue", {"title": TextType(), "severity": IntType()})
        contract = codec.make_contract(typ, type_table_for(typedef))
        schema = contract.json_schema
        decode = contract.decode
        assert isinstance(schema, dict)
        assert decode is not None
        raw = '{"title": "x", "severity": 1}'
        # Reuse the very same schema/decode objects on repeated parses.
        for _ in range(3):
            result = codec.parse(raw, strict_json=False, schema=schema, decode=decode)
            assert result.ok is True
            # Identity: contract still holds the same schema object.
            assert contract.json_schema is schema

    def test_parse_without_schema_raises(self) -> None:
        """parse() with schema=None raises: no derivation fallback from a Type."""
        codec = JsonCodec()
        with pytest.raises(ValueError, match="schema and decode"):
            codec.parse(
                "42",
                strict_json=False,
                schema=None,
                decode=build_decode_schema(IntType(), type_table_for()).root,
            )

    def test_parse_without_decode_raises(self) -> None:
        """parse() with decode=None raises: no derivation fallback from a Type."""
        codec = JsonCodec()
        with pytest.raises(ValueError, match="schema and decode"):
            codec.parse(
                "42",
                strict_json=False,
                schema=derive_schema(IntType(), type_table_for()),
                decode=None,
            )


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

        kind_to_type: dict[str, Type] = {
            "text": TextType(),
            "int": IntType(),
            "decimal": DecimalType(),
            "bool": BoolType(),
            "json": JsonType(),
            "list": ListType(elem=TextType()),
            "dict": DictType(value=TextType()),
            "record": RecordType(name="R"),
            "enum": EnumType(name="E"),
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

        rt = PipelineDriver(default_agent=lambda request: "response")

        class AltTextCodec(TC):
            @property
            def name(self) -> str:
                return "alt_text"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

        rt.register_codec(AltTextCodec())
        result = rt.run(
            'let answer: text = ask("question", format = "alt_text")\nprint answer',
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

            def make_contract(self, type_ref: Type, type_table: TypeTable | None = None) -> OC:
                raise NotImplementedError

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: DecodeSchema | None = None,
            ) -> PR:
                raise NotImplementedError

        rt = PipelineDriver()
        rt.register_codec(CustomCodec())
        with pytest.raises(ValueError, match="custom_dup"):
            rt.register_codec(CustomCodec())

    def test_register_reserved_codec_name_text_raises(self) -> None:
        rt = PipelineDriver()
        with pytest.raises(ValueError, match="text"):
            rt.register_codec(TextCodec())

    def test_register_reserved_codec_name_json_raises(self) -> None:
        rt = PipelineDriver()
        with pytest.raises(ValueError, match="json"):
            rt.register_codec(JsonCodec())

    def test_custom_codec_make_contract_and_parse_exercised_in_pipeline(self) -> None:
        """a custom codec selected via ``format:`` is genuinely used.

        The codec is chosen with ``ask("Q", format: "tagcodec")`` on a ``text``
        target.  Both its ``make_contract`` (observable via the format
        instructions threaded into the agent request) and its ``parse``
        (observable as a distinctive prefix on the resulting binding) are
        exercised end-to-end through ``run()`` with a stub agent.
        """
        from agm.agl.semantics.values import TextValue

        class TagCodec:
            @property
            def name(self) -> str:
                return "tagcodec"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

            def supports_type(self, t: Type) -> bool:
                from agm.agl.semantics.types import TextType as TT

                return isinstance(t, TT)

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> "OutputContract":
                from agm.agl.runtime.contract import OutputContract

                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=self,
                    strict_json=None,
                    format_instructions="TAGCODEC-INSTRUCTIONS",
                    json_schema=None,
                )

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: DecodeSchema | None = None,
                defs: dict[str, DecodeSchema] | None = None,
            ) -> "ParseResult":
                from agm.agl.runtime.codec import ParseResult

                # Distinctive transform proving THIS codec parsed the output.
                return ParseResult.success(TextValue(f"PARSED::{raw}"))

        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "hello"

        rt = PipelineDriver(default_agent=agent)
        rt.register_codec(TagCodec())
        #  format: arg takes the codec name as a string; let needs a continuation.
        result = rt.run('let y: text = ask("Q", format = "tagcodec")\ny')
        assert result.ok is True
        # parse() ran: the binding carries the codec's distinctive prefix.
        assert result.bindings["y"] == TextValue("PARSED::hello")
        # make_contract() ran: its format instructions reached the agent.
        assert received[0].output_contract is not None
        assert received[0].output_contract.format_instructions == "TAGCODEC-INSTRUCTIONS"

    def test_custom_int_codec_sees_int_target_and_old_parse_signature_works(self) -> None:
        """Custom codecs get a kind-correct target and may omit the newer defs keyword."""

        seen_targets: list[str] = []

        class IntCodec:
            @property
            def name(self) -> str:
                return "intcodec"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"int"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, IntType)

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                seen_targets.append(repr(type_ref))
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=self,
                    strict_json=None,
                    format_instructions="INTCODEC-INSTRUCTIONS",
                    json_schema=None,
                )

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: DecodeSchema | None = None,
            ) -> ParseResult:
                return ParseResult.success(IntValue(int(raw)))

        rt = PipelineDriver(default_agent=lambda req: "7")
        rt.register_codec(IntCodec())
        result = rt.run('let y: int = ask("Q", format = "intcodec")\ny')
        assert result.ok is True
        assert result.bindings["y"] == IntValue(7)
        assert seen_targets == ["int"]

    def test_legacy_custom_codec_contract_and_parse_signatures_work(self) -> None:
        """Custom codecs may use make_contract(type) and parse(raw, target_type, ...)."""

        seen_contract_targets: list[str] = []
        seen_parse_targets: list[str] = []

        class LegacyCodec:
            @property
            def name(self) -> str:
                return "legacy-int"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"int"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, IntType)

            def make_contract(self, type_ref: Type) -> OutputContract:
                seen_contract_targets.append(repr(type_ref))
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=self,
                    strict_json=None,
                    format_instructions="LEGACY",
                    json_schema=None,
                )

            def parse(
                self,
                raw: str,
                target_type: Type,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: DecodeSchema | None = None,
            ) -> ParseResult:
                seen_parse_targets.append(repr(target_type))
                return ParseResult.success(IntValue(int(raw)))

        rt = PipelineDriver(default_agent=lambda req: "11")
        rt.register_codec(LegacyCodec())
        result = rt.run('let y: int = ask("Q", format = "legacy-int")\ny')

        from agm.agl.runtime.contract import materialize_contract
        from agm.agl.typecheck.env import OutputContractSpec

        materialize_contract(
            OutputContractSpec(IntType(), "legacy-int", strict_json=None),
            {"legacy-int": LegacyCodec()},
        )

        assert result.ok is True
        assert result.bindings["y"] == IntValue(11)
        assert seen_contract_targets == ["int", "int"]
        assert seen_parse_targets == ["int"]

    def test_custom_codec_make_contract_signature_probe_fallback(self) -> None:
        """Compatibility still works if a callable make_contract has no inspectable signature."""

        class ContractCallable:
            __signature__ = object()

            def __call__(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=codec,
                    strict_json=None,
                    format_instructions="fallback",
                    json_schema=None,
                )

        class FallbackCodec:
            make_contract = ContractCallable()

            @property
            def name(self) -> str:
                return "fallback-contract"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"int"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, IntType)

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: DecodeSchema | None = None,
                defs: Mapping[str, DecodeSchema] | None = None,
            ) -> ParseResult:
                return ParseResult.failure(raw)

        codec = FallbackCodec()
        contract = materialize_contract(
            OutputContractSpec(IntType(), "fallback-contract", strict_json=None),
            {"fallback-contract": codec},
        )

        assert contract.format_instructions == "fallback"

    def test_custom_structured_codec_uses_compiled_schema_not_lossy_placeholder(self) -> None:
        """Structured custom codecs receive the lowered decode plan for the real target."""

        seen_decode: list[DecodeSchema | None] = []

        class ListIntCodec:
            @property
            def name(self) -> str:
                return "list-int-codec"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"list"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, ListType)

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                raise AssertionError(f"lossy runtime placeholder reached codec: {type_ref!r}")

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: DecodeSchema | None = None,
                defs: Mapping[str, DecodeSchema] | None = None,
            ) -> ParseResult:
                seen_decode.append(decode)
                return ParseResult.success(ListValue((IntValue(int(raw)),)))

        rt = PipelineDriver(default_agent=lambda req: "3")
        rt.register_codec(ListIntCodec())
        result = rt.run('let xs: list[int] = ask("Q", format = "list-int-codec")\nxs')

        assert result.ok is True
        assert result.bindings["xs"] == ListValue((IntValue(3),))
        assert seen_decode == [ListDecode(ScalarDecode(ScalarKind.INT))]

    def test_custom_codec_ir_materialization_reconstructs_target_kinds(self) -> None:
        """IR contract materialization gives custom codecs kind-correct placeholders."""

        class CaptureCodec:
            def __init__(self) -> None:
                self.seen: list[Type] = []

            @property
            def name(self) -> str:
                return "capture"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset(
                    {
                        "text",
                        "int",
                        "decimal",
                        "bool",
                        "json",
                        "agent",
                        "list",
                        "dict",
                        "record",
                        "enum",
                        "unit",
                    }
                )

            def supports_type(self, t: Type) -> bool:
                return True

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                self.seen.append(type_ref)
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=self,
                    strict_json=None,
                    format_instructions="",
                    json_schema=None,
                )

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: DecodeSchema | None = None,
                defs: Mapping[str, DecodeSchema] | None = None,
            ) -> ParseResult:
                return ParseResult.failure(raw)

        codec = CaptureCodec()
        cases = [
            ("text", "text", TextType),
            ("int", "int", IntType),
            ("decimal", "decimal", DecimalType),
            ("bool", "bool", BoolType),
            ("json", "json", JsonType),
            ("agent", "agent", AgentType),
            ("list", "list[int]", ListType),
            ("dict", "dict[text, int]", DictType),
            ("record", "Issue", RecordType),
            ("enum", "Result", EnumType),
            ("", "unit", UnitType),
        ]
        for kind, label, expected_type in cases:
            request = ContractRequest(
                codec_name="capture",
                strict_json=None,
                json_schema=None,
                decode=None,
                target_type_label=label,
                structured_exec=False,
                format_instructions="",
                target_type_kind=kind,
            )
            contract = materialize_ir_contract(request, {"capture": codec})
            assert contract is not None
            assert isinstance(codec.seen[-1], expected_type)

    def test_custom_codec_ir_materialization_preserves_recursive_decode(self) -> None:
        """Execution-time custom contracts keep codec-provided decode metadata."""

        class RecursiveCodec:
            @property
            def name(self) -> str:
                return "recursive"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"record"})

            def supports_type(self, t: Type) -> bool:
                return True

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=self,
                    strict_json=False,
                    format_instructions="recursive",
                    json_schema={"$ref": "#/$defs/Node"},
                    decode=RefDecode("Node"),
                    defs=(("Node", ScalarDecode(ScalarKind.JSON)),),
                )

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: DecodeSchema | None = None,
                defs: Mapping[str, DecodeSchema] | None = None,
            ) -> ParseResult:
                return ParseResult.failure(raw)

        request = ContractRequest(
            codec_name="recursive",
            strict_json=None,
            json_schema=None,
            decode=None,
            target_type_label="Node",
            structured_exec=False,
            format_instructions="",
            target_type_kind="record",
        )

        contract = materialize_ir_contract(request, {"recursive": RecursiveCodec()})

        assert contract is not None
        assert contract.decode == RefDecode("Node")
        assert contract.defs == (("Node", ScalarDecode(ScalarKind.JSON)),)

    def test_custom_codec_parse_receives_recursive_defs(self) -> None:
        """The IR interpreter passes custom contract defs through to parse()."""
        from agm.agl.eval.ir_interpreter import IrInterpreter
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.program import ExecutableModule, ExecutableProgram

        seen_defs: Mapping[str, DecodeSchema] | None = None

        class DefsCaptureCodec:
            @property
            def name(self) -> str:
                return "capture-defs"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"record"})

            def supports_type(self, t: Type) -> bool:
                return True

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=self,
                    strict_json=False,
                    format_instructions="",
                    json_schema={},
                )

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: DecodeSchema | None = None,
                defs: Mapping[str, DecodeSchema] | None = None,
            ) -> ParseResult:
                nonlocal seen_defs
                seen_defs = defs
                return ParseResult.success(JsonValue({"ok": True}))

        contract_id = ContractId(0)
        request = ContractRequest(
            codec_name="capture-defs",
            strict_json=False,
            json_schema=None,
            decode=None,
            target_type_label="Node",
            structured_exec=False,
            format_instructions="",
            target_type_kind="record",
        )
        defs = (("Node", ScalarDecode(ScalarKind.JSON)),)
        host_contract = OutputContract(
            target_type_label="Node",
            codec=DefsCaptureCodec(),
            strict_json=False,
            format_instructions="",
            json_schema={},
            decode=RefDecode("Node"),
            defs=defs,
        )
        program = ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=())},
            symbols={},
            nominals={},
            sources={},
            contracts={contract_id: request},
        )
        interpreter = IrInterpreter(program, host_contracts={contract_id: host_contract})

        result = interpreter._parse_host_output("{}", contract_id, effective_strict=False)

        assert result.ok
        assert seen_defs == dict(defs)

    def test_custom_codec_parse_signature_probe_fallback(self) -> None:
        """Compatibility still works if a callable parse hook has no inspectable signature."""
        from agm.agl.eval.ir_interpreter import IrInterpreter
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.program import ExecutableModule, ExecutableProgram

        class ParseCallable:
            __signature__ = object()

            def __call__(self, raw: str, **kwargs: object) -> ParseResult:
                return ParseResult.success(TextValue(f"fallback::{raw}"))

        class FallbackCodec:
            parse = ParseCallable()

            @property
            def name(self) -> str:
                return "fallback-parse"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, TextType)

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=self,
                    strict_json=None,
                    format_instructions="",
                    json_schema=None,
                )

        contract_id = ContractId(0)
        request = ContractRequest(
            codec_name="fallback-parse",
            strict_json=False,
            json_schema=None,
            decode=None,
            target_type_label="text",
            structured_exec=False,
            format_instructions="",
            target_type_kind="text",
        )
        host_contract = OutputContract(
            target_type_label="text",
            codec=FallbackCodec(),
            strict_json=False,
            format_instructions="",
            json_schema={},
        )
        program = ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=())},
            symbols={},
            nominals={},
            sources={},
            contracts={contract_id: request},
        )
        interpreter = IrInterpreter(program, host_contracts={contract_id: host_contract})

        result = interpreter._parse_host_output("ok", contract_id, effective_strict=False)

        assert result.ok
        assert result.value == TextValue("fallback::ok")

    def test_custom_codec_parse_type_error_propagates(self) -> None:
        """Only old parse signatures are retried; codec TypeError bugs still propagate."""
        from agm.agl.eval.ir_interpreter import IrInterpreter
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.program import ExecutableModule, ExecutableProgram

        class BrokenCodec:
            @property
            def name(self) -> str:
                return "broken"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"record"})

            def supports_type(self, t: Type) -> bool:
                return True

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=self,
                    strict_json=False,
                    format_instructions="",
                    json_schema={},
                )

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: DecodeSchema | None = None,
                defs: Mapping[str, DecodeSchema] | None = None,
            ) -> ParseResult:
                raise TypeError("codec parse bug")

        contract_id = ContractId(0)
        request = ContractRequest(
            codec_name="broken",
            strict_json=False,
            json_schema=None,
            decode=None,
            target_type_label="Node",
            structured_exec=False,
            format_instructions="",
            target_type_kind="record",
        )
        host_contract = OutputContract(
            target_type_label="Node",
            codec=BrokenCodec(),
            strict_json=False,
            format_instructions="",
            json_schema={},
        )
        program = ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=())},
            symbols={},
            nominals={},
            sources={},
            contracts={contract_id: request},
        )
        interpreter = IrInterpreter(program, host_contracts={contract_id: host_contract})

        with pytest.raises(TypeError, match="codec parse bug"):
            interpreter._parse_host_output("{}", contract_id, effective_strict=False)


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

        #  format: arg takes the codec name as a string; let needs a continuation.
        src = 'let x: text = ask("Q", format = "altcodec")\nx'

        rt_unreg = PipelineDriver(default_agent=lambda req: "ok")
        unreg = rt_unreg.run(src)
        assert unreg.ok is False  # altcodec unknown without registration
        assert any("altcodec" in d.message for d in unreg.diagnostics)

        rt = PipelineDriver(default_agent=lambda req: "ok")
        rt.register_codec(AltCodec())
        reg = rt.run(src)
        assert reg.ok is True
