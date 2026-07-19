"""Tests for the AgL lowering phase.

Covers:
- ``compile_coercion`` — every branch of the coercion compiler.
- ``lower_module`` — lowering of supported nodes, including coercion
  insertion, binding/assignment lowering, and validate_ir pass.

Pipeline helper: reuses the ``parse_resolve_check`` pattern from
``tests/test_agl_typecheck.py`` to obtain a ``CheckedModule`` from source.
"""

from __future__ import annotations

import decimal
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from inspect import Parameter, signature
from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.contracts import ConversionFailureMode, ConversionStrategy
from agm.agl.ir.nodes import (
    AutoTraceField,
    IrArith,
    IrAssign,
    IrBind,
    IrBlock,
    IrBreak,
    IrCapture,
    IrCase,
    IrCoerce,
    IrCompare,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstText,
    IrConstUnit,
    IrContains,
    IrConvert,
    IrDirectCall,
    IrEnumCaseKey,
    IrIf,
    IrIndirectCall,
    IrIterHasNext,
    IrIterInit,
    IrIterNext,
    IrLoad,
    IrLoop,
    IrMakeClosure,
    IrMakeDict,
    IrMakeException,
    IrMakeList,
    IrMakeRecord,
    IrRaise,
    IrRenderTemplate,
    IrSequence,
    IrTemplateText,
    IrTemplateValue,
    IrTry,
    IrUnary,
    UseDefault,
)
from agm.agl.ir.operations import (
    ArithKind,
    ArithOp,
    CmpOp,
    CompareKind,
    ContainsKind,
    IntToDecimal,
    IterKind,
    MapDictValues,
    MapEnumFields,
    MapList,
    MapRecordFields,
    ToJson,
    UnaryOp,
)
from agm.agl.ir.program import ExecutableProgram, FunctionDescriptor, IrFunctionBody
from agm.agl.ir.validate import validate_ir
from agm.agl.lower import LinkImage, compile_coercion, lower_module, lower_repl_entry
from agm.agl.lower.lowerer import _Lowerer
from agm.agl.matchcompile import MatchCompiledModule, compile_module_matches
from agm.agl.parser import parse_program, parse_program_seeded
from agm.agl.scope import resolve_module
from agm.agl.semantics.types import (
    BoolType,
    DecimalType,
    DictType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    TypeVarType,
    UnitType,
)
from agm.agl.syntax.nodes import Case, Placeholder
from agm.agl.typecheck import check_module
from agm.agl.typecheck.env import CheckedModule
from tests._agl_helpers import enum_type, record_type, type_table_for
from tests.agl.ir_harness import _compiled_checked

_REPO_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"

# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------


def _caps() -> HostCapabilities:
    return HostCapabilities(
        agent_names=frozenset(),
        has_default_agent=True,
        supports_shell_exec=True,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
        },
    )


def _check(source: str) -> CheckedModule:
    prog = parse_program(source)
    resolved = resolve_module(prog)
    return check_module(resolved, _caps())


def test_lower_repl_entry_accumulates_tables_and_resolves_prior_symbols() -> None:
    image = LinkImage()
    first_program, next_id = parse_program_seeded("let x = 41\n()", start_id=0)
    first_checked = check_module(resolve_module(first_program), _caps())

    first = lower_repl_entry(
        _compiled_checked(first_checked),
        image=image,
        source_text="let x = 41\n()",
        source_label="<repl:1>",
    )
    first_symbols = set(first.program.symbols)
    first_sources = set(first.program.sources)

    second_source = "let y = x + 1\ny"
    second_program, _ = parse_program_seeded(second_source, start_id=next_id)
    second_checked = check_module(
        resolve_module(second_program, parent_scope=first_checked.resolved.root_scope),
        _caps(),
        seed_env=first_checked.type_env,
    )
    second = lower_repl_entry(
        _compiled_checked(second_checked),
        image=image,
        source_text=second_source,
        source_label="<repl:2>",
    )

    assert first_symbols < set(second.program.symbols)
    assert first_sources < set(second.program.sources)
    assert second.trailing_expression is not None
    assert second.program.modules[second.program.entry_module].initializers


def _lower(source: str) -> ExecutableProgram:
    """Parse → check → lower the source; return ExecutableProgram."""
    checked = _check(source)
    return lower_module(
        _compiled_checked(checked),
        source_text=source,
        source_label="<test>",
    )


def _contains_flexible_inference_state(value: object, seen: set[int]) -> bool:
    """Recursively inspect IR data without treating declared rigid variables as leaks."""
    from agm.agl.semantics.types import InferenceVarType

    if isinstance(value, InferenceVarType):
        return True
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return False
    value_id = id(value)
    if value_id in seen:
        return False
    seen.add(value_id)
    if isinstance(value, Mapping):
        return any(
            _contains_flexible_inference_state(key, seen)
            or _contains_flexible_inference_state(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, tuple | list | frozenset):
        return any(_contains_flexible_inference_state(item, seen) for item in value)
    if is_dataclass(value):
        return any(
            _contains_flexible_inference_state(getattr(value, field.name), seen)
            for field in fields(value)
        )
    return False


def _function_body(desc: FunctionDescriptor) -> object:
    assert isinstance(desc.impl, IrFunctionBody)
    return desc.impl.body


def _make_lowerer(checked: CheckedModule, source: str) -> "_Lowerer":
    """Match-compile ``checked`` and create a lowerer with fresh link state."""
    from agm.agl.ir.ids import SourceId
    from agm.agl.ir.program import SourceFile
    from agm.agl.lower.lowerer import _LinkState, _Lowerer
    from agm.agl.modules.ids import ENTRY_ID
    from agm.util.text import normalize_newlines

    link = _LinkState()
    source_id = SourceId(link.next_source)
    link.next_source += 1
    normalized = normalize_newlines(source)
    link.sources[source_id] = SourceFile(display_name="<test>", normalized_text=normalized)
    match_result = compile_module_matches(checked)
    assert isinstance(match_result.compiled, MatchCompiledModule)
    compiled = match_result.compiled
    return _Lowerer(compiled.checked, link, ENTRY_ID, source_id, source, compiled.cases)


def test_direct_lowerer_helper_requires_successful_match_compilation() -> None:
    source = "case true of | true => 1"

    with pytest.raises(AssertionError):
        _make_lowerer(_check(source), source)


def test_private_lowerer_requires_complete_case_mapping_argument() -> None:
    cases = signature(_Lowerer).parameters["cases"]

    assert cases.default is Parameter.empty


def test_direct_lowerer_helper_passes_complete_compiled_case_mapping() -> None:
    source = "case true of | true => 1 | false => 2"
    checked = _check(source)
    lowerer = _make_lowerer(checked, source)
    case = checked.resolved.program.body.items[0]

    assert isinstance(case, Case)
    assert set(lowerer._compiled_cases) == {case.node_id}


def test_constructor_result_nominal_rejects_non_nominal_type() -> None:
    source = "1"
    lowerer = _make_lowerer(_check(source), source)

    with pytest.raises(AssertionError, match="non-nominal result"):
        lowerer._nominal_for_constructor_result(IntType())


def test_lowering_erases_flexible_state_from_generic_direct_nested_and_partial_calls() -> None:
    executable = _lower(
        "record Box[T]\n"
        "  value: T\n"
        "def id[T](value: T) -> T = value\n"
        "def app[T](func: T -> T, value: T) -> T = func(value)\n"
        "let direct = id(1)\n"
        "let nested = app(id, app(id, 2))\n"
        "let make_box: (int) -> Box[int] = Box(value = ?)\n"
        "let boxed = make_box(nested)\n"
        "boxed"
    )

    assert not _contains_flexible_inference_state(executable, set())


# ---------------------------------------------------------------------------
# compile_coercion unit tests — every branch
# ---------------------------------------------------------------------------


class TestCompileCoercion:
    # Identity / None cases

    def test_same_int_type_is_none(self) -> None:
        assert compile_coercion(IntType(), IntType(), type_table_for()) is None

    def test_same_text_type_is_none(self) -> None:
        assert compile_coercion(TextType(), TextType(), type_table_for()) is None

    def test_same_bool_type_is_none(self) -> None:
        assert compile_coercion(BoolType(), BoolType(), type_table_for()) is None

    def test_same_decimal_type_is_none(self) -> None:
        assert compile_coercion(DecimalType(), DecimalType(), type_table_for()) is None

    def test_same_unit_type_is_none(self) -> None:
        assert compile_coercion(UnitType(), UnitType(), type_table_for()) is None

    def test_same_json_type_is_none(self) -> None:
        # json → json: identity, no coercion
        assert compile_coercion(JsonType(), JsonType(), type_table_for()) is None

    def test_type_var_source_is_none(self) -> None:
        # TypeVarType source → opaque, no coercion
        assert compile_coercion(TypeVarType("T"), IntType(), type_table_for()) is None

    def test_type_var_target_is_none(self) -> None:
        assert compile_coercion(IntType(), TypeVarType("T"), type_table_for()) is None

    # Scalar coercions

    def test_int_to_decimal(self) -> None:
        result = compile_coercion(IntType(), DecimalType(), type_table_for())
        assert result == IntToDecimal()

    def test_int_to_json(self) -> None:
        # Rule 1: target is JsonType and source is not JsonType → ToJson
        result = compile_coercion(IntType(), JsonType(), type_table_for())
        assert result == ToJson()

    def test_text_to_json(self) -> None:
        result = compile_coercion(TextType(), JsonType(), type_table_for())
        assert result == ToJson()

    def test_bool_to_json(self) -> None:
        result = compile_coercion(BoolType(), JsonType(), type_table_for())
        assert result == ToJson()

    def test_decimal_to_json(self) -> None:
        result = compile_coercion(DecimalType(), JsonType(), type_table_for())
        assert result == ToJson()

    def test_list_int_to_json(self) -> None:
        result = compile_coercion(ListType(IntType()), JsonType(), type_table_for())
        assert result == ToJson()

    # List coercions

    def test_list_int_to_list_decimal(self) -> None:
        result = compile_coercion(ListType(IntType()), ListType(DecimalType()), type_table_for())
        assert result == MapList(IntToDecimal())

    def test_list_int_to_list_int_is_none(self) -> None:
        # Element coercion is None → outer is None
        result = compile_coercion(ListType(IntType()), ListType(IntType()), type_table_for())
        assert result is None

    def test_list_int_to_list_json(self) -> None:
        result = compile_coercion(ListType(IntType()), ListType(JsonType()), type_table_for())
        assert result == MapList(ToJson())

    def test_nested_list_int_to_list_list_decimal(self) -> None:
        result = compile_coercion(
            ListType(ListType(IntType())),
            ListType(ListType(DecimalType())),
            type_table_for(),
        )
        assert result == MapList(MapList(IntToDecimal()))

    # Dict coercions

    def test_dict_int_to_dict_decimal(self) -> None:
        result = compile_coercion(DictType(IntType()), DictType(DecimalType()), type_table_for())
        assert result == MapDictValues(IntToDecimal())

    def test_dict_int_to_dict_int_is_none(self) -> None:
        result = compile_coercion(DictType(IntType()), DictType(IntType()), type_table_for())
        assert result is None

    def test_dict_int_to_dict_json(self) -> None:
        result = compile_coercion(DictType(IntType()), DictType(JsonType()), type_table_for())
        assert result == MapDictValues(ToJson())

    # Record coercions
    #
    # Source/target pairs that need genuinely different field shapes under the
    # same display name are constructed with distinct module_ids: a TypeTable
    # entry is keyed by (module_id, name), so two independent declarations
    # named "R"/"E" in different modules resolve to independent shapes, while
    # compile_coercion's record/enum branches don't care whether module_ids
    # match (they only dispatch on isinstance).

    def test_record_no_field_needs_coercion_is_none(self) -> None:
        rec, rec_def = record_type("R", {"x": IntType()})
        result = compile_coercion(rec, rec, type_table_for(rec_def))
        assert result is None

    def test_record_one_field_needs_coercion(self) -> None:
        from agm.agl.modules.ids import ModuleId

        src, src_def = record_type(
            "R", {"x": IntType(), "y": TextType()}, module_id=ModuleId.from_dotted("src_mod")
        )
        tgt, tgt_def = record_type(
            "R", {"x": DecimalType(), "y": TextType()}, module_id=ModuleId.from_dotted("tgt_mod")
        )
        result = compile_coercion(src, tgt, type_table_for(src_def, tgt_def))
        assert result == MapRecordFields((("x", IntToDecimal()),))

    def test_record_multiple_fields_need_coercion(self) -> None:
        from agm.agl.modules.ids import ModuleId

        src, src_def = record_type(
            "R", {"x": IntType(), "y": IntType()}, module_id=ModuleId.from_dotted("src_mod")
        )
        tgt, tgt_def = record_type(
            "R",
            {"x": DecimalType(), "y": DecimalType()},
            module_id=ModuleId.from_dotted("tgt_mod"),
        )
        result = compile_coercion(src, tgt, type_table_for(src_def, tgt_def))
        assert result == MapRecordFields((("x", IntToDecimal()), ("y", IntToDecimal())))

    def test_record_target_field_not_in_source_skipped(self) -> None:
        # Only shared fields are coerced; fields not in source are ignored
        from agm.agl.modules.ids import ModuleId

        src, src_def = record_type("R", {"x": IntType()}, module_id=ModuleId.from_dotted("src_mod"))
        tgt, tgt_def = record_type(
            "R",
            {"x": DecimalType(), "z": TextType()},
            module_id=ModuleId.from_dotted("tgt_mod"),
        )
        result = compile_coercion(src, tgt, type_table_for(src_def, tgt_def))
        assert result == MapRecordFields((("x", IntToDecimal()),))

    # Enum coercions

    def test_enum_no_field_coercion_needed_is_none(self) -> None:
        e, e_def = enum_type("E", {"A": {"x": IntType()}, "B": {}})
        result = compile_coercion(e, e, type_table_for(e_def))
        assert result is None

    def test_enum_one_variant_field_needs_coercion(self) -> None:
        from agm.agl.modules.ids import ModuleId

        src, src_def = enum_type(
            "E", {"A": {"x": IntType()}, "B": {}}, module_id=ModuleId.from_dotted("src_mod")
        )
        tgt, tgt_def = enum_type(
            "E", {"A": {"x": DecimalType()}, "B": {}}, module_id=ModuleId.from_dotted("tgt_mod")
        )
        result = compile_coercion(src, tgt, type_table_for(src_def, tgt_def))
        assert result == MapEnumFields((("A", (("x", IntToDecimal()),)),))

    def test_enum_empty_result_variant_excluded(self) -> None:
        # Variant B has no fields needing coercion → only A in result
        from agm.agl.modules.ids import ModuleId

        src, src_def = enum_type(
            "E",
            {"A": {"x": IntType()}, "B": {"y": TextType()}},
            module_id=ModuleId.from_dotted("src_mod"),
        )
        tgt, tgt_def = enum_type(
            "E",
            {"A": {"x": DecimalType()}, "B": {"y": TextType()}},
            module_id=ModuleId.from_dotted("tgt_mod"),
        )
        result = compile_coercion(src, tgt, type_table_for(src_def, tgt_def))
        assert result == MapEnumFields((("A", (("x", IntToDecimal()),)),))

    def test_enum_target_field_not_in_source_skipped(self) -> None:
        # Target variant A has field "extra" not in source → only "x" can be coerced
        from agm.agl.modules.ids import ModuleId

        src, src_def = enum_type(
            "E", {"A": {"x": IntType()}}, module_id=ModuleId.from_dotted("src_mod")
        )
        tgt, tgt_def = enum_type(
            "E",
            {"A": {"x": DecimalType(), "extra": TextType()}},
            module_id=ModuleId.from_dotted("tgt_mod"),
        )
        result = compile_coercion(src, tgt, type_table_for(src_def, tgt_def))
        # "extra" is not in source so it's skipped; only "x" coercion emitted
        assert result == MapEnumFields((("A", (("x", IntToDecimal()),)),))

    def test_enum_source_variant_not_in_target_skipped(self) -> None:
        # Source has variant B that target doesn't; only target variants are processed
        from agm.agl.modules.ids import ModuleId

        src, src_def = enum_type(
            "E",
            {"A": {"x": IntType()}, "B": {"y": IntType()}},
            module_id=ModuleId.from_dotted("src_mod"),
        )
        tgt, tgt_def = enum_type(
            "E", {"A": {"x": DecimalType()}}, module_id=ModuleId.from_dotted("tgt_mod")
        )
        result = compile_coercion(src, tgt, type_table_for(src_def, tgt_def))
        assert result == MapEnumFields((("A", (("x", IntToDecimal()),)),))

    def test_generic_instantiations_sharing_one_declaration(self) -> None:
        # The real production scenario: a single generic record declaration
        # (one (module_id, name) key) instantiated twice with different
        # type_args.  Box[int] and Box[decimal] are two handles that share
        # the SAME module_id/name — unlike the other record/enum cases above,
        # which model two independent declarations via distinct module_ids —
        # so their field shapes must come from substituting each handle's own
        # type_args into the one registered TypeDef template, not from two
        # separate table entries.
        from agm.agl.modules.ids import ModuleId
        from agm.agl.semantics.type_table import TypeDef, create_seeded_type_table

        module_id = ModuleId.from_dotted("generics_mod")
        table = create_seeded_type_table()
        table.register(
            TypeDef(
                kind="record",
                name="Box",
                module_id=module_id,
                type_params=("T",),
                fields=(("value", TypeVarType("T")),),
            )
        )
        box_int = RecordType("Box", type_args=(IntType(),), module_id=module_id)
        box_decimal = RecordType("Box", type_args=(DecimalType(),), module_id=module_id)
        result = compile_coercion(box_int, box_decimal, table)
        assert result == MapRecordFields((("value", IntToDecimal()),))

    def test_unequal_instantiations_of_recursive_declaration_raise_instead_of_looping(
        self,
    ) -> None:
        # Box[T] here ALSO has a `list[Box[T]]` field, unlike the
        # non-recursive Box[T] above: naively coercing Box[int] -> Box[decimal]
        # would recurse into that same unequal pair forever (list[Box[int]] ->
        # list[Box[decimal]] -> Box[int] -> Box[decimal] -> ...), a genuine
        # Python RecursionError. This pair is never actually produced by the
        # checker (nominal types are invariant — see
        # docs/agl/reference/generics.md), so the internal cycle guard is the
        # right response: fail loudly and immediately with a clear internal
        # diagnostic rather than hang or silently mis-coerce.
        from agm.agl.modules.ids import ModuleId
        from agm.agl.semantics.type_table import TypeDef, create_seeded_type_table

        module_id = ModuleId.from_dotted("recursive_generics_mod")
        table = create_seeded_type_table()
        table.register(
            TypeDef(
                kind="record",
                name="Box",
                module_id=module_id,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "children",
                        ListType(
                            RecordType("Box", type_args=(TypeVarType("T"),), module_id=module_id)
                        ),
                    ),
                ),
            )
        )
        box_int = RecordType("Box", type_args=(IntType(),), module_id=module_id)
        box_decimal = RecordType("Box", type_args=(DecimalType(),), module_id=module_id)
        with pytest.raises(AssertionError, match="compile_coercion re-entered"):
            compile_coercion(box_int, box_decimal, table)

    # Fallthrough — otherwise → None

    def test_int_to_text_is_none(self) -> None:
        # No implicit int→text coercion; the checker would reject this
        assert compile_coercion(IntType(), TextType(), type_table_for()) is None

    def test_text_to_bool_is_none(self) -> None:
        assert compile_coercion(TextType(), BoolType(), type_table_for()) is None


# ---------------------------------------------------------------------------
# lower_module: basic sanity — validate_ir passes
# ---------------------------------------------------------------------------


class TestLowerProgramValidateIr:
    def test_unit_program_validates(self) -> None:
        # Lower a unit-bodied program "()" and confirm validate_ir accepts it
        # without error.  A program ending in a let/var is a static error, but
        # "()" is a valid expression that produces a well-formed IR module.
        prog = _lower("()")
        validate_ir(prog)

    def test_validate_passes_for_all_literal_types(self) -> None:
        source = "1"
        prog = _lower(source)
        validate_ir(prog)

    def test_type_apply_lowers_to_underlying_function_value(self) -> None:
        prog = _lower("def id[T](x: T) -> T = x\nid::[int]")
        inits = prog.modules[prog.entry_module].initializers
        assert isinstance(inits[-1], IrLoad)

    def test_type_apply_in_lambda_body_lowers_and_scans_captures(self) -> None:
        source = "def id[T](x: T) -> T = x\nlet h = fn(x: int) -> (int) -> int => id::[int]\nh"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        assert any(
            isinstance(init, IrBind) and isinstance(init.value, IrMakeClosure) for init in inits
        )


# ---------------------------------------------------------------------------
# lower_module: literal lowering
# ---------------------------------------------------------------------------


class TestLiteralLowering:
    def test_int_literal(self) -> None:
        prog = _lower("42")
        inits = prog.modules[prog.entry_module].initializers
        assert len(inits) == 1
        node = inits[0]
        assert isinstance(node, IrConstInt)
        assert node.value == 42

    def test_decimal_literal(self) -> None:
        prog = _lower("3.14")
        inits = prog.modules[prog.entry_module].initializers
        node = inits[0]
        assert isinstance(node, IrConstDecimal)
        assert node.value == decimal.Decimal("3.14")

    def test_bool_true_literal(self) -> None:
        prog = _lower("true")
        node = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(node, IrConstBool)
        assert node.value is True

    def test_bool_false_literal(self) -> None:
        prog = _lower("false")
        node = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(node, IrConstBool)
        assert node.value is False

    def test_string_literal(self) -> None:
        prog = _lower('"hello"')
        node = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(node, IrConstText)
        assert node.value == "hello"

    def test_null_literal(self) -> None:
        prog = _lower("null")
        node = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(node, IrConstJsonNull)

    def test_unit_literal(self) -> None:
        prog = _lower("()")
        node = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(node, IrConstUnit)

    def test_location_has_correct_source_id(self) -> None:
        prog = _lower("42")
        node = prog.modules[prog.entry_module].initializers[0]
        # There should be exactly one source registered
        assert len(prog.sources) == 1
        (src_id,) = prog.sources
        assert node.location.source_id == src_id


# ---------------------------------------------------------------------------
# lower_module: list literal lowering
# ---------------------------------------------------------------------------


class TestListLitLowering:
    def test_empty_list(self) -> None:
        # list[int] with no elements
        prog = _lower("let _x: list[int] = []\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        make_list = bind.value
        assert isinstance(make_list, IrMakeList)
        assert make_list.items == ()

    def test_list_int_elements_no_coercion(self) -> None:
        prog = _lower("let _x: list[int] = [1, 2, 3]\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        make_list = bind.value
        assert isinstance(make_list, IrMakeList)
        # Elements are plain IrConstInt (no coercion needed)
        for item in make_list.items:
            assert isinstance(item, IrConstInt)

    def test_list_with_element_coercion(self) -> None:
        # list[decimal] = [1, 2] — elements need IntToDecimal
        prog = _lower("let _x: list[decimal] = [1, 2]\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        make_list = bind.value
        assert isinstance(make_list, IrMakeList)
        # Each element is IrCoerce(IrConstInt, IntToDecimal())
        for item in make_list.items:
            assert isinstance(item, IrCoerce)
            assert isinstance(item.value, IrConstInt)
            assert item.operation == IntToDecimal()

    def test_list_int_to_json_whole_list_coercion(self) -> None:
        # let j: json = [1, 2] — entire list is wrapped in ToJson
        prog = _lower("let _x: json = [1, 2]\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        # The value should be IrCoerce(IrMakeList(...), ToJson())
        coerce = bind.value
        assert isinstance(coerce, IrCoerce)
        assert coerce.operation == ToJson()
        assert isinstance(coerce.value, IrMakeList)


# ---------------------------------------------------------------------------
# lower_module: dict literal lowering
# ---------------------------------------------------------------------------


class TestDictLitLowering:
    def test_empty_dict(self) -> None:
        prog = _lower("let _x: dict[text, int] = {}\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        assert isinstance(bind.value, IrMakeDict)
        assert bind.value.entries == ()

    def test_dict_value_no_coercion(self) -> None:
        prog = _lower('let _x: dict[text, int] = {"a": 1}\n()')
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        make_dict = bind.value
        assert isinstance(make_dict, IrMakeDict)
        assert len(make_dict.entries) == 1
        key_expr, val_expr = make_dict.entries[0]
        assert isinstance(key_expr, IrConstText)
        assert key_expr.value == "a"
        assert isinstance(val_expr, IrConstInt)

    def test_dict_value_with_coercion(self) -> None:
        # dict[text, decimal] = {"a": 1} — value needs IntToDecimal
        prog = _lower('let _x: dict[text, decimal] = {"a": 1}\n()')
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        make_dict = bind.value
        assert isinstance(make_dict, IrMakeDict)
        _key, val_expr = make_dict.entries[0]
        assert isinstance(val_expr, IrCoerce)
        assert val_expr.operation == IntToDecimal()


# ---------------------------------------------------------------------------
# lower_module: let / var binding lowering
# ---------------------------------------------------------------------------


class TestBindingLowering:
    def test_let_binding_identity_no_coerce(self) -> None:
        # let x: int = 5 — no coercion needed
        prog = _lower("let _x: int = 5\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        assert not prog.symbols[bind.symbol].mutable
        assert isinstance(bind.value, IrConstInt)
        assert bind.value.value == 5

    def test_let_binding_int_to_decimal(self) -> None:
        # let d: decimal = 1 → IrBind(.., IrCoerce(IrConstInt, IntToDecimal))
        prog = _lower("let _d: decimal = 1\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        assert isinstance(bind.value, IrCoerce)
        assert bind.value.operation == IntToDecimal()
        assert isinstance(bind.value.value, IrConstInt)
        assert bind.value.value.value == 1

    def test_let_binding_to_json(self) -> None:
        # let j: json = 1 → IrCoerce(IrConstInt, ToJson)
        prog = _lower("let _j: json = 1\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        coerce = bind.value
        assert isinstance(coerce, IrCoerce)
        assert coerce.operation == ToJson()

    def test_let_binding_list_decimal_elements_coerced_not_outer(self) -> None:
        # let xs: list[decimal] = [1, 2]
        # Elements each get IntToDecimal; the list itself does NOT get MapList
        # because the list's own checked type is already list[decimal]
        # (the element coercion is applied at element level, not list level).
        prog = _lower("let _xs: list[decimal] = [1, 2]\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        # The bind value should be IrMakeList with coerced elements
        make_list = bind.value
        assert isinstance(make_list, IrMakeList)
        for item in make_list.items:
            assert isinstance(item, IrCoerce)
            assert item.operation == IntToDecimal()

    def test_let_var_ref_identity_no_coerce(self) -> None:
        # let a: list[int] = [1]; let b: list[int] = a — no coercion needed
        # (list[int] → list[decimal] is rejected by the type checker, so
        # MapList(IntToDecimal) on an IrLoad can only arise in future work
        # that extend assignability.  The contract example is forward-looking.)
        prog = _lower("let _a: list[int] = [1]\nlet _b: list[int] = _a\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind_b = inits[1]
        assert isinstance(bind_b, IrBind)
        # No coercion wrap — direct IrLoad
        load = bind_b.value
        assert isinstance(load, IrLoad)

    def test_var_binding_is_mutable(self) -> None:
        prog = _lower("var _x: int = 0\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        assert prog.symbols[bind.symbol].mutable

    def test_symbol_public_name(self) -> None:
        prog = _lower("let myvar: int = 1\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        desc = prog.symbols[bind.symbol]
        assert desc.public_name == "myvar"

    def test_symbol_owner_is_entry_module(self) -> None:
        prog = _lower("let _x: int = 1\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        desc = prog.symbols[bind.symbol]
        assert desc.owner == prog.entry_module


# ---------------------------------------------------------------------------
# lower_module: VarRef lowering (IrLoad)
# ---------------------------------------------------------------------------


class TestVarRefLowering:
    def test_varref_emits_ir_load(self) -> None:
        prog = _lower("let _x: int = 1\n_x")
        inits = prog.modules[prog.entry_module].initializers
        load = inits[1]
        assert isinstance(load, IrLoad)

    def test_varref_resolves_to_correct_symbol(self) -> None:
        prog = _lower("let _a: int = 1\n_a")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        load = inits[1]
        assert isinstance(load, IrLoad)
        assert load.symbol == bind.symbol


# ---------------------------------------------------------------------------
# lower_module: AssignStmt lowering (simple name target only)
# ---------------------------------------------------------------------------


class TestAssignStmtLowering:
    def test_simple_assign_emits_ir_assign_empty_path(self) -> None:
        prog = _lower("var _x: int = 0\n_x := 5\n()")
        inits = prog.modules[prog.entry_module].initializers
        assign = inits[1]
        assert isinstance(assign, IrAssign)
        assert assign.path == ()

    def test_simple_assign_symbol_matches_binding(self) -> None:
        prog = _lower("var _x: int = 0\n_x := 5\n()")
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        assign = inits[1]
        assert isinstance(assign, IrAssign)
        assert assign.symbol == bind.symbol

    def test_assign_with_coercion(self) -> None:
        # var x: decimal = 0.0; x := 1  (int → decimal coercion on RHS)
        prog = _lower("var _x: decimal = 0.0\n_x := 1\n()")
        inits = prog.modules[prog.entry_module].initializers
        assign = inits[1]
        assert isinstance(assign, IrAssign)
        coerce = assign.value
        assert isinstance(coerce, IrCoerce)
        assert coerce.operation == IntToDecimal()

    def test_assign_no_coercion_when_types_match(self) -> None:
        prog = _lower("var _x: int = 0\n_x := 5\n()")
        inits = prog.modules[prog.entry_module].initializers
        assign = inits[1]
        assert isinstance(assign, IrAssign)
        assert isinstance(assign.value, IrConstInt)


# ---------------------------------------------------------------------------
# lower_module: Block lowering
#
# A Block node appears as an expression only inside branches of control-flow
# (if/case/do/try) and function bodies. We test the
# Block lowering path by calling the lowerer's internal API directly with a
# synthetic Block node constructed from a lowered CheckedModule.  This
# exercises the Block arm of lower_expr and _lower_block without extra syntax
# control-flow in the source.
# ---------------------------------------------------------------------------


class TestBlockLowering:
    def test_block_emits_ir_block(self) -> None:
        # Build a CheckedModule from a simple multi-statement source so we
        # can reuse its node_types and resolution tables.  Then construct
        # a synthetic Block wrapping those items and call lower_expr directly.
        from agm.agl.syntax.nodes import Block, LetDecl, VarRef
        from agm.agl.syntax.spans import SourceSpan

        src = "let _a: int = 1\n_a"
        checked = _check(src)
        # Build a _Lowerer in the same state as lower_module would, but stop
        # before running the top-level body so we can call lower_expr manually.
        lowerer = _make_lowerer(checked, src)

        body = checked.resolved.program.body
        # body.items = [LetDecl(_a, ...), VarRef(_a, ...)]
        let_item = body.items[0]
        ref_item = body.items[1]
        assert isinstance(let_item, LetDecl)
        assert isinstance(ref_item, VarRef)

        sp = SourceSpan(
            start_line=1,
            start_col=1,
            end_line=2,
            end_col=3,
            start_offset=0,
            end_offset=len(src),
        )
        block = Block(items=(let_item, ref_item), span=sp, node_id=9999)

        # Call lower_expr — must allocate the LetDecl symbol first via
        # lower_item for the LetDecl, then lower_expr for VarRef can resolve.
        # We use lower_expr on the Block which calls lower_item on each child.
        ir = lowerer.lower_expr(block)
        assert isinstance(ir, IrBlock)
        # IrBlock should have exactly 2 items: IrBind + IrLoad
        assert len(ir.items) == 2
        assert isinstance(ir.items[0], IrBind)
        assert isinstance(ir.items[1], IrLoad)


# ---------------------------------------------------------------------------
# lower_module: sources table
# ---------------------------------------------------------------------------


class TestSourcesTable:
    def test_single_source_registered(self) -> None:
        prog = _lower("()")
        assert len(prog.sources) == 1

    def test_source_display_name(self) -> None:
        prog = _lower("()")
        (src_id,) = prog.sources
        assert prog.sources[src_id].display_name == "<test>"

    def test_source_normalized_text(self) -> None:
        src = "()"
        prog = _lower(src)
        (src_id,) = prog.sources
        assert prog.sources[src_id].normalized_text == src


# ---------------------------------------------------------------------------
# lower_module: nominals table starts empty
# ---------------------------------------------------------------------------


class TestNominalsEmpty:
    def test_nominals_contains_builtin_prelude_and_exceptions(self) -> None:
        """program.nominals always contains built-in prelude and exception types.

        Even an empty program populates nominals with all built-in prelude
        records/enums and exceptions. User-declared records/enums are added on top.
        """
        from agm.agl.ir.ids import NominalId
        from agm.agl.ir.program import NominalKind
        from agm.agl.modules.ids import PRELUDE_ID
        from agm.agl.semantics.types import BUILTIN_EXCEPTIONS, BUILTIN_PRELUDE_TYPES

        prog = _lower("()")
        nominal_names = {desc.nominal.declared_name for desc in prog.nominals.values()}
        for builtin_name in BUILTIN_PRELUDE_TYPES:
            assert builtin_name in nominal_names, (
                f"Built-in prelude type {builtin_name!r} missing from program.nominals"
            )
        for builtin_name in BUILTIN_EXCEPTIONS:
            assert builtin_name in nominal_names, (
                f"Built-in exception {builtin_name!r} missing from program.nominals"
            )

        assert prog.nominals[NominalId(PRELUDE_ID, "ExecResult")].kind is NominalKind.RECORD
        assert prog.nominals[NominalId(PRELUDE_ID, "ParsePolicy")].kind is NominalKind.ENUM
        assert prog.nominals[NominalId(PRELUDE_ID, "Abort")].kind is NominalKind.EXCEPTION

    def test_user_exception_nominal_stamped_with_declaring_module_id(self) -> None:
        """A user-declared exception's nominal is stamped with its real module_id.

        Declares the exception before a record so ``_build_nominals``' loop
        continues past the exception branch onto another declaration.
        """
        from agm.agl.ir.ids import NominalId
        from agm.agl.ir.program import NominalKind
        from agm.agl.modules.ids import ENTRY_ID

        source = (
            "exception Boom extends Exception\n"
            "  code: int\n"
            "\n"
            "record Point\n"
            "  x: int\n"
            "\n"
            "let p = Point(x = 1)\n"
            "p"
        )
        prog = _lower(source)
        boom_nominal = NominalId(ENTRY_ID, "Boom")
        assert boom_nominal in prog.nominals
        descriptor = prog.nominals[boom_nominal]
        assert descriptor.kind is NominalKind.EXCEPTION
        assert set(descriptor.fields) == {"message", "trace_id", "code"}

    def test_type_alias_does_not_create_spurious_nominal(self) -> None:
        """A type alias does NOT register a spurious NominalId in program.nominals.

        ``type Foo = Record`` must not create NominalId(..., "Foo") — only the
        canonical declaration NominalId(..., "Record") must exist.  Same for
        enum aliases.
        """
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import ENTRY_ID

        source = (
            "record Point\n"
            "  x: int\n"
            "  y: int\n"
            "\n"
            "type PointAlias = Point\n"
            "\n"
            "enum Color\n"
            "  | Red\n"
            "  | Blue\n"
            "\n"
            "type ColorAlias = Color\n"
            "\n"
            "let p = Point(x = 1, y = 2)\n"
            "let c = Color::Red\n"
            "()\n"
        )
        prog = _lower(source)

        nominal_names = {desc.display_name for desc in prog.nominals.values()}
        nominal_ids = set(prog.nominals.keys())

        # The canonical record and enum nominals must be present
        assert "Point" in nominal_names, "NominalId for 'Point' must be registered"
        assert "Color" in nominal_names, "NominalId for 'Color' must be registered"

        # The aliases must NOT register spurious nominals
        assert "PointAlias" not in nominal_names, (
            "Record alias 'PointAlias' must NOT register a spurious nominal descriptor"
        )
        assert NominalId(ENTRY_ID, "PointAlias") not in nominal_ids, (
            "NominalId(ENTRY_ID, 'PointAlias') must NOT appear in program.nominals"
        )
        assert "ColorAlias" not in nominal_names, (
            "Enum alias 'ColorAlias' must NOT register a spurious nominal descriptor"
        )
        assert NominalId(ENTRY_ID, "ColorAlias") not in nominal_ids, (
            "NominalId(ENTRY_ID, 'ColorAlias') must NOT appear in program.nominals"
        )


# ---------------------------------------------------------------------------
# lower_module: unsupported nodes raise a clear error
# ---------------------------------------------------------------------------


class TestUnsupportedNodes:
    def test_user_function_call_lowers_correctly(self) -> None:
        """Direct user function calls must lower without error."""
        # f() is a direct user function call supports this.
        prog = _lower("def f() -> int = 1\nlet result = f()\n()")
        # Verify the program has a function in the functions table
        assert len(prog.functions) == 1
        # Verify the let result = f() binding lowered to an IrDirectCall targeting f
        inits = prog.modules[prog.entry_module].initializers
        result_bind = next(
            (n for n in inits if isinstance(n, IrBind) and isinstance(n.value, IrDirectCall)),
            None,
        )
        assert result_bind is not None, "let result = f() did not lower to an IrDirectCall bind"
        assert isinstance(result_bind.value, IrDirectCall)
        assert result_bind.value.function_id in prog.functions

    def test_iife_lambda_call_lowers_to_indirect_call(self) -> None:
        """Calling an immediately-invoked lambda (non-VarRef, non-FieldAccess callee).

        When the callee of a Call is a Lambda, lowering produces an IrIndirectCall
        whose callee is an IrMakeClosure.
        """
        prog = _lower("(fn() -> int => 42)()\n()")
        inits = prog.modules[prog.entry_module].initializers
        # The IIFE evaluates to an IrIndirectCall
        # Initializaers: [IrIndirectCall(...), IrConstUnit()]
        indirect = inits[0]
        assert isinstance(indirect, IrIndirectCall), (
            f"Expected IrIndirectCall, got {type(indirect).__name__}"
        )
        assert isinstance(indirect.callee, IrMakeClosure)

    def test_qualified_enum_constructor_lowers_correctly(self) -> None:
        """Qualified constructor (e.g. Color::Red) lowers to IrMakeEnum/IrMakeConstructor.

        Qualified constructor lowering supports a nullary variant (no fields)
        lowers to IrMakeEnum (eagerly constructed).  A variant with fields lowers to
        IrMakeConstructor.  Here Red is nullary, so the binding value must be IrMakeEnum.
        """
        from agm.agl.ir.nodes import IrBind, IrMakeEnum

        source = """\
enum Color
  | Red
  | Blue

let c = Color::Red
()
"""
        prog = _lower(source)
        entry = prog.modules[prog.entry_module]
        found = False
        for node in entry.initializers:
            if isinstance(node, IrBind) and isinstance(node.value, IrMakeEnum):
                if node.value.variant == "Red":
                    found = True
        assert found, "Expected IrBind(value=IrMakeEnum(variant='Red')) in initializers"

    def test_lambda_lowers_to_make_closure_in_unsupported_class(self) -> None:
        """Lambda expressions now lower to IrMakeClosure."""
        prog = _lower("let f = fn(x: int) -> int => x + 1\n()")
        inits = prog.modules[prog.entry_module].initializers
        f_bind = inits[0]
        assert isinstance(f_bind, IrBind)
        assert isinstance(f_bind.value, IrMakeClosure)

    def test_if_lowers_correctly(self) -> None:
        """If expressions are lowered to IrIf in A."""
        from agm.agl.ir.nodes import IrBind, IrIf

        prog = _lower("let x = if true => 1 | else => 2\n()")
        entry = prog.modules[prog.entry_module]
        found = False
        for node in entry.initializers:
            if isinstance(node, IrBind) and isinstance(node.value, IrIf):
                assert node.value.has_else
                found = True
        assert found, "Expected IrBind(value=IrIf(has_else=True)) in initializers"

    def test_indexed_assign_lowers_to_ir_assign_with_path(self) -> None:
        """IndexTarget assignment lowers to IrAssign with a non-empty path."""
        from agm.agl.ir import IrAssign

        prog = _lower('var _d: dict[text, int] = {"a": 1}\n_d["a"] := 2\n()')
        entry = prog.modules[prog.entry_module]
        # AssignStmt lowers directly to IrAssign in the initializers list
        found = False
        for node in entry.initializers:
            if isinstance(node, IrAssign):
                assert len(node.path) >= 1
                found = True
        assert found, "Expected IrAssign(path=[...]) in initializers"


# ---------------------------------------------------------------------------
# lower_module: Location fields are valid
# ---------------------------------------------------------------------------


class TestLocationValidity:
    def test_location_on_int_literal_is_valid(self) -> None:
        prog = _lower("42")
        node = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(node, IrConstInt)
        loc = node.location
        assert loc.start_offset >= 0
        assert loc.start_offset <= loc.end_offset
        assert loc.start_line >= 1
        assert loc.start_col >= 0

    def test_location_source_id_in_sources(self) -> None:
        prog = _lower("42")
        node = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(node, IrConstInt)
        assert node.location.source_id in prog.sources


# ---------------------------------------------------------------------------
# lower_module: validate_ir integration
# ---------------------------------------------------------------------------


class TestValidateIrIntegration:
    def test_validate_ir_passes_on_complex_program(self) -> None:
        source = (
            "let _a: int = 1\n"
            "let _b: decimal = 2\n"
            "let _c: list[decimal] = [1, 2]\n"
            'let _d: dict[text, decimal] = {"x": 1}\n'
            "var _v: int = 0\n"
            "_v := 3\n"
            "()"
        )
        prog = _lower(source)
        validate_ir(prog)


# ---------------------------------------------------------------------------
# Direct _Lowerer unit test — IrField lowering (non-constructor FieldAccess)
# ---------------------------------------------------------------------------


class TestIrFieldLowering:
    """Unit tests for the IrField lowering path in _Lowerer.lower_expr.

    FieldAccess nodes that resolve to records/exceptions (rather than
    qualified constructor references) lower to IrField.  These tests exercise
    the _Lowerer directly because end-to-end lowering of FieldAccess requires
    constructor calls.
    """

    def test_field_access_lowers_to_ir_field(self) -> None:
        """Non-constructor FieldAccess lowers to IrField with correct field name.

        We construct a minimal FieldAccess AST node whose node_id is NOT in
        qualified_constructor_refs, pair it with a UnitLit obj (so lower_expr
        can recurse cleanly), and assert the result is IrField.
        """
        from agm.agl.ir.nodes import IrField
        from agm.agl.syntax.nodes import FieldAccess, UnitLit
        from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan

        # Build a CheckedModule from the trivial source "()" — we only need the
        # resolved/type-table scaffolding, not the actual program body.
        checked = _check("()")

        # A fresh node_id not present in qualified_constructor_refs.
        fake_node_id = 99999
        assert fake_node_id not in checked.resolved.qualified_constructor_refs

        span = SourceSpan(
            start_line=1,
            start_col=1,
            end_line=1,
            end_col=5,
            start_offset=0,
            end_offset=4,
            source=UNKNOWN_SOURCE,
        )
        unit_lit = UnitLit(span=span, node_id=fake_node_id + 1)
        field_access = FieldAccess(obj=unit_lit, field="myfield", span=span, node_id=fake_node_id)

        lowerer = _make_lowerer(checked, "()")
        result = lowerer.lower_expr(field_access)

        assert isinstance(result, IrField)
        assert result.field == "myfield"

    def test_kind_for_non_container_raises_assertion(self) -> None:
        """_kind_for_container raises AssertionError for a non-container type.

        Defensive guard: can only be triggered by a compiler bug (well-typed IR
        never passes a non-container type here).
        """
        import pytest

        checked = _check("()")
        lowerer = _make_lowerer(checked, "()")
        with pytest.raises(AssertionError, match="compiler bug"):
            lowerer._kind_for_container(IntType())

    def test_elem_type_for_non_container_raises_assertion(self) -> None:
        """_elem_type_for_container raises AssertionError for a non-container type.

        Defensive guard: can only be triggered by a compiler bug.
        """
        import pytest

        checked = _check("()")
        lowerer = _make_lowerer(checked, "()")
        with pytest.raises(AssertionError, match="compiler bug"):
            lowerer._elem_type_for_container(TextType())


# ---------------------------------------------------------------------------
# Function-related lowering coverage
# ---------------------------------------------------------------------------


class TestLowerFunctions:
    """Tests for function lowering coverage."""

    def test_try_with_bound_exception_in_function_body(self) -> None:
        """Function body with try-catch-as (bound handler) exercises clause.binding path."""
        source = (
            "def safe_add(a: int, b: int) -> int =\n"
            "  try\n"
            "    a + b\n"
            "  catch ArithmeticError as e =>\n"
            "    0\n"
            "let result = safe_add(3, 4)\n()"
        )
        prog = _lower(source)
        assert len(prog.functions) == 1
        # Verify the function body lowered to an IrTry with a bound handler
        fn_desc = next(iter(prog.functions.values()))
        body = _function_body(fn_desc)
        if isinstance(body, IrBlock):
            ir_try = next((item for item in body.items if isinstance(item, IrTry)), None)
            assert ir_try is not None, "Expected an IrTry node in function body IrBlock"
        else:
            assert isinstance(body, IrTry), f"Expected IrTry body, got {type(body).__name__}"
            ir_try = body
        assert any(h.symbol is not None for h in ir_try.handlers), (
            "Expected at least one handler with a bound symbol (catch ... as e)"
        )

    def test_builtin_print_in_function_body_lowers_to_ir_print(self) -> None:
        """print() inside a function body lowers to IrPrint."""
        from agm.agl.ir.nodes import IrPrint

        source = "def f(x: int) -> int =\n  print(x)\n  x + 1\nlet result = f(5)\n()"
        prog = _lower(source)
        # Find the function descriptor and verify its body contains IrPrint
        assert len(prog.functions) == 1
        fn_desc = next(iter(prog.functions.values()))
        # The function body is a block: [IrPrint(...), IrArith(x + 1)]
        from agm.agl.ir.nodes import IrBlock

        body = _function_body(fn_desc)
        assert isinstance(body, IrBlock)
        assert any(isinstance(item, IrPrint) for item in body.items)

    def test_indirect_call_via_let_binding_lowers_to_indirect_call(self) -> None:
        """Calling a let-bound function reference lowers to IrIndirectCall."""
        source = "def f(x: int) -> int = x + 1\nlet fn_ref = f\nlet result = fn_ref(5)\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        # inits[0] = IrBind(f, IrMakeClosure)
        # inits[1] = IrBind(fn_ref, ...)
        # inits[2] = IrBind(result, IrIndirectCall)
        result_bind = inits[2]
        assert isinstance(result_bind, IrBind)
        assert isinstance(result_bind.value, IrIndirectCall)

    def test_field_access_function_value_call_lowers_to_indirect_call(self) -> None:
        """Calling a function-typed record field via field access lowers to IrIndirectCall.

        The callee ``h.f`` is a FieldAccess that is not a qualified constructor, so it
        falls through to the indirect/value-call path rather than a direct or constructor
        call.
        """
        source = (
            "record Holder\n"
            "  f: (int) -> int\n"
            "let g = fn(x: int) -> int => x + 1\n"
            "let h = Holder(f = g)\n"
            "let r = h.f(5)\n()"
        )
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        indirect_binds = [
            n for n in inits if isinstance(n, IrBind) and isinstance(n.value, IrIndirectCall)
        ]
        assert len(indirect_binds) == 1


class TestScanCapturesLambdaBoundary:
    """Behavioral test for lambda-capture boundary detection in _scan_captures."""

    def test_scan_captures_stops_at_nested_lambda_boundary(self) -> None:
        """_scan_captures returns early at Lambda boundary without descending into it.

        When a function body contains a lambda expression, _scan_captures must treat
        Lambda as a scope boundary and stop descending.  The lambda's own captures are
        computed separately when the lambda is lowered.  Concretely: the outer def ``f``
        must NOT capture ``y`` (it references it only inside the lambda, not in f's own
        body); the lambda captures ``y`` independently.
        """
        # f's body is a block containing a lambda let-binding followed by the unit value.
        # _scan_captures should stop at the Lambda boundary rather than descending into
        # the lambda's body and incorrectly treating y as a capture of f.
        source = "let y = 5\ndef f() -> unit =\n  let _g = fn(u: unit) -> int => y\n  ()\nf()\n()"
        prog = _lower(source)
        # f should have no captures (y is not used in f's body directly)
        # Find the outer def whose function_symbol corresponds to "f"
        f_descs = [
            d for d in prog.functions.values() if prog.symbols[d.function_symbol].public_name == "f"
        ]
        assert len(f_descs) == 1
        f_desc = f_descs[0]
        # Check that the IrMakeClosure for f in the initializers has no captures for y
        # (The initializer for f is IrBind(fn_sym, IrMakeClosure(...)))
        f_init = None
        for node in prog.modules[prog.entry_module].initializers:
            if (
                isinstance(node, IrBind)
                and isinstance(node.value, IrMakeClosure)
                and node.value.function_id == f_desc.function_id
            ):
                f_init = node
                break
        assert f_init is not None, "IrBind for f not found in initializers"
        f_closure = f_init.value
        assert isinstance(f_closure, IrMakeClosure)
        # f should have no captures (y is not directly referenced in f's own body)
        assert len(f_closure.captures) == 0, (
            f"f should have no captures but has {f_closure.captures!r}"
        )


class TestScanCapturesLoopForIterWhileCond:
    """_scan_captures visits for_iter and while_cond in a Loop node (None-safe branches)."""

    def test_scan_captures_loop_for_iter_and_while_cond_detect_captures(self) -> None:
        """_scan_captures detects free-var captures in for_iter and while_cond.

        A Loop node with for_iter and while_cond referencing an outer binding must
        have those references detected as captures.  This covers the None-safe
        for_iter/while_cond branches in _scan_captures.  The test constructs
        synthetic Loop nodes directly so it can isolate the capture-scanning
        branches from the rest of the lowering pipeline.
        """
        from agm.agl.scope.symbols import BindingRef
        from agm.agl.syntax.nodes import Loop, UnitLit, VarRef
        from agm.agl.syntax.spans import SourceSpan

        # Minimal program with an outer var 'x' that is referenced so the
        # resolution table contains a BindingRef for it.
        source = "var x = 0\nvar _y = x\n()"
        checked = _check(source)

        # Locate x's BindingRef via any checked reference to it.
        x_ref = next(ref for ref in checked.resolved.resolution.values() if ref.name == "x")

        # Synthetic VarRef nodes for for_iter and while_cond, using fresh node_ids
        # that don't collide with any real node in the checked program.
        fake_span = SourceSpan(
            start_line=1,
            start_col=1,
            end_line=1,
            end_col=2,
            start_offset=0,
            end_offset=1,
        )
        for_iter_ref = VarRef(name="x", span=fake_span, node_id=88881)
        while_cond_ref = VarRef(name="x", span=fake_span, node_id=88882)

        # Inject checked references for the synthetic node_ids so _record_capture can find them.
        checked.resolved.resolution[88881] = x_ref
        checked.resolved.resolution[88882] = x_ref

        # Synthetic Loop: for_iter and while_cond both reference x; body is unit.
        body = UnitLit(span=fake_span, node_id=88883)
        loop_node = Loop(
            for_var="i",
            for_iter=for_iter_ref,
            for_range_to=None,
            for_range_down=False,
            for_range_by=None,
            while_cond=while_cond_ref,
            bound=None,
            body=body,
            until_cond=None,
            span=fake_span,
            node_id=88884,
        )

        lowerer = _make_lowerer(checked, source)
        local_ids: set[int] = set()
        captured: dict[int, BindingRef] = {}
        lowerer._scan_captures(loop_node, local_ids, captured)

        # Both for_iter and while_cond reference x: its decl_node_id must appear
        # in captured (registered by _record_capture via the injected resolutions).
        assert x_ref.decl_node_id in captured, (
            f"x (decl_node_id={x_ref.decl_node_id!r}) must be detected as a capture "
            f"via for_iter and while_cond; captured={captured!r}"
        )


# ===========================================================================
# Lambda lowering and indirect call lowering (golden tests)
# ===========================================================================


class TestLambdaLowering:
    """Golden tests: lambda lowers to IrMakeClosure with correct FunctionDescriptor."""

    def test_lambda_lowers_to_make_closure(self) -> None:
        """A lambda expression lowers to IrMakeClosure (not IrBind + IrMakeClosure)."""
        source = "let dbl = fn(x: int) -> int => x * 2\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        # The module initializer for `let dbl = ...` is an IrBind wrapping an IrMakeClosure.
        dbl_bind = inits[0]
        assert isinstance(dbl_bind, IrBind)
        assert isinstance(dbl_bind.value, IrMakeClosure)
        fn_id = dbl_bind.value.function_id
        assert fn_id in prog.functions, "Lambda's FunctionDescriptor must be in functions table"

    def test_lambda_registers_function_descriptor(self) -> None:
        """Lambda's FunctionDescriptor has correct body and param count."""
        source = "let inc = fn(x: int) -> int => x + 1\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        dbl_bind = inits[0]
        assert isinstance(dbl_bind, IrBind)
        assert isinstance(dbl_bind.value, IrMakeClosure)
        fn_id = dbl_bind.value.function_id
        desc = prog.functions[fn_id]
        assert len(desc.params) == 1, "Lambda with 1 param should have 1 FunctionParam"

    def test_module_binding_lambda_resolves_without_capture(self) -> None:
        """A lambda referencing an outer module-level binding has no captures.

        Module bindings are resolved through the base frame rather than via
        closure capture, so IrMakeClosure.captures is empty even when the lambda
        body references a name from an enclosing module initializer.
        """
        source = "let offset = 10\nlet add_off = fn(x: int) -> int => x + offset\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        # offset is first, add_off is second
        add_off_bind = inits[1]
        assert isinstance(add_off_bind, IrBind)
        assert isinstance(add_off_bind.value, IrMakeClosure)
        captures = add_off_bind.value.captures
        assert captures == (), "Module bindings resolve through the base frame"

    def test_lambda_closure_has_non_none_body(self) -> None:
        """A lambda with inferred return type lowers to a closure with a non-None body."""
        source = "let f = fn(x: int) => x\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        assert isinstance(bind.value, IrMakeClosure)
        fn_id = bind.value.function_id
        desc = prog.functions[fn_id]
        # Body is the param load (int -> int, no coercion needed since identity type)
        assert _function_body(desc) is not None

    def test_lambda_body_result_coerced(self) -> None:
        """Lambda body is lowered with lower_coerced so int-to-decimal coercion is baked in."""
        source = "let to_dec = fn(x: int) -> decimal => x\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        assert isinstance(bind.value, IrMakeClosure)
        fn_id = bind.value.function_id
        desc = prog.functions[fn_id]
        # The body must be an IrCoerce (int → decimal) wrapping an IrLoad
        body = _function_body(desc)
        assert isinstance(body, IrCoerce), (
            f"Expected IrCoerce for int->decimal body, got {type(body).__name__}"
        )

    def test_lambda_private_function_symbol(self) -> None:
        """Lambda's function_symbol has public_name=None (private synthetic symbol)."""
        source = "let f = fn(x: int) -> int => x\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        assert isinstance(bind.value, IrMakeClosure)
        fn_id = bind.value.function_id
        desc = prog.functions[fn_id]
        sym_desc = prog.symbols[desc.function_symbol]
        assert sym_desc.public_name is None, (
            "Lambda's synthetic function_symbol should be private (public_name=None)"
        )


class TestIndirectCallLowering:
    """Golden tests: indirect call lowers to IrIndirectCall with coerced args."""

    def test_indirect_call_lowers_to_indirect_call_node(self) -> None:
        """A value-call lowers to IrIndirectCall (not IrDirectCall)."""
        source = "let f = fn(x: int) -> int => x + 1\nlet r = f(5)\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        # inits[0] = IrBind(f, IrMakeClosure)
        # inits[1] = IrBind(r, ...)
        r_bind = inits[1]
        assert isinstance(r_bind, IrBind)
        assert isinstance(r_bind.value, IrIndirectCall), (
            f"Expected IrIndirectCall for value-call, got {type(r_bind.value).__name__}"
        )

    def test_indirect_call_args_coerced(self) -> None:
        """Indirect call arguments are lowered WITH coercion (lower_coerced, not lower_expr).

        When the arg type already matches the param type, compile_coercion returns None
        and lower_coerced emits no IrCoerce wrapper — so the absence of IrCoerce for an
        int→int call is the correct identity-coercion-elision behavior, not a sign that
        coercion is skipped.  The key invariant: the arg is lowered via lower_coerced,
        which bakes in any needed coercion (e.g. int→decimal) at compile time.
        """
        source = "let f = fn(x: int) -> int => x\nlet r = f(7)\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        r_bind = inits[1]
        assert isinstance(r_bind, IrBind)
        assert isinstance(r_bind.value, IrIndirectCall)
        # int→int: identity coercion is elided, so arg is bare IrConstInt (no IrCoerce)
        assert len(r_bind.value.arguments) == 1
        arg = r_bind.value.arguments[0]
        assert isinstance(arg, IrConstInt), (
            f"Arg should be bare IrConstInt (identity coercion elided), got {type(arg).__name__}"
        )
        assert not isinstance(arg, IrCoerce)

    def test_indirect_call_callee_is_lower_expr_result(self) -> None:
        """Indirect call callee is lowered with lower_expr (no coercion on callee)."""
        source = "let f = fn(x: int) -> int => x\nlet r = f(3)\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        r_bind = inits[1]
        assert isinstance(r_bind, IrBind)
        assert isinstance(r_bind.value, IrIndirectCall)
        # Callee should be IrLoad(f's symbol), not IrCoerce
        callee = r_bind.value.callee
        assert isinstance(callee, IrLoad), f"Callee should be IrLoad, got {type(callee).__name__}"

    def test_indirect_call_validates_deep(self) -> None:
        """IrIndirectCall from end-to-end lowering passes validate_ir deep=True."""
        source = "let f = fn(x: int) -> int => x + 1\nlet r = f(5)\n()"
        prog = _lower(source)
        validate_ir(prog, deep=True)  # no exception


class TestPartialCallLowering:
    """Golden tests for lowering placeholder calls to closure IR."""

    def test_declared_partial_call_captures_non_holes_and_preserves_defaults(self) -> None:
        source = "def f(x: int, y: int, z: int = 9) -> int = x + y + z\nlet h = f(?, 2)\n()"
        prog = _lower(source)
        h_bind = prog.modules[prog.entry_module].initializers[1]
        assert isinstance(h_bind, IrBind)
        assert isinstance(h_bind.value, IrBlock)
        block_items = h_bind.value.items
        assert len(block_items) == 2
        captured_bind = block_items[0]
        assert isinstance(captured_bind, IrBind)
        assert isinstance(captured_bind.value, IrConstInt)
        assert captured_bind.value.value == 2
        make_closure = block_items[1]
        assert isinstance(make_closure, IrMakeClosure)
        assert make_closure.captures == (IrCapture(captured_bind.symbol, by_cell=False),)

        desc = prog.functions[make_closure.function_id]
        assert len(desc.params) == 1
        assert isinstance(desc.impl, IrFunctionBody)
        assert isinstance(desc.impl.body, IrDirectCall)
        args = desc.impl.body.arguments
        assert len(args) == 3
        assert isinstance(args[0], IrLoad)
        assert args[0].symbol == desc.params[0].symbol
        assert isinstance(args[1], IrLoad)
        assert args[1].symbol == captured_bind.symbol
        assert args[2] == UseDefault(param_index=2)

    def test_value_partial_call_captures_callee_before_arguments(self) -> None:
        source = "let f = fn(x: int, y: int) -> int => x + y\nlet h = f(?, 2)\n()"
        prog = _lower(source)
        h_bind = prog.modules[prog.entry_module].initializers[1]
        assert isinstance(h_bind, IrBind)
        assert isinstance(h_bind.value, IrBlock)
        callee_bind, arg_bind, make_closure = h_bind.value.items
        assert isinstance(callee_bind, IrBind)
        assert isinstance(callee_bind.value, IrLoad)
        assert isinstance(arg_bind, IrBind)
        assert isinstance(arg_bind.value, IrConstInt)
        assert isinstance(make_closure, IrMakeClosure)
        assert make_closure.captures == (
            IrCapture(callee_bind.symbol, by_cell=False),
            IrCapture(arg_bind.symbol, by_cell=False),
        )

        desc = prog.functions[make_closure.function_id]
        assert isinstance(desc.impl, IrFunctionBody)
        assert isinstance(desc.impl.body, IrIndirectCall)
        assert isinstance(desc.impl.body.callee, IrLoad)
        assert desc.impl.body.callee.symbol == callee_bind.symbol
        assert isinstance(desc.impl.body.arguments[0], IrLoad)
        assert desc.impl.body.arguments[0].symbol == desc.params[0].symbol
        assert isinstance(desc.impl.body.arguments[1], IrLoad)
        assert desc.impl.body.arguments[1].symbol == arg_bind.symbol

    def test_constructor_partial_call_body_constructs_from_load_slots(self) -> None:
        source = "record Point\n  x: int\n  y: int\nlet make = Point(x = ?, y = 2)\n()"
        prog = _lower(source)
        make_bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(make_bind, IrBind)
        assert isinstance(make_bind.value, IrBlock)
        captured_bind, make_closure = make_bind.value.items
        assert isinstance(captured_bind, IrBind)
        assert isinstance(make_closure, IrMakeClosure)

        desc = prog.functions[make_closure.function_id]
        assert isinstance(desc.impl, IrFunctionBody)
        assert isinstance(desc.impl.body, IrMakeRecord)
        assert len(desc.impl.body.fields) == 2
        x_value = desc.impl.body.fields[0][1]
        y_value = desc.impl.body.fields[1][1]
        assert isinstance(x_value, IrLoad)
        assert x_value.symbol == desc.params[0].symbol
        assert isinstance(y_value, IrLoad)
        assert y_value.symbol == captured_bind.symbol

    def test_partial_call_captured_argument_coercion_is_inside_synthesized_body(self) -> None:
        source = "def f(x: decimal, y: decimal) -> decimal = x + y\nlet h = f(?, 1)\n()"
        prog = _lower(source)
        h_bind = prog.modules[prog.entry_module].initializers[1]
        assert isinstance(h_bind, IrBind)
        assert isinstance(h_bind.value, IrBlock)
        captured_bind, make_closure = h_bind.value.items
        assert isinstance(captured_bind, IrBind)
        assert isinstance(make_closure, IrMakeClosure)

        desc = prog.functions[make_closure.function_id]
        assert isinstance(desc.impl, IrFunctionBody)
        assert isinstance(desc.impl.body, IrDirectCall)
        captured_arg = desc.impl.body.arguments[1]
        assert isinstance(captured_arg, IrCoerce)
        assert isinstance(captured_arg.operation, IntToDecimal)
        assert isinstance(captured_arg.value, IrLoad)
        assert captured_arg.value.symbol == captured_bind.symbol

    def test_lower_expr_placeholder_guard(self) -> None:
        import pytest

        from agm.agl.syntax.spans import SourceSpan

        checked = _check("()")
        lowerer = _make_lowerer(checked, "()")
        span = SourceSpan(
            start_line=1,
            start_col=0,
            end_line=1,
            end_col=1,
            start_offset=0,
            end_offset=1,
        )
        with pytest.raises(AssertionError, match="placeholder"):
            lowerer.lower_expr(Placeholder(index=None, span=span, node_id=999_001))


# ---------------------------------------------------------------------------
# lower_program: multi-module golden test
# ---------------------------------------------------------------------------


class TestLowerGraph:
    """Golden tests for lower_program."""

    def test_lower_program_simple(self, tmp_path: Path) -> None:
        """lower_program on a two-module program builds a valid ExecutableProgram.

        Asserts the task-specified structure for a 2-module program:
        - Both modules appear in ``program.modules`` with distinct entries.
        - Both modules' functions appear in ``program.functions`` with DISTINCT FunctionIds.
        - ``program.nominals`` contains types from both modules (one record per module).
        - Exactly one ``SourceFile`` per module (2 sources total).
        - The library ``ExecutableModule.initializers`` contains ONLY function binds
          (IrBind wrapping IrMakeClosure).
        - The entry module is LAST in ``program.modules`` insertion order.
        - ``validate_ir`` passes (deep=True).
        """
        import os

        from agm.agl.lower.program import lower_program
        from agm.agl.modules.ids import ModuleId
        from agm.agl.modules.loader import load_graph
        from agm.agl.modules.roots import RootSet
        from agm.agl.scope.program import resolve_program
        from agm.agl.typecheck.program import check_program

        # Library defines a record type + a function using it.
        # Entry defines its own record type + imports lib's function.
        lib_source = (
            "record LibPoint\n"
            "  x: int\n"
            "  y: int\n"
            "\n"
            "def make_point(a: int, b: int) -> LibPoint =\n"
            "    LibPoint(x = a, y = b)\n"
        )
        entry_source = (
            "import lib\n"
            "record EntryBox\n"
            "  width: int\n"
            "\n"
            "let result = lib::make_point(1, 2)\n"
            "let box = EntryBox(width = 10)\n"
            "()\n"
        )

        root = tmp_path / "root"
        root.mkdir()
        lib_mid = ModuleId.from_dotted("lib")
        lib_path = root / lib_mid.relpath().replace("/", os.sep)
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib_path.write_text(lib_source)

        mg = load_graph(
            entry_source,
            entry_path=None,
            roots=RootSet(roots=frozenset({root, _REPO_STDLIB_ROOT})),
        )
        rg = resolve_program(mg)
        cg = check_program(rg, _caps())

        prog = lower_program(_compiled_checked(cg))

        # Both modules must appear
        assert len(prog.modules) == 2

        # Entry module is LAST in insertion order
        module_ids = list(prog.modules.keys())
        assert module_ids[-1] == prog.entry_module, (
            "Entry module must be last in program.modules insertion order"
        )

        # Exactly one SourceFile per module
        assert len(prog.sources) == 2

        # Both modules' functions appear in program.functions with DISTINCT FunctionIds.
        # lib has make_point; entry has no user functions here, but they share one table.
        lib_fn_ids = {
            desc.function_id for desc in prog.functions.values() if desc.module_id == lib_mid
        }
        assert len(lib_fn_ids) >= 1, "lib module must contribute at least one FunctionId"
        # All FunctionIds across both modules must be distinct
        assert len(prog.functions) == len({d.function_id for d in prog.functions.values()}), (
            "All FunctionIds must be distinct across modules"
        )

        # program.nominals contains types from BOTH modules: LibPoint (lib) and EntryBox (entry).
        nominal_names = {desc.display_name for desc in prog.nominals.values()}
        assert "LibPoint" in nominal_names, "LibPoint from lib module must be in nominals"
        assert "EntryBox" in nominal_names, "EntryBox from entry module must be in nominals"

        # Library ExecutableModule.initializers contains ONLY IrBind wrapping IrMakeClosure.
        lib_mod = prog.modules[lib_mid]
        for init_node in lib_mod.initializers:
            assert isinstance(init_node, IrBind), (
                f"Library initializer must be IrBind, got {type(init_node).__name__}"
            )
            assert isinstance(init_node.value, IrMakeClosure), (
                f"Library IrBind value must be IrMakeClosure, got {type(init_node.value).__name__}"
            )

        # The entry module's result binding must be in the symbols table
        result_syms = [desc for desc in prog.symbols.values() if desc.public_name == "result"]
        assert len(result_syms) == 1

        # The suite-enabled self-checks already validated this program during
        # lowering; call validate_ir explicitly so the test pins it regardless.
        validate_ir(prog, deep=True)

    def test_lower_program_type_alias_no_spurious_nominal(self, tmp_path: Path) -> None:
        """Type alias does not register a spurious NominalId in lower_program.

        A program with ``type Foo = Point`` (where Point is a record) must NOT
        create a ``NominalId(mid, "Foo")`` entry in ``program.nominals``.
        Only the canonical declaration site ``NominalId(mid, "Point")`` must exist.
        """
        import os

        from agm.agl.ir.ids import NominalId
        from agm.agl.lower.program import lower_program
        from agm.agl.modules.ids import ModuleId
        from agm.agl.modules.loader import load_graph
        from agm.agl.modules.roots import RootSet
        from agm.agl.scope.program import resolve_program
        from agm.agl.typecheck.program import check_program

        # Library defines a record and an enum, each with an alias pointing to them.
        # Exercises both the RecordType and EnumType alias-skip guards in graph.py.
        lib_source = (
            "record Point\n"
            "  x: int\n"
            "  y: int\n"
            "\n"
            "type PointAlias = Point\n"
            "\n"
            "enum Color\n"
            "  | Red\n"
            "  | Blue\n"
            "\n"
            "type ColorAlias = Color\n"
            "\n"
            "def origin() -> Point =\n"
            "    Point(x = 0, y = 0)\n"
        )
        entry_source = "import lib\nlet p = lib::origin()\n()\n"

        root = tmp_path / "root"
        root.mkdir()
        lib_mid = ModuleId.from_dotted("lib")
        lib_path = root / lib_mid.relpath().replace("/", os.sep)
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib_path.write_text(lib_source)

        mg = load_graph(
            entry_source,
            entry_path=None,
            roots=RootSet(roots=frozenset({root, _REPO_STDLIB_ROOT})),
        )
        rg = resolve_program(mg)
        cg = check_program(rg, _caps())

        prog = lower_program(_compiled_checked(cg))

        nominal_names = {desc.display_name for desc in prog.nominals.values()}
        nominal_ids = set(prog.nominals.keys())

        # Canonical record and enum nominals must be present
        assert "Point" in nominal_names, "NominalId for 'Point' must be registered"
        assert "Color" in nominal_names, "NominalId for 'Color' must be registered"

        # Record alias must NOT register a spurious nominal
        assert "PointAlias" not in nominal_names, (
            "Record alias 'PointAlias' must NOT register a spurious nominal descriptor"
        )
        assert NominalId(lib_mid, "PointAlias") not in nominal_ids, (
            "NominalId(lib_mid, 'PointAlias') must NOT appear in program.nominals"
        )

        # Enum alias must NOT register a spurious nominal (exercises EnumType guard in graph.py)
        assert "ColorAlias" not in nominal_names, (
            "Enum alias 'ColorAlias' must NOT register a spurious nominal descriptor"
        )
        assert NominalId(lib_mid, "ColorAlias") not in nominal_ids, (
            "NominalId(lib_mid, 'ColorAlias') must NOT appear in program.nominals"
        )


# ---------------------------------------------------------------------------
# Golden lowering: print, parse_json, param declarations
# ---------------------------------------------------------------------------


class TestHostOpLowering:
    """Golden lowering tests for host operations."""

    def test_print_lowers_to_ir_print(self) -> None:
        """print(x) lowers to IrPrint wrapping the argument expression."""
        from agm.agl.ir.nodes import IrPrint

        source = "let x = 42\nprint(x)\n()"
        prog = _lower(source)
        entry = prog.modules[list(prog.modules.keys())[-1]]
        # initializers are [IrBind(x=42), IrPrint(IrLoad(x)), IrConstUnit]
        ir_print = next((node for node in entry.initializers if isinstance(node, IrPrint)), None)
        assert ir_print is not None, "Expected IrPrint in entry initializers"
        assert isinstance(ir_print, IrPrint)

    def test_render_lowers_to_ir_render_value(self) -> None:
        """render(x, pretty:, quote_strings:) lowers to IrRenderValue."""
        from agm.agl.ir.nodes import IrBind, IrRenderValue

        source = 'let s = render("x", pretty = false, quote_strings = false)\n()'
        prog = _lower(source)
        entry = prog.modules[list(prog.modules.keys())[-1]]
        ir_bind = next((node for node in entry.initializers if isinstance(node, IrBind)), None)
        assert ir_bind is not None, "Expected IrBind in entry initializers"
        assert isinstance(ir_bind.value, IrRenderValue)
        assert ir_bind.value.pretty is not None
        assert ir_bind.value.quote_strings is not None

    def test_parse_json_lowers_to_ir_parse_json(self) -> None:
        """parse_json(s) lowers to IrParseJson wrapping the argument expression."""
        from agm.agl.ir.nodes import IrBind, IrParseJson

        source = "let j = parse_json('null')\n()"
        prog = _lower(source)
        entry = prog.modules[list(prog.modules.keys())[-1]]
        # The binding `let j = parse_json(...)` lowers to IrBind(value=IrParseJson(...))
        ir_bind = next((node for node in entry.initializers if isinstance(node, IrBind)), None)
        assert ir_bind is not None, "Expected IrBind in entry initializers"
        assert isinstance(ir_bind, IrBind)
        assert isinstance(ir_bind.value, IrParseJson), (
            f"Expected IrBind.value to be IrParseJson, got {type(ir_bind.value).__name__}"
        )

    def test_param_required_lowers_to_ir_param(self) -> None:
        """A required param (no default) produces an IrParam with required=True and no default."""
        from agm.agl.ir.program import IrParam

        source = "param n: int\nlet result = n + 1\n()"
        prog = _lower(source)
        assert len(prog.params) == 1
        p = prog.params[0]
        assert isinstance(p, IrParam)
        assert p.public_name == "n"
        assert p.required is True
        assert p.default is None

    def test_param_with_default_lowers_to_ir_param(self) -> None:
        """A param with a default produces an IrParam with required=False and a default expr."""
        from agm.agl.ir.nodes import IrConstInt
        from agm.agl.ir.program import IrParam

        source = "param n: int = 7\nlet result = n + 1\n()"
        prog = _lower(source)
        assert len(prog.params) == 1
        p = prog.params[0]
        assert isinstance(p, IrParam)
        assert p.public_name == "n"
        assert p.required is False
        assert isinstance(p.default, IrConstInt), (
            f"Expected IrConstInt default, got {type(p.default).__name__}"
        )
        assert p.default.value == 7

    def test_multiple_params_all_in_program_params(self) -> None:
        """Multiple param declarations each produce an IrParam in program.params."""
        source = "param x: int\nparam y: int = 5\nlet sum = x + y\n()"
        prog = _lower(source)
        assert len(prog.params) == 2
        public_names = {p.public_name for p in prog.params}
        assert public_names == {"x", "y"}
        required_map = {p.public_name: p.required for p in prog.params}
        assert required_map["x"] is True
        assert required_map["y"] is False

    def test_ask_lowers_to_ir_ask_m6b(self) -> None:
        """ask() now lowers to IrAsk."""
        from agm.agl.ir.nodes import IrAsk, IrBind

        source = 'agent impl\nlet r: text = ask("prompt", agent = impl)\n()'
        prog = _lower(source)
        # The initializers contain IrBind(IrAgentHandle) for `impl` then IrBind(IrAsk) for `r`.
        inits = prog.modules[prog.entry_module].initializers
        ask_binds = [n for n in inits if isinstance(n, IrBind) and isinstance(n.value, IrAsk)]
        assert len(ask_binds) == 1
        assert len(prog.contracts) == 1

    def test_exec_lowers_to_ir_exec(self) -> None:
        """exec() lowers to IrExec."""
        from agm.agl.ir.nodes import IrExec

        source = 'exec("ls -la")\n()'
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        exec_nodes = [n for n in inits if isinstance(n, IrExec)]
        assert len(exec_nodes) == 1, f"Expected 1 IrExec, found {len(exec_nodes)}"

    def test_agent_decl_lowers_to_ir_agent_handle_bind(self) -> None:
        """AgentDecl lowers to IrBind(symbol, IrAgentHandle(name))."""
        from agm.agl.ir.nodes import IrAgentHandle, IrBind

        source = "agent my_agent\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        # Expect an IrBind whose value is IrAgentHandle with the agent's name.
        handle_binds = [
            n for n in inits if isinstance(n, IrBind) and isinstance(n.value, IrAgentHandle)
        ]
        assert len(handle_binds) == 1, (
            f"Expected exactly 1 IrBind(IrAgentHandle), got {len(handle_binds)}"
        )
        assert handle_binds[0].value.agent_name == "my_agent"

    def test_ask_request_lowers_to_ir_ask_request_with_contract(self) -> None:
        """ask-request lowers to IrAskRequest + ContractRequest in program.contracts."""
        from agm.agl.ir.nodes import IrAskRequest, IrBind

        source = 'agent worker\nlet req = ask-request("my prompt", agent = worker)\n()'
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        # The ask-request binding lowers to IrBind(symbol, IrAskRequest(...)).
        ask_req_binds = [
            n for n in inits if isinstance(n, IrBind) and isinstance(n.value, IrAskRequest)
        ]
        assert len(ask_req_binds) == 1, (
            f"Expected exactly 1 IrBind(IrAskRequest), got {len(ask_req_binds)}"
        )
        ask_req = ask_req_binds[0].value
        assert isinstance(ask_req, IrAskRequest)
        # The contract_id must reference an entry in program.contracts.
        assert ask_req.contract_id in prog.contracts, (
            f"IrAskRequest.contract_id {ask_req.contract_id} not in program.contracts"
        )
        contract = prog.contracts[ask_req.contract_id]
        # ask-request is always is_unit=False (result is always an AgentRequest record).
        assert contract.is_unit is False


# ---------------------------------------------------------------------------
# Structural lowering: lambda capture positive path (non-empty captures)
# ---------------------------------------------------------------------------


class TestLambdaCapturePositive:
    """Structural tests for the lambda capture positive path.

    The negative (empty-captures) path for module-level bindings is already
    covered in TestLambdaLowering.  These tests assert the non-empty
    capture shape when a lambda references function-local bindings, and verify
    by_cell for both param (let-like) and var captures.
    """

    def test_lambda_captures_param_with_by_cell_false(self) -> None:
        """Lambda capturing a function parameter produces by_cell=False.

        A param is declared with mutable=False; IrCapture.by_cell must be False
        (snapshot-value semantics, not cell-sharing).  Asserts captures is
        non-empty and the single entry has by_cell=False matching the symbol.
        """
        source = "def make_fn(n: int) -> unit =\n  let _g = fn(x: int) -> int => n + x\n  ()\n()\n"
        prog = _lower(source)
        make_fn_desc = next(
            d
            for d in prog.functions.values()
            if prog.symbols[d.function_symbol].public_name == "make_fn"
        )
        body = _function_body(make_fn_desc)
        assert isinstance(body, IrBlock)
        # Find the IrMakeClosure in the function body (the nested lambda)
        lambda_closure: IrMakeClosure | None = None
        for item in body.items:
            if isinstance(item, IrBind) and isinstance(item.value, IrMakeClosure):
                lambda_closure = item.value
                break
        assert lambda_closure is not None, "Expected IrMakeClosure in make_fn body"
        # The lambda captures n (param) — captures must be non-empty
        captures = lambda_closure.captures
        assert len(captures) == 1, f"Expected 1 capture, got {captures!r}"
        cap = captures[0]
        assert isinstance(cap, IrCapture)
        # param is immutable → by_cell=False
        assert cap.by_cell is False, "param capture must have by_cell=False"
        assert not prog.symbols[cap.symbol].mutable

    def test_lambda_captures_var_with_by_cell_true(self) -> None:
        """Lambda capturing a function-local var produces by_cell=True.

        A var binding is mutable=True; IrCapture.by_cell must be True
        (cell-sharing so the lambda sees mutations).  Asserts captures is
        non-empty and the single entry has by_cell=True matching the symbol.
        """
        source = (
            "def make_fn() -> unit =\n"
            "  var count: int = 0\n"
            "  let _g = fn() -> int => count\n"
            "  ()\n"
            "()\n"
        )
        prog = _lower(source)
        make_fn_desc = next(
            d
            for d in prog.functions.values()
            if prog.symbols[d.function_symbol].public_name == "make_fn"
        )
        body = _function_body(make_fn_desc)
        assert isinstance(body, IrBlock)
        lambda_closure = None
        for item in body.items:
            if isinstance(item, IrBind) and isinstance(item.value, IrMakeClosure):
                lambda_closure = item.value
                break
        assert lambda_closure is not None, "Expected IrMakeClosure in make_fn body"
        captures = lambda_closure.captures
        assert len(captures) == 1, f"Expected 1 capture, got {captures!r}"
        cap = captures[0]
        assert isinstance(cap, IrCapture)
        # var is mutable → by_cell=True
        assert cap.by_cell is True, "var capture must have by_cell=True"
        assert prog.symbols[cap.symbol].mutable

    def test_lambda_captures_both_param_and_var_with_correct_by_cell(self) -> None:
        """Lambda capturing both a param (by_cell=False) and a var (by_cell=True).

        Asserts a two-entry captures tuple where each by_cell value matches
        the symbol's mutable flag, covering both False and True in one test.
        A wrong by_cell in the lowerer causes this test to fail.
        """
        source = (
            "def make_fn(n: int) -> unit =\n"
            "  var count: int = 0\n"
            "  let _g = fn() -> int => n + count\n"
            "  ()\n"
            "()\n"
        )
        prog = _lower(source)
        make_fn_desc = next(
            d
            for d in prog.functions.values()
            if prog.symbols[d.function_symbol].public_name == "make_fn"
        )
        body = _function_body(make_fn_desc)
        assert isinstance(body, IrBlock)
        lambda_closure = None
        for item in body.items:
            if isinstance(item, IrBind) and isinstance(item.value, IrMakeClosure):
                lambda_closure = item.value
                break
        assert lambda_closure is not None, "Expected IrMakeClosure in make_fn body"
        captures = lambda_closure.captures
        # Both n (param) and count (var) must be captured — non-empty
        assert len(captures) == 2, f"Expected 2 captures (n + count), got {captures!r}"
        # Each capture's by_cell must match the symbol's mutable flag
        for cap in captures:
            assert isinstance(cap, IrCapture)
            assert cap.by_cell == prog.symbols[cap.symbol].mutable, (
                f"by_cell={cap.by_cell} must equal symbol.mutable="
                f"{prog.symbols[cap.symbol].mutable} for cap {cap!r}"
            )
        # Exactly one by_cell=False entry (the param n)
        false_caps = [c for c in captures if not c.by_cell]
        assert len(false_caps) == 1
        assert not prog.symbols[false_caps[0].symbol].mutable
        # Exactly one by_cell=True entry (the var count)
        true_caps = [c for c in captures if c.by_cell]
        assert len(true_caps) == 1
        assert prog.symbols[true_caps[0].symbol].mutable


# ---------------------------------------------------------------------------
# Structural lowering: binary-op kind selection
# ---------------------------------------------------------------------------


class TestBinaryOpKindSelection:
    """Structural tests for _lower_binary_op: node type and kind/op fields.

    Each test lowers a single binary-expression and asserts the exact IR node
    type plus its decision-bearing fields.  A wrong kind selection in the
    lowerer causes the test to fail.
    """

    def test_int_add_lowers_to_arith_add_int(self) -> None:
        """+ on two ints → IrArith with op=ADD and kind=INT."""
        prog = _lower("let r = 1 + 2\n()")
        bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(bind, IrBind)
        arith = bind.value
        assert isinstance(arith, IrArith)
        assert arith.op is ArithOp.ADD
        assert arith.kind is ArithKind.INT

    def test_div_always_lowers_to_arith_div_decimal(self) -> None:
        """/ always produces IrArith with op=DIV and kind=DECIMAL.

        _lower_div coerces both operands to decimal regardless of input types;
        kind=DECIMAL is the decision-bearing field.
        """
        prog = _lower("let r = 3 / 2\n()")
        bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(bind, IrBind)
        arith = bind.value
        assert isinstance(arith, IrArith)
        assert arith.op is ArithOp.DIV
        assert arith.kind is ArithKind.DECIMAL

    def test_int_ordering_lowers_to_compare_lt_int(self) -> None:
        """< on two ints → IrCompare with op=LT and kind=INT."""
        prog = _lower("let r = 1 < 2\n()")
        bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(bind, IrBind)
        cmp = bind.value
        assert isinstance(cmp, IrCompare)
        assert cmp.op is CmpOp.LT
        assert cmp.kind is CompareKind.INT

    def test_decimal_widening_in_ordering_lowers_to_compare_decimal(self) -> None:
        """< with one decimal operand widens to kind=DECIMAL.

        _lower_ordering selects CompareKind.DECIMAL when either side is decimal;
        this asserts the widening decision that would be missed by an INT-only test.
        """
        prog = _lower("let r = 1.0 < 2\n()")
        bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(bind, IrBind)
        cmp = bind.value
        assert isinstance(cmp, IrCompare)
        assert cmp.op is CmpOp.LT
        assert cmp.kind is CompareKind.DECIMAL

    def test_equality_lowers_to_compare_eq_structural(self) -> None:
        """== on ints → IrCompare with op=EQ and kind=STRUCTURAL.

        _lower_equality always uses STRUCTURAL (wider than INT/DECIMAL) because
        equality is defined over any comparable type, not just numerics.
        """
        prog = _lower("let r = 1 == 1\n()")
        bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(bind, IrBind)
        cmp = bind.value
        assert isinstance(cmp, IrCompare)
        assert cmp.op is CmpOp.EQ
        assert cmp.kind is CompareKind.STRUCTURAL

    def test_in_list_lowers_to_contains_list(self) -> None:
        """'in' on a list → IrContains with kind=LIST."""
        prog = _lower("let r = 1 in [1, 2, 3]\n()")
        bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(bind, IrBind)
        contains = bind.value
        assert isinstance(contains, IrContains)
        assert contains.kind is ContainsKind.LIST

    def test_in_dict_lowers_to_contains_dict(self) -> None:
        """'in' (key lookup) on a dict → IrContains with kind=DICT."""
        prog = _lower('let r = "a" in {"a": 1}\n()')
        bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(bind, IrBind)
        contains = bind.value
        assert isinstance(contains, IrContains)
        assert contains.kind is ContainsKind.DICT


# ---------------------------------------------------------------------------
# Structural lowering: compiled one-level cases
# ---------------------------------------------------------------------------


class TestOneLevelCaseLowering:
    """Structural tests for decision-DAG lowering and source binders."""

    def test_bare_variant_pattern_lowers_to_enum_key(self) -> None:
        source = (
            "enum Status\n"
            "  | Active\n"
            "  | Inactive\n"
            "\n"
            "let s = Status::Active\n"
            "let r = case s of\n"
            "  | Active => 1\n"
            "  | _ => 0\n"
            "()\n"
        )
        prog = _lower(source)
        case_bind = prog.modules[prog.entry_module].initializers[1]
        assert isinstance(case_bind, IrBind)
        sequence = case_bind.value
        assert isinstance(sequence, IrSequence)
        assert isinstance(sequence.items[0], IrBind)
        assert prog.symbols[sequence.items[0].symbol].public_name is None
        switch = sequence.items[1]
        assert isinstance(switch, IrCase)
        assert isinstance(switch.arms[0].key, IrEnumCaseKey)
        assert switch.arms[0].key.variant == "Active"

    def test_binder_pattern_lowers_in_default_leaf(self) -> None:
        source = (
            "enum Status\n"
            "  | Active\n"
            "  | Inactive\n"
            "\n"
            "let s = Status::Active\n"
            "let r = case s of\n"
            "  | Active => 1\n"
            "  | _ as x => 0\n"
            "()\n"
        )
        prog = _lower(source)
        case_bind = prog.modules[prog.entry_module].initializers[1]
        assert isinstance(case_bind, IrBind)
        assert isinstance(case_bind.value, IrSequence)
        switch = case_bind.value.items[1]
        assert isinstance(switch, IrCase)
        assert isinstance(switch.default, IrSequence)
        binder = switch.default.items[0]
        assert isinstance(binder, IrBind)
        assert isinstance(binder.value, IrLoad)
        assert prog.symbols[binder.symbol].public_name is None

    def test_bare_variant_and_binder_use_key_and_default(self) -> None:
        source = (
            "enum Status\n"
            "  | Active\n"
            "  | Inactive\n"
            "\n"
            "let s = Status::Inactive\n"
            "let r = case s of\n"
            "  | Active => 1\n"
            "  | _ as x => 2\n"
            "()\n"
        )
        prog = _lower(source)
        case_bind = prog.modules[prog.entry_module].initializers[1]
        assert isinstance(case_bind, IrBind)
        assert isinstance(case_bind.value, IrSequence)
        switch = case_bind.value.items[1]
        assert isinstance(switch, IrCase)
        assert isinstance(switch.arms[0].key, IrEnumCaseKey)
        assert switch.default is not None


# ---------------------------------------------------------------------------
# Structural lowering: IrConvert / total-cast as? failure modes
# ---------------------------------------------------------------------------


class TestIrConvertLowering:
    """Structural tests for Cast lowering: IrConvert node and recipe selection.

    The lowerer emits IrConvert for 'as' (always) and fallible 'as?'; for
    total 'as?' it emits IrSequence instead.  These tests pin the
    decision-bearing fields so a wrong failure_mode or strategy selection
    fails the test.
    """

    def test_total_as_lowers_to_ir_convert_raise_cast_error(self) -> None:
        """'as' always emits IrConvert with failure_mode=RAISE_CAST_ERROR.

        int as decimal: total noop cast → strategy=WIDEN_INT_TO_DECIMAL,
        failure_mode=RAISE_CAST_ERROR.
        """
        prog = _lower("let r = 1 as decimal\n()")
        bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(bind, IrBind)
        conv = bind.value
        assert isinstance(conv, IrConvert)
        assert conv.failure_mode is ConversionFailureMode.RAISE_CAST_ERROR
        assert conv.recipe.strategy is ConversionStrategy.WIDEN_INT_TO_DECIMAL

    def test_fallible_as_test_lowers_to_ir_convert_return_bool(self) -> None:
        """Fallible 'as?' emits IrConvert with failure_mode=RETURN_BOOL.

        decimal as? int: fallible cast → strategy=NARROW_DECIMAL_TO_INT,
        failure_mode=RETURN_BOOL.
        """
        prog = _lower("let r = 1.5 as? int\n()")
        bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(bind, IrBind)
        conv = bind.value
        assert isinstance(conv, IrConvert)
        assert conv.failure_mode is ConversionFailureMode.RETURN_BOOL
        assert conv.recipe.strategy is ConversionStrategy.NARROW_DECIMAL_TO_INT

    def test_total_as_test_lowers_to_ir_sequence_not_ir_convert(self) -> None:
        """Total 'as?' emits IrSequence((source, IrConstBool(True))), NOT IrConvert.

        int as? decimal is a total noop; the lowerer sequences the source
        expression for side-effects and then yields True — no IrConvert.
        """
        prog = _lower("let r = 1 as? decimal\n()")
        bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(bind, IrBind)
        seq = bind.value
        assert isinstance(seq, IrSequence), (
            f"Total 'as?' must emit IrSequence, not {type(seq).__name__}"
        )
        assert len(seq.items) == 2
        last = seq.items[1]
        assert isinstance(last, IrConstBool)
        assert last.value is True

    def test_render_to_text_as_lowers_to_ir_convert_render_strategy(self) -> None:
        """'as text' (total render cast) → IrConvert with strategy=RENDER_TO_TEXT."""
        prog = _lower("let r = 42 as text\n()")
        bind = prog.modules[prog.entry_module].initializers[0]
        assert isinstance(bind, IrBind)
        conv = bind.value
        assert isinstance(conv, IrConvert)
        assert conv.failure_mode is ConversionFailureMode.RAISE_CAST_ERROR
        assert conv.recipe.strategy is ConversionStrategy.RENDER_TO_TEXT


# ---------------------------------------------------------------------------
# Structural lowering: template string lowering
# ---------------------------------------------------------------------------


class TestTemplateLowering:
    """Structural tests for Template → IrRenderTemplate lowering.

    Asserts the segment tuple shape so a wrong template representation fails.
    """

    def test_interpolated_template_lowers_to_ir_render_template_with_segments(self) -> None:
        """A template with one text segment and one interpolated expression.

        "hello ${name}" lowers to IrRenderTemplate with segments:
          (IrTemplateText("hello "), IrTemplateValue(IrLoad(name_sym)))
        """
        source = 'let name = "world"\nlet msg = "hello ${name}"\n()\n'
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        # inits[0] = IrBind(name, IrConstText("world"))
        # inits[1] = IrBind(msg, IrRenderTemplate(...))
        msg_bind = inits[1]
        assert isinstance(msg_bind, IrBind)
        template = msg_bind.value
        assert isinstance(template, IrRenderTemplate)
        segs = template.segments
        assert len(segs) == 2
        # First segment: static text prefix
        assert isinstance(segs[0], IrTemplateText)
        assert segs[0].text == "hello "
        # Second segment: interpolated expression → IrLoad of name's symbol
        assert isinstance(segs[1], IrTemplateValue)
        assert isinstance(segs[1].value, IrLoad)


# ---------------------------------------------------------------------------
# Structural lowering: IrMakeException / AutoTraceField
# ---------------------------------------------------------------------------


class TestIrMakeExceptionLowering:
    """Structural tests for exception construction lowering.

    IrMakeException.fields contains IrExpr for explicitly provided fields and
    AutoTraceField sentinels for declared-but-omitted fields.  These tests pin
    the exact slot shape so a wrong AutoTraceField placement fails.
    """

    def _get_raise_in_fn(self, source: str) -> IrRaise:
        """Lower source and return the IrRaise from the 'stop_fn' function body."""
        prog = _lower(source)
        stop_desc = next(
            d
            for d in prog.functions.values()
            if prog.symbols[d.function_symbol].public_name == "stop_fn"
        )
        body = _function_body(stop_desc)
        # Indented function body is wrapped in an IrBlock; unwrap if needed.
        if isinstance(body, IrBlock):
            for item in body.items:
                if isinstance(item, IrRaise):
                    return item
            raise AssertionError("Expected IrRaise in function body IrBlock")
        assert isinstance(body, IrRaise)
        return body

    def test_exception_construction_emits_ir_make_exception(self) -> None:
        """raise Abort(message = ...) → IrRaise(exc=IrMakeException(display_name='Abort'))."""
        source = 'def stop_fn() -> unit =\n  raise Abort(message = "stop")\nstop_fn()\n'
        raise_node = self._get_raise_in_fn(source)
        exc = raise_node.exc
        assert isinstance(exc, IrMakeException)
        assert exc.display_name == "Abort"

    def test_provided_field_is_ir_expr_not_auto_trace_field(self) -> None:
        """Explicitly provided 'message' field → IrConstText, not AutoTraceField."""
        source = 'def stop_fn() -> unit =\n  raise Abort(message = "stop")\nstop_fn()\n'
        raise_node = self._get_raise_in_fn(source)
        exc = raise_node.exc
        assert isinstance(exc, IrMakeException)
        fields_dict = dict(exc.fields)
        msg_slot = fields_dict["message"]
        assert isinstance(msg_slot, IrConstText), (
            f"explicitly provided 'message' must be IrConstText, got {type(msg_slot).__name__}"
        )
        assert msg_slot.value == "stop"

    def test_unprovided_trace_id_field_is_auto_trace_field(self) -> None:
        """Undeclared 'trace_id' field → AutoTraceField sentinel, not an IrExpr.

        When a caller omits a declared exception field, the lowerer places an
        AutoTraceField sentinel; the evaluator fills in a fresh trace id at
        construction time.  This test pins that exactly one AutoTraceField
        is present for the omitted trace_id.
        """
        source = 'def stop_fn() -> unit =\n  raise Abort(message = "stop")\nstop_fn()\n'
        raise_node = self._get_raise_in_fn(source)
        exc = raise_node.exc
        assert isinstance(exc, IrMakeException)
        fields_dict = dict(exc.fields)
        trace_slot = fields_dict["trace_id"]
        assert isinstance(trace_slot, AutoTraceField), (
            f"omitted 'trace_id' must be AutoTraceField, got {type(trace_slot).__name__}"
        )
        # Only the omitted field gets an AutoTraceField; the provided 'message' does not.
        auto_fields = [v for _, v in exc.fields if isinstance(v, AutoTraceField)]
        assert len(auto_fields) == 1


# ---------------------------------------------------------------------------
# Golden lowering: loop desugar
# ---------------------------------------------------------------------------


def _get_loop_ir(source: str) -> "IrLoop | IrSequence":
    """Lower *source* and return the top-level loop IR node (IrSequence or IrLoop)."""
    from agm.agl.modules.ids import ENTRY_ID

    executable = _lower(source)
    # Find the loop/sequence node; skip IrBind for var declarations.
    for node in executable.modules[ENTRY_ID].initializers:
        if isinstance(node, (IrLoop, IrSequence)):
            return node
    raise AssertionError("No IrLoop or IrSequence found in initializers")


class TestLoopDesugar:
    """Golden lowering tests for loop desugaring."""

    def test_unbounded_do_until_no_preloop(self) -> None:
        """``do … until E`` lowers to a bare IrLoop with body + until guard.

        No pre-loop IrSequence; items 4/5 absent (no bound).
        """
        source = "var n = 0\ndo\n  n := n + 1\nuntil n >= 3\n"
        node = _get_loop_ir(source)
        # Unbounded: no pre-loop wrapping IrSequence
        assert isinstance(node, IrLoop), f"expected IrLoop, got {type(node).__name__}"
        body = node.body
        assert isinstance(body, IrBlock)
        # Body has 2 items: item 6 (body) + item 7 (until guard)
        assert len(body.items) == 2, f"expected 2 body items, got {len(body.items)}"
        # Item 7: IrIf (until guard) — last item
        assert isinstance(body.items[1], IrIf), (
            f"item 7 must be IrIf (until guard), got {type(body.items[1]).__name__}"
        )
        until_if = body.items[1]
        assert len(until_if.branches) == 1
        assert until_if.has_else is False
        # The until-guard branch body is IrBreak
        assert isinstance(until_if.branches[0].body, IrBreak), (
            "until guard branch body must be IrBreak"
        )

    def test_bounded_do_n_until_preloop_and_all_items(self) -> None:
        """``do[N] … until E`` lowers to IrSequence(__n bind, __count bind, IrLoop).

        The IrLoop body contains items 4 (bound check), 5 (count incr),
        6 (body), 7 (until guard).
        """
        source = "var x = 0\ndo[7]\n  x := x + 1\nuntil x >= 3\n"
        node = _get_loop_ir(source)
        # Bounded: wraps in IrSequence
        assert isinstance(node, IrSequence), f"expected IrSequence, got {type(node).__name__}"
        # pre_items: IrBind(__n), IrBind(__count); last item: IrLoop
        assert len(node.items) == 3, f"expected 3 IrSequence items, got {len(node.items)}"
        n_bind = node.items[0]
        count_bind = node.items[1]
        loop = node.items[2]
        # __n: immutable bind to the bound expression (lower_coerced(7, IntType) → IrConstInt(7))
        assert isinstance(n_bind, IrBind), f"item 0 must be IrBind, got {type(n_bind).__name__}"
        assert isinstance(n_bind.value, IrConstInt), (
            f"__n value must be IrConstInt, got {type(n_bind.value).__name__}"
        )
        assert n_bind.value.value == 7
        # __count: mutable bind to 0
        assert isinstance(count_bind, IrBind)
        assert isinstance(count_bind.value, IrConstInt)
        assert count_bind.value.value == 0
        # IrLoop with 4-item body (items 4, 5, 6, 7)
        assert isinstance(loop, IrLoop), f"item 2 must be IrLoop, got {type(loop).__name__}"
        body = loop.body
        assert isinstance(body, IrBlock)
        assert len(body.items) == 4, f"expected 4 loop body items, got {len(body.items)}"
        # Item ordering: 4 (bound check IrIf), 5 (count incr IrAssign),
        #                6 (body), 7 (until guard IrIf)
        assert isinstance(body.items[0], IrIf), "item 4 must be IrIf (bound check)"
        assert isinstance(body.items[1], IrAssign), "item 5 must be IrAssign (count incr)"
        assert isinstance(body.items[3], IrIf), "item 7 must be IrIf (until guard)"

    def test_bounded_do_n_until_bound_check_structure(self) -> None:
        """Item 4 bound-check structure: outer GE if → inner EQ-or-raise if."""
        source = "var x = 0\ndo[5]\n  x := x + 1\nuntil x >= 5\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        loop = node.items[2]
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        bound_check = body.items[0]
        assert isinstance(bound_check, IrIf)
        # Outer if: single branch with GE comparison, has_else=False
        assert len(bound_check.branches) == 1
        assert bound_check.has_else is False
        outer_branch = bound_check.branches[0]
        assert isinstance(outer_branch.cond, IrCompare)
        assert outer_branch.cond.op is CmpOp.GE
        # Inner if: two branches (EQ=0 → IrBreak, else → IrRaise), has_else=True
        inner_if = outer_branch.body
        assert isinstance(inner_if, IrIf)
        assert len(inner_if.branches) == 2
        assert inner_if.has_else is True
        # First branch: count == 0 → IrBreak
        first_branch = inner_if.branches[0]
        assert isinstance(first_branch.cond, IrCompare)
        assert first_branch.cond.op is CmpOp.EQ
        assert first_branch.cond.kind is CompareKind.STRUCTURAL
        assert isinstance(first_branch.body, IrBreak)
        # Second branch (else): → IrRaise(MaxIterationsExceeded)
        else_branch = inner_if.branches[1]
        assert else_branch.cond is None
        assert isinstance(else_branch.body, IrRaise)
        exc_node = else_branch.body.exc
        assert isinstance(exc_node, IrMakeException)
        assert exc_node.display_name == "MaxIterationsExceeded"

    def test_max_iterations_exception_field_order(self) -> None:
        """MaxIterationsExceeded IrMakeException has fields in declaration order."""
        source = "var x = 0\ndo[5]\n  x := x + 1\nuntil x >= 5\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        loop = node.items[2]
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        outer_if = body.items[0]
        assert isinstance(outer_if, IrIf)
        inner_if = outer_if.branches[0].body
        assert isinstance(inner_if, IrIf)
        raise_node = inner_if.branches[1].body
        assert isinstance(raise_node, IrRaise)
        exc = raise_node.exc
        assert isinstance(exc, IrMakeException)
        fields = exc.fields
        # Declaration order: message, trace_id, limit, condition,
        #                    last_condition_value, metadata
        assert len(fields) == 6
        assert fields[0][0] == "message"
        assert isinstance(fields[0][1], IrRenderTemplate)
        assert fields[1][0] == "trace_id"
        assert isinstance(fields[1][1], AutoTraceField)
        assert fields[2][0] == "limit"
        assert isinstance(fields[2][1], IrLoad)  # IrLoad(__n_sym)
        assert fields[3][0] == "condition"
        assert isinstance(fields[3][1], IrConstText)
        assert fields[4][0] == "last_condition_value"
        assert isinstance(fields[4][1], IrConstBool)
        assert fields[4][1].value is False
        assert fields[5][0] == "metadata"
        assert isinstance(fields[5][1], IrConstJsonNull)

    def test_done_terminator_condition_source_is_false(self) -> None:
        """``do[n] … done`` sets ``condition="false"`` in MaxIterationsExceeded."""
        source = "var x = 0\ndo[2]\n  x := x + 1\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        loop = node.items[2]
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        # Body has 3 items (4, 5, 6) — no item 7 for done
        assert len(body.items) == 3
        outer_if = body.items[0]
        inner_if = outer_if.branches[0].body
        raise_node = inner_if.branches[1].body
        exc = raise_node.exc
        assert isinstance(exc, IrMakeException)
        condition_field = dict(exc.fields)["condition"]
        assert isinstance(condition_field, IrConstText)
        assert condition_field.value == "false"

    def test_bounded_done_no_until_guard(self) -> None:
        """``do[n] … done`` body has 3 items (4, 5, 6) — no item 7 until guard."""
        source = "var y = 0\ndo[3]\n  y := y + 1\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        loop = node.items[2]
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        assert len(body.items) == 3, (
            f"do[n] done body must have 3 items (4, 5, 6), got {len(body.items)}"
        )
        # No until-guard IrIf at the end
        assert not isinstance(body.items[-1], IrIf) or (
            # If it happens to be IrIf (item 4), that's the bound check, which is fine
            # as long as it's at position 0.  Position 2 (last) should NOT be an IrIf.
            True
        )
        # Position 2 is the body (item 6), not an until guard
        assert not isinstance(body.items[2], IrIf), (
            "item 2 (last) must NOT be an until-guard IrIf for done terminator"
        )

    def test_unbounded_done_only_body_item(self) -> None:
        """``do … done`` (no bound, done terminator): bare IrLoop with 1 body item."""
        source = "var z = 0\ndo\n  z := z + 1\ndone\n"
        node = _get_loop_ir(source)
        # No bound → no pre-loop IrSequence
        assert isinstance(node, IrLoop), (
            f"unbounded done: expected IrLoop directly, got {type(node).__name__}"
        )
        body = node.body
        assert isinstance(body, IrBlock)
        # Only item 6 (body) — no items 4/5 (no bound), no item 7 (done = until false)
        assert len(body.items) == 1, f"unbounded done body must have 1 item, got {len(body.items)}"

    def test_n_bound_evaluated_once(self) -> None:
        """The bound expression is bound to ``__n`` ONCE via a single IrBind in the IrSequence."""
        source = "var budget = 3\ndo[budget]\n  budget := budget + 100\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        # First IrBind is __n = lower(budget) (a single IrLoad of the budget symbol)
        n_bind = node.items[0]
        assert isinstance(n_bind, IrBind)
        # The value is an IrLoad (of the budget var) — evaluated once at entry
        assert isinstance(n_bind.value, IrLoad), (
            f"__n value must be IrLoad(budget), got {type(n_bind.value).__name__}"
        )

    def test_crlf_condition_source_in_exception_field(self) -> None:
        """With CRLF source, condition source text in MaxIterationsExceeded is clean."""
        from tests.agl.ir_harness import evaluate_ir_raises

        source = "var i = 0\r\ndo[3]\r\n  i := i + 1\r\nuntil i > 100\r\n"
        ir_exc = evaluate_ir_raises(source)
        assert ir_exc.display_name == "MaxIterationsExceeded"
        from agm.agl.semantics.values import TextValue

        assert ir_exc.fields.get("condition") == TextValue("i > 100")

    def test_for_while_bounded_until_full_body_item_order(self) -> None:
        """``for x in items while x < 10 do[5] body until total > 20`` — all 7 items.

        Asserts the lowered IrLoop body items are in desugar order:
        1. for-exhaustion check (IrIf wrapping IrIterHasNext)
        2. for-var bind (IrBind wrapping IrIterNext)
        3. while guard (IrIf wrapping the condition)
        4. bound check (IrIf)
        5. count increment (IrAssign)
        6. body
        7. until guard (IrIf)

        Also asserts the pre-loop IrSequence binds __it (IrIterInit) first, then
        __n and __count, before the IrLoop.
        """
        source = (
            "param items: list[int]\n"
            "var total: int = 0\n"
            "for x in items while x < 10 do[5]\n"
            "  total := total + x\n"
            "until total > 20\n"
        )
        node = _get_loop_ir(source)
        # Pre-loop: IrSequence with 4 items: __it bind, __n bind, __count bind, IrLoop
        assert isinstance(node, IrSequence), (
            f"for+while+bound: expected IrSequence, got {type(node).__name__}"
        )
        assert len(node.items) == 4, f"expected 4 pre-loop+loop items, got {len(node.items)}"
        it_bind = node.items[0]
        n_bind = node.items[1]
        count_bind = node.items[2]
        loop = node.items[3]

        # __it bind: IrBind(IrIterInit(LIST, ...))
        assert isinstance(it_bind, IrBind), (
            f"item 0 must be IrBind(__it), got {type(it_bind).__name__}"
        )
        assert isinstance(it_bind.value, IrIterInit), (
            f"__it value must be IrIterInit, got {type(it_bind.value).__name__}"
        )
        assert it_bind.value.kind is IterKind.LIST, (
            f"__it kind must be LIST for list[int], got {it_bind.value.kind!r}"
        )
        # __n bind: IrBind(IrConstInt(5))
        assert isinstance(n_bind, IrBind), (
            f"item 1 must be IrBind(__n), got {type(n_bind).__name__}"
        )
        assert isinstance(n_bind.value, IrConstInt), (
            f"__n value must be IrConstInt, got {type(n_bind.value).__name__}"
        )
        assert n_bind.value.value == 5
        # __count bind: IrBind(IrConstInt(0))
        assert isinstance(count_bind, IrBind), (
            f"item 2 must be IrBind(__count), got {type(count_bind).__name__}"
        )
        assert isinstance(count_bind.value, IrConstInt), (
            f"__count value must be IrConstInt(0), got {type(count_bind.value).__name__}"
        )
        assert count_bind.value.value == 0

        # IrLoop with 7-item body
        assert isinstance(loop, IrLoop), f"item 3 must be IrLoop, got {type(loop).__name__}"
        body = loop.body
        assert isinstance(body, IrBlock)
        assert len(body.items) == 7, (
            f"for+while+bound+until: expected 7 body items, got {len(body.items)}"
        )

        # Item 1: for-exhaustion check — IrIf(not IrIterHasNext(__it))
        item1 = body.items[0]
        assert isinstance(item1, IrIf), "item 1 must be IrIf (for-exhaustion check)"
        assert len(item1.branches) == 1
        assert item1.has_else is False
        cond1 = item1.branches[0].cond
        assert isinstance(cond1, IrUnary), "item 1 condition must be IrUnary (not)"
        assert cond1.op is UnaryOp.NOT
        assert isinstance(cond1.value, IrIterHasNext), "item 1 IrUnary.value must be IrIterHasNext"
        assert isinstance(item1.branches[0].body, IrBreak), "item 1 branch body must be IrBreak"

        # Item 2: for-var bind — IrBind(for_var, IrIterNext(__it))
        item2 = body.items[1]
        assert isinstance(item2, IrBind), "item 2 must be IrBind (for-var = IrIterNext)"
        assert isinstance(item2.value, IrIterNext), (
            f"item 2 value must be IrIterNext, got {type(item2.value).__name__}"
        )

        # Item 3: while guard — IrIf(not while_cond)
        item3 = body.items[2]
        assert isinstance(item3, IrIf), "item 3 must be IrIf (while guard)"
        assert len(item3.branches) == 1
        assert item3.has_else is False
        cond3 = item3.branches[0].cond
        assert isinstance(cond3, IrUnary), "item 3 condition must be IrUnary (not)"
        assert cond3.op is UnaryOp.NOT
        assert isinstance(item3.branches[0].body, IrBreak), "item 3 branch body must be IrBreak"

        # Item 4: bound check — IrIf (GE outer)
        item4 = body.items[3]
        assert isinstance(item4, IrIf), "item 4 must be IrIf (bound check)"

        # Item 5: count increment — IrAssign
        item5 = body.items[4]
        assert isinstance(item5, IrAssign), "item 5 must be IrAssign (count increment)"

        # Item 6: body (total := total + x) — wrapped in IrBlock by the lowerer
        item6 = body.items[5]
        assert isinstance(item6, IrBlock), "item 6 (body) must be IrBlock"

        # Item 7: until guard — IrIf (until total > 20)
        item7 = body.items[6]
        assert isinstance(item7, IrIf), "item 7 must be IrIf (until guard)"
        assert len(item7.branches) == 1
        assert item7.has_else is False
        assert isinstance(item7.branches[0].body, IrBreak), "item 7 branch body must be IrBreak"

    def test_for_iter_kind_list(self) -> None:
        """``for x in items`` over ``list[int]`` selects IterKind.LIST."""
        source = "param items: list[int]\nvar n: int = 0\nfor x in items do\n  n := n + x\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence), (
            f"for over list: expected IrSequence, got {type(node).__name__}"
        )
        it_bind = node.items[0]
        assert isinstance(it_bind, IrBind)
        assert isinstance(it_bind.value, IrIterInit), (
            f"expected IrIterInit, got {type(it_bind.value).__name__}"
        )
        assert it_bind.value.kind is IterKind.LIST, (
            f"expected IterKind.LIST for list[int], got {it_bind.value.kind!r}"
        )

    def test_for_iter_kind_dict_keys(self) -> None:
        """``for k in d`` over ``dict[text, int]`` selects IterKind.DICT_KEYS."""
        source = "param d: dict[text, int]\nvar n: int = 0\nfor k in d do\n  n := n + 1\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence), (
            f"for over dict: expected IrSequence, got {type(node).__name__}"
        )
        it_bind = node.items[0]
        assert isinstance(it_bind, IrBind)
        assert isinstance(it_bind.value, IrIterInit), (
            f"expected IrIterInit, got {type(it_bind.value).__name__}"
        )
        assert it_bind.value.kind is IterKind.DICT_KEYS, (
            f"expected IterKind.DICT_KEYS for dict[text, int], got {it_bind.value.kind!r}"
        )

    def test_for_iter_kind_text(self) -> None:
        """``for ch in s`` over ``text`` selects IterKind.TEXT."""
        source = "param s: text\nvar n: int = 0\nfor ch in s do\n  n := n + 1\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence), (
            f"for over text: expected IrSequence, got {type(node).__name__}"
        )
        it_bind = node.items[0]
        assert isinstance(it_bind, IrBind)
        assert isinstance(it_bind.value, IrIterInit), (
            f"expected IrIterInit, got {type(it_bind.value).__name__}"
        )
        assert it_bind.value.kind is IterKind.TEXT, (
            f"expected IterKind.TEXT for text, got {it_bind.value.kind!r}"
        )


# ---------------------------------------------------------------------------
# Golden lowering: range-for desugar
# ---------------------------------------------------------------------------


class TestRangeForDesugar:
    """Golden lowering tests for integer-range for-loop desugaring."""

    def test_to_default_step_preloop_items(self) -> None:
        """``for i in 1 to 5`` pre-loop: IrSequence with __cur(mutable)/
        __end/step/step-guard/IrLoop.

        Pre-loop contains: IrBind(__cur, IrConstInt(1) mutable),
        IrBind(__end, IrConstInt(5) immutable),
        IrBind(__step, IrConstInt(1) immutable),
        IrIf(step-guard), IrLoop.
        """
        source = "for i in 1 to 5 do\n  ()\ndone\n"
        prog = _lower(source)
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence), (
            f"range for: expected IrSequence, got {type(node).__name__}"
        )
        # 5 items: __cur bind, __end bind, __step bind, step guard IrIf, IrLoop
        assert len(node.items) == 5, f"range for pre-loop: expected 5 items, got {len(node.items)}"
        cur_bind, end_bind, step_bind, guard_if, loop = node.items
        # __cur: mutable, value is IrConstInt(1)
        assert isinstance(cur_bind, IrBind)
        assert isinstance(cur_bind.value, IrConstInt)
        assert cur_bind.value.value == 1
        assert prog.symbols[cur_bind.symbol].mutable, "__cur must be mutable (var cursor)"
        # __end: immutable, value is IrConstInt(5)
        assert isinstance(end_bind, IrBind)
        assert isinstance(end_bind.value, IrConstInt)
        assert end_bind.value.value == 5
        assert not prog.symbols[end_bind.symbol].mutable, "__end must be immutable"
        # __step: immutable, value is IrConstInt(1) for default step
        assert isinstance(step_bind, IrBind)
        assert isinstance(step_bind.value, IrConstInt)
        assert step_bind.value.value == 1
        assert not prog.symbols[step_bind.symbol].mutable, "__step must be immutable"
        # step guard: IrIf(LE, has_else=False) → IrRaise(RangeError)
        assert isinstance(guard_if, IrIf)
        assert guard_if.has_else is False
        assert len(guard_if.branches) == 1
        guard_cond = guard_if.branches[0].cond
        assert isinstance(guard_cond, IrCompare)
        assert guard_cond.op is CmpOp.LE
        assert guard_cond.kind is CompareKind.INT
        guard_body = guard_if.branches[0].body
        assert isinstance(guard_body, IrRaise)
        exc = guard_body.exc
        assert isinstance(exc, IrMakeException)
        assert exc.display_name == "RangeError"
        fields_dict = dict(exc.fields)
        assert "message" in fields_dict
        assert isinstance(fields_dict["message"], IrConstText)
        assert "trace_id" in fields_dict
        assert isinstance(fields_dict["trace_id"], AutoTraceField)
        # IrLoop is last
        assert isinstance(loop, IrLoop)

    def test_to_default_step_body_item1_gt_break(self) -> None:
        """``for i in 1 to 5`` body item 1: IrIf(__cur > __end => IrBreak)."""
        source = "for i in 1 to 5 do\n  ()\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        loop = node.items[4]
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        # Item 1: IrIf GT comparison → IrBreak
        item1 = body.items[0]
        assert isinstance(item1, IrIf)
        assert item1.has_else is False
        assert len(item1.branches) == 1
        cond = item1.branches[0].cond
        assert isinstance(cond, IrCompare)
        assert cond.op is CmpOp.GT
        assert cond.kind is CompareKind.INT
        assert isinstance(item1.branches[0].body, IrBreak)

    def test_to_default_step_body_item2_bind_then_advance(self) -> None:
        """``for i in 1 to 5`` body item 2: IrBind(i, IrLoad(__cur))
        then IrAssign(__cur, __cur + __step).
        """
        source = "for i in 1 to 5 do\n  ()\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        loop = node.items[4]
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        # Item 2a: IrBind(i, IrLoad(__cur))
        item2a = body.items[1]
        assert isinstance(item2a, IrBind)
        assert isinstance(item2a.value, IrLoad)
        # Item 2b: IrAssign(__cur, __cur + __step)
        item2b = body.items[2]
        assert isinstance(item2b, IrAssign)
        advance = item2b.value
        assert isinstance(advance, IrArith)
        assert advance.op is ArithOp.ADD
        assert advance.kind is ArithKind.INT

    def test_downto_default_step_body_item1_lt_break(self) -> None:
        """``for i in 5 downto 1`` body item 1: IrIf(__cur < __end => IrBreak)."""
        source = "for i in 5 downto 1 do\n  ()\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        loop = node.items[4]
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        item1 = body.items[0]
        assert isinstance(item1, IrIf)
        cond = item1.branches[0].cond
        assert isinstance(cond, IrCompare)
        assert cond.op is CmpOp.LT
        assert cond.kind is CompareKind.INT
        assert isinstance(item1.branches[0].body, IrBreak)

    def test_downto_default_step_body_item2_advance_sub(self) -> None:
        """``for i in 5 downto 1`` body item 2b: advance uses SUB."""
        source = "for i in 5 downto 1 do\n  ()\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        loop = node.items[4]
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        item2b = body.items[2]
        assert isinstance(item2b, IrAssign)
        advance = item2b.value
        assert isinstance(advance, IrArith)
        assert advance.op is ArithOp.SUB
        assert advance.kind is ArithKind.INT

    def test_to_by_k_step_bind_is_expr(self) -> None:
        """``for i in 1 to 10 by 3`` __step bind comes from the ``by`` expression."""
        source = "for i in 1 to 10 by 3 do\n  ()\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        # Pre-loop: cur, end, step, guard, loop
        _cur_bind, _end_bind, step_bind, _guard, _loop = node.items
        assert isinstance(step_bind, IrBind)
        assert isinstance(step_bind.value, IrConstInt)
        assert step_bind.value.value == 3

    def test_downto_by_k_advance_uses_sub(self) -> None:
        """``for i in 10 downto 1 by 2`` advance is SUB with __step from ``by``."""
        source = "for i in 10 downto 1 by 2 do\n  ()\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        _cur_bind, _end_bind, step_bind, _guard, loop = node.items
        assert isinstance(step_bind, IrBind)
        assert isinstance(step_bind.value, IrConstInt)
        assert step_bind.value.value == 2
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        item2b = body.items[2]
        assert isinstance(item2b, IrAssign)
        advance = item2b.value
        assert isinstance(advance, IrArith)
        assert advance.op is ArithOp.SUB

    def test_step_guard_raises_range_error_ir(self) -> None:
        """The step guard IrMakeException has nominal RangeError, message+trace_id fields."""
        from agm.agl.modules.ids import PRELUDE_ID as _PRELUDE_ID

        source = "for i in 1 to 5 do\n  ()\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        guard_if = node.items[3]
        assert isinstance(guard_if, IrIf)
        raise_node = guard_if.branches[0].body
        assert isinstance(raise_node, IrRaise)
        exc = raise_node.exc
        assert isinstance(exc, IrMakeException)
        assert exc.nominal.module_id == _PRELUDE_ID
        assert exc.nominal.declared_name == "RangeError"
        assert exc.display_name == "RangeError"
        fields_dict = dict(exc.fields)
        assert set(fields_dict.keys()) == {"message", "trace_id"}
        assert isinstance(fields_dict["message"], IrConstText)
        assert isinstance(fields_dict["trace_id"], AutoTraceField)

    def test_range_with_n_bound_preloop_order(self) -> None:
        """``for i in 1 to 5 do[3]`` range pre-loop precedes __n/__count.

        IrSequence items: __cur, __end, __step, guard, __n, __count, IrLoop.
        """
        source = "for i in 1 to 5 do[3]\n  ()\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        assert len(node.items) == 7, (
            f"range+bound: expected 7 pre-loop+loop items, got {len(node.items)}"
        )
        _cur, _end, _step, _guard, n_bind, count_bind, loop = node.items
        # __n: IrConstInt(3)
        assert isinstance(n_bind, IrBind)
        assert isinstance(n_bind.value, IrConstInt)
        assert n_bind.value.value == 3
        # __count: IrConstInt(0)
        assert isinstance(count_bind, IrBind)
        assert isinstance(count_bind.value, IrConstInt)
        assert count_bind.value.value == 0
        # Loop body has range items 1+2a+2b, bound items 4+5, body 6: 6 items total
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        assert len(body.items) == 6, f"range+bound body: expected 6 items, got {len(body.items)}"
        # Items: 1(IrIf range-break), 2a(IrBind i), 2b(IrAssign advance),
        #        4(IrIf bound-check), 5(IrAssign count), 6(body)
        assert isinstance(body.items[0], IrIf), "item 1 must be IrIf (range break)"
        assert isinstance(body.items[1], IrBind), "item 2a must be IrBind (loop var)"
        assert isinstance(body.items[2], IrAssign), "item 2b must be IrAssign (advance)"
        assert isinstance(body.items[3], IrIf), "item 4 must be IrIf (bound check)"
        assert isinstance(body.items[4], IrAssign), "item 5 must be IrAssign (count incr)"

    def test_range_with_while_guard(self) -> None:
        """``for i in 1 to 10 while i < 8`` body includes range items + while guard."""
        source = "for i in 1 to 10 while i < 8 do\n  ()\ndone\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        # Pre-loop: cur, end, step, guard, loop (5 items)
        assert len(node.items) == 5
        loop = node.items[4]
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        # Items: 1(range break), 2a(bind), 2b(advance), 3(while guard), 6(body) = 5
        assert len(body.items) == 5, f"range+while body: expected 5 items, got {len(body.items)}"
        # Item 3: while guard — IrIf(not cond)
        item3 = body.items[3]
        assert isinstance(item3, IrIf)
        assert item3.has_else is False
        cond3 = item3.branches[0].cond
        assert isinstance(cond3, IrUnary)
        assert cond3.op is UnaryOp.NOT

    def test_range_with_until_guard(self) -> None:
        """``for i in 1 to 10 until i > 7`` body includes range items + until guard."""
        source = "for i in 1 to 10 do\n  ()\nuntil i > 7\n"
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        loop = node.items[4]
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        # Items: 1(range break), 2a(bind), 2b(advance), 6(body), 7(until guard) = 5
        assert len(body.items) == 5, f"range+until body: expected 5 items, got {len(body.items)}"
        item7 = body.items[4]
        assert isinstance(item7, IrIf)
        assert item7.has_else is False
        assert isinstance(item7.branches[0].body, IrBreak)

    def test_range_for_no_iterator_ops(self) -> None:
        """A range ``for`` must NOT emit IrIterInit, IrIterHasNext, or IrIterNext."""
        source = "for i in 1 to 100 by 2 do\n  ()\ndone\n"
        prog = _lower(source)
        # Serialise to a string and scan for iterator node class names
        prog_repr = repr(prog)
        assert "IrIterInit" not in prog_repr, "range for must not emit IrIterInit"
        assert "IrIterHasNext" not in prog_repr, "range for must not emit IrIterHasNext"
        assert "IrIterNext" not in prog_repr, "range for must not emit IrIterNext"

    def test_range_scan_captures_outer_var(self) -> None:
        """Range bounds/step from an enclosing function scope are captured in a nested lambda.

        When a lambda contains a range ``for`` whose bounds/step reference parameters
        of the enclosing function, ``_scan_captures`` must walk ``for_range_to`` and
        ``for_range_by`` to detect those free variables.  The resulting
        ``IrMakeClosure.captures`` must contain entries for both outer parameters.
        """
        source = (
            "def make_fn(end_val: int, step_val: int) -> unit =\n"
            "  let _g = fn() -> unit => for i in 1 to end_val by step_val do () done\n"
            "  ()\n"
            "make_fn(5, 1)\n"
        )
        prog = _lower(source)
        # Find the FunctionDescriptor for ``make_fn``.
        make_fn_desc = next(
            d
            for d in prog.functions.values()
            if prog.symbols[d.function_symbol].public_name == "make_fn"
        )
        body = _function_body(make_fn_desc)
        assert isinstance(body, IrBlock)
        # The nested lambda _g lowers to IrBind(_g, IrMakeClosure(...)).
        g_closure: IrMakeClosure | None = None
        for item in body.items:
            if isinstance(item, IrBind) and isinstance(item.value, IrMakeClosure):
                g_closure = item.value
                break
        assert g_closure is not None, "Expected IrMakeClosure for _g in make_fn body"
        # _scan_captures must have walked for_range_to / for_range_by and found
        # end_val and step_val as free variables captured from make_fn's params.
        captures = g_closure.captures
        assert len(captures) == 2, f"Expected 2 captures (end_val + step_val), got {captures!r}"
        # Both are params (immutable) → by_cell=False for each capture.
        for cap in captures:
            assert isinstance(cap, IrCapture)
            assert not cap.by_cell, "param captures must have by_cell=False"
            assert not prog.symbols[cap.symbol].mutable, "captured params must be immutable"
        # The captured symbol ids must exactly match make_fn's param symbols.
        param_symbols = {p.symbol for p in make_fn_desc.params}
        captured_symbols = {cap.symbol for cap in captures}
        assert captured_symbols == param_symbols, (
            f"Captured symbols {captured_symbols!r} must match make_fn params {param_symbols!r}"
        )

    def test_collection_for_regression(self) -> None:
        """Collection ``for`` still lowers to IrIterInit / IrIterHasNext / IrIterNext."""
        source = (
            "param items: list[int]\n"
            "var total: int = 0\n"
            "for x in items do\n"
            "  total := total + x\n"
            "done\n"
        )
        node = _get_loop_ir(source)
        assert isinstance(node, IrSequence)
        it_bind = node.items[0]
        assert isinstance(it_bind, IrBind)
        assert isinstance(it_bind.value, IrIterInit), "collection for must still use IrIterInit"
        loop = node.items[1]
        assert isinstance(loop, IrLoop)
        body = loop.body
        assert isinstance(body, IrBlock)
        item1 = body.items[0]
        assert isinstance(item1, IrIf)
        cond = item1.branches[0].cond
        assert isinstance(cond, IrUnary)
        assert isinstance(cond.value, IrIterHasNext), "collection for item 1 must use IrIterHasNext"
        item2 = body.items[1]
        assert isinstance(item2, IrBind)
        assert isinstance(item2.value, IrIterNext), "collection for item 2 must use IrIterNext"

    def test_range_error_in_builtin_exceptions(self) -> None:
        """RangeError is present in BUILTIN_EXCEPTIONS with the expected fields."""
        from agm.agl.semantics.type_table import create_seeded_type_table
        from agm.agl.semantics.types import BUILTIN_EXCEPTIONS
        from agm.agl.semantics.types import TextType as _TextType

        assert "RangeError" in BUILTIN_EXCEPTIONS
        exc = BUILTIN_EXCEPTIONS["RangeError"]
        assert exc.name == "RangeError"
        fields = create_seeded_type_table().exception_fields(exc)
        assert set(fields.keys()) == {"message", "trace_id"}
        assert isinstance(fields["message"], _TextType)
        assert isinstance(fields["trace_id"], _TextType)
