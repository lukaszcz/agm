"""Tests for the AgL lowering phase (M2-A).

Covers:
- ``compile_coercion`` — every branch of the coercion compiler.
- ``lower_program`` — lowering of the M2 node subset, including coercion
  insertion, binding/assignment lowering, and validate_ir pass.

Pipeline helper: reuses the ``parse_resolve_check`` pattern from
``tests/test_agl_typecheck.py`` to obtain a ``CheckedProgram`` from source.
"""

from __future__ import annotations

import decimal
from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.nodes import (
    IrAssign,
    IrBind,
    IrBlock,
    IrCoerce,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstText,
    IrConstUnit,
    IrIndirectCall,
    IrLoad,
    IrMakeClosure,
    IrMakeDict,
    IrMakeList,
)
from agm.agl.ir.operations import (
    IntToDecimal,
    MapDictValues,
    MapEnumFields,
    MapList,
    MapRecordFields,
    ToJson,
)
from agm.agl.ir.program import ExecutableProgram
from agm.agl.ir.validate import validate_ir
from agm.agl.lower import compile_coercion, lower_program
from agm.agl.lower.lowerer import _Lowerer
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.typecheck import check
from agm.agl.typecheck.env import CheckedProgram
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
    TypeVarType,
    UnitType,
)

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
            "json": frozenset(
                {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
            ),
        },
    )


def _check(source: str) -> CheckedProgram:
    prog = parse_program(source)
    resolved = resolve(prog)
    return check(resolved, _caps())


def _lower(source: str, *, validate: bool = True) -> ExecutableProgram:
    """Parse → check → lower the source; return ExecutableProgram."""
    checked = _check(source)
    return lower_program(
        checked,
        source_text=source,
        source_label="<test>",
        validate=validate,
    )


def _make_lowerer(checked: CheckedProgram, source: str) -> "_Lowerer":
    """Create a _Lowerer with a fresh _LinkState for unit tests."""
    from agm.agl._text import normalize_newlines
    from agm.agl.ir.ids import SourceId
    from agm.agl.ir.program import SourceFile
    from agm.agl.lower.lowerer import _LinkState, _Lowerer
    from agm.agl.modules.ids import ENTRY_ID
    link = _LinkState()
    source_id = SourceId(link.next_source)
    link.next_source += 1
    normalized = normalize_newlines(source)
    link.sources[source_id] = SourceFile(display_name="<test>", normalized_text=normalized)
    return _Lowerer(checked, link, ENTRY_ID, source_id, source)

# ---------------------------------------------------------------------------
# compile_coercion unit tests — every branch
# ---------------------------------------------------------------------------


class TestCompileCoercion:
    # Identity / None cases

    def test_same_int_type_is_none(self) -> None:
        assert compile_coercion(IntType(), IntType()) is None

    def test_same_text_type_is_none(self) -> None:
        assert compile_coercion(TextType(), TextType()) is None

    def test_same_bool_type_is_none(self) -> None:
        assert compile_coercion(BoolType(), BoolType()) is None

    def test_same_decimal_type_is_none(self) -> None:
        assert compile_coercion(DecimalType(), DecimalType()) is None

    def test_same_unit_type_is_none(self) -> None:
        assert compile_coercion(UnitType(), UnitType()) is None

    def test_same_json_type_is_none(self) -> None:
        # json → json: identity, no coercion
        assert compile_coercion(JsonType(), JsonType()) is None

    def test_type_var_source_is_none(self) -> None:
        # TypeVarType source → opaque, no coercion
        assert compile_coercion(TypeVarType("T"), IntType()) is None

    def test_type_var_target_is_none(self) -> None:
        assert compile_coercion(IntType(), TypeVarType("T")) is None

    # Scalar coercions

    def test_int_to_decimal(self) -> None:
        result = compile_coercion(IntType(), DecimalType())
        assert result == IntToDecimal()

    def test_int_to_json(self) -> None:
        # Rule 1: target is JsonType and source is not JsonType → ToJson
        result = compile_coercion(IntType(), JsonType())
        assert result == ToJson()

    def test_text_to_json(self) -> None:
        result = compile_coercion(TextType(), JsonType())
        assert result == ToJson()

    def test_bool_to_json(self) -> None:
        result = compile_coercion(BoolType(), JsonType())
        assert result == ToJson()

    def test_decimal_to_json(self) -> None:
        result = compile_coercion(DecimalType(), JsonType())
        assert result == ToJson()

    def test_list_int_to_json(self) -> None:
        result = compile_coercion(ListType(IntType()), JsonType())
        assert result == ToJson()

    # List coercions

    def test_list_int_to_list_decimal(self) -> None:
        result = compile_coercion(ListType(IntType()), ListType(DecimalType()))
        assert result == MapList(IntToDecimal())

    def test_list_int_to_list_int_is_none(self) -> None:
        # Element coercion is None → outer is None
        result = compile_coercion(ListType(IntType()), ListType(IntType()))
        assert result is None

    def test_list_int_to_list_json(self) -> None:
        result = compile_coercion(ListType(IntType()), ListType(JsonType()))
        assert result == MapList(ToJson())

    def test_nested_list_int_to_list_list_decimal(self) -> None:
        result = compile_coercion(
            ListType(ListType(IntType())),
            ListType(ListType(DecimalType())),
        )
        assert result == MapList(MapList(IntToDecimal()))

    # Dict coercions

    def test_dict_int_to_dict_decimal(self) -> None:
        result = compile_coercion(DictType(IntType()), DictType(DecimalType()))
        assert result == MapDictValues(IntToDecimal())

    def test_dict_int_to_dict_int_is_none(self) -> None:
        result = compile_coercion(DictType(IntType()), DictType(IntType()))
        assert result is None

    def test_dict_int_to_dict_json(self) -> None:
        result = compile_coercion(DictType(IntType()), DictType(JsonType()))
        assert result == MapDictValues(ToJson())

    # Record coercions

    def test_record_no_field_needs_coercion_is_none(self) -> None:
        rec = RecordType("R", {"x": IntType()})
        result = compile_coercion(rec, rec)
        assert result is None

    def test_record_one_field_needs_coercion(self) -> None:
        src = RecordType("R", {"x": IntType(), "y": TextType()})
        tgt = RecordType("R", {"x": DecimalType(), "y": TextType()})
        result = compile_coercion(src, tgt)
        assert result == MapRecordFields((("x", IntToDecimal()),))

    def test_record_multiple_fields_need_coercion(self) -> None:
        src = RecordType("R", {"x": IntType(), "y": IntType()})
        tgt = RecordType("R", {"x": DecimalType(), "y": DecimalType()})
        result = compile_coercion(src, tgt)
        assert result == MapRecordFields((("x", IntToDecimal()), ("y", IntToDecimal())))

    def test_record_target_field_not_in_source_skipped(self) -> None:
        # Only shared fields are coerced; fields not in source are ignored
        src = RecordType("R", {"x": IntType()})
        tgt = RecordType("R", {"x": DecimalType(), "z": TextType()})
        result = compile_coercion(src, tgt)
        assert result == MapRecordFields((("x", IntToDecimal()),))

    # Enum coercions

    def test_enum_no_field_coercion_needed_is_none(self) -> None:
        e = EnumType("E", {"A": {"x": IntType()}, "B": {}})
        result = compile_coercion(e, e)
        assert result is None

    def test_enum_one_variant_field_needs_coercion(self) -> None:
        src = EnumType("E", {"A": {"x": IntType()}, "B": {}})
        tgt = EnumType("E", {"A": {"x": DecimalType()}, "B": {}})
        result = compile_coercion(src, tgt)
        assert result == MapEnumFields((("A", (("x", IntToDecimal()),)),))

    def test_enum_empty_result_variant_excluded(self) -> None:
        # Variant B has no fields needing coercion → only A in result
        src = EnumType("E", {"A": {"x": IntType()}, "B": {"y": TextType()}})
        tgt = EnumType("E", {"A": {"x": DecimalType()}, "B": {"y": TextType()}})
        result = compile_coercion(src, tgt)
        assert result == MapEnumFields((("A", (("x", IntToDecimal()),)),))

    def test_enum_target_field_not_in_source_skipped(self) -> None:
        # Target variant A has field "extra" not in source → only "x" can be coerced
        src = EnumType("E", {"A": {"x": IntType()}})
        tgt = EnumType("E", {"A": {"x": DecimalType(), "extra": TextType()}})
        result = compile_coercion(src, tgt)
        # "extra" is not in source so it's skipped; only "x" coercion emitted
        assert result == MapEnumFields((("A", (("x", IntToDecimal()),)),))

    def test_enum_source_variant_not_in_target_skipped(self) -> None:
        # Source has variant B that target doesn't; only target variants are processed
        src = EnumType("E", {"A": {"x": IntType()}, "B": {"y": IntType()}})
        tgt = EnumType("E", {"A": {"x": DecimalType()}})
        result = compile_coercion(src, tgt)
        assert result == MapEnumFields((("A", (("x", IntToDecimal()),)),))

    # Fallthrough — otherwise → None

    def test_int_to_text_is_none(self) -> None:
        # No implicit int→text coercion; the checker would reject this
        assert compile_coercion(IntType(), TextType()) is None

    def test_text_to_bool_is_none(self) -> None:
        assert compile_coercion(TextType(), BoolType()) is None


# ---------------------------------------------------------------------------
# lower_program: basic sanity — validate_ir passes
# ---------------------------------------------------------------------------


class TestLowerProgramValidateIr:
    def test_empty_body_raises(self) -> None:
        # A program ending in a let/var is a static error; an empty body would
        # be "()" which is valid. Test a simple constant program validates.
        prog = _lower("()")
        validate_ir(prog)

    def test_validate_passes_for_all_literal_types(self) -> None:
        source = "1"
        prog = _lower(source)
        validate_ir(prog)


# ---------------------------------------------------------------------------
# lower_program: literal lowering
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
# lower_program: list literal lowering
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
# lower_program: dict literal lowering
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
# lower_program: let / var binding lowering
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
        # MapList(IntToDecimal) on an IrLoad can only arise in later milestones
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
# lower_program: VarRef lowering (IrLoad)
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
# lower_program: AssignStmt lowering (simple name target only)
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
# lower_program: Block lowering
#
# A Block node appears as an expression only inside branches of control-flow
# (if/case/do/try) and function bodies — all of which are M3+.  We test the
# Block lowering path by calling the lowerer's internal API directly with a
# synthetic Block node constructed from a lowered CheckedProgram.  This
# exercises the Block arm of lower_expr and _lower_block without needing M3
# control-flow in the source.
# ---------------------------------------------------------------------------


class TestBlockLowering:
    def test_block_emits_ir_block(self) -> None:
        # Build a CheckedProgram from a simple multi-statement source so we
        # can reuse its node_types and resolution tables.  Then construct
        # a synthetic Block wrapping those items and call lower_expr directly.
        from agm.agl.syntax.nodes import Block, LetDecl, VarRef
        from agm.agl.syntax.spans import SourceSpan

        src = "let _a: int = 1\n_a"
        checked = _check(src)
        # Build a _Lowerer in the same state as lower_program would, but stop
        # before running the top-level body so we can call lower_expr manually.
        lowerer = _make_lowerer(checked, src)

        body = checked.resolved.program.body
        # body.items = [LetDecl(_a, ...), VarRef(_a, ...)]
        let_item = body.items[0]
        ref_item = body.items[1]
        assert isinstance(let_item, LetDecl)
        assert isinstance(ref_item, VarRef)

        sp = SourceSpan(
            start_line=1, start_col=1, end_line=2, end_col=3,
            start_offset=0, end_offset=len(src),
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
# lower_program: sources table
# ---------------------------------------------------------------------------


class TestSourcesTable:
    def test_single_source_registered(self) -> None:
        prog = _lower("()")
        assert len(prog.sources) == 1

    def test_source_display_name(self) -> None:
        prog = _lower("()", validate=False)
        (src_id,) = prog.sources
        assert prog.sources[src_id].display_name == "<test>"

    def test_source_normalized_text(self) -> None:
        src = "()"
        prog = _lower(src)
        (src_id,) = prog.sources
        assert prog.sources[src_id].normalized_text == src


# ---------------------------------------------------------------------------
# lower_program: nominals table is empty in M2
# ---------------------------------------------------------------------------


class TestNominalsEmpty:
    def test_nominals_contains_builtin_exceptions(self) -> None:
        """program.nominals always contains at least the built-in exception types.

        Even an empty program populates nominals with all built-in exceptions
        (they are always in scope).  User-declared records/enums are added on top.
        """
        from agm.agl.ir.program import NominalKind
        from agm.agl.typecheck.types import BUILTIN_EXCEPTIONS

        prog = _lower("()")
        # All built-in exception names must appear in the table
        exception_names = {desc.nominal.declared_name for desc in prog.nominals.values()}
        for builtin_name in BUILTIN_EXCEPTIONS:
            assert builtin_name in exception_names, (
                f"Built-in exception {builtin_name!r} missing from program.nominals"
            )
        # Every entry in an empty program uses NominalKind.EXCEPTION (only builtins present)
        for _nominal_id, desc in prog.nominals.items():
            assert desc.kind is NominalKind.EXCEPTION

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
            "let p = Point(x: 1, y: 2)\n"
            "let c = Color.Red\n"
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
# lower_program: unsupported nodes raise a clear error
# ---------------------------------------------------------------------------


class TestUnsupportedNodes:
    def test_user_function_call_lowers_correctly(self) -> None:
        """Direct user function calls are supported in M4a and must lower without error."""
        # f() is a direct user function call — M4a supports this.
        prog = _lower("def f() -> int = 1\nlet result = f()\n()")
        # Verify the program has a function in the functions table
        assert len(prog.functions) == 1

    def test_iife_lambda_call_lowers_to_indirect_call(self) -> None:
        """Calling an immediately-invoked lambda (non-VarRef, non-FieldAccess callee).

        When the callee of a Call is a Lambda (M4b), lowering produces an IrIndirectCall
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
        """Qualified constructor (e.g. Color.Red) lowers to IrMakeEnum/IrMakeConstructor.

        M3d implemented qualified constructor lowering.  A nullary variant (no fields)
        lowers to IrMakeEnum (eagerly constructed).  A variant with fields lowers to
        IrMakeConstructor.  Here Red is nullary, so the binding value must be IrMakeEnum.
        """
        from agm.agl.ir.nodes import IrBind, IrMakeEnum

        source = """\
enum Color
  | Red
  | Blue

let c = Color.Red
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
        """Lambda expressions now lower to IrMakeClosure (M4b implemented)."""
        prog = _lower("let f = fn(x: int) -> int => x + 1\n()")
        inits = prog.modules[prog.entry_module].initializers
        f_bind = inits[0]
        assert isinstance(f_bind, IrBind)
        assert isinstance(f_bind.value, IrMakeClosure)

    def test_if_lowers_correctly(self) -> None:
        """If expressions are lowered to IrIf in M3f-A."""
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
        """IndexTarget assignment lowers to IrAssign with a non-empty path (M3c)."""
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
# lower_program: Location fields are valid
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
# lower_program: validate_ir integration
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
        prog = _lower(source, validate=True)
        validate_ir(prog)

    def test_validate_false_skips_validation(self) -> None:
        # Should not raise even without explicit validate call
        prog = _lower("()", validate=False)
        assert prog is not None


# ---------------------------------------------------------------------------
# Direct _Lowerer unit test — IrField lowering (non-constructor FieldAccess)
# ---------------------------------------------------------------------------


class TestIrFieldLowering:
    """Unit tests for the IrField lowering path in _Lowerer.lower_expr.

    FieldAccess nodes that resolve to records/exceptions (rather than
    qualified constructor references) lower to IrField.  These tests exercise
    the _Lowerer directly because end-to-end lowering of FieldAccess requires
    constructor calls (M4, deferred).
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

        # Build a CheckedProgram from the trivial source "()" — we only need the
        # resolved/type-table scaffolding, not the actual program body.
        checked = _check("()")

        # A fresh node_id not present in qualified_constructor_refs.
        fake_node_id = 99999
        assert fake_node_id not in checked.resolved.qualified_constructor_refs

        span = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=5,
            start_offset=0, end_offset=4, source=UNKNOWN_SOURCE,
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
# M4a lower: function-related coverage
# ---------------------------------------------------------------------------


class TestM4aLowerFunctions:
    """Tests for M4a function lowering coverage."""

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

    def test_builtin_print_in_function_body_lowers_to_ir_print(self) -> None:
        """print() inside a function body lowers to IrPrint (M6a implemented)."""
        from agm.agl.ir.nodes import IrPrint

        source = (
            "def f(x: int) -> int =\n"
            "  print(x)\n"
            "  x + 1\n"
            "let result = f(5)\n()"
        )
        prog = _lower(source)
        # Find the function descriptor and verify its body contains IrPrint
        assert len(prog.functions) == 1
        fn_desc = next(iter(prog.functions.values()))
        # The function body is a block: [IrPrint(...), IrArith(x + 1)]
        from agm.agl.ir.nodes import IrBlock
        assert isinstance(fn_desc.body, IrBlock)
        assert any(isinstance(item, IrPrint) for item in fn_desc.body.items)

    def test_indirect_call_via_let_binding_lowers_to_indirect_call(self) -> None:
        """Calling a let-bound function reference lowers to IrIndirectCall (M4b)."""
        source = (
            "def f(x: int) -> int = x + 1\n"
            "let fn_ref = f\n"
            "let result = fn_ref(5)\n()"
        )
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        # inits[0] = IrBind(f, IrMakeClosure)
        # inits[1] = IrBind(fn_ref, ...)
        # inits[2] = IrBind(result, IrIndirectCall)
        result_bind = inits[2]
        assert isinstance(result_bind, IrBind)
        assert isinstance(result_bind.value, IrIndirectCall)


class TestM4aLowerDefensivePaths:
    """Tests for M4a lowerer defensive code paths via internal API."""

    def test_lower_funcdef_without_preallocation_raises(self) -> None:
        """_lower_funcdef with a non-pre-allocated FuncDef raises AssertionError."""
        from agm.agl.syntax.nodes import FuncDef

        source = "def f(x: int) -> int = x + 1\nlet result = f(1)\n()"
        checked = _check(source)

        # Create lowerer and find FuncDef WITHOUT pre-allocating it
        lowerer = _make_lowerer(checked, source)
        funcdef = None
        for item in checked.resolved.program.body.items:
            if isinstance(item, FuncDef):
                funcdef = item
                break
        assert funcdef is not None

        # _lower_funcdef without pre-allocation raises AssertionError (compiler bug guard)
        with pytest.raises(AssertionError, match="was not pre-allocated"):
            lowerer._lower_funcdef(funcdef)

    def test_lower_direct_call_fallback_signature_by_name(self) -> None:
        """_lower_direct_call uses fallback get_function_signature when by_node_id returns None."""
        from agm.agl.syntax.nodes import FuncDef

        source = "def f(x: int) -> int = x + 1\nlet result = f(1)\n()"
        checked = _check(source)

        # Create lowerer and pre-allocate the funcdef
        lowerer = _make_lowerer(checked, source)
        for item in checked.resolved.program.body.items:
            if isinstance(item, FuncDef):
                lowerer._prealloc_funcdef(item)

        # Temporarily clear the by_node_id table to force fallback to by-name lookup
        type_env = checked.type_env
        original = dict(type_env._function_signatures_by_node_id)
        type_env._function_signatures_by_node_id.clear()
        try:
            # Now call _lower_funcdef — it will fall back to get_function_signature(name)
            for item in checked.resolved.program.body.items:
                if isinstance(item, FuncDef):
                    result = lowerer._lower_funcdef(item)
                    from agm.agl.ir.nodes import IrBind
                    assert isinstance(result, IrBind)
        finally:
            type_env._function_signatures_by_node_id.update(original)
    def test_lower_direct_call_fallback_sig_lookup_by_name(self) -> None:
        """_lower_direct_call fallback: get_function_signature(name) when by_node_id is None."""
        from agm.agl.syntax.nodes import FuncDef

        source = "def f(x: int) -> int = x + 1\nlet result = f(1)\n()"
        checked = _check(source)

        # Pre-allocate funcdef (phase 1)
        lowerer = _make_lowerer(checked, source)
        for item in checked.resolved.program.body.items:
            if isinstance(item, FuncDef):
                lowerer._prealloc_funcdef(item)

        # Clear by_node_id table to force both _lower_funcdef and _lower_direct_call fallbacks
        type_env = checked.type_env
        original = dict(type_env._function_signatures_by_node_id)
        type_env._function_signatures_by_node_id.clear()
        try:
            # Lower all items — will trigger fallback in both _lower_funcdef (472)
            # and _lower_direct_call (1103)
            for item in checked.resolved.program.body.items:
                lowerer.lower_item(item, top_level=True)
        finally:
            type_env._function_signatures_by_node_id.update(original)

    def test_field_access_callee_without_qcr_attempts_indirect_call(self) -> None:
        """FieldAccess callee not in qualified_constructor_refs falls through to indirect call.

        Since M4b, any callee that is not a constructor ref or direct function binding
        is lowered as an indirect call (IrIndirectCall).  A FieldAccess on a fake node
        with no resolution triggers an AssertionError inside lower_expr (compiler bug),
        which is the correct behavior for a malformed node in a unit test.
        """
        from agm.agl.syntax.nodes import Call, FieldAccess, VarRef

        source = "let x = 1\n()"
        checked = _check(source)
        lowerer = _make_lowerer(checked, source)

        # Get a real SourceSpan from the parsed program
        span = checked.resolved.program.body.items[0].span

        # FieldAccess callee not in qualified_constructor_refs → indirect call path.
        # The fake VarRef (node_id=9999) has no resolution, so lower_expr hits an
        # AssertionError inside the indirect call lowering path.
        fa = FieldAccess(
            obj=VarRef(name="x", node_id=9999, span=span, module_qualifier=None),
            field="method",
            node_id=10000,
            span=span,
        )
        fake_call = Call(callee=fa, args=(), named_args=(), node_id=10001, span=span)
        # AssertionError because the fake VarRef has no resolution (compiler bug path).
        with pytest.raises(AssertionError, match="compiler bug"):
            lowerer._lower_call(fake_call, 10001, span)

    def test_missing_required_arg_raises_assertion_error(self) -> None:
        """_lower_direct_call raises AssertionError when call is missing a required arg."""
        from agm.agl.syntax.nodes import Call, FuncDef, VarRef

        source = "def f(x: int) -> int = x + 1\nlet result = f(1)\n()"
        checked = _check(source)
        lowerer = _make_lowerer(checked, source)

        # Pre-allocate funcdef
        funcdef = next(
            item for item in checked.resolved.program.body.items if isinstance(item, FuncDef)
        )
        lowerer._prealloc_funcdef(funcdef)

        # Get a callee_ref for the function binding
        from agm.agl.scope.symbols import BinderKind, BindingRef
        span = funcdef.span
        from agm.agl.modules.ids import ENTRY_ID

        callee_ref = BindingRef(
            name="f",
            mutable=False,
            decl_span=span,
            decl_node_id=funcdef.node_id,
            kind=BinderKind.function_binding,
            module_id=ENTRY_ID,
        )

        # Create a fake call with NO args (f requires 1 arg)
        varref = VarRef(name="f", node_id=9999, span=span, module_qualifier=None)
        fake_call = Call(callee=varref, args=(), named_args=(), node_id=10001, span=span)

        with pytest.raises(AssertionError, match="compiler bug"):
            lowerer._lower_direct_call(fake_call, callee_ref, 10001, span)

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
        source = (
            "let y = 5\n"
            "def f() -> unit =\n"
            "  let _g = fn(u: unit) -> int => y\n"
            "  ()\n"
            "f()\n"
            "()"
        )
        prog = _lower(source)
        # f should have no captures (y is not used in f's body directly)
        # Find the outer def whose function_symbol corresponds to "f"
        f_descs = [
            d for d in prog.functions.values()
            if prog.symbols[d.function_symbol].public_name == "f"
        ]
        assert len(f_descs) == 1
        f_desc = f_descs[0]
        # Check that the IrMakeClosure for f in the initializers has no captures for y
        # (The initializer for f is IrBind(fn_sym, IrMakeClosure(...)))
        f_init = None
        for node in prog.modules[prog.entry_module].initializers:
            if (isinstance(node, IrBind)
                    and isinstance(node.value, IrMakeClosure)
                    and node.value.function_id == f_desc.function_id):
                f_init = node
                break
        assert f_init is not None, "IrBind for f not found in initializers"
        f_closure = f_init.value
        assert isinstance(f_closure, IrMakeClosure)
        # f should have no captures (y is not directly referenced in f's own body)
        assert len(f_closure.captures) == 0, (
            f"f should have no captures but has {f_closure.captures!r}"
        )


# ===========================================================================
# M4b — Lambda lowering and indirect call lowering (golden tests)
# ===========================================================================


class TestM4bLambdaLowering:
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

    def test_lambda_captures_outer_let(self) -> None:
        """Lambda that references an outer let captures it in its IrMakeClosure."""
        source = (
            "let offset = 10\n"
            "let add_off = fn(x: int) -> int => x + offset\n"
            "()"
        )
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        # offset is first, add_off is second
        add_off_bind = inits[1]
        assert isinstance(add_off_bind, IrBind)
        assert isinstance(add_off_bind.value, IrMakeClosure)
        captures = add_off_bind.value.captures
        assert len(captures) == 1, "Lambda should capture exactly one symbol (offset)"
        # by_cell=False because offset is a let binding
        assert captures[0].by_cell is False

    def test_lambda_inferred_return_type_lowered(self) -> None:
        """Lambda with inferred return type still gets body coercion from FunctionType."""
        source = "let f = fn(x: int) => x\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        bind = inits[0]
        assert isinstance(bind, IrBind)
        assert isinstance(bind.value, IrMakeClosure)
        fn_id = bind.value.function_id
        desc = prog.functions[fn_id]
        # Body is the param load (int -> int, no coercion needed since identity type)
        assert desc.body is not None

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
        assert isinstance(desc.body, IrCoerce), (
            f"Expected IrCoerce for int->decimal body, got {type(desc.body).__name__}"
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


class TestM4bIndirectCallLowering:
    """Golden tests: indirect call lowers to IrIndirectCall with coerced args."""

    def test_indirect_call_lowers_to_indirect_call_node(self) -> None:
        """A value-call lowers to IrIndirectCall (not IrDirectCall)."""
        source = (
            "let f = fn(x: int) -> int => x + 1\n"
            "let r = f(5)\n"
            "()"
        )
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
        source = (
            "let f = fn(x: int) -> int => x\n"
            "let r = f(7)\n"
            "()"
        )
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
        source = (
            "let f = fn(x: int) -> int => x\n"
            "let r = f(3)\n"
            "()"
        )
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        r_bind = inits[1]
        assert isinstance(r_bind, IrBind)
        assert isinstance(r_bind.value, IrIndirectCall)
        # Callee should be IrLoad(f's symbol), not IrCoerce
        callee = r_bind.value.callee
        assert isinstance(callee, IrLoad), (
            f"Callee should be IrLoad, got {type(callee).__name__}"
        )

    def test_indirect_call_validates_deep(self) -> None:
        """IrIndirectCall from end-to-end lowering passes validate_ir deep=True."""
        source = (
            "let f = fn(x: int) -> int => x + 1\n"
            "let r = f(5)\n"
            "()"
        )
        prog = _lower(source)
        validate_ir(prog, deep=True)  # no exception



# ---------------------------------------------------------------------------
# M5 — lower_graph: multi-module golden test
# ---------------------------------------------------------------------------


class TestLowerGraph:
    """Golden tests for lower_graph (M5 whole-graph module linking)."""

    def test_lower_graph_simple(self, tmp_path: Path) -> None:
        """lower_graph on a two-module program builds a valid ExecutableProgram.

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

        from agm.agl.lower.graph import lower_graph
        from agm.agl.modules.ids import ModuleId
        from agm.agl.modules.loader import load_graph
        from agm.agl.modules.roots import RootSet
        from agm.agl.scope.graph import resolve_graph
        from agm.agl.typecheck.graph import check_graph

        # Library defines a record type + a function using it.
        # Entry defines its own record type + imports lib's function.
        lib_source = (
            "record LibPoint\n"
            "  x: int\n"
            "  y: int\n"
            "\n"
            "def make_point(a: int, b: int) -> LibPoint =\n"
            "    LibPoint(x: a, y: b)\n"
        )
        entry_source = (
            "import lib\n"
            "record EntryBox\n"
            "  width: int\n"
            "\n"
            "let result = lib::make_point(1, 2)\n"
            "let box = EntryBox(width: 10)\n"
            "()\n"
        )

        root = tmp_path / "root"
        root.mkdir()
        lib_mid = ModuleId.from_dotted("lib")
        lib_path = root / lib_mid.relpath().replace("/", os.sep)
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib_path.write_text(lib_source)

        mg = load_graph(entry_source, entry_path=None, roots=RootSet(roots=frozenset([root])))
        rg = resolve_graph(mg)
        cg = check_graph(rg, _caps())

        prog = lower_graph(cg, validate=True)

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
            desc.function_id
            for desc in prog.functions.values()
            if desc.module_id == lib_mid
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
        result_syms = [
            desc for desc in prog.symbols.values()
            if desc.public_name == "result"
        ]
        assert len(result_syms) == 1

        # validate_ir must pass (already called with validate=True above,
        # but call again explicitly to be explicit)
        validate_ir(prog, deep=True)

    def test_lower_graph_type_alias_no_spurious_nominal(self, tmp_path: Path) -> None:
        """Type alias does not register a spurious NominalId in lower_graph.

        A program with ``type Foo = Point`` (where Point is a record) must NOT
        create a ``NominalId(mid, "Foo")`` entry in ``program.nominals``.
        Only the canonical declaration site ``NominalId(mid, "Point")`` must exist.
        """
        import os

        from agm.agl.ir.ids import NominalId
        from agm.agl.lower.graph import lower_graph
        from agm.agl.modules.ids import ModuleId
        from agm.agl.modules.loader import load_graph
        from agm.agl.modules.roots import RootSet
        from agm.agl.scope.graph import resolve_graph
        from agm.agl.typecheck.graph import check_graph

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
            "    Point(x: 0, y: 0)\n"
        )
        entry_source = (
            "import lib\n"
            "let p = lib::origin()\n"
            "()\n"
        )

        root = tmp_path / "root"
        root.mkdir()
        lib_mid = ModuleId.from_dotted("lib")
        lib_path = root / lib_mid.relpath().replace("/", os.sep)
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib_path.write_text(lib_source)

        mg = load_graph(entry_source, entry_path=None, roots=RootSet(roots=frozenset([root])))
        rg = resolve_graph(mg)
        cg = check_graph(rg, _caps())

        prog = lower_graph(cg, validate=True)

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

    def test_lower_graph_without_validate(self, tmp_path: Path) -> None:
        """lower_graph with validate=False skips validate_ir (covers the non-validate branch)."""
        import os

        from agm.agl.lower.graph import lower_graph
        from agm.agl.modules.ids import ModuleId
        from agm.agl.modules.loader import load_graph
        from agm.agl.modules.roots import RootSet
        from agm.agl.scope.graph import resolve_graph
        from agm.agl.typecheck.graph import check_graph

        lib_source = "def greet(n: int) -> int =\n    n + 1\n"
        entry_source = "import lib\nlet r = lib::greet(5)\n()\n"

        root = tmp_path / "root"
        root.mkdir()
        lib_mid = ModuleId.from_dotted("lib")
        lib_path = root / lib_mid.relpath().replace("/", os.sep)
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib_path.write_text(lib_source)

        mg = load_graph(entry_source, entry_path=None, roots=RootSet(roots=frozenset([root])))
        rg = resolve_graph(mg)
        cg = check_graph(rg, _caps())

        # validate=False (the default) skips validate_ir — covers the branch at graph.py:144
        prog = lower_graph(cg)
        assert len(prog.modules) == 2


# ---------------------------------------------------------------------------
# M6a golden lowering: print, parse_json, param declarations
# ---------------------------------------------------------------------------


class TestM6aLowering:
    """Golden lowering tests for M6a host operations."""

    def test_print_lowers_to_ir_print(self) -> None:
        """print(x) lowers to IrPrint wrapping the argument expression."""
        from agm.agl.ir.nodes import IrPrint

        source = "let x = 42\nprint(x)\n()"
        prog = _lower(source)
        entry = prog.modules[list(prog.modules.keys())[-1]]
        # initializers are [IrBind(x=42), IrPrint(IrLoad(x)), IrConstUnit]
        ir_print = next(
            (node for node in entry.initializers if isinstance(node, IrPrint)), None
        )
        assert ir_print is not None, "Expected IrPrint in entry initializers"
        assert isinstance(ir_print, IrPrint)

    def test_parse_json_lowers_to_ir_parse_json(self) -> None:
        """parse_json(s) lowers to IrParseJson wrapping the argument expression."""
        from agm.agl.ir.nodes import IrBind, IrParseJson

        source = "let j = parse_json('null')\n()"
        prog = _lower(source)
        entry = prog.modules[list(prog.modules.keys())[-1]]
        # The binding `let j = parse_json(...)` lowers to IrBind(value=IrParseJson(...))
        ir_bind = next(
            (node for node in entry.initializers if isinstance(node, IrBind)), None
        )
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
        """ask() now lowers to IrAsk (M6b implemented)."""
        from agm.agl.ir.nodes import IrAsk, IrBind

        source = "agent impl\nlet r: text = ask(\"prompt\", agent: impl)\n()"
        prog = _lower(source)
        # The initializers contain IrBind(IrAgentHandle) for `impl` then IrBind(IrAsk) for `r`.
        inits = prog.modules[prog.entry_module].initializers
        ask_binds = [n for n in inits if isinstance(n, IrBind) and isinstance(n.value, IrAsk)]
        assert len(ask_binds) == 1
        assert len(prog.contracts) == 1

    def test_exec_raises_not_implemented_m6c(self) -> None:
        """exec() lowers to NotImplementedError('M6c') — deferred to milestone M6c."""
        source = "exec(\"ls -la\")\n()"
        with pytest.raises(NotImplementedError, match="M6c"):
            _lower(source)

    def test_agent_decl_lowers_to_ir_agent_handle_bind(self) -> None:
        """AgentDecl lowers to IrBind(symbol, IrAgentHandle(name)) (M6b golden lowering)."""
        from agm.agl.ir.nodes import IrAgentHandle, IrBind

        source = "agent my_agent\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        # Expect an IrBind whose value is IrAgentHandle with the agent's name.
        handle_binds = [
            n for n in inits
            if isinstance(n, IrBind) and isinstance(n.value, IrAgentHandle)
        ]
        assert len(handle_binds) == 1, (
            f"Expected exactly 1 IrBind(IrAgentHandle), got {len(handle_binds)}"
        )
        assert handle_binds[0].value.agent_name == "my_agent"

    def test_ask_request_lowers_to_ir_ask_request_with_contract(self) -> None:
        """ask-request lowers to IrAskRequest + ContractRequest in program.contracts (M6b)."""
        from agm.agl.ir.nodes import IrAskRequest, IrBind

        source = "agent worker\nlet req = ask-request(\"my prompt\", agent: worker)\n()"
        prog = _lower(source)
        inits = prog.modules[prog.entry_module].initializers
        # The ask-request binding lowers to IrBind(symbol, IrAskRequest(...)).
        ask_req_binds = [
            n for n in inits
            if isinstance(n, IrBind) and isinstance(n.value, IrAskRequest)
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


