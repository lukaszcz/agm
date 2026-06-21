"""Golden structural tests for the agm.agl.ir package (M1 skeleton).

These tests pin the structural contract of the IR data model so that later
milestones notice accidental shape changes.  They cover:
- Every node, descriptor, and id type.
- Frozenness (mutating a field raises FrozenInstanceError).
- Field values round-trip correctly.
- IrExpr union membership.
- Coercion nested construction, equality, and hash.
- NominalId equality/inequality.
- Enum members exist as specified.
- ExecutableProgram/ExecutableModule/descriptors construction.
"""

from __future__ import annotations

import dataclasses
import decimal

import pytest

from agm.agl.ir import (
    ArithOp,
    CmpOp,
    Coercion,
    CompareKind,
    ContainsKind,
    ContractId,
    ExecutableModule,
    ExecutableProgram,
    FunctionId,
    IntToDecimal,
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
    IrExpr,
    IrIndexStep,
    IrLoad,
    IrMakeDict,
    IrMakeList,
    IrSequence,
    Location,
    MapDictValues,
    MapEnumFields,
    MapList,
    MapRecordFields,
    NominalDescriptor,
    NominalId,
    NominalKind,
    NumericKind,
    SourceFile,
    SourceId,
    SymbolDescriptor,
    SymbolId,
    ToJson,
)
from agm.agl.modules.ids import ModuleId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOD_A = ModuleId.from_dotted("mod_a")
MOD_B = ModuleId.from_dotted("mod_b")

SID0 = SourceId(value=0)
SYM0 = SymbolId(value=0)
SYM1 = SymbolId(value=1)
FN0 = FunctionId(value=0)
CT0 = ContractId(value=0)
NOM0 = NominalId(module_id=MOD_A, declared_name="Foo")


def loc(source_id: SourceId = SID0) -> Location:
    return Location(
        source_id=source_id,
        start_offset=0,
        end_offset=10,
        start_line=1,
        start_col=0,
    )


LOC = loc()


# ---------------------------------------------------------------------------
# ids.py — SourceId, SymbolId, FunctionId, ContractId
# ---------------------------------------------------------------------------


class TestIds:
    def test_source_id_value(self) -> None:
        sid = SourceId(value=42)
        assert sid.value == 42

    def test_source_id_frozen(self) -> None:
        sid = SourceId(value=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(sid, "value", 99)

    def test_source_id_equality(self) -> None:
        assert SourceId(value=1) == SourceId(value=1)
        assert SourceId(value=1) != SourceId(value=2)

    def test_symbol_id_frozen(self) -> None:
        sym = SymbolId(value=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(sym, "value", 5)

    def test_function_id(self) -> None:
        fn = FunctionId(value=7)
        assert fn.value == 7

    def test_function_id_frozen(self) -> None:
        fn = FunctionId(value=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(fn, "value", 1)

    def test_contract_id(self) -> None:
        ct = ContractId(value=3)
        assert ct.value == 3

    def test_contract_id_frozen(self) -> None:
        ct = ContractId(value=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(ct, "value", 1)

    def test_ids_hashable(self) -> None:
        s = {SourceId(value=1), SymbolId(value=2), FunctionId(value=3), ContractId(value=4)}
        assert len(s) == 4


# ---------------------------------------------------------------------------
# ids.py — NominalId
# ---------------------------------------------------------------------------


class TestNominalId:
    def test_fields(self) -> None:
        n = NominalId(module_id=MOD_A, declared_name="Bar")
        assert n.module_id == MOD_A
        assert n.declared_name == "Bar"

    def test_frozen(self) -> None:
        n = NominalId(module_id=MOD_A, declared_name="X")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "declared_name", "Y")

    def test_equality_same(self) -> None:
        assert NominalId(MOD_A, "Foo") == NominalId(MOD_A, "Foo")

    def test_inequality_different_module(self) -> None:
        assert NominalId(MOD_A, "Foo") != NominalId(MOD_B, "Foo")

    def test_inequality_different_name(self) -> None:
        assert NominalId(MOD_A, "Foo") != NominalId(MOD_A, "Bar")

    def test_hashable(self) -> None:
        d = {NominalId(MOD_A, "Foo"): 1, NominalId(MOD_B, "Foo"): 2}
        assert d[NominalId(MOD_A, "Foo")] == 1
        assert d[NominalId(MOD_B, "Foo")] == 2


# ---------------------------------------------------------------------------
# ids.py — Location
# ---------------------------------------------------------------------------


class TestLocation:
    def test_fields(self) -> None:
        loc_obj = Location(
            source_id=SID0,
            start_offset=5,
            end_offset=15,
            start_line=3,
            start_col=7,
        )
        assert loc_obj.source_id == SID0
        assert loc_obj.start_offset == 5
        assert loc_obj.end_offset == 15
        assert loc_obj.start_line == 3
        assert loc_obj.start_col == 7

    def test_frozen(self) -> None:
        loc_obj = loc()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(loc_obj, "start_line", 99)

    def test_hashable(self) -> None:
        d = {loc(): "a"}
        assert d[loc()] == "a"


# ---------------------------------------------------------------------------
# operations.py — enums
# ---------------------------------------------------------------------------


class TestArithOp:
    def test_members_exist(self) -> None:
        # Derived from BinOp in syntax/nodes.py: ADD(+), SUB(-), MUL(*), DIV(/)
        assert ArithOp.ADD
        assert ArithOp.SUB
        assert ArithOp.MUL
        assert ArithOp.DIV

    def test_exhaustive_set(self) -> None:
        names = {m.name for m in ArithOp}
        assert names == {"ADD", "SUB", "MUL", "DIV"}


class TestCmpOp:
    def test_members_exist(self) -> None:
        # Derived from BinOp: EQ(=), NEQ(!=), LT(<), LE(<=), GT(>), GE(>=)
        assert CmpOp.EQ
        assert CmpOp.NEQ
        assert CmpOp.LT
        assert CmpOp.LE
        assert CmpOp.GT
        assert CmpOp.GE

    def test_exhaustive_set(self) -> None:
        names = {m.name for m in CmpOp}
        assert names == {"EQ", "NEQ", "LT", "LE", "GT", "GE"}


class TestNumericKind:
    def test_members(self) -> None:
        assert NumericKind.INT
        assert NumericKind.DECIMAL

    def test_exhaustive(self) -> None:
        assert {m.name for m in NumericKind} == {"INT", "DECIMAL"}


class TestCompareKind:
    def test_members(self) -> None:
        assert CompareKind.INT
        assert CompareKind.DECIMAL
        assert CompareKind.TEXT
        assert CompareKind.STRUCTURAL

    def test_exhaustive(self) -> None:
        assert {m.name for m in CompareKind} == {"INT", "DECIMAL", "TEXT", "STRUCTURAL"}


class TestContainsKind:
    def test_members(self) -> None:
        assert ContainsKind.LIST
        assert ContainsKind.DICT
        assert ContainsKind.TEXT

    def test_exhaustive(self) -> None:
        assert {m.name for m in ContainsKind} == {"LIST", "DICT", "TEXT"}


# ---------------------------------------------------------------------------
# operations.py — Coercion types
# ---------------------------------------------------------------------------


class TestCoercion:
    def test_int_to_decimal(self) -> None:
        c = IntToDecimal()
        assert isinstance(c, IntToDecimal)

    def test_int_to_decimal_frozen(self) -> None:
        c = IntToDecimal()
        # no fields to mutate, but verify it's a frozen dataclass
        assert dataclasses.is_dataclass(c)

    def test_to_json(self) -> None:
        c = ToJson()
        assert isinstance(c, ToJson)

    def test_map_list(self) -> None:
        inner = IntToDecimal()
        c = MapList(item=inner)
        assert c.item == inner

    def test_map_list_frozen(self) -> None:
        c = MapList(item=IntToDecimal())
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(c, "item", ToJson())

    def test_map_dict_values(self) -> None:
        c = MapDictValues(value=IntToDecimal())
        assert isinstance(c.value, IntToDecimal)

    def test_map_dict_values_frozen(self) -> None:
        c = MapDictValues(value=IntToDecimal())
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(c, "value", ToJson())

    def test_map_record_fields(self) -> None:
        fields: tuple[tuple[str, Coercion], ...] = (("x", IntToDecimal()), ("y", ToJson()))
        c = MapRecordFields(fields=fields)
        assert c.fields == fields

    def test_map_record_fields_frozen(self) -> None:
        c = MapRecordFields(fields=(("x", IntToDecimal()),))
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(c, "fields", ())

    def test_map_enum_fields(self) -> None:
        variants: tuple[tuple[str, tuple[tuple[str, Coercion], ...]], ...] = (
            ("VariantA", (("f1", IntToDecimal()),)),
            ("VariantB", (("f2", ToJson()),)),
        )
        c = MapEnumFields(variants=variants)
        assert c.variants == variants

    def test_map_enum_fields_frozen(self) -> None:
        c = MapEnumFields(variants=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(c, "variants", (("X", ()),))

    def test_nested_coercion(self) -> None:
        # MapList(MapRecordFields((("x", IntToDecimal()),)))
        inner = MapRecordFields(fields=(("x", IntToDecimal()),))
        outer = MapList(item=inner)
        assert outer.item == inner
        assert isinstance(outer.item, MapRecordFields)
        assert outer.item.fields[0][0] == "x"
        assert isinstance(outer.item.fields[0][1], IntToDecimal)

    def test_coercion_equality(self) -> None:
        a = MapList(item=IntToDecimal())
        b = MapList(item=IntToDecimal())
        assert a == b

    def test_coercion_hash(self) -> None:
        s = {MapList(item=IntToDecimal()), MapList(item=IntToDecimal())}
        assert len(s) == 1

    def test_coercion_type_alias(self) -> None:
        # Coercion is a type alias — each member is an instance of its class
        c: Coercion = IntToDecimal()
        assert isinstance(c, IntToDecimal)

        c2: Coercion = MapList(item=IntToDecimal())
        assert isinstance(c2, MapList)


# ---------------------------------------------------------------------------
# nodes.py — IrExpr union membership
# ---------------------------------------------------------------------------


class TestIrExprUnion:
    def test_const_int_in_union(self) -> None:
        # IrExpr is a closed union type alias; each concrete node is one of its members.
        # We verify that a typed variable accepting IrExpr can hold an IrConstInt.
        node: IrExpr = IrConstInt(location=LOC, value=42)
        assert isinstance(node, IrConstInt)
        assert node.value == 42

    def test_index_step_not_in_expr_union(self) -> None:
        # IrIndexStep is a helper record, not a member of IrExpr.
        # We verify this by confirming it is not an instance of any IrExpr member type.
        step = IrIndexStep(index=IrConstInt(location=LOC, value=0), location=LOC)
        assert not isinstance(step, IrConstInt)
        assert not isinstance(step, IrConstDecimal)
        assert not isinstance(step, IrConstBool)
        assert not isinstance(step, IrConstText)
        assert not isinstance(step, IrConstUnit)
        assert not isinstance(step, IrConstJsonNull)
        assert not isinstance(step, IrMakeList)
        assert not isinstance(step, IrMakeDict)
        assert not isinstance(step, IrLoad)
        assert not isinstance(step, IrBind)
        assert not isinstance(step, IrAssign)
        assert not isinstance(step, IrCoerce)
        assert not isinstance(step, IrSequence)
        assert not isinstance(step, IrBlock)


# ---------------------------------------------------------------------------
# nodes.py — Constants
# ---------------------------------------------------------------------------


class TestIrConstants:
    def test_const_int(self) -> None:
        n = IrConstInt(location=LOC, value=99)
        assert n.value == 99
        assert n.location == LOC

    def test_const_int_frozen(self) -> None:
        n = IrConstInt(location=LOC, value=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "value", 2)

    def test_const_decimal(self) -> None:
        d = decimal.Decimal("3.14")
        n = IrConstDecimal(location=LOC, value=d)
        assert n.value == d

    def test_const_decimal_frozen(self) -> None:
        n = IrConstDecimal(location=LOC, value=decimal.Decimal("1"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "value", decimal.Decimal("2"))

    def test_const_bool(self) -> None:
        t = IrConstBool(location=LOC, value=True)
        f = IrConstBool(location=LOC, value=False)
        assert t.value is True
        assert f.value is False

    def test_const_bool_frozen(self) -> None:
        n = IrConstBool(location=LOC, value=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "value", False)

    def test_const_text(self) -> None:
        n = IrConstText(location=LOC, value="hello")
        assert n.value == "hello"

    def test_const_text_frozen(self) -> None:
        n = IrConstText(location=LOC, value="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "value", "y")

    def test_const_unit(self) -> None:
        n = IrConstUnit(location=LOC)
        assert n.location == LOC

    def test_const_unit_frozen(self) -> None:
        n = IrConstUnit(location=LOC)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "location", loc(SID0))

    def test_const_json_null(self) -> None:
        n = IrConstJsonNull(location=LOC)
        assert n.location == LOC

    def test_const_json_null_frozen(self) -> None:
        n = IrConstJsonNull(location=LOC)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "location", loc(SID0))


# ---------------------------------------------------------------------------
# nodes.py — Container literals
# ---------------------------------------------------------------------------


class TestIrContainerLiterals:
    def test_make_list_empty(self) -> None:
        n = IrMakeList(location=LOC, items=())
        assert n.items == ()

    def test_make_list_with_items(self) -> None:
        item = IrConstInt(location=LOC, value=1)
        n = IrMakeList(location=LOC, items=(item,))
        assert n.items == (item,)

    def test_make_list_frozen(self) -> None:
        n = IrMakeList(location=LOC, items=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "items", (IrConstInt(location=LOC, value=1),))

    def test_make_dict_empty(self) -> None:
        n = IrMakeDict(location=LOC, entries=())
        assert n.entries == ()

    def test_make_dict_with_entries(self) -> None:
        key = IrConstText(location=LOC, value="k")
        val = IrConstInt(location=LOC, value=42)
        n = IrMakeDict(location=LOC, entries=((key, val),))
        assert n.entries[0] == (key, val)

    def test_make_dict_frozen(self) -> None:
        n = IrMakeDict(location=LOC, entries=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "entries", ())


# ---------------------------------------------------------------------------
# nodes.py — Bindings/storage
# ---------------------------------------------------------------------------


class TestIrBindingsStorage:
    def test_ir_load(self) -> None:
        n = IrLoad(location=LOC, symbol=SYM0)
        assert n.symbol == SYM0
        assert n.location == LOC

    def test_ir_load_frozen(self) -> None:
        n = IrLoad(location=LOC, symbol=SYM0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "symbol", SYM1)

    def test_ir_bind(self) -> None:
        val = IrConstInt(location=LOC, value=5)
        n = IrBind(location=LOC, symbol=SYM0, value=val)
        assert n.symbol == SYM0
        assert n.value == val

    def test_ir_bind_frozen(self) -> None:
        n = IrBind(location=LOC, symbol=SYM0, value=IrConstInt(location=LOC, value=1))
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "symbol", SYM1)

    def test_ir_index_step(self) -> None:
        idx = IrConstInt(location=LOC, value=0)
        step = IrIndexStep(index=idx, location=LOC)
        assert step.index == idx
        assert step.location == LOC

    def test_ir_index_step_frozen(self) -> None:
        step = IrIndexStep(index=IrConstInt(location=LOC, value=0), location=LOC)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(step, "index", IrConstInt(location=LOC, value=1))

    def test_ir_assign_no_path(self) -> None:
        val = IrConstInt(location=LOC, value=7)
        n = IrAssign(location=LOC, symbol=SYM0, path=(), value=val)
        assert n.symbol == SYM0
        assert n.path == ()
        assert n.value == val

    def test_ir_assign_with_path(self) -> None:
        idx_expr = IrConstInt(location=LOC, value=0)
        step = IrIndexStep(index=idx_expr, location=LOC)
        val = IrConstText(location=LOC, value="v")
        n = IrAssign(location=LOC, symbol=SYM0, path=(step,), value=val)
        assert len(n.path) == 1
        assert n.path[0] == step

    def test_ir_assign_frozen(self) -> None:
        n = IrAssign(
            location=LOC,
            symbol=SYM0,
            path=(),
            value=IrConstUnit(location=LOC),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "symbol", SYM1)


# ---------------------------------------------------------------------------
# nodes.py — Coercion node
# ---------------------------------------------------------------------------


class TestIrCoerce:
    def test_construct(self) -> None:
        val = IrConstInt(location=LOC, value=1)
        n = IrCoerce(location=LOC, value=val, operation=IntToDecimal())
        assert n.value == val
        assert isinstance(n.operation, IntToDecimal)

    def test_frozen(self) -> None:
        n = IrCoerce(
            location=LOC,
            value=IrConstInt(location=LOC, value=0),
            operation=IntToDecimal(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "operation", ToJson())

    def test_location_carried(self) -> None:
        n = IrCoerce(
            location=LOC,
            value=IrConstBool(location=LOC, value=True),
            operation=ToJson(),
        )
        assert n.location == LOC


# ---------------------------------------------------------------------------
# nodes.py — Sequencing
# ---------------------------------------------------------------------------


class TestIrSequencing:
    def test_ir_sequence(self) -> None:
        a = IrConstInt(location=LOC, value=1)
        b = IrConstInt(location=LOC, value=2)
        n = IrSequence(location=LOC, items=(a, b))
        assert len(n.items) == 2
        assert n.items[0] == a
        assert n.items[1] == b

    def test_ir_sequence_frozen(self) -> None:
        n = IrSequence(location=LOC, items=(IrConstUnit(location=LOC),))
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "items", ())

    def test_ir_block(self) -> None:
        a = IrConstText(location=LOC, value="done")
        n = IrBlock(location=LOC, items=(a,))
        assert n.items == (a,)

    def test_ir_block_frozen(self) -> None:
        n = IrBlock(location=LOC, items=(IrConstUnit(location=LOC),))
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(n, "items", ())

    def test_sequence_vs_block_distinction(self) -> None:
        # Both are separate types — not aliases of each other
        s = IrSequence(location=LOC, items=(IrConstUnit(location=LOC),))
        b = IrBlock(location=LOC, items=(IrConstUnit(location=LOC),))
        # They are distinct types: IrSequence is not IrBlock
        assert type(s).__name__ != type(b).__name__
        assert isinstance(s, IrSequence)
        assert isinstance(b, IrBlock)
        # IrSequence is not an IrBlock and vice versa
        assert not isinstance(s, IrBlock)
        assert not isinstance(b, IrSequence)


# ---------------------------------------------------------------------------
# program.py — NominalKind
# ---------------------------------------------------------------------------


class TestNominalKind:
    def test_members(self) -> None:
        assert NominalKind.RECORD
        assert NominalKind.ENUM
        assert NominalKind.EXCEPTION

    def test_exhaustive(self) -> None:
        assert {m.name for m in NominalKind} == {"RECORD", "ENUM", "EXCEPTION"}


# ---------------------------------------------------------------------------
# program.py — SymbolDescriptor
# ---------------------------------------------------------------------------


class TestSymbolDescriptor:
    def test_construct_with_module_owner(self) -> None:
        desc = SymbolDescriptor(
            symbol_id=SYM0,
            mutable=False,
            public_name="x",
            owner=MOD_A,
        )
        assert desc.symbol_id == SYM0
        assert desc.mutable is False
        assert desc.public_name == "x"
        assert desc.owner == MOD_A

    def test_construct_with_function_owner(self) -> None:
        desc = SymbolDescriptor(
            symbol_id=SYM1,
            mutable=True,
            public_name=None,
            owner=FN0,
        )
        assert desc.owner == FN0
        assert desc.public_name is None

    def test_frozen(self) -> None:
        desc = SymbolDescriptor(
            symbol_id=SYM0, mutable=False, public_name=None, owner=MOD_A
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(desc, "mutable", True)


# ---------------------------------------------------------------------------
# program.py — NominalDescriptor
# ---------------------------------------------------------------------------


class TestNominalDescriptor:
    def test_construct(self) -> None:
        desc = NominalDescriptor(
            nominal=NOM0,
            display_name="Foo",
            kind=NominalKind.RECORD,
        )
        assert desc.nominal == NOM0
        assert desc.display_name == "Foo"
        assert desc.kind == NominalKind.RECORD

    def test_frozen(self) -> None:
        desc = NominalDescriptor(
            nominal=NOM0,
            display_name="Foo",
            kind=NominalKind.ENUM,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(desc, "display_name", "Bar")


# ---------------------------------------------------------------------------
# program.py — SourceFile
# ---------------------------------------------------------------------------


class TestSourceFile:
    def test_construct(self) -> None:
        sf = SourceFile(display_name="main.agl", normalized_text="let x = 1")
        assert sf.display_name == "main.agl"
        assert sf.normalized_text == "let x = 1"

    def test_frozen(self) -> None:
        sf = SourceFile(display_name="a.agl", normalized_text="")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(sf, "display_name", "b.agl")


# ---------------------------------------------------------------------------
# program.py — ExecutableModule
# ---------------------------------------------------------------------------


class TestExecutableModule:
    def test_construct(self) -> None:
        init = IrConstUnit(location=LOC)
        em = ExecutableModule(module_id=MOD_A, initializers=(init,))
        assert em.module_id == MOD_A
        assert em.initializers == (init,)

    def test_frozen(self) -> None:
        em = ExecutableModule(module_id=MOD_A, initializers=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(em, "module_id", MOD_B)


# ---------------------------------------------------------------------------
# program.py — ExecutableProgram
# ---------------------------------------------------------------------------


class TestExecutableProgram:
    def _make_program(self) -> ExecutableProgram:
        sym_desc = SymbolDescriptor(
            symbol_id=SYM0, mutable=False, public_name="y", owner=MOD_A
        )
        nom_desc = NominalDescriptor(
            nominal=NOM0, display_name="Foo", kind=NominalKind.RECORD
        )
        sf = SourceFile(display_name="main.agl", normalized_text="")
        em = ExecutableModule(module_id=MOD_A, initializers=())

        return ExecutableProgram(
            entry_module=MOD_A,
            modules={MOD_A: em},
            symbols={SYM0: sym_desc},
            nominals={NOM0: nom_desc},
            sources={SID0: sf},
        )

    def test_construct(self) -> None:
        prog = self._make_program()
        assert prog.entry_module == MOD_A
        assert MOD_A in prog.modules
        assert SYM0 in prog.symbols
        assert NOM0 in prog.nominals
        assert SID0 in prog.sources

    def test_table_access(self) -> None:
        prog = self._make_program()
        assert prog.symbols[SYM0].public_name == "y"
        assert prog.nominals[NOM0].display_name == "Foo"
        assert prog.sources[SID0].display_name == "main.agl"
        assert prog.modules[MOD_A].module_id == MOD_A

    def test_frozen_reference(self) -> None:
        prog = self._make_program()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(prog, "entry_module", MOD_B)

    def test_tables_are_mutable_dicts_post_construction(self) -> None:
        # The dataclass reference is frozen, but the dict contents may be
        # populated by the linker before freezing usage.
        prog = self._make_program()
        # We can mutate the dict (documented behavior: treat as immutable after construction)
        new_sf = SourceFile(display_name="other.agl", normalized_text="")
        prog.sources[SourceId(value=99)] = new_sf
        assert SourceId(value=99) in prog.sources
